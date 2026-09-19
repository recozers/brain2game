"""Brain Twin - Modal service.

TRIBE v2 inference (audio + video, text pathway off), rolling-window live sessions fed by 5 s phone
chunks, one-shot /predict for sample clips, and the phone capture page.

    modal deploy modal_app/tribe_service.py
"""
import asyncio
import base64
import json
import os
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import modal

APP_NAME = "brain-twin"
VERSION = 9
CACHE_DIR = "/cache"            # volume: HF hub cache lives at /cache/huggingface/hub
FEAT_DIR = "/tmp/feat"          # per-container neuralset feature cache (wiped after every window)
SESS_DIR = "/tmp/sessions"
WINDOW_S = float(os.environ.get("WINDOW_S", "20"))   # trailing window fed to the model (20 s: ~12 s per tick on H100)
TRAIL_DROP = 2                  # seconds dropped at the end of each window (no trailing context)
N_VERT = 20484
ACTIVE_Z = 0.3                  # system-level value counted as "active" (average-subject scale: peaks ~0.5-0.7)
TEXT_ON = os.environ.get("TEXT_ON", "1") != "0"   # transcribe speech (resident faster-whisper) and feed the Llama text pathway
MAX_PREDS_PER_RESPONSE = 60

HERE = Path(__file__).parent
CAPTURE_LOCAL = HERE / "capture.html"
ATLAS_LOCAL = HERE.parent / "app" / "static" / "assets" / "atlas.json"

vol = modal.Volume.from_name("brain-twin-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
    .pip_install("numpy==2.2.6", "torch==2.6.0", "torchvision==0.21.0", "torchaudio==2.6.0")
    .run_commands(
        "git clone --depth 1 https://github.com/facebookresearch/tribev2.git /tribev2",
        "cd /tribev2 && pip install -e .",
    )
    .pip_install("fastapi[standard]", "python-multipart", "faster-whisper>=1.1")
    .env(
        {
            "HF_HOME": f"{CACHE_DIR}/huggingface",
            "HF_HUB_DOWNLOAD_TIMEOUT": "300",
            "HF_HUB_HTTP_TIMEOUT": "300",
            "TOKENIZERS_PARALLELISM": "false",
            # ctranslate2 (faster-whisper) dlopens cuBLAS/cuDNN: point it at the pip-installed CUDA libs
            "LD_LIBRARY_PATH": "/usr/local/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/lib/python3.11/site-packages/nvidia/cudnn/lib",
        }
    )
    .add_local_file(str(CAPTURE_LOCAL), "/root/capture.html")
    .add_local_file(str(HERE / "fast_video.py"), "/root/fast_video.py")
    .add_local_file(str(HERE / "fast_text.py"), "/root/fast_text.py")
)
if ATLAS_LOCAL.exists():
    image = image.add_local_file(str(ATLAS_LOCAL), "/root/atlas.json")

app = modal.App(APP_NAME)


def _run(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _probe_duration(path: str) -> float | None:
    r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], timeout=60)
    try:
        return float(r.stdout.strip())
    except Exception:
        return None


def _synthetic_clip(path: str, seconds: int = 30, speech: bool = False) -> None:
    """Colour bars + tone (or a TTS sentence loop when speech=True): only used to warm the models."""
    audio = None
    if speech:
        try:
            from gtts import gTTS

            mp3 = path + ".speech.mp3"
            gTTS("The quick brown fox jumps over the lazy dog while the orchestra plays in the old town square. "
                 "She said the museum opens at nine, so we walked past the river and watched the boats.", lang="en").save(mp3)
            audio = mp3
        except Exception as e:
            print(f"gTTS unavailable ({e}); warm-up uses a tone", flush=True)
    if audio:
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=640x480:rate=30", "-stream_loop", "-1", "-i", audio,
               "-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-shortest", path]
    else:
        cmd = ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=size=640x480:rate=30", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000",
               "-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", path]
    _run(cmd)


class Session:
    def __init__(self, sid: str, svc: "TribeService"):
        self.sid = sid
        self.svc = svc
        self.dir = f"{SESS_DIR}/{sid}"
        os.makedirs(self.dir, exist_ok=True)
        self.chunks: list[dict] = []          # {index, path, duration, t_start}
        self.preds: dict[int, "np.ndarray"] = {}   # abs second -> float16 (N_VERT,)
        self.sys: dict[int, dict] = {}        # abs second -> {system_id: z}
        self.top: dict[int, list] = {}        # abs second -> [{name, z}]
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.pending = False
        self.busy = False
        self.finish_requested = False
        self.final_done = threading.Event()
        self.n_windows = 0
        self.last_infer_s = None
        self.last_window_s = None
        self.last_error = None
        self.created = time.time()
        self.thread = threading.Thread(target=self._worker, daemon=True, name=f"worker-{sid}")
        self.thread.start()

    # ---- ingest -------------------------------------------------------------
    def add_chunk(self, index: int, path: str, client_duration: float) -> float:
        dur = _probe_duration(path) or client_duration or 5.0
        with self.lock:
            t_start = sum(c["duration"] for c in self.chunks)
            self.chunks.append({"index": index, "path": path, "duration": float(dur), "t_start": t_start})
            self.pending = True
        self.event.set()
        return t_start + dur

    def seconds_received(self) -> float:
        with self.lock:
            return float(sum(c["duration"] for c in self.chunks))

    # ---- worker -------------------------------------------------------------
    def _worker(self):
        while True:
            self.event.wait()
            self.event.clear()
            while True:
                with self.lock:
                    if not self.pending:
                        break
                    self.pending = False
                    final = self.finish_requested
                try:
                    self._run_window(final)
                except Exception as e:  # keep the worker alive whatever happens
                    self.last_error = f"{type(e).__name__}: {e}"
                    print(f"[{self.sid}] window failed: {self.last_error}", flush=True)
                if final:
                    self.final_done.set()

    def _build_window(self) -> tuple[str, float, float]:
        with self.lock:
            chunks = list(self.chunks)
            n = self.n_windows
            self.n_windows += 1
        sel, total = [], 0.0
        for c in reversed(chunks):
            if total >= WINDOW_S:
                break
            sel.insert(0, c)
            total += c["duration"]
        list_path = f"{self.dir}/win_{n:05d}.txt"
        out = f"{self.dir}/win_{n:05d}.mp4"
        with open(list_path, "w") as f:
            for c in sel:
                f.write(f"file '{c['path']}'\n")
        r = _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", "-movflags", "+faststart", out])
        ok = r.returncode == 0 and (_probe_duration(out) or 0) > 0.5
        if not ok:
            r = _run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c:v", "libx264", "-preset", "ultrafast",
                 "-crf", "24", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-movflags", "+faststart", out]
            )
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg concat failed: {r.stderr[-800:]}")
        return out, sel[0]["t_start"], total

    def _run_window(self, final: bool):
        self.busy = True
        try:
            path, t_start, total = self._build_window()
            t0 = time.time()
            preds, rel_starts = self.svc.infer_file(path)
            dt = time.time() - t0
            drop = 0 if final else TRAIL_DROP
            keep = max(0, len(preds) - drop)
            new = 0
            with self.lock:
                for i in range(keep):
                    sec = int(round(t_start + rel_starts[i]))
                    if sec in self.preds:
                        continue
                    vec = preds[i]
                    self.preds[sec] = vec.astype("float16")
                    s, top = self.svc.stats_for(vec)
                    self.sys[sec] = s
                    self.top[sec] = top
                    new += 1
                self.last_infer_s = round(dt, 2)
                self.last_window_s = round(total, 2)
            print(f"[{self.sid}] window {self.n_windows} ({total:.1f}s) -> {len(preds)} s in {dt:.1f}s, {new} new seconds, "
                  f"predicted={len(self.preds)}", flush=True)
            try:
                os.remove(path)
            except OSError:
                pass
        finally:
            self.busy = False

    # ---- reads --------------------------------------------------------------
    def status(self) -> dict:
        with self.lock:
            received = float(sum(c["duration"] for c in self.chunks))
            n_pred = len(self.preds)
            newest = max(self.preds) if self.preds else -1
            return {
                "chunks_received": len(self.chunks),
                "seconds_received": round(received, 2),
                "seconds_predicted": n_pred,
                "newest_predicted_second": newest,
                "busy": self.busy,
                "pending": self.pending,
                "last_infer_s": self.last_infer_s,
                "last_window_s": self.last_window_s,
                "delay_estimate_s": round(received - newest, 1) if newest >= 0 else None,
                "windows": self.n_windows,
                "last_error": self.last_error,
                "finished": self.finish_requested,
            }

    def preds_since(self, since: int) -> dict:
        import numpy as np

        with self.lock:
            secs = sorted(k for k in self.preds if k >= since)[:MAX_PREDS_PER_RESPONSE]
            data = np.stack([self.preds[k] for k in secs]).astype("float16") if secs else np.zeros((0, N_VERT), "float16")
            regions = [{"second": k, "top": self.top[k]} for k in secs]
            systems = [{"second": k, "z": self.sys[k]} for k in secs]
        st = self.status()
        return {
            "sid": self.sid,
            "rate_hz": 1,
            "n_vertices": N_VERT,
            "seconds": secs,
            "data_b64": base64.b64encode(np.ascontiguousarray(data).tobytes()).decode(),
            "regions": regions,
            "systems": systems,
            "seconds_received": st["seconds_received"],
            "status": st,
        }

    def all_preds(self):
        import numpy as np

        with self.lock:
            secs = sorted(self.preds)
            arr = np.stack([self.preds[k] for k in secs]).astype("float32") if secs else np.zeros((0, N_VERT), "float32")
        return secs, arr


@app.cls(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    volumes={CACHE_DIR: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    scaledown_window=1200,
    min_containers=1,
    max_containers=1,
)
@modal.concurrent(max_inputs=32)
class TribeService:
    @modal.enter()
    def load(self):
        import numpy as np
        import torch

        t0 = time.time()
        os.makedirs(FEAT_DIR, exist_ok=True)
        os.makedirs(SESS_DIR, exist_ok=True)
        tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if tok:
            os.environ["HF_TOKEN"] = tok
        self.gpu_lock = threading.Lock()
        self.sessions: dict[str, Session] = {}
        self.sessions_lock = threading.Lock()
        self.gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

        import sys

        sys.path.insert(0, "/root")
        import fast_video  # noqa: F401  (monkeypatches neuralset before any extractor is built)

        self.fast_video = fast_video
        from tribev2 import TribeModel

        import fast_text  # noqa: F401  (resident faster-whisper replaces the whisperx subprocess)

        self.fast_text = fast_text

        model = None
        # num_workers 0: forked loader workers die when forked from the ASGI thread; text batch 32: the release config
        # runs Llama on 4 words at a time (25 forwards for a 30 s clip), 32 cuts that to ~4 forwards
        for update in ({"data.num_workers": 0, "data.text_feature.batch_size": 32}, {"data.num_workers": 0}, None):
            try:
                model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=FEAT_DIR, config_update=update)
                print(f"model loaded with config_update={update}", flush=True)
                break
            except Exception as e:
                print(f"from_pretrained with config_update={update} failed: {type(e).__name__}: {e}", flush=True)
        if model is None:
            raise RuntimeError("could not load TRIBE v2")
        self.model = model
        self.load_s = round(time.time() - t0, 1)

        # atlas -> vertex index sets
        self.atlas = None
        self.system_idx: dict[str, np.ndarray] = {}
        self.region_idx: dict[str, np.ndarray] = {}
        self.region_system: dict[str, str | None] = {}
        if os.path.exists("/root/atlas.json"):
            with open("/root/atlas.json") as f:
                a = json.load(f)
            labels = np.asarray(a["labels"])
            names = a["names"]
            for i, nm in enumerate(names):
                if i == 0:
                    continue
                idx = np.flatnonzero(labels == i)
                if len(idx):
                    self.region_idx[nm] = idx
                    self.region_system[nm] = None
            for sid_, sd in a["systems"].items():
                idx = np.concatenate([self.region_idx[r] for r in sd["regions"] if r in self.region_idx] or [np.zeros(0, int)])
                self.system_idx[sid_] = idx
                for r in sd["regions"]:
                    self.region_system[r] = sid_
            self.atlas = {"atlas": a.get("atlas"), "systems": {k: {"label": v["label"]} for k, v in a["systems"].items()}}
            print(f"atlas loaded: {a.get('atlas')} {len(self.region_idx)} regions, {len(self.system_idx)} systems", flush=True)
        else:
            print("no atlas.json baked in; region summaries disabled", flush=True)

        # warm the backbones with two different synthetic clips: the second call measures the resident path
        self.warm_s = self.warm2_s = None
        if TEXT_ON:
            try:
                fast_text.get_model()
                vol.commit()   # keep the downloaded whisper weights on the volume
            except Exception as e:
                print(f"whisper preload failed: {type(e).__name__}: {e}", flush=True)
        try:
            for i, sec in enumerate((30, 30)):
                warm = f"{SESS_DIR}/_warm{i}.mp4"
                _synthetic_clip(warm, sec, speech=(i == 1))
                t1 = time.time()
                preds, _ = self.infer_file(warm)
                dt = round(time.time() - t1, 1)
                if i == 0:
                    self.warm_s = dt
                else:
                    self.warm2_s = dt
                print(f"warm-up call {i}: preds {preds.shape} in {dt}s | fast_video {fast_video.stats()}", flush=True)
            try:
                m = next(iter(fast_video._MODELS.values()))
                print(f"vjepa2 processor: {m.processor}", flush=True)
            except Exception:
                pass
        except Exception as e:
            print(f"warm-up failed: {type(e).__name__}: {e}", flush=True)

    # ---- inference ----------------------------------------------------------
    def infer_file(self, path: str):
        """Video file -> (preds float32 (T, N_VERT), rel_starts float (T,)) ordered by time."""
        import numpy as np
        import pandas as pd
        from tribev2.demo_utils import get_audio_and_text_events

        t0 = time.time()
        ev = pd.DataFrame([{"type": "Video", "filepath": path, "start": 0, "timeline": "default", "subject": "default"}])
        n_words = 0
        if TEXT_ON:
            try:
                events = get_audio_and_text_events(ev, audio_only=False)
                n_words = int((events.type == "Word").sum())
                if n_words == 0:
                    events = get_audio_and_text_events(ev, audio_only=True)
            except Exception as e:
                print(f"text pathway failed ({type(e).__name__}: {e}); falling back to audio+video", flush=True)
                events = get_audio_and_text_events(ev, audio_only=True)
        else:
            events = get_audio_and_text_events(ev, audio_only=True)
        t1 = time.time()
        with self.gpu_lock:
            self.fast_text.FEATURE_TIMES.clear()
            preds, segments = self.model.predict(events, verbose=False)
            t2 = time.time()
            t_feat = t1 + sum(v["s"] for v in self.fast_text.FEATURE_TIMES.values())   # features vs head split from the prepare timings
            preds = np.asarray(preds, dtype=np.float32)
            try:
                starts = np.array([float(sg.start) for sg in segments], dtype=float)
                starts = starts - starts.min()
                order = np.argsort(starts, kind="stable")
                preds, starts = preds[order], starts[order]
            except Exception:
                starts = np.arange(len(preds), dtype=float)
        ft = " ".join(f"{k}={v['s']}s/{v['n_events']}" for k, v in self.fast_text.FEATURE_TIMES.items())
        print(f"infer {os.path.basename(path)}: events {t1 - t0:.1f}s ({n_words} words), features {t_feat - t1:.1f}s [{ft}], head {t2 - t_feat:.1f}s -> {preds.shape}", flush=True)
        return preds, starts

    def stats_for(self, vec) -> tuple[dict, list]:
        import numpy as np

        if not self.system_idx:
            return {}, []
        sysz = {k: round(float(vec[idx].mean()), 3) if len(idx) else 0.0 for k, idx in self.system_idx.items()}
        means = {nm: float(vec[idx].mean()) for nm, idx in self.region_idx.items()}
        top = sorted(means.items(), key=lambda kv: -kv[1])[:8]
        return sysz, [{"name": nm, "z": round(z, 3)} for nm, z in top]

    def summarize(self, secs: list[int], arr) -> dict:
        import numpy as np

        out = {"n_seconds_predicted": int(len(secs)), "duration_s": float(max(secs) + 1) if secs else 0.0,
               "systems": [], "regions": [], "timeline": []}
        if not self.system_idx or not secs:
            return out
        sysz = {k: arr[:, idx].mean(1) if len(idx) else np.zeros(len(secs)) for k, idx in self.system_idx.items()}
        systems = []
        for k, z in sysz.items():
            systems.append({"id": k, "label": self.atlas["systems"][k]["label"], "mean": round(float(z.mean()), 4),
                            "peak": round(float(z.max()), 4), "frac_active": round(float((z > ACTIVE_Z).mean()), 4)})
        systems.sort(key=lambda d: -d["mean"])
        for r, d in enumerate(systems, 1):
            d["rank"] = r
        regions = []
        for nm, idx in self.region_idx.items():
            z = arr[:, idx].mean(1)
            regions.append({"name": nm, "system": self.region_system.get(nm), "mean": round(float(z.mean()), 4),
                            "peak": round(float(z.max()), 4), "frac_active": round(float((z > ACTIVE_Z).mean()), 4)})
        regions.sort(key=lambda d: -d["mean"])
        timeline = [{"second": int(s), "z": {k: round(float(sysz[k][i]), 3) for k in sysz}} for i, s in enumerate(secs)]
        out.update({"systems": systems, "regions": regions, "timeline": timeline})
        return out

    # ---- sessions -----------------------------------------------------------
    def session(self, sid: str, create: bool = True) -> Session | None:
        with self.sessions_lock:
            s = self.sessions.get(sid)
            if s is None and create:
                s = Session(sid, self)
                self.sessions[sid] = s
            return s

    def finish(self, sid: str) -> dict:
        s = self.session(sid, create=False)
        if s is None:
            return {"error": "unknown session"}
        with s.lock:
            s.finish_requested = True
            s.pending = bool(s.chunks)
        if s.chunks:
            s.event.set()
            s.final_done.wait(timeout=120)
        secs, arr = s.all_preds()
        summary = self.summarize(secs, arr)
        summary.update({"sid": sid, "status": s.status()})
        return summary

    def predict_bytes(self, data: bytes, filename: str) -> dict:
        import numpy as np

        d = f"{SESS_DIR}/_oneshot"
        os.makedirs(d, exist_ok=True)
        ext = (Path(filename).suffix or ".mp4").lower()
        raw = f"{d}/{uuid.uuid4().hex}{ext}"
        with open(raw, "wb") as f:
            f.write(data)
        small = raw.rsplit(".", 1)[0] + "_480.mp4"
        r = _run(["ffmpeg", "-y", "-i", raw, "-vf", "scale='min(640,iw)':-2", "-r", "30", "-c:v", "libx264", "-preset", "ultrafast",
                  "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-ac", "1", "-movflags", "+faststart", small])
        path = small if r.returncode == 0 else raw
        t0 = time.time()
        preds, starts = self.infer_file(path)
        dt = time.time() - t0
        secs = [int(round(x)) for x in starts]
        sysz, tops = [], []
        for i, sec in enumerate(secs):
            s_, top = self.stats_for(preds[i])
            sysz.append({"second": sec, "z": s_})
            tops.append({"second": sec, "top": top})
        summary = self.summarize(secs, preds)
        for p in (raw, small):
            try:
                os.remove(p)
            except OSError:
                pass
        return {
            "filename": filename, "infer_s": round(dt, 2), "rate_hz": 1, "n_vertices": N_VERT, "seconds": secs,
            "data_b64": base64.b64encode(np.ascontiguousarray(preds.astype("float16")).tobytes()).decode(),
            "regions": tops, "systems": sysz, "summary": summary,
        }

    # ---- web ----------------------------------------------------------------
    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import HTMLResponse

        svc = self
        api = FastAPI(title="Brain Twin")
        api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

        @api.get("/")
        def root():
            return {"ok": True, "app": APP_NAME, "version": VERSION, "gpu": svc.gpu_name, "atlas": svc.atlas is not None,
                    "fast_video": svc.fast_video.stats(), "text_on": TEXT_ON, "fast_text": svc.fast_text.stats() if TEXT_ON else None,
                    "load_s": svc.load_s, "warm_s": getattr(svc, "warm_s", None), "warm2_s": getattr(svc, "warm2_s", None),
                    "sessions": list(svc.sessions)}

        @api.get("/capture", response_class=HTMLResponse)
        def capture():
            with open("/root/capture.html") as f:
                return f.read()

        @api.get("/atlas")
        def atlas():
            if not os.path.exists("/root/atlas.json"):
                raise HTTPException(404, "no atlas baked into this deployment")
            with open("/root/atlas.json") as f:
                return json.load(f)

        @api.post("/session/{sid}/chunk")
        async def chunk(sid: str, file: UploadFile = File(...), index: int = Form(...), duration: float = Form(0.0)):
            data = await file.read()
            s = svc.session(sid)
            ext = Path(file.filename or "").suffix.lower() or ".mp4"
            path = f"{s.dir}/chunk_{index:05d}{ext}"
            with open(path, "wb") as f:
                f.write(data)
            received = await asyncio.to_thread(s.add_chunk, index, path, duration)
            return {"ok": True, "index": index, "bytes": len(data), "seconds_received": round(received, 2)}

        @api.get("/session/{sid}/preds")
        def preds(sid: str, since: int = 0):
            s = svc.session(sid, create=False)
            if s is None:
                return {"sid": sid, "rate_hz": 1, "n_vertices": N_VERT, "seconds": [], "data_b64": "", "regions": [],
                        "systems": [], "seconds_received": 0.0, "status": {"chunks_received": 0, "unknown_session": True}}
            return s.preds_since(since)

        @api.get("/session/{sid}/status")
        def status(sid: str):
            s = svc.session(sid, create=False)
            if s is None:
                raise HTTPException(404, "unknown session")
            return s.status()

        @api.post("/session/{sid}/finish")
        async def finish(sid: str):
            return await asyncio.to_thread(svc.finish, sid)

        @api.post("/predict")
        async def predict(file: UploadFile = File(...)):
            import traceback

            from fastapi.responses import JSONResponse

            data = await file.read()
            try:
                return await asyncio.to_thread(svc.predict_bytes, data, file.filename or "clip.mp4")
            except Exception as e:
                tb = traceback.format_exc()[-1500:]
                print(f"/predict failed: {type(e).__name__}: {e}\n{tb}", flush=True)
                return JSONResponse({"error": f"{type(e).__name__}: {e}", "traceback": tb}, status_code=500)

        return api
