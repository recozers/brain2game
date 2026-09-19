"""Brain Twin - Modal service (worker pool edition).

Two classes on one app:
  * TribeWorker  - GPU containers (H100), WORKERS of them kept warm. Each takes one window of video
                   and returns TRIBE v2 predictions (audio + video + speech text pathways).
  * TribeService - one small CPU container: owns the live sessions, builds rolling windows from the
                   phone's 3 s chunks, dispatches them to the pool so ticks overlap, serves the API
                   and the phone capture page.

    WORKERS=3 GPU=H200 WINDOW_S=12 modal deploy modal_app/tribe_service.py
"""
import asyncio
import base64
import json
import math
import os
import statistics
import subprocess
import threading
import time
import uuid
from pathlib import Path

import modal

APP_NAME = "brain-twin"
VERSION = 15
CACHE_DIR = "/cache"            # volume: HF hub cache lives at /cache/huggingface/hub
FEAT_DIR = "/tmp/feat"          # per-worker neuralset feature cache (exca keeps an in-memory index: never wipe it)
SESS_DIR = "/tmp/sessions"
N_VERT = 20484
TRAIL_DROP = 2                  # seconds dropped at the end of each window (no trailing context)
ACTIVE_Z = 0.3                  # system-level value counted as "active" (average-subject scale: peaks ~0.5-0.7)
MAX_PREDS_PER_RESPONSE = 60
MAX_CHUNK_S = 60.0              # live chunks are ~3 s; reject unbounded/corrupt media timelines
# deploy-time knobs (baked into the image env so the containers see the same values)
WORKERS = int(os.environ.get("WORKERS", "3"))
GPU = os.environ.get("GPU", "H100")            # H100 | H200 (B200 needs torch>=2.7, which tribev2 pins out)
WINDOW_S = float(os.environ.get("WINDOW_S", "20"))
TEXT_ON = os.environ.get("TEXT_ON", "1") != "0"

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
    .env({"WORKERS": str(WORKERS), "WINDOW_S": str(WINDOW_S), "TEXT_ON": "1" if TEXT_ON else "0", "GPU": GPU})
    .add_local_file(str(CAPTURE_LOCAL), "/root/capture.html")
    .add_local_file(str(HERE / "fast_video.py"), "/root/fast_video.py")
    .add_local_file(str(HERE / "fast_text.py"), "/root/fast_text.py")
)
if ATLAS_LOCAL.exists():
    image = image.add_local_file(str(ATLAS_LOCAL), "/root/atlas.json")

app = modal.App(APP_NAME)


# ---------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------
def _run(cmd: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def _probe_duration(path: str) -> float | None:
    r = _run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path], timeout=60)
    try:
        duration = float(r.stdout.strip())
        return duration if math.isfinite(duration) and duration > 0 else None
    except Exception:
        return None


def _prepare_chunk(path: str) -> tuple[str, float]:
    """Remux into a consistent stream order; repair overflowing packet durations before concat.

    MediaRecorder can write a uint32-underflowed last-frame duration (~83 days at 600 Hz).
    Clamping only our Python duration leaves that broken timestamp in the actual video. Repair
    packet durations without re-encoding or moving the audio/video timestamps, then re-probe.
    """
    duration = _probe_duration(path)
    filters = []
    if duration is None or duration > MAX_CHUNK_S:
        r = _run(["ffprobe", "-v", "error", "-show_entries",
                  "stream=index,codec_type:packet=stream_index,pts_time,duration_time",
                  "-of", "json", path], timeout=30)
        if r.returncode:
            raise ValueError("could not read chunk timestamps")
        metadata = json.loads(r.stdout)
        for stream in metadata.get("streams", []):
            kind = stream.get("codec_type")
            if kind not in {"video", "audio"}:
                continue
            packets = [p for p in metadata.get("packets", []) if p["stream_index"] == stream["index"]]
            starts = [float(p["pts_time"]) for p in packets if "pts_time" in p]
            if not starts or any(not math.isfinite(t) or t < -1 or t > MAX_CHUNK_S for t in starts):
                raise ValueError("chunk packet timestamps exceed the live recording limit")
            durations = [float(p["duration_time"]) for p in packets if "duration_time" in p]
            good = [d for d in durations if math.isfinite(d) and 0 < d <= 1]
            if any(not math.isfinite(d) or d > MAX_CHUNK_S for d in durations):
                if not good:
                    raise ValueError("chunk has no usable packet durations")
                typical = statistics.median(good)
                stream_type = "v" if kind == "video" else "a"
                filters.extend([f"-bsf:{stream_type}",
                                f"setts=duration=if(gt(DURATION*TB\\,{MAX_CHUNK_S})\\,{typical}/TB\\,DURATION)"])
        print(f"repairing chunk {os.path.basename(path)}: reported duration={duration}, packet repair={bool(filters)}", flush=True)

    # Some phone chunks switch their audio/video stream order. Concat requires a stable order.
    ext = ".webm" if Path(path).suffix.lower() == ".webm" else ".mp4"
    out = str(Path(path).with_suffix("")) + ".normalized" + ext
    cmd = ["ffmpeg", "-y", "-v", "error", "-copyts", "-i", path, "-map", "0:v:0", "-map", "0:a:0?",
           "-c", "copy", *filters]
    if ext == ".mp4":
        cmd += ["-movflags", "+faststart"]
    try:
        r = _run(cmd + [out], timeout=30)
        if r.returncode:
            raise ValueError(f"could not normalize chunk: {r.stderr[-300:]}")
        repaired_duration = _probe_duration(out)
        if repaired_duration is None or repaired_duration > MAX_CHUNK_S:
            raise ValueError("chunk duration is invalid after timestamp repair")
        return out, repaired_duration
    except Exception:
        Path(out).unlink(missing_ok=True)
        raise


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


def _transcode_480(src: str) -> str:
    dst = src.rsplit(".", 1)[0] + "_480.mp4"
    r = _run(["ffmpeg", "-y", "-i", src, "-vf", "scale='min(640,iw)':-2", "-r", "30", "-c:v", "libx264", "-preset", "ultrafast",
              "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-ac", "1", "-movflags", "+faststart", dst])
    return dst if r.returncode == 0 else src


# ---------------------------------------------------------------------------------------------------
# GPU workers
# ---------------------------------------------------------------------------------------------------
@app.cls(
    image=image,
    gpu=GPU,
    cpu=8,
    memory=32768,
    volumes={CACHE_DIR: vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=3600,
    scaledown_window=1200,
    min_containers=WORKERS,
    max_containers=WORKERS,
)
class TribeWorker:
    @modal.enter()
    def load(self):
        import sys

        import numpy as np  # noqa: F401
        import torch

        t0 = time.time()
        os.makedirs(FEAT_DIR, exist_ok=True)
        os.makedirs(SESS_DIR, exist_ok=True)
        tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if tok:
            os.environ["HF_TOKEN"] = tok
        self.worker_id = (os.environ.get("MODAL_TASK_ID") or uuid.uuid4().hex)[-6:]
        self.gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        sys.path.insert(0, "/root")
        import fast_video  # noqa: F401  (monkeypatches neuralset before any extractor is built)

        self.fast_video = fast_video
        from tribev2 import TribeModel

        import fast_text  # noqa: F401  (resident faster-whisper replaces the whisperx subprocess)

        self.fast_text = fast_text
        model = None
        # num_workers 0: forked loader workers die in a threaded container; text batch 32 instead of the release config's 4
        for update in ({"data.num_workers": 0, "data.text_feature.batch_size": 32}, {"data.num_workers": 0}, None):
            try:
                model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=FEAT_DIR, config_update=update)
                print(f"[{self.worker_id}] model loaded with config_update={update}", flush=True)
                break
            except Exception as e:
                print(f"[{self.worker_id}] from_pretrained with config_update={update} failed: {type(e).__name__}: {e}", flush=True)
        if model is None:
            raise RuntimeError("could not load TRIBE v2")
        self.model = model
        self.load_s = round(time.time() - t0, 1)
        self.warm_s = self.warm2_s = None
        if TEXT_ON:
            try:
                fast_text.get_model()
                vol.commit()
            except Exception as e:
                print(f"[{self.worker_id}] whisper preload failed: {type(e).__name__}: {e}", flush=True)
        try:
            for i in range(2):
                warm = f"{SESS_DIR}/_warm{i}.mp4"
                _synthetic_clip(warm, 30, speech=(i == 1))
                t1 = time.time()
                preds, _, info = self.infer_file(warm)
                dt = round(time.time() - t1, 1)
                if i == 0:
                    self.warm_s = dt
                else:
                    self.warm2_s = dt
                print(f"[{self.worker_id}] warm-up {i}: {preds.shape} in {dt}s ({info['n_words']} words) | {fast_video.stats()}", flush=True)
        except Exception as e:
            print(f"[{self.worker_id}] warm-up failed: {type(e).__name__}: {e}", flush=True)
        print(f"[{self.worker_id}] ready: load {self.load_s}s warm {self.warm_s}/{self.warm2_s}s", flush=True)

    def infer_file(self, path: str):
        """Video file -> (preds float32 (T, N_VERT), rel_starts (T,), info)."""
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
                print(f"[{self.worker_id}] text pathway failed ({type(e).__name__}: {e}); audio+video only", flush=True)
                events = get_audio_and_text_events(ev, audio_only=True)
        else:
            events = get_audio_and_text_events(ev, audio_only=True)
        t1 = time.time()
        self.fast_text.FEATURE_TIMES.clear()
        preds, segments = self.model.predict(events, verbose=False)
        t2 = time.time()
        preds = np.asarray(preds, dtype=np.float32)
        try:
            starts = np.array([float(sg.start) for sg in segments], dtype=float)
            starts = starts - starts.min()
            order = np.argsort(starts, kind="stable")
            preds, starts = preds[order], starts[order]
        except Exception:
            starts = np.arange(len(preds), dtype=float)
        ft = {k: v["s"] for k, v in self.fast_text.FEATURE_TIMES.items()}
        feat_s = round(sum(ft.values()), 2)
        info = {"n_words": n_words, "timing": {"events_s": round(t1 - t0, 2), "features_s": feat_s, "head_s": round(t2 - t1 - feat_s, 2), "features": ft}}
        print(f"[{self.worker_id}] infer {os.path.basename(path)}: events {t1 - t0:.1f}s ({n_words} words), features {feat_s}s {ft}, "
              f"head {t2 - t1 - feat_s:.1f}s -> {preds.shape}", flush=True)
        return preds, starts, info

    @modal.method()
    def infer_window(self, data: bytes, name: str, transcode: bool = False) -> dict:
        import numpy as np

        d = f"{SESS_DIR}/in"
        os.makedirs(d, exist_ok=True)
        ext = (Path(name).suffix or ".mp4").lower()
        raw = f"{d}/{uuid.uuid4().hex}{ext}"
        with open(raw, "wb") as f:
            f.write(data)
        path = _transcode_480(raw) if transcode else raw
        t0 = time.time()
        try:
            preds, starts, info = self.infer_file(path)
        finally:
            for p in {raw, path}:
                try:
                    os.remove(p)
                except OSError:
                    pass
        info["timing"]["total_s"] = round(time.time() - t0, 2)
        return {"preds": np.ascontiguousarray(preds.astype(np.float16)).tobytes(), "shape": list(preds.shape),
                "starts": [float(x) for x in starts], "n_words": info["n_words"], "timing": info["timing"], "worker": self.worker_id}

    @modal.method()
    def stats(self) -> dict:
        return {"worker": self.worker_id, "gpu": self.gpu_name, "load_s": self.load_s, "warm_s": self.warm_s, "warm2_s": self.warm2_s,
                "fast_video": self.fast_video.stats(), "fast_text": self.fast_text.stats() if TEXT_ON else None}


# ---------------------------------------------------------------------------------------------------
# sessions (live in the API container)
# ---------------------------------------------------------------------------------------------------
def _video_segments(chunks: list[dict], start: float, duration: float = 1.0) -> list[dict]:
    """Locate a prediction interval in the exact chunk sequence supplied to inference."""
    parts, cursor = [], 0.0
    for chunk in chunks:
        end = cursor + chunk["duration"]
        lo, hi = max(start, cursor), min(start + duration, end)
        if hi > lo:
            parts.append({"index": chunk["index"], "offset_s": round(lo - cursor, 6),
                          "duration_s": round(hi - lo, 6)})
        cursor = end
        if cursor >= start + duration:
            break
    return parts


class Session:
    def __init__(self, sid: str, svc: "TribeService"):
        self.sid = sid
        self.svc = svc
        self.dir = f"{SESS_DIR}/{sid}"
        os.makedirs(self.dir, exist_ok=True)
        self.chunks: list[dict] = []          # present chunks sorted by index: {index, path, duration, t_start}
        self.by_index: dict[int, dict] = {}   # arrival order is NOT stimulus order: uploads can overtake each other
        self.preds: dict[int, "np.ndarray"] = {}
        self.sys: dict[int, dict] = {}
        self.top: dict[int, list] = {}
        self.video: dict[int, list] = {}     # source clip offsets retained with the winning prediction
        self.lock = threading.Lock()
        self.pending = False                  # chunks arrived since the last dispatch
        self.inflight = 0
        self.max_inflight = max(1, WORKERS)
        self.finish_requested = False
        self.final_done = threading.Event()
        self.finish_done = threading.Event()
        self.finish_result = None
        self.n_windows = 0
        self.last_infer_s = None
        self.last_window_s = None
        self.last_error = None
        self.timings: list[dict] = []
        self.created = time.time()

    # ---- ingest -------------------------------------------------------------
    def add_chunk(self, index: int, path: str, client_duration: float) -> float:
        path, dur = _prepare_chunk(path)
        with self.lock:
            if self.finish_requested:
                raise ValueError("session is already finishing; start a new session")
            self.by_index[index] = {"index": index, "path": path, "duration": float(dur)}
            self._rebuild_timeline()
            self.pending = True
            end = self.by_index[index]["t_start"] + dur
        self._maybe_dispatch()
        return end

    def _rebuild_timeline(self) -> None:
        """t_start from the sequence numbers: a chunk that has not arrived (yet) still occupies its slot,
        with the typical chunk duration, so later chunks stay at the right stimulus time. Call with the lock."""
        present = sorted(self.by_index)
        if not present:
            self.chunks = []
            return
        typical = sorted(c["duration"] for c in self.by_index.values())[len(self.by_index) // 2]
        t = 0.0
        for i in range(present[-1] + 1):
            c = self.by_index.get(i)
            if c is None:
                t += typical
                continue
            c["t_start"] = t
            t += c["duration"]
        self.chunks = [self.by_index[i] for i in present]

    def seconds_received(self) -> float:
        with self.lock:
            return float(max((c["t_start"] + c["duration"] for c in self.chunks), default=0.0))

    # ---- scheduling: every new chunk starts a window if a worker slot is free ----
    def _maybe_dispatch(self):
        with self.lock:
            if self.finish_requested or not self.pending or self.inflight >= self.max_inflight or not self.chunks:
                return
            self.pending = False
            self.inflight += 1
            n = self.n_windows
            self.n_windows += 1
            chunks = [c.copy() for c in self.chunks]
        threading.Thread(target=self._run_window, args=(chunks, n, False), daemon=True, name=f"win-{self.sid}-{n}").start()

    def _build_window(self, chunks: list[dict], n: int) -> tuple[str, float, float, list[dict]]:
        sel, total = [], 0.0
        for c in reversed(chunks):                       # newest first, only while the sequence stays contiguous
            if total >= WINDOW_S or (sel and c["index"] != sel[0]["index"] - 1):
                break
            sel.insert(0, c)
            total += c["duration"]
        list_path = f"{self.dir}/win_{n:05d}.txt"
        out = f"{self.dir}/win_{n:05d}.mp4"
        with open(list_path, "w") as f:
            for c in sel:
                f.write(f"file '{c['path']}'\n")
                f.write(f"duration {c['duration']:.6f}\n")
        r = _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c", "copy", "-movflags", "+faststart", out])
        duration = _probe_duration(out)
        ok = r.returncode == 0 and duration is not None and abs(duration - total) < 1.0
        if not ok:
            r = _run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_path, "-c:v", "libx264", "-preset", "ultrafast",
                      "-crf", "24", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-t", str(total),
                      "-movflags", "+faststart", out])
            if r.returncode != 0:
                raise RuntimeError(f"ffmpeg concat failed: {r.stderr[-800:]}")
            duration = _probe_duration(out)
            if duration is None or abs(duration - total) >= 1.0:
                raise RuntimeError("inference window duration does not match its source clips")
        return out, sel[0]["t_start"], total, sel

    def _run_window(self, chunks: list[dict], n: int, final: bool):
        try:
            path, t_start, total, window_chunks = self._build_window(chunks, n)
            with open(path, "rb") as f:
                data = f.read()
            try:
                os.remove(path)
            except OSError:
                pass
            t0 = time.time()
            preds, rel_starts, res = self.svc.infer_remote(data, os.path.basename(path), transcode=False)
            dt = time.time() - t0
            drop = 0 if final else TRAIL_DROP
            keep = max(0, len(preds) - drop)
            new = 0
            with self.lock:
                for i in range(keep):
                    if not math.isfinite(rel_starts[i]) or not 0 <= rel_starts[i] < total:
                        continue
                    sec = int(round(t_start + rel_starts[i]))
                    if sec in self.preds:
                        continue
                    vec = preds[i]
                    self.preds[sec] = vec.astype("float16")
                    s, top = self.svc.stats_for(vec)
                    self.sys[sec] = s
                    self.top[sec] = top
                    self.video[sec] = _video_segments(window_chunks, float(rel_starts[i]))
                    new += 1
                self.last_infer_s = round(dt, 2)
                self.last_window_s = round(total, 2)
                self.timings.append({"window": n, "window_s": round(total, 1), "round_trip_s": round(dt, 2), "new": new, "worker": res.get("worker"),
                                     "n_words": res.get("n_words"), **{k: v for k, v in res.get("timing", {}).items() if k != "features"}})
                self.timings = self.timings[-50:]
            print(f"[{self.sid}] window {n} ({total:.1f}s) on {res.get('worker')} -> {len(preds)} s in {dt:.1f}s, {new} new, "
                  f"predicted={len(self.preds)} inflight={self.inflight}", flush=True)
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            print(f"[{self.sid}] window {n} failed: {self.last_error}", flush=True)
        finally:
            with self.lock:
                self.inflight -= 1
            if final:
                self.final_done.set()
            else:
                self._maybe_dispatch()

    # ---- reads --------------------------------------------------------------
    def status(self) -> dict:
        with self.lock:
            received = float(max((c["t_start"] + c["duration"] for c in self.chunks), default=0.0))
            n_pred = len(self.preds)
            newest = max(self.preds) if self.preds else -1
            return {
                "chunks_received": len(self.chunks),
                "seconds_received": round(received, 2),
                "seconds_predicted": n_pred,
                "newest_predicted_second": newest,
                "busy": self.inflight > 0,
                "inflight": self.inflight,
                "workers": self.max_inflight,
                "pending": self.pending,
                "last_infer_s": self.last_infer_s,
                "last_window_s": self.last_window_s,
                "delay_estimate_s": round(received - newest, 1) if newest >= 0 else None,
                "windows": self.n_windows,
                "last_error": self.last_error,
                "finished": self.finish_requested,
                "finish_state": "done" if self.finish_done.is_set() else "finishing" if self.finish_requested else "recording",
            }

    def preds_since(self, since: int) -> dict:
        import numpy as np

        with self.lock:
            secs = sorted(k for k in self.preds if k >= since)[:MAX_PREDS_PER_RESPONSE]
            data = np.stack([self.preds[k] for k in secs]).astype("float16") if secs else np.zeros((0, N_VERT), "float16")
            regions = [{"second": k, "top": self.top[k]} for k in secs]
            systems = [{"second": k, "z": self.sys[k]} for k in secs]
            video = [{"second": k, "clips": self.video.get(k, [])} for k in secs]
        st = self.status()
        return {"sid": self.sid, "rate_hz": 1, "n_vertices": N_VERT, "seconds": secs,
                "data_b64": base64.b64encode(np.ascontiguousarray(data).tobytes()).decode(),
                "regions": regions, "systems": systems, "video": video,
                "seconds_received": st["seconds_received"], "status": st}

    def all_preds(self):
        import numpy as np

        with self.lock:
            secs = sorted(self.preds)
            arr = np.stack([self.preds[k] for k in secs]).astype("float32") if secs else np.zeros((0, N_VERT), "float32")
        return secs, arr


# ---------------------------------------------------------------------------------------------------
# API container (CPU)
# ---------------------------------------------------------------------------------------------------
@app.cls(image=image, cpu=4, memory=8192, timeout=3600, scaledown_window=1200, min_containers=1, max_containers=1)
@modal.concurrent(max_inputs=32)
class TribeService:
    @modal.enter()
    def load(self):
        import numpy as np

        os.makedirs(SESS_DIR, exist_ok=True)
        self.sessions: dict[str, Session] = {}
        self.sessions_lock = threading.Lock()
        self.worker = TribeWorker()
        self.calls = 0
        self.last_timing = None
        self.started = time.time()
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
            print(f"api: atlas {a.get('atlas')} {len(self.region_idx)} regions, {len(self.system_idx)} systems; {WORKERS} workers, window {WINDOW_S}s, text {TEXT_ON}", flush=True)
        else:
            print("api: no atlas.json baked in; region summaries disabled", flush=True)

    # ---- remote inference -------------------------------------------------------
    def infer_remote(self, data: bytes, name: str, transcode: bool):
        import numpy as np

        t0 = time.time()
        res = self.worker.infer_window.remote(data, name, transcode)
        dt = time.time() - t0
        preds = np.frombuffer(res["preds"], dtype=np.float16).reshape(res["shape"]).astype(np.float32)
        starts = np.asarray(res["starts"], dtype=float)
        self.calls += 1
        self.last_timing = {**res.get("timing", {}), "round_trip_s": round(dt, 2), "worker": res.get("worker"), "n_words": res.get("n_words")}
        return preds, starts, res

    # ---- atlas stats -----------------------------------------------------------
    def stats_for(self, vec) -> tuple[dict, list]:
        if not self.system_idx:
            return {}, []
        sysz = {k: round(float(vec[idx].mean()), 3) if len(idx) else 0.0 for k, idx in self.system_idx.items()}
        means = {nm: float(vec[idx].mean()) for nm, idx in self.region_idx.items()}
        top = sorted(means.items(), key=lambda kv: -kv[1])[:8]
        return sysz, [{"name": nm, "z": round(z, 3)} for nm, z in top]

    def summarize(self, secs: list[int], arr) -> dict:
        import numpy as np

        out = {"n_seconds_predicted": int(len(secs)), "duration_s": float(max(secs) + 1) if secs else 0.0, "systems": [], "regions": [], "timeline": []}
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

    # ---- sessions ---------------------------------------------------------------
    def session(self, sid: str, create: bool = True) -> Session | None:
        with self.sessions_lock:
            s = self.sessions.get(sid)
            if s is None and create:
                s = Session(sid, self)
                self.sessions[sid] = s
            return s

    def finish(self, sid: str, wait: bool = True) -> dict:
        s = self.session(sid, create=False)
        if s is None:
            return {"error": "unknown session"}
        with s.lock:
            start = not s.finish_requested
            if start:
                s.finish_requested = True
                s.pending = False
        if start:
            threading.Thread(target=self._finish_session, args=(s,), daemon=True, name=f"finish-{sid}").start()
        if wait:
            s.finish_done.wait()
        if s.finish_done.is_set():
            return s.finish_result
        return {"sid": sid, "state": "finishing", "status": s.status()}

    def _finish_session(self, s: Session) -> None:
        """Run once per session, independent of HTTP connections; cache the completed summary."""
        try:
            # Let all dispatched windows land before the final pass and summary.
            while True:
                with s.lock:
                    busy = s.inflight > 0
                if not busy:
                    break
                time.sleep(0.5)
            with s.lock:
                chunks = [c.copy() for c in s.chunks]
                if chunks:
                    n = s.n_windows
                    s.n_windows += 1
                    s.inflight += 1
            if chunks:
                s._run_window(chunks, n, True)
            secs, arr = s.all_preds()
            summary = self.summarize(secs, arr)
            summary.update({"sid": s.sid, "status": {**s.status(), "finish_state": "done"}, "timings": s.timings[-12:]})
            s.finish_result = summary
        except Exception as e:
            s.finish_result = {"error": f"{type(e).__name__}: {e}", "sid": s.sid}
        finally:
            s.finish_done.set()

    def predict_bytes(self, data: bytes, filename: str) -> dict:
        import numpy as np

        t0 = time.time()
        preds, starts, res = self.infer_remote(data, filename, transcode=True)
        dt = time.time() - t0
        secs = [int(round(x)) for x in starts]
        sysz, tops = [], []
        for i, sec in enumerate(secs):
            s_, top = self.stats_for(preds[i])
            sysz.append({"second": sec, "z": s_})
            tops.append({"second": sec, "top": top})
        summary = self.summarize(secs, preds)
        return {"filename": filename, "infer_s": round(dt, 2), "worker": res.get("worker"), "n_words": res.get("n_words"), "timing": res.get("timing"),
                "rate_hz": 1, "n_vertices": N_VERT, "seconds": secs,
                "data_b64": base64.b64encode(np.ascontiguousarray(preds.astype("float16")).tobytes()).decode(),
                "regions": tops, "systems": sysz, "summary": summary}

    # ---- web --------------------------------------------------------------------
    @modal.asgi_app()
    def web(self):
        from fastapi import FastAPI, File, Form, HTTPException, UploadFile
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

        svc = self
        api = FastAPI(title="Brain Twin")
        api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

        @api.get("/")
        def root():
            return {"ok": True, "app": APP_NAME, "version": VERSION, "workers": WORKERS, "gpu": GPU, "window_s": WINDOW_S, "text_on": TEXT_ON,
                    "atlas": svc.atlas is not None, "calls": svc.calls, "last_timing": svc.last_timing, "uptime_s": round(time.time() - svc.started),
                    "sessions": list(svc.sessions)}

        @api.get("/workers")
        async def workers():
            return await asyncio.to_thread(lambda: svc.worker.stats.remote())

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
            if s.finish_requested:
                raise HTTPException(409, "session is already finishing; start a new session")
            ext = Path(file.filename or "").suffix.lower() or ".mp4"
            path = f"{s.dir}/chunk_{index:05d}{ext}"
            with open(path, "wb") as f:
                f.write(data)
            try:
                received = await asyncio.to_thread(s.add_chunk, index, path, duration)
            except ValueError as e:
                raise HTTPException(422, str(e)) from e
            return {"ok": True, "index": index, "bytes": len(data), "seconds_received": round(received, 2)}

        @api.get("/session/{sid}/preds")
        def preds(sid: str, since: int = 0):
            s = svc.session(sid, create=False)
            if s is None:
                return {"sid": sid, "rate_hz": 1, "n_vertices": N_VERT, "seconds": [], "data_b64": "", "regions": [], "systems": [],
                        "seconds_received": 0.0, "status": {"chunks_received": 0, "unknown_session": True}}
            return s.preds_since(since)

        @api.get("/session/{sid}/status")
        def status(sid: str):
            s = svc.session(sid, create=False)
            if s is None:
                raise HTTPException(404, "unknown session")
            return s.status()

        @api.get("/session/{sid}/video/{index}")
        def video(sid: str, index: int):
            s = svc.session(sid, create=False)
            if s is None:
                raise HTTPException(404, "unknown session")
            with s.lock:
                chunk = s.by_index.get(index)
                path = chunk["path"] if chunk else None
            if not path or not os.path.isfile(path):
                raise HTTPException(404, "video chunk unavailable")
            media_type = "video/webm" if Path(path).suffix.lower() == ".webm" else "video/mp4"
            return FileResponse(path, media_type=media_type, headers={"Cache-Control": "no-store"})

        @api.post("/session/{sid}/finish")
        async def finish(sid: str, wait: bool = True):
            result = await asyncio.to_thread(svc.finish, sid, wait=wait)
            return JSONResponse(result, status_code=202 if result.get("state") == "finishing" else 200)

        @api.post("/predict")
        async def predict(file: UploadFile = File(...)):
            import traceback

            data = await file.read()
            try:
                return await asyncio.to_thread(svc.predict_bytes, data, file.filename or "clip.mp4")
            except Exception as e:
                tb = traceback.format_exc()[-1500:]
                print(f"/predict failed: {type(e).__name__}: {e}\n{tb}", flush=True)
                return JSONResponse({"error": f"{type(e).__name__}: {e}", "traceback": tb}, status_code=500)

        return api
