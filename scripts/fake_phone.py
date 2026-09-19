"""Pretend to be the phone: split a local clip into 3 s chunks and POST them to a live session in
real time, exactly like modal_app/capture.html does. Lets us test the live loop without a phone, and
doubles as the on-stage fallback (replay a canned clip through the real pipeline).

    python3 scripts/fake_phone.py samples/face_talk.mov --sid demo          # real time (3 s per chunk)
    python3 scripts/fake_phone.py clip.mp4 --sid demo --fast                # no pacing
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent


def load_env():
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def split(video: Path, out_dir: Path, chunk_s: float) -> list[Path]:
    """Cut exact chunk_s pieces by seeking + re-encoding each one (robust, always exact)."""
    total = duration(video)
    out = []
    i, t = 0, 0.0
    while t < total - 0.25:
        dst = out_dir / f"chunk_{i:04d}.mp4"
        cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{t:.3f}", "-t", f"{chunk_s:.3f}", "-i", str(video),
               "-vf", "scale='min(640,iw)':-2", "-r", "30", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-ac", "1", "-movflags", "+faststart", str(dst)]
        subprocess.run(cmd, check=True)
        out.append(dst)
        i += 1
        t += chunk_s
    return out


def duration(p: Path) -> float:
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                       capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except ValueError:
        return 5.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--sid", default="demo")
    ap.add_argument("--url", default=None)
    ap.add_argument("--chunk", type=float, default=3.0)
    ap.add_argument("--fast", action="store_true", help="do not pace to real time")
    a = ap.parse_args()
    url = (a.url or load_env().get("MODAL_BASE_URL") or os.environ.get("MODAL_BASE_URL") or "").rstrip("/")
    if not url:
        sys.exit("MODAL_BASE_URL missing (.env) and no --url")
    video = Path(a.video)
    with tempfile.TemporaryDirectory() as td:
        chunks = split(video, Path(td), a.chunk)
        print(f"{video.name}: {len(chunks)} chunks of ~{a.chunk:.0f}s -> {url}/session/{a.sid}", flush=True)
        t_session = time.time()
        with httpx.Client(timeout=120) as client:
            for i, c in enumerate(chunks):
                d = duration(c)
                t_due = t_session + sum(duration(x) for x in chunks[:i]) + d  # chunk i is "recorded" by then
                if not a.fast:
                    time.sleep(max(0.0, t_due - time.time()))
                t0 = time.time()
                with c.open("rb") as f:
                    r = client.post(f"{url}/session/{a.sid}/chunk", files={"file": (c.name, f, "video/mp4")},
                                    data={"index": str(i), "duration": f"{d:.2f}"})
                up = time.time() - t0
                st = client.get(f"{url}/session/{a.sid}/status").json()
                print(f"chunk {i:02d} {d:4.1f}s {c.stat().st_size/1e3:6.0f} KB up {up*1000:4.0f} ms | "
                      f"received {st['seconds_received']:5.1f}s predicted {st['seconds_predicted']:3d}s "
                      f"newest {st['newest_predicted_second']:3d} delay~{st['delay_estimate_s']} busy={st['busy']} "
                      f"infer={st['last_infer_s']}s err={st['last_error']}", flush=True)
            # let the tail catch up
            for _ in range(12):
                time.sleep(5)
                st = client.get(f"{url}/session/{a.sid}/status").json()
                print(f"  ... predicted {st['seconds_predicted']}s of {st['seconds_received']:.1f}s, busy={st['busy']} "
                      f"pending={st['pending']} infer={st['last_infer_s']}s", flush=True)
                if not st["busy"] and not st["pending"]:
                    break


if __name__ == "__main__":
    main()
