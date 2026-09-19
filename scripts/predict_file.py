"""Send a local clip through the deployed Modal /predict endpoint, save predictions next to it,
and refresh samples/baseline.json (per-system stats averaged over every sample so far).

    python3 scripts/predict_file.py samples/face_talk.mov [more clips...]
    python3 scripts/predict_file.py --baseline-only        # just rebuild baseline.json from saved .json files

MODAL_BASE_URL comes from .env (or --url).
"""
import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


def load_env():
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env


def predict(url: str, clip: Path) -> dict:
    import httpx

    t0 = time.time()
    with clip.open("rb") as f:
        r = httpx.post(f"{url}/predict", files={"file": (clip.name, f, "application/octet-stream")}, timeout=600)
    r.raise_for_status()
    out = r.json()
    out["_wall_s"] = round(time.time() - t0, 1)
    return out


def save(clip: Path, res: dict) -> Path:
    n = len(res["seconds"])
    arr = np.frombuffer(base64.b64decode(res["data_b64"]), dtype=np.float16).reshape(n, res["n_vertices"]).astype(np.float32)
    stem = SAMPLES / clip.stem
    np.savez_compressed(f"{stem}.npz", seconds=np.array(res["seconds"]), preds=arr)
    meta = {k: v for k, v in res.items() if k != "data_b64"}
    (SAMPLES / f"{clip.stem}.json").write_text(json.dumps(meta, indent=1))
    return stem


def rebuild_baseline() -> dict:
    per_sys: dict[str, list] = {}
    labels: dict[str, str] = {}
    n_clips = 0
    for jp in sorted(SAMPLES.glob("*.json")):
        if jp.name == "baseline.json":
            continue
        meta = json.loads(jp.read_text())
        systems = meta.get("summary", {}).get("systems", [])
        if not systems:
            continue
        n_clips += 1
        for s in systems:
            per_sys.setdefault(s["id"], []).append(s)
            labels[s["id"]] = s["label"]
    baseline = {"n_clips": n_clips, "clips": [p.stem for p in sorted(SAMPLES.glob("*.json")) if p.name != "baseline.json"], "systems": {}}
    for sid, rows in per_sys.items():
        means = np.array([r["mean"] for r in rows])
        baseline["systems"][sid] = {
            "label": labels[sid],
            "mean": round(float(means.mean()), 4),
            "std": round(float(means.std()), 4),
            "peak": round(float(np.mean([r["peak"] for r in rows])), 4),
            "frac_active": round(float(np.mean([r["frac_active"] for r in rows])), 4),
        }
    (SAMPLES / "baseline.json").write_text(json.dumps(baseline, indent=1))
    return baseline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="*")
    ap.add_argument("--url", default=None)
    ap.add_argument("--baseline-only", action="store_true")
    a = ap.parse_args()
    SAMPLES.mkdir(exist_ok=True)
    if not a.baseline_only:
        url = a.url or load_env().get("MODAL_BASE_URL") or os.environ.get("MODAL_BASE_URL")
        if not url:
            sys.exit("MODAL_BASE_URL missing: put it in .env or pass --url")
        url = url.rstrip("/")
        for c in a.clips:
            clip = Path(c)
            print(f"-> {clip.name} ({clip.stat().st_size/1e6:.1f} MB) ...", flush=True)
            res = predict(url, clip)
            stem = save(clip, res)
            top = ", ".join(f"{s['id']}={s['mean']:.2f}" for s in res["summary"]["systems"][:4])
            print(f"   {len(res['seconds'])} s predicted, infer {res['infer_s']}s, wall {res['_wall_s']}s -> {stem}.npz | top: {top}")
    b = rebuild_baseline()
    print(f"baseline.json rebuilt from {b['n_clips']} clip(s): " + ", ".join(f"{k}={v['mean']:.2f}" for k, v in b["systems"].items()))


if __name__ == "__main__":
    main()
