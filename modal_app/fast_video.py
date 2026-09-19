"""Speed patches for neuralset's HuggingFaceVideo extractor on the V-JEPA2 path.

Stock behaviour (neuralset 0.0.2): the V-JEPA2 giant model is re-instantiated from disk on every
call, every 4 s window is decoded frame by frame through moviepy, preprocessed on CPU and run in
fp32, one window at a time. On an H100 that is ~1.6 s per window, 2 windows per second of video.

These patches keep the model resident, decode each video once with ffmpeg straight to 256x256 at
16 fps, preprocess on the GPU (checked once against the HF processor) and run under bf16 autocast.
They do not touch the exca cache wrapper around _get_data, so feature caching keeps working.
Import this module before building the TribeModel.
"""
import logging
import subprocess
import threading
import time

import numpy as np
import PIL.Image
import torch
from neuralset.extractors import video as nsv

log = logging.getLogger("fast_video")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("fast_video - %(message)s"))
    log.addHandler(_h)

DEC_FPS = 16.0          # 64 frames per 4 s clip
SIZE = 256              # V-JEPA2 fpc64-256 input size (processor: resize shortest edge to 292, centre-crop 256)
RESIZE_EDGE = 292
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

_MODELS: dict = {}
_FRAMES: dict = {}
_LOCK = threading.Lock()
STATE = {"gpu_preproc": None, "windows": 0, "gpu_s": 0.0, "decode_s": 0.0, "decoded": 0, "check": None}

_Orig = nsv._HFVideoModel
_orig_predict = _Orig.predict
_orig_read = nsv._VideoImage._read


class ResidentHFVideoModel:
    """Drop-in for nsv._HFVideoModel that returns one shared instance per config."""

    MODELS = _Orig.MODELS
    check_layer_type = staticmethod(_Orig.check_layer_type)

    def __new__(cls, model_name, pretrained=True, layer_type="", num_frames=None):
        key = (model_name, pretrained, layer_type, num_frames)
        with _LOCK:
            m = _MODELS.get(key)
            if m is None:
                t0 = time.time()
                m = _Orig(model_name=model_name, pretrained=pretrained, layer_type=layer_type, num_frames=num_frames)
                if torch.cuda.is_available():
                    m.model.to("cuda")
                _MODELS[key] = m
                log.info("loaded %s once in %.1fs (resident from now on)", model_name, time.time() - t0)
        return m


def _predict_fast(self, images, audio=None):
    if "vjepa2" not in self.model_name:
        return _orig_predict(self, images, audio)
    arr = np.asarray(images)
    dev = self.model.device
    on_gpu = (arr.ndim == 4 and arr.shape[1] == SIZE and arr.shape[2] == SIZE and arr.dtype == np.uint8
              and STATE["gpu_preproc"] is True and dev.type == "cuda")
    if on_gpu:
        x = torch.from_numpy(np.ascontiguousarray(arr)).to(dev, non_blocking=True)
        x = x.permute(0, 3, 1, 2).float().div_(255.0)
        x = ((x - MEAN.to(dev)) / STD.to(dev)).unsqueeze(0)  # (1, T, 3, H, W)
        inputs = {"pixel_values_videos": x}
    else:
        inputs = self.processor(videos=list(arr), return_tensors="pt")
        nsv._fix_pixel_values(inputs)
        inputs = inputs.to(dev)
    t0 = time.time()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
        pred = self.model(**inputs)
    STATE["windows"] += 1
    STATE["gpu_s"] += time.time() - t0
    return pred


def _decode(path: str) -> np.ndarray:
    vf = (f"fps={DEC_FPS},scale='if(gt(iw,ih),-2,{RESIZE_EDGE})':'if(gt(iw,ih),{RESIZE_EDGE},-2)':flags=bilinear,"
          f"crop={SIZE}:{SIZE}")
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                       capture_output=True, check=True)
    n = len(r.stdout) // (SIZE * SIZE * 3)
    return np.frombuffer(r.stdout, np.uint8)[: n * SIZE * SIZE * 3].reshape(n, SIZE, SIZE, 3)


def _probe_frame(path: str) -> np.ndarray:
    """First source frame scaled to shortest edge 292 but NOT cropped (what the HF processor would see)."""
    import io

    vf = f"scale='if(gt(iw,ih),-2,{RESIZE_EDGE})':'if(gt(iw,ih),{RESIZE_EDGE},-2)':flags=bilinear"
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", vf, "-frames:v", "1", "-f", "image2", "-c:v", "png", "-"],
                       capture_output=True, check=True)
    return np.asarray(PIL.Image.open(io.BytesIO(r.stdout)).convert("RGB"))


def _equivalence_check(path: str, frames: np.ndarray) -> None:
    """Once: compare our ffmpeg-crop + GPU-normalise against the HF processor on the uncropped frame."""
    if STATE["gpu_preproc"] is not None or not _MODELS or not len(frames):
        return
    try:
        m = next(iter(_MODELS.values()))
        dev = m.model.device
        probe = _probe_frame(path)
        ref = m.processor(videos=[probe], return_tensors="pt")
        key = "pixel_values_videos" if "pixel_values_videos" in ref else list(ref.keys())[0]
        r = ref[key].float().reshape(-1, 3, SIZE, SIZE)[0].to(dev)
        x = torch.from_numpy(np.ascontiguousarray(frames[0])).to(dev).permute(2, 0, 1).float().div_(255.0)
        x = (x - MEAN[0].to(dev)) / STD[0].to(dev)
        d = (r - x).abs()
        mean, p99, frac = d.mean().item(), d.flatten().kthvalue(int(0.99 * d.numel())).values.item(), (d > 0.5).float().mean().item()
        ok = mean < 0.05 and frac < 0.02
        STATE["gpu_preproc"] = ok
        STATE["check"] = {"probe_shape": list(probe.shape), "mean_abs_diff": round(mean, 4), "p99": round(p99, 4), "frac_gt_0.5": round(frac, 4)}
        log.info("GPU preprocessing check vs HF processor: probe=%s mean|d|=%.4f p99=%.4f frac>0.5=%.4f -> %s",
                 probe.shape, mean, p99, frac, "ENABLED" if ok else "DISABLED (HF processor on our crops)")
    except Exception as e:
        STATE["gpu_preproc"] = False
        log.warning("equivalence check failed (%s); using HF processor", e)


def _read_fast(self):
    fn = getattr(self.video, "filename", None)
    if not fn:
        return _orig_read(self)
    frames = _FRAMES.get(fn)
    if frames is None:
        t0 = time.time()
        try:
            frames = _decode(fn)
        except Exception as e:  # fall back to moviepy for anything ffmpeg dislikes
            log.warning("ffmpeg decode failed for %s (%s); falling back to moviepy", fn, e)
            return _orig_read(self)
        with _LOCK:
            while len(_FRAMES) >= 3:
                _FRAMES.pop(next(iter(_FRAMES)))
            _FRAMES[fn] = frames
        STATE["decode_s"] += time.time() - t0
        STATE["decoded"] += 1
        log.info("decoded %s: %d frames @ %g fps in %.2fs", fn, len(frames), DEC_FPS, time.time() - t0)
        _equivalence_check(fn, frames)
    if not len(frames):
        return _orig_read(self)
    i = min(len(frames) - 1, max(0, int(round(self.time * DEC_FPS))))
    return PIL.Image.fromarray(frames[i])


def stats() -> dict:
    w = max(1, STATE["windows"])
    return {"windows": STATE["windows"], "gpu_s_per_window": round(STATE["gpu_s"] / w, 4),
            "gpu_preproc": STATE["gpu_preproc"], "check": STATE["check"], "videos_decoded": STATE["decoded"],
            "decode_s_total": round(STATE["decode_s"], 2), "resident_models": len(_MODELS)}


nsv._HFVideoModel = ResidentHFVideoModel
_Orig.predict = _predict_fast
nsv._VideoImage._read = _read_fast
log.info("patches installed: resident V-JEPA2, bf16 autocast, GPU preprocessing, one-pass ffmpeg decode")
