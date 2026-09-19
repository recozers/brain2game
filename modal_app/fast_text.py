"""Resident faster-whisper in place of tribev2's whisperx subprocess.

Stock tribev2 transcribes by shelling out to `uvx whisperx` (large-v3 + wav2vec2 alignment) on every
call: a fresh process and model load each time, 10-20 s before a single word comes back. This keeps
one faster-whisper model on the GPU and returns the same words DataFrame the pipeline expects
(text, start, duration, sequence_id, sentence). Import after tribev2 is importable, before inference.
"""
import logging
import os
import threading
import time

import pandas as pd
from tribev2 import eventstransforms as et

log = logging.getLogger("fast_text")
log.setLevel(logging.INFO)
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("fast_text - %(message)s"))
    log.addHandler(_h)

MODEL_NAME = os.environ.get("WHISPER_MODEL", "large-v3-turbo")
LANG = {"english": "en", "french": "fr", "spanish": "es", "dutch": "nl", "chinese": "zh"}
COLUMNS = ["text", "start", "duration", "sequence_id", "sentence"]
STATE = {"model": MODEL_NAME, "loaded": False, "calls": 0, "seconds": 0.0, "words": 0, "last_words": None, "last_s": None}
_MODEL = None
_LOCK = threading.Lock()


def get_model():
    global _MODEL
    with _LOCK:
        if _MODEL is None:
            import torch  # noqa: F401  (loads the CUDA runtime libs ctranslate2 dlopens)
            from faster_whisper import WhisperModel

            t0 = time.time()
            _MODEL = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
            STATE["loaded"] = True
            log.info("faster-whisper %s resident in %.1fs", MODEL_NAME, time.time() - t0)
    return _MODEL


def transcribe(wav_filename, language: str = "english") -> pd.DataFrame:
    """Drop-in for ExtractWordsFromAudio._get_transcript_from_audio."""
    model = get_model()
    t0 = time.time()
    rows = []
    try:
        segments, _info = model.transcribe(
            str(wav_filename), language=LANG.get(language, "en"), word_timestamps=True, beam_size=1,
            vad_filter=True, condition_on_previous_text=False,
        )
        for i, seg in enumerate(segments):
            sentence = (seg.text or "").replace('"', "").strip()
            for w in seg.words or []:
                text = (w.word or "").replace('"', "").strip()
                if not text:
                    continue
                rows.append({"text": text, "start": float(w.start), "duration": float(max(0.02, w.end - w.start)),
                             "sequence_id": i, "sentence": sentence})
    except Exception as e:  # a transcription failure must never take the brain down
        log.warning("transcription failed for %s: %s", wav_filename, e)
    dt = time.time() - t0
    STATE["calls"] += 1
    STATE["seconds"] += dt
    STATE["words"] += len(rows)
    STATE["last_words"] = len(rows)
    STATE["last_s"] = round(dt, 2)
    log.info("transcribed %s: %d words in %.2fs", os.path.basename(str(wav_filename)), len(rows), dt)
    return pd.DataFrame(rows, columns=COLUMNS)


def stats() -> dict:
    return dict(STATE, mean_s=round(STATE["seconds"] / STATE["calls"], 2) if STATE["calls"] else None)


et.ExtractWordsFromAudio._get_transcript_from_audio = staticmethod(transcribe)

# ---- keep Llama and w2v-bert resident: neuralset rebuilds its extractor objects on every call and
# each fresh HuggingFaceText / Wav2VecBert reloads its weights from disk (~5 s for Llama 3.2 3B) ----
from neuralset.extractors import audio as nsa  # noqa: E402
from neuralset.extractors import text as nst  # noqa: E402

_SHARED: dict = {}
_orig_load_model = nst.HuggingFaceText._load_model


def _shared_load_model(self, **kwargs):
    key = ("text", self.model_name, repr(sorted(kwargs.items())))
    m = _SHARED.get(key)
    if m is None:
        t0 = time.time()
        m = _orig_load_model(self, **kwargs)
        _SHARED[key] = m
        log.info("%s loaded once in %.1fs (resident from now on)", self.model_name, time.time() - t0)
    return m


nst.HuggingFaceText._load_model = _shared_load_model

for _cls in (nsa.HuggingFaceAudio, getattr(nsa, "Wav2Vec", None), getattr(nsa, "Wav2VecBert", None),
             getattr(nsa, "SeamlessM4T", None), getattr(nsa, "Whisper", None)):
    if _cls is None or "_get_sound_model" not in _cls.__dict__:
        continue
    _orig = _cls.__dict__["_get_sound_model"]

    def _make(orig, cname):
        def _shared_sound(self, model_name, *a, **k):
            key = ("audio", cname, model_name)
            m = _SHARED.get(key)
            if m is None:
                t0 = time.time()
                m = orig(self, model_name, *a, **k)
                _SHARED[key] = m
                log.info("%s %s loaded once in %.1fs (resident from now on)", cname, model_name, time.time() - t0)
            return m
        return _shared_sound

    setattr(_cls, "_get_sound_model", _make(_orig, _cls.__name__))


# ---- per-modality timing of the feature pre-computation (video / audio / text) ------------------
from neuralset.extractors import base as nsbase_ext  # noqa: E402

FEATURE_TIMES: dict = {}
_orig_prepare = nsbase_ext.BaseExtractor.prepare


def _timed_prepare(self, *args, **kwargs):
    t0 = time.time()
    out = _orig_prepare(self, *args, **kwargs)
    dt = time.time() - t0
    name = type(self).__name__
    n = len(args[0]) if args and hasattr(args[0], "__len__") else -1
    FEATURE_TIMES[name] = {"s": round(dt, 2), "n_events": n}
    if dt > 0.05:
        log.info("features %s: %s events in %.2fs", name, n, dt)
    return out


nsbase_ext.BaseExtractor.prepare = _timed_prepare
log.info("patch installed: ExtractWordsFromAudio now uses resident faster-whisper %s", MODEL_NAME)
