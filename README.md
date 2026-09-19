# Brain Twin (brain2game)

Point your phone at the world; a Modal H100 runs Meta's TRIBE v2 brain encoder on what it sees and
hears; the laptop paints what an *average human brain* would be doing about 15 s later on a 3D cortex;
press Finish and a Gemini agent writes you a browser game that exercises the least-driven system.

Honest framing: TRIBE v2 predicts the group-average fMRI response to a stimulus. This is a brain
twin / simulator, not a brain reader. Text pathway is off (audio + video only).

Design and API contracts: [CONTRACTS.md](CONTRACTS.md).

## One-time setup

```
# .env (git-ignored)
GEMINI_API_KEY=...
MODAL_BASE_URL=https://recozers--brain-twin-tribeservice-web.modal.run

pip install httpx google-genai python-dotenv playwright nilearn
python3 -m playwright install chromium
brew install ffmpeg
```

Modal: `modal token` already set up; volume `brain-twin-cache` holds the weights (seeded once with
`modal run scripts/seed_volume.py`); secret `huggingface-secret` supplies the HF token.

## Run

```
# 1. backend (once; keeps one H100 warm until you `modal app stop brain-twin`)
modal deploy modal_app/tribe_service.py

# 2. laptop app
uvicorn app.server:app --port 8001        # then open http://localhost:8001  (8000 is taken by an old http.server on this laptop)

#    stage fallback with no backend at all: http://localhost:8001/?mock=1
# 3. phone: scan the QR on the viewer (opens <MODAL_BASE_URL>/capture?sid=...), press Start
#    or, without a phone, replay a clip through the real pipeline:
python3 scripts/fake_phone.py samples/face_talk.mov --sid <sid shown in the viewer>
```

## Sample clips and baseline

```
python3 scripts/predict_file.py samples/*.mov      # -> samples/<name>.npz/.json + samples/baseline.json
python3 scripts/build_assets.py && python3 scripts/build_hires.py   # rebuild viewer meshes/atlas after a fresh clone (binaries are git-ignored)
```

The baseline (per-system stats averaged over the samples) is what the agent compares a session
against when deciding which system is under-driven.

## Layout

- `modal_app/tribe_service.py` — Modal app: TRIBE v2 inference, rolling 30 s window sessions, /predict, phone page
- `modal_app/fast_video.py` — speed patches for neuralset's V-JEPA2 extractor (resident model, bf16, ffmpeg decode)
- `modal_app/capture.html` — phone capture page (MediaRecorder rotated every 5 s)
- `app/server.py`, `app/agent.py` — laptop FastAPI + Gemini game agent; `app/static/` — three.js viewer (`brain.js` = full-res fsaverage renderer with pial/inflated morph and bloom, press `i` or the Inflate button; add `?hires=0` to fall back to the fsaverage5 mesh)
- `app/regions.py` — functional systems table shared by everything
- `scripts/` — asset builder, volume seeding, sample prediction, fake phone

## Demo-day runbook

1. Backend warm? `curl -s $MODAL_BASE_URL/ | head -c 200` should show `"version"` and `"gpu"`. If the app was stopped: `modal deploy modal_app/tribe_service.py` and wait ~2.5 min for the warm-up.
2. `uvicorn app.server:app --port 8001` from the repo root, open http://localhost:8001 on the laptop (a fresh `sid` is generated; the QR encodes it).
3. Phone: scan the QR, allow camera + mic, press **Start streaming**, keep the app in the foreground. Point it at faces, motion and speech; the brain lags ~15-20 s.
4. **Finish** → the agent log streams into the page (~45 s with `GEMINI_MODEL_CODE=gemini-3.8-flash`; ~3 min with the pro model) → the game takes over the screen, brain in the corner. **Back to brain** returns.

Fallbacks, in order:
- No phone / bad wifi: `python3 scripts/fake_phone.py <clip> --sid <sid shown on the viewer>` streams a clip through the real pipeline.
- No backend: `http://localhost:8001/?mock=1` (synthetic activity, mock Finish, built-in game).
- Gemini down: the agent falls back to `app/games/_fallback_motion.html` automatically. Pre-generated real games are kept in `app/games/` and served at `/game/<id>` (e.g. `20260919-114240-a18c` coherent-dot motion, `20260919-113855-6be8` rule-shift cards).

## License notes

The code in this repo was written in a one-day hackathon and is provided as-is. It depends on Meta's
[TRIBE v2](https://github.com/facebookresearch/tribev2) model and weights, which are released under
**CC-BY-NC 4.0 (non-commercial)**; anything you build on top inherits that restriction for the model
part. The fsaverage surfaces come from FreeSurfer via nilearn; the Glasser parcellation is the
HCP-MMP1.0 atlas projected to fsaverage.
