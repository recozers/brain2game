# Brain Twin — build contracts (hackathon, 2026-09-19)

Phone films the world → Modal runs TRIBE v2 (audio+video only, no text) on a rolling 30 s window →
laptop three.js paints predicted cortical activity ~15 s behind reality → "Finish" → Gemini agent
reads the session's region stats and writes a single-file browser game that targets the least-driven
system → user plays it on the laptop.

Honest framing everywhere in UI copy: TRIBE v2 predicts the *average* brain's fMRI response to a
stimulus. It is a brain twin / simulator, not a brain reader.

## Repo layout

```
agentichack/
  CONTRACTS.md
  .env                        # user-created, git-ignored: GEMINI_API_KEY=..., MODAL_BASE_URL=... (filled after deploy)
  modal_app/
    tribe_service.py          # Modal app "brain-twin": GPU inference, session API, phone capture page
    capture.html              # phone capture page, served by Modal at GET /capture
  app/
    server.py                 # laptop FastAPI on http://localhost:8000
    agent.py                  # Gemini: analyse session -> plan -> generate game HTML -> headless check
    regions.py                # functional systems table (shared by Modal summaries and the agent)
    static/
      index.html              # three.js viewer (no build step; three via CDN import map)
      viewer.js
      assets/                 # built by scripts/build_assets.py
        mesh.json             # {"n_vertices":20484,"n_faces":...,"hemi_offset":10242,"files":{...}}
        positions.f32         # Float32 little-endian, N*3, inflated surface, L then R, R shifted +x so hemis don't overlap
        faces.u32             # Uint32, F*3, indices into the combined vertex array
        sulc.f32              # Float32, N, sulcal depth for base shading
        atlas.json            # see below
    games/                    # generated games: <game_id>.html + <game_id>.json (plan/log)
  scripts/
    build_assets.py           # nilearn fsaverage5 -> app/static/assets/*
    seed_volume.py            # one-off: copy weights old volume -> fresh volume
    predict_file.py           # one-shot: local clip -> Modal /predict -> samples/<name>.npz + baseline update
  samples/                    # user's 30 s clips (.mov/.mp4), <name>.npz predictions, baseline.json
```

Local Python is 3.10 (pyenv). fastapi + uvicorn + modal are installed. Node 23 exists but the viewer
needs no build step. Modal image is Python 3.11.

## Vertex space

TRIBE v2 output: `(T, 20484)` float, 1 Hz, fsaverage5. Index 0..10241 = left hemisphere,
10242..20483 = right hemisphere (verify once in scripts/build_assets.py against tribev2's plotting code).
Values are z-score-like BOLD predictions; typical range roughly -2..+3; "active" means > 1.0.

## atlas.json

```json
{
  "n_vertices": 20484,
  "hemi_offset": 10242,
  "atlas": "destrieux",                       // or "glasser" if we get it
  "labels": [0, 0, 17, ...],                  // 20484 ints, index into names; 0 = "Unknown/Medial wall"
  "names": ["Unknown", "L_G_cuneus", ...],    // hemi-prefixed
  "systems": {                                // functional systems, see app/regions.py
    "motion": {"label": "Visual motion (MT+)", "regions": ["L_G_occipital_middle", "R_..."], "vertex_count": 812}
  }
}
```

`app/regions.py` is the single source of truth for the systems (id, label, one-line blurb, atlas
region names per atlas, `game_target: true|false`). Systems (ids are stable, do not rename):
`early_visual`, `motion`, `faces`, `places`, `objects`, `attention`, `auditory`, `somatomotor`,
`frontal_control`, `language` (narration only, game_target false), `default_mode` (game_target false).

## Modal API  (base URL in .env as MODAL_BASE_URL; CORS `*`; all JSON unless noted)

- `GET /capture?sid=<sid>` → phone capture page (HTML). Rotates a MediaRecorder every 5 s, POSTs each file.
- `POST /session/{sid}/chunk`  multipart: `file` (video/mp4), form fields `index` (int, 0-based), `duration` (float s)
  → `{"ok": true, "index": 3, "seconds_received": 20.0}`
- `GET /session/{sid}/preds?since=<int>` → newly predicted seconds with `second >= since`:
  ```json
  {"sid": "abc", "rate_hz": 1, "n_vertices": 20484,
   "seconds": [12, 13, 14],
   "data_b64": "<base64 of float16 array, shape (len(seconds), 20484), C order>",
   "regions": [{"second": 12, "top": [{"name": "L_S_calcarine", "z": 2.1}, ...]}],   // top 8 per second
   "systems": [{"second": 12, "z": {"early_visual": 1.8, "motion": 0.4, ...}}],
   "seconds_received": 25.0, "status": {...same as /status...}}
  ```
- `GET /session/{sid}/status` → `{"chunks_received": 5, "seconds_received": 25.0, "seconds_predicted": 18,
   "busy": true, "inflight": 2, "workers": 3, "last_infer_s": 7.9, "last_window_s": 20.0, "delay_estimate_s": 15.4}`
- `POST /session/{sid}/finish` → session summary (also stored on the container until restart):
  ```json
  {"sid": "abc", "duration_s": 240.0, "n_seconds_predicted": 230,
   "systems": [{"id": "motion", "label": "Visual motion (MT+)", "mean": 0.12, "peak": 1.9,
                "frac_active": 0.05, "rank": 7}],           // sorted most→least driven, rank 1 = most driven
   "regions":  [{"name": "L_S_calcarine", "system": "early_visual", "mean": 0.4, "peak": 2.6, "frac_active": 0.2}],
   "timeline": [{"second": 0, "z": {"early_visual": 1.2, "motion": 0.3, ...}}]}
  ```
- `POST /predict` multipart `file` → same shape as `/preds` (all seconds) plus `"summary"` (same shape as finish).
  Used for sample clips, the baseline, and (stretch) scoring generated games.

Modal service behaviour: one small CPU API container plus a pool of WORKERS warm H100 containers.
Each new 3 s chunk dispatches "predict the trailing 20 s window" to a free worker (ticks overlap);
when all workers are busy the newest chunk waits and the next free worker takes the latest window
(skip-to-latest, never queue).
Predictions for the last 2 s of a window are dropped (no trailing context) and filled by the next window.
First write wins per absolute second.

## Laptop API  (http://localhost:8000)

- `GET /` → viewer. `GET /config` → `{"modal_base_url": "...", "sid": "<new or ?sid= given>"}`
- `POST /finish` `{"sid": "..."}` → `{"job_id": "..."}`; `GET /finish/{job_id}` →
  `{"state": "running|done|error", "log": ["Fetching session summary", ...], "plan": {...}, "game_url": "/game/<id>"}`
- `GET /game/{id}` → the generated HTML. `GET /samples/baseline.json` → per-system baseline stats.

## Agent (app/agent.py) — Gemini via `google-genai` SDK, key from .env

1. **Analyse** (fast model, JSON mode): input = finish summary + baseline.json (if present) + regions.py
   system blurbs. Output `{"target_system": "motion", "why": "...", "evidence": {...},
   "game": {"title": "...", "concept": "...", "mechanic": "...", "controls": "mouse|keyboard"}}`.
   Only systems with `game_target: true`. Under-driven = lowest mean/frac_active vs baseline z.
2. **Generate** (strongest code model): one complete HTML file. Contract for the generated game:
   single file, no external requests, canvas-based, laptop keyboard/mouse, 60 s round with visible timer,
   header shows target system label + one-sentence "why" from the plan, no `alert()`,
   on round end `window.parent.postMessage({type: "brain-game-score", score, target: "<system id>"}, "*")`,
   exposes `window.startDemo()` that plays itself for 25 s (for future TRIBE scoring).
3. **Check**: load in headless Chromium (Playwright) for 5 s, collect console errors + a screenshot.
   On error, one regenerate pass with the errors appended. Save to app/games/<id>.html and <id>.json.

Model names are NOT hard-coded: on first use list models via the API and pick the newest `*-pro` for code
and newest `*-flash` for analysis, overridable with GEMINI_MODEL_CODE / GEMINI_MODEL_FAST in .env.

## Viewer (app/static)

- Loads mesh + atlas, renders inflated fsaverage5, sulc as base shading, per-vertex colour from the
  latest prediction, thresholded colormap (below 0.8 → base, 0.8..2.5 → warm ramp), smooth 1 s crossfade.
- Polls `/session/{sid}/preds?since=` every 1 s, buffers seconds, plays them at 1 Hz in arrival order.
  Shows "brain time" (stimulus second being displayed), "delay" (seconds_received - shown second) and a
  live bar of the 11 systems from the `systems` field.
- Shows a QR code (client-side lib from CDN) encoding `${modal_base_url}/capture?sid=${sid}`.
- Finish button → POST /finish, shows the agent log streaming in, then opens `/game/<id>` in an iframe
  taking over the viewer, with the brain shrunk to a corner.
