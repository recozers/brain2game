"""Brain Twin - laptop server (see CONTRACTS.md, "Laptop API").

Run from the repo root:   uvicorn app.server:app --port 8000
Mock the agent:           MOCK_GEMINI=1 uvicorn app.server:app --port 8000

Routes
  GET  /                    -> app/static/index.html (three.js viewer)
  GET  /config              -> {"modal_base_url": <.env MODAL_BASE_URL or "">, "sid": <?sid= or new 6-char id>}
  POST /finish              -> {"job_id"}; body {"sid": "...", "summary": {...optional finish summary...}}
  GET  /finish/{job_id}     -> {"state": "running|done|error", "log": [...], "plan": {...}|null,
                                "game_url": "/game/<id>"|null, "error": str|null}
  GET  /game/{id}           -> app/games/<id>.html
  GET  /samples/baseline.json
  /static/*  and  /*        -> app/static (the viewer's assets, both absolute and root-relative paths)
"""
from __future__ import annotations

import json
import os
import re
import secrets
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from . import agent
except ImportError:  # e.g. `uvicorn server:app` from inside app/
    import agent  # type: ignore

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
STATIC_DIR = APP_DIR / "static"
GAMES_DIR = APP_DIR / "games"
SAMPLES_DIR = ROOT / "samples"
FIXTURES_DIR = APP_DIR / "fixtures"

agent.load_env()

app = FastAPI(title="Brain Twin laptop server", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --------------------------------------------------------------------------------------------------
# Job store
# --------------------------------------------------------------------------------------------------

_jobs: Dict[str, Dict[str, Any]] = {}
_jobs_lock = threading.Lock()

_SID_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"


def _new_sid() -> str:
    return "".join(secrets.choice(_SID_ALPHABET) for _ in range(6))


def _job_log(job_id: str, line: str) -> None:
    line = " ".join(str(line).split())
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["log"].append(line)
    print(f"[finish {job_id}] {line}", flush=True)


def _modal_base_url() -> str:
    agent.load_env(override=True)  # .env may be filled in after the server started
    return (os.environ.get("MODAL_BASE_URL") or "").strip().rstrip("/")


def _fetch_summary(sid: str, log) -> Dict[str, Any]:
    base = _modal_base_url()
    if not base:
        raise RuntimeError("MODAL_BASE_URL is not set in .env and no summary was supplied in the request")
    url = f"{base}/session/{sid}/finish"
    log(f"Fetching session summary from Modal ({url})")
    with httpx.Client(timeout=120.0) as client:
        resp = client.post(url)  # CONTRACTS.md: POST /session/{sid}/finish
        if resp.status_code == 405:
            log("Modal answered 405 to POST; retrying as GET")
            resp = client.get(url)
        if resp.status_code >= 400:
            raise RuntimeError(f"Modal /session/{sid}/finish returned HTTP {resp.status_code}: {resp.text[:200]}")
        data = resp.json()
    if not isinstance(data, dict):
        raise RuntimeError("Modal finish response is not a JSON object")
    if "summary" in data and isinstance(data["summary"], dict) and "systems" not in data:
        data = data["summary"]
    if not data.get("systems") and not data.get("timeline"):
        detail = data.get("error") or data.get("detail") or "no predictions in the summary"
        raise RuntimeError(f"Modal has no session data for '{sid}' ({detail}). Was anything filmed with this sid?")
    data.setdefault("sid", sid)
    log(f"Summary received: {data.get('n_seconds_predicted', '?')} s predicted over {data.get('duration_s', '?')} s")
    return data


def _load_baseline(log) -> Optional[Any]:
    path = SAMPLES_DIR / "baseline.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        log("Loaded samples/baseline.json")
        return data
    except Exception as e:  # noqa: BLE001
        log(f"Ignoring unreadable samples/baseline.json ({type(e).__name__})")
        return None


def _run_finish_job(job_id: str, sid: str, summary: Optional[Dict[str, Any]]) -> None:
    def log(line: str) -> None:
        _job_log(job_id, line)

    try:
        t0 = time.time()
        if summary is None:
            summary = _fetch_summary(sid, log)
        else:
            summary = dict(summary)
            summary.setdefault("sid", sid)
            log("Using the session summary supplied with the request")
        baseline = _load_baseline(log)
        plan, game_id = agent.run_job(summary, baseline, log)
        with _jobs_lock:
            job = _jobs[job_id]
            job.update({"state": "done", "plan": plan, "game_id": game_id, "game_url": f"/game/{game_id}",
                        "finished": time.time(), "seconds": round(time.time() - t0, 1)})
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        msg = f"{type(e).__name__}: {' '.join(str(e).split())[:300]}"
        log(f"Error: {msg}")
        with _jobs_lock:
            job = _jobs[job_id]
            job.update({"state": "error", "error": msg, "finished": time.time()})


# --------------------------------------------------------------------------------------------------
# API routes (declared before the static mounts so they take precedence)
# --------------------------------------------------------------------------------------------------


class FinishBody(BaseModel):
    sid: Optional[str] = None
    summary: Optional[Dict[str, Any]] = None


@app.get("/", include_in_schema=False)
def index() -> Any:
    index_html = STATIC_DIR / "index.html"
    if index_html.exists():
        return FileResponse(index_html, media_type="text/html; charset=utf-8")
    return HTMLResponse(
        "<!doctype html><meta charset=utf-8><title>Brain Twin</title>"
        "<body style='font-family:system-ui;background:#0b0f1a;color:#e8ecf5;padding:40px'>"
        "<h1>Brain Twin</h1><p>The viewer (app/static/index.html) is not built yet.</p>"
        "<p>API: <code>GET /config</code>, <code>POST /finish</code>, <code>GET /finish/{job_id}</code>, "
        "<code>GET /game/{id}</code>, <code>GET /samples/baseline.json</code></p></body>", status_code=200)


@app.get("/config")
def config(sid: Optional[str] = None) -> Dict[str, str]:
    sid = (sid or "").strip()
    if not sid or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", sid):
        sid = _new_sid()
    return {"modal_base_url": _modal_base_url(), "sid": sid}


@app.post("/finish")
def finish(body: FinishBody) -> Dict[str, str]:
    sid = (body.sid or "").strip() or _new_sid()
    job_id = secrets.token_hex(4)
    with _jobs_lock:
        _jobs[job_id] = {"state": "running", "log": [], "plan": None, "game_url": None, "game_id": None,
                         "error": None, "sid": sid, "started": time.time(), "finished": None}
    _job_log(job_id, f"Finish requested for session {sid}")
    thread = threading.Thread(target=_run_finish_job, args=(job_id, sid, body.summary), name=f"finish-{job_id}",
                              daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/finish/{job_id}")
def finish_status(job_id: str) -> Dict[str, Any]:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="unknown job_id")
        return {"state": job["state"], "log": list(job["log"]), "plan": job["plan"], "game_url": job["game_url"],
                "error": job["error"], "sid": job["sid"], "job_id": job_id}


@app.get("/finish")
def finish_list() -> List[Dict[str, Any]]:
    with _jobs_lock:
        return [{"job_id": k, "state": v["state"], "sid": v["sid"], "game_url": v["game_url"], "started": v["started"]}
                for k, v in _jobs.items()]


_GAME_ID_RE = re.compile(r"^[A-Za-z0-9_-]+(\.html)?$")


@app.get("/game/{game_id}")
def game(game_id: str) -> Any:
    if not _GAME_ID_RE.match(game_id):
        raise HTTPException(status_code=404, detail="no such game")
    game_id = game_id[:-5] if game_id.endswith(".html") else game_id
    path = GAMES_DIR / f"{game_id}.html"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no such game")
    return FileResponse(path, media_type="text/html; charset=utf-8", headers={"Cache-Control": "no-store"})


@app.get("/game/{game_id}/plan")
def game_plan(game_id: str) -> Any:
    if not _GAME_ID_RE.match(game_id):
        raise HTTPException(status_code=404, detail="no such game")
    path = GAMES_DIR / f"{game_id}.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no such game")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/samples/baseline.json")
def baseline() -> Any:
    path = SAMPLES_DIR / "baseline.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="samples/baseline.json not found")
    return FileResponse(path, media_type="application/json")


@app.get("/fixtures/{name}")
def fixture(name: str) -> Any:
    if not re.fullmatch(r"[A-Za-z0-9_-]+\.json", name):
        raise HTTPException(status_code=404, detail="no such fixture")
    path = FIXTURES_DIR / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no such fixture")
    return FileResponse(path, media_type="application/json")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {"ok": True, "gemini_key": bool(os.environ.get("GEMINI_API_KEY")), "modal_base_url": _modal_base_url(),
            "mock_gemini": os.environ.get("MOCK_GEMINI", "") in ("1", "true", "yes"),
            "viewer_present": (STATIC_DIR / "index.html").exists(), "jobs": len(_jobs)}


# --------------------------------------------------------------------------------------------------
# Static: /static/* and, as a fallback for root-relative asset paths, /* -> app/static
# --------------------------------------------------------------------------------------------------

GAMES_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR), check_dir=False), name="static")
app.mount("/", StaticFiles(directory=str(STATIC_DIR), check_dir=False), name="static-root")
