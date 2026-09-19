"""Brain Twin - Gemini game-building agent (see CONTRACTS.md, "Agent").

Pipeline (run_job):  analyse(summary, baseline) -> plan
                     generate_game(plan)        -> single-file HTML
                     check_game(html_path)      -> console errors + screenshot (headless Chromium)
                     one repair pass on errors, then save app/games/<id>.html and <id>.json

Degrades gracefully: MOCK_GEMINI=1, a missing GEMINI_API_KEY, or any Gemini step that still fails after
two attempts switches to the hand-made game in app/games/_fallback_motion.html and says so in the log.

Model names are not hard-coded: the first call lists models via the API and picks the newest working
"*-pro" (code) and "*-flash" (analysis), probing each so retired names are skipped; GEMINI_MODEL_CODE /
GEMINI_MODEL_FAST in .env override that.
"""
from __future__ import annotations

import datetime as _dt
import json
import logging
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from .regions import GAME_TARGETS, SYSTEMS, by_id
except ImportError:  # imported as a plain module / run as a script
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from regions import GAME_TARGETS, SYSTEMS, by_id  # type: ignore

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
GAMES_DIR = APP_DIR / "games"
FALLBACK_GAME = GAMES_DIR / "_fallback_motion.html"
ENV_PATH = ROOT / ".env"

LogFn = Callable[[str], None]

# --------------------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------------------


def load_env(override: bool = False) -> None:
    """Load ROOT/.env into os.environ. Also tolerates a .env that holds just the bare Gemini key."""
    if not ENV_PATH.exists():
        return
    try:
        from dotenv import load_dotenv

        dotenv_logger = logging.getLogger("dotenv.main")
        prev = dotenv_logger.level
        dotenv_logger.setLevel(logging.ERROR)  # a bare-token line makes it warn; we handle that below
        try:
            load_dotenv(ENV_PATH, override=override)
        finally:
            dotenv_logger.setLevel(prev)
    except Exception:
        pass
    if not os.environ.get("GEMINI_API_KEY"):
        try:
            bare = [
                ln.strip()
                for ln in ENV_PATH.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.strip().startswith("#") and "=" not in ln
            ]
            if len(bare) == 1 and len(bare[0]) >= 20 and " " not in bare[0]:
                os.environ["GEMINI_API_KEY"] = bare[0]
        except Exception:
            pass


def _short(err: BaseException, n: int = 180) -> str:
    msg = " ".join(str(err).split())
    return f"{type(err).__name__}: {msg[:n]}" if msg else type(err).__name__


# --------------------------------------------------------------------------------------------------
# Gemini client + model selection
# --------------------------------------------------------------------------------------------------

_client_lock = threading.Lock()
_client_cache: Dict[str, Any] = {}
_models_cache: Dict[str, str] = {}
_model_names_cache: List[str] = []
_bad_models: set = set()

_EXCLUDE_TOKENS = ("image", "tts", "live", "embedding", "embed", "audio", "vision", "veo", "imagen", "aqa",
                   "robotics", "computer-use", "dialog", "transcribe", "translate", "omni", "customtools",
                   "deep-research", "thinking")


class ModelUnavailable(RuntimeError):
    """The API says this model name cannot be used (404 / retired); pick another one."""


def _client():
    """Lazily build the google-genai client (import happens here so the server starts without the SDK)."""
    load_env()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set (put it in .env)")
    with _client_lock:
        if _client_cache.get("key") == key and _client_cache.get("client") is not None:
            return _client_cache["client"]
        from google import genai

        logging.getLogger("google_genai").setLevel(logging.ERROR)  # hide the SDK's AFC chatter on the stage console
        client = genai.Client(api_key=key)
        _client_cache["key"] = key
        _client_cache["client"] = client
        return client


def _version_of(name: str) -> float:
    m = re.search(r"gemini-(\d+)(?:\.(\d+))?", name)
    if not m:
        return 0.0
    return float(f"{m.group(1)}.{m.group(2) or 0}")


def _is_stable(name: str) -> bool:
    return re.search(r"preview|exp|latest", name) is None


def _rank_key(name: str):
    # best first when sorted descending: newest version > stable > non-lite > non-8b > plainest name
    return (_version_of(name), _is_stable(name), "lite" not in name, "8b" not in name, -len(name), name)


def _clean_name(n: str) -> str:
    return n.split("/", 1)[-1] if n.startswith("models/") else n


def rank_models(names: List[str], bad: Optional[set] = None) -> Dict[str, List[str]]:
    """Ranked candidate lists: {"pro": [...], "flash": [...]} (best first). Pure function."""
    bad = bad or set()
    usable = []
    for n in names:
        n = _clean_name(n)
        low = n.lower()
        if not low.startswith("gemini") or n in bad:
            continue
        if any(tok in low for tok in _EXCLUDE_TOKENS):
            continue
        usable.append(n)
    pro = sorted([n for n in usable if "pro" in n.lower()], key=_rank_key, reverse=True)
    flash = sorted([n for n in usable if "flash" in n.lower()], key=_rank_key, reverse=True)
    return {"pro": pro, "flash": flash}


def choose_models(names: List[str], bad: Optional[set] = None) -> Dict[str, Optional[str]]:
    """Top picks without probing: newest *pro* for code, newest *flash* for analysis (a preview only loses to a
    stable model of the same or newer version). Unit-testable."""
    r = rank_models(names, bad)
    code = r["pro"][0] if r["pro"] else (r["flash"][0] if r["flash"] else None)
    fast = r["flash"][0] if r["flash"] else (r["pro"][0] if r["pro"] else None)
    return {"code": code, "fast": fast}


def _is_not_found(err: BaseException) -> bool:
    msg = str(err)
    code = getattr(err, "code", None) or getattr(err, "status_code", None)
    return code == 404 or "404" in msg[:40] or "NOT_FOUND" in msg or "no longer available" in msg or "not found" in msg.lower()


def _recommended_model(err: BaseException) -> Optional[str]:
    m = re.search(r"use models/([A-Za-z0-9._-]+)", str(err))
    return m.group(1) if m else None


def _probe(model: str) -> Tuple[bool, Optional[str]]:
    """Tiny generate call. Returns (usable, recommended_replacement). Only a 404-style answer counts as unusable."""
    from google.genai import types

    try:
        _client().models.generate_content(
            model=model, contents="Reply with OK.",
            config=types.GenerateContentConfig(max_output_tokens=8, http_options=types.HttpOptions(timeout=20_000)))
        return True, None
    except Exception as e:  # noqa: BLE001
        if _is_not_found(e):
            return False, _recommended_model(e)
        return True, None  # quota / transient errors: the model exists, keep it


def _list_model_names() -> List[str]:
    if _model_names_cache:
        return list(_model_names_cache)
    names: List[str] = []
    for m in _client().models.list():
        name = getattr(m, "name", "") or ""
        actions = getattr(m, "supported_actions", None) or []
        if actions and "generateContent" not in actions:
            continue
        if name:
            names.append(_clean_name(name))
    _model_names_cache.extend(names)
    return names


def _first_usable(candidates: List[str], role: str, log: Optional[LogFn], all_names: List[str]) -> Optional[str]:
    queue = [c for c in candidates if c not in _bad_models]
    tried = 0
    while queue and tried < 5:
        name = queue.pop(0)
        tried += 1
        ok, recommended = _probe(name)
        if ok:
            return name
        _bad_models.add(name)
        if log:
            log(f"{name} is not available for this key" + (f"; API suggests {recommended}" if recommended else ""))
        if recommended and recommended in all_names and recommended not in _bad_models and recommended not in queue:
            queue.insert(0, recommended)
    return None


def pick_models(log: Optional[LogFn] = None) -> Dict[str, str]:
    """Return {"code": ..., "fast": ...}. Lists models once per process, probes candidates so retired names are
    skipped, honours GEMINI_MODEL_CODE / GEMINI_MODEL_FAST from .env (never probed, never replaced)."""
    load_env()
    if _models_cache.get("code") and _models_cache.get("fast"):
        return dict(_models_cache)
    code = (os.environ.get("GEMINI_MODEL_CODE") or "").strip()
    fast = (os.environ.get("GEMINI_MODEL_FAST") or "").strip()
    if not (code and fast):
        names = _list_model_names()
        ranked = rank_models(names, _bad_models)
        if log:
            log(f"Gemini lists {len(names)} models; newest pro: {', '.join(ranked['pro'][:3]) or '-'}; "
                f"newest flash: {', '.join(ranked['flash'][:3]) or '-'}")
        if not code:
            code = _first_usable(ranked["pro"], "code", log, names) or _first_usable(ranked["flash"], "code", log, names) or ""
        if not fast:
            fast = _first_usable(ranked["flash"], "fast", log, names) or code
    if not code or not fast:
        raise RuntimeError("No usable Gemini pro/flash model found (set GEMINI_MODEL_CODE / GEMINI_MODEL_FAST in .env)")
    _models_cache.update({"code": code, "fast": fast})
    return {"code": code, "fast": fast}


def _resp_text(resp) -> str:
    text = None
    try:
        text = resp.text
    except Exception:
        text = None
    if text:
        return text
    reason = None
    try:
        cand = (resp.candidates or [None])[0]
        reason = getattr(cand, "finish_reason", None)
        if cand is not None and cand.content and cand.content.parts:
            text = "".join(getattr(p, "text", "") or "" for p in cand.content.parts)
    except Exception:
        pass
    if text:
        return text
    fb = getattr(resp, "prompt_feedback", None)
    raise RuntimeError(f"Empty response from Gemini (finish_reason={reason}, prompt_feedback={fb})")


def _gen(model: str, prompt: str, *, json_schema: Optional[dict] = None, temperature: float = 0.7,
         timeout_s: float = 120.0, system: Optional[str] = None) -> str:
    from google.genai import types

    client = _client()
    cfg: Dict[str, Any] = {"temperature": temperature, "http_options": types.HttpOptions(timeout=int(timeout_s * 1000))}
    if system:
        cfg["system_instruction"] = system
    if json_schema is not None:
        cfg["response_mime_type"] = "application/json"
        cfg["response_schema"] = json_schema
    try:
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=types.GenerateContentConfig(**cfg))
        except (TypeError, ValueError):
            if json_schema is None:
                raise
            cfg.pop("response_schema", None)  # SDK/model rejected the schema: rely on JSON mode + strict prompt
            resp = client.models.generate_content(model=model, contents=prompt, config=types.GenerateContentConfig(**cfg))
    except Exception as e:  # noqa: BLE001
        if _is_not_found(e):
            _bad_models.add(model)
            _models_cache.clear()  # next pick_models() walks the ranking again without this name
            raise ModelUnavailable(f"{model} is not available: {_short(e, 120)}") from e
        raise
    return _resp_text(resp)


# --------------------------------------------------------------------------------------------------
# Session arithmetic (deterministic, used for evidence and as a safety net for the model's choice)
# --------------------------------------------------------------------------------------------------


def _ensure_systems(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = summary.get("systems") or []
    if rows:
        return rows
    timeline = summary.get("timeline") or []
    acc: Dict[str, List[float]] = {}
    for entry in timeline:
        for sid, z in (entry.get("z") or {}).items():
            acc.setdefault(sid, []).append(float(z))
    rows = []
    for s in SYSTEMS:
        v = acc.get(s["id"]) or []
        if not v:
            continue
        rows.append({"id": s["id"], "label": s["label"], "mean": sum(v) / len(v), "peak": max(v),
                     "frac_active": sum(1 for x in v if x > 1.0) / len(v), "rank": 0})
    rows.sort(key=lambda r: -r["mean"])
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    summary["systems"] = rows
    return rows


def _baseline_map(baseline: Optional[Any]) -> Dict[str, Dict[str, Any]]:
    if not baseline:
        return {}
    sysb = baseline.get("systems", baseline) if isinstance(baseline, dict) else baseline
    out: Dict[str, Dict[str, Any]] = {}
    if isinstance(sysb, list):
        for row in sysb:
            if isinstance(row, dict) and row.get("id"):
                out[row["id"]] = row
    elif isinstance(sysb, dict):
        for k, v in sysb.items():
            if isinstance(v, dict):
                out[k] = v
    return out


def local_ranking(summary: Dict[str, Any], baseline: Optional[Any]) -> Tuple[List[Dict[str, Any]], bool]:
    """Game-target systems, least driven first. drive = mean + frac_active, or the deviation from baseline."""
    bmap = _baseline_map(baseline)
    rows = []
    for row in _ensure_systems(summary):
        sid = row.get("id")
        if sid not in GAME_TARGETS:
            continue
        mean = float(row.get("mean") or 0.0)
        frac = float(row.get("frac_active") or 0.0)
        b = bmap.get(sid) or {}
        bmean = b.get("mean")
        entry = {"id": sid, "label": by_id(sid)["label"], "mean": round(mean, 3), "peak": row.get("peak"),
                 "frac_active": round(frac, 3), "rank": row.get("rank")}
        if bmean is not None:
            bfrac = float(b.get("frac_active") or 0.0)
            std = float(b.get("std") or 0.0) or 1.0
            entry["baseline_mean"] = round(float(bmean), 3)
            entry["baseline_frac_active"] = round(bfrac, 3)
            entry["drive"] = round((mean - float(bmean)) / std + (frac - bfrac), 3)
        else:
            entry["drive"] = round(mean + frac, 3)
        rows.append(entry)
    rows.sort(key=lambda r: r["drive"])
    return rows, bool(bmap)


def _evidence(summary: Dict[str, Any], baseline: Optional[Any], target: str, note: str = "") -> Dict[str, Any]:
    ranking, has_baseline = local_ranking(summary, baseline)
    systems = _ensure_systems(summary)
    tgt = next((r for r in ranking if r["id"] == target), None) or {}
    most = systems[0] if systems else {}
    ev: Dict[str, Any] = {
        "target": {k: tgt.get(k) for k in ("mean", "peak", "frac_active", "rank") if k in tgt},
        "target_rank_of": len(systems),
        "baseline": ({"mean": tgt.get("baseline_mean"), "frac_active": tgt.get("baseline_frac_active")}
                     if has_baseline and "baseline_mean" in tgt else None),
        "most_driven": {"id": most.get("id"), "label": most.get("label"), "mean": most.get("mean")} if most else None,
        "candidates_least_driven_first": ranking[:5],
        "n_seconds_predicted": summary.get("n_seconds_predicted"),
        "duration_s": summary.get("duration_s"),
        "model_disclaimer": "Predicted response of an average brain (TRIBE v2 encoding model), not a measurement of the user.",
    }
    if note:
        ev["note"] = note
    return ev


# --------------------------------------------------------------------------------------------------
# 1. Analyse
# --------------------------------------------------------------------------------------------------

ANALYSIS_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "target_system": {"type": "STRING", "enum": list(GAME_TARGETS)},
        "why": {"type": "STRING"},
        "evidence_note": {"type": "STRING"},
        "game": {
            "type": "OBJECT",
            "properties": {
                "title": {"type": "STRING"},
                "concept": {"type": "STRING"},
                "mechanic": {"type": "STRING"},
                "controls": {"type": "STRING", "enum": ["mouse", "keyboard"]},
            },
            "required": ["title", "concept", "mechanic", "controls"],
        },
    },
    "required": ["target_system", "why", "evidence_note", "game"],
}


def _systems_blurbs() -> str:
    lines = []
    for s in SYSTEMS:
        flag = "game_target=true" if s["game_target"] else "game_target=FALSE (never choose)"
        hint = f' Game hint: {s["game_hint"]}' if s["game_hint"] else ""
        lines.append(f'- {s["id"]}: "{s["label"]}" - {s["blurb"]} [{flag}]{hint}')
    return "\n".join(lines)


def build_analysis_prompt(summary: Dict[str, Any], baseline: Optional[Any]) -> str:
    systems = _ensure_systems(summary)
    ranking, has_baseline = local_ranking(summary, baseline)
    regions = sorted(summary.get("regions") or [], key=lambda r: -float(r.get("mean") or 0))
    top_regions = [{k: r.get(k) for k in ("name", "system", "mean", "frac_active")} for r in regions[:10]]
    low_regions = [{k: r.get(k) for k in ("name", "system", "mean", "frac_active")} for r in regions[-6:]]
    bmap = _baseline_map(baseline)
    baseline_txt = json.dumps({k: {kk: v.get(kk) for kk in ("mean", "frac_active", "std") if kk in v}
                               for k, v in bmap.items()}) if bmap else "none available"
    duration = summary.get("duration_s") or "?"
    n = summary.get("n_seconds_predicted") or len(summary.get("timeline") or [])
    return f"""You are the analysis step of "Brain Twin", a live stage demo.
A phone filmed the world for about {duration} s. TRIBE v2, a brain encoding model, predicted the AVERAGE human
brain's fMRI response (fsaverage5 cortex, z-scores at 1 Hz, {n} predicted seconds) to that footage - audio and
video only, no text. This is a simulator of an average brain, not a reading of the user's brain; the "why"
must make that clear in plain words.

Functional systems (only game_target=true systems may be chosen):
{_systems_blurbs()}

Session statistics per system, sorted most -> least driven (rank 1 = most driven). mean/peak are predicted
z-scores; frac_active = share of seconds above z = 1.0:
{json.dumps(systems)}

Most driven regions: {json.dumps(top_regions)}
Least driven regions: {json.dumps(low_regions)}

Baseline (typical values from reference clips; when present judge "under-driven" RELATIVE to it): {baseline_txt}

Arithmetic ranking of the game-target systems, least driven first
(drive = {"deviation from baseline" if has_baseline else "mean + frac_active"}):
{json.dumps(ranking)}

Task:
1. Choose the single most under-driven game-target system. It is normally the first row of the ranking;
   deviate only for a good reason and explain it in evidence_note.
2. "why": one or two sentences (max 45 words) for a general audience on a big screen: say what the camera did
   not show much of, quote one number (for example the mean z or the rank), and make clear this is the
   predicted response of an average brain, not the player's.
3. "evidence_note": one sentence with the key numbers behind the choice.
4. "game": design a 60-second single-file browser canvas game for a laptop that drives exactly that system,
   following its game hint. title: 2-4 words. concept: 2-3 sentences of what the player sees and does.
   mechanic: a specific paragraph (what appears, how scoring works, how it gets harder, what ends a round).
   controls: "mouse" or "keyboard" (pick the one that suits the mechanic best).

Return only JSON with exactly these keys: target_system, why, evidence_note, game{{title, concept, mechanic, controls}}.
"""


def _parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def analyse(summary: Dict[str, Any], baseline: Optional[Any] = None, *, models: Optional[Dict[str, str]] = None,
            log: Optional[LogFn] = None, attempt: int = 1) -> Dict[str, Any]:
    """Fast model, JSON mode -> plan {target_system, why, evidence, game{title, concept, mechanic, controls}}."""
    models = models or pick_models(log)
    prompt = build_analysis_prompt(summary, baseline)
    if log:
        log(f"Asking {models['fast']} which system to target (attempt {attempt})")
    text = _gen(models["fast"], prompt, json_schema=ANALYSIS_SCHEMA, temperature=0.4, timeout_s=90)
    data = _parse_json(text)
    target = str(data.get("target_system") or "").strip()
    ranking, _ = local_ranking(summary, baseline)
    if target not in GAME_TARGETS:
        fallback_target = ranking[0]["id"] if ranking else "motion"
        if log:
            log(f"Model chose '{target or '?'}', which is not a game target; using {fallback_target} instead")
        target = fallback_target
    game = data.get("game") or {}
    controls = str(game.get("controls") or "keyboard").lower()
    plan = {
        "target_system": target,
        "target_label": by_id(target)["label"],
        "why": " ".join(str(data.get("why") or "").split()) or _default_why(summary, baseline, target),
        "evidence": _evidence(summary, baseline, target, note=str(data.get("evidence_note") or "")),
        "game": {
            "title": str(game.get("title") or f"{by_id(target)['label']} trainer").strip()[:60],
            "concept": str(game.get("concept") or "").strip(),
            "mechanic": str(game.get("mechanic") or "").strip(),
            "controls": "mouse" if controls.startswith("mouse") else "keyboard",
        },
    }
    if not plan["game"]["concept"] or not plan["game"]["mechanic"]:
        raise RuntimeError("Analysis JSON is missing the game concept/mechanic")
    return plan


def _default_why(summary: Dict[str, Any], baseline: Optional[Any], target: str) -> str:
    ranking, has_baseline = local_ranking(summary, baseline)
    tgt = next((r for r in ranking if r["id"] == target), None) or {}
    label = by_id(target)["label"]
    n = len(_ensure_systems(summary))
    mean = tgt.get("mean")
    rank = tgt.get("rank")
    rel = " below its usual baseline" if has_baseline and "baseline_mean" in tgt else ""
    mean_txt = f"a mean of {mean:.2f} z" if isinstance(mean, (int, float)) else "the lowest activity"
    return (f"The session barely engaged {label}: the average brain's predicted response there was {mean_txt}{rel}, "
            f"rank {rank} of {n} systems. That is a simulated average brain, not a reading of yours - this game gives it work.")


# --------------------------------------------------------------------------------------------------
# 2. Generate
# --------------------------------------------------------------------------------------------------

GAME_CONTRACT = """single file, no external requests, canvas-based, laptop keyboard/mouse, 60 s round with visible timer,
header shows target system label + one-sentence "why" from the plan, no `alert()`,
on round end `window.parent.postMessage({type: "brain-game-score", score, target: "<system id>"}, "*")`,
exposes `window.startDemo()` that plays itself for 25 s (for future TRIBE scoring)."""

CODE_SYSTEM_INSTRUCTION = ("You are a senior game developer who writes small, polished, dependency-free HTML5 canvas "
                           "games. You respond with a single complete HTML document and nothing else.")


def build_game_prompt(plan: Dict[str, Any]) -> str:
    target = plan["target_system"]
    sysinfo = by_id(target)
    game = plan.get("game") or {}
    return f"""Build a small browser game for "Brain Twin", a live demo. A phone filmed the world, a brain encoding model
(TRIBE v2) predicted how the AVERAGE human brain's cortex would respond to that footage, and the least-driven
functional system became the target. Your game gets that system working for 60 seconds on a laptop.

TARGET SYSTEM: {sysinfo['label']}  (system id: "{target}")
WHAT IT DOES: {sysinfo['blurb']}
DESIGN HINT FOR THIS SYSTEM: {sysinfo['game_hint'] or 'n/a'}
WHY (show this text in the header, verbatim): {plan.get('why', '')}

GAME PLAN (follow it):
- title: {game.get('title', '')}
- concept: {game.get('concept', '')}
- mechanic: {game.get('mechanic', '')}
- controls: {game.get('controls', 'keyboard')}

CONTRACT (from CONTRACTS.md, all mandatory):
{GAME_CONTRACT}

Clarifications of the contract:
- ONE complete HTML document with inline <style> and <script>. No <script src>, no <link>, no @import, no fetch/XHR,
  no images, fonts or data from the network. Everything is drawn on a <canvas> at runtime (procedural graphics).
- Must work opened from file:// and inside an iframe. Plain ES2017 JavaScript, no modules, no frameworks.
- Header (HTML above the canvas, one compact bar): the game title, the text "Target: {sysinfo['label']}", and the
  why sentence above. Also show the countdown timer and the score (in the header and/or on the canvas).
  The canvas fills the rest of the window and handles resize.
- Start screen with a one-line instruction. The round starts on click or any key. A round is exactly 60 seconds
  with a visible countdown. When it ends, show the final score on screen (never alert/prompt/confirm) and call
  window.parent.postMessage({{type: "brain-game-score", score: <integer>, target: "{target}"}}, "*") exactly once
  per round. Offer "play again".
- window.startDemo() must start a round that plays itself with a simple bot or scripted inputs for 25 seconds and
  then ends the round (posting the score). It must not need any user gesture. Never call it automatically.
- WebAudio sound effects are welcome but optional: create the AudioContext lazily on the first user gesture and
  wrap every audio call in try/catch so demo mode stays silent and error-free.
- No console errors, no uncaught exceptions, no unhandled promise rejections. Guard every DOM lookup.
- Keyboard: use event.code, preventDefault on arrows/space. Mouse: track position on the canvas. Support both
  where it makes sense, with "{game.get('controls', 'keyboard')}" as the primary input.
- Polish: readable typography, a coherent colour scheme, particles or flashes as feedback, a difficulty ramp,
  and a satisfying end screen. Keep the whole file under about 900 lines.
- Honest tone: the header may say "average brain"; never claim to read the player's mind.

Output: ONLY the HTML document, starting with <!DOCTYPE html> and ending with </html>. No markdown fences, no
commentary before or after.
"""


def extract_html(text: str) -> str:
    """Strip markdown fences; if several code blocks exist take the largest HTML-looking one."""
    text = text.strip()
    blocks = re.findall(r"```[a-zA-Z0-9_-]*[ \t]*\r?\n(.*?)```", text, re.S)
    candidates: List[str] = list(blocks)
    if re.search(r"<!doctype\s+html|<html", text, re.I):
        candidates.append(text)
    html_like = [c for c in candidates if re.search(r"<!doctype\s+html|<html|<canvas", c, re.I)]
    best = max(html_like or candidates or [text], key=len).strip()
    best = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", best)
    best = re.sub(r"\s*```$", "", best)
    m = re.search(r"<!doctype\s+html|<html", best, re.I)
    if m:
        best = best[m.start():]
    m = re.search(r"</html>", best, re.I)
    if m:
        best = best[: m.end()]
    return best.strip() + "\n"


def _validate_html(html: str) -> List[str]:
    problems = []
    low = html.lower()
    if "<canvas" not in low and "createelement('canvas')" not in low and 'createelement("canvas")' not in low:
        problems.append("no <canvas> element")
    if "startdemo" not in low:
        problems.append("window.startDemo is missing")
    if "postmessage" not in low or "brain-game-score" not in low:
        problems.append("round-end postMessage({type:'brain-game-score'}) is missing")
    if re.search(r"<script[^>]+src=|<link[^>]+href=|@import\s|fetch\(|XMLHttpRequest", html, re.I):
        problems.append("external resource or network call found (not allowed)")
    if re.search(r"\balert\s*\(", html):
        problems.append("alert() found (not allowed)")
    if "</html>" not in low:
        problems.append("document is truncated (no </html>)")
    return problems


def generate_game(plan: Dict[str, Any], *, models: Optional[Dict[str, str]] = None, log: Optional[LogFn] = None,
                  attempt: int = 1) -> str:
    """Strongest code model -> complete single-file HTML game that satisfies the contract."""
    models = models or pick_models(log)
    prompt = build_game_prompt(plan)
    if attempt > 1:
        prompt += ("\nREMINDER: the previous attempt was rejected. Output the COMPLETE document, include "
                   "window.startDemo(), the brain-game-score postMessage, a <canvas>, and no external resources.\n")
    if log:
        log(f"Asking {models['code']} for a game (attempt {attempt})")
    t0 = time.time()
    text = _gen(models["code"], prompt, temperature=0.8, timeout_s=420, system=CODE_SYSTEM_INSTRUCTION)
    html = extract_html(text)
    problems = _validate_html(html)
    if log:
        log(f"Received {len(html) // 1024} KB of HTML in {time.time() - t0:.0f} s")
    if problems:
        raise RuntimeError("generated HTML violates the contract: " + "; ".join(problems))
    return html


def repair_game(html: str, problems: List[str], *, models: Optional[Dict[str, str]] = None,
                log: Optional[LogFn] = None) -> str:
    models = models or pick_models(log)
    prompt = f"""The HTML5 canvas game below was loaded in headless Chromium for 5 seconds and produced these problems:
{chr(10).join('- ' + p for p in problems)}

Fix them without changing the game's concept. Keep the whole contract:
{GAME_CONTRACT}

Return the COMPLETE corrected HTML document only (starting with <!DOCTYPE html>, ending with </html>), no
markdown fences, no commentary.

--- CURRENT HTML ---
{html}
"""
    if log:
        log(f"Asking {models['code']} to repair the game")
    t0 = time.time()
    text = _gen(models["code"], prompt, temperature=0.4, timeout_s=420, system=CODE_SYSTEM_INSTRUCTION)
    fixed = extract_html(text)
    bad = _validate_html(fixed)
    if log:
        log(f"Repaired HTML received ({len(fixed) // 1024} KB, {time.time() - t0:.0f} s)")
    if bad:
        raise RuntimeError("repaired HTML violates the contract: " + "; ".join(bad))
    return fixed


# --------------------------------------------------------------------------------------------------
# 3. Check (headless Chromium via Playwright)
# --------------------------------------------------------------------------------------------------


def check_game(html_path: Path, seconds: float = 5.0, log: Optional[LogFn] = None,
               expect_label: Optional[str] = None) -> Dict[str, Any]:
    """Load file:// in headless Chromium for `seconds`, exercise start + startDemo(), collect console errors,
    save <name>.png next to the file. Returns a dict; raises if Playwright/Chromium are unavailable."""
    from playwright.sync_api import sync_playwright

    html_path = Path(html_path)
    png = html_path.with_suffix(".png")
    errors: List[str] = []
    external: List[str] = []
    result: Dict[str, Any] = {"errors": errors, "external_requests": external, "screenshot": None,
                              "has_canvas": None, "has_start_demo": None, "start_demo_ok": None,
                              "header_has_label": None, "messages": []}
    t0 = time.time()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            page.set_default_timeout(10_000)
            page.add_init_script(
                "window.__bg_msgs=[];window.addEventListener('message',function(e){try{"
                "window.__bg_msgs.push(JSON.parse(JSON.stringify(e.data)))}catch(_){}});")

            def _is_external(url: str) -> bool:
                return not url.startswith(("file://", "data:", "blob:", "about:"))

            page.route(lambda url: _is_external(url), lambda route: route.abort())
            page.on("request", lambda r: external.append(r.url) if _is_external(r.url) else None)
            page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
            page.on("pageerror", lambda e: errors.append(f"Uncaught: {e}"))
            page.goto(html_path.as_uri(), wait_until="load")
            page.wait_for_timeout(700)
            # manual start path: click the canvas area, press a few keys
            try:
                page.mouse.click(640, 460)
                for key in ("ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Space"):
                    page.keyboard.press(key)
            except Exception as e:  # input problems are not game errors
                errors.append(f"input simulation failed: {e}")
            page.wait_for_timeout(600)
            # demo path
            result["start_demo_ok"] = page.evaluate(
                "() => { try { if (typeof window.startDemo === 'function') { window.startDemo(); return true; }"
                " return false; } catch (e) { return 'threw: ' + (e && e.message ? e.message : e); } }")
            if isinstance(result["start_demo_ok"], str):
                errors.append("window.startDemo() " + result["start_demo_ok"])
            remaining = max(0.0, seconds - (time.time() - t0))
            page.wait_for_timeout(int(remaining * 1000))
            info = page.evaluate(
                "(label) => ({ canvas: !!document.querySelector('canvas'),"
                " demo: typeof window.startDemo === 'function',"
                " header: label ? (document.body.innerText || '').indexOf(label) >= 0 : null,"
                " msgs: window.__bg_msgs || [] })", expect_label or "")
            result["has_canvas"] = bool(info.get("canvas"))
            result["has_start_demo"] = bool(info.get("demo"))
            result["header_has_label"] = info.get("header")
            result["messages"] = info.get("msgs") or []
            page.screenshot(path=str(png))
            result["screenshot"] = str(png)
        finally:
            browser.close()
    result["seconds"] = round(time.time() - t0, 1)
    if log:
        log(f"Headless check: {len(errors)} console errors" + (f", {len(external)} external requests" if external else ""))
    return result


def _problems(check: Dict[str, Any]) -> List[str]:
    """Turn a check result into the list of things worth a repair pass."""
    out: List[str] = []
    seen = set()
    for e in check.get("errors") or []:
        line = " ".join(str(e).split())[:300]
        if line not in seen:
            seen.add(line)
            out.append("console error: " + line)
    if check.get("has_canvas") is False:
        out.append("no <canvas> element was rendered")
    if check.get("has_start_demo") is False:
        out.append("window.startDemo is not a function")
    if check.get("external_requests"):
        urls = sorted(set(check["external_requests"]))[:5]
        out.append("external requests are not allowed but the page requested: " + ", ".join(urls))
    return out[:12]


def _is_fatal(check: Optional[Dict[str, Any]]) -> bool:
    if not check:
        return False
    if check.get("has_canvas") is False:
        return True
    return any(str(e).startswith("Uncaught:") or str(e).startswith("window.startDemo() threw") for e in check.get("errors") or [])


# --------------------------------------------------------------------------------------------------
# Fallback / mock
# --------------------------------------------------------------------------------------------------

FALLBACK_GAME_INFO = {
    "title": "Disc Dodger",
    "concept": "Red discs stream across the screen from every edge while gold ones drift through. Steer a small ship "
               "to dodge the red and catch the gold for 60 seconds as everything speeds up.",
    "mechanic": "Discs spawn at the edges aimed across the play field, some at the player. Gold catch = 10 points plus "
                "2 per combo step; red hit = -15 and a short invulnerability; +1 per second survived. Spawn rate and "
                "speed ramp over the round. Optic-flow background keeps the whole field moving.",
    "controls": "keyboard",
}


def mock_plan(summary: Dict[str, Any], baseline: Optional[Any] = None) -> Dict[str, Any]:
    """Canned plan targeting 'motion' (used with MOCK_GEMINI=1, no key, or when Gemini is unreachable)."""
    target = "motion"
    ranking, has_baseline = local_ranking(summary, baseline)
    tgt = next((r for r in ranking if r["id"] == target), None) or {}
    n = len(_ensure_systems(summary))
    mean = tgt.get("mean")
    frac = tgt.get("frac_active")
    if isinstance(mean, (int, float)):
        why = (f"Almost nothing moved across the camera: the predicted motion cortex (MT+) averaged {mean:.2f} z with "
               f"{(frac or 0) * 100:.0f}% of seconds active, rank {tgt.get('rank')} of {n}. That is an average brain's "
               f"simulated response, not a reading of yours. Time to make it track.")
    else:
        why = ("Hardly anything moved across the camera in this session, so the motion-sensitive cortex of the "
               "average brain stayed quiet. Time to make it track.")
    return {
        "target_system": target,
        "target_label": by_id(target)["label"],
        "why": why,
        "evidence": _evidence(summary, baseline, target, note="canned plan (no Gemini)"),
        "game": dict(FALLBACK_GAME_INFO),
    }


def build_fallback_html(plan: Dict[str, Any]) -> str:
    """Copy the hand-made game, injecting the header text via window.BRAIN_GAME_META."""
    html = FALLBACK_GAME.read_text(encoding="utf-8")
    meta = {
        "target": plan.get("target_system", "motion"),
        "label": plan.get("target_label") or by_id(plan.get("target_system", "motion"))["label"],
        "why": plan.get("why", ""),
        "title": (plan.get("game") or {}).get("title") or FALLBACK_GAME_INFO["title"],
    }
    payload = json.dumps(meta).replace("</", "<\\/")
    placeholder = "/*META*/null/*META*/"
    if placeholder in html:
        return html.replace(placeholder, payload, 1)
    return html.replace("<script>", f"<script>window.BRAIN_GAME_META = {payload};</script>\n<script>", 1)


def _apply_fallback(plan: Dict[str, Any], reason: str, log: LogFn) -> Dict[str, Any]:
    plan = dict(plan)
    original = plan.get("target_system")
    plan["fallback"] = {"reason": reason, "analysed_target": original}
    if original != "motion":
        label = by_id(original)["label"] if original in [s["id"] for s in SYSTEMS] else str(original)
        log(f"Fallback game targets motion, not {label}; header will say so")
        plan["why"] = (f"{label} came out least driven, but the game generator did not deliver, so here is the standby "
                       f"motion game: an average brain's predicted MT+ response is easy to drive with moving discs.")
        plan["target_system"] = "motion"
    plan["target_label"] = by_id("motion")["label"]
    plan["game"] = dict(FALLBACK_GAME_INFO)
    return plan


# --------------------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------------------


def _retry(fn: Callable[[int], Any], attempts: int, log: LogFn, what: str) -> Any:
    last: Optional[BaseException] = None
    for i in range(1, attempts + 1):
        try:
            return fn(i)
        except Exception as e:  # noqa: BLE001 - any failure is retried once, then reported
            last = e
            log(f"{what} attempt {i} failed: {_short(e)}")
            if i < attempts:
                time.sleep(1.5)
    assert last is not None
    raise last


def run_job(summary: Dict[str, Any], baseline: Optional[Any], log_fn: LogFn) -> Tuple[Dict[str, Any], str]:
    """analyse -> generate -> check (+ one repair) -> app/games/<id>.html + <id>.json. Returns (plan, game_id)."""
    t_start = time.time()
    lines: List[str] = []

    def log(msg: str) -> None:
        lines.append(msg)
        try:
            log_fn(msg)
        except Exception:
            pass

    load_env()
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    game_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)
    timings: Dict[str, float] = {}
    summary = dict(summary or {})
    sid = summary.get("sid") or "?"
    systems = _ensure_systems(summary)
    n = summary.get("n_seconds_predicted") or len(summary.get("timeline") or [])
    log(f"Reading {n} s of predicted activity from session {sid}")
    if systems:
        most, least = systems[0], systems[-1]
        log(f"Most driven: {most.get('label')} (mean {float(most.get('mean') or 0):.2f} z); "
            f"least driven overall: {least.get('label')} (mean {float(least.get('mean') or 0):.2f} z)")
    ranking, has_baseline = local_ranking(summary, baseline)
    if has_baseline:
        log(f"Baseline found: judging {len(ranking)} game-target systems relative to it")
    else:
        log(f"No baseline: comparing the {len(ranking)} game-target systems against each other")
    if ranking:
        log("Least-driven candidates: " + ", ".join(f"{r['label']} ({r['drive']:+.2f})" for r in ranking[:3]))

    mock = os.environ.get("MOCK_GEMINI", "").strip() in ("1", "true", "yes")
    models: Optional[Dict[str, str]] = None
    plan: Optional[Dict[str, Any]] = None
    html: Optional[str] = None
    fallback_reason: Optional[str] = None

    if mock:
        log("MOCK_GEMINI=1: skipping Gemini and using the canned motion plan")
        fallback_reason = "MOCK_GEMINI=1"
    elif not os.environ.get("GEMINI_API_KEY"):
        log("GEMINI_API_KEY is not set: using the canned plan and the built-in game")
        fallback_reason = "GEMINI_API_KEY not set"
    else:
        try:
            models = pick_models(log)
            log(f"Gemini models: {models['fast']} for analysis, {models['code']} for the game")
        except Exception as e:  # noqa: BLE001
            log(f"Could not set up Gemini ({_short(e)}); using the canned plan and the built-in game")
            fallback_reason = f"Gemini unavailable: {_short(e)}"

    if models and not fallback_reason:
        t0 = time.time()
        try:
            plan = _retry(lambda i: analyse(summary, baseline, models=pick_models(log), log=log, attempt=i), 2, log, "Analysis")
        except Exception as e:  # noqa: BLE001
            log(f"Analysis failed twice ({_short(e)}); using the canned plan and the built-in game")
            fallback_reason = f"analysis failed: {_short(e)}"
        timings["analyse"] = round(time.time() - t0, 1)
        if plan:
            log(f"Least-driven system: {plan['target_label']} ({timings['analyse']} s)")
            log(f"Why: {plan['why']}")
            log(f"Game idea: {plan['game']['title']} - {plan['game']['concept']}")
            t0 = time.time()
            try:
                html = _retry(lambda i: generate_game(plan, models=pick_models(log), log=log, attempt=i), 2, log, "Game generation")
            except Exception as e:  # noqa: BLE001
                log(f"Game generation failed twice ({_short(e)}); falling back to the built-in motion game")
                fallback_reason = f"generation failed: {_short(e)}"
            timings["generate"] = round(time.time() - t0, 1)

    if plan is None:
        plan = mock_plan(summary, baseline)
        log(f"Least-driven system (canned): {plan['target_label']}")
        log(f"Why: {plan['why']}")

    fallback_used = False
    if html is None:
        plan = _apply_fallback(plan, fallback_reason or "no game generated", log)
        html = build_fallback_html(plan)
        fallback_used = True
        log(f"Using the built-in game '{plan['game']['title']}'")

    html_path = GAMES_DIR / f"{game_id}.html"
    html_path.write_text(html, encoding="utf-8")
    log(f"Saved game to app/games/{game_id}.html ({len(html) // 1024} KB)")

    check: Optional[Dict[str, Any]] = None
    repair: Optional[Dict[str, Any]] = None
    t0 = time.time()
    try:
        check = check_game(html_path, seconds=5.0, log=log, expect_label=plan.get("target_label"))
    except Exception as e:  # noqa: BLE001
        log(f"Headless check skipped ({_short(e)})")
    timings["check"] = round(time.time() - t0, 1)

    problems = _problems(check) if check else []
    if check and not problems:
        pass  # "Headless check: 0 console errors" already logged by check_game
    elif check and problems:
        for pr in problems[:3]:
            log("  " + pr)
        if not fallback_used and models:
            t0 = time.time()
            repair = {"problems": problems, "ok": False}
            try:
                broken_path = GAMES_DIR / f"{game_id}.broken.html"
                broken_path.write_text(html, encoding="utf-8")
                fixed = repair_game(html, problems, models=pick_models(log), log=log)
                html_path.write_text(fixed, encoding="utf-8")
                check2 = check_game(html_path, seconds=5.0, log=log, expect_label=plan.get("target_label"))
                problems2 = _problems(check2)
                repair.update({"problems_after": problems2, "check_after": check2})
                if not problems2:
                    log("Repair worked: the game runs clean")
                    html, check, repair["ok"] = fixed, check2, True
                elif _is_fatal(check2) and not _is_fatal(check):
                    log("Repair made it worse; keeping the first version")
                    html_path.write_text(html, encoding="utf-8")
                elif _is_fatal(check2):
                    log("Still crashing after repair; falling back to the built-in motion game")
                    plan = _apply_fallback(plan, "generated game crashed twice", log)
                    html = build_fallback_html(plan)
                    html_path.write_text(html, encoding="utf-8")
                    fallback_used = True
                    check = check_game(html_path, seconds=3.0, log=log, expect_label=plan.get("target_label"))
                else:
                    log(f"Still {len(problems2)} issue(s) after repair; shipping the repaired game anyway")
                    html, check = fixed, check2
            except Exception as e:  # noqa: BLE001
                log(f"Repair failed ({_short(e)})")
                if _is_fatal(check):
                    log("Generated game crashes; falling back to the built-in motion game")
                    plan = _apply_fallback(plan, "generated game crashed and repair failed", log)
                    html = build_fallback_html(plan)
                    html_path.write_text(html, encoding="utf-8")
                    fallback_used = True
                    try:
                        check = check_game(html_path, seconds=3.0, log=log, expect_label=plan.get("target_label"))
                    except Exception:
                        pass
                else:
                    log("Shipping the game with console warnings")
            timings["repair"] = round(time.time() - t0, 1)

    timings["total"] = round(time.time() - t_start, 1)
    meta = {
        "id": game_id,
        "sid": sid,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "mock": mock,
        "fallback_used": fallback_used,
        "fallback_reason": fallback_reason if fallback_used else None,
        "models": dict(_models_cache) if _models_cache else models,
        "timings": timings,
        "plan": plan,
        "check": check,
        "repair": repair,
        "log": lines,
    }
    (GAMES_DIR / f"{game_id}.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    log(f"Game ready ({timings['total']} s total)")
    return plan, game_id


# --------------------------------------------------------------------------------------------------
# CLI: python3 app/agent.py [summary.json] [baseline.json]
# --------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    summary_path = Path(sys.argv[1]) if len(sys.argv) > 1 else APP_DIR / "fixtures" / "summary_example.json"
    baseline_path = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "samples" / "baseline.json"
    summary_data = json.loads(summary_path.read_text(encoding="utf-8"))
    baseline_data = json.loads(baseline_path.read_text(encoding="utf-8")) if baseline_path.exists() else None
    plan_out, gid = run_job(summary_data, baseline_data, lambda m: print(m, flush=True))
    print(json.dumps({"game_id": gid, "plan": plan_out}, indent=1))
