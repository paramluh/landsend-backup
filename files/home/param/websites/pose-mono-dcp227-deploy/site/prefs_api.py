"""10-29 prefsync (user via main -1e: "my monoplex settings were no caches to browser?"). The playground came back on a NEW quick-tunnel
hostname; the page's settings live in the address bar + localStorage, and localStorage is per origin, so a bare URL on the new hostname came
up with defaults. This module keeps the user's LAST-USED settings on the server, so they survive tunnel / origin / browser changes.

  GET  /api/prefs[?ws=<name>]   {"ws": name, "prefs": {"monoplex": {"q": <settings query>, "t": unix}, "duplex": {...}, "livedup": {...},
                                 "mono_weights": <weights id>, "last": <model>, "t": unix}}   ({} when nothing is saved for that workspace)
  POST /api/prefs  {"ws": name|"", "model": monoplex|duplex|livedup, "q": "<query string>", "mw": <weights id>|absent}
                   the page posts its current settings (the string urlSave builds) on every user change, debounced; agent pages
                   (navigator.webdriver) never post, and a body with hidden / agent_test is ignored (agent runs never write prefs).

Stored in genweb/state/prefs.json: {"version": 1, "ws": {"<ws or ''>": {...as above...}}}, written atomically (tmp + os.replace) under
an flock (two app servers overlap for 30 s during a zero-downtime deploy).
Registered by server.py: prefs_api.setup(app, globals())."""
import fcntl
import json
import os
import re
import time

from aiohttp import web
import actor_selection

HERE = os.path.dirname(os.path.abspath(__file__))
STATE = f"{HERE}/state"
PREFS = f"{STATE}/prefs.json"
MODELS = ("monoplex", "duplex", "livedup")
WS_RE = re.compile(r"^[A-Za-z0-9_-]{0,40}$")
MW_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
MAXQ = 16000


def load():
    try:
        with open(PREFS) as f:
            d = json.load(f)
        if isinstance(d, dict) and isinstance(d.get("ws"), dict):
            return d
    except (OSError, ValueError):
        pass
    return {"version": 1, "ws": {}}


def save(d):
    os.makedirs(STATE, exist_ok=True)
    tmp = f"{PREFS}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, PREFS)


def put(ws, model, q, mw=None):
    """record one model's settings (+ the monoplex weights pick) for a workspace; returns the workspace entry"""
    if model == "monoplex":
        q = actor_selection.normalize_query(q)
    os.makedirs(STATE, exist_ok=True)
    with open(f"{PREFS}.lock", "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = load()
        e = d["ws"].setdefault(ws, {})
        now = round(time.time(), 3)
        e[model] = {"q": q, "t": now}
        if mw:
            e["mono_weights"] = mw
        e["last"] = model
        e["t"] = now
        save(d)
        return e


async def get_prefs(req):
    ws = req.query.get("ws", "") or ""
    if not WS_RE.match(ws):
        return web.json_response({"error": "bad ws"}, status=400)
    prefs = load()["ws"].get(ws, {})
    mono = prefs.get("monoplex")
    if isinstance(mono, dict) and isinstance(mono.get("q"), str):
        mono["q"] = actor_selection.normalize_query(mono["q"])
    return web.json_response({"ws": ws, "prefs": prefs}, headers={"Cache-Control": "no-store"})


async def post_prefs(req):
    try:
        b = await req.json()
    except Exception:                                                   # noqa: BLE001  (any bad body)
        return web.json_response({"error": "bad json"}, status=400)
    if not isinstance(b, dict):
        return web.json_response({"error": "bad body"}, status=400)
    if b.get("hidden") or b.get("agent_test") or b.get("agent"):        # agent / hidden pages never write the user's prefs
        return web.json_response({"ok": True, "ignored": "agent"})
    ws, model, q, mw = str(b.get("ws") or ""), b.get("model"), b.get("q"), b.get("mw")
    if not WS_RE.match(ws) or model not in MODELS or not isinstance(q, str) or len(q) > MAXQ:
        return web.json_response({"error": "bad prefs"}, status=400)
    q = q.lstrip("?")
    if mw is not None and (not isinstance(mw, str) or not MW_RE.match(mw)):
        mw = None
    e = put(ws, model, q, mw if model == "monoplex" else None)
    return web.json_response({"ok": True, "t": e.get("t")})


def setup(app, g):
    app.router.add_get("/api/prefs", get_prefs)
    app.router.add_post("/api/prefs", post_prefs)
