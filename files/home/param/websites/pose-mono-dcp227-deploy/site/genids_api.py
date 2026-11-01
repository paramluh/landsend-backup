"""10-29 genids (user via main -1e: "give all my generations ids so we have a language"). The user-facing id of a generation is the 10-28
short video id v<N> (genweb/VIDS.json {run_id: "v<N>", "_next": N}); batches get B<N>. This module:
  assign_vids(ROOT, RUNS)  the ONE v-id assigner (server.py assign_vids at Generate time + genids/assign.py every 1 s, so Live mic calls get
                           theirs within seconds): every visible run dir without one gets the next v<N> (run-id order), under VIDS.lock;
                           ids already given are never changed. From 10-29 agent runs get none: request "hidden", a label the page hides
                           (window.agentRun), or a live call the page never registered (livedup_import's rule); a dir with no
                           request.json yet waits.
  GET /api/genids          {"t", "b": {batch_id: n}, "rb": {run_id: n}}  B<n> per batch (GEN_IDS.json, written by genids/assign.py; ETag)
  GET /api/gid/{B37|v670}  B: {b, batch, run_id (first sample), runs, v, model, ws}   v: {v, run_id, batch, b, model, ws, hidden}
Registered by server.py: genids_api.setup(app, globals())."""
import fcntl
import json
import os
import re

from aiohttp import web

C = {}
S = {"mt": None, "body": b"{}", "etag": None, "b": {}, "v": {}, "runs": {}, "batches": {}}
RID = re.compile(r"^[0-9]{8}-[0-9]{6}-[0-9a-f]{4}$")
AGENT = re.compile(r"\b(check|test|repro|verify)\b|\(aloedemo\)", re.I)     # = the page's window.agentRun label rule
SKIP = {}                                                                     # run dir -> (request.json mtime, skip)


def hidden_req(q):
    if q.get("hidden") or AGENT.search(str(q.get("label") or "")):
        return True
    return q.get("model") == "livedup" and (not q.get("url_query") or str(q.get("label") or "").startswith("agent test"))


def vid_skip(rd):
    p = f"{rd}/request.json"
    try:
        m = os.path.getmtime(p)
    except OSError:
        return True
    h = SKIP.get(rd)
    if h and h[0] == m:
        return h[1]
    try:
        with open(p) as f:
            q = json.load(f)
    except (OSError, ValueError):
        return True                                                     # mid-write: next pass
    v = not isinstance(q, dict) or hidden_req(q)
    SKIP[rd] = (m, v)
    return v


def assign_vids(root, runs):
    f = f"{root}/VIDS.json"
    with open(f"{root}/VIDS.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            with open(f) as fh:
                d = json.load(fh)
        except OSError:
            d = {}
        except ValueError:
            return 0                                                    # never overwrite a map we could not read
        if not isinstance(d, dict):
            return 0
        n, add = int(d.get("_next", 1)), 0
        for rid in sorted(os.listdir(runs)):
            if RID.match(rid) and rid not in d and not vid_skip(f"{runs}/{rid}"):
                d[rid], n, add = f"v{n}", n + 1, add + 1
        if add:
            d["_next"] = n
            tmp = f"{f}.tmp{os.getpid()}"
            with open(tmp, "w") as fh:
                json.dump(d, fh, indent=1)
            os.replace(tmp, f)
    return add


def _load():
    p = f"{C['ROOT']}/GEN_IDS.json"
    try:
        mt = os.path.getmtime(p)
    except OSError:
        return S
    if S["mt"] != mt:
        try:
            with open(p) as f:
                d = json.load(f)
        except (OSError, ValueError):
            return S
        runs, bats = d.get("runs") or {}, d.get("batches") or {}
        num = lambda s: int(str(s)[1:]) if s else 0                    # noqa: E731
        comp = {"t": d.get("updated"), "b": {k: num(e.get("b")) for k, e in bats.items()},
                "rb": {k: num(e.get("b")) for k, e in runs.items() if e.get("b")}}
        S.update(mt=mt, body=json.dumps(comp, separators=(",", ":")).encode(), etag=f'"b{mt:.3f}"', runs=runs, batches=bats,
                 b={num(e.get("b")): k for k, e in bats.items()}, v={str(e.get("v")): k for k, e in runs.items() if e.get("v")})
    return S


async def genids(req):
    s = _load()
    if s["etag"] and req.headers.get("If-None-Match") == s["etag"]:
        return web.Response(status=304, headers={"ETag": s["etag"], "Cache-Control": "no-cache"})
    return web.Response(body=s["body"], content_type="application/json", headers={"ETag": s["etag"] or "", "Cache-Control": "no-cache"})


async def gid(req):
    m = re.match(r"^([bv]?)([0-9]{1,7})$", req.match_info["x"].strip(), re.I)
    if not m:
        return web.json_response({"error": "ids look like v670 or B37"}, status=404)
    s, kind, n = _load(), (m.group(1) or "v").lower(), int(m.group(2))
    if kind == "v":
        rid = s["v"].get(f"v{n}")
        if not rid:
            return web.json_response({"error": f"no v{n}"}, status=404)
        e, q = s["runs"].get(rid) or {}, C["jread"](f"{C['RUNS']}/{rid}/request.json", {}) or {}
        return web.json_response({"v": f"v{n}", "run_id": rid, "batch": e.get("batch"), "b": e.get("b"), "model": q.get("model") or "monoplex",
                                  "mono_weights": q.get("mono_weights") or "base", "ws": q.get("ws"), "hidden": bool(q.get("hidden"))},
                                 headers={"Cache-Control": "no-store"})
    bid = s["b"].get(n)
    if not bid:
        return web.json_response({"error": f"no B{n}"}, status=404)
    e = s["batches"].get(bid) or {}
    runs = sorted((r for r, x in s["runs"].items() if x.get("batch") == bid), key=lambda r: int(str(s["runs"][r].get("v") or "v0")[1:]))
    q = (C["jread"](f"{C['RUNS']}/{runs[0]}/request.json", {}) or {}) if runs else {}
    return web.json_response({"b": f"B{n}", "batch": bid, "run_id": runs[0] if runs else None, "runs": runs, "v": e.get("v"),
                              "model": q.get("model") or e.get("model") or "monoplex", "mono_weights": q.get("mono_weights") or "base",
                              "ws": q.get("ws")}, headers={"Cache-Control": "no-store"})


def setup(app, g):
    C.update({k: g[k] for k in ("ROOT", "RUNS", "jread") if k in g})
    app.router.add_get("/api/genids", genids)
    app.router.add_get("/api/gid/{x}", gid)
