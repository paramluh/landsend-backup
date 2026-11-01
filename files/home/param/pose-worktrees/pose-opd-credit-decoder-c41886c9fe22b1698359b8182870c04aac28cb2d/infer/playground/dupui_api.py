"""dupui (10-26): duplex batch download with the conversation audio. Mounted by server.py: import dupui_api; dupui_api.setup(app, globals())
GET /api/dupui          -> {"ok": true} (the page shows the extra download link only when this route exists)
GET /api/dupzip/{bid}   -> server.batch_zip's files (each sample's exact avatar mp4 + the compare re-render, same names) plus every run's
                           conversation audio: mix.wav (A + B as B heard them), a.wav (A as fed to B), b.wav (B), a_own.wav (A on its own
                           timeline, both-sides runs only)."""
import io
import os
import asyncio
import zipfile

from aiohttp import web

C = {}
WAVS = ("mix.wav", "a.wav", "b.wav", "a_own.wav")


async def ping(req):
    return web.json_response({"ok": True}, headers={"Cache-Control": "no-store"})


async def dupzip(req):
    bid = req.match_info["bid"]
    cards = C["batch_cards"](bid)
    if not cards:
        return web.json_response({"error": "no such batch"}, status=404)
    path = await asyncio.to_thread(_zip_file, bid, cards)               # 10-26 aloedemo: in a thread, on disk (never in RAM / on the event loop)
    return web.FileResponse(path, headers={"Content-Disposition": f'attachment; filename="{bid}-conversation.zip"', "Cache-Control": "no-store"})


def _zip_file(bid, cards):
    os.makedirs(f"{C['ROUT']}/../zips", exist_ok=True)
    path = f"{C['ROUT']}/../zips/{bid}-conversation.zip"
    with zipfile.ZipFile(path + ".tmp", "w", zipfile.ZIP_STORED) as z:
        for i, c in enumerate(cards):
            stem = f"{bid}_{i + 1:02d}_seed{c['seed']}_{c['run_id']}"
            _, p = C["exact_src"](c["run_id"])
            if p:
                z.write(p, arcname=f"{stem}{'_vansh' if c.get('video_weights') == 'vansh' else ''}.mp4")
            o = (c.get("cmp") or {}).get("other")
            if o and os.path.exists(f"{C['ROUT']}/{c['run_id']}.{o}.mp4"):
                z.write(f"{C['ROUT']}/{c['run_id']}.{o}.mp4", arcname=f"{stem}_{o}.mp4")
            for n in WAVS:
                f = f"{C['RUNS']}/{c['run_id']}/{n}"
                if os.path.isfile(f):
                    z.write(f, arcname=f"{stem}_{n}")
    os.replace(path + ".tmp", path)
    return path


def setup(app, g):
    C.update({k: g[k] for k in ("RUNS", "ROUT", "batch_cards", "exact_src")})
    app.router.add_get("/api/dupui", ping)
    app.router.add_get("/api/dupzip/{bid}", dupzip)
