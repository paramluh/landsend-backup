"""Serve the Rosie listening gallery and its public audio assets."""
from pathlib import Path
from aiohttp import web

ROOT = Path(__file__).resolve().parent
PUBLIC = ROOT / "public"


async def index(request):
    return web.FileResponse(PUBLIC / "index.html", headers={"Cache-Control": "no-cache"})


app = web.Application()
app.router.add_get("/", index)
app.router.add_static("/", PUBLIC, show_index=False)
web.run_app(app, host="127.0.0.1", port=8920, access_log=None)
