"""Serve cached pose coordinates for a synchronized video overlay."""
from pathlib import Path

from aiohttp import web


def setup(app, context):
    runs = Path(context["RUNS"])
    valid_id = context["RID"]
    read_json = context["jread"]

    async def pose(request):
        rid = request.match_info["rid"]
        if not valid_id.fullmatch(rid):
            return web.json_response({"state": "unavailable", "message": "Unknown generation."}, status=404)
        directory = runs / rid
        if not (directory / "request.json").is_file():
            return web.json_response({"state": "unavailable", "message": "Unknown generation."}, status=404)
        output = directory / "pose_overlay.json"
        tokens = directory / "tokens.npz"
        try:
            if output.is_file() and tokens.is_file() and output.stat().st_mtime_ns >= tokens.stat().st_mtime_ns:
                return web.FileResponse(output, headers={"Cache-Control": "private, max-age=0, must-revalidate"})
        except FileNotFoundError:
            pass
        status = read_json(str(directory / "pose_overlay.status.json"), {}) or {}
        if status.get("state") in ("error", "failed", "unavailable"):
            return web.json_response({"state": "unavailable", "message": "Pose overlay could not be prepared for this generation."},
                                     status=422, headers={"Cache-Control": "no-store"})
        generation = read_json(str(directory / "status.json"), {}) or {}
        if not tokens.is_file() and generation.get("state") in ("done", "error", "failed", "cancelled"):
            return web.json_response({"state": "unavailable", "message": "This generation has no saved pose data."},
                                     status=404, headers={"Cache-Control": "no-store"})
        return web.json_response({"state": "preparing" if tokens.is_file() else "waiting",
                                  "message": "Preparing pose overlay…" if tokens.is_file() else "Pose becomes available after generation."},
                                 status=202, headers={"Retry-After": "2", "Cache-Control": "no-store"})

    app.router.add_get("/api/run/{rid}/pose", pose)
