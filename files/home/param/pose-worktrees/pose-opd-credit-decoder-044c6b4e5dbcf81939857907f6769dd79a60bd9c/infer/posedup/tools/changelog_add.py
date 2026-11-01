#!/usr/bin/env python3
"""10-26 (user via main): EVERY deploy to a demo site appends one changelog line. One CHANGELOG.json per site, served from disk by the
site's server; the page's header badge = the newest entry's version, click = what is running + the changelog (times shown in PT).

  changelog_add.py <site> <who> "<one plain line of what changed>" [--at <ISO UTC>] [--set key=value ...] [--page] [--server]
    <site>: live  (= aloe-posedup-live/CHANGELOG.json, the live pose-duplex demo)
            genweb (= aloe-posedup-live/genweb/CHANGELOG.json, the monoplex/duplex playground)
            motlive (= ppm/mot/live/CHANGELOG.json, the MoT 14B live demo)
            or a path to a CHANGELOG.json
    --set:  update the 'running' block (speech, video_engine, video_weights, video_sampler, ...)
    --page / --server: this deploy changed the page / the server: running.page / running.server = the new version

Version = v<PT date>-<n>, n = 1 + the number of entries already on that PT date. Atomic (flock + tmp + replace); prints the new version.

{"site": str, "running": {key: str}, "entries": [{"t": ISO UTC, "v": version, "what": str, "who": agent}, ...newest first]}"""
import argparse
import datetime as dt
import fcntl
import json
import os
from zoneinfo import ZoneInfo

H = "/data/home/amartyavarma/ppm/aloe-posedup-live"
SITES = {"live": f"{H}/CHANGELOG.json", "genweb": f"{H}/genweb/CHANGELOG.json",
         "motlive": "/data/home/amartyavarma/ppm/mot/live/CHANGELOG.json"}          # 10-26: the MoT 14B live demo (ppm/mot/live)
PT = ZoneInfo("America/Los_Angeles")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("site")
    ap.add_argument("who")
    ap.add_argument("what")
    ap.add_argument("--at", default=None, help="ISO UTC time of the deploy (default now)")
    ap.add_argument("--set", action="append", default=[], help="running key=value")
    ap.add_argument("--page", action="store_true")
    ap.add_argument("--server", action="store_true")
    a = ap.parse_args()
    path = SITES.get(a.site, a.site)
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            with open(path) as f:
                d = json.load(f)
        except FileNotFoundError:
            d = {"site": a.site, "running": {}, "entries": []}
        at = dt.datetime.fromisoformat(a.at.replace("Z", "+00:00")) if a.at else dt.datetime.now(dt.timezone.utc)
        day = at.astimezone(PT).strftime("%Y.%m.%d")
        n = 1 + sum(1 for e in d["entries"] if e["v"].startswith(f"v{day}-"))
        v = f"v{day}-{n}"
        d["entries"].append({"t": at.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "v": v, "what": a.what, "who": a.who})
        d["entries"].sort(key=lambda e: e["t"], reverse=True)
        for kv in a.set:
            k, _, val = kv.partition("=")
            d["running"][k] = val
        if a.page:
            d["running"]["page"] = v
        if a.server:
            d["running"]["server"] = v
        with open(path + ".tmp", "w") as f:
            json.dump(d, f, indent=1)
        os.replace(path + ".tmp", path)
    print(v)


if __name__ == "__main__":
    main()
