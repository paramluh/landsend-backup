#!/usr/bin/env python3
"""posefix page: avatar mp4s WITH B's voice, same names under a/ (main's plain page). For each a/<kind>-<name>-<n>.mp4 listed in the
plain page: video = the render clips/<kind>/<name>.mp4 (frame 0 = feed frame 0 = B Mimi frame 0), audio = the run's own B waveform
(sim npz 'b' for mono / dupM, the conversation's real B stem for gtc), both cut to n code frames (n x 80 ms); H.264 350 kbps + AAC 64 kbps.
10-25 (user): a WHITE FLASH (3 video frames = 0.12 s) burnt in and an 880 Hz BEEP (150 ms) mixed into the audio at the prefill -> generated
switch (switch_s = prefill code frames x 80 ms; none when there is no prefill), so the switch shows in any player.
usage: python mux.py <list of a/*.mp4 names> [workers]      (library: mux_one(src, audio, n_frames, switch_s, out))"""
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf

P = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix"
FF = "/data/home/amartyavarma/tools/ffmpeg-static/ffmpeg-7.0.2-amd64-static/ffmpeg"
CONVS = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/convs"


def mux_one(src, a, n, switch_s, out):
    """src video (frame 0 = code frame 0), a float32 24 kHz B audio from code frame 0, n code frames, switch_s (None = no flash) -> out."""
    dur = n * 0.08
    a = np.asarray(a, np.float32)[:int(round(dur * 24000))].copy()
    vf = []
    if switch_s is not None and 0 < switch_s < dur:
        i0 = int(round(switch_s * 24000))
        t = np.arange(int(0.15 * 24000)) / 24000.0
        beep = 0.35 * np.sin(2 * np.pi * 880 * t) * np.minimum(1, np.minimum(t / 0.01, (0.15 - t) / 0.01))
        if len(a) < i0 + len(beep):
            a = np.concatenate([a, np.zeros(i0 + len(beep) - len(a), np.float32)])
        a[i0:i0 + len(beep)] = np.clip(a[i0:i0 + len(beep)] + beep, -1, 1)
        vf = ["-vf", f"drawbox=x=0:y=0:w=iw:h=ih:color=white@1:t=fill:enable='between(t,{switch_s:.3f},{switch_s + 0.119:.3f})'"]
    with tempfile.NamedTemporaryFile(suffix=".wav", dir="/tmp", delete=False) as t_:
        wav = t_.name
    sf.write(wav, a, 24000)
    tmp = out + ".tmp.mp4"
    cmd = [FF, "-y", "-loglevel", "error", "-i", src, "-i", wav, "-t", f"{dur:.3f}", "-map", "0:v:0", "-map", "1:a:0"] + vf + [
        "-c:v", "libx264", "-preset", "veryfast", "-b:v", "350k", "-maxrate", "450k", "-bufsize", "900k", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-b:a", "64k", "-ar", "24000", "-movflags", "+faststart", tmp]
    r = subprocess.run(cmd, capture_output=True, text=True)
    os.unlink(wav)
    if r.returncode:
        return "ffmpeg " + r.stderr[-200:]
    os.replace(tmp, out)
    return f"ok {os.path.getsize(out) / 1e6:.2f} MB"


def job(fn):
    base = os.path.basename(fn)[:-4]
    kind, rest = base.split("-", 1)
    run, n = rest.rsplit("-", 1)
    n = int(n)
    src = f"{P}/clips/{kind}/{run}.mp4"
    if not os.path.isfile(src):
        return fn, "no render"
    sw = None
    if kind == "gtc":
        a = np.load(f"{CONVS}/{run[:2]}.npz")[f"stem_{run[3]}"].astype(np.float32)
    else:
        a = np.load(f"{P}/out/{kind}/{run}.npz")["b"].astype(np.float32)
        j = json.load(open(f"{P}/out/{kind}/{run}.json"))
        sw = (j.get("pose_prefill_frames") or 0) * 0.08 or None       # the prefill -> generated switch
    return fn, mux_one(src, a, n, sw, f"{P}/page/{fn}")


if __name__ == "__main__":
    names = [ln.strip() for ln in open(sys.argv[1]) if ln.strip()]
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    with ThreadPoolExecutor(W) as ex:
        res = list(ex.map(job, names))
    bad = [r for r in res if not r[1].startswith("ok")]
    print(f"{len(res) - len(bad)} ok, {len(bad)} not: {bad[:10]}")
    print("total MB", round(sum(os.path.getsize(f"{P}/page/{fn}") for fn, s in res if s.startswith("ok")) / 1e6, 1))
