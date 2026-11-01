"""NODE-LOCAL DECODED A-AUDIO ROW CACHE (SPEED FIX 10-16, see the run dir's SPEED_FIX.md). CPU only, no torch.

Why: the Voxtral trainers' DataLoader workers ran audio_s4q.ffmpeg_row_audio for every micro, i.e. ffmpeg on the WHOLE ~23 min 48 kHz stereo
WLT segment FLAC (~1.9 CPU-s + a ~134 MB pipe read per 10 min row) on the training rank's own 5 logical CPUs (2.5 physical cores) while the
FSDP forward ran -> forward 4-5 s instead of 1.5-2.0 s (nodediag A/B 10-16, jobs 22341 / 22357; backward / nanosteps unchanged).

What: tools/stage_rowcache.py runs the SAME function (audio_s4q.ffmpeg_row_audio: same ffmpeg binary + command, same slice / zero-pad /
gain) once per row at staging and stores its float32 output as .npy under ALOE_ROWCACHE (node-local). row_audio() returns np.load() of that
file when present, else falls back to the on-the-fly decode -> the returned array is bit-identical either way (dtype / shape asserted here;
equality of the two paths verified by tools/stage_rowcache.py --verify N and the 10-16 slinky-4 CPU check).

Key = sha256 of (absolute FLAC path, its size in bytes, frame0, n_frames, repr(gain_db), ffmpeg binary path): any change of the source
file, the window, the gain or the decoder binary misses the cache (= on-the-fly decode), never a stale hit.
ALOE_ROWCACHE unset / empty = cache off (pure on-the-fly, the pre-fix behaviour)."""
import hashlib
import json
import os

import numpy as np

ROOT = os.environ.get("ALOE_ROWCACHE", "")
STATS = {"hit": 0, "miss": 0}


def _au():
    import audio_s4q as AU
    return AU


def key(e, gain_db):
    AU = _au()
    src = os.path.abspath(e["flac"])
    k = json.dumps([src, int(os.path.getsize(src)), int(e["frame0"]), int(e["n_frames"]), repr(float(gain_db)), str(AU.FFMPEG_EXE)])
    return hashlib.sha256(k.encode()).hexdigest()


def path_for(e, gain_db, root=None):
    h = key(e, gain_db)
    return f"{root or ROOT}/{h[:2]}/{h}.npy"


def _check(w, e):
    AU = _au()
    n = int(e["n_frames"]) * AU.SAMPLES_PER_FRAME_24K
    assert w.dtype == np.float32 and w.shape == (n,), ("row cache entry dtype / shape", e.get("row_key"), w.dtype, w.shape, n)
    return w


def row_audio(e, gain_db):
    """= audio_s4q.ffmpeg_row_audio(e, gain_db), from the node-local cache when staged (bit-identical)."""
    if ROOT:
        p = path_for(e, gain_db)
        if os.path.exists(p):
            STATS["hit"] += 1
            return _check(np.load(p, allow_pickle=False), e)
    STATS["miss"] += 1
    return _au().ffmpeg_row_audio(e, gain_db)


def build(e, gain_db, root=None):
    """decode with the on-the-fly function and store it atomically; -> 'hit' (already cached) or 'built'."""
    p = path_for(e, gain_db, root)
    if os.path.exists(p):
        return "hit"
    w = _check(_au().ffmpeg_row_audio(e, gain_db), e)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = f"{p}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        np.save(f, w, allow_pickle=False)
    os.replace(tmp, p)
    return "built"


def verify(e, gain_db, root=None):
    """cached array == a fresh on-the-fly decode, bit for bit (dtype, shape, bytes)."""
    p = path_for(e, gain_db, root)
    a = np.load(p, allow_pickle=False)
    b = _au().ffmpeg_row_audio(e, gain_db)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
