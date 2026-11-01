"""pose-duplex samples: pose tokens -> per-speaker stick-figure skeleton MP4s for the 2-side samples page (CPU only, no CFG, no video model).
Reused verbatim from ppm/samples/aloe-pose-ft-430/work/infer_pose_2side.py render() + ppm/code/infer_ppm.py decode_pose / POSE_EDGES /
POSE_COL (copied here, originals untouched): PoseTokenizer InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f (HF cache, offline) decode
-> shoulder-width-normalised offsets -> denormalised with the cut's frame-0 root + shoulder width (frame0-root-frame0-horizontal-shoulder-
width-v1) -> 480x270 canvas, 25 fps (2 pose frames / 12.5 Hz token, frame g*2 at time g*0.08 s), libx264 yuv420p +faststart, no audio.
Duration = 2T frames / 25 fps = T*0.08 s (syncs with the 24 kHz wavs: token g = sample g*1920). Labels: PROMPT (real) / GENERATED / REAL.

per run dir (runs/B-00N/): pose_s1.npy pose_s2.npy gt_pose_s1.npy gt_pose_s2.npy (int16 [32, T], cb-local 0..2047, -1 = none)
  -> pose_gen_s1.mp4 pose_gen_s2.mp4 pose_gt_s1.mp4 pose_gt_s2.mp4 + pose_render.json + POSE_DONE
-1 columns are filled with the nearest valid column (all -1 -> blank 'no pose' frames).
frame-0 normalisation per speaker, first hit wins: meta.json pose_norm/norm[spk] {frame0_root [y, x], frame0_width}; else the source cut's
w00.SPEAKER_0{0,1}.pose.json (s1 = SPEAKER_00, s2 = SPEAKER_01, or meta pose_speakers {s1: .., s2: ..}) found from meta row_key / prov.row_key
'.../<session>/<seg>/cNN' under data/aloe-handover-full/DATA/files/*/; else a canonical default (median Aloe framing).
prefill end (label switch) = meta prompt_s (or prefill_s); else the first column where pose_sX != gt_pose_sX.

   python -I render_pose.py <run dir> [<run dir> ...]     (on a compute node; ~env: HF_HUB_OFFLINE=1)"""
import glob
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, "/data/home/amartyavarma/ref/pose-tokenizer")
POSE_TOK = ("InternalCan/pose-tokenizer-32cb-200k-noshift", "1d6ff4f0de89408cb054da618e309d4334596252")
DATA = "/data/home/amartyavarma/data/aloe-handover-full/DATA/files"
DEPTH = 32
H, W = 480, 270
FPS = 25
DEFAULT_NORM = {"frame0_root": [0.63, 0.50], "frame0_width": 0.85, "src": "default"}   # typical Aloe portrait framing (fallback only)

POSE_EDGES = {
    "face": [(0, 1), (0, 2), (1, 3), (2, 4)],
    "body": [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)],
    "lh": [(i + 13, j + 13) for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
                                         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]] + [(9, 13)],
    "rh": [(i + 34, j + 34) for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
                                         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]] + [(10, 34)],
}
POSE_COL = {"face": (200, 200, 200), "body": (80, 170, 255), "lh": (90, 220, 120), "rh": (255, 140, 80)}     # BGR


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def fill(pc):
    """[32, T] with -1 columns -> nearest valid column; returns (codes int64, n_filled) or (None, T) if nothing valid."""
    pc = np.asarray(pc).astype(np.int64)
    ok = (pc >= 0).all(0) & (pc < 2048).all(0)
    if not ok.any():
        return None, pc.shape[1]
    idx = np.where(ok)[0]
    T = pc.shape[1]
    near = idx[np.clip(np.searchsorted(idx, np.arange(T)), 0, len(idx) - 1)]
    prev = idx[np.clip(np.searchsorted(idx, np.arange(T)) - 1, 0, len(idx) - 1)]
    pick = np.where(np.abs(prev - np.arange(T)) < np.abs(near - np.arange(T)), prev, near)
    return pc[:, pick], int((~ok).sum())


def pose_json_norm(pj):
    import ast
    d = json.load(open(pj))
    nm = d["normalization"]
    nm = nm if isinstance(nm, dict) else ast.literal_eval(nm)
    return {"frame0_root": nm["root0_yx_normalized"], "frame0_width": nm["shoulder_width0_normalized"], "orig_w": nm.get("orig_w"),
            "orig_h": nm.get("orig_h"), "src": pj}


def find_norm(M, spk):
    for k in ("pose_norm", "norm"):
        x = (M.get(k) or {}).get(spk)
        if x and "frame0_root" in x and "frame0_width" in x:
            return dict(x) | {"src": f"meta.{k}"}
    rk = M.get("row_key") or (M.get("prov") or {}).get("row_key") or ""
    parts = rk.strip("/").split("/")
    if len(parts) >= 3:
        sess, seg = (parts[-3], parts[-2]) if parts[-1].startswith("c") and parts[-2].startswith("seg") else (parts[-2], parts[-1])
        S = (M.get("pose_speakers") or {}).get(spk) or {"s1": "SPEAKER_00", "s2": "SPEAKER_01"}[spk]
        hits = sorted(glob.glob(f"{DATA}/*/{sess}/{seg}/w00.{S}.pose.json"))
        if hits:
            try:
                return pose_json_norm(hits[0])
            except Exception as e:  # noqa: BLE001
                log(f"pose.json unreadable {hits[0]}: {e}")
    return dict(DEFAULT_NORM)


def decode(ptok, codes_list):
    """list of [32, T] int64 (same T) -> list of (pos [2T, 55, 2] (y, x) offsets-not-yet-placed, conf [2T, 55]) raw r3 arrays."""
    import torch
    from pose_tokenizer.data.kinematic import NUM_JOINTS
    with torch.no_grad():
        cs = [torch.as_tensor(np.stack([c[k] for c in codes_list])) for k in range(DEPTH)]
        rec = ptok.decode(cs).float().cpu().numpy()                      # [B, 2T, 165]
    return [r.reshape(r.shape[0], NUM_JOINTS, 3) for r in rec]


def place(r3, c0):
    """= infer_ppm.decode_pose after the tokenizer call: the clip's frame-0 root + shoulder width."""
    from pose_tokenizer.data.kinematic import offsets_to_positions
    T = r3.shape[0]
    root = np.asarray(c0["frame0_root"], np.float32).reshape(1, 2)
    wd = np.float32(c0["frame0_width"])
    gs = np.repeat(np.concatenate([root, [[wd]]], 1), T, 0)
    pos = offsets_to_positions(gs[:, :2], r3[:, :, :2] * gs[:, 2:3, None])
    return np.asarray(pos), np.clip(r3[:, :, 2], 0, 1)


def draw(p, c, tag):
    import cv2
    img = np.full((H, W, 3), 24, np.uint8)
    if p is not None:
        for g_, edges in POSE_EDGES.items():
            col = POSE_COL[g_]
            for i, j in edges:
                al = 0.15 + 0.85 * min(c[i], c[j])
                if al < 0.2:
                    continue
                cl = tuple(int(x * al) for x in col)
                cv2.line(img, (int(p[i, 1] * W), int(p[i, 0] * H)), (int(p[j, 1] * W), int(p[j, 0] * H)), cl, 3 if g_ == "body" else 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "no pose", (95, H // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1, cv2.LINE_AA)
    if tag:
        cv2.putText(img, tag, (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 255) if tag == "GENERATED" else (180, 180, 180), 1, cv2.LINE_AA)
    return img


def write_mp4(job):
    """job = (out path, pos or None, conf or None, n_frames, prefill frames or None (None = 'REAL' throughout))."""
    import imageio_ffmpeg
    out, pos, conf, n, pf = job
    tmp = out[:-4] + ".tmp.mp4"
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-crf", "26", "-preset", "veryfast", "-g", "25", "-pix_fmt", "yuv420p",
           "-threads", "2", "-movflags", "+faststart", tmp]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(n):
        tag = "REAL" if pf is None else ("PROMPT (real)" if f < pf else "GENERATED")
        p.stdin.write(draw(None if pos is None else pos[f], None if conf is None else conf[f], tag).tobytes())
    p.stdin.close()
    assert p.wait() == 0, f"ffmpeg failed for {out}"
    os.replace(tmp, out)
    return out


def render_run(rd, ptok, ex):
    t0 = time.time()
    M = json.load(open(f"{rd}/meta.json")) if os.path.isfile(f"{rd}/meta.json") else {}
    raw = {f"{k}_{s}": np.load(f"{rd}/{'gt_' if k == 'gt' else ''}pose_{s}.npy") for k in ("gen", "gt") for s in ("s1", "s2")}
    T = raw["gen_s1"].shape[1]
    for k, v in raw.items():
        assert v.ndim == 2 and v.shape[0] == DEPTH, (k, v.shape)
        assert v.shape[1] == T, ("pose arrays differ in T", k, v.shape, T)
    ps = M.get("prompt_s", M.get("prefill_s"))
    filled, nfill, pf = {}, {}, {}
    for k, v in raw.items():
        filled[k], nfill[k] = fill(v)
    for s in ("s1", "s2"):
        if ps is not None:
            pf[s] = int(round(float(ps) * 12.5)) * 2
        else:
            a, b = raw[f"gen_{s}"], raw[f"gt_{s}"]
            d = np.where((a != b).any(0) & (a >= 0).all(0))[0]
            pf[s] = int(d[0]) * 2 if len(d) else 2 * T
    keys = [k for k in raw if filled[k] is not None]
    dec = dict(zip(keys, decode(ptok, [filled[k] for k in keys]))) if keys else {}
    t_dec = time.time() - t0
    norms = {s: find_norm(M, s) for s in ("s1", "s2")}
    jobs = []
    for k in raw:
        kind, s = k.split("_")
        n = 2 * T
        if k in dec:
            pos, conf = place(dec[k], norms[s])
            assert pos.shape[0] == n, ("decoder frames != 2T", k, pos.shape, n)
        else:
            pos = conf = None
        jobs.append((f"{rd}/pose_{kind}_{s}.mp4", pos, conf, n, pf[s] if kind == "gen" else None))
    outs = list(ex.map(write_mp4, jobs))
    info = {"T_tokens": T, "frames": 2 * T, "fps": FPS, "duration_s": round(T * 0.08, 3), "prefill_frames": pf, "prompt_s": ps,
            "filled_columns": nfill, "norm": norms, "tokenizer": f"{POSE_TOK[0]}@{POSE_TOK[1]}", "size": f"{W}x{H}",
            "timing_s": {"decode": round(t_dec, 1), "total": round(time.time() - t0, 1)}, "node": os.uname().nodename,
            "files": [os.path.basename(o) for o in outs], "done": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    json.dump(info, open(f"{rd}/pose_render.json", "w"), indent=1)
    open(f"{rd}/POSE_DONE", "w").write("ok\n")
    log(f"[render_pose] {rd}: T {T} -> {2 * T} frames ({T * 0.08:.2f} s) x4 in {time.time() - t0:.1f} s (decode {t_dec:.1f} s); "
        f"filled {nfill}; norm {[norms[s]['src'] for s in ('s1', 's2')]}")


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HOME", "/data/home/amartyavarma/.cache/huggingface")
    import torch
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 8)))
    from pose_tokenizer import PoseTokenizer
    t0 = time.time()
    ptok = PoseTokenizer.from_pretrained(POSE_TOK[0], device="cpu", revision=POSE_TOK[1])
    log(f"[render_pose] tokenizer loaded in {time.time() - t0:.1f} s")
    rc = 0
    import multiprocessing as mp
    with ProcessPoolExecutor(4, mp_context=mp.get_context("spawn")) as ex:
        for rd in sys.argv[1:]:
            rd = rd.rstrip("/")
            try:
                render_run(rd, ptok, ex)
            except Exception as e:  # noqa: BLE001
                rc = 1
                log(f"[render_pose] FAILED {rd}: {type(e).__name__}: {e}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
