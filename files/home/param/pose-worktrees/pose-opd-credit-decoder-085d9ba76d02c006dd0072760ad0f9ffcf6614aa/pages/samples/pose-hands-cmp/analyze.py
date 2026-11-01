"""pose-hands-cmp ANALYSIS: hand movement of GT vs 8B (SAM v2pure step 636, the video tab's model) vs 32B (PPM Aloe pose FT 22091 duo,
monoplex FT 22317, PPM main) on the same held-out Aloe tracks C-iii-sK (conversation iii, speaker s1 / s2), 15 s real prefill + 45 s generated.
Pose tokens -> keypoints with the one shared tokenizer (InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0, = PPM infer_ppm.POSE_TOK =
SAM pose_infer.render) -> body-relative coordinates: kinematic offsets summed from a zero root (units = the track's frame-0 shoulder width;
y multiplied by orig_h / orig_w so both axes are in shoulder widths), minus the current frame's mid-shoulder.
Writes <out>/metrics.json (per source x track metrics + aggregates) and <out>/tracks/<src>/<track>.npz (rel positions, conf, speed trace).
   python analyze.py <work root> <out dir>      (qwen-s2 venv, CPU or GPU)"""
import glob
import json
import os
import sys

import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, welch

sys.path.insert(0, "/data/home/amartyavarma/ref/pose-tokenizer")
from pose_tokenizer import PoseTokenizer  # noqa: E402
from pose_tokenizer.data.kinematic import offsets_to_positions  # noqa: E402

W, OUT = sys.argv[1], sys.argv[2]
os.makedirs(f"{OUT}/tracks", exist_ok=True)
POSE_TOK = ("InternalCan/pose-tokenizer-32cb-200k-noshift", "1d6ff4f0de89408cb054da618e309d4334596252")
S = "/data/home/amartyavarma/ppm/samples"
NPC, NTOT, FPS = 188, 750, 25                     # tokens (12.5 Hz); decoded poses are 25 fps (2 per token)
G0, G1 = 2 * NPC, 2 * NTOT                        # generated window in 25 fps frames
WR, EL = [9, 10], [7, 8]
FING = list(range(14, 34)) + list(range(35, 55))  # 40 finger joints (hand-root joints 13 / 34 excluded)
HANDS = [list(range(13, 34)), list(range(34, 55))]
TH_STILL, TH_GEST = 0.01, 0.04                    # shoulder widths per 25 fps frame (0.25 / 1.0 SW per s)
TH_SPEAK = -42.0                                  # dBFS, 200 ms smoothed frame RMS
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
ptok = PoseTokenizer.from_pretrained(POSE_TOK[0], device=dev, revision=POSE_TOK[1])

gt = {}
for i in range(16):
    for spk in ("s1", "s2"):
        tr = f"C-{i:03d}-{spk}"
        g = json.load(open(f"{W}/gt8b/{tr}.json"))
        z = np.load(f"{W}/gt8b/{tr}.npz")
        gt[tr] = dict(g, pose=z["pose"].astype(np.int64), audio=z["audio"])
rows_mono = json.load(open(f"{W}/rows-mono/rows.json"))
MONO = {r["row"]: f"C-{r['conv']:03d}-{r['duo_spk']}" for r in rows_mono}


def duo_dir(i):
    return f"{S}/aloe-pose-ft-430/runs-pose/B-{i:03d}" if i < 8 else f"{W}/runs-duo/B-{i:03d}"


def mono_dir(row):
    j = int(row.split("-")[1])
    return f"{S}/aloe-uncond-ft-430/runs/{row}" if j < 8 else f"{W}/runs-mono/{row}"


# ---------------------------------------------------------------- sources: key -> (label, group, {track: (pose [32,750], wav, words|None)})
SRC = {}


def add(key, label, group, track, pose, wav, words, check_gt=None):
    if check_gt is not None:
        assert np.array_equal(np.asarray(check_gt, np.int64), gt[track]["pose"]), ("GT pose differs", key, track)
    assert pose.shape == (32, NTOT), (key, track, pose.shape)
    SRC.setdefault(key, {"label": label, "group": group, "tracks": {}})["tracks"][track] = (pose.astype(np.int64), wav, words)


for i in range(16):
    for spk in ("s1", "s2"):
        tr = f"C-{i:03d}-{spk}"
        d = duo_dir(i)
        if os.path.isfile(f"{d}/DONE"):
            add("gt", "Ground truth", "gt", tr, gt[tr]["pose"], f"{d}/gt_{spk}.wav", gt[tr]["words"])
        for v, lab in (("nocfg", "8B no CFG"), ("cfg3", "8B pose CFG 3")):
            e = f"{W}/runs-8b/{v}/{tr}"
            if os.path.isfile(f"{e}/DONE"):
                z = np.load(f"{e}/codes.npz")
                assert np.array_equal(z["pose"][:, :NPC].astype(np.int64), gt[tr]["pose"][:, :NPC])
                add("8b" if v == "nocfg" else "8b_cfg3", lab, "8b", tr, z["pose"], f"{e}/gen.wav", None)
        for key, run, lab in (("duo", None, "32B pose FT (duo) no CFG"), ("duo_cfg3", "runs-posecfg3", "32B pose FT (duo) pose CFG 3"),
                              ("duo_k250", "runs-posek250", "32B pose FT (duo) cb1-31 top-k 250"),
                              ("duo_cfg3_k250", f"{S}/aloe-pose-ft-430-cfg3k250/runs-pose", "32B pose FT (duo, fast) pose CFG 3 + cb1-31 top-k 250")):
            e = d if run is None else (f"{run}/B-{i:03d}" if run.startswith("/") else f"{S}/aloe-pose-ft-430/{run}/B-{i:03d}")
            if os.path.isfile(f"{e}/DONE"):
                z = np.load(f"{e}/codes.npz")
                wj = json.load(open(f"{e}/words.json"))
                add(key, lab, "32b", tr, z[f"pose_{spk}"], f"{e}/{spk}.wav", wj[spk], check_gt=z[f"gt_pose_{spk}"])
MONO_K = [os.path.basename(x)[len("runs-k"):] for x in sorted(glob.glob(f"{S}/aloe-uncond-ft-430-cfg3k250/runs-k*"))]   # adjusted top-k variants
for row, tr in MONO.items():
    j = int(row.split("-")[1])
    extra = tuple((f"mono_cfg3_k{k}", f"{S}/aloe-uncond-ft-430-cfg3k250/runs-k{k}/{row}", f"32B monoplex FT (fast) pose CFG 3 + cb1-31 top-k {k}")
                  for k in MONO_K)
    for key, d, lab in (("mono", mono_dir(row), "32B monoplex FT no CFG"),
                        ("mono_cfg3", f"{S}/aloe-uncond-ft-430/runs-posecfg3/{row}", "32B monoplex FT pose CFG 3"),
                        ("mono_rep12", f"{S}/aloe-uncond-ft-430/runs-poserep12/{row}", "32B monoplex FT pose rep-pen 1.2"),
                        ("mono_adapt", f"{S}/aloe-uncond-ft-430/runs-poseadapt/{row}", "32B monoplex FT rep-pen 1.2 when silent"),
                        ("ppm", f"{W}/runs-ppmmain-mono/{row}", "PPM main step 12529 (mono, zero-shot)"),
                        ("mono_cfg3_k250", f"{S}/aloe-uncond-ft-430-cfg3k250/runs/{row}", "32B monoplex FT (fast) pose CFG 3 + cb1-31 top-k 250"),
                        ("mono_k140", f"{S}/aloe-uncond-ft-430-cfg3k250/runs-nocfg-k140/{row}", "32B monoplex FT (fast) no CFG, cb1-31 top-k 140")) + extra:
        if j >= 8 and key in ("mono_cfg3", "mono_rep12", "mono_adapt"):
            continue
        if os.path.isfile(f"{d}/DONE"):
            z = np.load(f"{d}/codes.npz")
            wj = json.load(open(f"{d}/words.json"))
            add(key, lab, "32b", tr, z["pose_s1"], f"{d}/s1.wav", wj["s1"], check_gt=z["gt_pose_s1"])
print({k: len(v["tracks"]) for k, v in SRC.items()}, flush=True)


# ---------------------------------------------------------------- decode
def decode(pc, norm):
    with torch.no_grad():
        rec = ptok.decode([torch.as_tensor(pc[k], device=dev) for k in range(32)]).float().cpu().numpy()
    r3 = rec.reshape(rec.shape[0], 55, 3)
    off = r3[:, :, :2].copy()
    rel0 = offsets_to_positions(np.zeros((len(off), 2), np.float32), off)            # frame-0-root relative, shoulder-width units
    iso = rel0 * np.array([float(norm["orig_h"]) / float(norm["orig_w"]), 1.0], np.float32)
    mid = (iso[:, 5] + iso[:, 6]) / 2
    rel = iso - mid[:, None]
    w0 = float(norm["frame0_width"])
    img = offsets_to_positions(np.repeat(np.array([norm["frame0_root"]], np.float32), len(off), 0), off * w0)   # image coords (render)
    return rel.astype(np.float32), np.clip(r3[:, :, 2], 0, 1).astype(np.float32), img.astype(np.float32)


def energy_db(wav):
    x, sr = sf.read(wav, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    hop = sr // FPS
    n = len(x) // hop
    e = 10 * np.log10(np.mean(x[:n * hop].reshape(n, hop).astype(np.float64) ** 2, 1) + 1e-10)
    out = np.full(G1, -100.0)
    out[:min(n, G1)] = e[:G1]
    return out


def code_stats(pc):
    g = pc[:, NPC:]
    c0 = g[0]
    _, cnt = np.unique(c0, return_counts=True)
    p = cnt / cnt.sum()
    runs = np.diff(np.flatnonzero(np.r_[True, c0[1:] != c0[:-1], True]))
    return {"cb0_distinct": int(len(cnt)), "cb0_entropy_bits": float(-(p * np.log2(p)).sum()), "cb0_repeat": float((c0[1:] == c0[:-1]).mean()),
            "cb0_mean_run": float(runs.mean()), "cb_change_frac": float((g[:, 1:] != g[:, :-1]).mean()),
            "tuple_distinct": int(len({tuple(col) for col in g.T}))}


def wmetrics(rel, conf, db, a, b):
    """metrics over 25 fps frames [a, b)."""
    v = np.linalg.norm(np.diff(rel, axis=0), axis=-1)          # [T-1, 55]: speed between frame t and t+1
    v = v[a:b - 1]
    ws = v[:, WR].mean(1)                                       # mean of the two wrists, per frame
    wmax = v[:, WR].max(1)
    pr = rel[a:b]
    out = {"wrist_speed": float(ws.mean()), "wrist_p50": float(np.median(ws)), "wrist_p90": float(np.percentile(ws, 90)),
           "wrist_p99": float(np.percentile(ws, 99)), "path_45s": float(v[:, WR].sum(0).mean() * (1124 / len(ws))),
           "wrist_pos_std": float(np.sqrt(pr[:, WR].var(0).sum(-1)).mean()), "wrist_ms_vel": float((v[:, WR] ** 2).mean()),
           "elbow_speed": float(v[:, EL].mean()), "finger_speed": float(v[:, FING].mean())}
    art = []
    for h, root in ((list(range(14, 34)), 13), (list(range(35, 55)), 34)):
        q = pr[:, h] - pr[:, root:root + 1]
        art.append(np.linalg.norm(np.diff(q, axis=0), axis=-1).mean())
    out["finger_artic_speed"] = float(np.mean(art))
    out["still_frac"] = float((wmax < TH_STILL).mean())
    s = gaussian_filter1d(wmax, 2)
    pk, _ = find_peaks(s, height=TH_GEST, distance=10, prominence=TH_GEST / 2)
    out["gestures_45s"] = float(len(pk) * 1124 / len(s))
    y = pr[:, WR, 0]
    out["hand_height_range"] = float((np.percentile(y, 95, 0) - np.percentile(y, 5, 0)).mean())
    out["hand_height_mean"] = float(y.mean())
    P = 0
    for j in WR:
        for ax in (0, 1):
            f, pw = welch(pr[:, j, ax], fs=FPS, nperseg=min(256, len(pr)), detrend="linear")
            P = P + pw / 2
    df = f[1] - f[0]
    band = lambda lo, hi: float(P[(f >= lo) & (f < hi)].sum() * df)  # noqa: E731
    out.update(band_05_3=band(0.5, 3.0), band_3_125=band(3.0, 12.6), band_01_125=band(0.1, 12.6))
    out["band_05_3_frac"] = out["band_05_3"] / max(out["band_01_125"], 1e-12)
    out["hf_frac"] = out["band_3_125"] / max(out["band_01_125"], 1e-12)
    hc = np.stack([conf[a:b, h].mean(1) for h in HANDS], 1)
    out["hand_visible_frac"] = float((hc > 0.3).mean())
    out["wrist_conf"] = float(conf[a:b, WR].mean())
    # speech link (own audio)
    e = db[a:b - 1]
    es = gaussian_filter1d(e, 5)
    sp = es > TH_SPEAK
    out["speak_frac"] = float(sp.mean())
    out["speed_speaking"] = float(ws[sp].mean()) if sp.sum() > 25 else None
    out["speed_silent"] = float(ws[~sp].mean()) if (~sp).sum() > 25 else None
    out["speak_silent_ratio"] = (out["speed_speaking"] / out["speed_silent"]) if out["speed_speaking"] and out["speed_silent"] else None
    vs = gaussian_filter1d(ws, 6)
    eq = gaussian_filter1d(np.maximum(e, -70), 6)
    out["r_speed_energy"] = float(np.corrcoef(vs, eq)[0, 1]) if vs.std() > 0 and eq.std() > 0 else None
    best = None
    for lag in range(-25, 26):                                 # hand leads speech for lag < 0
        x1, x2 = (vs[:lag], eq[-lag:]) if lag < 0 else ((vs[lag:], eq[:len(eq) - lag]) if lag > 0 else (vs, eq))
        if x1.std() > 0 and x2.std() > 0:
            r = float(np.corrcoef(x1, x2)[0, 1])
            if best is None or r > best[0]:
                best = (r, lag / FPS)
    out["r_max"], out["r_max_lag_s"] = best if best else (None, None)
    on = np.flatnonzero(sp[1:] & ~sp[:-1]) + 1
    on = [t for t in on if t >= 10 and not sp[max(0, t - 10):t].any() and t + 25 <= len(ws)]
    out["n_onsets"] = len(on)
    out["onset_speed_ratio"] = float(np.mean([ws[t:t + 25].mean() for t in on]) / ws.mean()) if on and ws.mean() > 0 else None
    return out


RES = {}
for key, sd in SRC.items():
    os.makedirs(f"{OUT}/tracks/{key}", exist_ok=True)
    RES[key] = {"label": sd["label"], "group": sd["group"], "tracks": {}}
    for tr, (pc, wav, words) in sorted(sd["tracks"].items()):
        rel, conf, img = decode(pc, gt[tr]["norm"])
        db = energy_db(wav)
        m = {"gen": wmetrics(rel, conf, db, G0, G1), "prefill": wmetrics(rel, conf, db, 0, G0), "codes": code_stats(pc)}
        v = np.linalg.norm(np.diff(rel, axis=0), axis=-1)[:, WR].mean(1)
        trace = gaussian_filter1d(np.r_[v, v[-1]], 3)[::2]                                 # 12.5 Hz, 750 points
        np.savez_compressed(f"{OUT}/tracks/{key}/{tr}.npz", rel=rel, conf=conf, img=img, trace=trace.astype(np.float32), db=db.astype(np.float32))
        RES[key]["tracks"][tr] = m
    print(key, len(RES[key]["tracks"]), flush=True)


# ---------------------------------------------------------------- aggregates (each source vs GT on ITS OWN track set)
def agg(key, part="gen"):
    trs = sorted(set(RES[key]["tracks"]) & set(RES["gt"]["tracks"]))
    keys = [k for k, x in RES[key]["tracks"][trs[0]][part].items() if isinstance(x, (int, float)) or x is None]
    out = {"n": len(trs), "tracks": trs, "mean": {}, "gt_mean": {}, "pct_gt": {}}
    for k in keys:
        a = [RES[key]["tracks"][t][part][k] for t in trs]
        b = [RES["gt"]["tracks"][t][part][k] for t in trs]
        ok = [i for i in range(len(trs)) if a[i] is not None and b[i] is not None]
        if not ok:
            continue
        ma, mb = float(np.mean([a[i] for i in ok])), float(np.mean([b[i] for i in ok]))
        out["mean"][k], out["gt_mean"][k] = ma, mb
        out["pct_gt"][k] = 100 * ma / mb if mb else None
    for k in RES[key]["tracks"][trs[0]]["codes"]:
        out["mean"]["code_" + k] = float(np.mean([RES[key]["tracks"][t]["codes"][k] for t in trs]))
        out["gt_mean"]["code_" + k] = float(np.mean([RES["gt"]["tracks"][t]["codes"][k] for t in trs]))
    rng = np.random.default_rng(0)
    a = np.array([RES[key]["tracks"][t][part]["wrist_speed"] for t in trs])
    b = np.array([RES["gt"]["tracks"][t][part]["wrist_speed"] for t in trs])
    bs = [a[ix].mean() / b[ix].mean() for ix in (rng.integers(0, len(trs), len(trs)) for _ in range(2000))]
    out["wrist_speed_pct_ci95"] = [float(100 * np.percentile(bs, 2.5)), float(100 * np.percentile(bs, 97.5))]
    return out


AGG = {k: {"gen": agg(k, "gen"), "prefill": agg(k, "prefill")} for k in RES}
json.dump({"res": RES, "agg": AGG, "consts": {"TH_STILL": TH_STILL, "TH_GEST": TH_GEST, "TH_SPEAK": TH_SPEAK, "NPC": NPC, "NTOT": NTOT}},
          open(f"{OUT}/metrics.json", "w"))
for k in AGG:
    g = AGG[k]["gen"]
    print(f"{k:11s} n={g['n']:2d} wrist {g['pct_gt']['wrist_speed']:6.1f}% GT (CI {g['wrist_speed_pct_ci95'][0]:.0f}-{g['wrist_speed_pct_ci95'][1]:.0f}) "
          f"still {g['mean']['still_frac']:.2f} (gt {g['gt_mean']['still_frac']:.2f}) gest {g['mean']['gestures_45s']:.1f} (gt {g['gt_mean']['gestures_45s']:.1f}) "
          f"cb0 distinct {g['mean']['code_cb0_distinct']:.0f} (gt {g['gt_mean']['code_cb0_distinct']:.0f}) prefill wrist {AGG[k]['prefill']['pct_gt']['wrist_speed']:.0f}%",
          flush=True)
print("done")
