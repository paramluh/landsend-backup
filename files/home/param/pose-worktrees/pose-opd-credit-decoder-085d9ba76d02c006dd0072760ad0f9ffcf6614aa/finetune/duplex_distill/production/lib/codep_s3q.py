"""pose-s3 CODEPENDENCE evals (10-09; used only by train_pose_s3q.codep.py -> after the swap, train_pose_s3q.py). Forward passes only.

Per rank: a FIXED subset of held-out pose clips (the rank's pose eval rows, whole clips in row order until CODEP_FRAMES positions), each clip
its own attention segment, scored in FIVE PASSES, ONE FORWARD PER VARIANT with identical length / layout / scored positions / light frames
(only the inputs below differ), so a real and a counterfactual pass are bitwise comparable (a packed multi-variant batch is not: GEMM
rounding depends on the row position):
  real         the clip as trained
  pose_shuf    the trunk's pose input (codes + frame-0 state) from ANOTHER subset clip (seeded derangement; its frames tiled / cropped to this
               clip's length, then the same cd3 delay)
  pose_zero    no pose input (codes absent, frame-0 state zero)
  speech_shuf  the trunk's speech input (Mimi codes + word lane; also the pose tower's lane input) from the partner clip, tiled / cropped
  speech_sil   silence: word lane PAD, Mimi codes absent
Every variant is scored on the clip's REAL targets (text, speech cb0 / light, pose cb0 / light; same light frames). Towers' teacher-forced
code inputs + labels stay real (only the trunk / lane inputs change), so a difference is the effect of the counterfactual INPUT.
  pose -> speech   dCE = CE(variant) - CE(real) for pose_shuf / pose_zero: normalised word CE (nats / word), speech cb0 (nats / frame), speech
                   light (nats / frame, mean over cb1..31)
  speech -> pose   dCE for speech_shuf / speech_sil: pose cb0, pose light (same units)
  timing           Pearson r over frames of P(NEW_WORD at frame f) (head probe targets = NEW_WORD at every scored position) vs pose MOTION
                   magnitude at f = || z_f - z_(f-1) ||, z = the pose tokenizer's RVQ latent of frame f (sum over the 32 codebooks of
                   out_proj(normalise(codebook[code])), the decoder's input; frame-0-relative because the codes are), for the real input and for
                   pose_shuf (r against the real clip's motion and against the shown partner motion)."""
import json
import os
import struct

import numpy as np

import pose_s3q as PS

SP = PS.SP
VARIANTS = ("real", "pose_shuf", "pose_zero", "speech_shuf", "speech_sil")
SPEECH_V = (0, 1, 2)            # passes whose speech heads / towers are scored (pose -> speech; real; NEW_WORD probes on 0, 1)
POSE_V = (0, 3, 4)              # passes whose pose towers are scored (speech -> pose; real)
TOK = "/data/home/amartyavarma/.cache/huggingface/hub/models--InternalCan--pose-tokenizer-32cb-200k-noshift/snapshots/1d6ff4f0de89408cb054da618e309d4334596252"
_RVQ = {}


def rvq_tables(root=TOK):
    """[32, 2048, 2048] would be large: return per-codebook (codebook [2048, 16] normalised, W [2048, 16], b [2048]) of the tokenizer's quantizer
    (weight-norm out_proj: W = g * v / ||v||, norm over the input dims per output channel)."""
    if "t" in _RVQ:
        return _RVQ["t"]
    f = open(f"{root}/model.safetensors", "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    base = 8 + n

    def get(k):
        h = hdr[k]
        assert h["dtype"] == "F32", (k, h["dtype"])
        a, b = h["data_offsets"]
        f.seek(base + a)
        return np.frombuffer(f.read(b - a), np.float32).reshape(h["shape"])
    cfg = json.load(open(f"{root}/config.json"))
    out = []
    for c in range(int(cfg["n_codebooks"])):
        q = f"quantizer.quantizers.{c}."
        cb = get(q + "codebook.weight")
        if cfg.get("normalize_codes", True):
            cb = cb / np.maximum(np.linalg.norm(cb, axis=1, keepdims=True), 1e-12)
        g, v = get(q + "out_proj.parametrizations.weight.original0")[:, 0, 0], get(q + "out_proj.parametrizations.weight.original1")[:, :, 0]
        W = g[:, None] * v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        out.append((cb.astype(np.float32), W.astype(np.float32), get(q + "out_proj.bias").astype(np.float32)))
    _RVQ["t"] = out
    return out


def pose_latent(pose):
    """raw pose codes [32, n] (0..2047) -> RVQ latent z [n, 2048] (float32)."""
    t = rvq_tables()
    z = None
    for c, (cb, W, b) in enumerate(t):
        e = cb[np.asarray(pose[c], np.int64)] @ W.T + b
        z = e if z is None else z + e
    return z


def select_clips(rows, budget):
    """whole clips of the rank's eval rows, in row order, total packed span (SLOT + n + GAP each) <= budget; clips longer than budget / 2 are
    skipped (so the budget is met with >= 2 clips); falls back to the 2 shortest clips if fewer than 2 fit."""
    allc = [(r, c) for r, row in enumerate(rows) for c in row["clips"]]
    span = lambda c: PS.SLOT + int(c["n"]) + PS.GAP  # noqa: E731
    out, tot = [], 0
    for r, c in allc:
        L = span(c)
        if L <= budget // 2 and tot + L <= budget:
            out.append((r, c)); tot += L
    if len(out) < 2:
        out = sorted(allc, key=lambda x: span(x[1]))[:2]
        out = [x for x in allc if any(x is y for y in out)]
    assert len(out) >= 2, "codependence subset needs >= 2 clips"
    return out


def derangement(k, seed):
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), 0xC0DE]))
    while True:
        p = rng.permutation(k)
        if k < 2 or not (p == np.arange(k)).any():
            return p


def tile(a, n):
    """[.., m] -> [.., n] repeating along the last axis (crop when m >= n)."""
    m = a.shape[-1]
    return a[..., np.arange(n) % m]


def clip_row(row, c, codes=None, pose=None, lane=None, key=None):
    """a single-clip synthetic row ([SLOT | clip | GAP]) from the clip's real arrays, with optional replaced clip-frame arrays."""
    s0, n = int(c["slot"]), int(c["n"])
    L = PS.SLOT + n + PS.GAP
    sl = slice(s0, s0 + L)
    cod, pos, ln, kd = (np.array(np.asarray(row[k])[..., sl]) for k in ("codes", "pose", "lane", "kind"))
    if codes is not None:
        cod[:, PS.SLOT:PS.SLOT + n] = codes
    if pose is not None:
        pos[:, PS.SLOT:PS.SLOT + n] = pose
    if lane is not None:
        ln[:L] = lane
    cc = dict(c, slot=0, start=PS.SLOT)
    return {"codes": cod, "pose": pos, "lane": ln, "kind": kd, "row": -1, "n_frames": L, "row_key": key or f"codep|{row['row_key']}|{c.get('clip_id', s0)}",
            "clips": [cc], "split": row["split"], "dataset": row["dataset"], "n_words": 0, "n_pieces": 0}


def build(rows, cfg, rank, budget, seed, light_frac, plight_frac):
    """-> {"base": the REAL micro (numpy; pose_inputs layout) with NEW_WORD probes appended to its scored positions, "var": {variant: (ids_B,
    Mimi depth_B, pdepth, gfeat)} = the trunk inputs of each variant, "info", probe motion arrays}. ONE forward per variant: identical length,
    segments, scored positions, labels and light frames in every pass (only these inputs differ), so passes are bitwise comparable."""
    sel = select_clips(rows, budget)
    K = len(sel)
    perm = derangement(K, seed)
    real = [clip_row(rows[r], c) for r, c in sel]
    b = PS.pose_micro(real, cfg, rank, light_frac, plight_frac)
    N = int(sum(b["segs"]))
    starts = np.cumsum([0] + b["segs"][:-1])
    pd = int(cfg["pose_delay"])
    var = {}
    for v in VARIANTS:
        ids_b, dep, pdep, gf = b["ids"][1].copy(), b["depth"][1].copy(), b["pdepth"].copy(), b["gfeat"].copy()
        if v != "real":
            for i, (r, c) in enumerate(sel):
                n = int(c["n"]); s0 = int(starts[i])
                pr, pc = sel[perm[i]]
                p0 = int(pc["start"]); pn = int(pc["n"])
                src = rows[pr]
                if v in ("pose_shuf", "pose_zero"):
                    pose = tile(np.asarray(src["pose"])[:, p0:p0 + pn], n) if v == "pose_shuf" else np.full((32, n), -1, np.int16)
                    cr = clip_row(rows[r], c, pose=pose)
                    _, _, p_ = PS.segment_arrays(cr, 0, cr["n_frames"], pd, cfg)
                    pdep[s0:s0 + len(p_)] = p_
                    gf[i] = b["gfeat"][perm[i]] if v == "pose_shuf" else 0.0
                else:
                    lane = np.asarray(rows[r]["lane"])[int(c["slot"]):int(c["slot"]) + PS.SLOT + n + PS.GAP].copy()
                    if v == "speech_shuf":
                        lane[PS.SLOT:PS.SLOT + n] = tile(np.asarray(src["lane"])[p0:p0 + pn], n)
                        codes = tile(np.asarray(src["codes"])[:, p0:p0 + pn], n)
                    else:
                        lane[:] = SP.PAD_ID
                        codes = np.full((32, n), -1, np.int16)
                    cr = clip_row(rows[r], c, codes=codes, lane=lane)
                    a_, d_, _ = PS.segment_arrays(cr, 0, cr["n_frames"], pd, cfg)
                    ids_b[s0:s0 + len(a_)] = a_; dep[s0:s0 + len(d_)] = d_
        var[v] = (ids_b, dep, pdep, gf.astype(np.float32))
    # NEW_WORD probes (P(NW) at every scored position) appended to the scored positions of EVERY pass (identical layout)
    ns = len(b["sel"])
    base = dict(b)
    base["sel"] = np.concatenate([b["sel"], b["sel"]]); base["y"] = np.concatenate([b["y"], np.full(ns, SP.NW_ID, np.int64)])
    base["ch"] = np.ones(2 * ns, np.int64); base["rid"] = np.concatenate([b["rid"], b["rid"]])
    for k in ("word", "pad", "eob", "nw"):
        base[k] = np.concatenate([b[k], np.full(ns, k == "nw")])
    base["depth_real"], base["pdepth_real"] = b["depth"][1].copy(), b["pdepth"].copy()
    # timing: per probe -> the lane frame it predicts (q + 1 in its clip segment) and that frame's motion (the real clip / the shown partner)
    seg_of = np.repeat(np.arange(K), b["segs"]); q_in = np.concatenate([np.arange(L) for L in b["segs"]])
    tgt = b["sel"] + 1
    si, qj = seg_of[tgt], q_in[tgt]
    mot_real, mot_shown, ok = np.zeros(ns, np.float32), np.zeros(ns, np.float32), np.zeros(ns, bool)
    zs = []
    for i, (r, c) in enumerate(sel):
        n = int(c["n"]); p0 = int(c["start"])
        z = pose_latent(np.asarray(rows[r]["pose"])[:, p0:p0 + n])
        zs.append(np.concatenate([[np.nan], np.linalg.norm(np.diff(z, axis=0), axis=1)]).astype(np.float32))   # motion of clip frame k (k >= 1)
    for i in range(K):
        zp = zs[perm[i]]
        m = si == i
        f = qj[m] - PS.SLOT
        n = int(sel[i][1]["n"])
        good = (f >= 1) & (f < n)
        mr = np.where(good, zs[i][np.clip(f, 0, n - 1)], 0.0)
        ms = np.where(good, zp[np.clip(f, 0, n - 1) % len(zp)], 0.0)
        good &= np.isfinite(mr) & np.isfinite(ms)
        mot_real[m], mot_shown[m], ok[m] = np.nan_to_num(mr), np.nan_to_num(ms), good
    info = {"K": K, "N": N, "V": len(VARIANTS), "ns": ns, "variants": VARIANTS, "probe_v": (0, 1), "perm": perm.tolist(),
            "clips": [c.get("clip_id", "") for _, c in sel]}
    return {"base": base, "var": var, "info": info, "probe_motion_real": mot_real, "probe_motion_shown": mot_shown, "probe_ok": ok}


def variant_micro(base, var_in, torchify=None):
    """the base (real-target) micro with ONE variant's trunk inputs."""
    o = dict(base)
    ids_b, dep, pdep, gf = var_in
    o["ids"] = np.stack([base["ids"][0], ids_b]); o["depth"] = np.stack([base["depth"][0], dep])
    o["pdepth"], o["gfeat"] = pdep, gf
    return torchify(o) if torchify is not None else o


def reduce_variant(c, v, out):
    """per-variant sums of ONE pass (torch tensors) -> list: word (sum, n), cb0, light (mean over 31 cb), pose cb0, pose light; + timing sums
    (6 per motion kind) when v is a probe variant."""
    import torch
    b, ns = c["base"], c["info"]["ns"]
    dev = next(x for x in out.values() if torch.is_tensor(x)).device
    vals = [0.0] * 10                                                  # parts a pass does not score stay (0, 0): no metric
    if "text_stats" in out:
        st = out["text_stats"]
        word = torch.as_tensor(np.asarray(b["word"][:ns]), device=dev)
        lab = torch.as_tensor(np.asarray(b["cb0"]) >= 0, device=dev)
        vals[0:6] = [float((st[3, :ns] * word).sum()), float(word.sum()), float((out["cb0_pos"] * lab).sum()), float(lab.sum()),
                     float(out["light_frame"].sum() / 31.0), float(out["light_frame"].numel())]
    if "pose_cb0_pos" in out:
        plab = torch.as_tensor(np.asarray(b["pcb0"]) >= 0, device=dev)
        vals[6:10] = [float((out["pose_cb0_pos"] * plab).sum()), float(plab.sum()), float(out["pose_light_frame"].sum() / 31.0),
                      float(out["pose_light_frame"].numel())]
    if v in c["info"]["probe_v"]:
        st = out["text_stats"]
        ok = torch.as_tensor(c["probe_ok"], device=dev).double()
        p = st[2, ns:2 * ns].double()
        for mk in ("probe_motion_real", "probe_motion_shown"):
            m = torch.as_tensor(c[mk], device=dev).double()
            vals += [float(ok.sum()), float((p * ok).sum()), float((m * ok).sum()), float((p * p * ok).sum()), float((m * m * ok).sum()), float((p * m * ok).sum())]
    return vals


def summarize(t, V=len(VARIANTS), n_probe=2):
    """all-reduced flat sums (variant 0..V-1 x 10, then the probe variants' timing sums 12 each) -> codependence/* metrics."""
    d, k = {}, 0
    s = {}
    for v in range(V):
        s[v] = t[k:k + 10]; k += 10
    nm = VARIANTS
    m = lambda v, i: s[v][2 * i] / max(s[v][2 * i + 1], 1.0)  # noqa: E731
    keys = ("word_ce_norm", "cb0_ce", "light_ce", "pose_cb0_ce", "pose_light_ce")
    for v in range(V):
        for i, kk in enumerate(keys):
            if s[v][2 * i + 1] > 0:
                d[f"codependence/{nm[v]}/{kk}"] = m(v, i)
    for v in (1, 2):                                                      # pose -> speech
        for i, kk in enumerate(keys[:3]):
            d[f"codependence/pose_to_speech/d_{kk}_{nm[v]}"] = m(v, i) - m(0, i)
    for v in (3, 4):                                                      # speech -> pose
        for i, kk in enumerate(keys[3:], 3):
            d[f"codependence/speech_to_pose/d_{kk}_{nm[v]}"] = m(v, i) - m(0, i)
    for pv in (0, 1)[:n_probe]:
        for mk in ("real_motion", "shown_motion"):
            n, sx, sy, sxx, syy, sxy = t[k:k + 6]; k += 6
            cov = sxy / n - (sx / n) * (sy / n) if n else 0.0
            vx, vy = (sxx / n - (sx / n) ** 2 if n else 0.0), (syy / n - (sy / n) ** 2 if n else 0.0)
            d[f"codependence/timing/r_pnw_{mk}_{nm[pv]}_input"] = cov / max((vx * vy) ** 0.5, 1e-30)
            d["codependence/timing/frames"] = n
    d["codependence/frames_per_variant"] = s[0][3]                     # = scored cb0 frames of the real pass (all ranks)
    return d


if __name__ == "__main__":
    print(os.path.isfile(f"{TOK}/model.safetensors"), len(rvq_tables()))
