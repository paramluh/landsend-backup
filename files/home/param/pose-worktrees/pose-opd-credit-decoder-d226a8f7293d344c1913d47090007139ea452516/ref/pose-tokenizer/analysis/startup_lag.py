"""Startup-instability + temporal-lag diagnostics for the causal pose tokenizer.

Read-only. Runs single-pass inference (no chunking) on two causal checkpoints
(a no-shift `causal_loss_shift=0` model and a loss-shift `causal_loss_shift=1`
model), reusing cached RTMW npz poses, and produces:

  A. Startup: encoder-vs-decoder localization
     A1  encoder code stability under nested prefixes (context starvation)
     A2  decoder cold-start with fixed "good" codes
     A3  warm-up prefix intervention on the full pipeline
     A4  raw (undeshifted) per-frame L1 for both models (de-shift re-index test)
  B. Lag reality
     B1  per-clip frame-shift sweep (argmin d*, L1 drop vs d=0)
     B2  lag / shift-benefit vs GT motion speed
     B3  per-region (body/hands/face) optimal shift
  C. no-shift vs loss-shift residual lag (from B)

Outputs CSVs + JSON + (optional) plots under OUTDIR, and prints a summary.

Run:
    cd /home/james/pose-tokenizer
    uv run python analysis/startup_lag.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Reuse the preprocessing / model-resolution helpers from the inference script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from infer_reconstruct_video import (  # noqa: E402
    resolve_model_and_config,
    load_raw_pose_npz,
    infer_coord_order,
    preprocess_to_features,
    resample_to_fps,
    make_model_input,
    split_model_output,
    load_norm_stats,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CKPTS = {
    "noshift": "/mnt/somfs/james-checkpoints/pose-tokenizer/base-causal-pretrain-32cb-pose20k-subset1.2tb/step_200000",
    "shift":   "/mnt/somfs/james-checkpoints/pose-tokenizer-james/32cb_size_2048/base-causal-2x-wide-32cb-qd0.5-lr1e-4-lossshift1-root-frame-0-cb_size_2048/step_150000",
}
NPZ_GLOB = "/home/james/aloe_valds_recon_32cb/*_rtmwx.npz"
OUTDIR = Path("/home/james/pose_tokenizer_diag")
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# Cap clip length so single-pass inference stays within the training window
# regime (segment_frames is 80 for the shift model, 128 for no-shift). Keeps
# the startup + lag analysis in-distribution and avoids long-clip OOD drift.
CLIP_CAP = 80

# Region joint groups in the 55-joint SELECTED space.
FACE_J = list(range(0, 5))
BODY_J = list(range(5, 13))
HAND_J = list(range(13, 55))
REGIONS = {"all": list(range(55)), "body": BODY_J, "hands": HAND_J, "face": FACE_J}


# ---------------------------------------------------------------------------
# Model / data helpers
# ---------------------------------------------------------------------------

def load_model(ck):
    model, cfg, name, ckdir = resolve_model_and_config(ck, DEVICE)
    model.eval()
    shift = int(getattr(cfg, "causal_loss_shift", 0) or 0)
    inf = model.config.input_features
    tf = float(cfg.target_fps or 25)
    nc, ns = load_norm_stats(ckdir, inf)
    return dict(model=model, cfg=cfg, inf=inf, tf=tf, shift=shift, nc=nc, ns=ns)


def to_coords(rec165):
    """(T,165) interleaved (y,x,conf) per joint -> (T,110) coords only."""
    T = rec165.shape[0]
    return rec165.reshape(T, 55, 3)[:, :, :2].reshape(T, 110)


def clip_features(npz, tf, inf, nc, ns):
    """Return model_input (T,165) normalized as training, and gt coords (T,110)."""
    rk_, sc, sf, w, h = load_raw_pose_npz(npz)
    order = infer_coord_order(rk_, h, w, "auto")
    kf, conf, gs = preprocess_to_features(rk_, sc, h, w, order)
    kf, conf, gs = resample_to_fps([kf, conf, gs], sf, tf)
    mi = make_model_input(kf, conf, inf)
    if nc is not None and ns is not None:
        mi = (mi - nc) / ns
    mi = mi[:CLIP_CAP]; kf = kf[:CLIP_CAP]
    return mi.astype(np.float32), kf.astype(np.float32)  # (T,165) model input, (T,110) gt coords


@torch.no_grad()
def forward_single(model, mi):
    """Single-pass forward. mi: (T,165) -> recon coords (T,110)."""
    x = torch.from_numpy(mi).float().unsqueeze(0).to(DEVICE)
    rec = model(x)["reconstruction"].squeeze(0).cpu().numpy()  # (T,165)
    return to_coords(rec)


def region_l1(gt, rec, idx, warm=0):
    """Mean L1 over joint indices idx, frames [warm:], on (T,110) coord arrays."""
    T = min(len(gt), len(rec))
    g = gt[:T].reshape(T, 55, 2)[warm:, idx]
    r = rec[:T].reshape(T, 55, 2)[warm:, idx]
    return float(np.abs(g - r).mean())


def deshift(rec, s):
    if s > 0 and rec.shape[0] > s:
        return np.concatenate([rec[s:], np.repeat(rec[-1:], s, axis=0)], axis=0)
    return rec


# ---------------------------------------------------------------------------
# A. Startup: encoder vs decoder
# ---------------------------------------------------------------------------

@torch.no_grad()
def encode_codes_latent(model, mi):
    """(T,165) -> (codes stacked (n_cb,T'), continuous latent z (C,T'))."""
    x = torch.from_numpy(mi).float().unsqueeze(0).to(DEVICE)
    z = model.encoder(model.preprocess(x))          # (1,C,T')
    z_q, codes, _, _ = model.quantizer(z)
    codes = torch.stack([c.squeeze(0) for c in codes], 0).cpu().numpy()  # (n_cb,T')
    return codes, z.squeeze(0).cpu().numpy()        # (n_cb,T'), (C,T')


@torch.no_grad()
def A1_encoder_prefix(model, clips, M=24, prefixes=(0, 2, 4, 8, 16, 32, 48)):
    """Right-edge-aligned content window of M frames; vary real prefix length.
    Compare its codes/latent to the deepest-prefix (gold) version.
    Returns dict p -> (code_disagree_frac, latent_l2)."""
    Ktok = M // 2
    agg = {p: [] for p in prefixes}
    lat = {p: [] for p in prefixes}
    for mi, _ in clips:
        T = mi.shape[0]
        pmax = max(pp for pp in prefixes if pp + M <= T) if any(pp + M <= T for pp in prefixes) else None
        if pmax is None:
            continue
        a = T - M                                     # content = suffix [a:T]
        # gold: use the largest available prefix (deepest context)
        gp = min(pmax, a)
        gold_codes, gold_z = encode_codes_latent(model, mi[a - gp:T])
        gold_codes = gold_codes[:, -Ktok:]; gold_z = gold_z[:, -Ktok:]
        for p in prefixes:
            if p > a or p + M > T:
                continue
            codes, z = encode_codes_latent(model, mi[a - p:T])
            codes = codes[:, -Ktok:]; z = z[:, -Ktok:]
            agg[p].append(float((codes != gold_codes).mean()))
            lat[p].append(float(np.abs(z - gold_z).mean()))
    return {p: (float(np.mean(v)) if v else float("nan"),
               float(np.mean(lat[p])) if lat[p] else float("nan"))
            for p, v in agg.items()}


@torch.no_grad()
def A2_decoder_coldstart(model, clips, maxpos=16):
    """Fix good codes from full-clip encode. Decode full vs decode a suffix of
    codes; compare the SAME content frames (right-edge aligned). Per-position
    (from suffix start) L1 isolates decoder cold-start."""
    acc = np.zeros(maxpos); cnt = np.zeros(maxpos)
    for mi, _ in clips:
        x = torch.from_numpy(mi).float().unsqueeze(0).to(DEVICE)
        _, codes, _, _ = model.encode(x)                       # good codes
        Tt = codes[0].shape[1]
        j = Tt // 2                                            # start suffix mid-stream
        if Tt - j < 4:
            continue
        full = to_coords(model.decode(codes).squeeze(0).cpu().numpy())     # (T,110)
        suf = to_coords(model.decode([c[:, j:] for c in codes]).squeeze(0).cpu().numpy())
        # right-edge aligned: suffix covers the last (Tt-j)*2 input frames
        Lsuf = suf.shape[0]
        full_tail = full[-Lsuf:]
        d = np.abs(suf - full_tail).mean(axis=1)  # per-frame divergence
        n = min(maxpos, len(d)); acc[:n] += d[:n]; cnt[:n] += 1
    return acc / np.maximum(cnt, 1)


@torch.no_grad()
def A2b_decoder_context(model, clips, K=3, ps=(0, 1, 2, 4, 8, 16, 32)):
    """Decoder-context sweep with FIXED warm codes (the clean design).

    Encode the whole clip once -> codes A B C ... H. Decode the last K tokens
    (the target, e.g. F G H) with varying amounts of preceding code context by
    decoding right-edge-aligned suffixes: decode(...H), decode(GH), decode(EFGH),
    decode(CDEFGH), ..., decode(ABC...H). The target's CODES are identical in
    every decode; only the decoder's preceding context changes. Measure the
    target frames' L1 vs GT and vs the deepest-context decode, as a function of
    how many preceding tokens `p` the decoder was given.
    """
    vs_gt = {p: [] for p in ps}
    vs_deep = {p: [] for p in ps}
    for mi, gt in clips:
        x = torch.from_numpy(mi).float().unsqueeze(0).to(DEVICE)
        _, codes, _, _ = model.encode(x)
        Ttok = codes[0].shape[1]
        tstart = Ttok - K                     # target tokens = [tstart:] (last K)
        if tstart < max(ps):
            continue
        pmax = min(max(ps), tstart)
        deep = to_coords(model.decode([c[:, tstart - pmax:] for c in codes]).squeeze(0).cpu().numpy())
        deep_tgt = deep[-2 * K:]              # target frames (last 2K), deepest context
        gt_tgt = gt[-2 * K:]
        for p in ps:
            if p > tstart:
                continue
            dec = to_coords(model.decode([c[:, tstart - p:] for c in codes]).squeeze(0).cpu().numpy())
            tgt = dec[-2 * K:]
            vs_gt[p].append(float(np.abs(tgt - gt_tgt).mean()))
            vs_deep[p].append(float(np.abs(tgt - deep_tgt).mean()))
    out = {}
    for p in ps:
        out[p] = (float(np.mean(vs_gt[p])) if vs_gt[p] else float("nan"),
                  float(np.mean(vs_deep[p])) if vs_deep[p] else float("nan"))
    return out


@torch.no_grad()
def A3_prefix_intervention(model, clips, Ns=(0, 2, 4, 8, 16), nframes=15):
    """Prepend warm-up frames (repeat-frame-0 and reflect), drop them from
    output, and measure per-frame L1 vs GT for the first `nframes` target frames."""
    out = {}
    for kind in ("zero", "repeat0", "reflect"):
        for N in Ns:
            acc = np.zeros(nframes); cnt = np.zeros(nframes)
            for mi, gt in clips:
                if N == 0:
                    pref = mi[:0]
                elif kind == "zero":
                    pref = np.zeros((N, mi.shape[1]), np.float32)
                elif kind == "repeat0":
                    pref = np.repeat(mi[:1], N, axis=0)
                else:  # reflect early frames
                    pref = mi[1:N + 1][::-1] if mi.shape[0] > N else np.repeat(mi[:1], N, axis=0)
                inp = np.concatenate([pref, mi], axis=0) if N else mi
                rec = forward_single(model, inp)[N:]              # drop warm-up outputs
                d = region_l1_perframe(gt, rec, nframes)
                n = min(nframes, len(d)); acc[:n] += d[:n]; cnt[:n] += 1
            out[f"{kind}_N{N}"] = (acc / np.maximum(cnt, 1)).tolist()
            if kind == "zero" and N > 0:
                # zero-prefix is identical to N=0 baseline except for output cropping; keep only N=0
                pass
    return out


def region_l1_perframe(gt, rec, nframes):
    T = min(len(gt), len(rec), nframes)
    g = gt[:T].reshape(T, 55, 2); r = rec[:T].reshape(T, 55, 2)
    return np.abs(g - r).mean(axis=(1, 2))


@torch.no_grad()
def A5_enc_vs_dec(model, clips):
    """Attribute the cold-start to encoder vs decoder IN THE SAME OUTPUT UNITS.

    At a deep mid-clip token t0 (output frame f0=2*t0), measure L1 (vs the fully
    warm reference reconstruction) of two independent single-stage starvations:
      encoder-cold: replace token t0 with the code the encoder emits when that
        content is at the very start of its input (no past context), decode with
        the FULL (warm) code context.
      decoder-cold: keep the warm codes, but decode only the suffix starting at
        t0 so the decoder has no preceding code context.
      both-cold: suffix decode AND the cold code (approximates the true clip start).
    """
    enc, dec, both = [], [], []
    for mi, gt in clips:
        x = torch.from_numpy(mi).float().unsqueeze(0).to(DEVICE)
        _, codes_full, _, _ = model.encode(x)
        Ttok = codes_full[0].shape[1]
        t0 = Ttok // 2
        if t0 < 4 or Ttok - t0 < 4:
            continue
        f0 = 2 * t0
        R_ww = to_coords(model.decode(codes_full).squeeze(0).cpu().numpy())     # warm/warm reference
        if f0 >= R_ww.shape[0]:
            continue

        # decoder-cold: warm codes, decode suffix [t0:] (right-edge aligned -> frame0 == f0)
        R_suf = to_coords(model.decode([c[:, t0:] for c in codes_full]).squeeze(0).cpu().numpy())
        ref_tail = R_ww[-R_suf.shape[0]:]
        dec.append(float(np.abs(R_suf[0] - ref_tail[0]).mean()))

        # encoder-cold: cold code for f0 content (encode with no prefix), splice, warm decode
        cold_codes, _ = encode_codes_latent(model, mi[f0:])   # (n_cb, Twin); token 0 is starved
        spliced = [c.clone() for c in codes_full]
        for k in range(len(spliced)):
            spliced[k][0, t0] = int(cold_codes[k, 0])
        R_cw = to_coords(model.decode(spliced).squeeze(0).cpu().numpy())
        enc.append(float(np.abs(R_cw[f0] - R_ww[f0]).mean()))

        # both-cold: cold code at suffix start + cold decoder
        suf_codes = [c[:, t0:].clone() for c in codes_full]
        for k in range(len(suf_codes)):
            suf_codes[k][0, 0] = int(cold_codes[k, 0])
        R_cc = to_coords(model.decode(suf_codes).squeeze(0).cpu().numpy())
        both.append(float(np.abs(R_cc[0] - ref_tail[0]).mean()))
    return dict(encoder=float(np.mean(enc)), decoder=float(np.mean(dec)),
                both=float(np.mean(both)), n=len(enc))


@torch.no_grad()
def A4_perframe_raw(models, clips, nframes=30):
    """Per-frame L1 vs frame index for both models under three alignments:
      raw   : recon[t] vs gt[t]           (no compensation)
      plus1 : recon[t+1] vs gt[t]         (advance by 1 = remove the measured
              intrinsic +1 lag; isolates cold-start from lag)
      model : advance by the model's own causal_loss_shift
    Averaged over clips. Lets us separate lag from context-starvation cold-start."""
    res = {}
    for tag, m in models.items():
        accs = {k: (np.zeros(nframes), np.zeros(nframes)) for k in ("raw", "plus1", "model")}
        for mi, gt in m["clips"]:
            rec = forward_single(m["model"], mi)
            variants = {"raw": rec, "plus1": deshift(rec, 1), "model": deshift(rec, m["shift"])}
            for k, r in variants.items():
                d = region_l1_perframe(gt, r, nframes)
                n = min(nframes, len(d)); accs[k][0][:n] += d[:n]; accs[k][1][:n] += 1
        res[tag] = {k: (a / np.maximum(c, 1)).tolist() for k, (a, c) in accs.items()}
        res[tag]["shift"] = m["shift"]
    return res


# ---------------------------------------------------------------------------
# B. Lag
# ---------------------------------------------------------------------------

@torch.no_grad()
def B_shift_sweep(model, clips, ds=range(-3, 4), warm=8):
    """Per-clip: L1(d)=mean_t|rec[t+d]-gt[t]| over frames>=warm, per region.
    Returns list of per-clip dicts with region -> {d: L1}, argmin, motion."""
    rows = []
    for name, (mi, gt) in clips:
        rec = forward_single(model, mi)
        T = min(len(gt), len(rec))
        g = gt[:T].reshape(T, 55, 2); r = rec[:T].reshape(T, 55, 2)
        motion = float(np.abs(np.diff(g[warm:], axis=0)).mean())
        row = {"clip": name, "motion": motion, "regions": {}}
        for reg, idx in REGIONS.items():
            l1 = {}
            for d in ds:
                if d >= 0:
                    a = r[warm + d:T, idx]; b = g[warm:T - d, idx]
                else:
                    a = r[warm:T + d, idx]; b = g[warm - d:T, idx]
                if len(a) < 4:
                    continue
                l1[d] = float(np.abs(a - b).mean())
            dstar = min(l1, key=l1.get)
            row["regions"][reg] = {"l1": l1, "dstar": dstar,
                                    "l1_0": l1.get(0, float("nan")),
                                    "l1_best": l1[dstar],
                                    "drop_frac": (l1.get(0, np.nan) - l1[dstar]) / l1.get(0, np.nan)}
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Plotting (optional)
# ---------------------------------------------------------------------------

def try_plots(a1, a2, a3, a4, bres):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plots] matplotlib unavailable ({e}); skipping plots")
        return
    # A4 per-frame ramp
    fig, ax = plt.subplots(figsize=(7, 4))
    for tag, r in a4.items():
        ax.plot(r["raw"], label=f"{tag} raw")
        ax.plot(r["plus1"], "--", label=f"{tag} +1-aligned")
    ax.set_xlabel("frame index"); ax.set_ylabel("L1"); ax.set_title("A4: per-frame L1 (startup ramp)")
    ax.legend(fontsize=7); ax.set_xlim(0, 20); fig.tight_layout(); fig.savefig(OUTDIR / "A4_perframe.png", dpi=110); plt.close(fig)
    # A1 encoder starvation
    fig, ax = plt.subplots(figsize=(7, 4))
    ps = sorted(a1.keys()); ax.plot(ps, [a1[p][0] for p in ps], "o-", label="code disagreement")
    ax2 = ax.twinx(); ax2.plot(ps, [a1[p][1] for p in ps], "s--", color="tab:red", label="latent L1")
    ax.set_xlabel("real prefix frames"); ax.set_ylabel("code disagree frac"); ax2.set_ylabel("latent L1")
    ax.set_title("A1: encoder code stability vs prefix"); fig.tight_layout()
    fig.savefig(OUTDIR / "A1_encoder_prefix.png", dpi=110); plt.close(fig)
    # A2 decoder cold-start
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(a2, "o-")
    ax.set_xlabel("frames from suffix start"); ax.set_ylabel("L1 vs full-decode")
    ax.set_title("A2: decoder cold-start (same codes)"); fig.tight_layout()
    fig.savefig(OUTDIR / "A2_decoder_coldstart.png", dpi=110); plt.close(fig)
    # A3 prefix intervention
    fig, ax = plt.subplots(figsize=(7, 4))
    for k, v in a3.items():
        if k.startswith("repeat0") or k.startswith("reflect"):
            ax.plot(v, label=k)
    ax.set_xlabel("target frame"); ax.set_ylabel("L1"); ax.set_title("A3: warm-up prefix intervention")
    ax.legend(fontsize=7); ax.set_xlim(0, 14); fig.tight_layout()
    fig.savefig(OUTDIR / "A3_prefix.png", dpi=110); plt.close(fig)
    # B2 motion vs dstar-benefit
    fig, ax = plt.subplots(figsize=(7, 4))
    for tag, rows in bres.items():
        mo = [r["motion"] for r in rows]
        drop = [r["regions"]["all"]["drop_frac"] for r in rows]
        ax.scatter(mo, drop, label=tag)
    ax.set_xlabel("GT motion (mean|Δ|)"); ax.set_ylabel("L1 drop frac at d*"); ax.set_title("B2: lag-benefit vs motion")
    ax.legend(); fig.tight_layout(); fig.savefig(OUTDIR / "B2_motion.png", dpi=110); plt.close(fig)
    print(f"[plots] wrote PNGs to {OUTDIR}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    npzs = sorted(glob.glob(NPZ_GLOB))
    print(f"[setup] {len(npzs)} clips, device={DEVICE}")

    models = {}
    for tag, ck in CKPTS.items():
        m = load_model(ck)
        clips = []
        named = []
        for npz in npzs:
            mi, gt = clip_features(npz, m["tf"], m["inf"], m["nc"], m["ns"])
            clips.append((mi, gt))
            named.append((os.path.basename(npz).replace("_rtmwx.npz", ""), (mi, gt)))
        m["clips"] = clips; m["named"] = named
        models[tag] = m
        print(f"[setup] {tag}: shift={m['shift']} inf={m['inf']} tf={m['tf']} clips={len(clips)}")

    summary = {}

    # ---- A. startup ----
    print("\n[A1] encoder code stability vs real prefix (BOTH models)")
    a1_both = {t: A1_encoder_prefix(models[t]["model"], models[t]["clips"]) for t in ("noshift", "shift")}
    a1 = a1_both["noshift"]  # kept for plotting back-compat
    for t in ("noshift", "shift"):
        print(f"  -- {t}: code_disagree | latentL1, by real prefix frames --")
        for p in sorted(a1_both[t]):
            print(f"     prefix={p:>2d}  code_disagree={a1_both[t][p][0]:.3f}  latentL1={a1_both[t][p][1]:.4f}")

    print("\n[A2] decoder cold-start with fixed good codes (noshift)")
    a2 = A2_decoder_coldstart(models["noshift"]["model"], models["noshift"]["clips"])
    print("   L1(suffix vs full) by frame-from-suffix-start:", np.round(a2[:10], 4).tolist())

    print("\n[A2b] decoder-context sweep, FIXED warm codes (decode last 3 tokens with p preceding)")
    a2b = {t: A2b_decoder_context(models[t]["model"], models[t]["clips"]) for t in ("noshift", "shift")}
    for t in ("noshift", "shift"):
        print(f"  -- {t}: target-frame L1 (vs GT | vs deepest-context) by preceding tokens p --")
        for p in sorted(a2b[t]):
            g, d = a2b[t][p]
            print(f"     p={p:>2d}  vsGT={g:.4f}  vsDeep={d:.4f}")

    print("\n[A3] warm-up prefix intervention (noshift)")
    a3 = A3_prefix_intervention(models["noshift"]["model"], models["noshift"]["clips"])
    for k in ["repeat0_N0", "repeat0_N4", "repeat0_N16", "reflect_N16"]:
        if k in a3: print(f"   {k:<14s} first5={np.round(a3[k][:5],4).tolist()}")

    print("\n[A4] per-frame L1 ramp, both models (raw / +1-aligned / model-deshift)")
    a4 = A4_perframe_raw(models, None)
    for tag, r in a4.items():
        ss = float(np.mean(r["raw"][15:25]))
        print(f"   {tag:<8s} steady~{ss:.4f} | raw f0={r['raw'][0]:.4f} f1={r['raw'][1]:.4f} "
              f"({r['raw'][0]/ss:.1f}x) | +1 f0={r['plus1'][0]:.4f} f1={r['plus1'][1]:.4f} "
              f"({r['plus1'][0]/ss:.1f}x) | model-deshift f0={r['model'][0]:.4f}")

    print("\n[A5] encoder vs decoder cold-start attribution (output L1 at a mid-clip token)")
    a5 = {}
    for tag in ("noshift", "shift"):
        a5[tag] = A5_enc_vs_dec(models[tag]["model"], models[tag]["clips"])
        r = a5[tag]
        share = r["encoder"] / (r["encoder"] + r["decoder"]) * 100 if (r["encoder"] + r["decoder"]) else float("nan")
        print(f"   {tag:<8s} encoder={r['encoder']:.4f}  decoder={r['decoder']:.4f}  both={r['both']:.4f}  "
              f"(encoder share={share:.0f}%, n={r['n']})")

    # ---- B. lag ----
    bres = {}
    for tag in ("noshift", "shift"):
        bres[tag] = B_shift_sweep(models[tag]["model"], models[tag]["named"])
    print("\n[B1/B3] per-clip optimal shift d* (region=all) and drop_frac")
    for tag in ("noshift", "shift"):
        print(f"  -- {tag} (raw, undeshifted) --")
        for r in bres[tag]:
            a = r["regions"]["all"]
            print(f"     {r['clip']:<11s} d*={a['dstar']:+d}  L1_0={a['l1_0']:.4f} L1*={a['l1_best']:.4f} "
                  f"drop={a['drop_frac']*100:+.0f}%  motion={r['motion']:.4f}")
        # aggregate
        import statistics as st
        dstars = [r["regions"]["all"]["dstar"] for r in bres[tag]]
        print(f"     >> median d*={st.median(dstars)}  frac(d*>0)={np.mean([d>0 for d in dstars]):.0%}")
        for reg in ("body", "hands", "face"):
            rd = [r["regions"][reg]["dstar"] for r in bres[tag]]
            print(f"        region {reg:<5s} median d*={st.median(rd):+.1f}  frac>0={np.mean([d>0 for d in rd]):.0%}")

    summary = dict(A1=a1_both, A2=a2.tolist(), A2b=a2b, A3=a3, A4=a4, A5=a5,
                   B={t: [{"clip": r["clip"], "motion": r["motion"],
                            "regions": {rg: {"dstar": v["dstar"], "l1_0": v["l1_0"],
                                             "l1_best": v["l1_best"], "drop_frac": v["drop_frac"]}
                                        for rg, v in r["regions"].items()}}
                           for r in bres[t]] for t in bres})
    (OUTDIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n[done] summary -> {OUTDIR/'summary.json'}")
    try_plots(a1, a2, a3, a4, bres)


if __name__ == "__main__":
    main()
