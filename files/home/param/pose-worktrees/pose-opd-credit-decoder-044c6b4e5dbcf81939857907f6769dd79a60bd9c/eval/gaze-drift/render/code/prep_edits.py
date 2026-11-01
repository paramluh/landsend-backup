"""Gaze-drift causal render tests: build vdata dirs whose pose latents come from EDITED decoded keypoints.
Reimplements demo-v3 pose.tokens_to_latents (video/demo-v3-1019/pose/__init__.py) step for step, with an edit hook between
decode/offsets_to_positions and raw_to_recon. Production code is imported, never modified.
Edits (pixel space, head = COCO-WB 0..4, shoulders 5/6, prefill = first 2P pose frames @25 fps; only frames >= 2P are edited):
  E0 none      unmodified (must reproduce the sample's own vdata pose latents)
  E1 smooth    head points rel. to shoulder midpoint Gaussian-smoothed in time, sigma 3 frames
  E2 headlock  head points rel. to shoulder midpoint := their prefill mean
  E3 shape     keep generated eye midpoint (translation); head shape (offsets rel. eye midpoint / eye distance) := prefill mean,
               rescaled by the prefill-mean eye distance
Confidences are never edited.
   python prep_edits.py ROW:RID:K [...]          (venv-distilled, 1 GPU)"""
import json
import os
import sys

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from safetensors.torch import load_file, save_file

DEMO = "/data/home/amartyavarma/video/demo-v3-1019"
sys.path.insert(0, DEMO)
import pose as POSE  # noqa: E402
from pose import H, W, SEL, global_state, decode  # noqa: E402
from pose.kinematic import offsets_to_positions, raw_to_recon  # noqa: E402
from pose.raster import render_pose_canvas  # noqa: E402

REQ = "/data/home/amartyavarma/ppm/ppm-pose-server/requests"
OUT = "/data/home/amartyavarma/ppm/gaze-drift/render/vdata"
HEAD = [0, 1, 2, 3, 4]


def edit_kp(kp, mode, F0):
    """kp [T, 133, 3] (x, y, conf) px -> edited copy. F0 = first generated pose frame (25 fps)."""
    kp = kp.clone()
    xy = kp[:, :, :2].numpy().astype(np.float64)
    smid = (xy[:, 5] + xy[:, 6]) / 2                                     # [T, 2]
    rel = xy[:, HEAD] - smid[:, None]                                    # [T, 5, 2]
    if mode == "none":
        return kp
    if mode == "smooth":
        rs = gaussian_filter1d(rel, sigma=3.0, axis=0, mode="nearest")
        new = smid[:, None] + rs
    elif mode == "headlock":
        new = smid[:, None] + rel[:F0].mean(0)[None]
    elif mode == "halfmotion":                                           # ALL 55 points: deviation from a 1 s (25-frame) moving average x0.5
        sel = np.array(SEL)
        a = xy[:, sel]
        ma = uniform_filter1d(a, size=25, axis=0, mode="nearest")
        xy[F0:, sel] = (ma + 0.5 * (a - ma))[F0:]
        kp[:, :, :2] = torch.from_numpy(xy).float()
        return kp
    elif mode == "orient":                                               # remove the SLOW head-orientation offset, keep fast motion
        emid = (xy[:, 1] + xy[:, 2]) / 2
        ed = np.linalg.norm(xy[:, 1] - xy[:, 2], axis=-1)
        shape = (xy[:, HEAD] - emid[:, None]) / ed[:, None, None]         # [T, 5, 2]
        lp = uniform_filter1d(shape, size=50, axis=0, mode="nearest")      # 2 s moving average
        new = emid[:, None] + (shape - (lp - shape[:F0].mean(0)[None])) * ed[:, None, None]
    elif mode == "shape":
        emid = (xy[:, 1] + xy[:, 2]) / 2
        ed = np.linalg.norm(xy[:, 1] - xy[:, 2], axis=-1)                # [T]
        shape = (xy[:, HEAD] - emid[:, None]) / ed[:, None, None]
        new = emid[:, None] + shape[:F0].mean(0)[None] * ed[:F0].mean()
    else:
        raise ValueError(mode)
    xy[F0:, HEAD] = new[F0:]
    kp[:, :, :2] = torch.from_numpy(xy).float()
    return kp


@torch.no_grad()
def latents(m, codes, frame0_pose, mode, F0):
    """= pose.tokens_to_latents with the edit hook; -> (out dict, kp [T, 133, 3] edited, kp_orig)."""
    codes = torch.as_tensor(codes).long()
    codes = codes[:, :codes.shape[1] // 2 * 2]
    yxc = torch.as_tensor(frame0_pose).float().numpy()
    kp0 = torch.from_numpy(np.ascontiguousarray(yxc[:, [1, 0, 2]]))
    gs = global_state(kp0.numpy())
    dec = decode(m['tok'], codes)
    T = len(dec)
    f = dec.view(T, 55, 3)
    g = gs.expand(T, 3)
    pos = offsets_to_positions(g[:, :2], f[..., :2] * g[:, None, 2:3])
    kp = torch.zeros(T, 133, 3)
    kp[:, SEL, :2] = torch.stack([pos[..., 1] * W, pos[..., 0] * H], -1)
    kp[:, SEL, 2] = f[..., 2]
    kpe = edit_kp(kp, mode, F0)                                          # ---- the hook ----
    recon, g = raw_to_recon(torch.cat([kpe, kpe[-1:]]), kp0, H, W)
    canvas = render_pose_canvas(recon[None].cuda(), g[None].cuda(), H, W)
    z = m['enc'].encode(canvas)[0].transpose(0, 1).to(torch.bfloat16).contiguous().cpu()
    assert z.shape == (T // 4 + 1, 48, 30, 16), z.shape
    return {'latents': z, 'n_frames': torch.tensor(T + 1), 'valid_frames': torch.tensor(T)}, kpe, kp


GTV = "/data/home/amartyavarma/video/studio-posemono/sam8b-eval-cfg"
PR = "/data/home/amartyavarma/video/demo-data/prompts"


def main_gt(m, specs):
    """GT:ROW:CLIP -> E7 (GT pose, head locked to prefill-mean offset from shoulder midpoint) and E8 (GT pose, all motion halved toward
    a 1 s moving average), both with GT Mimi, frames >= 50 edited; plus an unedited re-encode checked against the GT vdata latents."""
    rep = {}
    for spec in specs:
        _, row, clip = spec.split(":")
        z = np.load(f"{GTV}/runs/{row}/codes.npz")
        f0 = load_file(f"{PR}/{clip}/llm_inputs.safetensors")["frame0_pose"].float()
        gv = f"{GTV}/vdata/GT-{clip}"
        ref = load_file(f"{gv}/pose_latents_electron_v3.safetensors")["latents"].float()
        kps = {}
        for e, mode in (("GT", "none"), ("E7", "headlock"), ("E8", "halfmotion")):
            out, kpe, kp = latents(m, torch.from_numpy(z["gt_pose"].astype(np.int64)), f0, mode, 50)
            d = out["latents"].float() - ref
            r = {"mode": mode, "gt_tokens": list(z["gt_pose"].shape), "lat_rel_diff_vs_GT_vdata": float(d.norm() / ref.norm()),
                 "head_px_change_mean": float((kpe[50:, HEAD, :2] - kp[50:, HEAD, :2]).norm(dim=-1).mean()),
                 "all_px_change_mean": float((kpe[50:, SEL, :2] - kp[50:, SEL, :2]).norm(dim=-1).mean())}
            kps[e] = kpe[:, :13].numpy()
            if e != "GT":
                key = f"GAZE-{e}-{row}"
                vd = f"{OUT}/{key}"
                os.makedirs(vd, exist_ok=True)
                save_file(out, f"{vd}/pose_latents_electron_v3.safetensors")
                save_file(load_file(f"{gv}/mimi_local.safetensors"), f"{vd}/mimi_local.safetensors")
                if not os.path.lexists(f"{vd}/video_latents.safetensors"):
                    os.symlink(f"{gv}/video_latents.safetensors", f"{vd}/video_latents.safetensors")
                r["key"] = key
            rep[f"{e}-{row}"] = r
            print(e, row, json.dumps(r), flush=True)
        np.savez_compressed(f"{OUT}/kpgt-{row}.npz", **kps, prefill_pose_frames=50)
    json.dump(rep, open(f"{OUT}/prep_gt_report.json", "w"), indent=1)


@torch.no_grad()
def encode_kp(m, kp, kp0):
    """kp [T, 133, 3] (x, y, conf) px -> pose latents dict exactly as tokens_to_latents' tail."""
    T = len(kp)
    recon, g = raw_to_recon(torch.cat([kp, kp[-1:]]), kp0, H, W)
    canvas = render_pose_canvas(recon[None].cuda(), g[None].cuda(), H, W)
    z = m['enc'].encode(canvas)[0].transpose(0, 1).to(torch.bfloat16).contiguous().cpu()
    assert z.shape == (T // 4 + 1, 48, 30, 16), z.shape
    return {'latents': z, 'n_frames': torch.tensor(T + 1), 'valid_frames': torch.tensor(T)}


def main_pp(m, specs):
    """PP:ROW:CLIP:RID:K -> E10: REAL pose for 20 s = GT keypoints 0..299 then played backwards 299..100 (ping-pong, no jump),
    + the generated sample's 20 s Mimi. Horizon test: does gaze drift after 12 s even with real pose?"""
    rep = {}
    for spec in specs:
        _, row, clip, rid, k = spec.split(":")
        z = np.load(f"{GTV}/runs/{row}/codes.npz")
        f0 = load_file(f"{PR}/{clip}/llm_inputs.safetensors")["frame0_pose"].float()
        _, _, kp = latents(m, torch.from_numpy(z["gt_pose"].astype(np.int64)), f0, "none", 50)
        kp0 = torch.from_numpy(np.ascontiguousarray(f0.numpy()[:, [1, 0, 2]]))
        pp = torch.cat([kp, kp.flip(0)[:200]])                                       # 300 + 200 = 500 frames
        out = encode_kp(m, pp, kp0)
        sd = f"{REQ}/{rid}/s{k}"
        key = f"GAZE-E10-{row}"
        vd = f"{OUT}/{key}"
        os.makedirs(vd, exist_ok=True)
        save_file(out, f"{vd}/pose_latents_electron_v3.safetensors")
        save_file(load_file(f"{sd}/vdata/mimi_local.safetensors"), f"{vd}/mimi_local.safetensors")
        if not os.path.lexists(f"{vd}/video_latents.safetensors"):
            os.symlink(f"{sd}/vdata/video_latents.safetensors", f"{vd}/video_latents.safetensors")
        gl = load_file(f"{GTV}/vdata/GT-{clip}/pose_latents_electron_v3.safetensors")["latents"].float()
        rep[key] = {"frames": len(pp), "latents": list(out["latents"].shape), "first12s_rel_diff_vs_GT_vdata": float((out["latents"][:75].float() - gl[:75]).norm() / gl[:75].norm())}
        np.savez_compressed(f"{OUT}/kppp-{row}.npz", E10=pp[:, :13].numpy())
        print(key, json.dumps(rep[key]), flush=True)
    json.dump(rep, open(f"{OUT}/prep_pp_report.json", "w"), indent=1)


ARML = [7, 9] + list(range(91, 112))       # left elbow, wrist, left hand (hangs off wrist 9)  -> attached to shoulder 5
ARMR = [8, 10] + list(range(112, 133))     # right elbow, wrist, right hand (wrist 10)          -> attached to shoulder 6


def main_hy(m, specs):
    """HY:ROW:CLIP:RID:K -> 20 s hybrids of the generated pose G and the REAL pose R (GT keypoints 0..299 then 299..100, as E10h),
    frames >= 50 edited, positions + confidences taken from the named source, generated sample's Mimi:
      E11 real body + generated head (head 0..4 = R shoulder-mid + G head offsets from G shoulder-mid)
      E12 generated body + real head (head 0..4 = G shoulder-mid + R head offsets from R shoulder-mid)
      E13 generated head + torso, both arm chains (elbow, wrist, hand) real, each attached to its generated shoulder"""
    rep = {}
    for spec in specs:
        _, row, clip, rid, k = spec.split(":")
        sd = f"{REQ}/{rid}/s{k}"
        zs = np.load(f"{sd}/codes.npz")
        zg = np.load(f"{GTV}/runs/{row}/codes.npz")
        f0s = torch.from_numpy(zs["frame0_pose"]).float()
        f0g = load_file(f"{PR}/{clip}/llm_inputs.safetensors")["frame0_pose"].float()
        _, _, G = latents(m, torch.from_numpy(zs["pose"].astype(np.int64)), f0s, "none", 50)
        _, _, Rk = latents(m, torch.from_numpy(zg["gt_pose"].astype(np.int64)), f0g, "none", 50)
        Rr = torch.cat([Rk, Rk.flip(0)[:200]])
        assert len(Rr) == len(G) == 500
        kp0 = torch.from_numpy(np.ascontiguousarray(f0s.numpy()[:, [1, 0, 2]]))
        sm = lambda K: (K[:, 5, :2] + K[:, 6, :2]) / 2  # noqa: E731
        out = {}
        e11 = Rr.clone(); e11[:, :5, :2] = sm(Rr)[:, None] + (G[:, :5, :2] - sm(G)[:, None]); e11[:, :5, 2] = G[:, :5, 2]
        e12 = G.clone(); e12[:, :5, :2] = sm(G)[:, None] + (Rr[:, :5, :2] - sm(Rr)[:, None]); e12[:, :5, 2] = Rr[:, :5, 2]
        e13 = G.clone()
        for idx, sh in ((ARML, 5), (ARMR, 6)):
            e13[:, idx, :2] = G[:, sh:sh + 1, :2] + (Rr[:, idx, :2] - Rr[:, sh:sh + 1, :2]); e13[:, idx, 2] = Rr[:, idx, 2]
        kps = {}
        for e, kk in (("E11", e11), ("E12", e12), ("E13", e13)):
            kk[:50] = G[:50]
            o = encode_kp(m, kk, kp0)
            key = f"GAZE-{e}-{row}"
            vd = f"{OUT}/{key}"
            os.makedirs(vd, exist_ok=True)
            save_file(o, f"{vd}/pose_latents_electron_v3.safetensors")
            save_file(load_file(f"{sd}/vdata/mimi_local.safetensors"), f"{vd}/mimi_local.safetensors")
            if not os.path.lexists(f"{vd}/video_latents.safetensors"):
                os.symlink(f"{sd}/vdata/video_latents.safetensors", f"{vd}/video_latents.safetensors")
            kps[e] = kk[:, :13].numpy()
            nose = kk[50:, 0, :2]
            wr = kk[50:, [9, 10], :2]
            sw = (kk[50:, 5, :2] - kk[50:, 6, :2]).norm(dim=-1)
            rep[key] = {"wrist_nose_dist_sw_median": float(((wr - nose[:, None]).norm(dim=-1).min(1).values / sw).median())}
            print(key, json.dumps(rep[key]), flush=True)
        rep[f"frame0_maxdiff-{row}"] = float((f0s - f0g).abs().max())
        np.savez_compressed(f"{OUT}/kphy-{row}.npz", **kps)
    json.dump(rep, open(f"{OUT}/prep_hy_report.json", "w"), indent=1)


def main():
    torch.set_num_threads(4)
    m = POSE.load()
    report = {}
    if sys.argv[1].startswith("GT:"):
        return main_gt(m, sys.argv[1:])
    if sys.argv[1].startswith("HY:"):
        return main_hy(m, sys.argv[1:])
    if sys.argv[1].startswith("PP:"):
        return main_pp(m, sys.argv[1:])
    for spec in sys.argv[1:]:
        row, rid, k = spec.split(":")
        sd = f"{REQ}/{rid}/s{k}"
        z = np.load(f"{sd}/codes.npz")
        P = int(z["prefill_frames"])
        F0 = 2 * P
        ref = load_file(f"{sd}/vdata/pose_latents_electron_v3.safetensors")
        kps = {}
        modes = [tuple(x.split("=")) for x in os.environ.get("MODES", "E0=none,E1=smooth,E2=headlock,E3=shape").split(",")]
        for e, mode in modes:
            key = f"GAZE-{e}-{row}"
            out, kpe, kp = latents(m, torch.from_numpy(z["pose"].astype(np.int64)), torch.from_numpy(z["frame0_pose"]).float(), mode, F0)
            vd = f"{OUT}/{key}"
            os.makedirs(vd, exist_ok=True)
            save_file(out, f"{vd}/pose_latents_electron_v3.safetensors")
            for fn in ("mimi_local.safetensors",):
                save_file(load_file(f"{sd}/vdata/{fn}"), f"{vd}/{fn}")
            if not os.path.lexists(f"{vd}/video_latents.safetensors"):
                os.symlink(f"{sd}/vdata/video_latents.safetensors", f"{vd}/video_latents.safetensors")
            d = (out["latents"].float() - ref["latents"].float())
            r = {"mode": mode, "rid": rid, "k": int(k), "prefill_pose_frames": F0, "T": int(kp.shape[0]),
                 "lat_maxabs_diff_vs_sample": float(d.abs().max()), "lat_rel_diff_vs_sample": float(d.norm() / ref["latents"].float().norm()),
                 "lat_rel_diff_prefill_latents": float(d[:F0 // 4 + 1].norm() / ref["latents"][:F0 // 4 + 1].float().norm()),
                 "head_px_change_mean": float((kpe[F0:, HEAD, :2] - kp[F0:, HEAD, :2]).norm(dim=-1).mean())}
            report[key] = r
            kps[e] = kpe[:, :13].numpy()
            print(key, json.dumps(r), flush=True)
        np.savez_compressed(f"{OUT}/kp-{row}.npz", **kps, prefill_pose_frames=F0)
    json.dump(report, open(f"{OUT}/prep_report.json", "w"), indent=1)


if __name__ == "__main__":
    main()
