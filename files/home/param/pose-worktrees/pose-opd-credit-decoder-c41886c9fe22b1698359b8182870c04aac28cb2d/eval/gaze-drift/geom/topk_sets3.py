"""Follow-up: stillness/motion metrics + cb0 code-change rate on 8 sets (fs, d1-3, l250, c1-3) vs GT, matched 2-12 s (and 12-20 s).
Decodes the 4 new sets (same video-path decode as decode_all.py) into kp2.npz; reuses kp.npz for the rest. Writes topk_sets.json."""
import json
import sys

import numpy as np
import torch
from safetensors.torch import load_file

torch.set_num_threads(6)
sys.path.insert(0, "/data/home/amartyavarma/ref/pose-tokenizer")
sys.path.insert(0, "/data/home/amartyavarma/video/demo-v3-1019")
from pose_tokenizer import PoseTokenizer  # noqa: E402
from pose.kinematic import offsets_to_positions  # noqa: E402

G = "/data/home/amartyavarma/ppm/gaze-drift/geom"
CTL = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20/control"
REQ = "/data/home/amartyavarma/ppm/ppm-pose-server/requests"
PR = "/data/home/amartyavarma/video/demo-data/prompts"
H, W, FPS, PF = 480, 256, 25, 50
SEL = list(range(13)) + list(range(91, 133))
CLIP = {"S-05": "10_27286bdd", "S-13": "06_74807fe0", "S-15": "10_27286bdd", "S-20": "19_47473a46", "E-03": "03_b437950e",
        "E-05": "05_e25c6859", "E-09": "09_d9ef3c94", "E-12": "12_72cb45d3", "E-13": "13_956c09af", "E-15": "15_d1e1322c",
        "E-18": "18_1ffa9083"}
ROWS = list(CLIP)
OLD, CACHED, NEW = ["fs"], ["l250", "gz_k0l0"], ["gz_l500", "gz_l1000", "gz_t125", "gz_t15"]
SAMP = "/data/home/amartyavarma/ppm/gaze-drift/samp"
SETS = OLD + CACHED + NEW
tok = PoseTokenizer.from_pretrained("InternalCan/pose-tokenizer-32cb-200k-noshift", device="cpu",
                                    revision="1d6ff4f0de89408cb054da618e309d4334596252")


def gstate(f0):
    p = np.asarray(f0, np.float32)[:, [1, 0, 2]][SEL]
    yx = np.stack([p[:, 1] / H, p[:, 0] / W], -1)
    r0, sw = (yx[5] + yx[6]) / 2, abs(yx[5, 1] - yx[6, 1])
    return np.array([r0[0], r0[1], sw if sw > 0 else 1.0], np.float32)


@torch.no_grad()
def dec(codes, gs):
    c = torch.as_tensor(np.asarray(codes)).long()
    c = c[:, :c.shape[1] // 2 * 2]
    d = tok.decode([x[None] for x in c.unbind(0)]).float()[0]
    f = d.view(len(d), 55, 3)
    g = torch.from_numpy(gs).expand(len(d), 3)
    pos = offsets_to_positions(g[:, :2], f[..., :2] * g[:, None, 2:3])
    return torch.stack([pos[..., 1] * W, pos[..., 0] * H, f[..., 2]], -1).numpy().astype(np.float32)


z0 = np.load(f"{G}/kp.npz")
z2 = np.load(f"{G}/kp2.npz")
KP, CODES, PARAMS = {}, {}, {}
for s in SETS:
    rmap = json.load(open(f"{SAMP}/{s}.json" if s.startswith("gz_") else f"{CTL}/xs17641_{s}.json"))
    for r, (rid, k) in rmap.items():
        z = np.load(f"{REQ}/{rid}/s{k}/codes.npz")
        CODES[(s, r)] = z["pose"].astype(np.int64)
        e = json.load(open(f"{REQ}/{rid}/s{k}/meta.json"))["engine_params"]
        PARAMS[s] = {kk: e.get(kk) for kk in ["cb0_temp", "pose_rep", "pose_light_rep"] + ["pose_topk", "pose_light_topk", "light_topk", "cb0_topk", "pose_cfg_on", "pose_cfg_s", "pose_temp", "pose_light_temp", "pose_topp", "pose_light_topp"]}
        KP[(s, r)] = z0[f"{s}__{r}"] if s in OLD else z2[f"{s}__{r}"] if s in CACHED else dec(z["pose"], gstate(z["frame0_pose"]))
    print("set", s, PARAMS[s], flush=True)
for r in ROWS:
    KP[("GT", r)] = z0[f"gtpkg__{CLIP[r]}"]
    CODES[("GT", r)] = load_file(f"{PR}/{CLIP[r]}/gt/pose_tokens.safetensors")["codes"].numpy().astype(np.int64)
np.savez_compressed(f"{G}/kp3.npz", **{f"{s}__{r}": v for (s, r), v in KP.items() if s in NEW})

nrm = lambda v: np.linalg.norm(v, axis=-1)  # noqa: E731
WIN = {"2-12": (50, 300), "12-20": (300, 500)}


def metrics(kp, codes, a, b):
    xy = kp[..., :2]
    sh = (xy[:, 5] + xy[:, 6]) / 2
    sw0 = np.median(nrm(xy[:PF, 5] - xy[:PF, 6]))
    eye0 = np.median(nrm(xy[:PF, 1] - xy[:PF, 2]))
    head = nrm(np.diff((xy[:, :5] - sh[:, None]) / eye0, axis=0)).mean(1)
    torso = nrm(np.diff(sh, axis=0)) / sw0
    arms = nrm(np.diff(xy[:, 7:11] - sh[:, None], axis=0)).mean(1) / sw0
    fing = nrm(np.diff(xy[:, 13:55] - xy[:, [9] * 21 + [10] * 21], axis=0)).mean(1) / sw0
    s = slice(a, b - 1)
    ta, tb = a // 2, b // 2                                                  # token frames (12.5 Hz)
    c = codes[:, ta:tb]
    ch = (c[:, 1:] != c[:, :-1]).mean(1)                                     # per-codebook change rate
    allj = nrm(np.diff(xy, axis=0)) / sw0                                    # [T-1, 55] per-joint step, shoulder widths
    cf = kp[..., 2]
    out_canvas = ((xy[..., 0] < -0.25 * W) | (xy[..., 0] > 1.25 * W) | (xy[..., 1] < -0.25 * H) | (xy[..., 1] > 1.25 * H))
    deg = {"max_joint_jump_sw": float(allj[s].max()), "frac_frames_jump_gt_0.5sw": float((allj[s].max(1) > 0.5).mean()),
           "conf_all_mean": float(cf[a:b].mean()), "conf_body_mean": float(cf[a:b, :13].mean()), "conf_hands_mean": float(cf[a:b, 13:].mean()),
           "frac_frames_body_conf_lt_0.3": float((cf[a:b, :13].mean(1) < 0.3).mean()), "frac_kp_far_off_canvas": float(out_canvas[a:b].mean()),
           "sw_ratio_dev": float(np.abs(nrm(xy[a:b, 5] - xy[a:b, 6]) / sw0 - 1).mean())}
    return {**deg, "head_still_frac": float((head[s] < 0.01).mean()), "head_jit": float(head[s].mean()), "torso": float(torso[s].mean()),
            "arms": float(arms[s].mean()), "fingers": float(fing[s].mean()), "cb0_change": float(ch[0]), "light_change": float(ch[1:].mean()),
            "cb0_unique_frac": float(len(np.unique(c[0])) / c.shape[1])}


per = {}
for (s, r), kp in KP.items():
    for wn, (a, b) in WIN.items():
        if b <= len(kp):
            per[f"{s}/{r}/{wn}"] = metrics(kp, CODES[(s, r)], a, b)
K = list(per[f"fs/{ROWS[0]}/2-12"])
summ = {}
for k in K:
    summ[k] = {}
    for wn in WIN:
        summ[k][wn] = {s: float(np.median([per[f"{s}/{r}/{wn}"][k] for r in ROWS])) for s in ["GT"] + SETS if f"{s}/{ROWS[0]}/{wn}" in per}
    summ[k]["ratio_gen_over_gt_2-12"] = {s: float(np.median([per[f"{s}/{r}/2-12"][k] / per[f"GT/{r}/2-12"][k] for r in ROWS
                                                            if per[f"GT/{r}/2-12"][k] > 0])) for s in SETS}
json.dump({"params": PARAMS, "summary": summ, "per_sample": per,
           "note": "2-12 s matched vs GT; head_jit eye-dists/frame rel shoulders; torso/arms/fingers shoulder-widths/frame; "
                   "cb0_change = fraction of consecutive 12.5 Hz token frames where pose cb0 changes; light_change = mean over cb1-31"},
          open(f"{G}/topk_sets3.json", "w"), indent=1)
hdr = ["GT"] + SETS
print(f"{'metric (2-12 s median)':22s}" + "".join(f"{h:>8s}" for h in hdr))
for k in K:
    print(f"{k:22s}" + "".join(f"{summ[k]['2-12'][h]:8.4f}" for h in hdr))
    print(f"{'  ratio gen/GT':22s}{'':8s}" + "".join(f"{summ[k]['ratio_gen_over_gt_2-12'][h]:8.2f}" for h in SETS))
    print(f"{'  12-20 s':22s}{'':8s}" + "".join(f"{summ[k]['12-20'][h]:8.4f}" for h in SETS))

# closeness to GT: |log still ratio| + mean |log motion ratio| (head_jit, torso, arms, fingers) + |log jump-rate ratio| + |log wobble ratio| (set medians)
eps = 1e-3
def L(x, y):
    return abs(np.log((x + eps) / (y + eps)))
print("\nCLOSENESS (lower = closer to GT; set medians, 2-12 s)")
sc = {}
for st in SETS:
    m = {k: summ[k]["2-12"][st] for k in K}
    g = {k: summ[k]["2-12"]["GT"] for k in K}
    parts = {"still": L(m["head_still_frac"], g["head_still_frac"]),
             "motion": float(np.mean([abs(np.log(summ[k]["ratio_gen_over_gt_2-12"][st])) for k in ["head_jit", "torso", "arms", "fingers"]])),
             "jump": L(m["frac_frames_jump_gt_0.5sw"], g["frac_frames_jump_gt_0.5sw"]), "wobble": L(m["sw_ratio_dev"], g["sw_ratio_dev"])}
    sc[st] = {"total": float(sum(parts.values())), **{k: round(v, 3) for k, v in parts.items()}}
    print(st, sc[st])
R = json.load(open(f"{G}/topk_sets3.json")); R["closeness"] = sc; json.dump(R, open(f"{G}/topk_sets3.json", "w"), indent=1)
