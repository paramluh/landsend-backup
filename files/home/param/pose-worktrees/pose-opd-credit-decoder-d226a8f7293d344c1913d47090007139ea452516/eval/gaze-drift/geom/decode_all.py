"""Decode all 44 generated samples + 10 GT clips (package pose_tokens and runs/<row>/codes.npz gt_pose) to 25 fps keypoints,
exactly as demo-v3 pose.tokens_to_latents does up to raw_to_recon: PoseTokenizer.decode -> offsets_to_positions with
global_state(frame0_pose) -> kp px (x = pos_x * 256, y = pos_y * 480), conf = decoded conf (unclipped, as video path).
Writes geom/kp.npz : per key '<cond>/<row>' -> [T, 55, 3] (x px, y px, conf)."""
import json
import os
import sys

import numpy as np
import torch
from safetensors.torch import load_file

torch.set_num_threads(4)
sys.path.insert(0, "/data/home/amartyavarma/ref/pose-tokenizer")
sys.path.insert(0, "/data/home/amartyavarma/video/demo-v3-1019")
from pose_tokenizer import PoseTokenizer  # noqa: E402
from pose.kinematic import offsets_to_positions  # demo-v3 kinematic (same as video path)  # noqa: E402

TOK = ("InternalCan/pose-tokenizer-32cb-200k-noshift", "1d6ff4f0de89408cb054da618e309d4334596252")
H, W = 480, 256
SEL = list(range(13)) + list(range(91, 133))
CTL = "/data/home/amartyavarma/video/studio-posemono/pose-ft-samples/s20/control"
REQ = "/data/home/amartyavarma/ppm/ppm-pose-server/requests"
PR = "/data/home/amartyavarma/video/demo-data/prompts"
RUNS = "/data/home/amartyavarma/video/studio-posemono/sam8b-eval-cfg/runs"
CLIP = {"S-05": "10_27286bdd", "S-13": "06_74807fe0", "S-15": "10_27286bdd", "S-20": "19_47473a46", "E-03": "03_b437950e",
        "E-05": "05_e25c6859", "E-09": "09_d9ef3c94", "E-12": "12_72cb45d3", "E-13": "13_956c09af", "E-15": "15_d1e1322c",
        "E-18": "18_1ffa9083"}
OUT = "/data/home/amartyavarma/ppm/gaze-drift/geom"

tok = PoseTokenizer.from_pretrained(TOK[0], device="cpu", revision=TOK[1])
tok.eval() if hasattr(tok, "eval") else None


def gstate(frame0_pose_yxc):
    yxc = np.asarray(frame0_pose_yxc, np.float32)
    p = yxc[:, [1, 0, 2]][SEL]                       # (x, y, conf) px
    yx = np.stack([p[:, 1] / H, p[:, 0] / W], -1)
    root0, sw0 = (yx[5] + yx[6]) / 2, abs(yx[5, 1] - yx[6, 1])
    return np.array([root0[0], root0[1], sw0 if sw0 > 0 else 1.0], np.float32)


@torch.no_grad()
def dec(codes, gs):
    codes = torch.as_tensor(np.asarray(codes)).long()
    codes = codes[:, :codes.shape[1] // 2 * 2]
    d = tok.decode([c[None] for c in codes.unbind(0)]).float()[0].cpu()
    T = len(d)
    f = d.view(T, 55, 3)
    g = torch.from_numpy(gs).expand(T, 3)
    pos = offsets_to_positions(g[:, :2], f[..., :2] * g[:, None, 2:3])
    kp = torch.stack([pos[..., 1] * W, pos[..., 0] * H, f[..., 2]], -1)
    return kp.numpy().astype(np.float32)


res, info = {}, {}
for cond in ["fs", "d1", "d2", "d3"]:
    rmap = json.load(open(f"{CTL}/xs17641_{cond}.json"))
    for row, (rid, k) in rmap.items():
        D = f"{REQ}/{rid}/s{k}"
        z = np.load(f"{D}/codes.npz")
        meta = json.load(open(f"{D}/meta.json"))
        assert meta["clip"] == CLIP[row], (row, meta["clip"])
        gs = gstate(z["frame0_pose"])
        gs_file = z["global_state"].astype(np.float32)
        res[f"{cond}/{row}"] = dec(z["pose"], gs)
        info[f"{cond}/{row}"] = dict(rid=rid, k=k, T=int(z["pose"].shape[1]), prefill=int(z["prefill_frames"]),
                                     gs_diff=float(np.abs(gs - gs_file).max()),
                                     pose_light_topk=meta["engine_params"].get("pose_light_topk"),
                                     pose_cfg=meta["engine_params"].get("pose_cfg", meta["engine_params"].get("pose_cfg_scale")))
        print(cond, row, info[f"{cond}/{row}"], flush=True)

for row, clip in CLIP.items():
    li = load_file(f"{PR}/{clip}/llm_inputs.safetensors")
    gs = gstate(li["frame0_pose"].float().numpy())
    pt = load_file(f"{PR}/{clip}/gt/pose_tokens.safetensors")
    key = "codes"
    codes = pt[key].numpy().astype(np.int64)
    if codes.shape[0] != 32:
        codes = codes.T
    if f"gtpkg/{clip}" not in res:
        res[f"gtpkg/{clip}"] = dec(codes, gs)
        info[f"gtpkg/{clip}"] = dict(T=int(codes.shape[1]), gs_pkg_vs_llm=float(np.abs(gs - gstate(pt["frame0_pose"].numpy())).max()), gs_file_diff=float(np.abs(gs - pt["global_state"].numpy()).max()))
    z = np.load(f"{RUNS}/{row}/codes.npz")
    gp = z["gt_pose"]
    if f"gtrun/{clip}" not in res:
        res[f"gtrun/{clip}"] = dec(gp, gs)
        m = min(gp.shape[1], codes.shape[1])
        info[f"gtrun/{clip}"] = dict(T=int(gp.shape[1]), same_as_pkg_prefix=bool((gp[:, :m] == codes[:, :m]).all()))
    # does the generated prefill equal the package GT tokens' first 25 frames?
    zz = np.load(f"{REQ}/{json.load(open(f'{CTL}/xs17641_fs.json'))[row][0]}/s{json.load(open(f'{CTL}/xs17641_fs.json'))[row][1]}/codes.npz")
    P = int(zz["prefill_frames"])
    info[f"prefill_match/{row}"] = dict(pkg=bool((zz["pose"][:, :P] == codes[:, :P]).all()),
                                        pkg_off=[int(o) for o in range(0, codes.shape[1] - P) if (zz["pose"][:, :P] == codes[:, o:o + P]).all()][:3],
                                        gs_pkg_vs_sample=float(np.abs(gs - gstate(zz["frame0_pose"])).max()))
    print(row, clip, info[f"gtpkg/{clip}"], info[f"gtrun/{clip}"], info[f"prefill_match/{row}"], flush=True)

np.savez_compressed(f"{OUT}/kp.npz", **{k.replace("/", "__"): v for k, v in res.items()})
json.dump(info, open(f"{OUT}/decode_info.json", "w"), indent=1)
print("DONE", len(res))
