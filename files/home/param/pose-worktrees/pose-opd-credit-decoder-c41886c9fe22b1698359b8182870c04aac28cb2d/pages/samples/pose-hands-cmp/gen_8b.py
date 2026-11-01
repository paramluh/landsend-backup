"""pose-hands-cmp: the 8B SAM pose model the VIDEO TAB uses (InternalCan/sam-pose-aloe-v2pure-step636 = local export
/data/home/amartyavarma/sam-aloe-ft-v2pure/export-step-636/model.safetensors, video/code/t2v/settings.yaml llm.weights) on the SAME held-out
Aloe tracks as the 32B (extract_gt.py: C-iii-sK).
Per track: upfront text = that speaker's real words of the whole 60 s window (prefill part included, as SAM's pose_infer default
"the clip's own transcript, all of it"); prefill = the first 188 frames (15.04 s) of the speaker's REAL Mimi + pose tokens; generate to
frame 750 (45 s). END is masked (audio cb0 END logit -inf), so every track gets the full 45 s (= the demo's end_mode mask_continue: END
regenerated with END -inf); everything else is SAM's own fast generate (pose_infer_fast.generate / pose_infer_fast_cfg.generate).
Variants (--variant, comma list):
  nocfg : temp 1.0 audio + pose (all 32 codebooks), audio top-k 30, pose top-k 30, NO CFG, no rep-pen, no min-p
  cfg3  : same + pose CFG 3.0 on all 32 pose codebooks, uncond = digital-silence Mimi (the SAM line's / video tab's CFG; top-k 30 here)
  vtab  : the video tab's pose recipe (t2v/settings.yaml): pose CFG 3.0 all 32 codebooks + pose top-k 250 (audio top-k 30)
bf16. -> <out root>/<variant>/C-iii-sK/: codes.npz (audio, pose [32, 750], prefill_frames), gen.wav (24 kHz Mimi decode), meta.json, DONE
   python gen_8b.py <gt8b dir> <out root> <tracks comma list> [--variant nocfg,cfg3]      (venvs/sam-pose)"""
import argparse
import json
import os
import sys
import time

os.environ["CONFIG_PATH"] = "/data/home/amartyavarma/qwen-1ch/infer/sam_pose_scripts/config_local.yaml"
os.environ["CKPT"] = "/data/home/amartyavarma/models/sam-51960"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
ROOT = "/data/home/amartyavarma/qwen-1ch/infer/sam_pose/code"
sys.path.insert(0, f"{ROOT}/infer")
sys.path.append("/data/home/amartyavarma/ref/pose-tokenizer")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import pose_infer as PI  # noqa: E402
import pose_infer_fast_cfg as PC  # noqa: E402
from sam_lib import load_tokenizer, mimi_decode  # noqa: E402
import soundfile as sf  # noqa: E402

CKPT = "/data/home/amartyavarma/sam-aloe-ft-v2pure/export-step-636"
END = PI.END
ap = argparse.ArgumentParser()
ap.add_argument("gt")
ap.add_argument("out")
ap.add_argument("tracks")
ap.add_argument("--variant", default="nocfg,cfg3")
a = ap.parse_args()
VAR = {"nocfg": dict(pose_cfg=0.0), "cfg3": dict(pose_cfg=3.0, pose_cfg_scope="all", pose_cfg_uncond="silence"),
       "vtab": dict(pose_cfg=3.0, pose_cfg_scope="all", pose_cfg_uncond="silence")}   # vtab = the video tab's recipe: + pose top-k 250
PTOPK = {"nocfg": 30, "cfg3": 30, "vtab": 250}


class NoEnd(nn.Module):
    """audio cb0 head with the END logit set to -inf (END can never be sampled; top-k picks among the 2048 codes)."""

    def __init__(self, lin):
        super().__init__()
        self.lin = lin

    def forward(self, x):
        y = self.lin(x)
        y[..., END] = float("-inf")
        return y


t0 = time.time()
model = PI.load_model(CKPT, "cuda")
assert model.lm_head.out_features == END + 1
model.lm_head = NoEnd(model.lm_head)
tok = load_tokenizer()
print(f"[8b] loaded {CKPT} in {time.time() - t0:.0f}s on {torch.cuda.get_device_name()}", flush=True)
for tr in a.tracks.split(","):
    g = json.load(open(f"{a.gt}/{tr}.json"))
    z = np.load(f"{a.gt}/{tr}.npz")
    audio, pose = torch.tensor(z["audio"].astype(np.int64)), torch.tensor(z["pose"].astype(np.int64))
    P, N = int(g["Npc"]), int(g["Ntot"])
    ids = tok(g["text"], add_special_tokens=False)["input_ids"]
    seed = 1000 * int(g["conv"]) + (0 if g["spk"] == "s1" else 1)
    for v in a.variant.split(","):
        out = f"{a.out}/{v}/{tr}"
        if os.path.isfile(f"{out}/DONE"):
            print(f"[8b] {v} {tr} exists", flush=True)
            continue
        os.makedirs(out, exist_ok=True)
        t1 = time.time()
        ga, gp, ended = PC.generate(model, ids, audio, pose, P, N, temp=1.0, topk=30, seed=seed, pose_temp=1.0, pose_topk=PTOPK[v],
                                    pose_rep=1.0, rep_window=0, **VAR[v])
        gen_s = time.time() - t1
        assert not ended and ga.shape == (32, N) and gp.shape == (32, N), (ended, ga.shape, gp.shape)
        assert torch.equal(ga[:, :P], audio[:, :P]) and torch.equal(gp[:, :P], pose[:, :P])
        assert int(ga.max()) < 2048 and int(gp.max()) < 2048
        wav = mimi_decode(ga[None], device="cuda")[0]
        sf.write(f"{out}/gen.wav", wav, 24000)
        np.savez_compressed(f"{out}/codes.npz", audio=ga.numpy().astype(np.int16), pose=gp.numpy().astype(np.int16), prefill_frames=P)
        meta = {"track": tr, "variant": v, "ckpt": CKPT, "hf": "InternalCan/sam-pose-aloe-v2pure-step636@cbd9c1e3", "seed": seed,
                "prefill_frames": P, "total_frames": N, "text_tokens": len(ids), "text": g["text"], "end_masked": True,
                "sampling": {"temp": 1.0, "topk": 30, "pose_temp": 1.0, "pose_topk": PTOPK[v], "pose_rep": 1.0, "min_p": None, **VAR[v]},
                "pose_gen_cb0_unique": int(len(np.unique(gp[0, P:].numpy()))), "gen_s": round(gen_s, 1), "dtype": "bf16",
                "s_per_frame": round(PC.LAST.get("loop_s", 0) / max(PC.LAST.get("frames", 1), 1), 4),
                "made": time.strftime("%Y-%m-%d %H:%M:%S %Z")}
        json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)
        open(f"{out}/DONE", "w").write("ok\n")
        print(f"[8b] {v} {tr}: {gen_s:.0f}s, {meta['s_per_frame']} s/frame, cb0 uniq {meta['pose_gen_cb0_unique']}, text {len(ids)} tok", flush=True)
print(f"[8b] all done in {time.time() - t0:.0f}s", flush=True)
