"""SAM 8B pose from OUR generated speech + text (10-19): the 8B SAM Aloe pose fine-tune the pose-demo website / video tab runs
(InternalCan/sam-pose-aloe-v2pure-step636 @ cbd9c1e3 = local export sam-aloe-ft-v2pure/export-step-636; = gen_8b.py's model), SAM's own fast
CFG generate with audio teacher-forcing (pose_infer_fast_cfg.generate audio_tf=True): upfront text = our full-AR video's generated words
(the model's own lane, 0-60 s, prefill words included), audio = our generated Mimi codes [32, 750] forced at every frame, pose prefill = the
package's real prefill_pose (frames 0..24), pose generated to frame 750. Sampling (the SAM line's standing rule): pose CFG 3 on all 32 pose
codebooks, uncond = digital-silence Mimi; no CFG on audio; temp 1, top-k 30 (audio + pose); no rep-pen; END masked; bf16.
-> runs-sam8b/<row>/codes.npz (pose [32, 750], audio = ours), meta.json, DONE
   python gen_sam8b.py S-05,S-13,S-15,S-20      (venvs/sam-pose)"""
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
from sam_lib import load_tokenizer  # noqa: E402

CKPT = "/data/home/amartyavarma/sam-aloe-ft-v2pure/export-step-636"
W = "/data/home/amartyavarma/ppm/samples/aloe-posemono-poseonly"
END = PI.END
SEEDS = {"S-05": 5, "S-13": 13, "S-15": 15, "S-20": 20}


class NoEnd(nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.lin = lin

    def forward(self, x):
        y = self.lin(x)
        y[..., END] = float("-inf")
        return y


t0 = time.time()
model = PI.load_model(CKPT, "cuda")
model.lm_head = NoEnd(model.lm_head)
tok = load_tokenizer()
for row in sys.argv[1].split(","):
    out = f"{W}/runs-sam8b/{row}"
    os.makedirs(out, exist_ok=True)
    z = np.load(f"{W}/runs-studio/{row}/codes.npz")
    wj = json.load(open(f"{W}/runs-studio/{row}/words.json"))
    text = " ".join(w["text"] for w in wj["s1"])
    audio = torch.tensor(z["s1"].astype(np.int64))
    pose = torch.zeros(32, 750, dtype=torch.long)
    pose[:, :25] = torch.tensor(z["prefill_pose"].astype(np.int64))
    ids = tok(text, add_special_tokens=False)["input_ids"]
    t1 = time.time()
    ga, gp, ended = PC.generate(model, ids, audio, pose, 25, 750, temp=1.0, topk=30, seed=SEEDS[row], pose_temp=1.0, pose_topk=30,
                                pose_rep=1.0, rep_window=0, pose_cfg=3.0, pose_cfg_scope="all", pose_cfg_uncond="silence", audio_tf=True)
    assert ga.shape == (32, 750) and gp.shape == (32, 750) and torch.equal(ga.cpu(), audio) and torch.equal(gp[:, :25].cpu(), pose[:, :25])
    np.savez_compressed(f"{out}/codes.npz", pose=gp.cpu().numpy().astype(np.int16), audio=ga.cpu().numpy().astype(np.int16))
    json.dump({"row": row, "ckpt": CKPT, "hf": "InternalCan/sam-pose-aloe-v2pure-step636@cbd9c1e3", "seed": SEEDS[row], "text": text,
               "text_tokens": len(ids), "audio": "our full-AR generated Mimi codes, teacher-forced (audio_tf)", "prefill_pose_frames": 25,
               "sampling": {"temp": 1.0, "topk": 30, "pose_temp": 1.0, "pose_topk": 30, "pose_cfg": 3.0, "scope": "all", "uncond": "silence",
                            "audio_cfg": None, "rep_pen": None}, "gen_s": round(time.time() - t1, 1),
               "pose_gen_cb0_unique": int(len(np.unique(gp[0, 25:].cpu().numpy())))}, open(f"{out}/meta.json", "w"), indent=1)
    open(f"{out}/DONE", "w").write("ok\n")
    print(f"[sam8b] {row} done in {time.time() - t1:.0f} s, {len(ids)} text tokens", flush=True)
print(f"[sam8b] all done {time.time() - t0:.0f} s", flush=True)
