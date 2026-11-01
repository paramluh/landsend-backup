"""aloe-8b-vs-32b: render the 32B's pose with the 8B SAM page's OWN skeleton renderer (sam_pose pose_infer.render, imported unchanged: same
tokenizer InternalCan/pose-tokenizer-32cb-200k-noshift@1d6ff4f0, bones drawn only when both joints have confidence > 0.25, same colours
(face / body / left hand / right hand), 2 px lines, 25 fps, canvas 480 high x orig aspect, frame-0 state = the cache's global_state0 (= the 8B
page's inputs/{idx}.json), label 'prefill (real)' before 2.0 s then 'generated' = exactly how the 8B page's gen.mp4 was rendered
(pose_infer.main: render(gp, wav, global_state, orig_hw, gen.mp4, prefill_s=P / FPS)).
Per row and mode: <W>/render/<mode>/<row>.mp4 from runs-<mode>-k140/<row>/codes.npz (pose_s1 [32, n], prefill = real) + s1.wav.
   python render_8bstyle.py <work root>      (venvs/sam-pose; CPU)"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

os.environ["CONFIG_PATH"] = "/data/home/amartyavarma/qwen-1ch/infer/sam_pose_scripts/config_local.yaml"
os.environ["CKPT"] = "/data/home/amartyavarma/models/sam-51960"
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, "/data/home/amartyavarma/qwen-1ch/infer/sam_pose/code/infer")
sys.path.append("/data/home/amartyavarma/ref/pose-tokenizer")
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

W = sys.argv[1]
rows = json.load(open(f"{W}/rows/rows.json"))
S8 = "/data/home/amartyavarma/sam-aloe-ft-v2pure/samples"


def one(job):
    import pose_infer as PI
    torch.set_num_threads(2)
    mode, r = job
    d = f"{W}/runs-{mode}-k140/{r['row']}"
    z = np.load(f"{d}/codes.npz")
    pc = z["pose_s1"].astype(np.int64)
    n = int(r["n"])
    assert pc.shape == (32, n), pc.shape
    inp = np.load(f"{S8}/inputs/{r['idx']}.npz")
    assert np.array_equal(pc[:, :25], inp["pose"][:, :25]), "prefill pose != the 8B page's real pose"
    m8 = json.load(open(f"{S8}/inputs/{r['idx']}.json"))
    assert m8["key"] == r["key"] and m8["global_state0"] == r["global_state0"]
    wav, sr = sf.read(f"{d}/s1.wav", dtype="float32")
    assert sr == 24000
    os.makedirs(f"{W}/render/{mode}", exist_ok=True)
    out = f"{W}/render/{mode}/{r['row']}.mp4"
    PI.render(torch.tensor(pc), wav, np.asarray([m8["global_state0"]], np.float32), tuple(m8["orig_hw"]), out + ".tmp.mp4", prefill_s=25 / 12.5)
    os.replace(out + ".tmp.mp4", out)
    return out


if __name__ == "__main__":
    jobs = [(mode, r) for mode in ("gen", "force") for r in rows]
    with ProcessPoolExecutor(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")) // 2) as ex:
        for o in ex.map(one, jobs):
            print("rendered", o, flush=True)
    print("done", len(jobs))
