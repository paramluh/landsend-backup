"""aloe-8b-vs-32b HAND-MOTION METRICS, the same maths for every column = ppm/samples/pose-hands-cmp/work/analyze.py's functions decode /
energy_db / code_stats / wmetrics / agg + constants, executed unchanged (as aloe-v2pure-200/work/analyze_v2pure.py does). Window per tile = its
generated part: 25 fps frames [50, 2 x its length) (2.0 s to the tile's end: the real clip's end for the real / 32B tiles, END or the 25 s cap
for the 8B tile); gestures scaled to 45 s; one shared tokenizer decode + each clip's own frame-0 normalisation (v2pure cache).
Sources per clip: gt = real pose (cache = the 8B page's inputs) + Mimi decode of the real codes (32B run's gt_s1.wav), b8 = the 8B page's
CFG 3 / pose top-k 30 column (out/1014-215517/{idx}-pk30: codes.npz pose + gen.wav), force / gen = the 32B k140 runs (codes.npz pose_s1 + s1.wav).
   python analyze.py <work root> <out dir>      (venvs/qwen-s2; CPU or GPU)"""
import ast
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
os.makedirs(OUT, exist_ok=True)
SRCF = "/data/home/amartyavarma/ppm/samples/pose-hands-cmp/work/analyze.py"
O8 = "/data/home/amartyavarma/sam-aloe-ft-v2pure/samples/out/1014-215517"
S8 = "/data/home/amartyavarma/sam-aloe-ft-v2pure/samples"
tree = ast.parse(open(SRCF).read())
WANT_F = {"decode", "energy_db", "code_stats", "wmetrics", "agg"}
WANT_C = {"POSE_TOK", "WR", "EL", "FING", "HANDS", "TH_STILL", "TH_GEST", "TH_SPEAK"}
body = []
for nd in tree.body:
    if isinstance(nd, ast.FunctionDef) and nd.name in WANT_F:
        body.append(nd)
    elif isinstance(nd, ast.Assign):
        names = {t.id for t in nd.targets if isinstance(t, ast.Name)} | {e.id for t in nd.targets if isinstance(t, ast.Tuple) for e in t.elts}
        if names & WANT_C:
            body.append(nd)
assert {nd.name for nd in body if isinstance(nd, ast.FunctionDef)} == WANT_F
NS = {"np": np, "sf": sf, "torch": torch, "gaussian_filter1d": gaussian_filter1d, "find_peaks": find_peaks, "welch": welch,
      "offsets_to_positions": offsets_to_positions, "FPS": 25}
exec(compile(ast.Module(body=body, type_ignores=[]), SRCF, "exec"), NS)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
NS["dev"] = dev
NS["ptok"] = PoseTokenizer.from_pretrained(NS["POSE_TOK"][0], device=dev, revision=NS["POSE_TOK"][1])
NPF = 25
rows = json.load(open(f"{W}/rows/rows.json"))
LAB = {"gt": "Real recording", "b8": "8B · CFG 3 · pose top-k 30 · text upfront", "force": "32B · text forced", "gen": "32B · all streams"}
RES = {k: {"label": LAB[k], "group": k, "tracks": {}} for k in LAB}
for r in rows:
    tr, n, idx = r["row"], int(r["n"]), r["idx"]
    real = np.load(f"{S8}/inputs/{idx}.npz")["pose"].astype(np.int64)
    m8 = json.load(open(f"{O8}/{idx}-pk30/meta.json"))
    assert m8["clip"] == r["key"] and m8["pose_cfg"] == 3.0 and m8["pose_topk"] == 30 and m8["prefill_frames"] == NPF, m8
    src = {"gt": (real, f"{W}/runs-gen-k140/{tr}/gt_s1.wav"), "b8": (np.load(f"{O8}/{idx}-pk30/codes.npz")["pose"], f"{O8}/{idx}-pk30/gen.wav")}
    for k in ("force", "gen"):
        z = np.load(f"{W}/runs-{k}-k140/{tr}/codes.npz")
        assert np.array_equal(z["gt_pose_s1"].astype(np.int64), real), ("real codes differ", k, tr)
        src[k] = (z["pose_s1"], f"{W}/runs-{k}-k140/{tr}/s1.wav")
    for k, (pc, wav) in src.items():
        pc = pc.astype(np.int64)
        L = pc.shape[1]
        assert pc.shape[0] == 32 and (L == n or k == "b8"), (k, tr, pc.shape)
        assert np.array_equal(pc[:, :NPF], real[:, :NPF]), ("prefill pose != real", k, tr)
        NS.update(NPC=NPF, NTOT=L, G0=2 * NPF, G1=2 * L)
        rel, conf, img = NS["decode"](pc, r["norm"]["s1"])
        db = NS["energy_db"](wav)
        RES[k]["tracks"][tr] = {"gen": NS["wmetrics"](rel, conf, db, 2 * NPF, 2 * L), "prefill": NS["wmetrics"](rel, conf, db, 0, 2 * NPF),
                                "codes": NS["code_stats"](pc), "window_s": [NPF / 12.5, L / 12.5]}
    g = RES["gt"]["tracks"][tr]["gen"]
    print(tr, idx, " | ".join(f"{k} wrist {100 * RES[k]['tracks'][tr]['gen']['wrist_speed'] / g['wrist_speed']:.0f}% "
                              f"gest {RES[k]['tracks'][tr]['gen']['gestures_45s']:.1f} still {RES[k]['tracks'][tr]['gen']['still_frac']:.2f}"
                              for k in LAB), flush=True)
NS["RES"] = RES
AGG = {k: {"gen": NS["agg"](k, "gen"), "prefill": NS["agg"](k, "prefill")} for k in RES}
json.dump({"res": RES, "agg": AGG, "consts": {"TH_STILL": NS["TH_STILL"], "TH_GEST": NS["TH_GEST"], "TH_SPEAK": NS["TH_SPEAK"], "NPF": NPF,
                                              "window": "generated = 25 fps frames [50, 2 L): 2.0 s to each tile's end", "source": SRCF}},
          open(f"{OUT}/metrics.json", "w"))
for k in AGG:
    g = AGG[k]["gen"]
    print(f"{k:6s} n={g['n']} wrist {g['mean']['wrist_speed']:.4f} = {g['pct_gt']['wrist_speed']:6.1f}% real (CI {g['wrist_speed_pct_ci95'][0]:.0f}-"
          f"{g['wrist_speed_pct_ci95'][1]:.0f}) gest/45s {g['mean']['gestures_45s']:.1f} (real {g['gt_mean']['gestures_45s']:.1f}) still "
          f"{100 * g['mean']['still_frac']:.0f}% (real {100 * g['gt_mean']['still_frac']:.0f}%) r {g['mean']['r_speed_energy']:.2f}", flush=True)
print("done")
