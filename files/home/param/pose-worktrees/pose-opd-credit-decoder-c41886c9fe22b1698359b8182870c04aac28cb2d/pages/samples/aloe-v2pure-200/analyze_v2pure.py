"""aloe-v2pure-200 HAND-MOTION METRICS = ppm/samples/pose-hands-cmp/work/analyze.py's maths, run on this run's clips: its functions decode /
energy_db / code_stats / wmetrics / agg and constants (joints, thresholds, FPS) are taken from analyze.py's own source (ast) and executed
unchanged; only its hard-coded 60 s window becomes each clip's window: the generated part = 25 fps frames [2 x 25, 2 x n) (2.0 s to the clip's
real end), the prefill = [0, 50). Gestures / path are per 45 s exactly as analyze.py scales them (count x 1124 / window frames).
Sources per clip: gt = the real recording (real pose codes + the Mimi decode of the real audio, gt_s1.wav), gen = 'all streams generated',
force = 'text forced' (each with its own generated audio s1.wav). Writes <out>/metrics.json (res / agg as analyze.py) + prints the summary.
   python analyze_v2pure.py <work root> <out dir>      (qwen-s2 venv, env_fast.sh; GPU or CPU)"""
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
got = {nd.name for nd in body if isinstance(nd, ast.FunctionDef)}
assert got == WANT_F, got
NS = {"np": np, "sf": sf, "torch": torch, "gaussian_filter1d": gaussian_filter1d, "find_peaks": find_peaks, "welch": welch,
      "offsets_to_positions": offsets_to_positions, "FPS": 25}
exec(compile(ast.Module(body=body, type_ignores=[]), SRCF, "exec"), NS)
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
NS["dev"] = dev
NS["ptok"] = PoseTokenizer.from_pretrained(NS["POSE_TOK"][0], device=dev, revision=NS["POSE_TOK"][1])
NPF = 25                                                         # prefill tokens (2.0 s); decoded poses are 25 fps (2 per token)
rows = json.load(open(f"{W}/rows/rows.json"))
LAB = {"gt": "Real recording", "gen": "all streams generated", "force": "text forced"}
RUNS = {"gen": "runs-gen", "force": "runs-force"}
for k, lab, d in (("gen_k140", "all streams · CFG3 top-k 140", "runs-gen-k140"), ("force_k140", "text forced · CFG3 top-k 140", "runs-force-k140")):
    if all(os.path.isfile(f"{W}/{d}/{r['row']}/DONE") for r in rows):                   # 10-18 k140 variant (pose cb0 top-k 30, cb1-31 140)
        LAB[k], RUNS[k] = lab, d
RES = {k: {"label": LAB[k], "group": k, "tracks": {}} for k in LAB}
for r in rows:
    tr, n = r["row"], int(r["n"])
    NS.update(NPC=NPF, NTOT=n, G0=2 * NPF, G1=2 * n)
    zg = np.load(f"{W}/runs-gen/{tr}/codes.npz")
    zf = np.load(f"{W}/runs-force/{tr}/codes.npz")
    assert np.array_equal(zg["gt_pose_s1"], zf["gt_pose_s1"]) and np.array_equal(zg["gt_s1"], zf["gt_s1"]), ("real codes differ", tr)
    src = {"gt": (zg["gt_pose_s1"], f"{W}/runs-gen/{tr}/gt_s1.wav")}
    for k, d in RUNS.items():
        z = np.load(f"{W}/{d}/{tr}/codes.npz")
        assert np.array_equal(z["gt_pose_s1"], zg["gt_pose_s1"]) and np.array_equal(z["gt_s1"], zg["gt_s1"]), ("real codes differ", d, tr)
        src[k] = (z["pose_s1"], f"{W}/{d}/{tr}/s1.wav")
    for k, (pc, wav) in src.items():
        pc = pc.astype(np.int64)
        assert pc.shape == (32, n), (k, tr, pc.shape)
        assert np.array_equal(pc[:, :NPF], zg["gt_pose_s1"][:, :NPF].astype(np.int64)), ("prefill pose != real", k, tr)
        rel, conf, img = NS["decode"](pc, r["norm"]["s1"])
        db = NS["energy_db"](wav)
        RES[k]["tracks"][tr] = {"gen": NS["wmetrics"](rel, conf, db, 2 * NPF, 2 * n), "prefill": NS["wmetrics"](rel, conf, db, 0, 2 * NPF),
                                "codes": NS["code_stats"](pc), "window_s": [NPF / 12.5, n / 12.5]}
    g = RES["gt"]["tracks"][tr]["gen"]
    print(tr, f"{n / 12.5:.1f}s", " | ".join(f"{k} wrist {100 * RES[k]['tracks'][tr]['gen']['wrist_speed'] / g['wrist_speed']:.0f}% "
                                            f"gest {RES[k]['tracks'][tr]['gen']['gestures_45s']:.1f} still {RES[k]['tracks'][tr]['gen']['still_frac']:.2f} "
                                            f"r {RES[k]['tracks'][tr]['gen']['r_speed_energy']:.2f}" for k in LAB), flush=True)
print("sources", list(LAB), flush=True)
NS["RES"] = RES
AGG = {k: {"gen": NS["agg"](k, "gen"), "prefill": NS["agg"](k, "prefill")} for k in RES}
json.dump({"res": RES, "agg": AGG, "consts": {"TH_STILL": NS["TH_STILL"], "TH_GEST": NS["TH_GEST"], "TH_SPEAK": NS["TH_SPEAK"], "NPF": NPF,
                                              "window": "generated = 25 fps frames [50, 2 n): 2.0 s to each clip's end", "source": SRCF}},
          open(f"{OUT}/metrics.json", "w"))
for k in AGG:
    g = AGG[k]["gen"]
    print(f"{k:6s} n={g['n']:2d} wrist {g['mean']['wrist_speed']:.4f} = {g['pct_gt']['wrist_speed']:6.1f}% real (CI {g['wrist_speed_pct_ci95'][0]:.0f}-"
          f"{g['wrist_speed_pct_ci95'][1]:.0f}) gest/45s {g['mean']['gestures_45s']:.1f} (real {g['gt_mean']['gestures_45s']:.1f}) still "
          f"{g['mean']['still_frac']:.2f} (real {g['gt_mean']['still_frac']:.2f}) r {g['mean']['r_speed_energy']:.2f} (real "
          f"{g['gt_mean']['r_speed_energy']:.2f}) vrange {g['mean']['hand_height_range']:.2f} (real {g['gt_mean']['hand_height_range']:.2f})", flush=True)
print("done")
