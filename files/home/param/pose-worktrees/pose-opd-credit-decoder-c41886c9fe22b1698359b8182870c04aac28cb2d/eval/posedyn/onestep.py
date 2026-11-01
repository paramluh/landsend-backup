"""10-27 user ("why does temp 1 no cfg not work" -> "yes" to the test): is the monoplex pose model's low motion COMPOUNDING (free rollout
drifts to stillness) or BLURRED one-step predictions? Same engine, weights and code path as the playground (genweb/worker.py Worker.run,
pose-only mode = the actor's real speech codes + timed words forced every frame), base export-357, CFG off, T 1.0:
  FREE      prefill 1.92 s (real pose for 26 frames), then pose sampled and fed back (= a playground pose-only run)
  ONESTEP   real pose forced on EVERY frame; at each frame the engine still draws a pose code from the model (one step ahead of REAL
            history); the draw is captured just before the force replaces it (patched FastPPM.step, otherwise unchanged)
  REAL      the actor's own pose tokens
for every actor opening on disk (12 s each), 2 seeds, top-k at the playground defaults (cb0 30, light 151) and top-k OFF (0).
Metrics on frames 26..T (the same window for all three): gesturemetric strokes/min + hands-moving %, plus cb0 change rate.
-> out/results.json"""
import inspect
import json
import os
import sys
import textwrap

G = "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb"
sys.path.insert(0, G)
import worker as W  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

FP = W.FP
src = textwrap.dedent(inspect.getsource(FP.FastPPM.step))
import re
mm = list(re.finditer(r'^( *)if force is not None and force\.get\("pose"\) is not None:\n', src, re.M))
assert len(mm) == 1, ("patch anchor", len(mm))
ind = mm[0].group(1)
src = src[:mm[0].start()] + f"{ind}self._pdraw = self.pkp[dp].clone()   # posedyn: the model's own draw, before any force\n" + src[mm[0].start():]
ns = {}
exec(compile(src, "fast_ppm_step_patched", "exec"), FP.__dict__, ns)
FP.FastPPM.step = ns["step"]

sys.path.insert(0, "/data/home/amartyavarma/ppm/agents1026/gesturemetric")
import gm  # noqa: E402

OUT = "/data/home/amartyavarma/ppm/posedyn/out"
os.makedirs(OUT, exist_ok=True)
W.WID = "posedyn"
wk = W.Worker()
persons = sorted(set(p for r in ("openings-sam", "openings") for p in os.listdir(f"{G}/{r}")
                     if os.path.isfile(f"{G}/{r}/{p}/pose_tokens.safetensors") and os.path.isfile(f"{G}/{r}/{p}/words.json")))
print("persons", persons, flush=True)


def metrics(codes):
    c = np.asarray(codes, np.int64)
    k55, swp = gm.k55_of(c, "01_ab63d745")
    ws = gm.wrist_speed(k55, swp)
    st, _ = gm.strokes(ws)
    mins = len(ws) / 25 / 60
    return {"strokes": int(len(st)), "per_min": round(len(st) / mins, 2), "moving": round(float((ws.max(1) >= gm.PEAK_THR).mean()), 4),
            "cb0_change": round(float((c[1:, 0] != c[:-1, 0]).mean()), 4), "frames": int(len(c))}


res = json.load(open(f"{OUT}/results.json")) if os.path.isfile(f"{OUT}/results.json") else []   # resume: keep finished runs
DONE = {(r["person"], r["topk"], r["seed"], r["mode"]) for r in res}
orig_step = wk.eng.step
for person in persons:
    for topk in ("default", "off"):
        for seed in (101, 202):
            prm = {"pose_only": 1} if topk == "default" else {"pose_only": 1, "pose_topk": 0, "pose_light_topk": 0}
            for mode, pfs in (("free", 1.92), ("onestep", 11.68)):
                if (person, topk, seed, mode) in DONE:
                    continue
                draws = {}

                def step(f, **kw):
                    out = orig_step(f, **kw)
                    if getattr(wk.eng, "_pdraw", None) is not None:
                        draws[f] = (wk.eng._pdraw[0].cpu().numpy() - W.OFF).astype(np.int64)
                    return out
                wk.eng.step = step
                rd = f"{OUT}/runs/{person}_{topk}_{seed}_{mode}"
                os.makedirs(rd, exist_ok=True)
                req = {"run_id": os.path.basename(rd), "seed": seed, "clip_s": 12.0, "params": prm, "prompt": {"voice": person},
                       "prefill": {"person": person, "s": pfs, "src": "start", "pose": True, "audio": True, "words": True}}
                wk.run(rd, req)
                wk.eng.step = orig_step
                z = np.load(f"{rd}/tokens.npz")
                gen = z["pose"].astype(np.int64)                       # [n, 32] what the run kept (free: sampled; onestep: real)
                n = len(gen)
                lo = 26
                if mode == "free":
                    r = {"mode": "free", "m": metrics(gen[lo:n])}
                else:
                    seq = np.stack([draws[f] for f in sorted(draws) if 0 <= f - W.CD < n])[:n]   # draw of frame g = f - CD
                    r = {"mode": "onestep", "m": metrics(seq[lo:n]), "real": metrics(gen[lo:n]),
                         "onestep_eq_real_cb0": round(float((seq[lo:n, 0] == gen[lo:n, 0]).mean()), 4)}
                r.update(person=person, topk=topk, seed=seed, n=n)
                res.append(r)
                print(json.dumps(r), flush=True)
                json.dump(res, open(f"{OUT}/results.json", "w"), indent=1)
