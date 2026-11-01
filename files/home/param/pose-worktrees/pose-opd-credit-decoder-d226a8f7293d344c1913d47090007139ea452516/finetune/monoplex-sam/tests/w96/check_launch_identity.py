"""w96 LAUNCH-TIME IDENTITY CHECK (CPU, seconds): a 1-step world-96 run of the -sam code on the ORIGINAL stores (env ALOE_STOP_AT_STEP=1,
ALOE_LOSSCHECK_DUMMY=1, ALOE_RESUME=never, own log / save / export dirs) vs job 24272 (world 64): its step-0 evals, step-0 loss check and
step-1 line. Tolerance classes (REPORT.md; noise floor from the small-scale test, tests/w96/out/compare.json):
  EXACT   counts, LRs, decoder LRs, tokens_step totals, rows digest, prompt fractions, loss-check rank-0 numbers (log, 5 decimals)
  EVAL    step-0 eval + eval_ref23051rule (every key but time_s): rel <= 1e-6 (only the float64 order of the per-rank sums differs)
  FWD     step-1 forward statistics (word CE / top-1 / pad / nw / text CE, cb0, pose cb0, the weighted / unweighted losses): rel <= 1e-5
          (same rows on the same rank indices, unchanged weights; only the fp32 normaliser scale WORLD / tot differs)
  DEC     decoder statistics (light CE, pose light CE and their losses): rel <= TOL_DEC (nanostep updates see world-scaled bf16 gradients)
  GN      per-group gradient norms: rel <= TOL_GN (bf16 backward noise; FA3 backward is not deterministic even at the same world)
usage: python check_launch_identity.py <new run dir (metrics.jsonl + prompt_stats_*.json)> <new log> [ref run dir] [ref log]
exit 0 = PASS (GO may continue), 1 = FAIL (GO must stop and resume MoT)."""
import json
import math
import re
import sys

REF_DIR = "/data/home/amartyavarma/ppm/aloe-pose-mono-8020/runs/ppm-aloe-pose-mono-8020-j24229"
REF_LOG = "/data/home/amartyavarma/ppm/aloe-pose-mono-8020/logs/posemono8020-24272.log"
import os  # noqa: E402
# CHECK_MODE norm64 (default: the run uses ALOE_NORM_WORLD=64 -> world-64 bf16 numbers, only fp32 reduction order + FA3 noise differ) or
# plain (ALOE_NORM_WORLD unset: gradients carry the x1.5 bf16 rounding; small-scale layout effect: gn <= 1.5e-3, light CE <= 4e-7)
MODE = os.environ.get("CHECK_MODE", "norm64")
assert MODE in ("norm64", "plain"), MODE
TOL_EVAL, TOL_FWD = 1e-6, 1e-5
TOL_DEC, TOL_GN = (1e-4, 5e-3) if MODE == "norm64" else (1e-3, 2e-2)
TOL_DEC, TOL_GN = float(os.environ.get("CHECK_TOL_DEC", TOL_DEC)), float(os.environ.get("CHECK_TOL_GN", TOL_GN))
new_dir, new_log = sys.argv[1], sys.argv[2]
ref_dir = sys.argv[3] if len(sys.argv) > 3 else REF_DIR
ref_log = sys.argv[4] if len(sys.argv) > 4 else REF_LOG


def rows(d):
    out = {}
    for line in open(f"{d}/metrics.jsonl"):
        r = json.loads(line)
        kind = "eval" if any(k.startswith("eval_speech/") for k in r) else "ref" if any(k.startswith("eval_ref23051rule") for k in r) else "train"
        out.setdefault((r["step"], kind), r)                      # first occurrence (a resumed run appends; the check run has one)
    return out


N, R = rows(new_dir), rows(ref_dir)
fails, notes = [], []


def cmp(key_sel, a, b, tol, cls):
    for k in sorted(k for k in b if key_sel(k)):
        if k not in a:
            fails.append(f"{cls} {k}: missing in the new run"); continue
        x, y = float(a[k]), float(b[k])
        r = abs(x - y) / max(abs(y), 1e-30) if tol > 0 else float(x != y)
        ok = (r <= tol) if tol > 0 else (x == y)
        notes.append((cls, k, y, x, r, ok))
        if not ok or not math.isfinite(x):
            fails.append(f"{cls} {k}: ref {y!r} new {x!r} rel {r:.2e} > {tol:g}")


for kind in ("eval", "ref"):
    if (0, kind) not in N:
        fails.append(f"step-0 {kind} missing in the new run")
    else:
        cmp(lambda k: "/" in k and not k.endswith("time_s"), N[(0, kind)], R[(0, kind)], TOL_EVAL, f"EVAL[{kind}]")
if (1, "train") not in N:
    fails.append("step 1 missing in the new run")
else:
    a, b = N[(1, "train")], R[(1, "train")]
    exact = lambda k: k in ("train/text_targets", "train/cb0_labels", "train/light_frames", "train/pose_cb0_labels", "train/pose_frames", "train/rounds",  # noqa: E731
                            "train_speech/n_word", "train/dummies") or k.startswith(("train/lr_", "train/dec_lr", "train/dec_p_lr"))
    dec = lambda k: "light" in k                                                                                                   # noqa: E731
    gn = lambda k: k.startswith("train/gn_")                                                                                       # noqa: E731
    skip = lambda k: k in ("step", "time_pt") or k.startswith(("train/step_time", "train/data_wait", "train/micro_s", "train/peak_gib", "train/passab"))  # noqa: E731
    cmp(lambda k: exact(k), a, b, 0, "EXACT")
    cmp(lambda k: gn(k), a, b, TOL_GN, "GN")
    cmp(lambda k: dec(k) and not exact(k), a, b, TOL_DEC, "DEC")
    cmp(lambda k: not (exact(k) or gn(k) or dec(k) or skip(k)), a, b, TOL_FWD, "FWD")
for nm in ("train", "eval"):
    pa, pb = json.load(open(f"{new_dir}/prompt_stats_{nm}.json")), json.load(open(f"{ref_dir}/prompt_stats_{nm}.json"))
    cmp(lambda k: k.startswith("prompt_") or k == "n", pa, pb, 0, f"EXACT[prompt {nm}]")
# log lines: rows digest, tokens_step totals, loss check (rank-0 numbers), world / dummies
pat = {"rowsdg": r"\[data\] rows digest (\w+)", "tokens": r"\[pack\] tokens_step: step 0 totals over \d+ ranks x 1 rounds: (.*); per-micro",
       "lc_vals": r"LOSS CHECK step 0: .*\(rank 0: (.*)\)", "world": r"\] aloe-ft \S+: world (\d+)", "dummies": r"; (\d+) dummy micros;"}


def grab(path):
    txt = open(path, errors="replace").read()
    return {k: (m.group(1) if (m := re.search(p, txt)) else None) for k, p in pat.items()}


gl, gr = grab(new_log), grab(ref_log)
for k in ("rowsdg", "tokens", "lc_vals"):
    ok = gl[k] is not None and gl[k] == gr[k]
    notes.append(("LOG", k, gr[k], gl[k], 0.0, ok))
    if not ok:
        fails.append(f"LOG {k}: ref {gr[k]!r} new {gl[k]!r}")
if gl["world"] != "96" or gl["dummies"] != "11447":
    fails.append(f"LOG world / dummies: {gl['world']} / {gl['dummies']} (want 96 / 11447)")
for cls, k, y, x, r, ok in notes:
    if not ok or cls in ("GN", "DEC", "LOG") or k.endswith(("word_ce_norm", "train/loss")):
        print(f"{'ok  ' if ok else 'FAIL'} {cls:16s} {k:44s} ref {y!s:>22.22} new {x!s:>22.22} rel {r:.2e}")
print(f"\nmode {MODE} (tolerances EVAL {TOL_EVAL:g} FWD {TOL_FWD:g} DEC {TOL_DEC:g} GN {TOL_GN:g}); {len(notes)} values compared; {len(fails)} failures")
print("PASS: world 96 reproduces job 24272's step-0 evals, loss check and step 1" if not fails else "FAIL:\n  " + "\n  ".join(fails[:40]))
sys.exit(0 if not fails else 1)
