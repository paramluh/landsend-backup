# pose5 (10-27): = code-811/fresh/append/eval_log_append.py with ONLY the output dir changed (pose5/, so nothing is written into code-811):
# appends one line per new eval of the 8,1,1 run resumed from dcp-17850 (npl5 + s2p; jobs >= MIN_JOB) to pose5/EVALS_pose5.md; same columns as
# fresh/EVALS_append3.md (17850 = the resume eval); stops at step END or on pose5/EVAL_LOG_STOP.
# usage: MIN_JOB=<first pose5 job id> END=<PPM_MAX_STEPS> [MIN_STEP=17850] python3 eval_log_pose5.py   (info only)
import json, glob, time, os
F = "/data/home/amartyavarma/ppm/pose5"
OUT = f"{F}/{os.environ.get('EVAL_OUT', 'EVALS_pose5.md')}"; MIN_STEP = int(os.environ.get("MIN_STEP", "17850"))
MIN_JOB, END = int(os.environ["MIN_JOB"]), int(os.environ["END"])
K = ["eval_speech/cb0_ce", "eval_speech/light_ce", "eval_speech/word_ce_norm", "eval_speech_s3/light_ce", "eval_pose/cb0_ce", "eval_pose/light_ce",
     "eval_pose/pose_cb0_ce", "eval_pose/pose_light_ce", "eval_gemma/word_ce_norm", "eval_chat/word_ce_norm"]
if not os.path.exists(OUT):
    open(OUT, "w").write(f"# 8,1,1 run, pose5 leg (resume from dcp-{MIN_STEP} + npl5 + s2p, jobs >= {MIN_JOB}): evals (auto-appended by pose5/kit/eval_log_pose5.py; {MIN_STEP} = the resume eval; compare fresh/EVALS_append3.md = 27653)\n\n| step | time | " + " | ".join(k[5:] for k in K) + " |\n|" + "---|" * (len(K) + 2) + "\n")
seen = set()
for l in open(OUT):
    if l.startswith("| 1"): seen.add(int(l.split("|")[1]))
while True:
    for p in [q for q in sorted(glob.glob("/data/home/amartyavarma/ppm/runs/ppm-v1-j*/metrics.jsonl")) if int(q.split("-j")[1].split("/")[0]) >= MIN_JOB]:
        for l in open(p):
            try: r = json.loads(l)
            except Exception: continue
            if "eval_speech/cb0_ce" in r and r["step"] >= MIN_STEP and r["step"] not in seen:
                seen.add(r["step"])
                open(OUT, "a").write(f"| {r['step']} | {r.get('time_pt', '')} | " + " | ".join(f"{r[k]:.4f}" if k in r else "" for k in K) + " |\n")
    if os.path.exists(f"{F}/EVAL_LOG_STOP") or any(s >= END for s in seen): break
    time.sleep(120)
