# 10-22 fresh pose APPEND run: appends one line per new eval of the 8,1,1 run resumed from dcp-17352 onto the appended plans (jobs >= MIN_JOB) to
# fresh/EVALS_append.md; same columns as EVALS_16861_v4.md (17352 = the resume eval = the end of 27114); stops at step END or on fresh/EVAL_LOG_STOP.
# usage: MIN_JOB=<first append job id> END=<PPM_MAX_STEPS> python3 eval_log_append.py   (info only)
import json, glob, time, os
F = "/data/home/amartyavarma/ppm/code-811/fresh"
# 10-23 (delivery 3): EVAL_OUT / MIN_STEP select the leg (default = the npl2 leg: EVALS_append.md from 17352; npl3 leg: EVALS_append3.md from 17641)
OUT = f"{F}/{os.environ.get('EVAL_OUT', 'EVALS_append.md')}"; MIN_STEP = int(os.environ.get("MIN_STEP", "17352"))
MIN_JOB, END = int(os.environ["MIN_JOB"]), int(os.environ["END"])
K = ["eval_speech/cb0_ce", "eval_speech/light_ce", "eval_speech/word_ce_norm", "eval_speech_s3/light_ce", "eval_pose/cb0_ce", "eval_pose/light_ce",
     "eval_pose/pose_cb0_ce", "eval_pose/pose_light_ce", "eval_gemma/word_ce_norm", "eval_chat/word_ce_norm"]
if not os.path.exists(OUT):
    open(OUT, "w").write(f"# 8,1,1 fresh-data run, APPENDED pose plans (resume from dcp-{MIN_STEP}, jobs >= {MIN_JOB}): evals (auto-appended by fresh/append/eval_log_append.py; {MIN_STEP} = the resume eval)\n\n| step | time | " + " | ".join(k[5:] for k in K) + " |\n|" + "---|" * (len(K) + 2) + "\n")
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
