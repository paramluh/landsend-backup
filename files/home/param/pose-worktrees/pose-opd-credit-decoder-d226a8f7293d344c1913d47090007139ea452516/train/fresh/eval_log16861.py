# appends one line per new 8,1,1 fresh-run eval (job 27030 chain: runs/ppm-v1-j*.metrics with step >= 17012) to fresh/EVALS.md; info only
import json, glob, time, os
F = "/data/home/amartyavarma/ppm/code-811/fresh"; OUT = f"{F}/EVALS_16861.md"
K = ["eval_speech/cb0_ce", "eval_speech/light_ce", "eval_speech/word_ce_norm", "eval_speech_s3/light_ce", "eval_pose/cb0_ce", "eval_pose/light_ce",
     "eval_pose/pose_cb0_ce", "eval_pose/pose_light_ce", "eval_gemma/word_ce_norm", "eval_chat/word_ce_norm"]
if not os.path.exists(OUT):
    open(OUT, "w").write("# 8,1,1 fresh-data run: evals (auto-appended by fresh/eval_log.py; restart from dcp-16861 (job 27100+); 16861 = start eval = PPM end)\n\n| step | time | " + " | ".join(k[5:] for k in K) + " |\n|" + "---|" * (len(K) + 2) + "\n")
seen = set()
for l in open(OUT):
    if l.startswith("| 1"): seen.add(int(l.split("|")[1]))
while True:
    for p in sorted(glob.glob("/data/home/amartyavarma/ppm/runs/ppm-v1-j27[1-9]*/metrics.jsonl")):
        for l in open(p):
            try: r = json.loads(l)
            except Exception: continue
            if "eval_speech/cb0_ce" in r and r["step"] >= 16861 and r["step"] not in seen:
                seen.add(r["step"])
                open(OUT, "a").write(f"| {r['step']} | {r.get('time_pt', '')} | " + " | ".join(f"{r[k]:.4f}" if k in r else "" for k in K) + " |\n")
    if os.path.exists(f"{F}/EVAL_LOG_STOP") or any(s >= 17352 for s in seen): break
    time.sleep(120)
