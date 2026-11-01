"""step-0 eval of the FT (weights = dcp-17012) vs the 8,1,1 run's last eval (job 26391, step 16996 = 16 steps earlier): common eval keys."""
import json, sys
ft = [json.loads(l) for l in open(sys.argv[1])]
ref = [json.loads(l) for l in open("/data/home/amartyavarma/ppm/runs/ppm-v1-j26391/metrics.jsonl")]
r = [x for x in ref if "eval_speech/word_ce_norm" in x][-1]
f = [x for x in ft if x["step"] == 0 and "eval_speech/word_ce_norm" in x][0]
ks = sorted(k for k in f if k in r and k.endswith("word_ce_norm"))
for k in ks:
    print(f"{k}: ft step0 {f[k]:.4f} | 811 step {r['step']} {r[k]:.4f} | diff {f[k] - r[k]:+.4f}")
print("eval_ttv3/word_ce_norm step0", round(f.get("eval_ttv3/word_ce_norm", float("nan")), 4), "cb0", round(f.get("eval_ttv3/cb0_ce", float("nan")), 4),
      "light", round(f.get("eval_ttv3/light_ce", float("nan")), 4))
