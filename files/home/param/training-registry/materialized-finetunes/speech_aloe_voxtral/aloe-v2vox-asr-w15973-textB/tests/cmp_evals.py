import json, sys
def ev(p):
    o = {}
    for ln in open(p):
        d = json.loads(ln)
        if "eval_speech/word_ce_norm" in d:
            o[d["step"]] = d
    return o
a = ev("/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973/runs/ppm-aloe-v2vox-asr-w15973-j26232/metrics.jsonl")
b = ev("/data/home/amartyavarma/ppm/aloe-v2vox-asr-w15973-textB/runs/ppm-aloe-v2vox-asr-w15973-textB-j26422/metrics.jsonl")
K = ["eval_speech/word_ce_norm", "eval_asstB_speech/word_ce_norm", "eval_userB_speech/word_ce_norm"]
print("step | asr-w15973 (26232) all / asstB / userB | textB (26422) all / asstB / userB")
for s in sorted(set(a) & set(b)):
    print(s, "|", " / ".join(f"{a[s][k]:.4f}" for k in K), "|", " / ".join(f"{b[s][k]:.4f}" for k in K))
