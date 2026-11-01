import json, sys, numpy as np
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-posedup-live/posedrift")
from analyze import load, window
rs, real = load()
dup = [r for r in rs if r["model"] == "duplex-498" and r["gen_s"] >= 20]
mono = [r for r in rs if r["model"] == "monoplex-357" and r["gen_s"] >= 20]
aloe = [r for r in real if r["kind"] == "aloe"]
def fmt(sub, w=(0, 30)):
    f = lambda m: np.mean([x for x in (window(r, m, *w) for r in sub) if x is not None])
    return f"n={len(sub):3d} conf={f('conf'):.3f} body={f('body_conf'):.3f} hand={f('hand_conf'):.3f} floor={f('below_floor'):.3f} red={f('red'):.3f} lean={f('lean'):.2f} shear={f('shear'):.3f} hroll={f('head_roll'):.2f} pitch={f('pitch'):.3f}"
print("REAL aloe 0-30", fmt(aloe)); print("REAL aloe 0-120", fmt(aloe, (0, 120)))
for name, S in (("DUP", dup), ("MONO", mono)):
    for k in sorted({(r["topk"], r["cfg"], r["cfg_s"]) for r in S}, key=str):
        sub = [r for r in S if (r["topk"], r["cfg"], r["cfg_s"]) == k]
        print(name, k, fmt(sub), "clients", sorted({str(r["client"]) for r in sub}))
def frac(bins, f): return np.mean([f(b) for b in bins]) if bins else float("nan")
for name, sub in (("real", aloe), ("dup cfg0", [r for r in dup if not r["cfg"]]), ("dup cfg1", [r for r in dup if r["cfg"]])):
    B = [b for r in sub for b in r["bins"]]
    print(f"{name}: bins={len(B)} lean>10 {100*frac(B, lambda b: b['lean']>10):.1f}% shear>0.85 {100*frac(B, lambda b: b['shear']>0.85):.1f}% hroll>15 {100*frac(B, lambda b: b['head_roll']>15):.1f}% body<0.4 {100*frac(B, lambda b: b['body_conf']<0.4):.1f}% calls_with_lean>10: {sum(any(b['lean']>10 for b in r['bins']) for r in sub)}/{len(sub)}")
print("LONG duplex calls (>=60 s): id client cfg topk end_s | body first->last | lean max | hroll first->last | shear max")
for r in sorted([r for r in dup if r["end_s"] >= 60], key=lambda r: (r["cfg"], str(r["topk"]))):
    b = r["bins"]
    print(f"{r['id']} {str(r['client'])[:10]:10s} cfg{r['cfg']} {r['cfg_s']} {r['topk']} {r['end_s']:6.1f} | {b[0]['body_conf']:.3f}->{b[-1]['body_conf']:.3f} | {max(x['lean'] for x in b):5.1f} | {b[0]['head_roll']:5.1f}->{b[-1]['head_roll']:5.1f} | {max(x['shear'] for x in b):.2f}")
