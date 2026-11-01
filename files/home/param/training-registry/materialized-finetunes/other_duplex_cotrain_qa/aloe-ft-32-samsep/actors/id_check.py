"""Which manifest field is the voice? For the top-30 actor_ids' candidate segments (cached ECAPA embeddings), compare within-group cosine
when grouping by speaker_<lane>_actor_id vs speaker_<lane>_account_uid; and count distinct account_uids per actor_id."""
import csv, glob, json, collections
import numpy as np
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors"
csv.field_size_limit(1 << 30)
man = {m["recording_key"]: m for m in csv.DictReader(open("/data/home/amartyavarma/data/aloe-handover-full/DATA/manifest.csv"))}
segs = []
for fj in sorted(glob.glob(f"{D}/segs/segs_*.json")):
    segs += json.load(open(fj))
E = np.load(f"{D}/segs/emb.npz")["E"]; E = E / np.linalg.norm(E, axis=1, keepdims=True)
def conv(k): return "/".join(k.split("/")[2:4])
acc = [man[conv(s["row_key"])][f"speaker_{s['lane'][-2:]}_account_uid"] for s in segs]
# global account-level stats
allacc = collections.Counter(); pairs = collections.Counter()
for m in man.values():
    for L in ("00", "01"):
        allacc[m[f"speaker_{L}_account_uid"]] += 1
        pairs[(m[f"speaker_{L}_actor_id"] == m[f"speaker_{L}_account_uid"])] += 1
print("manifest lanes actor_id==account_uid:", dict(pairs))
by = collections.defaultdict(list)
for i, s in enumerate(segs): by[s["rank"]].append(i)
for r in sorted(by):
    idx = by[r]; aid = segs[idx[0]]["actor_id"]
    ac = collections.Counter(acc[i] for i in idx)
    # within-account mean cosine (pairs from different conversations), and across-account
    wi, ac_ = [], []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            i, j = idx[a], idx[b]
            if segs[i]["row_key"] == segs[j]["row_key"]: continue
            (wi if acc[i] == acc[j] else ac_).append(float(E[i] @ E[j]))
    print(f"rank {r:2d} {aid[:8]} accounts {len(ac):2d} top {ac.most_common(1)[0][0][:8]}x{ac.most_common(1)[0][1]:2d} self={int(aid in ac)} "
          f"same-acc cos {np.mean(wi) if wi else float('nan'):.2f} (n{len(wi)})  diff-acc cos {np.mean(ac_) if ac_ else float('nan'):.2f} (n{len(ac_)})")
