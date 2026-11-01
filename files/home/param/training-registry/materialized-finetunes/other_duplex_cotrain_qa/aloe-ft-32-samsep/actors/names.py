"""Names for the speaker accounts. No metadata name field exists in the handover data (manifest / pair.json / speaker_mapping carry only
account_uid + actor_id). The only names are the ones identified from each conversation's transcript (system_prompts/SPEAKER_NN/episode.json
profile.a_speaker = the name this lane's speaker goes by in that conversation, identify.confidence). Per account_uid (and per actor_id, for
comparison): the counter of those names over its first-chunk train+eval conversations -> actors/names.json"""
import collections, csv, json, os, sys
from concurrent.futures import ThreadPoolExecutor
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors"
DATA = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
STORES = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores"
csv.field_size_limit(1 << 30)
man = {m["recording_key"]: m for m in csv.DictReader(open(f"{DATA}/manifest.csv"))}
jobs = []
for k in range(4):
    for ln in open(f"{STORES}/m1n{k}/rows.jsonl"):
        r = json.loads(ln)
        if int(r["chunk_index"]) == 0 and r["split"] in ("train", "eval"):
            m = man[r["src_conversation"]]
            for L in ("00", "01"):
                jobs.append((r["split"], m, L))
def one(j):
    sp, m, L = j
    p = f"{DATA}/{m['relative_dir']}/system_prompts/SPEAKER_{L}/episode.json"
    try:
        d = json.load(open(p))
        pr = d["profile"]
        return sp, m[f"speaker_{L}_account_uid"], m[f"speaker_{L}_actor_id"], pr.get("a_speaker"), (pr.get("identify") or {}).get("confidence"), d.get("identity_confidence")
    except Exception as e:
        return sp, m[f"speaker_{L}_account_uid"], m[f"speaker_{L}_actor_id"], None, "missing:" + type(e).__name__, None
with ThreadPoolExecutor(48) as ex:
    res = list(ex.map(one, jobs))
acc, act = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
conf = collections.Counter()
for sp, a, b, name, c, ic in res:
    conf[str(c)] += 1
    nm = (name or "?").strip()
    acc[a][nm] += 1
    act[b][nm] += 1
def purity(C):
    tot = sum(C.values()); top = C.most_common(1)[0][1]
    return top / tot
big_acc = [a for a, C in acc.items() if sum(C.values()) >= 20]
big_act = [a for a, C in act.items() if sum(C.values()) >= 20]
print("lanes", len(res), "confidence", dict(conf))
print("accounts>=20 convs:", len(big_acc), "median top-name share", sorted(purity(acc[a]) for a in big_acc)[len(big_acc)//2])
print("actor_ids>=20 convs:", len(big_act), "median top-name share", sorted(purity(act[a]) for a in big_act)[len(big_act)//2])
json.dump({"account_uid": {a: C.most_common() for a, C in acc.items()}, "actor_id": {a: C.most_common() for a, C in act.items()}},
          open(f"{D}/names.json", "w"))
