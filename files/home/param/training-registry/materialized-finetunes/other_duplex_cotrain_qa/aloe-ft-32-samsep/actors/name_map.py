"""account_uid -> name: the top transcript-identified name if it covers >= 60 % of the account's named conversations and >= 2 of them, else
'name unknown' (shares over the conversations with an identified name; 'unknown' / hedged answers left out). Writes actors/name_map.json and prints the Kate-like accounts."""
import json, re
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors"
N = json.load(open(f"{D}/names.json"))["account_uid"]
st = json.load(open(f"{D}/actor_stats_acc.json"))
out = {}
for a in st["actors"]:
    C = [(n, c) for n, c in N.get(a["actor_id"], []) if re.fullmatch(r"[A-Z][A-Za-z'-]+( [A-Z][A-Za-z'-]+)?", n or "") and n.lower() != "unknown"]
    tot = sum(c for _, c in C)
    if C and C[0][1] >= 2 and C[0][1] / tot >= 0.6:
        out[a["actor_id"]] = {"name": C[0][0], "share": round(C[0][1] / tot, 2), "n": tot, "alts": C[1:3]}
    else:
        out[a["actor_id"]] = {"name": None, "share": round(C[0][1] / tot, 2) if C else 0, "n": tot, "alts": C[:3]}
json.dump(out, open(f"{D}/name_map.json", "w"))
known = sum(1 for v in out.values() if v["name"])
print("named", known, "of", len(out))
for i, a in enumerate(st["actors"][:30], 1):
    v = out[a["actor_id"]]; print(i, a["actor_id"][:8], v["name"], v["share"], v["n"], v["alts"][:2])
print("--- Kate-like")
for i, a in enumerate(st["actors"], 1):
    v = out[a["actor_id"]]
    names = [n for n, _ in N.get(a["actor_id"], [])]
    if any(re.fullmatch(r"(Kate|Katie|Kat|Katy|Kathy|Katherine|Kathryn|Catherine|Cate|Caitlin|Kaitlyn)", n or "") for n in names[:3]):
        t, e = a["train"], a["eval"]
        print(f"rank {i} {a['actor_id']} name={v['name']} share={v['share']} names={N[a['actor_id']][:3]} train {t.get('speech_s',0)/3600:.2f} h {t.get('rows',0)} rows eval {e.get('speech_s',0)/3600:.2f} h {e.get('rows',0)} rows")
