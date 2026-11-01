"""Daisy = a PPM podcast persona: rows whose persona prompt names the speaker Daisy ('Your name is Daisy' / 'You are Daisy' / 'You go by Daisy') (qwen-1ch-data/persona/processed, status ok). For each
such row-side: episode key (tree/podcast/episode), side, chunk, row seconds (end_sec - start_sec), the partner side's persona name in the
same chunk, prompt tokens, full prompt. -> daisy/daisy.json"""
import collections, json, re
import pyarrow.parquet as pq
P = "/data/home/amartyavarma/qwen-1ch-data/persona/processed"
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/daisy"
NAME = re.compile(r"[Yy]our name is ([A-Z][a-z]+)")
hits = [json.loads(l) for l in open(f"{D}/hits.jsonl")]
OWN = re.compile(r"\b(?:[Yy]our name is|[Yy]ou are|[Yy]ou're|[Yy]ou go by)\s+Daisy\b(?!['’])")
dz = [h for h in hits if OWN.search(h["system_prompt"] or "")]
mention = len(hits) - len(dz)
files = sorted({h["file"] for h in dz})
rows = []
for f in files:
    t = pq.read_table(f"{P}/{f}", columns=["ek", "side", "chunk_index", "status", "system_prompt", "start_sec", "end_sec", "prompt_tokens"]).to_pylist()
    idx = {(r["ek"], r["side"], r["chunk_index"]): r for r in t}
    for h in dz:
        if h["file"] != f:
            continue
        r = idx[(h["ek"], h["side"], h["chunk_index"])]
        o = idx.get((h["ek"], "user" if h["side"] == "assistant" else "assistant", h["chunk_index"]))
        pn = NAME.search((o or {}).get("system_prompt") or "") if o and o["status"] == "ok" else None
        rows.append({"ek": h["ek"], "node": f.split("/")[-1][:-8], "side": h["side"], "chunk": h["chunk_index"], "sec": (r["end_sec"] or 0) - (r["start_sec"] or 0),
                     "start_sec": r["start_sec"], "partner_name": pn.group(1) if pn else None, "partner_status": (o or {}).get("status"),
                     "tokens": r["prompt_tokens"], "prompt": r["system_prompt"], "who": (lambda m: r["system_prompt"][m.start():m.start() + 260])(OWN.search(r["system_prompt"]))})
eps = collections.defaultdict(list)
for r in rows:
    eps[r["ek"]].append(r)
pods = collections.Counter(r["ek"].split("/")[1] for r in rows)
print("row-sides", len(rows), "episodes", len(eps), "podcasts", len(pods), "hours", round(sum(r["sec"] for r in rows) / 3600, 2),
      "sides", dict(collections.Counter(r["side"] for r in rows)), "trees", dict(collections.Counter(r["ek"][0] for r in rows)),
      "nodes", dict(collections.Counter(r["node"] for r in rows)), "other Daisy mentions", mention)
print("top podcasts", pods.most_common(8))
json.dump({"rows": rows, "mentions_not_name": mention}, open(f"{D}/daisy.json", "w"))
