"""where is each package's 0-12 s window in our v2pure cache? text + first-word-time alignment (read-only)."""
import json
import re
import pyarrow.parquet as pq
CACHE = "/data/home/amartyavarma/aloe-pose/sam-cache-active-v2pure"
P = "/data/home/amartyavarma/video/demo-data/prompts"
plan = json.load(open(f"{CACHE}/plan.json"))
norm = lambda s: re.sub(r"[^a-z' ]", "", s.lower()).split()  # noqa: E731
for idn in ("10_27286bdd", "06_74807fe0", "19_47473a46"):
    m = json.load(open(f"{P}/{idn}/meta.json"))
    rid, spk = m["recording_id"], f"speaker_{m['speaker']}"
    sh = [s for s, l in plan["shards"].items() if rid in l][0]
    ci = pq.read_table(f"{CACHE}/{sh}/items.parquet").to_pandas()
    it = ci[(ci["rid"] == rid) & (ci["speaker"] == spk)].sort_values("start")
    pw = norm(m["text"])
    print(f"== {idn} rid {rid[:8]} {spk} chunk {m['chunk_index']} first_word_s {m['first_word_s']:.2f} words_in_prefill {m['words_in_prefill']}")
    print("   package text:", " ".join(pw[:16]))
    hits = []
    for _, r in it.iterrows():
        ws = json.loads(r["words"])
        tw = [norm(w[2]) for w in ws]
        flat = [x for t in tw for x in t]
        for j in range(len(flat)):
            if flat[j:j + 5] == pw[:5]:
                hits.append((float(r["start"]), r["key"], j, len(flat)))
    print("   items of speaker:", len(it), "first starts:", [round(float(s), 2) for s in it["start"].values[:6]])
    print("   5-word hits:", hits[:5])
