"""Find 'Daisy' in the PPM podcast persona prompts (qwen-1ch-data/persona/processed/<run>/<sXXXX>/<node>.parquet, column system_prompt,
status ok rows) -> daisy/hits.jsonl (ek, side, chunk_index, run, node, prompt_tokens, is_name, snippet, system_prompt)."""
import glob, json, re, sys
from multiprocessing import Pool
import pyarrow.parquet as pq, pyarrow.compute as pc
P = "/data/home/amartyavarma/qwen-1ch-data/persona/processed"
OUT = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/daisy/hits.jsonl"
NAME = re.compile(r"(?:[Yy]our name is|[Yy]ou are|[Yy]ou're|name:?)\s+Daisy\b")
def one(f):
    t = pq.read_table(f, columns=["ek", "side", "chunk_index", "status", "system_prompt", "prompt_tokens"])
    m = pc.match_substring_regex(pc.fill_null(t.column("system_prompt"), ""), r"\bDaisy\b")
    t = t.filter(m)
    out = []
    for r in t.to_pylist():
        sp = r["system_prompt"] or ""
        i = sp.find("Daisy")
        out.append({**r, "file": f[len(P) + 1:], "is_name": bool(NAME.search(sp)), "snippet": sp[max(0, i - 120):i + 120]})
    return out, t.num_rows
if __name__ == "__main__":
    fs = sorted(glob.glob(f"{P}/*/*/*.parquet"))
    print("files", len(fs), flush=True)
    n = 0
    with open(OUT, "w") as fh, Pool(4) as pool:
        for out, k in pool.imap_unordered(one, fs, chunksize=4):
            for r in out:
                fh.write(json.dumps(r) + "\n")
            n += k
    print("hits", n, flush=True)
