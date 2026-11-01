"""Compare SAM test-slice lean-v12.16 system_prompts with the PPM processed persona rows (same ek/side/chunk). Writes sam_check.json."""
import glob, json, os, collections
import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
from concurrent.futures import ThreadPoolExecutor
OUT = os.path.dirname(os.path.abspath(__file__))
X = "/data/home/amartyavarma/data/sam-subsets-test/extracted"
P = "/data/home/amartyavarma/qwen-1ch-data/persona/processed"
TREE = {"700k-70kh-v1": "A", "intelligent-30kh-v1": "B", "700k": "A", "intelligent": "B"}
SIDE = {"speaker_0": "user", "speaker_1": "assistant"}
sam = {}; vers = collections.Counter()
for f in glob.glob(f"{X}/*/*/*/system_prompts/speaker_*/chunk_*.json"):
    parts = f.split("/")
    tree, pod, ep, spk = parts[-6], parts[-5], parts[-4], parts[-2]
    j = json.load(open(f)); vers[j.get("version")] += 1
    sam[(f"{TREE.get(tree, tree)}/{pod}/{ep}", SIDE[spk], int(j["chunk_index"]))] = j["system_prompt"]
eks = sorted({k[0] for k in sam})
print("sam prompts", len(sam), "episodes", len(eks), dict(vers), flush=True)
fs = [f for n in (2, 7, 8, 13) for f in glob.glob(f"{P}/*/s*/slinky-{n}.parquet")]
def scan(f):
    t = pq.read_table(f, columns=["ek"])
    m = pc.is_in(t["ek"], value_set=pa.array(eks))
    if not pc.any(m).as_py():
        return []
    t = pq.read_table(f, columns=["ek", "side", "chunk_index", "status", "system_prompt"]).filter(m)
    return t.to_pylist()
hits = []
with ThreadPoolExecutor(16) as ex:
    for r in ex.map(scan, fs):
        hits += r
C = collections.Counter(); diffs = []
for r in hits:
    k = (r["ek"], r["side"], r["chunk_index"])
    if k not in sam:
        C[f"ppm_only_{r['status']}"] += 1; continue
    if r["status"] != "ok":
        C[f"sam_has_ppm_{r['status']}"] += 1; continue
    same = r["system_prompt"] == sam[k]; C["identical" if same else "different"] += 1
    if not same and len(diffs) < 3:
        diffs.append({"key": k, "ppm": r["system_prompt"][:400], "sam": sam[k][:400]})
C["sam_prompts"] = len(sam); C["sam_not_in_ppm"] = len(set(sam) - {(r["ek"], r["side"], r["chunk_index"]) for r in hits})
json.dump({"counts": dict(C), "versions": dict(vers), "files_scanned": len(fs), "diffs": diffs}, open(f"{OUT}/sam_check.json", "w"), indent=1)
print(dict(C))
