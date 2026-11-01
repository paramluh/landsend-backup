"""aloe-v2vox-raw-w14503 pre-launch checks (CPU only, run on a compute node):
1. init w-14503 complete: COMPLETE + .metadata; every storage file present with size >= max(offset + length) of its chunks; key set (non-pose)
   == local-11488's (23081's init) key set (rank0.pt + rep.pt), shapes equal.
2. resolved CFG of the new launch (env file + 3-node launch env) vs 23081's logged cfg: print every differing key.
usage: python -I precheck.py <new env already sourced in the environment>"""
import json, os, pickle, sys

NEW = "/data/home/amartyavarma/ppm/aloe-v2vox-raw-w14503"
INIT = os.environ["QQA_INIT"]
REF_INIT = "/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488"
sys.path.insert(0, f"{NEW}/code"); sys.path.insert(0, f"{NEW}/code/lib")
import torch  # noqa: E402

# ---- 1. init completeness + keys
assert os.path.isfile(f"{INIT}/COMPLETE") and os.path.isfile(f"{INIT}/.metadata"), INIT
md = pickle.load(open(f"{INIT}/.metadata", "rb"))
need = {}
for idx, st in md.storage_data.items():
    need[st.relative_path] = max(need.get(st.relative_path, 0), st.offset + st.length)
bad = []
for f, n in sorted(need.items()):
    p = f"{INIT}/{f}"
    sz = os.path.getsize(p) if os.path.isfile(p) else -1
    if sz < n:
        bad.append((f, sz, n))
print(f"[init] {INIT}: COMPLETE={open(f'{INIT}/COMPLETE').read().strip()} storage files {len(need)}, short/missing {len(bad)} {bad[:4]}; "
      f"total bytes needed {sum(need.values()) / 1e9:.1f} GB")
keys = {k: tuple(v.size) for k, v in md.state_dict_metadata.items() if hasattr(v, "size")}
POSE = ("pose_head.", "tower_p.", "dec_p.", "decp_bb.", "inp.pose.")
r0 = torch.load(f"{REF_INIT}/rank0.pt", map_location="cpu", mmap=True, weights_only=True)
rep = torch.load(f"{REF_INIT}/rep.pt", map_location="cpu", mmap=True, weights_only=True)
meta = json.load(open(f"{REF_INIT}/meta.json"))
refk = set(r0) | set(rep)
newk = set(keys)
np_new = {k for k in newk if not k.startswith(POSE)}
np_ref = {k for k in refk if not k.startswith(POSE)}
print(f"[init] keys: w-14503 {len(newk)} ({len(np_new)} non-pose) | local-11488 {len(refk)} ({len(np_ref)} non-pose)")
print(f"[init] non-pose keys only in w-14503: {sorted(np_new - np_ref)[:10]} | only in local-11488: {sorted(np_ref - np_new)[:10]}")
print(f"[init] pose-prefixed keys only in w-14503 (skipped by the trainer): {len({k for k in newk if k.startswith(POSE)} - refk)}")
# shapes: rep.pt tensors are full; rank files are dim-0 shards (full shape = sum over ranks) -> compare rep tensors + non-dim0 dims
shp_bad = [k for k in np_new & set(rep) if tuple(rep[k].shape) != keys[k]]
shp_bad += [k for k in np_new & set(r0) if k not in rep and r0[k].dim() > 1 and tuple(r0[k].shape[1:]) != keys[k][1:]]
print(f"[init] shape mismatches (rep full / shard trailing dims): {len(shp_bad)} {shp_bad[:5]}")
ok_init = not bad and np_new == np_ref and not shp_bad
print(f"[init] VERDICT {'PASS' if ok_init else 'FAIL'}")

# ---- 2. resolved CFG vs 23081's logged cfg
import train_aloe_v2vox_raw as T  # noqa: E402
cur = T._flat(json.loads(json.dumps(T.CFG, default=str)))
ref = T._flat(json.load(open(f"{NEW}/logs/cfg_23081.json")))
rows = [(k, ref.get(k, "<absent>"), cur.get(k, "<absent>")) for k in sorted(set(ref) | set(cur)) if ref.get(k, "<absent>") != cur.get(k, "<absent>")]
print(f"[cfg] resolved config vs 23081's logged cfg: {len(rows)} of {len(set(ref) | set(cur))} keys differ")
for k, a, b in rows:
    print(f"   {k}: 23081 {a} -> new {b}")
n, badk = T.cfg_diff_v2vox()
print(f"[cfg] trainer's own startup check vs 22420: {n} keys differ, outside allowed: {badk}")
