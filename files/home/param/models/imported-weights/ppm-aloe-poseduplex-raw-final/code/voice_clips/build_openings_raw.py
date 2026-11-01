"""pdxhost 10-30: genweb/openings-raw = genweb/openings-sam with B's Mimi codes swapped to the RAW s2h-m1 lane codes (the raw-lane
mimi-stream-v5 codes the 33111 'pdxraw' model trained on, POSEDUP_RAW_CODES=1). Same rule as poseduplex-raw/code/data_aloe_posedup.RawRows:
sam-sep row -> mapped raw row via ALOE_SAMSEP_ROW_MAP (old_of_new), codes read from the raw store at the SAME frames and side. Everything
else (pose tokens, words, clipwin words / pose, frame0) is copied byte for byte. Asserted per package: store digests pinned (= RAW_CODES.md),
mapped raw row provenance == sam-sep row (row_key, n_frames, chunk_index, frame_start, split, speaker dirs, src_conversation), the sam-sep
slice in the store == the openings-sam file (so the window is the right one), same shape / dtype / valid mask. meta.json written LAST."""
import hashlib, json, os, shutil, sys
import numpy as np
import pyarrow.parquet as pq
from safetensors.numpy import load_file, save_file

SAM = "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb/openings-sam"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/data/home/amartyavarma/ppm/aloe-posedup-live/genweb/openings-raw"
SS = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores"
RS = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1/stores"
MAP = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/data/row_map.json"
RAW_DIG = ["c1c3b2517152741137cc4bb72dd6bdbb", "09bdd43fef84574175caacf4abdddd1e", "ecb1c731e8989b3f2e1b5266b51ea0fd", "3b16fb8dd2660a2e8c1fe32c0d361d7b"]
SAM_DIG = ["7318ffb3ea75cbcc3ff48b63b6cf8b88", "31ecb2273542310096936c0baf49efa8", "1759200f7818c6017dfb6a46e37fa5df", "611b2e7edb1503cc600a6489e49dcf40"]
COLS = ("row_key", "n_frames", "chunk_index", "frame_start", "split", "user_speaker_dir", "assistant_speaker_dir", "src_conversation")
j = json.load(open(MAP))
assert [j["old_digest"][f"m1n{k}"] for k in range(4)] == RAW_DIG and [j["new_digest"][f"m1n{k}"] for k in range(4)] == SAM_DIG
for k in range(4):
    for root, dig in ((SS, SAM_DIG), (RS, RAW_DIG)):
        nd = json.load(open(f"{root}/m1n{k}/NODE.json")); h = hashlib.blake2b(digest_size=16)   # = s3rows.load_node digest
        for s_, dg in zip(nd["shards"], nd["ids_blake2b"]):
            h.update(s_.encode()); h.update(dg.encode())
        assert h.hexdigest() == dig[k], (root, k, h.hexdigest())
ST = {}
def store(k):
    if k not in ST:
        mp = j["map"][f"m1n{k}"]; no = {int(o): int(n) for o, n in mp["new_of_old"].items()}
        oon = np.full(len(no), -1, np.int64)
        for o, n in no.items(): oon[n] = o
        assert (oon >= 0).all() and len(np.unique(oon)) == len(oon)
        tn = pq.read_table(f"{SS}/m1n{k}/rows.parquet", columns=["row", *COLS]).to_pydict()
        to = pq.read_table(f"{RS}/m1n{k}/rows.parquet", columns=["row", *COLS]).to_pydict()
        assert tn["row"] == list(range(len(oon))) and to["row"] == list(range(int(mp["old_rows"])))
        cod = {sd: (np.load(f"{SS}/m1n{k}/rows-00000-of-00001.codes_{sd}.npy", mmap_mode="r"),
                    np.load(f"{RS}/m1n{k}/rows-00000-of-00001.codes_{sd}.npy", mmap_mode="r")) for sd in ("user", "assistant")}
        assert cod["user"][0].shape[0] == len(oon) and cod["user"][1].shape[0] == int(mp["old_rows"]), "single-shard stores, row = node row"
        ST[k] = dict(oon=oon, tn=tn, to=to, cod=cod, key2row={rk: i for i, rk in enumerate(tn["row_key"])})
    return ST[k]
def find(row_key):
    for k in range(4):
        s = store(k)
        if row_key in s["key2row"]:
            return k, s["key2row"][row_key]
    raise KeyError(row_key)
def raw_slice(k, r, side, a, b, sam_file_codes):
    s = store(k); o = int(s["oon"][r])
    for c in COLS:
        assert s["tn"][c][r] == s["to"][c][o], ("mapped raw row != sam-sep row", k, r, c)
    ss, rr = s["cod"][side]
    sam = np.asarray(ss[r][:, a:b]); raw = np.asarray(rr[o][:, a:b])
    assert np.array_equal(sam.astype(np.int64), np.asarray(sam_file_codes, np.int64)), ("sam-sep store slice != openings-sam file", k, r, side, a, b)
    assert raw.shape == sam.shape and raw.dtype == sam.dtype and np.array_equal(raw >= 0, sam >= 0), ("raw shape/dtype/mask", k, r)
    return raw, o, float((raw[0] != sam[0]).mean())
def sha(a): return hashlib.sha256(np.ascontiguousarray(np.asarray(a, np.int64)).tobytes()).hexdigest()
os.makedirs(OUT, exist_ok=True); rep = {}
for p in sorted(os.listdir(SAM)):
    d = f"{SAM}/{p}"
    if not os.path.isfile(f"{d}/meta.json"): continue
    m = json.load(open(d + "/meta.json")); ml, vc = m["sources"]["mimi_local"], m["sources"]["voice_clip"]
    k = int(ml["store_node"][-1]); r = int(ml["node_row"]); side = ml["side"]
    a, b = (int(x) for x in ml["rows"].split("[:, ")[1].rstrip("]").split(":"))
    assert store(k)["tn"]["row_key"][r] == ml["store_row"], (p, "store_row")
    fm = load_file(f"{d}/mimi_local.safetensors")["codes"]
    raw_m, om, dm = raw_slice(k, r, side, a, b, fm)
    kv, rv = find(vc["src_row_key"]); a2, b2 = (int(x) for x in vc["frames_in_src_row"])
    fv = load_file(f"{d}/voice_clip.safetensors")["codes"]
    raw_v, ov, dv = raw_slice(kv, rv, vc["src_side"], a2, b2, fv)
    od = f"{OUT}/{p}"; os.makedirs(od, exist_ok=True)
    for f in os.listdir(d):
        if os.path.isfile(f"{d}/{f}") and f not in ("meta.json", "mimi_local.safetensors", "voice_clip.safetensors", "clipwin_meta.json"):
            shutil.copy2(f"{d}/{f}", f"{od}/{f}")
    save_file({"codes": np.ascontiguousarray(raw_m.astype(fm.dtype))}, f"{od}/mimi_local.safetensors")
    save_file({"codes": np.ascontiguousarray(raw_v.astype(fv.dtype))}, f"{od}/voice_clip.safetensors")
    if os.path.isfile(f"{d}/clipwin_meta.json"):
        cm = json.load(open(f"{d}/clipwin_meta.json"))
        cm["voice_clip_codes_sha256"] = cm["clipwin_codes_sha256"] = sha(raw_v)
        cm["raw_codes"] = "pdxhost 10-30: codes = RAW s2h-m1 lane (mapped raw row, same frames); pose / words unchanged from openings-sam"
        json.dump(cm, open(f"{od}/clipwin_meta.json", "w"), indent=1)
    raw_info = {"what": "B Mimi codes swapped to the RAW s2h-m1 lane codes (raw-lane mimi-stream-v5, lufs23-local; = what 33111 pdxraw trained on as B, POSEDUP_RAW_CODES=1, RawRows rule); everything else = openings-sam byte for byte",
                "row_map": MAP, "raw_stores": RS, "raw_digests": RAW_DIG,
                "mimi_local": {"raw_node": f"m1n{k}", "raw_row": om, "sam_row": r, "side": side, "frames": [a, b], "sha256_int64": sha(raw_m), "cb0_frac_differs_from_sam": round(dm, 4)},
                "voice_clip": {"raw_node": f"m1n{kv}", "raw_row": ov, "sam_row": rv, "side": vc["src_side"], "frames": [a2, b2], "sha256_int64": sha(raw_v), "cb0_frac_differs_from_sam": round(dv, 4)},
                "built_from": d}
    m["kind"] = "s2handover-raw"; m["raw_codes"] = raw_info
    json.dump(m, open(f"{od}/meta.json", "w"), indent=1)                 # LAST = ready
    rep[p] = {"mimi_cb0_diff": round(dm, 4), "clip_cb0_diff": round(dv, 4)}
    print(p, rep[p], flush=True)
shutil.copy2(f"{SAM}/SUMMARY.json", f"{OUT}/SUMMARY.json")
open(f"{OUT}/README.md", "w").write("# genweb/openings-raw (pdxhost 10-30)\n\nThe 15 genweb/openings-sam packages with B's Mimi codes (mimi_local.safetensors = conversation start, voice_clip.safetensors = the training-rule voice clip) swapped to the RAW s2h-m1 lane codes of the mapped raw row at the same frames (ALOE_SAMSEP_ROW_MAP; = poseduplex-raw/code/data_aloe_posedup.RawRows, what job 33111 'pdxraw' trained on as B). Pose tokens, words, clip-window words / pose and frame0 are byte-identical to openings-sam. Built by poseduplex-raw/host211/tools/build_openings_raw.py; every package's meta.json carries a raw_codes block (rows, frames, sha256, cb0 fraction that differs from SAM). Read by the pose-duplex live server via stage-lb/instP/OPENINGS_ROOT.txt while it serves the raw-codes model.\n")
json.dump(rep, open(f"{OUT}/BUILD_REPORT.json", "w"), indent=1)
print("DONE", len(rep), "packages ->", OUT)
