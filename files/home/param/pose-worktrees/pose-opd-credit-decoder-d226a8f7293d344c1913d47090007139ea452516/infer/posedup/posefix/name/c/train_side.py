"""posefix NAME-C: the poseduplex TRAINER's micro for Kate's row (s2h-m1-samsep m1n0 row 2937, swap 0 = assistant = Kate on B) with B's voice
clip FORCED on (want_voice -> True; the clip = data_ppm.voice_clips rule, unchanged), exactly data_aloe_posedup / v2vox / v2 code paths."""
import json, os, sys
E = {"ALOE_SAMSEP_STORES": "/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores", "ALOE_STORES_FROM_DATA": "1",
     "ALOE_SAMSEP_SRC": "/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores",
     "ALOE_SAMSEP_DIGESTS": "7318ffb3ea75cbcc3ff48b63b6cf8b88,31ecb2273542310096936c0baf49efa8,1759200f7818c6017dfb6a46e37fa5df,611b2e7edb1503cc600a6489e49dcf40",
     "ALOE_SAMSEP_ROW_MAP": "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/data/row_map.json",
     "ALOE_M1VOX_INDEX": "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/data/plan/M1VOX_INDEX_samsep.json",
     "ALOE_RAWGAIN": "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/data/plan/RAWGAIN_samsep.json", "ALOE_RAWGAIN_PLAN_DIGEST": "d1a3e3721328da0a",
     "VOX_A_GAIN": "raw", "ALOE_FLAC_ROOT": "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/data/flac",
     "POSEDUP_POSE": "1", "POSEDUP_ITEMS": "all", "POSEDUP_POSE_INDEX": "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/data/POSE_INDEX.json",
     "POSEDUP_POSE_INDEX_SHA": "7114d5a04f43778b29244e2ef1867d6d960d619f1515bf3c25f4b7f155f8d8a1", "POSEDUP_POSE_SHA_EVERY_READ": "1",
     "POSEDUP_POSE_LIGHT_FRAC": "1.0", "POSEDUP_SPEECH_LIGHT_FRAC": "0.1", "POSEDUP_VOICE_RULE": "v2_30", "POSEDUP_EPOCH_SEED": "20261011"}
os.environ.update(E)
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/code")
import numpy as np
import data_aloe_posedup as DP
OUT = sys.argv[1]
cfg = DP.aloe_cfg()
print("cfg new_word", cfg.get("new_word"), "sp_cap", cfg.get("sp_cap"), "voice_seed", cfg.get("voice_seed"), flush=True)
st = DP.Stores()
node, rows, sib = st.open(0)
row = rows[2937]
print("row", row["row_key"], "n_frames", row["n_frames"], "chunk", row["chunk_index"], flush=True)
drawn = DP.A2.want_voice(row["row_key"], 0, cfg["voice_seed"])
print("trainer voice draw for (row, swap 0):", drawn, flush=True)
DP.A2.want_voice = lambda *a, **k: True                      # force B's clip on (rule unchanged)
item = (0, 2937, 0)
m = st.conv(item, cfg, 0, audio=True)
print("micro keys", sorted(m.keys()) if m else None, flush=True)
pv = json.loads(m["prov"])
print("prov", json.dumps(pv)[:600], flush=True)
vc = DP.D.voice_clips(row, sib, cfg)
print("trainer voice_clips rule for this row:", {d: (None if c is None else {k: v for k, v in c.items() if k != "src_row"}) for d, c in vc.items()}, flush=True)
# no trainer clip exists for Kate's row -> a TRAINER-FORMAT clip dict (data_ppm.voice_clips' fields) on the samsep package's window 2416:2541
cl = {"w0": 2416, "L": 125, "src": "later_2nd_half", "src_row": row, "src_side": "assistant", "src_key": row["row_key"], "src_chunk": 0,
      "blank_lane": True, "side": "assistant", "n_chunks": 1}
ccodes = np.asarray(row["codes_assistant"])[:, 2416:2541].astype(np.int64)
spx = DP.D.SPR.build_prefix(row, [{"kind": "speech", "track": 1, **cl}], {"new_word": cfg["new_word"]})
np.savez(OUT.replace(".npz", "_prefix.npz"), ids=spx["ids"], depth=spx["depth"], present=spx["present"], P=int(spx["P"]), clip_codes=ccodes)
print("trainer-format prefix P", spx["P"], flush=True)
np.savez(OUT, ids=m["ids"], depth=m["depth"], present=m["present"], pid=m["pid"], P=int(m["P"]), n=int(row["n_frames"]),
         pdepth=m.get("pdepth", np.zeros(0)), clip_codes=ccodes, clip_meta=json.dumps({k: v for k, v in cl.items() if k != "src_row"}, default=str),
         wav=m.get("wav", np.zeros(0, np.float32)), vin=np.asarray(m.get("vin", (0, 0))), prov=m["prov"], aref=json.dumps(m.get("aref", {})))
print("saved", OUT, "P", m["P"], "T", len(m["pid"]), "clip", {k: v for k, v in cl.items() if k not in ("src_row",)}, flush=True)
