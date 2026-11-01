import io, json, sys, numpy as np, pyarrow.parquet as pq
sys.path.insert(0, "/data/home/amartyavarma/qwen-1ch/stage3/data/rows_build/scripts")
import s3rows as F
HF = "/data/home/amartyavarma/data/aloe-sam-sep/hf"
ses, seg = "cc3d4bbe-a355-4332-a59b-a11f269b5589", "seg00"
ix = pq.read_table(f"{HF}/index/pairs.parquet").to_pandas()
r = ix[(ix.session == ses) & (ix.seg == seg)].iloc[0]
print({k: r[k] for k in ix.columns if not k.startswith("bytes_") and k not in ("qc_json",)})
qc = json.loads(r["qc_json"]) if isinstance(r["qc_json"], str) else r["qc_json"]
print("qc_json:", json.dumps(qc)[:3000])
pf = pq.ParquetFile(f"{HF}/{r.token_shard}")
row = pf.read_row_group(int(r.token_row) // 64, columns=["session", "seg", "SPEAKER_00_isolated_npy", "SPEAKER_01_isolated_npy", "mix_SPEAKER_00_npy", "mix_SPEAKER_01_npy", "mimi_json"]).to_pylist()[int(r.token_row) % 64]
assert row["session"] == ses
iso0, iso1 = (np.load(io.BytesIO(row[f"SPEAKER_0{i}_isolated_npy"])) for i in (0, 1))
mx0, mx1 = (np.load(io.BytesIO(row[f"mix_SPEAKER_0{i}_npy"])) for i in (0, 1))
S = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1/stores/m1n1"
t = pq.read_table(f"{S}/rows.parquet").to_pandas()
rows = t[t.src_conversation == f"{ses}/{seg}"]
print(rows[["row", "row_key", "chunk_index", "frame_start", "n_frames", "split", "src_frames"]])
nod = F.load_node(S); R = F.Rows(nod)
raw = {0: np.full((32, int(rows.src_frames.iloc[0])), -1, np.int16), 1: None}
raw[1] = raw[0].copy()
W = {0: [], 1: []}
for _, x in rows.iterrows():
    rr = R[int(x.row)]; a, n = int(x.frame_start), int(x.n_frames)
    raw[0][:, a:a+n] = rr["codes_assistant"][:, :n]; raw[1][:, a:a+n] = rr["codes_user"][:, :n]
    W[0] += [w["start_frame"] + a for w in rr["words_assistant"]]; W[1] += [w["start_frame"] + a for w in rr["words_user"]]
T = raw[0].shape[1]
print("T", T, iso0.shape)
for a in range(0, T, 375):
    b = min(a + 375, T); v = raw[0][0, a:b] >= 0
    if not v.any(): continue
    f = lambda x, y: round(float((x[0, a:b][v] == y[0, a:b][v]).mean()), 2)
    print(f"{a/12.5:6.0f}s words L0 {sum(a<=w<b for w in W[0]):3d} L1 {sum(a<=w<b for w in W[1]):3d} | iso0~raw0 {f(iso0, raw[0])} iso0~raw1 {f(iso0, raw[1])} | iso1~raw1 {f(iso1, raw[1])} iso1~raw0 {f(iso1, raw[0])} | mix0~raw0 {f(mx0, raw[0])} mix0~raw1 {f(mx0, raw[1])} | iso0~iso1 {f(iso0, iso1)} raw0~raw1 {f(raw[0], raw[1])}")
mj = json.loads(row["mimi_json"]); print("mimi_json flags", json.dumps({k: v for k, v in mj.items() if "flag" in k.lower()})[:1000])
