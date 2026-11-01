"""Voice embeddings for every speaker lane in the Aloe pool (none-crossbleed handover calls with sam-sep codes): per (call, lane) the longest
own-speech segment (words.json _separated_speaker runs, gaps < 0.5 s closed, partner silent within 1 s, 3-6 s; Ashley's lanes: up to 3),
decoded from that lane's SAM-isolated Mimi codes (aloe-sam-sep HF parquet) with transformers MimiModel (1 s left context trimmed), resampled
to 16 kHz, ECAPA (speechbrain spkrec-ecapa-voxceleb, local copy) -> allvoices/emb_<SHARD>.npz (E [N,192], meta json)
usage: embed.py SHARD NSHARD (parquet files i with i % NSHARD == SHARD)"""
import csv, glob, io, json, sys
import numpy as np, torch
import pyarrow.parquet as pq
sys.path.append(glob.glob("/data/home/amartyavarma/venvs/voicebank/lib/python3*/site-packages")[0])
import torchaudio  # noqa: E402  (voicebank's, same torch 2.8)
from speechbrain.inference.speaker import EncoderClassifier  # noqa: E402
SH, NSH = int(sys.argv[1]), int(sys.argv[2])
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/allvoices"
DATA = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
HF = "/data/home/amartyavarma/data/aloe-sam-sep/hf"
ASH = "TyypZEov3LSf5Gp8N6kSnBTZYXy2"
CTX = 13
csv.field_size_limit(1 << 30)
pairs = {(p["session"], p["seg"]): p for p in pq.read_table(f"{HF}/index/pairs.parquet", columns=["session", "seg", "token_shard", "token_row"]).to_pylist()}
files = sorted({p["token_shard"] for p in pairs.values()})
mine = {f for i, f in enumerate(files) if i % NSH == SH}
man = [m for m in csv.DictReader(open(f"{DATA}/manifest.csv")) if m["relative_dir"].split("/")[1] == "none" and (m["session_id"], m["seg"]) in pairs
       and pairs[(m["session_id"], m["seg"])]["token_shard"] in mine]
def segs(W, L, c0, T, k):
    own = np.zeros(T, bool); oth = np.zeros(T, bool)
    for w in W:
        a, b = int((w["start"] - c0) / 0.08), int((w["end"] - c0) / 0.08) + 1
        (own if w.get("_separated_speaker") == f"SPEAKER_{L}" else oth)[max(0, a):max(0, min(T, b))] = True
    idx = np.flatnonzero(own); o = own.copy()
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= 6:
            o[a:b] = True
    d = np.diff(np.concatenate([[0], o.astype(int), [0]]))
    out = []
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        e = min(e, s + 75)
        if e - s >= 38 and not oth[max(0, s - 12):min(T, e + 12)].any():
            out.append((int(s), int(e)))
    return sorted(out, key=lambda x: x[0] - x[1])[:k]
plan = {}
for m in man:
    try:
        W = json.load(open(f"{DATA}/{m['relative_dir']}/w00.words.json"))["words"]
    except Exception:
        continue
    for L in ("00", "01"):
        acc = m[f"speaker_{L}_account_uid"]
        for s, e in segs(W, L, float(m["clip_start_sec"]), int(m["mimi_T"]), 3 if acc == ASH else 1):
            plan.setdefault((m["session_id"], m["seg"]), []).append({"acc": acc, "key": m["recording_key"], "lane": L, "s": s, "e": e})
print("shard", SH, "calls", len(man), "segments", sum(len(v) for v in plan.values()), flush=True)
dev = torch.device("cuda")
from transformers import MimiModel  # noqa: E402
mimi = MimiModel.from_pretrained("/data/home/amartyavarma/models/mimi").to(dev).eval()
ec = EncoderClassifier.from_hparams(source="/data/home/amartyavarma/data/voicebank-yodas3/models/spkrec-ecapa-voxceleb",
                                    savedir="/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/speakers/ecapa", run_opts={"device": "cuda:0"})
E, meta = [], []
for f in sorted(mine):
    pf = pq.ParquetFile(f"{HF}/{f}")
    for rg in range(pf.metadata.num_row_groups):
        t = pf.read_row_group(rg, columns=["session", "seg", "SPEAKER_00_isolated_npy", "SPEAKER_01_isolated_npy"]).to_pylist()
        for r in t:
            for p in plan.get((r["session"], r["seg"]), []):
                C = np.load(io.BytesIO(r[f"SPEAKER_{p['lane']}_isolated_npy"])).astype(np.int64)
                lo = max(0, p["s"] - CTX)
                with torch.no_grad():
                    w = mimi.decode(torch.as_tensor(C[:, lo:p["e"]], device=dev)[None]).audio_values[0, 0].float()
                    w = w[(p["s"] - lo) * 1920:(p["e"] - lo) * 1920].clamp(-1, 1)
                    w16 = torchaudio.functional.resample(w[None], 24000, 16000)
                    emb = ec.encode_batch(w16)[0, 0].float().cpu().numpy()
                E.append(emb / np.linalg.norm(emb)); meta.append(p)
    print("shard", SH, "file", f, "done", len(E), flush=True)
np.savez(f"{D}/emb_{SH}.npz", E=np.stack(E).astype(np.float32), meta=json.dumps(meta))
print("shard", SH, "saved", len(E), flush=True)
