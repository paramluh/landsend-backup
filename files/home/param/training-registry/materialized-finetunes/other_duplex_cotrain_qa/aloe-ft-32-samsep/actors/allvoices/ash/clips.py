"""Listen clips for Ashley and her nearest voices (usage: clips.py ACC1,ACC2,...): for each account, up to 3 of its none-crossbleed calls with
sam-sep codes; per call the 25 s window of its own lane with the most of its own words (handover words.json, _separated_speaker); audio =
that lane's SAM-isolated Mimi codes (aloe-sam-sep HF parquet SPEAKER_0N_isolated_npy) decoded with transformers MimiModel (1 s left
context trimmed); text = its own words in the window -> brianna/clips.json + audio/*.mp3|wav"""
import csv, io, json, os
import numpy as np, soundfile as sf, torch
import pyarrow.parquet as pq
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/allvoices/ash"
DATA = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
HF = "/data/home/amartyavarma/data/aloe-sam-sep/hf"
import sys
ACCS = sys.argv[1].split(",")
WIN, CTX = 313, 13
csv.field_size_limit(1 << 30)
man = list(csv.DictReader(open(f"{DATA}/manifest.csv")))
pairs = {(p["session"], p["seg"]): p for p in pq.read_table(f"{HF}/index/pairs.parquet", columns=["session", "seg", "token_shard", "token_row"]).to_pylist()}
plan = []
for acc in ACCS:
    cands = []
    for m in man:
        for L in ("00", "01"):
            if m[f"speaker_{L}_account_uid"] != acc or m["relative_dir"].split("/")[1] != "none" or (m["session_id"], m["seg"]) not in pairs:
                continue
            W = json.load(open(f"{DATA}/{m['relative_dir']}/w00.words.json"))["words"]
            c0, T = float(m["clip_start_sec"]), int(m["mimi_T"])
            own = np.zeros(T, bool); words = []
            for w in W:
                if w.get("_separated_speaker") == f"SPEAKER_{L}":
                    a, b = int((w["start"] - c0) / 0.08), int((w["end"] - c0) / 0.08) + 1
                    own[max(0, a):max(0, min(T, b))] = True
                    words.append((a, w["text"]))
            cs = np.convolve(own.astype(float), np.ones(WIN), "valid")
            if len(cs) == 0:
                continue
            j = int(np.argmax(cs))
            cands.append((float(cs[j]), m, L, j, " ".join(t for a, t in words if j <= a < j + WIN)))
    cands.sort(key=lambda x: -x[0])
    for sc, m, L, j, txt in cands[:3]:
        p = pairs[(m["session_id"], m["seg"])]
        plan.append({"acc": acc, "key": m["recording_key"], "lane": L, "j": j, "own_s": sc * 0.08, "text": txt, "shard": p["token_shard"], "row": int(p["token_row"])})
    print(acc[:8], "calls with codes", len(cands), "picked", min(3, len(cands)), flush=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import MimiModel
mimi = MimiModel.from_pretrained("/data/home/amartyavarma/models/mimi").to(dev).eval()
out = []
for shard in sorted({p["shard"] for p in plan}):
    pf = pq.ParquetFile(f"{HF}/{shard}")
    starts = np.cumsum([0] + [pf.metadata.row_group(i).num_rows for i in range(pf.metadata.num_row_groups)])
    for p in [q for q in plan if q["shard"] == shard]:
        rg = int(np.searchsorted(starts, p["row"], side="right") - 1)
        t = pf.read_row_group(rg, columns=["session", "seg", f"SPEAKER_{p['lane']}_isolated_npy"]).slice(p["row"] - starts[rg], 1).to_pylist()[0]
        assert f"{t['session']}/{t['seg']}" == p["key"], (t["session"], p["key"])
        C = np.load(io.BytesIO(t[f"SPEAKER_{p['lane']}_isolated_npy"])).astype(np.int64)
        j = p["j"]; lo = max(0, j - CTX)
        with torch.no_grad():
            w = mimi.decode(torch.as_tensor(C[:, lo:j + WIN], device=dev)[None]).audio_values[0, 0].float().cpu().numpy()
        w = np.clip(w[(j - lo) * 1920:(j - lo + WIN) * 1920], -1, 1)
        fn = f"{p['acc'][:8]}_{sum(1 for o in out if o['acc'] == p['acc']) + 1}.mp3"
        sf.write(f"{D}/audio/{fn}", w, 24000, format="MP3"); sf.write(f"{D}/audio/{fn[:-4]}.wav", w, 24000, subtype="PCM_16")
        out.append({**{k: v for k, v in p.items() if k not in ("shard", "row")}, "file": fn, "start_s": round(j * 0.08, 1)})
        print(fn, round(p["own_s"], 1), flush=True)
json.dump(out, open(f"{D}/clips.json", "w"), indent=0)
