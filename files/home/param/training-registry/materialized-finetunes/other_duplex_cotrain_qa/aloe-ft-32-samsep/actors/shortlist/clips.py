"""Shortlist clips: for the top-12 speakers by all-chunk own speech (+ Ada / 'Daisy' fQ85gun7), 2 excerpts each = 45 s continuous windows of
their lane's SAM-isolated Mimi codes from 2 different first-chunk TRAIN rows (the window with the most of their words), decoded with transformers
MimiModel (1 s left context trimmed) -> shortlist/audio/<rank>_<j>.mp3|wav + shortlist/clips.json"""
import csv, json, os, sys
import numpy as np, soundfile as sf, torch
sys.path.insert(0, "/data/home/amartyavarma/qwen-1ch/stage3/data/rows_build/scripts")
import s3rows as F  # noqa: E402
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/shortlist"
DATA = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
STORES = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores"
os.makedirs(f"{D}/audio", exist_ok=True)
HA = json.load(open(f"{D}/hours_all.json"))["rows"]
pick = [r["account"] for r in HA[:12]]
if "fQ85gun79yZYPSwmn7F4wY1sOd73" not in pick:
    pick.append("fQ85gun79yZYPSwmn7F4wY1sOd73")
csv.field_size_limit(1 << 30)
man = {m["recording_key"]: m for m in csv.DictReader(open(f"{DATA}/manifest.csv"))}
cand = {a: [] for a in pick}
for k in range(4):
    for ln in open(f"{STORES}/m1n{k}/rows.jsonl"):
        r = json.loads(ln)
        if int(r["chunk_index"]) != 0 or r["split"] != "train":
            continue
        m = man[r["src_conversation"]]
        for s in ("assistant", "user"):
            a = m[f"speaker_{r[f'{s}_speaker_dir'][-2:]}_account_uid"]
            if a in cand:
                cand[a].append((k, int(r["row"]), s, r["src_conversation"]))
st = {k: F.Rows(F.load_node(f"{STORES}/m1n{k}")) for k in range(4)}
def mask(ws, n):
    z = np.zeros(n, bool)
    for w in ws:
        z[int(w["start_frame"]):max(int(w["end_frame"]), int(w["start_frame"]) + 1)] = True
    return z
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import MimiModel
mimi = MimiModel.from_pretrained("/data/home/amartyavarma/models/mimi").to(dev).eval()
WIN, CTX = 563, 13
out = {}
for idx, acc in enumerate(pick, 1):
    scored = []
    for k, i, s, conv in cand[acc]:
        rw = st[k][i]; n = int(rw["n_frames"]); a = mask(rw[f"words_{s}"], n)
        cs = np.convolve(a.astype(float), np.ones(WIN), "valid")
        if len(cs):
            scored.append((float(cs.max()), k, i, s, conv, int(np.argmax(cs))))
    scored.sort(reverse=True)
    res = []
    for sc, k, i, s, conv, j in scored[:2]:
        rw = st[k][i]
        lo = max(0, j - CTX)
        c = np.asarray(rw[f"codes_{s}"])[:, lo:j + WIN].astype(np.int64)
        with torch.no_grad():
            w = mimi.decode(torch.as_tensor(c, device=dev)[None]).audio_values[0, 0].float().cpu().numpy()
        w = np.clip(w[(j - lo) * 1920:(j - lo + WIN) * 1920], -1, 1)
        fn = f"s{idx:02d}_{len(res) + 1}.mp3"
        sf.write(f"{D}/audio/{fn}", w, 24000, format="MP3"); sf.write(f"{D}/audio/{fn[:-4]}.wav", w, 24000, subtype="PCM_16")
        words = [x.get("text", "") for x in rw[f"words_{s}"] if j <= int(x["start_frame"]) < j + WIN]
        res.append({"file": fn, "conv": conv, "lane": "S00" if s == "assistant" else "S01", "start_s": round(j / 12.5, 1), "own_s": round(sc / 12.5, 1),
                    "transcript": " ".join(words)})
    out[acc] = res
    print(idx, acc[:8], len(cand[acc]), [r["own_s"] for r in res], flush=True)
json.dump({"pick": pick, "clips": out}, open(f"{D}/clips.json", "w"), indent=0)
