"""One speaker's page data. usage: speaker_data.py ACCOUNT SLUG
Rows: every sam-sep store row (all chunks, all splits) with the account on a lane; 8 excerpts = 60 s continuous windows of HER lane's
SAM-isolated Mimi codes from 8 different first-chunk TRAIN rows (the minute with the most of her own words), decoded with transformers
MimiModel (1 s left context trimmed); transcript = her store word lane in the window -> speakers/<slug>/data.json + audio/*.mp3|wav"""
import json, math, os, sys
import csv
import numpy as np, soundfile as sf, torch
sys.path.insert(0, "/data/home/amartyavarma/qwen-1ch/stage3/data/rows_build/scripts")
import s3rows as F  # noqa: E402
ACC, SLUG = sys.argv[1], sys.argv[2]
D = f"/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/speakers/{SLUG}"
DATA = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
STORES = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores"
NM = json.load(open("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/name_map.json"))
os.makedirs(f"{D}/audio", exist_ok=True)
csv.field_size_limit(1 << 30)
man = {m["recording_key"]: m for m in csv.DictReader(open(f"{DATA}/manifest.csv"))}
HA = {r["account"]: r for r in json.load(open("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/shortlist/hours_all.json"))["rows"]}
calls = [m for m in man.values() for L in ("00", "01") if m[f"speaker_{L}_account_uid"] == ACC and m["relative_dir"].split("/")[1] == "none"]
chunks_if_all = sum(math.ceil(float(m["clip_duration_sec"]) / 600) for m in calls if m["recording_key"] in set(HA[ACC]["keys"]))
rows = []
for k in range(4):
    for ln in open(f"{STORES}/m1n{k}/rows.jsonl"):
        r = json.loads(ln)
        m = man[r["src_conversation"]]
        for s in ("assistant", "user"):
            if m[f"speaker_{r[f'{s}_speaker_dir'][-2:]}_account_uid"] == ACC:
                rows.append((k, r, s))
st = {k: F.Rows(F.load_node(f"{STORES}/m1n{k}")) for k in range(4)}
def mask(ws, n):
    z = np.zeros(n, bool)
    for w in ws:
        z[int(w["start_frame"]):max(int(w["end_frame"]), int(w["start_frame"]) + 1)] = True
    return z
per = []
for k, r, s in rows:
    rw = st[k][int(r["row"])]
    assert rw["row_key"] == r["row_key"]
    n = int(rw["n_frames"]); a = mask(rw[f"words_{s}"], n)
    m = man[r["src_conversation"]]; L = r[f"{s}_speaker_dir"][-2:]; pacc = m[f"speaker_{'01' if L == '00' else '00'}_account_uid"]
    per.append({"k": k, "row": int(r["row"]), "row_key": r["row_key"], "conv": r["src_conversation"], "split": r["split"], "chunk": int(r["chunk_index"]),
                "side": s, "lane": f"S{L}", "n": n, "own_s": float(a.sum()) / 12.5, "partner": pacc, "partner_name": (NM.get(pacc) or {}).get("name")})
first_train = [p for p in per if p["chunk"] == 0 and p["split"] == "train"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import MimiModel
mimi = MimiModel.from_pretrained("/data/home/amartyavarma/models/mimi").to(dev).eval()
WIN, CTX = 750, 13
ex, seen = [], set()
for p in sorted(first_train, key=lambda p: -p["own_s"]):
    if p["conv"] in seen:
        continue
    seen.add(p["conv"])
    rw = st[p["k"]][p["row"]]
    a = mask(rw[f"words_{p['side']}"], p["n"])
    cs = np.convolve(a.astype(float), np.ones(WIN), "valid")
    j = int(np.argmax(cs)) if len(cs) else 0
    lo = max(0, j - CTX)
    c = np.asarray(rw[f"codes_{p['side']}"])[:, lo:j + WIN].astype(np.int64)
    with torch.no_grad():
        w = mimi.decode(torch.as_tensor(c, device=dev)[None]).audio_values[0, 0].float().cpu().numpy()
    w = np.clip(w[(j - lo) * 1920:(j - lo + WIN) * 1920], -1, 1)
    fn = f"{SLUG}_{len(ex) + 1:02d}.mp3"
    sf.write(f"{D}/audio/{fn}", w, 24000, format="MP3")
    sf.write(f"{D}/audio/{fn[:-4]}.wav", w, 24000, subtype="PCM_16")
    words = [x.get("text", "") for x in rw[f"words_{p['side']}"] if j <= int(x["start_frame"]) < j + WIN]
    ex.append({**p, "file": fn, "start_s": round(j / 12.5, 1), "her_s": round(float(a[j:j + WIN].sum()) / 12.5, 1), "transcript": " ".join(words)})
    if len(ex) == 8:
        break
json.dump({"account": ACC, "slug": SLUG, "all": {k: v for k, v in HA[ACC].items() if k != "keys"}, "chunks_if_all": chunks_if_all,
           "rows": per, "excerpts": ex}, open(f"{D}/data.json", "w"), indent=0)
print("rows", len(per), "first-chunk train", len(first_train), "excerpts", len(ex), "chunks_if_all", chunks_if_all)
