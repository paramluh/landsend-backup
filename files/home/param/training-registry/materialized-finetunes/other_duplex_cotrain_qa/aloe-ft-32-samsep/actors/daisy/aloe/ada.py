"""Daisy = Aloe account fQ85gun79yZYPSwmn7F4wY1sOd73 (a partner calls her Daisy before any introduction; she goes by Ada in her calls).
Stats over every handover conversation of hers and over the sam-sep FT stores; 8 excerpts = 60 s continuous windows of HER lane's SAM-isolated
Mimi codes (the sam-sep store codes the FT trains on), from 8 different conversations, each the window with the most of her own words,
decoded with transformers MimiModel (1 s left context trimmed); transcript = her store word lane in the window -> aloe/ada.json + aloe/audio/*.mp3"""
import collections, csv, json, os, sys
import numpy as np, soundfile as sf, torch
sys.path.insert(0, "/data/home/amartyavarma/qwen-1ch/stage3/data/rows_build/scripts")
import s3rows as F  # noqa: E402
ACC = "fQ85gun79yZYPSwmn7F4wY1sOd73"
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/daisy/aloe"
DATA = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
STORES = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores"
NM = json.load(open("/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/name_map.json"))
os.makedirs(f"{D}/audio", exist_ok=True)
csv.field_size_limit(1 << 30)
man = {m["recording_key"]: m for m in csv.DictReader(open(f"{DATA}/manifest.csv"))}
C = json.load(open(f"{D}/cands.json"))[ACC]
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
    n = int(rw["n_frames"]); o = "user" if s == "assistant" else "assistant"
    a, b = mask(rw[f"words_{s}"], n), mask(rw[f"words_{o}"], n)
    m = man[r["src_conversation"]]; L = r[f"{s}_speaker_dir"][-2:]; P = "01" if L == "00" else "00"
    pacc = m[f"speaker_{P}_account_uid"]
    per.append({"k": k, "row": int(r["row"]), "row_key": r["row_key"], "conv": r["src_conversation"], "split": r["split"], "chunk": int(r["chunk_index"]),
                "side": s, "lane": f"S{L}", "n": n, "own_s": float(a.sum()) / 12.5, "partner_s": float(b.sum()) / 12.5, "partner": pacc,
                "partner_name": (NM.get(pacc) or {}).get("name")})
first_train = [p for p in per if p["chunk"] == 0 and p["split"] == "train"]
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import MimiModel
mimi = MimiModel.from_pretrained("/data/home/amartyavarma/models/mimi").to(dev).eval()
WIN, CTX = 750, 13
ex = []
for p in sorted(first_train, key=lambda p: -p["own_s"])[:8]:
    rw = st[p["k"]][p["row"]]
    a = mask(rw[f"words_{p['side']}"], p["n"])
    cs = np.convolve(a.astype(float), np.ones(WIN), "valid")
    j = int(np.argmax(cs)) if len(cs) else 0
    lo = max(0, j - CTX)
    c = np.asarray(rw[f"codes_{p['side']}"])[:, lo:j + WIN].astype(np.int64)
    with torch.no_grad():
        w = mimi.decode(torch.as_tensor(c, device=dev)[None]).audio_values[0, 0].float().cpu().numpy()
    w = np.clip(w[(j - lo) * 1920:(j - lo + WIN) * 1920], -1, 1)
    fn = f"ada_{len(ex) + 1:02d}.mp3"
    sf.write(f"{D}/audio/{fn}", w, 24000, format="MP3")
    sf.write(f"{D}/audio/{fn[:-4]}.wav", w, 24000, subtype="PCM_16")
    words = [x["text"] if "text" in x else x.get("word", "") for x in rw[f"words_{p['side']}"] if j <= int(x["start_frame"]) < j + WIN]
    ex.append({**p, "file": fn, "start_s": round(j / 12.5, 1), "her_s": round(float(a[j:j + WIN].sum()) / 12.5, 1), "transcript": " ".join(words)})
    print(fn, p["conv"][:8], p["lane"], ex[-1]["her_s"], flush=True)
json.dump({"account": ACC, "convs": C["convs"], "rows": per, "excerpts": ex}, open(f"{D}/ada.json", "w"), indent=0)
print("rows", len(per), "first-chunk train", len(first_train), "own h (FT train)", round(sum(p["own_s"] for p in first_train) / 3600, 2))
