"""40 s excerpts of each extracted Daisy row with BOTH sides: the window with the most clean back-and-forth (frames where only one lane
has words: min(Daisy-only, partner-only) + 0.3 x both - frames where both lanes have words, which are stem bleed); echoed words dropped from the transcript, decoded from the row's Mimi codes (both lanes, transformers MimiModel /data/home/amartyavarma/models/mimi, 1 s left
context trimmed), stereo with Daisy panned left (0.85 / 0.15) and the partner right -> daisy/audio/<tag>.mp3 + <tag>.json. usage: excerpts.py FROM TO (slice of the sorted rows)"""
import glob, json, os, sys
import numpy as np, soundfile as sf, torch
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/daisy"
SR, HOP, WIN, CTX = 24000, 1920, 500, 13
os.makedirs(f"{D}/audio", exist_ok=True)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)
from transformers import MimiModel
mimi = MimiModel.from_pretrained("/data/home/amartyavarma/models/mimi").to(dev).eval()
def mask(ws, n):
    z = np.zeros(n, bool)
    for w in ws:
        z[int(w["start_frame"]):max(int(w["end_frame"]), int(w["start_frame"]) + 1)] = True
    return z
out = []
for f in sorted(glob.glob(f"{D}/rows/*.json"))[int(sys.argv[1]):int(sys.argv[2])]:
    m = json.load(open(f)); tag = os.path.basename(f)[:-5]
    z = np.load(f"{D}/rows/{tag}.npz"); n = m["n_frames"]
    me, ot = m["side"], ("user" if m["side"] == "assistant" else "assistant")
    a, b = mask(m[f"words_{me}"], n), mask(m[f"words_{ot}"], n)
    ao, bo, ab = a & ~b, b & ~a, a & b                   # the word lanes echo each other where the stems bleed: score clean turns only
    ca, cb, cab = (np.convolve(x.astype(float), np.ones(WIN), "valid") for x in (ao, bo, ab))
    j = int(np.argmax(np.minimum(ca, cb) + 0.3 * (ca + cb) - cab))
    lo = max(0, j - CTX)
    lanes = {}
    for s in (me, ot):
        c = torch.as_tensor(z[f"codes_{s}"][:, lo:j + WIN].astype(np.int64), device=dev)[None]
        with torch.no_grad():
            w = mimi.decode(c).audio_values[0, 0].float().cpu().numpy()
        lanes[s] = np.clip(w[(j - lo) * HOP:(j - lo + WIN) * HOP], -1, 1)
    st = np.stack([0.85 * lanes[me] + 0.15 * lanes[ot], 0.15 * lanes[me] + 0.85 * lanes[ot]], 1)
    sf.write(f"{D}/audio/{tag}.mp3", np.clip(st, -1, 1), SR, format="MP3")
    W = {s: [(w["start_frame"], (w.get("text") or "").strip()) for w in m[f"words_{s}"] if j <= w["start_frame"] < j + WIN] for s in (me, ot)}
    key = lambda x: x[1].lower().strip(".,?!")  # noqa: E731
    echo = {(sf_, key((sf_, t))) for sf_, t in W[me] if any(abs(sf_ - s2) <= 4 and key((s2, t2)) == key((sf_, t)) for s2, t2 in W[ot])}
    dup = lambda f0, t: any(abs(f0 - e0) <= 4 and k == key((f0, t)) for e0, k in echo)  # noqa: E731
    ws = sorted([(f0, "D", t) for f0, t in W[me] if not dup(f0, t)] + [(f0, "P", t) for f0, t in W[ot] if not dup(f0, t)])
    turns = []
    for _, who, wd in ws:
        if turns and turns[-1][0] == who:
            turns[-1][1].append(wd)
        else:
            turns.append([who, [wd]])
    out.append({**{k: m[k] for k in ("ek", "side", "chunk", "partner_name", "who", "node", "row_key")}, "file": f"{tag}.mp3",
                "start_s": round(j / 12.5, 1), "daisy_s": round(float(ao[j:j + WIN].sum()) / 12.5, 1), "partner_s": round(float(bo[j:j + WIN].sum()) / 12.5, 1), "both_s": round(float(ab[j:j + WIN].sum()) / 12.5, 1),
                "turns": [[w, " ".join(x).strip()] for w, x in turns]})
    json.dump(out[-1], open(f"{D}/audio/{tag}.json", "w"), indent=0)
    print(tag, j, out[-1]["daisy_s"], out[-1]["partner_s"], flush=True)
