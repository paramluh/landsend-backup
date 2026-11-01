"""Daisy in Aloe: the accounts a partner calls 'Daisy' (handover _meta/cut_decisions.jsonl, context_leak_name_before_intro: the partner used
the app's name before any introduction) -> per account: every handover conversation (all buckets), lane, minutes, bucket, FT-train
membership, names heard in their own conversations (episode.json profile.a_speaker); plus own-speech segments (words.json
_separated_speaker = their lane, partner silent within 1 s, 3-10 s, 2 per conversation, up to 8 conversations) decoded from the handover
raw-lane Mimi codes -> aloe/cands.json + aloe/segs/<acc8>_<i>.wav"""
import collections, csv, json, os, re
import numpy as np, soundfile as sf, torch
D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors/daisy/aloe"
DATA = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
STORES = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores"
os.makedirs(f"{D}/segs", exist_ok=True)
csv.field_size_limit(1 << 30)
man = list(csv.DictReader(open(f"{DATA}/manifest.csv")))
# who is called Daisy, and by whom
called = collections.Counter()
for l in open(f"{DATA}/_meta/cut_decisions.jsonl"):
    if not re.search(r"\bdaisy\b", l, re.I):
        continue
    d = json.loads(l)
    pr = (d.get("cut") or {}).get("premise", "")
    m = re.search(r"addresses (S[01])|knows (S[01])'s name|(S[01]) as \"Daisy", pr)
    tgt = None
    for g in re.findall(r"\b(S[01])\b", pr):
        tgt = g
    # the addressee is the lane named after 'addresses' / 'knows', else the one not speaking
    mm = re.search(r"addresses (S[01])|knows (S[01])", pr)
    if mm:
        tgt = mm.group(1) or mm.group(2)
    elif "listener" in pr:
        tgt = None
    for lane, acc in d["account_uids"].items():
        if tgt is None or lane.endswith(tgt[-1]):
            called[acc] += 1
print("called Daisy:", called)
ft = set()
for k in range(4):
    for ln in open(f"{STORES}/m1n{k}/rows.jsonl"):
        r = json.loads(ln)
        if int(r["chunk_index"]) == 0 and r["split"] == "train":
            ft.add(r["src_conversation"])
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformers import MimiModel
mimi = MimiModel.from_pretrained("/data/home/amartyavarma/models/mimi").to(dev).eval()
out = {}
for acc in called:
    convs = []
    for m in man:
        for L in ("00", "01"):
            if m[f"speaker_{L}_account_uid"] == acc:
                convs.append((m, L))
    rec = {"account": acc, "called_daisy": called[acc], "convs": [], "segs": []}
    for m, L in convs:
        try:
            nm = json.load(open(f"{DATA}/{m['relative_dir']}/system_prompts/SPEAKER_{L}/episode.json"))["profile"].get("a_speaker")
        except Exception:
            nm = None
        rec["convs"].append({"key": m["recording_key"], "lane": L, "min": float(m["clip_duration_sec"]) / 60, "bucket": m["relative_dir"].split("/")[1],
                             "ft_train": m["recording_key"] in ft, "name": nm, "partner": m[f"speaker_{'01' if L == '00' else '00'}_account_uid"]})
    for m, L in convs[:8]:
        try:
            W = json.load(open(f"{DATA}/{m['relative_dir']}/w00.words.json"))["words"]
            C = np.load(f"{DATA}/{m['relative_dir']}/w00.SPEAKER_{L}.codes.npy").astype(np.int64)
        except Exception as e:
            continue
        c0 = float(m["clip_start_sec"]); n = C.shape[1]
        own = np.zeros(n, bool); oth = np.zeros(n, bool)
        for w in W:
            a, b = int((w["start"] - c0) / 0.08), int((w["end"] - c0) / 0.08) + 1
            (own if w.get("_separated_speaker") == f"SPEAKER_{L}" else oth)[max(0, a):max(0, b)] = True
        d = np.diff(np.concatenate([[0], own.astype(int), [0]]))
        got = 0
        for s, e in sorted(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)), key=lambda x: x[0] - x[1]):
            e = min(e, s + 125)
            if e - s < 38 or oth[max(0, s - 12):e + 12].any():
                continue
            lo = max(0, s - 13)
            with torch.no_grad():
                w = mimi.decode(torch.as_tensor(C[None, :, lo:e], device=dev)).audio_values[0, 0].float().cpu().numpy()
            w = np.clip(w[(s - lo) * 1920:(e - lo) * 1920], -1, 1)
            fn = f"segs/{acc[:8]}_{len(rec['segs']):02d}.wav"
            sf.write(f"{D}/{fn}", w, 24000, subtype="PCM_16")
            rec["segs"].append({"file": fn, "key": m["recording_key"], "lane": L, "s": round(s * 0.08, 1), "dur": round((e - s) * 0.08, 1)})
            got += 1
            if got == 2:
                break
    names = collections.Counter(c["name"] for c in rec["convs"])
    print(acc[:8], "called", called[acc], "convs", len(rec["convs"]), "min", round(sum(c["min"] for c in rec["convs"]), 1),
          "ft_train", sum(c["ft_train"] for c in rec["convs"]), "buckets", dict(collections.Counter(c["bucket"] for c in rec["convs"])),
          "names", names.most_common(4), "segs", len(rec["segs"]), flush=True)
    out[acc] = rec
json.dump(out, open(f"{D}/cands.json", "w"), indent=0)
