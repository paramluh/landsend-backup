#!/usr/bin/env python3
"""posefix NAME-B (10-27; user: "why is the name lauren at 90% chance"): the duplex training data side.
(1) name stats over the s2h-m1-samsep training conversations (both lanes = possible B sides; the FT trains both directions):
    self-intros per side (strict "my name is X" / loose "I'm X"), distinct ACTORS per intro name, raw word counts of Lauren / Rosie / Kate,
    and the share among sides introducing with a female name (name-list gender proxy; no gender metadata exists in the manifest);
(2) WeSpeaker (voxceleb resnet221-LM, the samsep handoff copy) embeddings of 10 s SAM-code windows (Mimi-decoded, 16 kHz) for every actor
    that self-introduces as Lauren / Laura / Rosie, + a seeded random sample of other actors; cosine to Kate (actor 06 = 74807fe0: her live
    voice clip + 3 other Kate windows, mean embedding); the 20 nearest actors with the names they use.
-> /data/home/amartyavarma/ppm/aloe-posedup-live/posefix/name/names_spk.json"""
import csv, glob, json, os, random, re, subprocess, sys
from collections import Counter, defaultdict
import numpy as np

D = "/data/home/amartyavarma/data/aloe-handover-full/DATA"
ST = "/data/home/amartyavarma/qwen-1ch-data/s2h-m1-samsep/stores"
H = "/data/home/amartyavarma/ppm/sam-sep-handoff"
OUT = "/data/home/amartyavarma/ppm/aloe-posedup-live/posefix/name"
A06 = "74807fe0-d903-4674-8ec0-a2355bc17cb9"
N_RAND = int(os.environ.get("N_RAND", "120"))
STRICT = re.compile(r"\b(?i:my name is|my name's|name is|call me)\s+([A-Z][a-zA-Z'-]+)")
LOOSE = re.compile(r"\b(?i:i'm|i am|it's|this is)\s+([A-Z][a-zA-Z'-]+)")
STOP = set("I Okay OK So Yeah Yes No Not Good Great Fine Sorry Just Really From English British American Doing Well Here Going Very Like The "
           "A An And But Oh Um Uh Ah Hi Hey Hello Sure Also Actually Pretty Quite So-so Alright All Nvidia".split())
FEMALE = set("Lauren Laura Laurie Rosie Rose Kate Katie Kathy Sarah Sara Ella Anna Aisha Mary Emma Emily Amy Jessica Jess Hannah Sophie Sophia "
             "Olivia Grace Lucy Charlotte Chloe Megan Rachel Rebecca Becky Helen Holly Zoe Ruby Lily Amelia Isabella Isabel Mia Maria Alice "
             "Jennifer Jenny Lisa Linda Karen Susan Nicole Michelle Natalie Victoria Abigail Elizabeth Liz Beth Ellie Georgia Jade Leah Molly "
             "Naomi Fatima Amina Ayesha Shamim Sasha Mandy Alina Kelly Kim Tanya Tina Nina Rosa Julia Julie Claire Clara Fiona Gemma Joanna "
             "Jo Katherine Catherine Caroline Danielle Diana Eva Eve Freya Harriet Imogen Iris Jasmine Kayla Lara Lena Maya Nadia Phoebe "
             "Priya Rose Samantha Sienna Stella Tara Tess Vanessa Yasmin Zara Ashley".split())

conv = {}       # (session, seg) -> list of (node, row, lane_dir_assistant, chunk_index)
for f in sorted(glob.glob(f"{ST}/m1n*/rows.jsonl")):
    node = f.split("/")[-2]
    for l in open(f):
        r = json.loads(l)
        p = r["row_key"].split("/")
        conv.setdefault((p[2], p[3]), []).append((node, int(r["row"]), r["assistant_speaker_dir"], int(r["chunk_index"]), r.get("split")))
man = {(r["session_id"], r["seg"]): r for r in csv.DictReader(open(f"{D}/manifest.csv"))}


def words_of(key):
    r = man.get(key)
    if not r:
        return None
    w = json.load(open(f"{D}/{r['wlt_words_path']}"))
    w = w["words"] if isinstance(w, dict) else w
    out = {"SPEAKER_00": [], "SPEAKER_01": []}
    for x in w:
        s = x.get("_separated_speaker") or x.get("speaker")
        if s in out:
            out[s].append((float(x["start"]), str(x.get("word", x.get("text", ""))).strip()))
    return out


def intros(ws):
    txt, pos = "", []
    for _, t in ws:
        pos.append(len(txt)); txt += t + " "
    res = []
    for kind, rx in (("strict", STRICT), ("loose", LOOSE)):
        for m in rx.finditer(txt):
            nm = m.group(1).strip(".,!?'\"")
            if nm in STOP or len(nm) < 2:
                continue
            wi = max(i for i, p in enumerate(pos) if p <= m.start(1))
            res.append((kind, nm, ws[wi][0]))
    return res


side_names, actor_names, actor_sides = {}, defaultdict(Counter), defaultdict(list)
raw = Counter()
for k in sorted(conv):
    w = words_of(k)
    if w is None:
        continue
    r = man[k]
    for spk in ("SPEAKER_00", "SPEAKER_01"):
        a = r[f"speaker_{spk[-2:]}_actor_id"]
        it = intros(w[spk])
        side_names[(k, spk)] = it
        actor_sides[a].append((k, spk, w[spk]))
        for _, n, _ in it:
            actor_names[a][n] += 1
        for _, t in w[spk]:
            t2 = t.strip(".,!?'\"")
            if t2 in ("Lauren", "Rosie", "Kate", "Laura"):
                raw[t2] += 1
strict_sides, loose_sides = Counter(), Counter()
fem_strict = 0
for (k, spk), it in side_names.items():
    s = {n for kind, n, _ in it if kind == "strict"}
    lz = {n for _, n, _ in it}
    for n in s:
        strict_sides[n] += 1
    for n in lz:
        loose_sides[n] += 1
    if s & FEMALE:
        fem_strict += 1
actors_by_name = defaultdict(set)
for a, c in actor_names.items():
    for n in c:
        actors_by_name[n].add(a)
fem_total = sum(c for n, c in strict_sides.items() if n in FEMALE)
res = {"n_conversations": len({k for k, _ in side_names}), "n_sides": len(side_names), "n_actors": len(actor_sides),
       "raw_word_counts": dict(raw),
       "top30_strict_sides": strict_sides.most_common(30), "top30_loose_sides": loose_sides.most_common(30),
       "target": {n: {"strict_sides": strict_sides.get(n, 0), "loose_sides": loose_sides.get(n, 0), "actors": len(actors_by_name.get(n, ()))}
                  for n in ("Lauren", "Laura", "Rosie", "Kate")},
       "female_name_strict_sides": fem_strict,
       "share_of_female_strict_intros": {n: round(strict_sides.get(n, 0) / max(fem_total, 1), 4) for n in ("Lauren", "Laura", "Rosie", "Kate")},
       "kate06_names": dict(actor_names.get(A06, {}))}
print(json.dumps({k: v for k, v in res.items() if k not in ("top30_loose_sides",)}), flush=True)

# ---------------- embeddings: stage "decode" (venv with transformers: Mimi decode -> wav) then stage "embed" (venv with wespeaker)
STAGE = os.environ.get("STAGE", "decode")
FF = f"{H}/env/ffmpeg/wrap/ffmpeg"
TMP = os.path.join(OUT, "wav"); os.makedirs(TMP, exist_ok=True)
if STAGE == "decode":
    import torch
    from transformers import MimiModel
    torch.set_num_threads(int(os.environ.get("NT", "4")))
    mimi = MimiModel.from_pretrained(f"{H}/mimi_hub/models--kyutai--mimi/snapshots/89091b3e466eb6a9d11e537bf26b144f194978f7").eval()
else:
    import wespeaker
    os.environ.setdefault("WESPEAKER_HOME", f"{H}/wespeaker")
    spk_model = wespeaker.load_model("english"); spk_model.set_device("cpu"); spk_model.set_vad(False); spk_model.set_resample_rate(16000)
CODES = {}


def codes(node, side):
    if (node, side) not in CODES:
        CODES[(node, side)] = np.load(f"{ST}/{node}/rows-00000-of-00001.codes_{side}.npy", mmap_mode="r")
    return CODES[(node, side)]


def window_for(actor):
    """a 10 s (125-frame) window of this actor's SAM lane where it says the most words (any of its conversations)."""
    best = None
    for k, spk, ws in actor_sides[actor]:
        ts = np.asarray([t for t, _ in ws])
        if len(ts) < 10:
            continue
        for t0 in np.arange(0, ts.max(), 5.0):
            n = int(((ts >= t0) & (ts < t0 + 10)).sum())
            if best is None or n > best[0]:
                best = (n, k, spk, float(t0))
    if best is None:
        return None
    _, k, spk, t0 = best
    f0 = int(round(t0 * 12.5))
    for node, row, adir, ci, _ in conv[k]:
        if 7500 * ci <= f0 < 7500 * (ci + 1) - 125:
            side = "assistant" if adir == spk else "user"
            a = np.asarray(codes(node, side)[row])
            a = a if a.shape[0] == 32 else a.T
            c = a[:, f0 - 7500 * ci: f0 - 7500 * ci + 125]
            if c.min() >= 0:
                return c, f"{k[0][:8]}/{k[1]}/{spk}@{t0:.0f}s"
    return None


def decode_codes(c, name):
    with torch.no_grad():
        x = mimi.decode(torch.from_numpy(np.asarray(c, np.int64))[None]).audio_values[0, 0].numpy()
    p = f"{TMP}/{name}.wav"
    subprocess.run([FF, "-y", "-loglevel", "error", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "-", "-ar", "16000", p],
                   input=(np.clip(x, -1, 1) * 32767).astype("<i2").tobytes(), check=True)
    return p


def embed_wav(p):
    e = np.asarray(spk_model.extract_embedding(p), np.float64).reshape(-1)
    return e / np.linalg.norm(e)


targets = set()
for n in ("Lauren", "Laura", "Rosie"):
    targets |= actors_by_name.get(n, set())
others = sorted(set(actor_sides) - targets - {A06})
random.Random(0).shuffle(others)
todo = [(a, "target") for a in sorted(targets)] + [(a, "random") for a in others[:N_RAND]] + [(A06, "kate06")]
if STAGE == "decode":
    done = []
    for i, (a, why) in enumerate(todo):
        w = window_for(a)
        if w is None:
            continue
        done.append({"actor": a, "why": why, "window": w[1], "wav": decode_codes(w[0], f"{why}_{a[:8]}")})
        if i % 20 == 0:
            print(f"decoded {i + 1}/{len(todo)}", flush=True)
    json.dump({"res": res, "done": done}, open(f"{OUT}/decoded.json", "w"))
    print("DECODED", len(done), flush=True)
    sys.exit(0)
done = json.load(open(f"{OUT}/decoded.json"))["done"]


# Kate reference: her live voice clip (handoff A_kate_voiceclip_sam.wav) + 3 other Kate windows (handoff R_kate_*)
kref = []
for f in sorted(glob.glob(f"{H}/work/spk/A_kate_voiceclip_sam.wav") + glob.glob(f"{H}/work/spk/R_kate_*.wav")):
    e = np.asarray(spk_model.extract_embedding(f), np.float64).reshape(-1)
    kref.append(e / np.linalg.norm(e))
K = np.mean(kref, 0); K /= np.linalg.norm(K)
emb = []
for i, d in enumerate(done):
    e = embed_wav(d["wav"])
    a = d["actor"]
    emb.append({"actor": a, "why": d["why"], "window": d["window"], "cos_kate": round(float(e @ K), 3),
                "names": actor_names.get(a, Counter()).most_common(4), "n_sides": len(actor_sides[a])})
    if i % 20 == 0:
        print(f"embedded {i + 1}/{len(done)}", flush=True)
emb.sort(key=lambda r: -r["cos_kate"])
res["kate_ref_self_cos"] = [round(float(e @ K), 3) for e in kref]
res["embedded"] = len(emb)
res["nearest20"] = [r for r in emb if r["why"] != "kate06"][:20]
res["kate06_window_cos"] = [r["cos_kate"] for r in emb if r["why"] == "kate06"]
for n in ("Lauren", "Rosie", "Laura"):
    rs = [r for r in emb if any(nm == n for nm, _ in r["names"])]
    rank = {r["actor"]: i + 1 for i, r in enumerate([x for x in emb if x["why"] != "kate06"])}
    res[f"{n}_actors_cos"] = [(r["actor"][:8], r["cos_kate"], rank.get(r["actor"])) for r in rs]
rnd = [r["cos_kate"] for r in emb if r["why"] == "random"]
res["random_actor_cos"] = {"n": len(rnd), "p50": round(float(np.median(rnd)), 3) if rnd else None,
                           "p90": round(float(np.percentile(rnd, 90)), 3) if rnd else None, "max": round(float(max(rnd)), 3) if rnd else None}
json.dump(res, open(f"{OUT}/names_spk.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("kate_ref_self_cos", "embedded", "random_actor_cos", "kate06_window_cos", "Lauren_actors_cos",
                                       "Rosie_actors_cos", "Laura_actors_cos")}), flush=True)
for r in res["nearest20"]:
    print("NEAR", r["cos_kate"], r["why"], r["actor"][:8], r["names"], r["window"], flush=True)
print("DONE", flush=True)
