"""Aloe pose-duplex duo samples page (pdsamples) = the PPM 2-side page (samples/2side-local-10386-v2), data loading adapted to voxsamples runs.
two-sided PPM samples page (copy of setup-logs/klweb/s3sys/art-aloe-minp/build_art.py's data model): per sample, versions Model / Real
recording, each with a mono MP3 per speaker (track 0 = speaker 1, track 1 = speaker 2), waveform peaks and the word lanes; system prompt
cards (persona text, voice clip) per speaker -> data.json + a/*.mp3 + files.json next to index.html.
+ pose (aloe-pose-ft-430 build_page.py video lane): runs with POSE_DONE (code/render_pose.py) get per-speaker skeleton videos per version
  (Model = pose_gen_s{1,2}.mp4, Real recording = pose_gt_s{1,2}.mp4) copied to a/<run>-<gen|gt>-pose-s{1,2}.mp4; runs without POSE_DONE: no videos.
   python build_page.py [runs dir]"""
import json, os, sys, glob, shutil
import numpy as np, soundfile as sf

OUT = os.path.dirname(os.path.abspath(__file__)); os.makedirs(f"{OUT}/a", exist_ok=True)
RUNS = sys.argv[1] if len(sys.argv) > 1 else "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples/runs"
NTOT = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SR = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/samples"
EXTRA = [] and [("k500", "Pose top-k 500, cb0 + light (both sides generated)", f"{SR}/runs-k500"),
         ("cb0k500", "Pose cb0 top-k 500, light 30 (both sides generated)", f"{SR}/runs-cb0k500")]
NB = 2400


def peaks(y):
    n = max(len(y) // NB * NB, NB)
    y2 = np.zeros(n, np.float32); y2[:min(n, len(y))] = y[:n]
    pk = np.abs(y2).reshape(NB, -1).max(1)
    return [round(float(v), 3) for v in pk / max(1e-6, float(pk.max()))]


def vid(src, fn):
    shutil.copyfile(src, f"{OUT}/{fn}.tmp"); os.replace(f"{OUT}/{fn}.tmp", f"{OUT}/{fn}")
    return fn


def mp3(y, sr, fn):
    sf.write(f"{OUT}/{fn}", np.clip(y, -1, 1), sr, format="MP3", subtype="MPEG_LAYER_III")
    return fn


convs, files = [], {"data.json": "data.json"}
meta = {"ckpts": []}
for d in sorted(glob.glob(f"{RUNS}/B-*/DONE")):
    rd = os.path.dirname(d); nm = os.path.basename(rd)
    M = json.load(open(f"{rd}/meta.json")); W = json.load(open(f"{rd}/words.json"))
    step = int(M["step"])
    if not meta["ckpts"]:
        meta["ckpts"].append({"tag": f"local-{step}", "step": step, "vox_step": M.get("vox_step")})
    side1 = M["prov"]["chan_B_data_side"]; side2 = "user" if side1 == "assistant" else "assistant"
    roles = {"A": f"speaker 1 ({side1} side)", "B": f"speaker 2 ({side2} side)"}
    vers = []
    pose = os.path.isfile(f"{rd}/POSE_DONE") and all(os.path.isfile(f"{rd}/pose_{k}_{s}.mp4") for k in ("gen", "gt") for s in ("s1", "s2"))
    vlist = [("gen", "Pose-duplex FT, pose top-k 30 (both sides generated)" if any(os.path.isfile(f"{x}/{nm}/POSE_DONE") for _, _, x in EXTRA)
              else "Pose-duplex FT (both sides generated)", "", W, rd, "gen")]
    for tag, lab, xd in EXTRA:                                       # pose top-k A/B arms (main 10-25): same conversation + seeds, own run dir
        if os.path.isfile(f"{xd}/{nm}/DONE") and os.path.isfile(f"{xd}/{nm}/POSE_DONE"):
            vlist.append(("gen", lab, "", json.load(open(f"{xd}/{nm}/words.json")), f"{xd}/{nm}", tag))
    vlist.append(("gt", "Real recording", "gt_", W["gt"], rd, "gt"))
    for kind, label, pre, ws, vd, ft in vlist:
        v = {"label": label, "tracks": []}
        if pose:
            v["video"] = {}
            for C, spk in (("A", "s1"), ("B", "s2")):
                vf = vid(f"{vd}/pose_{kind}_{spk}.mp4", f"a/{nm}-{ft}-pose-{spk}.mp4"); files[vf] = vf
                v["video"][C] = {"src": vf, "cap": "real pose in the prefill, generated after" if kind == "gen" else "real pose"}
        for C, spk in (("A", "s1"), ("B", "s2")):
            x, sr = sf.read(f"{vd}/{pre}{spk}.wav", dtype="float32")
            v["dur"] = len(x) / sr
            fn = mp3(x, sr, f"a/{nm}-{ft}-{spk}.mp3"); files[fn] = fn
            v["tracks"].append({"ch": C, "role": roles[C], "src": fn, "peaks": peaks(x),
                                "words": [[round(w["t"], 2), w["text"], bool(w["prompt"])] for w in ws[spk]]})
        vers.append(v)
    cards = {}
    for C, spk in (("A", "s1"), ("B", "s2")):
        c = dict(M["cards"][spk]); clip = None
        if c["text"].startswith("system"):                               # runs made before infer_ppm_2side stripped the chat header
            c["text"] = c["text"][len("system"):].strip()
        if c["clip"]:
            y, sr = sf.read(f"{rd}/{c['clip']['wav']}", dtype="float32")
            fn = mp3(y, sr, f"a/{nm}-clip-{spk}.mp3"); files[fn] = fn
            clip = {"src": fn, "seconds": c["clip"]["seconds"]}
        cards[C] = {"role": roles[C], "text": c["text"] and {"text": c["text"], "n_tokens": c["n_tokens"]}, "clip": clip}
    smp = M["sampling"] | {"line_delay_ms": M.get("line_delay_ms")}
    pers = "persona" if any(M["cards"][s]["text"].removeprefix("system").strip() for s in ("s1", "s2")) else ""
    clips = "voice clips" if any(M["cards"][s]["clip"] for s in ("s1", "s2")) else ""
    title = f"{nm} · " + ((clips + " only") if clips and not pers else " + ".join(x for x in (pers, clips) if x) or "no system prompt")
    convs.append({"name": nm, "title": title, "ck": {f"local-{step}": {"step": step, "sampling": smp, "versions": vers, "cards": cards, "score": None}},
                  "mode": "continue", "np_s": M["prompt_s"], "ng_s": M["gen_s"], "row_key": M["prov"]["row_key"],
                  "order": [{"kind": o["kind"], "ch": "A" if o["spk"] == "s1" else "B"} for o in M["order"]],
                  "stats": {"words": M["gen_words"], "audio": {k: M["audio"][k] for k in ("s1_gen", "s2_gen")}}})
meta["n_done"], meta["n_total"] = len(convs), NTOT
json.dump({"meta": meta, "convs": convs}, open(f"{OUT}/data.json", "w"), separators=(",", ":"))
json.dump(files, open(f"{OUT}/files.json", "w"))
print(len(convs), "examples;", round(sum(os.path.getsize(f"{OUT}/{f}") for f in files) / 1e6, 1), "MB")
