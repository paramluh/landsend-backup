"""Stage B of the per-actor voice clips: ECAPA speaker check + assembly.
Every candidate segment (actors/segs/segs_*.json, from decode_segs.py) gets an ECAPA embedding (speechbrain spkrec-ecapa-voxceleb, local copy
under data/voicebank-yodas3/models, 16 kHz). Per actor: the voice centroid = mean of the 60 % of segments closest to the running centroid
(5 rounds, starting from the mean of all); a segment is kept if its cosine to the centroid is >= THR. The clip = kept segments in the
fixed conversation order, at most 2 per conversation, until >= 60 s, joined with 0.4 s of silence.
usage: python -I voice_select.py [THR] -> actors/audio_v3/r<rank>.mp3 + actors/audio_v3/clips.json (+ actors/segs/emb.npz cache)"""
import json
import glob
import os
import sys

import numpy as np
import soundfile as sf
import torch
import torchaudio

D = "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/actors"
MODELS = "/data/home/amartyavarma/data/voicebank-yodas3/models"
SR, GAP, TARGET = 24000, 0.4, 60.0
TAG = os.environ.get("ACT_TAG", "")


def main():
    thr = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    segs = []
    for fj in sorted(glob.glob(f"{D}/segs{TAG}/segs_*.json")):
        segs += json.load(open(fj))
    cache = f"{D}/segs{TAG}/emb.npz"
    if os.path.exists(cache) and int(np.load(cache)["n"]) == len(segs):
        E = np.load(cache)["E"]
    else:
        from speechbrain.inference.speaker import EncoderClassifier
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        ec = EncoderClassifier.from_hparams(source=f"{MODELS}/spkrec-ecapa-voxceleb", savedir=f"{D}/segs{TAG}/ecapa", run_opts={"device": dev})
        E = []
        for s in segs:
            w, sr = sf.read(f"{D}/{s['file']}", dtype="float32")
            x = torchaudio.functional.resample(torch.from_numpy(w)[None], sr, 16000).to(dev)
            with torch.no_grad():
                E.append(ec.encode_batch(x)[0, 0].float().cpu().numpy())
        E = np.stack(E)
        np.savez(cache, E=E, n=len(segs))
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    os.makedirs(f"{D}/audio_v3{TAG}", exist_ok=True)
    by = {}
    for i, s in enumerate(segs):
        by.setdefault(s["rank"], []).append(i)
    out = []
    for rank in sorted(by):
        idx = np.array(by[rank])
        X = E[idx]
        c = X.mean(0); c /= np.linalg.norm(c)
        for _ in range(5):
            sim = X @ c
            keep = sim >= np.quantile(sim, 0.4)
            c = X[keep].mean(0); c /= np.linalg.norm(c)
        sim = X @ c
        conv_sim = {}
        for i, v in zip(idx, sim):
            conv_sim.setdefault(segs[i]["row_key"], []).append(float(v))
        bad_convs = [k for k, v in conv_sim.items() if np.mean(v) < thr]
        pieces, used, got = [], [], 0.0
        for i, v in zip(idx, sim):
            if v < thr:
                continue
            s = segs[i]
            w, _ = sf.read(f"{D}/{s['file']}", dtype="float32")
            pieces += [w, np.zeros(int(GAP * SR), np.float32)]
            used.append({**s, "sim": round(float(v), 3)})
            got += len(w) / SR
            if got >= TARGET:
                break
        fn = f"r{rank:02d}.mp3"
        if pieces:
            sf.write(f"{D}/audio_v3{TAG}/{fn}", np.concatenate(pieces[:-1]), SR, format="MP3")
        lanes = {}
        for u in used:
            lanes.setdefault(u["row_key"], u["lane"])
        rec = {"rank": rank, "actor_id": segs[idx[0]]["actor_id"], "file": fn if pieces else None, "speech_s": round(got, 1),
               "n_segments": len(used), "n_convs": len(lanes), "lane_convs": {L: sum(1 for v in lanes.values() if v == L) for L in ("SPEAKER_00", "SPEAKER_01")},
               "cand_segments": int(len(idx)), "cand_convs": len(conv_sim), "dropped_segments": int((sim < thr).sum()), "outlier_convs": len(bad_convs),
               "sim_median": round(float(np.median(sim)), 3), "sim_p10": round(float(np.quantile(sim, 0.1)), 3), "segments": used}
        out.append(rec)
        print(f"rank {rank:2d} cand {len(idx):3d} segs/{len(conv_sim):2d} convs  sim med {np.median(sim):.2f} p10 {np.quantile(sim, .1):.2f} "
              f"min {sim.min():.2f}  dropped {int((sim < thr).sum()):2d} segs, outlier convs {len(bad_convs):2d}  clip {got:.0f}s from {len(lanes)} convs",
              flush=True)
    json.dump({"thr": thr, "clips": out}, open(f"{D}/audio_v3{TAG}/clips.json", "w"), indent=0)


if __name__ == "__main__":
    main()
