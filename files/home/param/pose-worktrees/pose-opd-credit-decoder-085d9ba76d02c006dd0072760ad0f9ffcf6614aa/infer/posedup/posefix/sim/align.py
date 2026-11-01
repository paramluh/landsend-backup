#!/usr/bin/env python3
"""posefix WORD ALIGNMENT with our own model (no ASR): the person's real 12 s (gt/mimi_local audio + gt/pose_tokens pose, voice clip =
gt/mimi_local[:, :125] as the live preset) teacher-forced through the MONOPLEX (export-357); B's word lane = the known transcript
(meta.json 'text') placed by DP on the model's own lane log-probs: score(lane) = sum_f log p_f(lane[f]) with the training placement
(NEW_WORD on the PAD frame before a word, its pieces on consecutive frames, PAD elsewhere). p_f = text head on hn[P + f - 1] of a teacher-
forced pass over the current lane guess (the model is causal, so the lane history matters): iterate (uniform guess -> DP -> re-run) until
the starts stop changing. Writes <out>/<person>.words.json ([[start_s, start_s, text], ...]) + the prefill bundle <out>/<person>.npz
(= make_prefill.py's format) and prints the counts to check against demo-data/prefill_words.json.
usage: python align.py <ck> <person> <out dir> [iters]"""
import json
import os
import sys

MI = "/data/home/amartyavarma/ppm/aloe-pose-monoplex-infer"
sys.path.insert(0, MI)
sys.path.insert(0, f"{MI}/ppm")
os.environ.setdefault("PPM_CD", "2")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import infer as MIF  # noqa: E402
import fast_ppm as FP  # noqa: E402
import ppm_io as IO  # noqa: E402
from safetensors.numpy import load_file  # noqa: E402

CD, OFF, PAD, NW = FP.CD, FP.OFF, FP.PAD, IO.SP.NW_ID
ck, person, out = sys.argv[1], sys.argv[2], sys.argv[3]
ITERS = int(sys.argv[4]) if len(sys.argv) > 4 else 4
os.makedirs(out, exist_ok=True)
pk = f"/data/home/amartyavarma/video/demo-data/prompts/{person}"
meta = json.load(open(f"{pk}/meta.json"))
audio = np.asarray(load_file(f"{pk}/gt/mimi_local.safetensors")["codes"], np.int64)
pose = np.asarray(load_file(f"{pk}/gt/pose_tokens.safetensors")["codes"], np.int64)
n = audio.shape[1]                                                 # 150 code frames (12 s)
L = n + CD                                                         # lane frames 0..n+1
texts = meta["text"].split()
pieces = IO.SP.word_pieces(IO.tok(), [{"text": t} for t in texts])
K = len(texts)
lens = [len(p) for p in pieces]
print(f"{person}: {K} words, {sum(lens)} pieces, first_word_s {meta.get('first_word_s')}, words_in_prefill {meta.get('words_in_prefill')}", flush=True)


def lane_of(starts):
    return np.asarray(MIF.lane_from_words([[s * 0.08 + 1e-3, s * 0.08 + 1e-3, t] for s, t in zip(starts, texts)], L), np.int64)


def dp(lp):
    """lp [L, V] log-probs of lane frame f -> best word start frames (training placement)."""
    base = lp[:, PAD]
    NEG = -1e30
    best = np.full((K, L), NEG)
    arg = np.zeros((K, L), np.int64)
    gain = []
    for k in range(K):
        g = np.full(L, NEG)
        for s in range(L - lens[k] + 1):
            v = sum(lp[s + j, pieces[k][j]] - base[s + j] for j in range(lens[k]))
            if s >= 1:
                v += lp[s - 1, NW] - base[s - 1]
            g[s] = v
        gain.append(g)
    best[0] = gain[0]
    for k in range(1, K):
        pm, pa = NEG, -1
        run = np.full(L, NEG); runa = np.zeros(L, np.int64)
        for s in range(L):                                         # prefix max of best[k-1] over s' <= s - lens[k-1] - 1
            j = s - lens[k - 1] - 1
            if j >= 0 and best[k - 1][j] > pm:
                pm, pa = best[k - 1][j], j
            run[s], runa[s] = pm, pa
        best[k] = np.where(run > NEG / 2, gain[k] + run, NEG)
        arg[k] = runa
    s = int(np.argmax(best[K - 1]))
    st = [s]
    for k in range(K - 1, 0, -1):
        s = int(arg[k][s]); st.append(s)
    return st[::-1], float(best[K - 1].max())


with torch.inference_mode():
    devs = [torch.device("cuda:0"), torch.device("cuda:1")]
    torch.cuda.set_device(devs[-1])
    m, ldev, cfgs, binfo = FP.build_model(ck, devs)
    mimi = IO.Mimi(devs[-1])
    sil = mimi.encode(np.zeros((2048 + 200) * 1920, np.float32))[:, :2048 + 200]
    eng = FP.FastPPM(m, ldev, cfgs, devs, n_inst=1, cfg_rows=False, max_s=16.0, mimi_stream=False, silence=sil)
    sp = MIF.system_block(audio[:, :125], None)
    P = int(sp["P"])
    # CAUSAL CONSTRAINED DECODING (the DP on a teacher-forced guess just copies the guess): step the engine frame by frame with the real
    # audio + pose forced; before frame f the lane distribution comes from hn[f - 1] (eng.hn_prev after step f - 1); between words the
    # next transcript word starts (NEW_WORD) when P(NEW_WORD) > P(PAD) (or when the remaining frames only just fit the remaining words),
    # its pieces are then forced on the following frames.
    eng.capture()
    lane0 = np.full(L, PAD, np.int64)
    x = MIF.assemble(sp, lane0, audio, pose, n, 0)
    params = dict(MIF.TONIGHT)

    def decode(thr):
        """causal constrained decoding; a word starts (NEW_WORD here) when 1 - P(PAD) > thr (the model expects speech in the lane)."""
        eng.start([dict(x, a_mode="empty")], 1, params, f0=0)
        st, k, pos, pend = [], 0, 0, False
        for f in range(L):
            z = torch.softmax(eng.text_logits(eng.hn_prev[:1]).float(), -1)[0]
            ppad = float(z[PAD])
            if pos > 0:
                tok = pieces[k - 1][pos]; pos = pos + 1 if pos + 1 < lens[k - 1] else 0
            elif pend:
                tok = pieces[k][0]; st.append(f); pos = 1 if lens[k] > 1 else 0; k += 1; pend = False
            elif k < K:
                need = sum(lens[k:]) + (K - k)
                pend = (1 - ppad) > thr or (L - f) <= need
                tok = NW if pend else PAD
            else:
                tok = PAD
            g = f - CD
            frc = {"lane": np.asarray([tok], np.int64)}
            if 0 <= g < n:
                frc["codes"] = (audio[:, g] + OFF)[None]; frc["pose"] = (pose[:, g] + OFF)[None]
            eng.step(f, force=frc, cold_pad=False)
            eng.result(f, False)
        return st

    exp = (2, 5, int(meta.get("words_in_prefill") or 10))           # first word frame, words in frames 0-12, words in frames 0-24
    best = None
    hist = []
    for thr in (0.5, 0.3, 0.2, 0.1, 0.05):
        st = decode(thr)
        for d in (0, 1, 2, 3):                                        # causal lag: the model reacts after hearing -> shift earlier
            s2 = [max(0, a - d) for a in st]
            c = (s2[0], sum(1 for a in s2 if a < 13), sum(1 for a in s2 if a < 25))
            err = abs(c[0] - exp[0]) + abs(c[1] - exp[1]) + abs(c[2] - exp[2])
            hist.append({"thr": thr, "shift": d, "counts": c, "err": err})
            if best is None or err < best[0]:
                best = (err, thr, d, s2)
        print(f"thr {thr}: counts (first, <13, <25) no shift {(st[0], sum(1 for a in st if a < 13), sum(1 for a in st if a < 25))}, last word frame {st[-1]}", flush=True)
    err, thr, d, starts = best
    print(f"best thr {thr} shift {d} err {err}", flush=True)
    words = [[round(s * 0.08 + 1e-3, 3), round(s * 0.08 + 1e-3, 3), t] for s, t in zip(starts, texts)]
    json.dump({"words": words, "method": "monoplex export-357 causal constrained lane decoding over the known transcript, real audio + pose forced (no ASR)", "trace": hist},
              open(f"{out}/{person}.words.json", "w"), indent=0)
    lane = lane_of(starts)
    np.savez(f"{out}/{person}.npz", lane=lane, audio=audio, pose=pose)
    c104 = sum(1 for s in starts if s < 13); c2 = sum(1 for s in starts if s < 25)
    print(json.dumps({"first_word_s": starts[0] * 0.08, "words_before_1.04s": c104, "words_before_2.0s": c2,
                      "expected (prefill_words.json)": {"first_word_s": meta.get("first_word_s"), "1.04s": 5, "2.0s": meta.get("words_in_prefill")}}))
sys.stdout.flush()
os._exit(0)
