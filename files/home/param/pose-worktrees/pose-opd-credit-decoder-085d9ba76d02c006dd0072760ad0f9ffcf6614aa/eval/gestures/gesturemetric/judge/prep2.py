"""gesturemetric (2) CPU-mini judge: per-track features -> judge/feat/<split>/<conv>__<side>.npz
  pose   float16 [2T, 47] per 25 fps frame: 15 points (body 0-12 + left / right hand centroid) x (x, y) / frame-0 shoulder width (raw
         canvas coords: window-mean shoulder-midpoint removal happens at train time), 2 hand spreads, 15 conf masks (conf > 0.3);
         points with conf <= 0.3 are 0. Same decode as gm.k55_of (person 01 frame, filters none).
  w_step int32 [n] own-word onset step (t0 / 0.08), w_vec float16 [n, D] lexical vector (Qwen3-0.6B layer-LAYER contextual state, mean
         over the word's pieces; or the static input embedding with --static), w_txt [n]
  mimi   int16 [T, 8] own-side Mimi cb0-7 (same T / clock as the pose tokens)
TRAIN = POSE_INDEX 'train' c00 convs with both sides ok (sorted, first N), EVAL = real_extract.eval_convs() (first N).
usage: prep2.py SPLIT N_CONVS MAX_TOK [SHARD/NSHARD] [--static] [--bench] [--stopfile PATH (exit 3 between tracks if it exists)]
       [--feat-dir DIR (default judge/feat; relative to judge/)]
env GM_DEVICE=cuda: pose decode on the GPU (score.decode CUDA path) and Qwen3-0.6B on the GPU in bf16 (default cpu, fp32)."""
import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
J = os.path.dirname(os.path.abspath(__file__))
H = os.path.dirname(J)
sys.path.insert(0, H)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gm  # noqa: E402
import real_extract as RE  # noqa: E402
import score as S  # noqa: E402

QWEN = "/data/home/amartyavarma/models/Qwen3-0.6B"
LAYER = 14
DEV = os.environ.get("GM_DEVICE", "cpu")
BODY = list(range(13))
LH, RH = list(range(13, 34)), list(range(34, 55))
_Q = {}


def convs(split, n):
    if split == "eval":
        return RE.eval_convs()[:n]
    pix = json.load(open(RE.POSE_INDEX))["rows"]
    tr = {}
    for d in pix.values():
        for v in d.values():
            if v["split"] == "train" and v["row_key"].endswith("/c00"):
                s = v["sides"]
                if all(s.get(x, {}).get("ok") and s[x]["T"] >= 1500 for x in ("assistant", "user")):
                    tr[v["conv"]] = v
    return [tr[c] for c in sorted(tr)][:n]


def pose_feats(k55, sw):
    """k55 [F, 55, 3] px -> float16 [F, 47]."""
    xy, cf = k55[..., :2] / sw, k55[..., 2] > 0.3
    F = k55.shape[0]
    pts = np.zeros((F, 15, 2)); m = np.zeros((F, 15))
    pts[:, :13] = np.where(cf[:, :13, None], xy[:, :13], 0); m[:, :13] = cf[:, :13]
    spread = np.zeros((F, 2))
    for i, J_ in enumerate((LH, RH)):
        c = cf[:, J_]
        n = c.sum(1)
        cen = (xy[:, J_] * c[..., None]).sum(1) / np.maximum(n, 1)[:, None]
        ok = n >= 3
        pts[:, 13 + i] = np.where(ok[:, None], cen, 0); m[:, 13 + i] = ok
        d = np.linalg.norm(xy[:, J_] - cen[:, None], axis=-1)
        spread[:, i] = np.where(ok, (d * c).sum(1) / np.maximum(n, 1), 0)
    return np.concatenate([pts.reshape(F, 30), spread, m], 1).astype(np.float16)


def qwen():
    if "m" not in _Q:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(QWEN)
        mdl = AutoModel.from_pretrained(QWEN, torch_dtype=torch.bfloat16 if DEV.startswith("cuda") else torch.float32)
        mdl.layers = mdl.layers[:LAYER]          # only up to the mid layer
        mdl.eval().to(DEV)
        _Q["m"], _Q["t"] = mdl, tok
    return _Q["m"], _Q["t"]


def word_vecs(words, static=False, max_tok=2048):
    """words [str] -> float16 [n, 1024]: per word mean of its pieces' layer-LAYER states (contextual, chunks of <= max_tok tokens)."""
    mdl, tok = qwen()
    out = np.zeros((len(words), 1024), np.float16)
    if not words:
        return out
    # chunk by words so each chunk stays < max_tok pieces
    i = 0
    while i < len(words):
        j, text, spans = i, "", []
        while j < len(words) and len(spans) < 900:
            w = words[j]
            a = len(text) + (1 if text else 0)
            text = text + (" " if text else "") + w
            spans.append((a, len(text)))
            j += 1
        enc = tok(text, return_offsets_mapping=True, return_tensors="pt", truncation=True, max_length=max_tok)
        offs = enc.pop("offset_mapping")[0].numpy()
        enc = {k: v.to(DEV) for k, v in enc.items()}
        with torch.no_grad():
            if static:
                h = mdl.embed_tokens(enc["input_ids"])[0].float().cpu().numpy()
            else:
                h = mdl(**enc, use_cache=False).last_hidden_state[0].float().cpu().numpy()
        k = 0
        for wi, (a, b) in enumerate(spans):
            sel = [t for t in range(k, len(offs)) if offs[t][1] > a and offs[t][0] < b]
            if sel:
                out[i + wi] = h[sel].mean(0)
                k = sel[-1]
        i = j
    return out


def main():
    split, n_conv, max_tok = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    shard = [a for a in sys.argv[4:] if a.replace("/", "").isdigit() and a.count("/") == 1]
    sh, nsh = map(int, shard[0].split("/")) if shard else (0, 1)
    static, bench = "--static" in sys.argv, "--bench" in sys.argv
    stopf = sys.argv[sys.argv.index("--stopfile") + 1] if "--stopfile" in sys.argv else None
    torch.set_grad_enabled(False)
    fd = sys.argv[sys.argv.index("--feat-dir") + 1] if "--feat-dir" in sys.argv else "feat"
    od = os.path.join(J, fd, split)
    os.makedirs(od, exist_ok=True)
    done = 0
    for ci, v in enumerate(convs(split, n_conv)):
        if ci % nsh != sh:
            continue
        by = RE.words_of(v)
        for side in ("assistant", "user"):
            name = v["conv"].replace("/", "_") + "__" + side
            fo = os.path.join(od, name + ".npz")
            if os.path.exists(fo) and not bench:
                continue
            if stopf and os.path.exists(stopf):
                print(f"STOP file {stopf} present: exiting cleanly", flush=True)
                sys.exit(3)
            t0 = time.time()
            s = v["sides"][side]
            codes = S._fix_codes(np.load(os.path.join(RE.ALOE, s["path"])))
            k55, sw = gm.k55_of(codes, S.DEFAULT_PERSON, max_tok, DEV)
            pf = pose_feats(k55, sw)
            T = pf.shape[0] // 2
            t1 = time.time()
            own = [w for w in by.get(s["spk"], []) if w[0] < T * 0.08]
            txt = [w[2] for w in own]
            wv = word_vecs(txt, static)
            t2 = time.time()
            mp = os.path.join(RE.ALOE, os.path.dirname(s["path"]), f"w00.{s['spk']}.codes.npy")
            mc = np.load(mp)
            mc = (mc if mc.shape[0] == 32 else mc.T)[:8, :T].T.astype(np.int16)
            np.savez(fo + ".tmp.npz", pose=pf, w_step=np.array([int(w[0] / 0.08) for w in own], np.int32), w_vec=wv,
                     w_txt=np.array(txt, dtype=object), mimi=mc, T=T, sw=sw, conv=v["conv"], side=side)
            os.replace(fo + ".tmp.npz", fo)
            print(f"{time.strftime('%H:%M:%S', time.gmtime())} {split} {name} T={T} words={len(own)} pose {t1 - t0:.1f}s words {t2 - t1:.1f}s total {time.time() - t0:.1f}s",
                  flush=True)
            done += 1
            if bench and done >= 2:
                return


if __name__ == "__main__":
    main()
