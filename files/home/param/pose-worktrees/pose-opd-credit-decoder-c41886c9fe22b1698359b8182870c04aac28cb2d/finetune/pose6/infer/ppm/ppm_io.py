# VENDORED for aloe-pose-monoplex-infer. Upstream: ppm/code/infer_ppm.py (the subset fast_ppm.py and infer.py use), copied verbatim
# except MIMI (= $MIMI_DIR, else kyutai/mimi at the pinned revision) and the unused tasks / eval paths dropped.
"""PPM helpers for the fast engine: KV-cached backbone Stack (prefill), lane Grammar, Mimi codec, wav io, audio stats, the Qwen3
tokenizer, lane <-> words, the D33 link scale."""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
os.environ.setdefault("PPM_CD", "2")

import numpy as np  # noqa: E402
import torch  # noqa: E402

import model_ppm as M  # noqa: E402
import speech_q as SP  # noqa: E402
import fused_triton as FO  # noqa: E402
import fa3_op  # noqa: E402

CD, DEPTH, CBS, FPS = SP.CD, SP.DEPTH, SP.CBS, SP.FPS
PAD, NW, EOB, TE = SP.PAD_ID, SP.NW_ID, SP.EOB_ID, 151643
OFF = np.arange(DEPTH, dtype=np.int64) * CBS
BF = torch.bfloat16
MIMI_REPO, MIMI_REV = "kyutai/mimi", "89091b3e466eb6a9d11e537bf26b144f194978f7"     # CC-BY-4.0
MIMI = os.environ.get("MIMI_DIR") or MIMI_REPO


def decp_bb_alpha_of(c, step):
    """D33: the trainer's link scale at optimizer step `step` from its cfg (= train_ppm.decp_bb_alpha)."""
    a, f0, fn = float(c.get("decp_bb_alpha", 1.0)), int(c.get("decp_bb_fade_from", -1)), int(c.get("decp_bb_fade_steps", 0))
    if f0 >= 0 and fn > 0:
        a *= min(1.0, max(0.0, 1.0 - (int(step) - f0) / fn))
    return a


class Stack:
    """Qwen3 layers (model_ppm.Block .layer) with a KV cache; the fused training layer (FO rms / qk_prep / add_rms / silu_mul + FA3)."""

    def __init__(self, blocks, devs, cfg, lmax):
        self.L = [b.layer for b in blocks]
        self.devs = devs
        nkv, hd = cfg.num_key_value_heads, cfg.head_dim
        self.k = [torch.zeros(lmax, nkv, hd, dtype=BF, device=d) for d in devs]
        self.v = [torch.zeros(lmax, nkv, hd, dtype=BF, device=d) for d in devs]
        self.rot = {d: M.Rot(cfg, d) for d in sorted(set(devs), key=str)}
        self.lmax = lmax

    def run(self, e, s):
        """e [1, n, H] embeddings of positions s .. s+n-1 -> final hidden [1, n, H] (pre final norm), on the last layer's device."""
        n = e.shape[1]
        assert s + n <= self.lmax, ("KV cache too short", s + n, self.lmax)
        pe, h = {}, e
        for i, L in enumerate(self.L):
            d = self.devs[i]
            if h.device != d:
                h = h.to(d)
            if d not in pe:
                pid = torch.arange(s, s + n, device=d)[None]
                pe[d] = self.rot[d](h, pid)
            with torch.cuda.device(d):                                  # Triton kernels launch on the CURRENT device
                h = self._layer(L, h, pe[d], self.k[i], self.v[i], s)
        return h

    @staticmethod
    def _layer(L, h, pe, kc, vc, s):
        at, ln1, ln2, mlp = L.self_attn, L.input_layernorm, L.post_attention_layernorm, L.mlp
        T = h.shape[1]
        cos, sin = pe
        x = FO.rms(h, ln1.weight, float(ln1.variance_epsilon))
        q, k = FO.qk_prep(at.q_proj(x), at.k_proj(x), at.q_norm.weight, at.k_norm.weight, float(at.q_norm.variance_epsilon), cos, sin)
        v = at.v_proj(x).view(T, -1, at.head_dim)
        kc[s:s + T] = k[0]
        vc[s:s + T] = v
        cu_q = torch.tensor([0, T], dtype=torch.int32, device=h.device)
        cu_k = torch.tensor([0, s + T], dtype=torch.int32, device=h.device)
        a = fa3_op.fa3_varlen_func(q[0], kc[:s + T], vc[:s + T], cu_q, cu_k, T, s + T, at.scaling, True)
        h, x = FO.add_rms(h, at.o_proj(a.reshape(1, T, -1)), ln2.weight, float(ln2.variance_epsilon))
        return h + mlp.down_proj(FO.silu_mul(mlp.gate_proj(x), mlp.up_proj(x)))


class Grammar:
    """lane grammar masks over the full head vocab (stage-3 TextGrammar). mode nw: after NEW_WORD a word piece; after PAD / EOB not a word
    piece; chat: words / PAD / EOB, no NEW_WORD; free: words / PAD / NEW_WORD (/ EOB)."""

    def __init__(self, V, dev, mode, allow_eob):
        self.mode = mode
        base = torch.zeros(V, dtype=torch.bool, device=dev)
        base[:TE] = True
        base[PAD] = True
        if mode != "chat":
            base[NW] = True
        if allow_eob:
            base[EOB] = True
        self.base = base
        self.after_nw = torch.zeros(V, dtype=torch.bool, device=dev)
        self.after_nw[:TE] = True
        self.after_pad = base.clone()
        self.after_pad[:TE] = False

    def mask(self, prev):
        if self.mode == "nw":
            if prev == NW:
                return self.after_nw
            if prev in (PAD, EOB):
                return self.after_pad
        return self.base


class Mimi:
    def __init__(self, dev):
        from transformers import MimiModel
        self.dev = dev
        kw = {} if os.environ.get("MIMI_DIR") else {"revision": MIMI_REV}
        self.m = MimiModel.from_pretrained(MIMI, **kw).to(dev).eval()

    @torch.no_grad()
    def decode(self, codes):
        """codes [32, F] local -> float32 [F * 1920] at 24 kHz."""
        if codes.shape[1] == 0:
            return np.zeros(0, np.float32)
        c = torch.as_tensor(np.asarray(codes, np.int64), device=self.dev)[None]
        return self.m.decode(c).audio_values[0, 0].float().cpu().numpy()[:c.shape[-1] * 1920]

    @torch.no_grad()
    def encode(self, wav24):
        x = torch.as_tensor(np.asarray(wav24, np.float32), device=self.dev)[None, None]
        return self.m.encode(x, num_quantizers=DEPTH).audio_codes[0].cpu().numpy().astype(np.int64)     # [32, F]


def read_wav24(path):
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != 24000:
        from scipy.signal import resample_poly
        g = math.gcd(int(sr), 24000)
        x = resample_poly(x, 24000 // g, int(sr) // g).astype(np.float32)
    return x


def write_wav(path, x, sr=24000):
    import soundfile as sf
    sf.write(path, np.clip(x, -1, 1), sr, subtype="PCM_16")


def audio_stats(x):
    if len(x) == 0:
        return {"s": 0.0}
    r = float(np.sqrt(np.mean(x ** 2)))
    w = 2400
    fr = np.sqrt(np.mean(x[:len(x) // w * w].reshape(-1, w) ** 2, 1)) if len(x) >= w else np.array([r])
    return {"s": round(len(x) / 24000, 2), "rms": round(r, 4), "rms_dbfs": round(20 * math.log10(max(r, 1e-9)), 1),
            "peak": round(float(np.abs(x).max()), 4), "clip_frac": round(float((np.abs(x) >= 0.999).mean()), 6),
            "silent_frac_100ms": round(float((fr < 0.003).mean()), 3)}


_TOK = [None]


def tok():
    if _TOK[0] is None:
        from transformers import AutoTokenizer
        _TOK[0] = AutoTokenizer.from_pretrained(M.QWEN32)
    return _TOK[0]


def lane_words(lane, f0=0, t_off=0, prompt_end=None):
    """word pieces of a lane grouped into words (a new word = a piece starting with 'Ġ' or following a non-word frame).
    lane index f -> time (f - t_off) / 12.5. -> [{text, frame, t, prompt}]"""
    tk = tok()
    words, cur = [], None
    for f in range(f0, len(lane)):
        t = int(lane[f])
        if t < TE:
            sp = tk.convert_ids_to_tokens(t).startswith("Ġ")
            if cur is None or sp:
                cur = {"pieces": [t], "frame": f}
                words.append(cur)
            else:
                cur["pieces"].append(t)
        else:
            cur = None
    return [{"text": tk.decode(w["pieces"]).strip(), "frame": int(w["frame"]), "t": round((w["frame"] - t_off) / FPS, 2),
             "prompt": bool(prompt_end is not None and w["frame"] < prompt_end)} for w in words]


def words_lane(words, n, offset_s=0.0):
    """[[text, start, end], ...] (seconds) -> lane [n] with NEW_WORD (the speech rows' placement: speech_q.lane + insert_new_word)."""
    ws = [{"text": str(w[0]), "start": float(w[1]) - offset_s, "end": float(w[2]) - offset_s} for w in words]
    ws = [w for w in ws if w["start"] >= 0]
    pieces = SP.word_pieces(tok(), ws)
    pid, offs = [], [0]
    for p in pieces:
        pid += p; offs.append(len(pid))
    wif = np.zeros(n, np.int64)
    lane = SP.lane(ws, pid, offs, n, wif)
    SP.insert_new_word(lane, wif)
    return lane
