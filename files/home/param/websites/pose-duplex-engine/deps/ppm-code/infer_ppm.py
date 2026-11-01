#!/usr/bin/env python3
"""PPM INFERENCE (pose + podcast + text-duplex model): one entry point for every task, any PPM checkpoint. bf16 only, single process,
model split over 2+ GPUs (backbone layers spread over --devs; head, both towers and both decoders on the last device).

  python infer_ppm.py --ckpt <ckpt> --task speech|pose|text [options]        (wrapper: infer_ppm.sh, runs it on slinky-15 via srun)

CHECKPOINTS (--ckpt): (a) a trainer export <export_root>/<run>/w-N (consolidated bf16 DCP, safety_ppm.export_weights), or (b) an inference
  copy of a node-local training save made by ckpt_local_export.py (<dir>/meta.json format ppm-local-bf16-shards-v1: 128 FSDP2 dim-0 shards,
  concatenated here in rank order). Model sizes (layers, tower / decoder depth, projector rank, tower vocab) come from the checkpoint.
  Every parameter of the model must be in the checkpoint (asserted, no silent random tensors).

GENERATION = the trainer's exact per-position layout (model_ppm / data_ppm), autoregressive with KV caches:
  position T: text lane of channel B sampled from the head at hn[T-1] (P_B projector + lm_head with the PAD / NEW_WORD / EOB own rows);
  speech tower step T: text_emb(lane_B[T]) + code_emb(B Mimi codes at T-1) + cond(hn[T-1]) -> cb0 (4B tower head) -> 0.6B decoder cb1..31
  (autoregressive over the 31 codebooks) = the Mimi frame entering the backbone at T (frame T - P - CD; CD = 2); pose rows also run the pose
  tower + pose decoder (pose codes at POSE_DELAY = CD); backbone input at T = sum_c present_c (emb_c(lane_c) + aud_c(codes_c)) [+ pose_emb].
  Channel A is never predicted: it is fed (row data / provided audio+words / silence).
SAMPLING defaults (user standing rules): no min-p, no audio repetition penalty, CFG off (none exists here), bf16. Temperatures / top-k =
  the stage-3 duplex inference reference (infer_s3_fast: text 0.8 / top-k 50, audio 0.8 / top-k 250); pose = the SAM / Vansh pose reference
  (0.8, top-k 30, repetition penalty 1.2 on pose cb0 history). NO lane grammar (10-28 user order, permanent): the word lane is drawn from
  the full in-vocab distribution (ids < n_in) in every task; --free-text / --allow-eob are ignored. Every other knob is a flag.
NO Whisper / ASR anywhere: transcripts are the model's own word lanes.
D33 (decp_bb backbone -> pose decoder link, temporary): honoured when the checkpoint has it: pose decoder input slot 0 = proj(hv[T]) +
  alpha x decp_bb(hn[T - 1]), alpha = the trainer's at the checkpoint step (local copy: meta cfg; DCP export: its train_cfg.json; flag
  --decp-bb-alpha overrides). Checkpoints without it: unchanged.

TASKS
  speech  podcast continuation on a held-out row (evalsets/speech_s3: the 103 stage-3 eval rows, held out of PPM training from step 170;
          micro files {B,A}-iii.pkl = the PPM speech micro incl. its training system prompt). Prefill system prompt + the first
          --prompt-s seconds of both channels, then generate B for --gen-s seconds while A is fed from the row (--a-mode row) or silence.
          --sys row|empty|custom (custom: --persona-a/-b, --context-a/-b text, --voice-a/-b wav clips of 3-10 s). --a-wav [--a-words
          json] replaces the row: A = that audio (+ word lane), B silent during the prompt.
          -> b.wav, b_gen.wav, a.wav, stereo.wav (L = A, R = B), words.json (both lanes' words with times), transcript.txt, codes.npz, meta.json
  pose    held-out pose row (evalsets/pose_infer, pose-v2/v3 eval split): prefill the SLOT + first --prompt-s seconds of clip 0, then
          generate B's text lane + Mimi + pose codes for --gen-s seconds. Pose codes -> keypoints with the pose tokenizer
          (InternalCan/pose-tokenizer-32cb-200k-noshift @ 1d6ff4f0, frame-0 root / shoulder width of the clip) -> mp4 (model | ground
          truth side by side, model audio) + frame png. -> pose.mp4, frame.png, b.wav, words.json, transcript.txt, codes.npz, meta.json
  text    --text-mode duplex: a held-out Gemma text-duplex conversation (gemma-txtdup-v1 eval packs): prefill --prompt-s seconds of
          both lanes, generate B for --gen-s seconds with A fed from the data. --text-mode chat: --user messages (repeatable) or
          --chat-file (json list of {role, content}; user turns are the script): A = the user turn rendered as in training (render_q
          timing), B forced to PAD while A speaks, then B free-runs until it emits <|im_end|> (or --max-turn-s).
          -> transcript.txt, text.json, meta.json
Outputs: <out-root>/<ckpt tag>/<task>/<name>/ (default out-root /data/home/amartyavarma/ppm/samples)."""
import argparse
import json
import math
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(HERE, "vendor"), os.path.join(HERE, "lib"), HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("PPM_CD", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import model_ppm as M  # noqa: E402
import speech_q as SP  # noqa: E402
import fused_triton as FO  # noqa: E402
import fa3_op  # noqa: E402

CD, DEPTH, CBS, FPS = SP.CD, SP.DEPTH, SP.CBS, SP.FPS
PAD, NW, EOB, TE = SP.PAD_ID, SP.NW_ID, SP.EOB_ID, 151643
SPEC = M.SPEC
OFF = np.arange(DEPTH, dtype=np.int64) * CBS
SAMPLES = "/data/home/amartyavarma/ppm/samples"
SPEECH_EVAL = "/data/home/amartyavarma/ppm/evalsets/speech_s3"
POSE_EVAL = "/data/home/amartyavarma/ppm/evalsets/pose_infer"
GEMMA = "/data/home/amartyavarma/data/gemma-txtdup-v1"
MIMI = "/home/param/models/mimi"
POSE_TOK = ("InternalCan/pose-tokenizer-32cb-200k-noshift", "1d6ff4f0de89408cb054da618e309d4334596252")
BF = torch.bfloat16
T0 = time.time()


def log(*a):
    print(f"[{time.strftime('%H:%M:%S')} +{time.time() - T0:6.1f}s]", *a, flush=True)


# ================================================================================================ checkpoint
def decp_bb_alpha_of(c, step):
    """D33: the trainer's link scale at optimizer step `step` from its cfg (= train_ppm.decp_bb_alpha)."""
    a, f0, fn = float(c.get("decp_bb_alpha", 1.0)), int(c.get("decp_bb_fade_from", -1)), int(c.get("decp_bb_fade_steps", 0))
    if f0 >= 0 and fn > 0:
        a *= min(1.0, max(0.0, 1.0 - (int(step) - f0) / fn))
    return a


def ckpt_info(path):
    path = os.path.abspath(path.rstrip("/"))
    if os.path.isfile(f"{path}/meta.json"):
        meta = json.load(open(f"{path}/meta.json"))
        assert meta.get("format") == "ppm-local-bf16-shards-v1", meta.get("format")
        assert os.path.isfile(f"{path}/COMPLETE"), ("inference copy not COMPLETE (ckpt_local_export.py --finalize)", path)
        c = meta["cfg"]
        return {"kind": "local", "path": path, "step": int(meta["step"]), "run": meta.get("run_base") or "ppm",
                "layers": int(c["layers"]), "tower_layers": int(c["tower_layers"]), "dec_layers": int(c["dec_layers"]),
                "proj_r": int(c["proj_r"]), "tower_vocab": int(c["tower_vocab"]), "cd": int(c.get("cd", CD)), "meta": meta,
                "decp_bb": bool(c.get("decp_bb", 0)), "decp_bb_alpha": decp_bb_alpha_of(c, meta["step"]) if c.get("decp_bb", 0) else None,
                "tag": f"{meta.get('run_base') or 'ppm'}-{os.path.basename(path)}"}
    assert os.path.isfile(f"{path}/.metadata"), ("not a PPM checkpoint (no meta.json / .metadata)", path)
    import torch.distributed.checkpoint as dcp
    md = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
    shp = {k: tuple(v.size) for k, v in md.items() if hasattr(v, "size")}

    def nlay(prefix):
        return 1 + max(int(k[len(prefix):].split(".")[0]) for k in shp if k.startswith(prefix))
    step = int(os.path.basename(path).split("-")[-1]) if os.path.basename(path).split("-")[-1].isdigit() else -1
    run = os.path.basename(os.path.dirname(path))
    bb = "decp_bb.weight" in shp
    side = json.load(open(f"{path}/train_cfg.json")) if os.path.isfile(f"{path}/train_cfg.json") else {}
    if bb and "decp_bb_alpha" not in side:
        log(f"[ckpt] WARNING {path} has the D33 decp_bb link but no train_cfg.json: alpha = 1.0 unless --decp-bb-alpha is given")
    return {"kind": "dcp", "path": path, "step": step, "run": run, "layers": nlay("layers."), "tower_layers": nlay("tower_s.layers."),
            "dec_layers": nlay("dec_s.model.layers."), "proj_r": shp["head.proj.G"][0], "tower_vocab": shp["tower_s.text_emb.weight"][0],
            "cd": CD, "shapes": shp, "decp_bb": bb, "decp_bb_alpha": float(side.get("decp_bb_alpha", 1.0)) if bb else None,
            "tag": f"{run}-{os.path.basename(path)}"}


def build_model(ci, devs):
    """PPM module (no FSDP), bf16 parameters placed on devs; weights from the checkpoint. -> (model, layer device list, configs)."""
    assert ci["cd"] == CD, ("checkpoint code delay != PPM_CD", ci["cd"], CD)
    c32, c4, c06 = M.cfg32(ci["layers"]), M.cfg4b(ci["tower_layers"]), M.cfg06(ci["dec_layers"])
    torch.set_default_dtype(BF)
    bb = bool(ci.get("decp_bb"))
    with torch.device("meta"):
        m = M.PPM(c32, c4, c06, ci["proj_r"], ci["tower_vocab"], **({"decp_bb": True} if bb else {}))
    torch.set_default_dtype(torch.float32)
    nL, nd = ci["layers"], len(devs)
    # backbone spread: the last device also holds head + towers + decoders (~22 GB), so it gets fewer layers
    if nd == 1:
        ldev = [devs[0]] * nL
    else:
        w = [1.0] * (nd - 1) + [0.45]
        cuts = np.cumsum(w) / sum(w) * nL
        ldev = [devs[int(np.searchsorted(cuts, i + 0.5))] for i in range(nL)]
    m.inp.to_empty(device=devs[0])
    for L, d in zip(m.layers, ldev):
        L.to_empty(device=d)
    for mod in (m.head, m.tower_s, m.tower_p, m.dec_s, m.dec_p) + ((m.decp_bb,) if bb else ()):
        mod.to_empty(device=devs[-1])
    for dec in (m.dec_s, m.dec_p):
        dec.model.rotary_emb = type(dec.model.rotary_emb)(config=c06, device=devs[-1])
    if bb:
        m.decp_bb_alpha = float(ci["decp_bb_alpha"])
        log(f"[build] D33 decp_bb link present: pose decoder slot 0 += {m.decp_bb_alpha:g} x decp_bb(hn[T - 1])")
    for n, p in m.named_parameters():
        assert p.dtype == BF, (n, p.dtype)
    load_weights(m, ci)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ldev, (c32, c4, c06)


@torch.no_grad()
def load_weights(m, ci):
    t0 = time.time()
    params = {n: p for n, p in m.named_parameters()}
    if ci["kind"] == "dcp":
        import torch.distributed.checkpoint as dcp
        want = set(params)
        have = set(ci["shapes"])
        assert want <= have, ("checkpoint lacks model tensors", sorted(want - have)[:6])
        for n, p in params.items():
            assert tuple(p.shape) == tuple(ci["shapes"][n]), (n, tuple(p.shape), ci["shapes"][n])
        dcp.load(dict(params), checkpoint_id=ci["path"], no_dist=True)
        extra = sorted(have - want)
        log(f"[load] DCP {ci['path']}: {len(want)} tensors" + (f" (ignored {len(extra)} extra: {extra[:4]})" if extra else ""))
    else:
        meta = ci["meta"]
        rows = {n: 0 for n in params}
        for r in range(int(meta["world"])):
            sd = torch.load(f"{ci['path']}/rank{r}.pt", map_location="cpu", weights_only=True)
            for n, t in sd.items():
                if n not in params and n.startswith("pose_head."):   # D32 training-only aux head: not part of the inference model
                    continue
                p = params[n]
                k = t.shape[0] if t.dim() else 0
                if k:
                    p[rows[n]:rows[n] + k].copy_(t, non_blocking=False)
                    rows[n] += k
            del sd
        rep = torch.load(f"{ci['path']}/rep.pt", map_location="cpu", weights_only=True)
        for n, t in rep.items():
            params[n].copy_(t)
            rows[n] = params[n].shape[0]
        bad = [n for n, p in params.items() if rows[n] != p.shape[0]]
        assert not bad, ("tensors not fully covered by the shards", [(n, rows[n], params[n].shape[0]) for n in bad[:6]])
        log(f"[load] local shards {ci['path']} (step {ci['step']}, {meta['world']} ranks + rep.pt): {len(params)} tensors")
    nonfin = [n for n, p in params.items() if not torch.isfinite(p).all()]
    assert not nonfin, ("non-finite weights", nonfin[:4])
    log(f"[load] done in {time.time() - t0:.0f}s; |emb_B| {float(m.inp.emb[1].weight.float().norm()):.1f} "
        f"|tower_s.cb0_head| {float(m.tower_s.cb0_head.weight.float().norm()):.2f} |dec_s.heads| {float(m.dec_s.heads.weight.float().norm()):.2f}")


# ================================================================================================ engine
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


def sample(z, temp, topk, gen, mask=None):
    """z [V] fp32 logits -> int (top-k then temperature-softmax sampling; temp <= 0 = argmax)."""
    if mask is not None:
        z = z.masked_fill(~mask, float("-inf"))
    if temp <= 0:
        return int(z.argmax())
    z = z / temp
    if topk and topk < z.shape[-1]:
        v, i = torch.topk(z, topk)
        return int(i[torch.multinomial(torch.softmax(v, -1), 1, generator=gen)])
    return int(torch.multinomial(torch.softmax(z, -1), 1, generator=gen))


def sample_rows(z, temp, topk, gen):
    """z [R, V] -> [R] long (independent rows), on device, no host sync."""
    if temp <= 0:
        return z.argmax(-1)
    z = z / temp
    if topk and topk < z.shape[-1]:
        v, i = torch.topk(z, topk, dim=-1)
        return i.gather(1, torch.multinomial(torch.softmax(v, -1), 1, generator=gen))[:, 0]
    return torch.multinomial(torch.softmax(z, -1), 1, generator=gen)[:, 0]


class DecoderAR:
    """0.6B depth decoder, autoregressive over cb1..cb31 with a 32-slot KV cache (training: one causal pass over [proj(hv), e(cb0),
    e(cb1..cb30)]; hidden after e(cb_{d-1}) predicts cb_d). Plain torch / HF modules (eager), SDPA with GQA."""

    def __init__(self, dec, c06, dev):
        self.d, self.dev = dec, dev
        nkv, hd = c06.num_key_value_heads, c06.head_dim
        self.nl = len(dec.model.layers)
        self.k = torch.zeros(self.nl, 1, nkv, DEPTH + 1, hd, dtype=BF, device=dev)
        self.v = torch.zeros_like(self.k)
        self.W = dec.heads.weight.view(DEPTH - 1, CBS, -1)
        pid = torch.arange(DEPTH + 1, device=dev)[None]
        x0 = torch.zeros(1, DEPTH + 1, c06.hidden_size, dtype=BF, device=dev)
        self.cos, self.sin = dec.model.rotary_emb(x0, pid)                 # [1, 33, hd]

    def _run(self, x, s):
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        n = x.shape[1]
        cos, sin = self.cos[:, s:s + n], self.sin[:, s:s + n]
        h = x
        for i, L in enumerate(self.d.model.layers):
            at = L.self_attn
            y = L.input_layernorm(h)
            q = at.q_norm(at.q_proj(y).view(1, n, -1, at.head_dim)).transpose(1, 2)
            k = at.k_norm(at.k_proj(y).view(1, n, -1, at.head_dim)).transpose(1, 2)
            v = at.v_proj(y).view(1, n, -1, at.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)
            self.k[i, :, :, s:s + n] = k
            self.v[i, :, :, s:s + n] = v
            a = F.scaled_dot_product_attention(q, self.k[i, :, :, :s + n], self.v[i, :, :, :s + n], is_causal=(n > 1 and s == 0),
                                               scale=at.scaling, enable_gqa=True)
            h = h + at.o_proj(a.transpose(1, 2).reshape(1, n, -1))
            h = h + L.mlp(L.post_attention_layernorm(h))
        return self.d.model.norm(h)

    def __call__(self, hv, cb0, temp, topk, gen, zb=None):
        """hv [1, Hv] tower state, cb0 [1] long (local code) -> codes [32] long local (cb0 + sampled cb1..31). zb [1, Hl] (D33 pose decoder):
        added to slot 0 (= model_ppm.Decoder.inputs)."""
        p = self.d.proj(hv) if zb is None else self.d.proj(hv) + zb.to(hv.dtype)
        x = torch.cat([p[:, None], self.d.cb0_emb(cb0)[:, None]], 1)
        out = [cb0]
        hl = self._run(x, 0)[:, -1]
        for d in range(1, DEPTH):
            z = (hl @ self.W[d - 1].t()).float()
            c = sample_rows(z, temp, topk, gen)
            out.append(c)
            if d < DEPTH - 1:
                hl = self._run(self.d.model.embed_tokens(c + (d - 1) * CBS)[:, None], d + 1)[:, -1]
        return torch.cat(out)


class Grammar:
    """10-28 USER ORDER (universal + permanent): NO lane grammar. The word lane is drawn from the full in-vocab distribution: every mask
    (any mode, any previous token) = ids < n_in, the vocabulary bound only (never an id the input table never trained); PAD / NEW_WORD /
    EOB / word pieces are all allowed after anything. mode / allow_eob are accepted for call compatibility and IGNORED (there is no switch
    back). base / after_nw / after_pad are kept (= the same in-vocab mask) for engines that read them."""

    def __init__(self, V, dev, mode=None, allow_eob=None, n_in=None):
        n = int(V) if n_in is None else min(int(V), int(n_in))
        self.mode = "none"
        self.n_in = n
        self.base = torch.arange(int(V), device=dev) < n
        self.after_nw = self.base
        self.after_pad = self.base

    def mask(self, prev=None):
        return self.base


class PPMGen:
    def __init__(self, m, ldev, cfgs, devs, lmax, a):
        self.m, self.a = m, a
        c32, c4, c06 = cfgs
        self.d0, self.dl = devs[0], devs[-1]
        self.lmax = lmax
        self.bb = Stack(list(m.layers), ldev, c32, lmax)
        self.ts = Stack(list(m.tower_s.layers), [self.dl] * len(m.tower_s.layers), c4, lmax)
        self.tp = Stack(list(m.tower_p.layers), [self.dl] * len(m.tower_p.layers), c4, lmax)
        self.ds = DecoderAR(m.dec_s, c06, self.dl)
        self.dp = DecoderAR(m.dec_p, c06, self.dl)
        h = m.head
        self.R = torch.cat([h.spec_out, h.nw_out, h.eob_out], 0)
        self.spec = torch.tensor(list(SPEC), device=self.dl)
        self.gen = torch.Generator(device=self.dl)
        self.gen.manual_seed(int(a.seed))
        self.V = h.lm_head.weight.shape[0]
        # vocabulary bound for the word-lane draw (10-28 no grammar): never an id without a trained input row (B lane table + tower text table)
        rows = [self.V] + [int(t.weight.shape[0]) for t in (getattr(getattr(m, "inp", None), "emb", [None, None])[1],
                                                             getattr(getattr(m, "tower_s", None), "text_emb", None)) if t is not None]
        self.n_in = min(rows)

    # ---- pieces
    def embed(self, ids, depth, present, pdepth):
        """numpy ids [2, n], depth [2, n, 32] (global ids, -1 none) or None, present [2, n] or None, pdepth [n, 32] or None -> [1, n, H]."""
        t = lambda x: None if x is None else torch.as_tensor(np.asarray(x), device=self.d0)  # noqa: E731
        return self.m.inp(t(ids), t(depth), t(present), t(pdepth))[None]

    def hnorm(self, h):
        n = self.m.head.norm
        return FO.rms(h[0].to(self.dl), n.weight, float(n.variance_epsilon))                   # [n, H]

    def text_logits(self, hn):
        """hn [1, H] -> [V] fp32 (P_B projector, lm_head, own PAD / NW / EOB rows)."""
        x = self.m.head.proj(hn)
        z = (x @ self.m.head.lm_head.weight.t()).float()
        z[:, self.spec] = (x @ self.R.t()).float()
        return z[0]

    def tower_in(self, tw, ids, dprev, hprev):
        """ids [n] lane B, dprev [n, 32] codes of the previous position (-1 none), hprev [n, H] hn of the previous position (0 none)."""
        ids = torch.as_tensor(np.asarray(ids), device=self.dl)
        dprev = torch.as_tensor(np.asarray(dprev), device=self.dl)
        e = tw.text_emb(ids) + tw.code_emb(dprev).to(BF) + tw.cond(hprev).to(BF)
        return e[None]

    def tower_head(self, tw, h):
        hv = FO.rms(h[0], tw.norm.weight, float(tw.norm.variance_epsilon))
        return hv, (hv @ tw.cb0_head.weight.t()).float()

    # ---- prefill
    @torch.no_grad()
    def prefill(self, ids, depth, present, pdepth, kind):
        """teacher-forced positions 0..n-1 (numpy arrays of the trainer's micro). -> hn [n, H]."""
        n = ids.shape[1]
        e = self.embed(ids, depth, present, pdepth)
        hn = self.hnorm(self.bb.run(e, 0))
        if kind in ("speech", "pose"):
            hprev = torch.cat([torch.zeros_like(hn[:1]), hn[:-1]], 0)
            dprev = np.concatenate([np.full((1, DEPTH), -1, np.int64), depth[1, :-1]], 0)
            self.last_ts = self.ts.run(self.tower_in(self.m.tower_s, ids[1], dprev, hprev), 0)
            if kind == "pose":
                pprev = np.concatenate([np.full((1, DEPTH), -1, np.int64), pdepth[:-1]], 0)
                self.last_tp = self.tp.run(self.tower_in(self.m.tower_p, ids[1], pprev, hprev), 0)
        torch.cuda.synchronize()
        return hn

    @torch.no_grad()
    def parity(self, ids, depth, present, pdepth, kind, t0, K=8, nlight=64, light_pos=None):
        """teacher-forced checks on the prompt (positions 0..n-1): (1) losses vs the trainer's eval scale: B text CE / normalised word CE
        (positions >= t0), speech cb0 CE, 0.6B CE (first nlight coded frames); (2) KV-cache consistency: prefill n - K then K single steps
        must reproduce the full prefill's hn / tower cb0 logits. -> dict."""
        n = ids.shape[1]
        hn = self.prefill(ids, depth, present, pdepth, kind)
        hv, z0 = self.tower_head(self.m.tower_s, self.last_ts)
        out = {}
        sel = torch.arange(t0, n - 1, device=self.dl)
        y = torch.as_tensor(ids[1, t0 + 1:n], device=self.dl)
        x = self.m.head.proj(hn[sel])
        z = (x @ self.m.head.lm_head.weight.t()).float()
        z[:, self.spec] = (x @ self.R.t()).float()
        ce = torch.logsumexp(z, -1) - z.gather(1, y[:, None])[:, 0]
        isw = y < TE
        wce = torch.logsumexp(z[:, :TE], -1) - z.gather(1, y[:, None])[:, 0]
        out["text_ce"] = round(float(ce.mean()), 4)
        out["sums"] = {"text_ce": float(ce.sum()), "n_text": int(ce.numel()), "word_norm": float(wce[isw].sum()), "n_word": int(isw.sum())}
        out["word_ce_norm"] = round(float(wce[isw].mean()), 4) if bool(isw.any()) else None
        out["pad_ce"] = round(float(ce[y == PAD].mean()), 4) if bool((y == PAD).any()) else None
        lab = torch.as_tensor(depth[1, :, 0], device=self.dl)
        m = lab >= 0
        m[:t0] = False
        if bool(m.any()):
            c0s = F.cross_entropy(z0[m], lab[m], reduction="sum")
            out["cb0_ce"] = round(float(c0s) / int(m.sum()), 4)
            out["sums"].update(cb0=float(c0s), n_cb0=int(m.sum()))
            if light_pos is not None:
                lp = torch.as_tensor(np.asarray(light_pos), device=self.dl)
                fr = lp[torch.randperm(lp.numel(), generator=torch.Generator().manual_seed(1234))[:2048].to(self.dl)]   # the trainer's eval pick
            else:
                fr = torch.nonzero(m)[:nlight, 0]
            dd = self.ds
            codes = torch.as_tensor(depth[1], device=self.dl)[fr]                    # global ids
            xx = torch.cat([dd.d.proj(hv[fr])[:, None], dd.d.cb0_emb(codes[:, 0])[:, None],
                            dd.d.model.embed_tokens(codes[:, 1:DEPTH - 1] - CBS)], 1)
            hl = dd.d.model(inputs_embeds=xx, use_cache=False).last_hidden_state
            lo = torch.einsum("ndh,dvh->ndv", hl[:, 1:], dd.W).float()
            labels = codes[:, 1:] - torch.arange(1, DEPTH, device=self.dl) * CBS
            ls_ = F.cross_entropy(lo.reshape(-1, CBS), labels.reshape(-1), reduction="sum")
            out["light_ce"] = round(float(ls_) / labels.numel(), 4)
            out["sums"].update(light=float(ls_), n_light=int(labels.numel()))
            # the AR decoder must give the same cb1 logits as the full pass (cache path)
            f0 = fr[:1]
            x0 = torch.cat([dd.d.proj(hv[f0])[:, None], dd.d.cb0_emb(codes[:1, 0])[:, None]], 1)
            hl_ar = dd._run(x0, 0)[:, -1]
            out["dec_ar_vs_full_maxdiff_cb1_logit"] = round(float(((hl_ar @ dd.W[0].t()).float() - lo[0, 0]).abs().max()), 4)
        # (2) incremental vs full
        hn2 = self.prefill(ids[:, :n - K], depth[:, :n - K], present[:, :n - K], None if pdepth is None else pdepth[:n - K], kind)
        hl = hn2[-1:]
        dmax, zmax = 0.0, 0.0
        for T in range(n - K, n):
            if kind in ("speech", "pose"):
                h = self.ts.run(self.tower_in(self.m.tower_s, [int(ids[1, T])], depth[1, T - 1][None], hl), T)
                _, zz = self.tower_head(self.m.tower_s, h)
                zmax = max(zmax, float((zz[0] - z0[T]).abs().max()))
            dep = None if depth is None else depth[:, T:T + 1]
            pd = None if pdepth is None else pdepth[T:T + 1]
            hl = self.hnorm(self.bb.run(self.embed(ids[:, T:T + 1], dep, present[:, T:T + 1], pd), T))
            dmax = max(dmax, float((hl[0].float() - hn[T].float()).abs().max()))
        out["incr_vs_prefill_max_abs_hn"] = round(dmax, 4)
        out["hn_abs_mean"] = round(float(hn.float().abs().mean()), 4)
        out["incr_vs_prefill_max_abs_cb0_logit"] = round(zmax, 4)
        return out

    # ---- one generated position
    @torch.no_grad()
    def step_codes(self, which, T, tok, dprev, hprev, temp, topk, rep=1.0, hist=None):
        tw, st, dec = (self.m.tower_s, self.ts, self.ds) if which == "speech" else (self.m.tower_p, self.tp, self.dp)
        h = st.run(self.tower_in(tw, [tok], dprev[None], hprev), T)
        hv, z0 = self.tower_head(tw, h)
        z0 = z0[0]
        if rep != 1.0 and hist is not None and len(hist):
            ix = torch.as_tensor(np.unique(np.asarray(hist)), device=self.dl)
            v = z0[ix]
            z0[ix] = torch.where(v > 0, v / rep, v * rep)
        c0 = sample_rows(z0[None], temp, topk, self.gen)
        zb = None
        if which == "pose" and hasattr(self.m, "decp_bb"):                     # D33: + alpha x decp_bb(hn[T - 1]) (hprev = hn[T - 1])
            zb = self.m.decp_bb(hprev) * float(self.m.decp_bb_alpha)
        codes = dec(hv, c0, temp if which == "speech" else self.a.pose_temp, topk if which == "speech" else self.a.pose_topk_light, self.gen, zb=zb)
        return codes.cpu().numpy().astype(np.int64)


# ================================================================================================ audio / words helpers
class Mimi:
    def __init__(self, dev):
        from transformers import MimiModel
        self.dev = dev
        self.m = MimiModel.from_pretrained(MIMI).to(dev).eval()

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


def lane_counts(lane):
    lane = np.asarray(lane)
    return {"frames": int(len(lane)), "word_pieces": int((lane < TE).sum()), "new_word": int((lane == NW).sum()), "pad": int((lane == PAD).sum()),
            "eob": int((lane == EOB).sum())}


def transcript(chans, prompt_t=None, gap_s=1.0):
    """chans {name: words} -> utterance lines '[t] NAME: words' sorted by start (an utterance = one speaker's words with gaps < gap_s;
    overlapping speech shows as interleaved lines), with a marker at the generation start."""
    utt = []
    for n, ws in chans.items():
        cur = None
        for w in ws:
            split = prompt_t is not None and cur is not None and cur["t0"] < prompt_t <= w["t"]
            if cur is None or w["t"] - cur["t1"] >= gap_s or split:
                cur = {"t0": w["t"], "t1": w["t"], "n": n, "w": []}
                utt.append(cur)
            cur["w"].append(w["text"]); cur["t1"] = w["t"]
    utt.sort(key=lambda u: (u["t0"], u["n"]))
    lines, mark = [], prompt_t is not None
    for u in utt:
        if mark and u["t0"] >= prompt_t:
            lines.append(f"---------------- generation starts at {prompt_t:.2f}s ----------------"); mark = False
        lines.append(f"[{u['t0']:7.2f}s] {u['n']}: {' '.join(u['w'])}")
    if mark:
        lines.append(f"---------------- generation starts at {prompt_t:.2f}s ----------------")
    return "\n".join(lines)


# ================================================================================================ speech
SIL_CODES = "/data/home/amartyavarma/setup-logs/s3hestia/ourrows/silence_codes.json"   # = the trainer's xdep 'silent' / s3probe / Demeter


def sil_seq(n):
    """[n, 32] global ids: Mimi codes of digital silence (the trainer's xdep silence sequence, tiled over time)."""
    q = np.asarray(json.load(open(SIL_CODES))["seq"], np.int64)        # [32, L]
    assert q.shape[0] == DEPTH and q.min() >= 0 and q.max() < CBS, q.shape
    return q[:, np.arange(n) % q.shape[1]].T + OFF


def speech_prefix(a, mimi):
    """custom system prompt -> sysprompt.build_prefix dict."""
    import sysprompt as SPR
    import persona_text_s3q as PT
    if PT._TOK[0] is None:                                               # data_ppm._tok_init: the Qwen3-32B tokenizer
        PT._TOK[0] = tok()
    segs = []
    for c, w in ((0, a.voice_a), (1, a.voice_b)):
        if w:
            codes = mimi.encode(read_wav24(w))
            L = codes.shape[1]
            assert L >= 38, ("voice clip shorter than 3 s", w, L / FPS)
            codes = codes[:, :125]
            L = codes.shape[1]
            src = {"codes_user": codes, "codes_assistant": codes}
            segs.append({"kind": "speech", "track": c, "w0": 0, "L": L, "src_row": src, "src_side": "user", "blank_lane": True})
    for c, (p, ctx) in ((0, (a.persona_a, a.context_a)), (1, (a.persona_b, a.context_b))):
        av = {k: v for k, v in (("persona", p), ("starting_context", ctx)) if v}
        if av:
            ids = [PT.SOP] + PT.encode(PT.block_text(av, [k for k in PT.SECTIONS if k in av])) + [PT.EOP]
            segs.append({"kind": "text", "track": c, "ids": ids})
    if not segs:
        segs = [{"kind": "text", "track": 1, "ids": []}]
    sp = SPR.build_prefix({}, segs, {"new_word": True})
    assert sp["P"] <= 2000, ("system prompt over the training cap (2,000 positions)", sp["P"])
    return sp


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


def run_parity_full(a, g, out):
    """teacher-forced losses over FULL eval rows (the trainer's eval_speech_s3 lane-B micros, files B-000..), no generation: checks this
    inference stack against the trainer's logged eval numbers (per-row means, then the mean over rows)."""
    res = []
    for i in range(a.parity_full):
        mic = pickle.load(open(f"{SPEECH_EVAL}/B-{i:03d}.pkl", "rb"))
        n = mic["ids"].shape[1]
        r = g.parity(mic["ids"], mic["depth"], mic["present"], None, "speech", int(mic["P"]), K=4, light_pos=mic["light"])
        res.append(r)
        log(f"[parity-full] B-{i:03d} (P {mic['P']}, {n} positions): " + str({k: v for k, v in r.items() if k != "sums"}))
    keys = [k for k in res[0] if isinstance(res[0][k], float)]
    mean = {k: round(float(np.mean([r[k] for r in res if r.get(k) is not None])), 4) for k in keys}
    S = {k: sum(r["sums"][k] for r in res) for k in res[0]["sums"]}
    mean["POOLED_word_ce_norm"] = round(S["word_norm"] / S["n_word"], 4)
    mean["POOLED_text_ce"] = round(S["text_ce"] / S["n_text"], 4)
    mean["POOLED_cb0_ce"] = round(S["cb0"] / S["n_cb0"], 4)
    mean["POOLED_light_ce"] = round(S["light"] / S["n_light"], 4)
    json.dump({"rows": res, "mean": mean}, open(f"{out}/parity_full.json", "w"), indent=1)
    log(f"[parity-full] mean over {len(res)} rows: {mean}")
    return {"task": "speech", "parity_full": mean}


def run_speech(a, g, mimi, out):
    if a.parity_full:
        return run_parity_full(a, g, out)
    Npc, Ng = int(round(a.prompt_s * FPS)), int(round(a.gen_s * FPS))
    if a.a_wav:
        aw = read_wav24(a.a_wav)
        acodes = mimi.encode(aw)
        nconv = Npc + Ng
        ids_c = np.full((2, nconv), PAD, np.int64)
        dep_c = np.full((2, nconv + CD, DEPTH), -1, np.int64)
        fa = min(acodes.shape[1], nconv)
        if a.a_words:
            ids_c[0] = words_lane(json.load(open(a.a_words)), nconv)
        dep_c[0, CD:CD + fa] = acodes[:, :fa].T + OFF
        dep_c[0, CD + fa:] = sil_seq(nconv - fa)
        dep_c[1, CD:] = sil_seq(nconv)                                             # B silent during the prompt (overwritten by generation)
        ids_c = np.concatenate([ids_c, np.full((2, CD), PAD, np.int64)], 1)
        pres_c = np.ones((2, nconv + CD), np.float32)
        conv = {"ids": ids_c, "depth": dep_c, "present": pres_c}
        src = {"a_wav": a.a_wav, "a_words": a.a_words}
        P0 = 0
        mic = None
    else:
        mic = pickle.load(open(f"{SPEECH_EVAL}/{a.row}.pkl", "rb"))
        assert mic["kind"] == "speech"
        P0 = int(mic["P"])
        conv = {k: mic[k][:, P0:] for k in ("ids", "depth", "present")}
        src = {"evalset": SPEECH_EVAL, "row_file": a.row, "prov": json.loads(mic["prov"])}
    if a.sys == "row":
        assert mic is not None, "--sys row needs a row (no --a-wav)"
        pre = {k: mic[k][:, :P0] for k in ("ids", "depth", "present")}
    else:
        if a.sys == "empty":
            for f in ("persona_a", "persona_b", "context_a", "context_b", "voice_a", "voice_b"):
                setattr(a, f, "")
        sp = speech_prefix(a, mimi)
        pre = {k: sp[k] for k in ("ids", "depth", "present")}
    P = pre["ids"].shape[1]
    ids = np.concatenate([pre["ids"], conv["ids"]], 1)
    depth = np.concatenate([pre["depth"], conv["depth"]], 1)
    present = np.concatenate([pre["present"], conv["present"]], 1)
    Np, Tend = P + Npc, P + Npc + Ng
    Tdata = ids.shape[1]
    if Tend > Tdata:                                                     # beyond the row: A silent
        k = Tend - Tdata
        ids = np.concatenate([ids, np.full((2, k), PAD, np.int64)], 1)
        depth = np.concatenate([depth, np.stack([sil_seq(k)] * 2)], 1)
        present = np.concatenate([present, np.ones((2, k), np.float32)], 1)
    gt = {"ids": ids.copy(), "depth": depth.copy()}
    if a.a_mode == "silence":                                            # A: PAD lane from the generation start, silent codes after CD
        ids[0, Np:] = PAD
        depth[0, Np + CD:] = sil_seq(depth.shape[1] - Np - CD)
    log(f"[speech] {src.get('row_file', src.get('a_wav'))}: system prompt P {P} ({a.sys}), prompt {a.prompt_s}s ({Npc} frames), "
        f"generate {a.gen_s}s ({Ng} frames), A = {a.a_mode}")
    par = None
    if a.parity:
        par = g.parity(ids[:, :Np], depth[:, :Np], present[:, :Np], None, "speech", P)
        log(f"[parity] {par}")
    hn = g.prefill(ids[:, :Np], depth[:, :Np], present[:, :Np], None, "speech")
    log(f"[speech] prefill {Np} positions done")
    gram = Grammar(g.V, g.dl, n_in=g.n_in)                              # 10-28: no lane grammar (in-vocab bound only)
    lane_b = ids[1].copy()
    dep_b = depth[1].copy()
    hlast = hn[-1:]
    ts = []
    for T in range(Np, Tend):
        t1 = time.time()
        z = g.text_logits(hlast)
        tokb = sample(z, a.text_temp, a.text_topk, g.gen, gram.mask(int(lane_b[T - 1])))
        lane_b[T] = tokb
        assert T >= P + CD
        dep_b[T] = g.step_codes("speech", T, tokb, dep_b[T - 1], hlast, a.audio_temp, a.audio_topk) + OFF
        ids_t = np.array([[ids[0, T]], [tokb]], np.int64)
        dep_t = np.stack([depth[0, T:T + 1], dep_b[T:T + 1]])
        e = g.embed(ids_t, dep_t, present[:, T:T + 1], None)
        hlast = g.hnorm(g.bb.run(e, T))
        ts.append(time.time() - t1)
        if (T - Np) % 125 == 0:
            log(f"[speech] gen frame {T - Np}/{Ng} ({np.mean(ts):.3f}s/frame)")
    # ---- outputs
    F_ = Tend - P - CD                                                   # conversation frames with codes in [0, F_)
    cb = (dep_b[P + CD:Tend] - OFF).T
    ca = (depth[0, P + CD:Tend] - OFF).T
    assert cb.min() >= 0 and cb.max() < CBS and ca.min() >= 0, ("codes out of range", cb.min(), cb.max(), ca.min())
    wb, wa = mimi.decode(cb), mimi.decode(ca)
    sp = Npc * 1920
    write_wav(f"{out}/b.wav", wb)
    write_wav(f"{out}/b_gen.wav", wb[sp:])
    write_wav(f"{out}/a.wav", wa)
    n = max(len(wa), len(wb))
    st = np.zeros((n, 2), np.float32)
    st[:len(wa), 0], st[:len(wb), 1] = wa, wb
    write_wav(f"{out}/stereo.wav", st)
    words = {"B": lane_words(lane_b[:Tend], P, P, Np), "A": lane_words(ids[0, :Tend], P, P, Np)}
    gtw = {"B": lane_words(gt["ids"][1, :min(Tend, Tdata)], P, P, Np), "A": lane_words(gt["ids"][0, :min(Tend, Tdata)], P, P, Np)}
    tr = transcript({"A (fed)": words["A"], "B": words["B"]}, a.prompt_s)
    open(f"{out}/transcript.txt", "w").write("# B before the marker = the real conversation (prompt); after it = model output. A is always fed (never predicted). Text = the model's own word lanes (no ASR).\n" + tr + "\n")
    json.dump({"B": words["B"], "A": words["A"], "gt_continuation": {"B": [w for w in gtw["B"] if not w["prompt"]]},
               "lane_B": lane_b[P:Tend].tolist(), "lane_A": ids[0, P:Tend].tolist(), "t0_frame": 0}, open(f"{out}/words.json", "w"))
    np.savez_compressed(f"{out}/codes.npz", b=cb.astype(np.int16), a=ca.astype(np.int16), lane_b=lane_b[P:Tend], lane_a=ids[0, P:Tend])
    gen_words = [w for w in words["B"] if not w["prompt"]]
    meta = {"task": "speech", "source": src, "P": P, "sys": a.sys, "prompt_s": a.prompt_s, "gen_s": a.gen_s, "a_mode": a.a_mode,
            "frames_with_codes": F_, "b_audio": audio_stats(wb), "b_gen_audio": audio_stats(wb[sp:]), "a_audio": audio_stats(wa),
            "b_gen_lane": lane_counts(lane_b[Np:Tend]), "b_gen_words": len(gen_words), "b_gen_text": " ".join(w["text"] for w in gen_words),
            "gt_b_cont_text": " ".join(w["text"] for w in gtw["B"] if not w["prompt"]),
            "b_gen_cb0_unique": int(len(np.unique(cb[0, Npc:]))), "speed_s_per_frame": round(float(np.mean(ts)), 4), "parity": par}
    return meta


# ================================================================================================ pose
def run_pose(a, g, mimi, out):
    import data_ppm as D
    row = pickle.load(open(f"{POSE_EVAL}/{a.pose_row}.pkl", "rb"))
    cfg = D.default_cfg(pose_delay=CD, light_frac=0.1, pose_light_frac=0.1)
    mic = D.pose_micro_ppm(row, cfg, 0)
    mic = D.packed(mic, cfg)
    c0 = row["clips"][0]
    nclip = int(c0["n"])
    Npc, Ng = int(round(a.prompt_s * FPS)), int(round(a.gen_s * FPS))
    Np, Tend = 1 + Npc, 1 + Npc + Ng                                     # SLOT at 0, clip frame k at position 1 + k
    if Tend > 1 + nclip:
        log(f"[pose] WARNING clip 0 has {nclip} frames ({nclip / FPS:.1f}s) < prompt + gen; the model continues past the clip's end "
            f"(ground truth then shows the next clip / gap)")
    ids, depth, present, pdepth = mic["ids"], mic["depth"], mic["present"], mic["pdepth"]
    assert ids.shape[1] >= Tend + CD
    log(f"[pose] {a.pose_row} ({row['row_key']}, {row['dataset']}, clip 0 {nclip / FPS:.1f}s {c0.get('language')}, "
        f"{c0.get('orig_w')}x{c0.get('orig_h')}): prompt {a.prompt_s}s, generate {a.gen_s}s")
    hn = g.prefill(ids[:, :Np], depth[:, :Np], present[:, :Np], pdepth[:Np], "pose")
    gram = Grammar(g.V, g.dl, n_in=g.n_in)                              # 10-28: no lane grammar (in-vocab bound only)
    lane_b, dep_b, pdep = ids[1].copy(), depth[1].copy(), pdepth.copy()
    hlast = hn[-1:]
    hist, ts = [], []
    for T in range(Np, Tend):
        t1 = time.time()
        z = g.text_logits(hlast)
        tokb = sample(z, a.text_temp, a.text_topk, g.gen, gram.mask(int(lane_b[T - 1])))
        lane_b[T] = tokb
        dep_b[T] = g.step_codes("speech", T, tokb, dep_b[T - 1], hlast, a.audio_temp, a.audio_topk) + OFF
        pc = g.step_codes("pose", T, tokb, pdep[T - 1], hlast, a.pose_temp, a.pose_topk, a.pose_rep,
                          hist[-a.pose_rep_win:] if a.pose_rep_win else hist)
        pdep[T] = pc + OFF
        hist.append(int(pc[0]))
        ids_t = np.array([[PAD], [tokb]], np.int64)
        dep_t = np.stack([depth[0, T:T + 1], dep_b[T:T + 1]])
        e = g.embed(ids_t, dep_t, present[:, T:T + 1], pdep[T:T + 1])
        hlast = g.hnorm(g.bb.run(e, T))
        ts.append(time.time() - t1)
        if (T - Np) % 125 == 0:
            log(f"[pose] gen frame {T - Np}/{Ng} ({np.mean(ts):.3f}s/frame)")
    # clip frame k sits at position 1 + k + CD in depth / pdepth: frames [0, Tend - 1 - CD)
    F_ = Tend - 1 - CD
    cb = (dep_b[1 + CD:Tend] - OFF).T
    pc = (pdep[1 + CD:Tend] - OFF).T
    assert cb.shape[1] == F_ and cb.min() >= 0 and pc.min() >= 0 and pc.max() < CBS, (cb.shape, cb.min(), pc.min(), pc.max())
    nf = min(Tend + CD, ids.shape[1]) - 1 - CD
    gpc = (pdepth[1 + CD:1 + CD + nf] - OFF).T
    gcb = (depth[1, 1 + CD:1 + CD + nf] - OFF).T
    wb = mimi.decode(cb)
    write_wav(f"{out}/b.wav", wb)
    write_wav(f"{out}/b_gen.wav", wb[Npc * 1920:])
    gt_ok = gpc.shape[1] >= F_ and (gpc[:, :F_] >= 0).all() and (gcb[:, :F_] >= 0).all()
    if gt_ok:
        write_wav(f"{out}/gt.wav", mimi.decode(gcb[:, :F_]))
    words = {"B": lane_words(lane_b[:Tend], 1, 1, Np)}
    gtw = lane_words(ids[1, :Tend], 1, 1, Np)
    open(f"{out}/transcript.txt", "w").write("# B before the marker = the real conversation (prompt); after it = model output. A is always fed (never predicted). Text = the model's own word lanes (no ASR).\n" + transcript({"B": words["B"]}, a.prompt_s) + "\n\nGROUND TRUTH CONTINUATION: "
                                              + " ".join(w["text"] for w in gtw if not w["prompt"]) + "\n")
    json.dump({"B": words["B"], "gt_continuation": [w for w in gtw if not w["prompt"]], "lane_B": lane_b[1:Tend].tolist()},
              open(f"{out}/words.json", "w"))
    np.savez_compressed(f"{out}/codes.npz", audio=cb.astype(np.int16), pose=pc.astype(np.int16), lane_b=lane_b[1:Tend])
    vid = render_pose(a, pc, gpc[:, :F_] if gt_ok else None, wb, c0, Npc, out)
    gen_words = [w for w in words["B"] if not w["prompt"]]
    return {"task": "pose", "source": {"evalset": POSE_EVAL, "row_file": a.pose_row, "row_key": row["row_key"], "dataset": row["dataset"],
                                        "clip0": {k: c0.get(k) for k in ("clip_id", "video_id", "language", "orig_h", "orig_w", "n")}},
            "prompt_s": a.prompt_s, "gen_s": a.gen_s, "frames_with_codes": F_, "b_audio": audio_stats(wb), "b_gen_audio": audio_stats(wb[Npc * 1920:]),
            "b_gen_lane": lane_counts(lane_b[Np:Tend]), "b_gen_words": len(gen_words), "b_gen_text": " ".join(w["text"] for w in gen_words),
            "gt_cont_text": " ".join(w["text"] for w in gtw if not w["prompt"]),
            "pose_gen_cb0_unique": int(len(np.unique(pc[0, Npc:]))), "audio_gen_cb0_unique": int(len(np.unique(cb[0, Npc:]))),
            "video": vid, "speed_s_per_frame": round(float(np.mean(ts)), 4)}


POSE_EDGES = {
    "face": [(0, 1), (0, 2), (1, 3), (2, 4)],
    "body": [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)],
    "lh": [(i + 13, j + 13) for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
                                         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]] + [(9, 13)],
    "rh": [(i + 34, j + 34) for i, j in [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
                                         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]] + [(10, 34)],
}
POSE_COL = {"face": (200, 200, 200), "body": (80, 170, 255), "lh": (90, 220, 120), "rh": (255, 140, 80)}     # BGR


def decode_pose(ptok, pc, c0, dev):
    """pose codes [32, F] -> positions [2F, 55, 2] (y, x in [0, 1]) and confidences [2F, 55] (the clip's frame-0 root + shoulder width)."""
    from pose_tokenizer.data.kinematic import NUM_JOINTS, offsets_to_positions
    with torch.no_grad():
        rec = ptok.decode([torch.as_tensor(pc[k].astype(np.int64), device=dev) for k in range(DEPTH)]).float().cpu().numpy()
    T = rec.shape[0]
    r3 = rec.reshape(T, NUM_JOINTS, 3)
    root = np.asarray(c0["frame0_root"], np.float32).reshape(1, 2)
    wd = np.float32(c0["frame0_width"])
    gs = np.repeat(np.concatenate([root, [[wd]]], 1), T, 0)
    pos = offsets_to_positions(gs[:, :2], r3[:, :, :2] * gs[:, 2:3, None])
    return np.asarray(pos), np.clip(r3[:, :, 2], 0, 1)


def render_pose(a, pc, gpc, wav, c0, Npc, out):
    import cv2
    sys.path.insert(0, "/data/home/amartyavarma/ref/pose-tokenizer")
    from pose_tokenizer import PoseTokenizer
    dev = "cuda:%d" % torch.cuda.current_device()
    ptok = PoseTokenizer.from_pretrained(POSE_TOK[0], device=dev, revision=POSE_TOK[1])
    pos, conf = decode_pose(ptok, pc, c0, dev)
    gpos = gconf = None
    if gpc is not None:
        gpos, gconf = decode_pose(ptok, gpc, c0, dev)
    oh, ow = float(c0.get("orig_h") or 1280), float(c0.get("orig_w") or 720)
    H = 640
    W = int(round(H * ow / oh)) // 2 * 2
    T = pos.shape[0]
    prompt_f = 2 * Npc                                                   # 25 fps frames of the prompt

    def draw(p, c, label, gen):
        img = np.full((H, W, 3), 24, np.uint8)
        for g_, edges in POSE_EDGES.items():
            col = POSE_COL[g_]
            for i, j in edges:
                al = 0.15 + 0.85 * min(c[i], c[j])
                if al < 0.2:
                    continue
                pt1 = (int(p[i, 1] * W), int(p[i, 0] * H)); pt2 = (int(p[j, 1] * W), int(p[j, 0] * H))
                cl = tuple(int(x * al) for x in col)
                cv2.line(img, pt1, pt2, cl, 4 if g_ == "body" else 2, cv2.LINE_AA)
        cv2.putText(img, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        if gen is not None:
            cv2.putText(img, "GENERATED" if gen else "PROMPT (ground truth)", (10, H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (80, 200, 255) if gen else (180, 180, 180), 2, cv2.LINE_AA)
        return img
    frames = []
    for f in range(T):
        im = draw(pos[f], conf[f], "model", f >= prompt_f)
        if gpos is not None:
            gi = draw(gpos[min(f, len(gpos) - 1)], gconf[min(f, len(gpos) - 1)], "ground truth", None)
            im = np.concatenate([im, np.full((H, 8, 3), 70, np.uint8), gi], 1)
        frames.append(im)
    vid = np.stack(frames)[:, :, :, ::-1]                               # BGR -> RGB
    hw = vid.shape[1:3]
    import imageio_ffmpeg
    import subprocess
    write_wav(f"{out}/_v.wav", wav)
    cmd = [imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-hide_banner", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{hw[1]}x{hw[0]}", "-r", "25", "-i", "-", "-i", f"{out}/_v.wav", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-shortest", f"{out}/pose.mp4"]
    subprocess.run(cmd, input=np.ascontiguousarray(vid).tobytes(), check=True)
    os.remove(f"{out}/_v.wav")
    fi = min(T - 1, prompt_f + (T - prompt_f) // 2)
    cv2.imwrite(f"{out}/frame.png", frames[fi])
    json.dump({"fps": 25, "frames": int(T), "prompt_frames": int(prompt_f), "pos": np.round(pos.reshape(T, -1), 4).tolist(),
               "conf": np.round(conf, 2).tolist()}, open(f"{out}/pose_keypoints.json", "w"))
    mv = float(np.abs(np.diff(pos[prompt_f:], axis=0)).mean()) if T - prompt_f > 2 else 0.0
    return {"mp4": f"{out}/pose.mp4", "frame_png": f"{out}/frame.png", "frame_index": int(fi), "frames_25fps": int(T),
            "gen_mean_joint_motion_per_frame": round(mv, 5), "x_range": [round(float(pos[..., 1].min()), 3), round(float(pos[..., 1].max()), 3)],
            "y_range": [round(float(pos[..., 0].min()), 3), round(float(pos[..., 0].max()), 3)], "mean_conf_gen": round(float(conf[prompt_f:].mean()), 3)}


# ================================================================================================ text
def run_text(a, g, out):
    if a.text_mode == "duplex":
        return run_duplex(a, g, out)
    return run_chat(a, g, out)


def run_duplex(a, g, out):
    import data_ppm as D
    import txtdup_s3q as TXD
    st = TXD.Store(GEMMA)
    ev = st.pack("eval", a.pack)
    i = ev[a.conv % len(ev)]
    rec = st.get(i)
    cfg = D.default_cfg()
    mic = D.text_micro([rec], cfg, [a.swap])
    ids, present = mic["ids"], mic["present"]
    N = ids.shape[1]
    Npc, Ng = int(round(a.prompt_s * FPS)), int(round(a.gen_s * FPS))
    Np, Tend = Npc, min(Npc + Ng, N)
    log(f"[duplex] Gemma eval pack {a.pack} conv {i} ({rec['conv_id']}, {N} frames = {N / FPS:.0f}s, swap {a.swap}): prompt {a.prompt_s}s, "
        f"generate {(Tend - Np) / FPS:.1f}s")
    hn = g.prefill(ids[:, :Np], None, present[:, :Np], None, "text")
    gram = Grammar(g.V, g.dl, n_in=g.n_in)                              # 10-28: no lane grammar (in-vocab bound only)
    lane_b = ids[1].copy()
    hlast = hn[-1:]
    for T in range(Np, Tend):
        z = g.text_logits(hlast)
        lane_b[T] = sample(z, a.text_temp, a.text_topk, g.gen, gram.mask(int(lane_b[T - 1])))
        e = g.embed(np.array([[ids[0, T]], [lane_b[T]]], np.int64), None, present[:, T:T + 1], None)
        hlast = g.hnorm(g.bb.run(e, T))
    wb, wa, gb = lane_words(lane_b[:Tend], 0, 0, Np), lane_words(ids[0, :Tend], 0, 0, Np), lane_words(ids[1, :Tend], 0, 0, Np)
    tr = transcript({"A (fed)": wa, "B": wb}, a.prompt_s)
    gtr = transcript({"A": wa, "B (data)": gb}, a.prompt_s)
    open(f"{out}/transcript.txt", "w").write("# B before the marker = the real conversation (prompt); after it = model output. A is always fed (never predicted). Text = the model's own word lanes (no ASR).\n" + tr + "\n\n==== ground truth (same window) ====\n" + gtr + "\n")
    json.dump({"B": wb, "A": wa, "gt_B": gb, "lane_B": lane_b[:Tend].tolist()}, open(f"{out}/text.json", "w"))
    gw = [w for w in wb if not w["prompt"]]
    return {"task": "text", "mode": "duplex", "source": {"store": GEMMA, "pack": a.pack, "conv_index": int(i), "conv_id": rec["conv_id"], "swap": a.swap},
            "prompt_s": a.prompt_s, "gen_s": round((Tend - Np) / FPS, 2), "b_gen_lane": lane_counts(lane_b[Np:Tend]), "b_gen_words": len(gw),
            "b_gen_text": " ".join(w["text"] for w in gw), "transcript": tr}


def chat_user_frames(msgs_so_far, user_text, rng_seed):
    """the user turn's frames as training renders them (render_q timing): render [.., user, assistant 'ok'] and take lane A from the
    user's first content frame to its last, plus the render's gap to the assistant's first frame."""
    q1 = "/home/param/websites/pose-duplex-engine/deps/qwen-1ch"
    if q1 not in sys.path:
        sys.path.append(q1)
    import render_q as RM
    import chat_ppm as CH
    tk = RM.load_tok()
    cfg = RM.cfg_from({})
    row = RM.build_row_q([{"role": "user", "content": user_text}, {"role": "assistant", "content": "Okay."}], cfg, tk,
                         np.random.default_rng(rng_seed))
    lanes, P, _ = CH.conv_lanes(row, keep_eob=True)
    A, B = lanes[0], lanes[1]
    fa = np.flatnonzero(A != PAD)
    fb = np.flatnonzero(B != PAD)
    assert len(fa) and len(fb), "render produced no user / assistant frames"
    return A[fa[0]:fa[-1] + 1].copy(), int(fb[0] - fa[-1] - 1), int(fa[0])


def run_chat(a, g, out):
    users = list(a.user or [])
    system = a.system
    if a.chat_file:
        msgs = json.load(open(a.chat_file))
        users += [m["content"] for m in msgs if m["role"] == "user"]
        system = system or next((m["content"] for m in msgs if m["role"] == "system"), "")
    if getattr(a, "interactive", False):
        def typed():                                                     # live CLI: one user turn per line; blank / EOF / "/exit" ends
            while True:
                try:
                    u = input("\nyou> ").strip()
                except EOFError:
                    return
                if not u or u in ("/exit", "/quit"):
                    return
                yield u
        users = typed()
    else:
        assert users, "chat needs --user or --chat-file (or --interactive)"
    tk = tok()
    ids = [[], []]
    pres = [[], []]

    def push(ta, tb, pa=1.0, pb=1.0, fp=False):
        ids[0].append(ta); ids[1].append(tb); pres[0].append(pa); pres[1].append(pb)
    if system:
        for t in tk(system, add_special_tokens=False)["input_ids"]:
            push(PAD, int(t), 0.0, 1.0, True)
    push(PAD, PAD, fp=True)                                              # the 1ch PREPEND frame
    turns, pos = [], None
    g_lane = Grammar(g.V, g.dl, n_in=g.n_in)                            # 10-28: no lane grammar (in-vocab bound only)
    maxf = int(round(a.max_turn_s * FPS))
    T, hlast = 0, None
    for k, u in enumerate(users):
        ua, gap, lead = chat_user_frames(None, u, a.seed + k)
        if k == 0:
            for _ in range(lead - 1):                                    # lead counts the PREPEND frame already pushed
                push(PAD, PAD, fp=True)
        else:
            for _ in range(10):
                push(PAD, PAD, fp=True)
        for t in ua:
            push(int(t), PAD, fp=True)
        # teacher-forced span [T, len): run the backbone over it (B forced PAD / system text)
        n0 = len(ids[0])
        arr = np.array(ids, np.int64)[:, T:n0]
        pr = np.array(pres, np.float32)[:, T:n0]
        h = g.bb.run(g.embed(arr, None, pr, None), T)
        hlast = g.hnorm(h)[-1:]
        T = n0
        start = T
        reply = []
        live = getattr(a, "interactive", False)
        shown = ""
        if live:
            print("\nmodel> ", end="", flush=True)
        for _ in range(maxf):
            z = g.text_logits(hlast)
            prev = ids[1][-1]
            tb = sample(z, a.text_temp, a.text_topk, g.gen, g_lane.mask(prev))
            push(PAD, tb)
            reply.append(tb)
            if live and tb < TE:                                         # live stream: print the new text as each word piece arrives
                cur = tk.decode([t for t in reply if t < TE])
                if cur.startswith(shown):
                    print(cur[len(shown):], end="", flush=True)
                    shown = cur
            e = g.embed(np.array([[PAD], [tb]], np.int64), None, np.ones((2, 1), np.float32), None)
            hlast = g.hnorm(g.bb.run(e, T))
            T += 1
            if tb == EOB:
                break
        words = [t for t in reply if t < TE]
        turns.append({"user": u, "assistant": tk.decode(words), "frames": len(reply), "ended_with_im_end": bool(reply and reply[-1] == EOB),
                      "first_word_frame": next((i for i, t in enumerate(reply) if t < TE), None), "start_frame": start})
        log(f"[chat] turn {k}: {len(reply)} frames, EOB {turns[-1]['ended_with_im_end']}")
        if getattr(a, "interactive", False):
            print(turns[-1]["assistant"][len(shown):] + ("" if turns[-1]["ended_with_im_end"] else "  [no <|im_end|> within --max-turn-s]"),
                  flush=True)
    lines = ([f"SYSTEM: {system}"] if system else []) + [s for t in turns for s in (f"USER: {t['user']}", f"ASSISTANT: {t['assistant']}"
                                                         + ("" if t["ended_with_im_end"] else "  [no <|im_end|> within --max-turn-s]"))]
    tr = "\n\n".join(lines)
    open(f"{out}/transcript.txt", "w").write(tr + "\n")
    json.dump({"turns": turns, "lane_A": ids[0], "lane_B": ids[1]}, open(f"{out}/text.json", "w"))
    return {"task": "text", "mode": "chat", "turns": turns, "transcript": tr}


# ================================================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--task", required=True, choices=("speech", "pose", "text"))
    ap.add_argument("--devs", default="0,1", help="CUDA devices (backbone spread over all; head/towers/decoders on the last)")
    ap.add_argument("--out-root", default=SAMPLES)
    ap.add_argument("--tag", default="", help="checkpoint tag for the output dir (default <run>-<ckpt dir name>)")
    ap.add_argument("--name", default="", help="sample dir name (default from the source + seed)")
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--prompt-s", type=float, default=None, help="seconds of real conversation prefilled (default speech 20, pose 4, duplex 20)")
    ap.add_argument("--gen-s", type=float, default=None, help="seconds generated (default speech 30, pose 20, duplex 40)")
    ap.add_argument("--text-temp", type=float, default=0.8)
    ap.add_argument("--text-topk", type=int, default=50)
    ap.add_argument("--audio-temp", type=float, default=0.8)
    ap.add_argument("--audio-topk", type=int, default=250)
    ap.add_argument("--pose-temp", type=float, default=0.8)
    ap.add_argument("--pose-topk", type=int, default=30, help="pose cb0 top-k")
    ap.add_argument("--pose-topk-light", type=int, default=30, help="pose cb1..31 (0.6B decoder) top-k")
    ap.add_argument("--pose-rep", type=float, default=1.2, help="repetition penalty on pose cb0 history (SAM / Vansh reference; 1.0 = off)")
    ap.add_argument("--pose-rep-win", type=int, default=0, help="frames of pose cb0 history for --pose-rep (0 = all generated)")
    ap.add_argument("--parity", type=int, default=0, help="speech: teacher-forced loss + KV-cache checks on the prompt before generating")
    ap.add_argument("--parity-full", type=int, default=0, help="speech: teacher-forced losses on the first N full eval rows only (no generation)")
    ap.add_argument("--free-text", type=int, default=1, help="IGNORED (10-28 user order: no lane grammar, ever)")
    ap.add_argument("--allow-eob", type=int, default=1, help="IGNORED (10-28 user order: no lane grammar; EOB allowed like any token)")
    ap.add_argument("--decp-bb-alpha", type=float, default=None, help="D33 link scale override (default: the trainer's at the checkpoint step)")
    # speech
    ap.add_argument("--row", default="B-092", help="speech eval micro: {B,A}-000..102 (B = dataset orientation, A = flipped)")
    ap.add_argument("--a-mode", default="row", choices=("row", "silence"))
    ap.add_argument("--sys", default="row", choices=("row", "empty", "custom"))
    for c in "ab":
        ap.add_argument(f"--persona-{c}", default="")
        ap.add_argument(f"--context-{c}", default="")
        ap.add_argument(f"--voice-{c}", default="", help="wav clip (3-10 s used; longer is cropped to 10 s)")
    ap.add_argument("--a-wav", default="", help="channel A audio (replaces the row's conversation)")
    ap.add_argument("--a-words", default="", help="json [[word, start_s, end_s], ...] for --a-wav's text lane")
    # pose
    ap.add_argument("--pose-row", default="row114976", help="evalsets/pose_infer/<name>.pkl (MANIFEST.json lists them)")
    # text
    ap.add_argument("--text-mode", default="duplex", choices=("duplex", "chat"))
    ap.add_argument("--pack", type=int, default=0, help="duplex: Gemma eval pack")
    ap.add_argument("--conv", type=int, default=0, help="duplex: conversation index inside the pack")
    ap.add_argument("--swap", type=int, default=0, help="duplex: 0 = data speaker b on channel B")
    ap.add_argument("--user", action="append", help="chat: a user turn (repeat for multi-turn)")
    ap.add_argument("--system", default="")
    ap.add_argument("--chat-file", default="")
    ap.add_argument("--interactive", action="store_true", help="chat: type user turns live (needs a terminal; infer_ppm.sh adds --pty)")
    ap.add_argument("--max-turn-s", type=float, default=120.0)
    a = ap.parse_args()
    dflt = {"speech": (20.0, 30.0), "pose": (4.0, 20.0), "text": (20.0, 40.0)}[a.task]
    a.prompt_s = dflt[0] if a.prompt_s is None else a.prompt_s
    a.gen_s = dflt[1] if a.gen_s is None else a.gen_s
    assert os.environ.get("GEMM_PRECISION", "bf16") == "bf16", "bf16 only"
    devs = [torch.device(f"cuda:{int(x)}") for x in a.devs.split(",")]
    torch.cuda.set_device(devs[-1])
    ci = ckpt_info(a.ckpt)
    if a.decp_bb_alpha is not None:
        assert ci["decp_bb"], "--decp-bb-alpha: the checkpoint has no D33 decp_bb link"
        ci["decp_bb_alpha"] = float(a.decp_bb_alpha)
    tag = a.tag or ci["tag"]
    if a.task == "speech":
        nm = a.name or (f"parity-full-{a.parity_full}" if a.parity_full else None) or (f"{a.row}-p{a.prompt_s:g}-g{a.gen_s:g}-{a.sys}-A{a.a_mode}-s{a.seed}" if not a.a_wav else
                        f"awav-{os.path.splitext(os.path.basename(a.a_wav))[0]}-s{a.seed}")
    elif a.task == "pose":
        nm = a.name or f"{a.pose_row}-p{a.prompt_s:g}-g{a.gen_s:g}-s{a.seed}"
    else:
        nm = a.name or (f"duplex-pack{a.pack}-conv{a.conv}-p{a.prompt_s:g}-g{a.gen_s:g}-s{a.seed}" if a.text_mode == "duplex" else f"chat-s{a.seed}")
    out = f"{a.out_root}/{tag}/{a.task}/{nm}"
    os.makedirs(out, exist_ok=True)
    log(f"[ppm-infer] ckpt {ci['path']} ({ci['kind']}, step {ci['step']}; layers {ci['layers']}/{ci['tower_layers']}/{ci['dec_layers']}) "
        f"task {a.task} -> {out}")
    torch.manual_seed(a.seed)
    with torch.inference_mode():
        m, ldev, cfgs = build_model(ci, devs)
        for d in devs:
            log(f"[mem] {d}: {torch.cuda.memory_allocated(d) / 2 ** 30:.1f} GiB weights")
        lmax = 2304 + 7600 if a.task == "speech" else 7700
        g = PPMGen(m, ldev, cfgs, devs, lmax, a)
        mimi = Mimi(devs[-1]) if a.task in ("speech", "pose") else None
        t1 = time.time()
        res = {"speech": lambda: run_speech(a, g, mimi, out), "pose": lambda: run_pose(a, g, mimi, out), "text": lambda: run_text(a, g, out)}[a.task]()
    res.update({"ckpt": ci["path"], "ckpt_kind": ci["kind"], "step": ci["step"], "tag": tag, "seed": a.seed, "out": out,
                "sampling": {k: getattr(a, k) for k in ("text_temp", "text_topk", "audio_temp", "audio_topk", "pose_temp", "pose_topk",
                                                         "pose_topk_light", "pose_rep", "pose_rep_win")}, "grammar": "none",
                "min_p": None, "audio_rep_pen": 1.0, "cfg": "off", "dtype": "bf16", "gen_wall_s": round(time.time() - t1, 1),
                "made": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "argv": sys.argv})
    json.dump(res, open(f"{out}/meta.json", "w"), indent=1)
    log(f"[ppm-infer] done -> {out}")
    for k in ("b_gen_text", "transcript"):
        if k in res:
            print(f"---- {k} ----\n{res[k][:3000]}", flush=True)
    for k in ("b_audio", "b_gen_audio", "b_gen_lane", "video"):
        if k in res:
            print(f"{k}: {res[k]}", flush=True)


if __name__ == "__main__":
    main()
