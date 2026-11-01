#!/usr/bin/env python3
"""LIVE PPM ENGINE for the Voxtral-A Aloe fine-tune (aloe-m1vox-ft, job 22335): B (words with NEW_WORD timing + Mimi) generated frame by
frame while channel A is HEARD from a live 24 kHz stream through the frozen Voxtral Realtime encoder + the run's trained adapter x s.

MODEL     = the trainer's model_ppm (aloe-m1vox-ft/code copy, VOX EDIT): PPM(nopose=True, vadapter={4 x 1280 -> SwiGLU 4096 -> 5120, norm
          none}) + vgain (s = exp(theta)). No pose parts exist. bf16 weights; backbone layers spread over the GPUs (infer_ppm's split: the
          last GPU also holds head + speech tower + 0.6B decoder); front (Voxtral + adapter), streaming Mimi decoder on the FIRST GPU.
CHECKPOINTS  (a) consolidated bf16 DCP export (export-366: every tensor incl. vadapter.*, vgain.theta), (b) ppm-local-bf16-shards-v1
          copies (the run's node-local exports, or the step-11488 init local-11488 + the step-1972 ASR adapter + the calibrated s init).
          s: the trainer's fp32 s at the checkpoint step from the run's metrics.jsonl (train/vox_a_scale) when available (the exports
          store theta in bf16, a ~1.6 % step in s); else exp(bf16 theta) (or the given s for the init + adapter test path).
LAYOUT    = the training micro (data_aloe_vox.speech_micro_vox): system prompt [0, P) (B's voice clip + persona blocks, A's voice clip
          never), then conversation position P + f = frame f: A = PAD text, no Mimi, present_A = 0, + s x adapter(Voxtral(A frame f))
          (same frame, no delay); B = lane token of frame f, B Mimi at code delay 2 (positions P, P + 1 carry no B codes).
PER FRAME T = P + f (all CUDA graphs except the very first front frame; device-side frame index, no host sync inside a frame):
  front  (GPU 0)  Voxtral streaming step on A samples [1920 f - 330, 1920 (f + 1) + 72) (= training's full-row front up to bf16
                  round-off; 3 ms lookahead) -> adapter -> x s -> a_vec.
  text   (head)   lane_B[T] ~ head(hn[T-1]) (P_B projector + lm_head + own PAD / NEW_WORD / EOB rows), NO lane grammar since 10-28 (was: after NEW_WORD a
                  word piece, after PAD no word piece, EOB off), temperature / top-k / top-p; raw P(PAD), P(NEW_WORD) recorded.
  audio  (head)   speech tower step T (text_emb(lane_B[T]) + code_emb(B codes at T - 1) + cond(hn[T - 1])) -> cb0 (temp / top-k) ->
                  0.6B decoder cb1..31 (fused KV-cached Triton path = the stage-3/4 live engines' FusedLight, temp / top-k) for
                  T >= P + 2; earlier conversation positions only advance the tower's KV cache (no B codes, as training).
  trunk  (both)   e = emb_B(lane_B[T]) + aud_B(B codes of T) + a_vec (A's tables x present 0) -> 64 backbone layers -> hn[T].
  B out  (GPU 0)  streaming Mimi decoder (graphed) on B's codes of T = B frame f - 2.
SAMPLING  = the s4live / model-2 live engine (live_engine.py, copied): live device tensors (one capture serves every setting): in-vocab bound only (NO lane grammar, 10-28) ->
          CFG -> frequency / CTRL repetition penalty (B ring window) -> temperature -> word min-p -> top-k -> top-p; audio: cb0 CFG -> CTRL
          repetition penalty per codebook (window) -> temperature / top-k. Standing rules: temperature 1 everywhere, audio top-k 30 (cb0 + light),
          text top-k off, min-p 0, repetition penalties 1, frequency penalty 0, CFG off (defaults; every knob live on the page), bf16 only.
          10-17: the text knobs act on the WORD stage of FACTORISED TEXT SAMPLING (below; default on, word min-p 0.05 absolute).
ADAPTIVE REPETITION PENALTY (10-17, live knobs, default OFF): per frame T a device-side SPEAKING flag = B's lane holds a real word piece
          (id < TE: not PAD / NEW_WORD / EOB) at some conversation position in [T - speak_window_frames, T - 1] (causal: frame T's own
          token is not yet drawn when the flag is computed, at the top of the text graph). With adaptive_rep_on the frame's audio CTRL
          repetition penalty (cb0 + cb1-31, window audio_rep_win) = audio_rep_speaking | audio_rep_silent and the text one (window
          text_rep_win) = text_rep_speaking | text_rep_silent, in place of the static audio_rep_pen / text_rep; off -> the static values
          exactly (same kernels, same RNG draws: byte-identical output).
FACTORISED TEXT SAMPLING (10-17, user spec, DEFAULT ON: factorised_text 1; live knobs factorised_text, cat_temp): B's lane draw is split
          into stage 1 = the CATEGORY, drawn at its own temperature cat_temp (default 1) from the categories of the trainer's factorised
          text CE (heads_s2q.ce_head with wn; train_aloe_* prep: w_cat_word x WORD + w_tim x PAD + w_nw x NEW_WORD + w_eob x EOB): WORD = all
          word pieces (id < TE) collapsed, P(WORD) = the sum of their softmax; PAD, NEW_WORD and EOB (always allowed since 10-28; before: only when allow_eob, else masked by the
          grammar as before) each their own category; no penalty / top-k / top-p / min-p at this stage. Stage 2, used only when the category
          is WORD = the word piece from the distribution renormalised over word pieces only, with EVERY text knob applied there, over words
          only: frequency penalty + CTRL text repetition penalty (static or adaptive speaking / silent; B's ring windows) -> text_temp ->
          min-p (abs / rel) -> top-k -> top-p (-> word_greedy). PAD / NEW_WORD / EOB are never penalised, truncated or temperature-shifted
          by the word knobs. cat_temp 1 + neutral word knobs = exactly the joint distribution (P(WORD) x P(w | WORD) = P(w)). Separate
          generator for the category draw (gen_c, seed + 2), so factorised_text 0 (the joint fallback, for comparison) = the pre-10-17 path
          byte for byte. Default min-p: 0.05 absolute on the word stage with factorised_text 1; the joint fallback keeps min-p 0 (switching
          the mode moves min-p to that mode's default unless it was changed: MINP_DEFAULT).
PAD BIAS  (10-18, user: 'allow me to downweight pad'; live knob pad_bias, default 0, range -10..+2): added to B's PAD logit of the
          CFG-combined, in-vocab-masked (no grammar, 10-28) lane logits right before t_text_draw, i.e. before any temperature: factorised = the PAD category
          logit of stage 1 (then / cat_temp; cat_temp 1 = exactly pad_bias nats), joint = the PAD token logit (then / text_temp). Negative
          = B stays silent less and speaks more. Device tensor in the text graph (no re-capture); 0 -> z + 0.0 = z: byte-identical output.
          The reported raw P(PAD) / P(NEW_WORD) stay the model's own (unbiased) probabilities; 'as drawn' includes the bias.
ADAPTIVE PAD BIAS (10-24, user: 'if a word hasn't been spoken for 3 s it kicks in, otherwise 1'; live knobs, default OFF in every
          profile: adaptive_pad_on, pad_silent_s, pad_bias_silent, pad_bias_other): per frame T, frames_since = T - (conversation position
          of B's last SPOKEN lane token: a word piece id < TE or NEW_WORD; PAD / EOB / other ids never count), held in the persistent device
          scalar lastspk = P at start() (silence counts from the first conversation frame, so the bias can never kick in before the
          threshold has passed) and updated in-graph from the lane token as finally committed (top of the audio graph, i.e. after the
          cold-start PAD / parity forcing). Causal like _speaking: frame T's own token is not drawn yet when frames_since is computed.
          SILENT iff frames_since x 80 ms >= pad_silent_s, i.e. frames_since >= pad_silent_s x 12.5 - 1e-3 in fp32 (= ceil: 3 s -> 38
          frames = 3.04 s, the same as round(37.5); never before the threshold, also where round() would go down, e.g. 1 s -> 13 frames,
          not 12). The PAD bias in
          effect = adaptive_pad_on ? (SILENT ? pad_bias_silent : pad_bias_other) : pad_bias, added at exactly the PAD BIAS point above
          (factorised and joint alike). Off -> the static pad_bias itself (torch.where picks the same value: byte-identical output);
          on with pad_bias_silent == pad_bias_other == x -> exactly static pad_bias x. Reported per frame (self.padad, read with the
          rep-pen state in result(): one device->host copy as before): on flag, seconds since B's last spoken token, bias in effect.
CFG       batch-2 trunk / tower / head: row 1 = the uncond stream (LivePPM doc); the 0.6B decoder runs for row 0 only."""
import json
import math
import os
import sys
import time

AC = "/home/param/pose-repo/finetune/duplex_distill/production"          # the run's own code (model_ppm VOX EDIT, data, lib, vendor)
PC = "/home/param/pose-repo/train"                         # infer_ppm (Stack / Grammar)
QI = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("PPM_CD", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("VOXTRAL_DIR", "/home/param/models/aloe-ft-vox17127-nopersona/assets/voxtral-realtime")   # = the trainer's VOX_MODEL


def _front_paths(paths):
    for p in reversed(paths):
        if p in sys.path:
            sys.path.remove(p)
        sys.path.insert(0, p)


_front_paths([AC, f"{AC}/lib", f"{AC}/vendor"])
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import model_ppm as M  # noqa: E402   (aloe-m1vox-ft copy)
import fused_triton as FO  # noqa: E402
import fa3_op  # noqa: E402
import speech_q as SP  # noqa: E402
sys.path.insert(0, PC)
import infer_ppm as IP  # noqa: E402   (uses the already-imported model_ppm / speech_q / fused_triton of the run's code)
assert IP.M is M, "infer_ppm must use the run's model_ppm"
M.QWEN32 = "/home/param/models/ppm-811-17850-base-bf16/configs/Qwen3-32B"
M.QWEN4B_CFG = "/home/param/pose-repo/infer/engine/ppm/configs/Qwen3-4B"
IP.MIMI = "/home/param/models/mimi"
_front_paths([AC, f"{AC}/lib", f"{AC}/vendor"])
for _p in (f"{QI}/s4fast", f"{QI}/s4self", f"{QI}/s3fast"):
    if _p not in sys.path:
        sys.path.append(_p)                                        # appended: never shadows a PPM module
import vox_stream as VS  # noqa: E402
from vox_fast import VoxStreamFast  # noqa: E402
from mimi_dec_stream import MimiDecStream  # noqa: E402
import light_kernels as LK  # noqa: E402

BF = torch.bfloat16
CD, DEPTH, CBS, FPS = SP.CD, SP.DEPTH, SP.CBS, SP.FPS
PAD, NW, EOB, TE = SP.PAD_ID, SP.NW_ID, SP.EOB_ID, 151643
OFF = np.arange(DEPTH, dtype=np.int64) * CBS
SPF = VS.SPF24
NINF = float("-inf")
assert CD == 2, CD
VADAPT = {"enc_hidden": 1280, "hidden": 4096, "stack": 4, "norm": "none"}     # = aloe_m1vox.env VOX_* (asserted against the export's cfg)
VOX_ADAPTER = "/data/home/amartyavarma/ppm/asr-adapter/ckpt-final/asrppm-nonorm/final-s1972/adapter-bf16-w1972"
S_INIT_RECAL = 0.0030924472                                      # 22335's calibrated s init (log: [calib] ... s_init = 0.0030924472)
RUN_METRICS = "/data/home/amartyavarma/ppm/aloe-m1vox-ft/runs/ppm-aloe-m1vox-ft-j22335/metrics.jsonl"
T0 = time.time()


def log(*a):
    print(f"[ppmlive {time.strftime('%H:%M:%S')}]", *a, flush=True)


# ================================================================================================ checkpoint
def ckpt_info(path):
    path = os.path.abspath(path.rstrip("/"))
    if os.path.isfile(f"{path}/meta.json"):
        meta = json.load(open(f"{path}/meta.json"))
        assert meta.get("format") == "ppm-local-bf16-shards-v1", meta.get("format")
        assert os.path.isfile(f"{path}/COMPLETE"), ("local copy not COMPLETE", path)
        c = meta["cfg"]
        if "layers" in c:                                        # ckpt_local_export copy (PPM prod format)
            dims = {k: int(c[k]) for k in ("layers", "tower_layers", "dec_layers", "proj_r", "tower_vocab")}
        else:                                                    # aloe trainer export_local (cfg = the trainer CFG): PPM full sizes
            dims = {"layers": 64, "tower_layers": 36, "dec_layers": 28, "proj_r": 2048, "tower_vocab": M.TOWER_VOCAB}
        return dict(kind="local", path=path, step=int(meta["step"]), meta=meta, **dims)
    assert os.path.isfile(f"{path}/.metadata") and os.path.isfile(f"{path}/COMPLETE"), ("not a COMPLETE PPM DCP export", path)
    import torch.distributed.checkpoint as dcp
    md = dcp.FileSystemReader(path).read_metadata().state_dict_metadata
    shp = {k: tuple(v.size) for k, v in md.items() if hasattr(v, "size")}

    def nlay(prefix):
        return 1 + max(int(k[len(prefix):].split(".")[0]) for k in shp if k.startswith(prefix))
    side = json.load(open(f"{path}/train_cfg.json")) if os.path.isfile(f"{path}/train_cfg.json") else {}
    step = int(side.get("step", -1))
    if step < 0:
        b = os.path.basename(path).split("-")[-1]
        step = int(b) if b.isdigit() else -1
    return dict(kind="dcp", path=path, step=step, shapes=shp, side=side, layers=nlay("layers."), tower_layers=nlay("tower_s.layers."),
                dec_layers=nlay("dec_s.model.layers."), proj_r=shp["head.proj.G"][0], tower_vocab=shp["tower_s.text_emb.weight"][0])


def run_metrics_for(ci):
    """10-17 (aloe-v2vox hot swap): the CHECKPOINT'S OWN run metrics (its train_cfg.json / meta.json cfg: log_root / run_tag / metrics.jsonl),
    never another run's (the fixed RUN_METRICS = 22335's would hand a v2vox step-253 export m1vox's step-253 s). None if the cfg lacks them."""
    c = (ci.get("side") or {}).get("cfg") or (ci.get("meta") or {}).get("cfg") or {}
    if c.get("log_root") and c.get("run_tag"):
        return f"{c['log_root']}/{c['run_tag']}/metrics.jsonl"
    return None


def s_from_metrics(step, path):
    """the trainer's fp32 s after optimizer step `step` (train/vox_a_scale of that step's metrics row in `path`) or None."""
    if not path or not os.path.isfile(path):
        return None
    best = None
    for ln in open(path):
        try:
            d = json.loads(ln)
        except ValueError:
            continue
        if int(d.get("step", -1)) == int(step) and "train/vox_a_scale" in d:
            best = float(d["train/vox_a_scale"])
    return best


def build_model(ck, devs, adapter=None, s_value=None):
    """-> (model, ldev, cfgs, info). adapter: None = from the checkpoint (must exist) | a VOX_ADAPTER-style DCP dir (init test path)."""
    ci = ckpt_info(ck)
    c32, c4, c06 = M.cfg32(ci["layers"]), M.cfg4b(ci["tower_layers"]), M.cfg06(ci["dec_layers"])
    torch.set_default_dtype(BF)
    with torch.device("meta"):
        m = M.PPM(c32, c4, c06, ci["proj_r"], ci["tower_vocab"], vadapter=VADAPT, nopose=True)
    torch.set_default_dtype(torch.float32)
    nL, nd = ci["layers"], len(devs)
    if nd == 1:
        ldev = [devs[0]] * nL
    else:
        w = [1.0] * (nd - 1) + [0.45]
        cuts = np.cumsum(w) / sum(w) * nL
        ldev = [devs[int(np.searchsorted(cuts, i + 0.5))] for i in range(nL)]
    d0, dl = devs[0], devs[-1]
    m.inp.to_empty(device=d0)
    for L, d in zip(m.layers, ldev):
        L.to_empty(device=d)
    for mod in (m.head, m.tower_s, m.dec_s):
        mod.to_empty(device=dl)
    m.dec_s.model.rotary_emb = type(m.dec_s.model.rotary_emb)(config=c06, device=dl)
    m.vadapter.to_empty(device=d0)
    m.vgain = M.VGain().to(d0)                                   # theta fp32 (= the trainer's fp32 master; the export holds bf16)
    load_weights(m, ci, adapter)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    th = float(m.vgain.theta.float().reshape(-1)[0])
    s_ck = math.exp(th)
    mpath = run_metrics_for(ci) if adapter is None else None
    s_met = s_from_metrics(ci["step"], mpath) if (ci["kind"] == "dcp" or ci["meta"].get("run_tag")) and adapter is None else None
    th_meta = (ci.get("meta") or {}).get("vgain_theta_fp32") if ci["kind"] == "local" and adapter is None else None   # 10-24 merged exports
    if s_value is not None:
        s_use, src = float(s_value), "given"
    elif th_meta is not None and abs(float(th_meta) - th) <= 0.04:   # the fp32 theta the export's meta.json records (e.g. a merged LoRA FT)
        s_use, src = math.exp(float(th_meta)), f"exp(meta.json vgain_theta_fp32 {float(th_meta):.10g})"
    elif s_met is not None and abs(math.log(s_met) - th) <= 0.04:  # bf16 theta step near -5.8 is 2^-5: the logged fp32 s must round to it
        s_use, src = s_met, f"trainer fp32 s at step {ci['step']} ({mpath})"
    else:
        s_use, src = s_ck, "exp(checkpoint theta)"
    with torch.no_grad():
        m.vgain.theta.fill_(math.log(s_use))
    info = {"ck": ci["path"], "step": ci["step"], "kind": ci["kind"], "s": s_use, "s_src": src, "s_ckpt_theta": s_ck, "s_metrics": s_met, "metrics": mpath,
            "run_tag": ((ci.get("side") or {}).get("cfg") or (ci.get("meta") or {}).get("cfg") or {}).get("run_tag"),
            "total_steps": ((ci.get("side") or {}).get("cfg") or (ci.get("meta") or {}).get("cfg") or {}).get("steps"),
            "adapter": adapter or "checkpoint", "ldev": [str(d) for d in sorted(set(ldev), key=str)],
            "layers_per_dev": {str(d): sum(1 for x in ldev if x == d) for d in devs}}
    log(f"[build] {ci['path']} step {ci['step']} ({ci['kind']}): s = {s_use:.8g} ({src}; ckpt theta -> {s_ck:.8g}); layers "
        f"{info['layers_per_dev']}; head / tower / decoder on {dl}, front on {d0}")
    return m, ldev, (c32, c4, c06), info


@torch.no_grad()
def load_weights(m, ci, adapter):
    t0 = time.time()
    params = {n: p for n, p in m.named_parameters()}
    want = set(params)
    if adapter is not None:
        want -= {n for n in params if n.startswith(("vadapter.", "vgain."))}
    if ci["kind"] == "dcp":
        import torch.distributed.checkpoint as dcp
        have = set(ci["shapes"])
        assert want <= have, ("checkpoint lacks model tensors", sorted(want - have)[:6])
        sd = {}
        for n in sorted(want):
            assert tuple(params[n].shape) == tuple(ci["shapes"][n]), (n, tuple(params[n].shape), ci["shapes"][n])
            sd[n] = params[n] if params[n].dtype == BF else torch.empty(params[n].shape, dtype=BF, device=params[n].device)
        dcp.load(sd, checkpoint_id=ci["path"], no_dist=True)
        for n in want:
            if sd[n] is not params[n]:
                params[n].copy_(sd[n].float())
        extra = sorted(have - want)
        log(f"[load] DCP {ci['path']}: {len(want)} tensors ({time.time() - t0:.0f}s); not used: {extra[:6]}{' ...' if len(extra) > 6 else ''}")
    else:
        meta = ci["meta"]
        rows = {n: 0 for n in want}
        skipped = set()
        for r in range(int(meta["world"])):
            sd = torch.load(f"{ci['path']}/rank{r}.pt", map_location="cpu", weights_only=True)
            for n, t in sd.items():
                if n not in want:
                    skipped.add(n.split(".")[0]); continue
                k = t.shape[0] if t.dim() else 0
                if k:
                    params[n][rows[n]:rows[n] + k].copy_(t.to(params[n].dtype))
                    rows[n] += k
            del sd
        rep = torch.load(f"{ci['path']}/rep.pt", map_location="cpu", weights_only=True)
        for n, t in rep.items():
            if n in want:
                params[n].copy_(t.to(params[n].dtype)); rows[n] = params[n].shape[0]
        bad = [n for n in want if rows[n] != params[n].shape[0]]
        assert not bad, ("tensors not fully covered by the shards", [(n, rows[n], params[n].shape[0]) for n in bad[:6]])
        log(f"[load] local shards {ci['path']} (step {ci['step']}, world {meta['world']}): {len(want)} tensors ({time.time() - t0:.0f}s); "
            f"skipped prefixes {sorted(skipped)}")
    if adapter is not None:
        import pickle
        import torch.distributed.checkpoint as dcp
        md = pickle.load(open(f"{adapter}/.metadata", "rb")).state_dict_metadata
        keys = sorted(k for k in md if k.startswith("vadapter."))
        sd = {k: torch.empty(tuple(md[k].size), dtype=md[k].properties.dtype) for k in keys}
        dcp.load(sd, checkpoint_id=adapter, no_dist=True)
        names = {n for n in params if n.startswith("vadapter.")}
        assert set(sd) == names, ("adapter keys", sorted(set(sd) ^ names))
        for n in names:
            params[n].copy_(sd[n].to(params[n].dtype))
        m.vgain.theta.fill_(math.log(S_INIT_RECAL))
        log(f"[load] vadapter <- {adapter} ({len(sd)} tensors), s <- {S_INIT_RECAL} (22335's calibrated init)")
    nonfin = [n for n, p in params.items() if not torch.isfinite(p).all()]
    assert not nonfin, ("non-finite weights", nonfin[:4])


# ================================================================================================ sampling (= qwen-1ch/infer/s4live*/live_engine.py, copied)
def _greedy_or(z, temp, q):
    return torch.where(temp <= 0, z.argmax(-1), q)


def _topk_mask(z, k):
    """z [N, V] fp32, k long [] (0 = off) -> z with everything below the k-th largest value at -inf (ties kept)."""
    V = z.shape[-1]
    vals = z.sort(-1, descending=True).values
    thr = vals.gather(-1, (k.clamp(1, V) - 1).view(1, 1).expand(z.shape[0], 1))
    return torch.where((k > 0) & (z < thr), torch.full_like(z, NINF), z)


def _topp_mask(z, p):
    """nucleus: keep the smallest set of top tokens whose mass reaches p (the top token always kept); p >= 1 -> unchanged."""
    pr = torch.softmax(z, -1)
    ps = pr.sort(-1, descending=True).values
    n = ((ps.cumsum(-1) - ps) < p).sum(-1, keepdim=True).clamp_min(1)
    thr = ps.gather(-1, n - 1)
    return torch.where((p < 1.0) & (pr < thr), torch.full_like(z, NINF), z)


def t_sample(z, gen, temp, topk, topp=None):
    """temperature / top-k / top-p from device tensors; temp <= 0 -> argmax."""
    zz = z / temp.clamp_min(1e-4)
    zz = _topk_mask(zz, topk)
    if topp is not None:
        zz = _topp_mask(zz, topp)
    q = torch.multinomial(torch.softmax(zz, -1), 1, generator=gen)[:, 0]
    return _greedy_or(z, temp, q)


def t_word_min_p(z, is_word, p, rel):
    """= fast_engine.word_min_p with p (fp32 []) and the mode (rel: long [] 1 relative / 0 absolute) as tensors; p <= 0 -> unchanged."""
    lp = torch.log_softmax(z.float(), -1)
    w = is_word[None]
    lw = lp.masked_fill(~w, NINF)
    lse_w = torch.logsumexp(lw, -1, keepdim=True)
    lpw = lw - lse_w
    top = lpw.max(-1, keepdim=True).values
    lgp = torch.log(p.clamp_min(1e-30))
    keep = torch.where(rel > 0, lpw >= top + lgp, (lpw >= lgp) | (lpw == top)) | (p <= 0)
    kw = w & keep
    lse_k = torch.logsumexp(lp.masked_fill(~kw, NINF), -1, keepdim=True)
    out = torch.where(kw, lp + (lse_w - lse_k), torch.where(w, torch.full_like(lp, NINF), lp))
    return torch.where(torch.isfinite(lse_w), out, lp)


def t_penalties(z, cnt, fpen, rep, is_word, cnt_rep=None):
    """frequency penalty (z -= fpen x cnt) + CTRL repetition penalty on present word pieces, then the word pieces are shifted back to
    their pre-penalty total mass (text_freq_pen.apply's convention: P(PAD) / P(NEW_WORD) untouched). fpen 0 and rep 1 -> unchanged."""
    zf = z.float()
    zp = zf - fpen * cnt
    pres = (cnt if cnt_rep is None else cnt_rep) > 0
    zp = torch.where(pres & (rep != 1.0), torch.where(zp > 0, zp / rep, zp * rep), zp)
    w = is_word[None]
    l0 = torch.logsumexp(zf.masked_fill(~w, NINF), -1, keepdim=True)
    l1 = torch.logsumexp(zp.masked_fill(~w, NINF), -1, keepdim=True)
    out = torch.where(w, zp + (l0 - l1), zf)
    return torch.where(torch.isfinite(l0) & torch.isfinite(l1), out, zf)


def t_text_draw(z, is_word, cnt, cnt2, fact, ct, tt, tk, tp, mp, mrel, fp, rp, wg, gen_t, gen_c):
    """B's lane draw (module doc FACTORISED TEXT SAMPLING) from in-vocab-masked (no grammar, 10-28), CFG-combined fp32 logits z [N, V] (word pieces = ids < TE,
    is_word; every knob a device tensor []). Stage 2 (word knobs) runs on z2 = z restricted to word pieces when factorised and a word piece
    is allowed (else z: the joint path / a frame where no word piece is in vocabulary, where its draw is unused): penalties -> temperature -> min-p
    -> top-k -> top-p -> draw (gen_t) -> word_greedy. Stage 1 (factorised only, gen_c): categories [WORD = logsumexp(z[:, :TE]), z[:, TE:]
    each its own] / cat_temp -> draw; WORD -> the stage-2 word piece. fact 0 -> the joint draw exactly (the pre-10-17 ops, same gen_t use).
    -> (q [N] long, p_draw [N] fp32 = the probability of q under the distribution it was drawn from)."""
    w = is_word[None]
    zw = z.masked_fill(~w, NINF)
    useW = (fact > 0) & torch.isfinite(zw).any(-1, keepdim=True)
    z2 = torch.where(useW, zw, z)
    zp = t_penalties(z2, cnt, fp, rp, is_word, cnt2)
    zs = t_word_min_p(zp / tt.clamp_min(1e-4), is_word, mp, mrel)
    zs = _topp_mask(_topk_mask(zs, tk), tp)
    ps = torch.softmax(zs, -1)
    q2 = torch.multinomial(ps, 1, generator=gen_t)[:, 0]
    q2 = _greedy_or(zp, tt, q2)
    qw = zp.masked_fill(~w, NINF).argmax(-1)
    q2 = torch.where((wg > 0) & (q2 < TE), qw, q2)
    p2 = ps.gather(1, q2[:, None])[:, 0]
    zc = torch.cat([torch.logsumexp(z[:, :TE], -1, keepdim=True), z[:, TE:]], -1) / ct.clamp_min(1e-4)
    pc = torch.softmax(zc, -1)
    ic = torch.multinomial(pc, 1, generator=gen_c)[:, 0]
    qc = torch.where(ic == 0, q2, ic + (TE - 1))
    pq = torch.where(ic == 0, pc[:, 0] * p2, pc.gather(1, ic[:, None])[:, 0])
    f = fact > 0
    return torch.where(f, qc, q2), torch.where(f, pq, p2)


def t_text_cat_probs(z, ct):
    """stage-1 category probabilities of t_text_draw (tests / diagnostics): -> [N, 1 + V - TE] (column 0 = WORD, column 1 + i = id TE + i)."""
    return torch.softmax(torch.cat([torch.logsumexp(z[:, :TE], -1, keepdim=True), z[:, TE:]], -1) / ct.clamp_min(1e-4), -1)


def ctrl_penalty(z, pres, pen):
    """= s3fast/fast_engine.ctrl_penalty: CTRL repetition penalty on fp32 logits z [N, V] where pres [N, V] (bool)."""
    return torch.where(pres, torch.where(z > 0, z / pen, z * pen), z)


def presence(dep_win, OFF_, CBS_, valid_pos=None):
    """= s3fast/fast_engine.presence: dep_win [S, W, 32] global codes (-1 none) -> [S, 32, CBS] bool."""
    S, W, K = dep_win.shape
    ok = dep_win >= 0
    if valid_pos is not None:
        ok = ok & valid_pos[None, :, None]
    ix = torch.where(ok, dep_win - OFF_, torch.full_like(dep_win, CBS_)).transpose(1, 2)
    pres = torch.zeros(S, K, CBS_ + 1, dtype=torch.float32, device=dep_win.device)
    pres.scatter_(2, ix, 1.0)
    return pres[..., :CBS_] > 0


WMAX_T, WMAX_A = 256, 256
WMAX_S = 256                                                     # adaptive rep-pen: SPEAKING look-back buffer (speak_window_frames <= 250)


class LiveRing:
    """= live_engine.LiveRing: B's last word pieces with their push index (variable window <= WMAX_T, read as a tensor)."""

    def __init__(self, V, TE_, dev, excl):
        self.V, self.TE, self.excl = V, TE_, excl
        self.ring = torch.full((1, WMAX_T), -1, dtype=torch.long, device=dev)
        self.pidx = torch.full((1, WMAX_T), -1, dtype=torch.long, device=dev)
        self.ptr = torch.zeros(1, dtype=torch.long, device=dev)
        self.cnt = torch.zeros(1, V, dtype=torch.float32, device=dev)
        self.cnt2 = torch.zeros(1, V, dtype=torch.float32, device=dev)

    def counts(self, win, out=None):
        out = self.cnt if out is None else out
        ok = (self.ring >= 0) & (self.pidx >= self.ptr - win)
        out.zero_()
        out.scatter_add_(1, self.ring.clamp_min(0), ok.to(torch.float32))
        return out

    def push(self, q):
        w = (q >= 0) & (q < self.TE)
        for x in self.excl:
            w = w & (q != x)
        pos = (self.ptr % WMAX_T)[:, None]
        self.ring.scatter_(1, pos, torch.where(w[:, None], q[:, None], self.ring.gather(1, pos)))
        self.pidx.scatter_(1, pos, torch.where(w[:, None], self.ptr[:, None], self.pidx.gather(1, pos)))
        self.ptr.add_(w.long())

    def reset(self):
        self.ring.fill_(-1); self.pidx.fill_(-1); self.ptr.zero_()

    def state(self):
        return [self.ring, self.pidx, self.ptr]


# ---- LIVE SAMPLING PARAMETERS (= live_engine.PARAMS; every knob is a device tensor read INSIDE the graphs)
PARAMS = {
    "factorised_text": (int, 0, 1),        # 10-17 FACTORISED text sampling (module doc): 1 = category (PAD / NEW_WORD / WORD [/ EOB]) then word; 0 = joint
    "cat_temp":      (float, 0.05, 3.0),    # stage-1 category temperature (factorised only; no penalties / truncation at that stage)
    "pad_bias":      (float, -10.0, 2.0),   # 10-18 added to B's PAD logit before the draw (factorised: PAD category; joint: PAD token); < 0 = speak more
    "adaptive_pad_on": (int, 0, 1),         # 10-24 ADAPTIVE PAD BIAS (module doc): 1 = pad_bias_silent / pad_bias_other replace pad_bias
    "pad_silent_s":    (float, 0.5, 20.0),  # SILENT = no spoken B lane token (word piece / NEW_WORD) for >= this many seconds (frames >= s x 12.5)
    "pad_bias_silent": (float, -10.0, 2.0), # PAD bias while B is SILENT (adaptive on)
    "pad_bias_other":  (float, -10.0, 2.0), # PAD bias otherwise (adaptive on)
    "text_temp":     (float, 0.05, 3.0),    # text temperature (factorised: over word pieces only)
    "text_topk":     (int, 0, 256),       # text top-k (factorised: over word pieces only; joint: over the in-vocab lane tokens) (0 = off)
    "text_topp":     (float, 0.05, 1.0),   # text nucleus (factorised: over word pieces only) (1 = off)
    "min_p":     (float, 0.0, 0.5),   # min-p on the WORD choice (0 = off; fast_engine.word_min_p rule)
    "min_p_rel": (int, 0, 1),          # 0 = absolute (top word always kept), 1 = relative to the top word
    "freq_pen":      (float, 0.0, 3.0),    # frequency penalty strength (logit -= pen x count in window; word mass kept)
    "freq_win":      (int, 1, 200),        # window = B's last N word pieces
    "text_rep":      (float, 1.0, 2.0),    # CTRL repetition penalty on word pieces present in text_rep_win (1 = off; word mass kept)
    "text_rep_win":  (int, 1, 200),        # its window (B's last N word pieces)
    "cb0_temp":      (float, 0.05, 3.0),    # audio codebook 0 temperature
    "cb0_topk":      (int, 0, 256),       # audio codebook 0 top-k (0 = off; 10-23: 0 allowed, was clamped to 1 = greedy)
    "light_temp":    (float, 0.05, 3.0),    # codebooks 1-31 (0.6B decoder) temperature
    "light_topk":    (int, 0, 256),       # codebooks 1-31 top-k (0 = off; 10-23: 0 allowed)
    "audio_trunc":   (int, 0, 1),          # 10-27 user: which audio truncation is active: 0 = top-k (cb0_topk / light_topk), 1 = top-p (below)
    "cb0_topp":      (float, 0.05, 1.0),   # 10-27: audio codebook 0 nucleus (used only when audio_trunc = 1; 1 = off)
    "light_topp":    (float, 0.05, 1.0),   # 10-27: codebooks 1-31 nucleus (used only when audio_trunc = 1; 1 = off)
    "audio_rep_pen":     (float, 1.0, 2.0),    # CTRL repetition penalty per codebook (1 = off)
    "audio_rep_win": (int, 1, 256),        # its window in frames (conversation only)
    "word_greedy":   (int, 0, 1),
    "cfg_on":        (int, 0, 1),          # audio CFG on B's text logits (the uncond row of the batch-2 engine)
    "cfg_s":         (float, 1.0, 3.0),    # CFG strength: z = uncond + s (cond - uncond); 1 = cond exactly
    "cfg_cb0":       (int, 0, 1),          # (with cfg_on) the same CFG on B's cb0 logits (before the audio rep-pen), sampled once
    "adaptive_rep_on":     (int, 0, 1),        # adaptive repetition penalty: per-frame SPEAKING / SILENT penalties replace the static ones
    "speak_window_frames": (int, 1, 250),      # SPEAKING = B emitted a word piece in the last N frames (25 = 2 s)
    "audio_rep_speaking":  (float, 1.0, 2.0),  # audio CTRL rep-pen (cb0 + cb1-31, window audio_rep_win) while SPEAKING
    "audio_rep_silent":    (float, 1.0, 2.0),  # ... while SILENT
    "text_rep_speaking":   (float, 1.0, 2.0),  # text CTRL rep-pen (window text_rep_win) while SPEAKING
    "text_rep_silent":     (float, 1.0, 2.0),  # ... while SILENT
}
ADAPT_KEYS = ("adaptive_rep_on", "speak_window_frames", "audio_rep_speaking", "audio_rep_silent", "text_rep_speaking", "text_rep_silent")
PADAD_KEYS = ("adaptive_pad_on", "pad_silent_s", "pad_bias_silent", "pad_bias_other")          # 10-24 adaptive PAD bias (next to pad_bias)
PADAD_DEFAULTS = {"adaptive_pad_on": 0, "pad_silent_s": 3.0, "pad_bias_silent": -1.0, "pad_bias_other": 0.0}   # off in EVERY profile
# standing sampling rules: audio top-k 30 (cb0 + light), temperature 1 everywhere, NO repetition / frequency penalty, CFG off; 10-17 user:
# factorised text sampling ON with min-p 0.05 (absolute: min_p_rel 0, the demos' mode default) on the word stage; the joint fallback keeps min-p 0
# 10-17 ~09:20 PT user: defaults = the user's own session settings (text temp 0.8, freq pen 0.3, adaptive rep on, audio rep 1.05 while speaking)
MINP_DEFAULT = {1: 0.05, 0: 0.0}                                 # factorised_text -> default min-p (start / mode switch: LivePPM.set_params)
DEFAULTS = {"factorised_text": 1, "cat_temp": 1.0, "pad_bias": 0.0,
            "text_temp": 0.8, "text_topk": 0, "text_topp": 1.0, "min_p": MINP_DEFAULT[1], "min_p_rel": 0, "freq_pen": 0.3, "freq_win": 20,
            "text_rep": 1.0, "text_rep_win": 64, "cb0_temp": 1.0, "cb0_topk": 30, "light_temp": 1.0, "light_topk": 30, "audio_rep_pen": 1.0, "audio_rep_win": 64,
            "word_greedy": 0, "cfg_on": 0, "cfg_s": 1.5, "cfg_cb0": 0,
            "adaptive_rep_on": 1, "speak_window_frames": 25, "audio_rep_speaking": 1.05, "audio_rep_silent": 1.0, "text_rep_speaking": 1.0, "text_rep_silent": 1.0,
            **PADAD_DEFAULTS, "audio_trunc": 0, "cb0_topp": 1.0, "light_topp": 1.0}
# 10-23 user ('the live Aloe demo with temp 1 and nothing else for everything: completely unbiased model generation'): PPMLIVE_DEFAULTS=unbiased
# (set by that demo's launch only; every other importer keeps the defaults above) -> temperature 1 on every stream, no top-k / top-p / min-p /
# frequency / repetition penalty / PAD bias / CFG, joint lane draw (factorised_text 0 = one categorical draw over the in-vocab-masked (no grammar, 10-28) lane logits;
# at these settings the factorised draw is the same distribution). Every knob stays a live control; only its default changes.
PROFILE = os.environ.get("PPMLIVE_DEFAULTS", "") or "fantastic"
if PROFILE == "unbiased":
    MINP_DEFAULT = {1: 0.0, 0: 0.0}
    DEFAULTS.update({"factorised_text": 0, "cat_temp": 1.0, "pad_bias": 0.0, "text_temp": 1.0, "text_topk": 0, "text_topp": 1.0, "min_p": 0.0,
                     "freq_pen": 0.0, "text_rep": 1.0, "cb0_temp": 1.0, "cb0_topk": 0, "light_temp": 1.0, "light_topk": 0, "audio_rep_pen": 1.0,
                     "word_greedy": 0, "cfg_on": 0, "cfg_cb0": 0, "adaptive_rep_on": 0, "audio_rep_speaking": 1.0,
                     "audio_rep_silent": 1.0, "text_rep_speaking": 1.0, "text_rep_silent": 1.0, **PADAD_DEFAULTS})
GROUPS = {k: ("adaptive repetition penalty" if k in ADAPT_KEYS else "audio CFG" if k.startswith("cfg") else "audio" if k.startswith(("cb0", "light", "audio"))
              else "text") for k in PARAMS}
WORDS_ONLY = " Words only: with factorised sampling it acts on the word choice alone (after B decided to say a word); PAD / NEW_WORD timing never touched."
HINTS = {
    "factorised_text": ("Factorised text sampling (default on): each frame B first decides PAD / NEW_WORD / WORD (EOB too when allowed) from the "
                        "model's probabilities (category temperature, PAD bias), then, only if WORD, which word piece, renormalised over word pieces. Every other text knob below (temperature, "
                        "top-k, top-p, min-p, frequency / repetition penalties, adaptive text rep-pen) applies to that word choice only, so PAD / "
                        "NEW_WORD timing is never penalised, truncated or temperature-shifted. Off = the old joint sampling (the knobs act on the "
                        "whole lane distribution; min-p default 0), for comparison."),
    "cat_temp": ("Category temperature: temperature of the PAD / NEW_WORD / WORD decision (factorised sampling only; no penalties or "
                 "truncation there). 1 = the model's own probabilities."),
    "pad_bias": ("Added to B's PAD logit before sampling: negative = B stays silent less and speaks more (-1 makes PAD e = 2.7x less likely "
                 "relative to the other choices; 0 = off, the model's own timing; positive = more silence). Factorised sampling: the PAD / "
                 "NEW_WORD / WORD decision (in nats when the category temperature is 1); joint sampling: the PAD token logit. "
                 "Ignored while the adaptive PAD bias is on."),
    "adaptive_pad_on": ("Adaptive PAD bias: when on, the PAD bias depends on how long B has been silent and replaces the static PAD bias: "
                        "'PAD bias while silent' once B has not spoken (no word or NEW_WORD in its lane) for the silence threshold, "
                        "'PAD bias otherwise' the rest of the time. The timer starts at the first conversation frame, so it never kicks "
                        "in before the threshold. Off = the static PAD bias above."),
    "pad_silent_s": ("Silence threshold (seconds, 80 ms frames): B counts as silent once its last spoken lane token (word piece or "
                     "NEW_WORD; PAD / EOB never count) is at least this long ago (adaptive PAD bias only)."),
    "pad_bias_silent": ("PAD bias while B has been silent for the threshold or longer (adaptive PAD bias only): negative = B starts "
                        "speaking sooner (-1 makes PAD e = 2.7x less likely; same units as the PAD bias)."),
    "pad_bias_other": ("PAD bias the rest of the time, i.e. while B spoke within the threshold (adaptive PAD bias only): 0 = the model's "
                       "own timing; positive = more pauses."),
    "text_temp": "text temperature of the word choice." + WORDS_ONLY,
    "text_topk": "keep the k most likely word pieces (0 = off)." + WORDS_ONLY,
    "text_topp": "nucleus over word pieces: smallest set reaching this mass (1 = off)." + WORDS_ONLY,
    "min_p": "min-p on the word choice (0 = off; default 0.05 with factorised sampling, 0 with joint)." + WORDS_ONLY,
    "min_p_rel": "min-p mode: 0 absolute (p_w >= p, top word kept), 1 relative (p_w >= p x top word); p_w = probability among word pieces",
    "freq_pen": "frequency penalty: word-piece logit -= strength x count among B's last N word pieces." + WORDS_ONLY,
    "freq_win": "frequency penalty window: B's last N word pieces",
    "text_rep_win": "text repetition penalty window: B's last N word pieces",
    "text_rep": "CTRL repetition penalty on word pieces present in the window (1 = off)." + WORDS_ONLY,
    "cb0_temp": "audio codebook 0 (semantic, 4B speech tower) temperature", "cb0_topk": "audio codebook 0 top-k (0 = off)",
    "light_temp": "codebooks 1-31 (0.6B decoder) temperature", "light_topk": "codebooks 1-31 top-k (0 = off)",
    "audio_trunc": "audio truncation: 0 = top-k (the two top-k knobs), 1 = top-p / nucleus (the two top-p knobs); the other pair is ignored",
    "cb0_topp": "audio codebook 0 top-p / nucleus (only when audio truncation = top-p; 1 = off)",
    "light_topp": "codebooks 1-31 top-p / nucleus (only when audio truncation = top-p; 1 = off)",
    "audio_rep_pen": "CTRL repetition penalty per codebook on codes used in the window (1 = off)",
    "audio_rep_win": "audio repetition window (conversation frames)",
    "word_greedy": "when B says a word, take the most likely word piece (after the word penalties; timing / category still sampled)",
    "cfg_cb0": "with CFG on: also apply it to B's cb0 audio logits (uncond + s (cond - uncond), before the audio rep-pen)",
    "adaptive_rep_on": "on: each frame uses the SPEAKING or SILENT penalties below instead of the static audio / text repetition penalties",
    "speak_window_frames": "SPEAKING = B emitted a word piece (not PAD / NEW_WORD / EOB) in its last N frames (80 ms each; 25 = 2 s)",
    "audio_rep_speaking": "audio repetition penalty (cb0 + cb1-31, audio repetition window) while B is SPEAKING (1 = off)",
    "audio_rep_silent": "audio repetition penalty (cb0 + cb1-31, audio repetition window) while B is SILENT (1 = off)",
    "text_rep_speaking": "text repetition penalty (text repetition window) while B is SPEAKING (1 = off); words only with factorised sampling",
    "text_rep_silent": "text repetition penalty (text repetition window) while B is SILENT (1 = off); words only with factorised sampling"}
FKEYS = [k for k, v in PARAMS.items() if v[0] is float]
IKEYS = [k for k, v in PARAMS.items() if v[0] is int]
UNCOND_SRC = ("quiet_frames", "digital_zero")
QUIET_BANK = "/home/param/models/aloe-ft-vox17127-nopersona/assets/quiet_bank_m2.npz"   # = s4live's uncond bank (quiet A frames, training level)


GRAMMAR_GONE = ("free_text", "allow_eob")   # 10-28 USER ORDER: no lane grammar, ever: legacy knobs, dropped silently wherever they arrive


def validate(d):
    """-> (clean dict of the known keys, clamped / typed, list of problems). Unknown keys are reported, never applied."""
    out, bad = {}, []
    for k, v in (d or {}).items():
        if k in GRAMMAR_GONE:
            continue                                                    # 10-28 USER ORDER: no lane grammar, ever: legacy key dropped silently
        if k not in PARAMS:
            bad.append(f"unknown {k}"); continue
        t, lo, hi = PARAMS[k]
        try:
            x = t(float(v)) if t is int else float(v)
        except (TypeError, ValueError):
            bad.append(f"{k}: not a number"); continue
        if x != x:
            bad.append(f"{k}: NaN"); continue
        if not lo <= x <= hi:
            bad.append(f"{k}={x} outside [{lo}, {hi}] (clamped)"); x = min(max(x, lo), hi)
        out[k] = x
    return out, bad


# ================================================================================================ fused 0.6B decoder (stage-3 FusedLight, triton path)
class FusedDec:
    """the 0.6B depth decoder's Qwen3 layers with the fused kernels (vendor fused_triton rms / add_rms + light_kernels gemv / light_attn /
    silu_mul_gu) and a static per-frame KV cache = qwen-1ch/infer/s3fast/fast_engine.FusedLight.step_triton (same LightTower class:
    heads_s2q). step(x, j): j = 0 (cb1: the 2 tokens proj(hv), cb0_emb at positions 0, 1), j >= 1 (cb_{j+1}: one token at position j + 1)."""

    def __init__(self, dec, dev, max_pos=64):
        cfg = dec.model.config
        self.dec, self.dev = dec, dev
        self.nh, self.kv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.layers = []
        for L in dec.model.layers:
            at, mlp = L.self_attn, L.mlp
            assert at.q_proj.bias is None and at.o_proj.bias is None
            self.layers.append(dict(ln1=L.input_layernorm.weight, e1=float(L.input_layernorm.variance_epsilon),
                                    wqkv=torch.cat([at.q_proj.weight, at.k_proj.weight, at.v_proj.weight], 0).contiguous(), wo=at.o_proj.weight,
                                    qn=at.q_norm.weight, kn=at.k_norm.weight, eqk=float(at.q_norm.variance_epsilon), sc=float(at.scaling),
                                    ln2=L.post_attention_layernorm.weight, e2=float(L.post_attention_layernorm.variance_epsilon),
                                    wgu=torch.cat([mlp.gate_proj.weight, mlp.up_proj.weight], 0).contiguous(), wd=mlp.down_proj.weight))
        self.nw, self.ne = dec.model.norm.weight, float(dec.model.norm.variance_epsilon)
        self.K = [torch.zeros(1, max_pos, self.kv, self.D, dtype=BF, device=dev) for _ in self.layers]
        self.V = [torch.zeros(1, max_pos, self.kv, self.D, dtype=BF, device=dev) for _ in self.layers]
        with torch.cuda.device(dev):
            c, s = dec.model.rotary_emb(torch.zeros(1, 1, 8, dtype=BF, device=dev), torch.arange(max_pos, device=dev)[None])
        c, s = c[0].contiguous(), s[0].contiguous()
        self.cs = []
        for j in range(DEPTH - 1):
            pos = torch.tensor([0, 1] if j == 0 else [j + 1], device=dev)
            self.cs.append((c.index_select(0, pos)[None].contiguous(), s.index_select(0, pos)[None].contiguous()))
        self.W = dec.heads.weight.view(DEPTH - 1, CBS, -1)

    @staticmethod
    def lin(x, w):
        return LK.gemv(x, w) if x.numel() // x.shape[-1] <= 16 else F.linear(x, w)

    def step(self, x, j):
        S, L, Hl = x.shape
        c, s = self.cs[j]
        t0 = 0 if j == 0 else j + 1
        nh, kv, D = self.nh, self.kv, self.D
        res = x.reshape(1, S * L, Hl)
        xn = FO.rms(res, self.layers[0]["ln1"], self.layers[0]["e1"])
        mo = None
        for i, Ly in enumerate(self.layers):
            if i:
                res, xn = FO.add_rms(res, mo, Ly["ln1"], Ly["e1"])
            qkv = self.lin(xn[0], Ly["wqkv"])
            a = LK.light_attn(qkv, Ly["qn"], Ly["kn"], Ly["eqk"], c[0], s[0], self.K[i], self.V[i], t0, L, nh, kv, D, Ly["sc"])
            a = self.lin(a, Ly["wo"])[None]
            res, h2 = FO.add_rms(res, a, Ly["ln2"], Ly["e2"])
            mo = self.lin(LK.silu_mul_gu(self.lin(h2[0], Ly["wgu"])), Ly["wd"])[None]
        _, xn = FO.add_rms(res, mo, self.nw, self.ne)
        return xn.view(S, L, Hl)[:, -1]

    def head(self, hl, d):
        return self.lin(hl, self.W[d - 1]).float()

    def gen(self, hv, c0, gen, temp, topk, pres=None, rp=None, topp=None):
        """hv [1, Hv] tower state, c0 [1] local cb0 -> codes [1, 32] local (cb0 + sampled cb1..31); pres [1, 32, CBS] / rp: the CTRL audio
        repetition penalty per codebook (= live_engine._audio_live)."""
        dec = self.dec
        x = torch.cat([dec.proj(hv)[:, None].to(BF), dec.cb0_emb(c0)[:, None].to(BF)], 1)
        out = [c0]
        for d in range(1, DEPTH):
            zd = self.head(self.step(x, d - 1), d)
            if pres is not None:
                zd = ctrl_penalty(zd, pres[:, d], rp)
            cd = t_sample(zd, gen, temp, topk, topp)
            out.append(cd)
            if d < DEPTH - 1:
                x = dec.model.embed_tokens(cd + (d - 1) * CBS)[:, None].to(BF)
        return torch.stack(out, 1)


# ================================================================================================ the engine
class LivePPM:
    """built once per loaded checkpoint (graphs captured once, static buffers sized for P_MAX + max_s); start(prefix, seed, params) per
    session; step(T) / result(T) per frame (module doc). BATCH 2 in the trunk / speech tower / text head: row 0 = cond (your mic), row 1 =
    the CFG uncond stream (= s4live's LiveS4U: the same prompt, A hears quiet frames (s4live's bank, training level) or digital zero
    through the same Voxtral front + adapter x s, precomputed once as latent tables; B's lane / codes are row 0's, nothing is sampled
    from row 1). It runs every frame (its KV stays aligned, so CFG can be switched on at any frame); the 0.6B decoder runs for row 0 only."""

    # 10-28 hybrid-live audio top-p (user): True = top-k AND top-p on the audio codes, nucleus after top-k (= t_sample's / the text lane's
    # order; cb0_topk / light_topk then cb0_topp / light_topp), audio_trunc unused. Read at graph capture (set it before capture()).
    # False (default, every other user of this file) = the 10-27 audio_trunc switch exactly (top-k OR top-p).
    k_then_p = False

    def __init__(self, m, ldev, cfgs, devs, max_s=300.0, p_max=2000, light_fused=True, mimi=None, enc=None):
        self.m, self.ldev, self.devs = m, ldev, devs
        c32, c4, c06 = cfgs
        self.d0, self.dl = devs[0], devs[-1]
        d0, dl = self.d0, self.dl
        self.max_s = float(max_s)
        self.n_fr = int(round(self.max_s * FPS)) + 8
        self.lmax = int(p_max) + self.n_fr + CD + 8
        self.bb = IP.Stack(list(m.layers), ldev, c32, self.lmax)
        self.ts = IP.Stack(list(m.tower_s.layers), [dl] * len(m.tower_s.layers), c4, self.lmax)
        # batch-2 KV caches; the eager prefill (Stack.run) writes row 0 through views, start() copies the prefix into row 1
        self.bbK = [torch.zeros(2, *k.shape, dtype=BF, device=k.device) for k in self.bb.k]
        self.bbV = [torch.zeros(2, *v.shape, dtype=BF, device=v.device) for v in self.bb.v]
        self.bb.k, self.bb.v = [k[0] for k in self.bbK], [v[0] for v in self.bbV]
        self.tsK = [torch.zeros(2, *k.shape, dtype=BF, device=k.device) for k in self.ts.k]
        self.tsV = [torch.zeros(2, *v.shape, dtype=BF, device=v.device) for v in self.ts.v]
        self.ts.k, self.ts.v = [k[0] for k in self.tsK], [v[0] for v in self.tsV]
        torch.cuda.empty_cache()
        self.fa = fa3_op._mod().flash_attn_interface
        self.TE, self.PAD, self.NW, self.EOB, self.CD = TE, PAD, NW, EOB, CD
        h = m.head
        self.R = torch.cat([h.spec_out, h.nw_out, h.eob_out], 0)
        self.spec = torch.tensor(list(M.SPEC), device=dl)
        self.V = h.lm_head.weight.shape[0]
        V = self.V
        # 10-28 USER ORDER: no lane grammar, ever: no grammar stacks; the lane draw is masked by self.in_vocab only (vocabulary bound)
        self.is_word = torch.arange(V, device=dl) < TE
        self.is_pad = torch.arange(V, device=dl) == PAD                 # 10-18 pad_bias: the PAD column of the lane logits
        # 10-18 pad mask (= aloe-m1vox-live-pure): lane ids the model can take back as input (speech tower text_emb rows; trunk emb_B rows).
        # The 254 unused Qwen padding rows >= 151682 have no tower input row: drawing one (free_text 1) crashed the server next frame.
        self.n_in = min(int(m.tower_s.text_emb.weight.shape[0]), int(m.inp.emb[1].weight.shape[0]), V)
        self.in_vocab = torch.arange(V, device=dl) < self.n_in
        self.gen_t = torch.Generator(device=dl)
        self.gen_a = torch.Generator(device=dl)
        self.gen_c = torch.Generator(device=dl)                         # factorised text: the stage-1 category draw (own stream: joint = pre-10-17)
        self.light = FusedDec(m.dec_s, dl) if light_fused else None
        self.ring = LiveRing(V, TE, dl, (PAD, NW, EOB))
        self.arWA = torch.arange(WMAX_A, device=dl)
        self.arWS = torch.arange(WMAX_S, device=dl)
        # 10-24: ONE per-frame state buffer (one device->host copy in result()): [0:3] adaptive rep-pen (SPEAKING flag, audio rep-pen in
        # effect, text rep-pen in effect), [3:6] adaptive PAD bias (on flag, seconds since B's last spoken lane token, PAD bias in effect)
        self.fstate = torch.zeros(6, dtype=torch.float32, device=dl)
        self.adapt = self.fstate[0:3]
        self.padad = self.fstate[3:6]
        self.lastspk = torch.zeros(1, dtype=torch.long, device=dl)    # 10-24 adaptive PAD bias: position of B's last spoken lane token (start: P)
        self.segs = []
        for i, d in enumerate(ldev):
            if not self.segs or self.segs[-1][0] != d:
                self.segs.append((d, []))
            self.segs[-1][1].append(i)
        assert self.segs[-1][0] == dl
        self.sdevs = sorted(set(devs) | {d for d, _ in self.segs}, key=lambda d: d.index)
        self.Td = {d: torch.zeros(1, dtype=torch.long, device=d) for d in self.sdevs}
        self.T2 = {d: torch.zeros(2, dtype=torch.long, device=d) for d in self.sdevs}
        self.rope = {}
        for d in self.sdevs:
            x = torch.zeros(1, self.lmax, 1, dtype=BF, device=d)
            self.rope[d] = self.bb.rot[d](x, torch.arange(self.lmax, device=d)[None]) if d in self.bb.rot else None
        self.rope_t = self.ts.rot[dl](torch.zeros(1, self.lmax, 1, dtype=BF, device=dl), torch.arange(self.lmax, device=dl)[None])
        H = c32.hidden_size
        self.H = H
        self.laneB = torch.full((self.lmax,), PAD, dtype=torch.long, device=dl)
        self.depB = torch.full((self.lmax, DEPTH), -1, dtype=torch.long, device=dl)
        self.hn_prev = torch.zeros(2, H, dtype=BF, device=dl)
        self.z0 = torch.zeros(2, CBS, dtype=torch.float32, device=dl)
        self.zt = torch.zeros(2, V, dtype=torch.float32, device=dl)
        self.probs = torch.zeros(2, dtype=torch.float32, device=dl)
        self.ptok = torch.zeros(2, dtype=torch.float32, device=dl)
        self.cfgd = torch.zeros(4, dtype=torch.float32, device=dl)       # text: mean |zc - zu| (finite), TV(cond, uncond); cb0: the same
        self.Pd = torch.zeros(1, dtype=torch.long, device=dl)
        self.pk = {d: torch.zeros(1 + DEPTH, dtype=torch.long, device=d) for d in self.sdevs}
        self.pres = torch.tensor([[0.0], [1.0]], dtype=torch.float32, device=d0)            # conversation: present_A 0, present_B 1
        self.idsA = torch.full((1, 1), PAD, dtype=torch.long, device=d0)
        self.depA = torch.full((1, 1, DEPTH), -1, dtype=torch.long, device=d0)
        self.xo = [torch.zeros(1, 2, H, dtype=BF, device=d) for d, _ in self.segs]
        self.xi = [None] + [torch.zeros(1, 2, H, dtype=BF, device=d) for d, _ in self.segs[1:]]
        self.off_t = torch.as_tensor(OFF, device=dl)
        self.OFF0 = torch.as_tensor(OFF, device=d0)
        self.pf = torch.zeros(len(FKEYS), dtype=torch.float32, device=dl)
        self.pi = torch.zeros(len(IKEYS), dtype=torch.long, device=dl)
        self.params = {}
        self.set_params(DEFAULTS)
        with torch.cuda.device(d0):
            self.enc = enc if enc is not None else VS.load_audio_tower(d0)
            self.vx = VoxStreamFast(self.enc, 1, self.n_fr, d0, self.fa, FO)
            mm = mimi if mimi is not None else VS.MimiStream(d0).mimi
            self.mimi = mm
            self.mdec = MimiDecStream(mm, 1, d0)
        self.S0 = VS.CTX24
        self.abuf = torch.zeros(1, self.S0 + self.n_fr * SPF + VS.LOOK24 + 2 * SPF, device=d0)
        self.arWin = torch.arange(VS.CTX24 + SPF + VS.LOOK24, device=d0)
        self.Pd0 = torch.zeros(1, dtype=torch.long, device=d0)
        self.a_vec0 = torch.zeros(2, H, dtype=BF, device=d0)                                    # row 0 your mic, row 1 the uncond source
        self.s_t = torch.exp(m.vgain.theta.float()).to(BF)                                      # = VGain: y * exp(theta).to(y.dtype)
        self.usrc = torch.zeros(1, dtype=torch.long, device=d0)
        self.utab = self._uncond_tables()                                                       # [2 sources, n_fr, H]
        self.cbB = torch.zeros(1, DEPTH, dtype=torch.long, device=d0)
        self.ybuf = torch.zeros(1, SPF, device=d0)
        self.vrec = torch.zeros(self.n_fr, H, dtype=BF, device=d0)
        self.graphs = {}
        self.P = None
        self.last_ptok = [0.0, 0.0]
        self.last_cfgd = None
        self.last_adapt = [0.0, 1.0, 1.0]
        self.last_padad = [0.0, 0.0, 0.0]
        self.uncond_src = UNCOND_SRC[0]

    @torch.no_grad()
    def _uncond_tables(self):
        """the uncond stream's A input per conversation frame f for each source (= s4live LiveS4U: one streaming Voxtral pass over the
        source waveform, then adapter x s): quiet_frames = s4live's quiet bank frames (training level) tiled in order; digital_zero = zeros."""
        d0, nfr = self.d0, self.n_fr
        z = np.load(QUIET_BANK)
        bank = z["audio"].astype(np.float32)
        idx = np.arange(nfr) % bank.shape[0]
        srcs = {"quiet_frames": bank[idx].reshape(-1), "digital_zero": np.zeros(nfr * SPF, np.float32)}
        tab = torch.zeros(len(UNCOND_SRC), nfr, self.H, dtype=BF, device=d0)
        with torch.cuda.device(d0):
            for si, name in enumerate(UNCOND_SRC):
                wav = torch.as_tensor(np.ascontiguousarray(srcs[name]), dtype=torch.float32, device=d0)[None]
                vx = VS.VoxStream(self.enc, 1, nfr + 8, d0, self.fa)
                f = 0
                while f < nfr:
                    n = min(125, nfr - f)
                    lat = vx.encode(VS.window(wav, f, n).to(d0), n)
                    tab[si, f:f + n] = (self.m.vadapter(lat)[0] * self.s_t).to(BF)
                    f += n
                del vx, wav
        torch.cuda.empty_cache()
        log(f"uncond A tables: {', '.join(f'{n} rms {float(tab[i].float().pow(2).mean().sqrt()):.4f}' for i, n in enumerate(UNCOND_SRC))} "
            f"({nfr} frames; quiet bank {bank.shape[0]} frames)")
        return tab

    # ---------------------------------------------------------------------------------------------- params
    def prm(self, k):
        return self.pf[FKEYS.index(k)] if k in FKEYS else self.pi[IKEYS.index(k)]

    def set_params(self, d):
        clean, bad = validate(d)
        of = self.params.get("factorised_text")
        if ("factorised_text" in clean and of is not None and int(clean["factorised_text"]) != int(of) and "min_p" not in clean
                and abs(float(self.params.get("min_p", -1.0)) - MINP_DEFAULT[int(of)]) < 1e-9):
            clean["min_p"] = MINP_DEFAULT[int(clean["factorised_text"])]   # mode switch: min-p moves to that mode's default unless it was changed
        new = dict(self.params, **clean)
        self.pf.copy_(torch.tensor([float(new.get(k, 0.0)) for k in FKEYS], dtype=torch.float32))
        self.pi.copy_(torch.tensor([int(new.get(k, 0)) for k in IKEYS], dtype=torch.long))
        self.params = new
        return clean, bad

    def cb0_cfg_live(self):
        return bool(self.params.get("cfg_on")) and bool(self.params.get("cfg_cb0"))

    # ---------------------------------------------------------------------------------------------- pieces (batch 2: h [1, 2, H] = 2 rows at ONE position)
    def _layer_step2(self, L, h, cos, sin, kc, vc, sl):
        at, ln1, ln2, mlp = L.self_attn, L.input_layernorm, L.post_attention_layernorm, L.mlp
        x = FO.rms(h, ln1.weight, float(ln1.variance_epsilon))
        q, k = FO.qk_prep(at.q_proj(x), at.k_proj(x), at.q_norm.weight, at.k_norm.weight, float(at.q_norm.variance_epsilon), cos, sin)
        hd = at.head_dim
        v = at.v_proj(x).view(2, 1, -1, hd)
        a = self.fa.flash_attn_with_kvcache(q.view(2, 1, -1, hd), kc, vc, k=k.view(2, 1, -1, hd), v=v, cache_seqlens=sl,
                                            softmax_scale=at.scaling, causal=True, num_splits=1)
        h, x = FO.add_rms(h, at.o_proj(a.reshape(1, 2, -1)), ln2.weight, float(ln2.variance_epsilon))
        return h + mlp.down_proj(FO.silu_mul(mlp.gate_proj(x), mlp.up_proj(x)))

    def text_logits(self, hn):
        x = self.m.head.proj(hn)
        z = (x @ self.m.head.lm_head.weight.t()).float()
        z[:, self.spec] = (x @ self.R.t()).float()
        return z

    def _speaking(self, T):
        """adaptive rep-pen (module doc): -> bool [] = B's lane holds a word piece at a conversation position in [T - N, T - 1]."""
        idx = T - WMAX_S + self.arWS
        lw = self.laneB.index_select(0, idx.clamp_min(0))
        ok = (idx >= self.Pd) & (idx >= T - self.prm("speak_window_frames"))
        return ((lw < TE) & (lw != PAD) & (lw != NW) & (lw != EOB) & ok).any()

    def _rep_eff(self, spk, static, speaking, silent):
        """the repetition penalty in effect this frame: adaptive_rep_on ? (SPEAKING ? speaking : silent) : static (the static value itself)."""
        return torch.where(self.prm("adaptive_rep_on") > 0, torch.where(spk, self.prm(speaking), self.prm(silent)), self.prm(static))

    def _pad_bias_eff(self, T):
        """adaptive PAD bias (module doc): -> fp32 [] = the PAD bias in effect at frame T: adaptive_pad_on ? (SILENT ? pad_bias_silent :
        pad_bias_other) : pad_bias (the static value itself). SILENT = frames since B's last spoken lane token (self.lastspk, P at start)
        >= pad_silent_s x FPS (- 1e-3: fp32 slack, so e.g. 2.4 s = 30 frames exactly). Causal: frame T's own token is not drawn yet.
        Writes self.padad (on, seconds since, bias in effect)."""
        fs = (T - self.lastspk)[0].float()
        silent = fs >= self.prm("pad_silent_s") * FPS - 1e-3
        on = self.prm("adaptive_pad_on") > 0
        pb = torch.where(on, torch.where(silent, self.prm("pad_bias_silent"), self.prm("pad_bias_other")), self.prm("pad_bias"))
        self.padad[0].copy_(on.float())
        self.padad[1].copy_(fs / FPS)
        self.padad[2].copy_(pb)
        return pb

    def _text(self):
        """= live_engine._tail for channel B: CFG combine on the raw logits -> in-vocab bound (NO lane grammar, 10-28) -> t_text_draw: factorised (default) = category
        (PAD / NEW_WORD / WORD [/ EOB], cat_temp) then, if WORD, the word piece over word pieces only with frequency / repetition penalty over
        B's ring window -> temperature -> word min-p -> top-k -> top-p -> word-greedy; joint (factorised_text 0) = those knobs on the whole
        lane distribution (pre-10-17). First: the frame's SPEAKING flag and the adaptive rep-pens (self.adapt, read by _audio of the same frame).
        10-24: the PAD bias in effect (static, or adaptive from B's silence time: _pad_bias_eff) at the PAD BIAS point."""
        T = self.Td[self.dl]
        spk = self._speaking(T)
        self.adapt[0].copy_(spk.float())
        self.adapt[1].copy_(self._rep_eff(spk, "audio_rep_pen", "audio_rep_speaking", "audio_rep_silent"))
        self.adapt[2].copy_(self._rep_eff(spk, "text_rep", "text_rep_speaking", "text_rep_silent"))
        z2 = self.text_logits(self.hn_prev)
        self.zt.copy_(z2)
        zc, zu = z2[0:1], z2[1:2]
        sc = torch.where(self.prm("cfg_on") > 0, self.prm("cfg_s"), torch.ones_like(self.prm("cfg_s")))
        zmix = torch.where(sc == 1.0, zc, zu + sc * (zc - zu))
        fin = torch.isfinite(zc) & torch.isfinite(zu)
        self.cfgd[0].copy_(torch.where(fin, (zc - zu).abs(), torch.zeros_like(zc)).sum() / fin.sum().clamp_min(1))
        self.cfgd[1].copy_(0.5 * (torch.softmax(zc, -1) - torch.softmax(zu, -1)).abs().sum())
        pr = torch.softmax(zc, -1)
        self.probs[0].copy_(pr[0, PAD]); self.probs[1].copy_(pr[0, NW])
        tt, tk, tp = self.prm("text_temp"), self.prm("text_topk"), self.prm("text_topp")
        mp, mrel = self.prm("min_p"), self.prm("min_p_rel")
        fp, fw, rp = self.prm("freq_pen"), self.prm("freq_win"), self.adapt[2]
        cnt = self.ring.counts(fw)
        cnt2 = self.ring.counts(self.prm("text_rep_win"), self.ring.cnt2)
        z = zmix.masked_fill(~self.in_vocab[None], NINF)   # 10-28 USER ORDER: no lane grammar, ever: full in-vocab distribution (PAD / NEW_WORD / EOB / words after anything)
        z = torch.where(self.is_pad[None], z + self._pad_bias_eff(T), z)  # 10-18 PAD BIAS (module doc): 0 -> z exactly; 10-24 adaptive (off = pad_bias)
        q, pq = t_text_draw(z, self.is_word, cnt, cnt2, self.prm("factorised_text"), self.prm("cat_temp"), tt, tk, tp, mp, mrel, fp, rp,
                            self.prm("word_greedy"), self.gen_t, self.gen_c)
        self.laneB.index_copy_(0, T, q)
        self.ring.push(q)
        self.ptok[0].copy_(pr.gather(1, q[:, None])[0, 0])
        self.ptok[1].copy_(pq[0])

    def _presence_live(self):
        """= live_engine._presence_live (B only): codes of the last audio_rep_win frames (<= WMAX_A), conversation only -> [1, 32, CBS]."""
        T = self.Td[self.dl]
        idx = T - WMAX_A + self.arWA
        win = self.depB.index_select(0, idx.clamp_min(0))[None]
        return presence(win, self.off_t, CBS, (idx >= self.Pd + CD) & (idx >= T - self.prm("audio_rep_win")))

    def _audio(self, full):
        m, dl = self.m, self.dl
        tw = m.tower_s
        T = self.Td[dl]
        sl = self.T2[dl].to(torch.int32)
        tok = self.laneB.index_select(0, T)
        # 10-24 adaptive PAD bias: B's lane token of T as finally committed (after the cold-start PAD / parity forcing in step()) -> the
        # last-spoken position read by the NEXT frame's text graph (word piece or NEW_WORD = spoken; PAD / EOB / anything else = not)
        self.lastspk.copy_(torch.where((tok < TE) | (tok == NW), T, self.lastspk))
        dprev = self.depB.index_select(0, T - 1)
        e = tw.text_emb(tok) + tw.code_emb(dprev).to(BF) + tw.cond(self.hn_prev).to(BF)       # [2, Hv]: rows differ in hn[T-1] only
        h = e[None]
        T2 = self.T2[dl]
        cos, sin = self.rope_t[0].index_select(1, T2), self.rope_t[1].index_select(1, T2)
        for i, L in enumerate(self.ts.L):
            h = self._layer_step2(L, h, cos, sin, self.tsK[i], self.tsV[i], sl)
        hv = FO.rms(h[0], tw.norm.weight, float(tw.norm.variance_epsilon))                    # [2, Hv]
        z0 = (hv @ tw.cb0_head.weight.t()).float()
        self.z0.copy_(z0)
        if full:
            zc, zu = z0[0:1], z0[1:2]
            sa = torch.where((self.prm("cfg_on") > 0) & (self.prm("cfg_cb0") > 0), self.prm("cfg_s"), torch.ones_like(self.prm("cfg_s")))
            zB = torch.where(sa == 1.0, zc, zu + sa * (zc - zu))
            self.cfgd[2].copy_((zc - zu).abs().mean())
            self.cfgd[3].copy_(0.5 * (torch.softmax(zc, -1) - torch.softmax(zu, -1)).abs().sum())
            pres = self._presence_live()
            rp = self.adapt[1]                                                                  # = audio_rep_pen unless adaptive_rep_on (_text)
            if self.k_then_p:                                                                   # 10-28 hybrid-live: top-k then top-p (class doc)
                k0, kl = self.prm("cb0_topk"), self.prm("light_topk")
                p0, pl = self.prm("cb0_topp"), self.prm("light_topp")
            else:
                tr = self.prm("audio_trunc") > 0                                                # 10-27 user: top-k OR top-p on the audio codes
                k0, kl = (torch.where(tr, torch.zeros_like(self.prm(x)), self.prm(x)) for x in ("cb0_topk", "light_topk"))
                p0, pl = (torch.where(tr, self.prm(x), torch.ones_like(self.prm(x))) for x in ("cb0_topp", "light_topp"))
            c0 = t_sample(ctrl_penalty(zB, pres[:, 0], rp), self.gen_a, self.prm("cb0_temp"), k0, p0)
            codes = self.light.gen(hv[0:1], c0, self.gen_a, self.prm("light_temp"), kl, pres, rp, topp=pl)[0]
            self.depB.index_copy_(0, T, (codes + self.off_t)[None])
        self.pk[dl].copy_(torch.cat([self.laneB.index_select(0, T), self.depB.index_select(0, T)[0]]))

    def _seg(self, si):
        d, idx = self.segs[si]
        T2 = self.T2[d]
        sl = T2.to(torch.int32)
        if si == 0:
            pk = self.pk[d]
            ids = torch.cat([self.idsA, pk[0:1].view(1, 1)], 0)                                   # [2, 1]
            dep = torch.cat([self.depA, pk[1:].view(1, 1, DEPTH)], 0)                             # [2, 1, 32]
            h = (self.m.inp(ids, dep, self.pres, None) + self.a_vec0)[None]                       # [1, 2, H]: trainer inp(...) + s x adapter
        else:
            h = self.xi[si]
        cos, sin = self.rope[d][0].index_select(1, T2), self.rope[d][1].index_select(1, T2)
        for i in idx:
            h = self._layer_step2(self.bb.L[i], h, cos, sin, self.bbK[i], self.bbV[i], sl)
        if si == len(self.segs) - 1:
            n = self.m.head.norm
            self.hn_prev.copy_(FO.rms(h[0], n.weight, float(n.variance_epsilon)))
        else:
            self.xo[si].copy_(h)

    def _a_rows(self, lat, f):
        self.a_vec0[0:1].copy_((self.m.vadapter(lat).reshape(1, -1) * self.s_t).to(BF))
        self.a_vec0[1:2].copy_(self.utab.view(-1, self.H).index_select(0, self.usrc * self.n_fr + f))

    def _front_eager(self, f):
        idx = self.S0 + f * SPF - VS.CTX24 + self.arWin
        lat = self.vx.encode(self.abuf.index_select(1, idx), 1)
        self._a_rows(lat, torch.tensor([f], device=self.d0))

    def _front_graphable(self):
        f = self.Td[self.d0] - self.Pd0
        idx = self.S0 + f * SPF - VS.CTX24 + self.arWin
        lat = self.vx.step_graphable(self.abuf.index_select(1, idx))
        self._a_rows(lat, f)

    def _mdec(self):
        loc = (self.cbB - self.OFF0[None]).clamp(0, CBS - 1)
        self.ybuf.copy_(self.mdec.step(loc))

    def _mdec_state(self):
        md = self.mdec
        out = list(md.K) + list(md.V) + [md.slot_pos, md.k, md.ups_tail]
        for _, _, st in md.ops:
            for s in (st if isinstance(st, list) else [st]):
                if s is not None:
                    out.append(s)
        return out

    # ---------------------------------------------------------------------------------------------- graphs
    def set_T(self, T):
        for d in self.sdevs:
            with torch.cuda.device(d):
                self.Td[d].fill_(int(T)); self.T2[d].fill_(int(T))

    @torch.no_grad()
    def capture(self, T=64, P=32):
        """capture every graph once (warm-ups / capture write KV slot T and scratch state; every slot is rewritten before it is read in a
        session, and start() resets every stream state + re-seeds the generators)."""
        t0 = time.time()
        self.set_T(T)
        self.Pd0.fill_(P); self.Pd.fill_(P)
        self.laneB.fill_(PAD); self.depB.fill_(-1)

        def cap(name, dev, fn, gens=()):
            with torch.cuda.device(dev):
                s = torch.cuda.Stream(dev)
                s.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(s):
                    for _ in range(2):
                        fn()
                torch.cuda.current_stream(dev).wait_stream(s)
                torch.cuda.synchronize(dev)
                g = torch.cuda.CUDAGraph()
                for gen in gens:
                    g.register_generator_state(gen)
                with torch.cuda.graph(g, stream=s):
                    fn()
                torch.cuda.synchronize(dev)
            self.graphs[name] = g
        cap("text", self.dl, self._text, (self.gen_t, self.gen_c))
        cap("audio", self.dl, lambda: self._audio(False))
        cap("audio_full", self.dl, lambda: self._audio(True), (self.gen_a,))
        for si, (d, _) in enumerate(self.segs):
            cap(f"seg{si}", d, lambda si=si: self._seg(si))
        vx = self.vx
        sv = (vx.mel_prev.clone(), vx.c1_prev.clone(), vx.kd.clone(), self.a_vec0.clone())
        vx.kd.fill_(1)
        cap("front", self.d0, self._front_graphable)
        vx.mel_prev.copy_(sv[0]); vx.c1_prev.copy_(sv[1]); vx.kd.copy_(sv[2]); self.a_vec0.copy_(sv[3])
        st = [t.clone() for t in self._mdec_state()]
        cap("mdec", self.d0, self._mdec)
        for t, s_ in zip(self._mdec_state(), st):
            t.copy_(s_)
        self.ring.reset()
        for d in self.sdevs:
            torch.cuda.synchronize(d)
        log(f"captured {len(self.graphs)} graphs ({', '.join(self.graphs)}) in {time.time() - t0:.1f}s; " +
            ", ".join(f"cuda:{d.index} {torch.cuda.memory_allocated(d) / 2 ** 30:.1f} GiB" for d in self.sdevs))

    # ---------------------------------------------------------------------------------------------- session
    @torch.no_grad()
    def start(self, pre, seed, params=None, uncond_src="quiet_frames"):
        """pre = the system-prompt prefix (sysprompt.build_prefix dict: ids [2, P], depth [2, P, 32], present [2, P]): prefill [0, P)
        eagerly (backbone + speech tower KV, row 0; copied to the uncond row 1: the prefix is the same), reset every stream state, seed the
        generators (text seed + 1, audio seed); uncond_src picks the uncond stream's A table (CFG)."""
        P = int(pre["P"])
        assert 5 <= P and P + self.n_fr + CD <= self.lmax, ("prefix too long for the static buffers", P, self.lmax)
        assert uncond_src in UNCOND_SRC, uncond_src
        ids, depth, present = (np.asarray(pre[k]) for k in ("ids", "depth", "present"))
        assert ids.shape == (2, P) and depth.shape == (2, P, DEPTH) and present.shape == (2, P)
        dl, d0 = self.dl, self.d0
        self.P = P
        self.laneB.fill_(PAD); self.depB.fill_(-1)
        self.laneB[:P].copy_(torch.as_tensor(ids[1], device=dl)); self.depB[:P].copy_(torch.as_tensor(depth[1], device=dl))
        e = self.m.inp(torch.as_tensor(ids, device=d0), torch.as_tensor(depth, device=d0), torch.as_tensor(present, device=d0))[None]
        h = self.bb.run(e, 0)
        n = self.m.head.norm
        with torch.cuda.device(dl):                                   # Triton kernels launch on the CURRENT device (stream order!)
            hn = FO.rms(h[0].to(dl), n.weight, float(n.variance_epsilon))
            tw = self.m.tower_s
            hprev = torch.cat([torch.zeros_like(hn[:1]), hn[:-1]], 0)
            dprev = np.concatenate([np.full((1, DEPTH), -1, np.int64), depth[1, :-1]], 0)
            et = tw.text_emb(torch.as_tensor(ids[1], device=dl)) + tw.code_emb(torch.as_tensor(dprev, device=dl)).to(BF) + tw.cond(hprev).to(BF)
            self.ts.run(et[None], 0)
            self.hn_prev.copy_(hn[-1:].expand(2, -1))
            self.Pd.fill_(P)
            self.ring.reset()
            self.lastspk.fill_(P)                                     # 10-24 adaptive PAD bias: silence counts from the first conversation frame
            self.fstate.zero_()
        for K_ in (self.bbK, self.bbV, self.tsK, self.tsV):
            for t in K_:
                with torch.cuda.device(t.device):
                    t[1, :P].copy_(t[0, :P])
        self.prefill_hn = hn
        with torch.cuda.device(d0):
            self.vx.reset(); self.abuf.zero_(); self.a_vec0.zero_(); self.mdec.reset(); self.ybuf.zero_()
            self.Pd0.fill_(P)
            self.usrc.fill_(UNCOND_SRC.index(uncond_src))
        self.uncond_src = uncond_src
        self.gen_t.manual_seed(int(seed) + 1)
        self.gen_a.manual_seed(int(seed))
        self.gen_c.manual_seed(int(seed) + 2)
        self.params = {}
        p0 = dict(DEFAULTS, **(params or {}))
        if "min_p" not in (params or {}):                             # min-p default follows the text mode (MINP_DEFAULT)
            p0["min_p"] = MINP_DEFAULT[int(validate({"factorised_text": p0["factorised_text"]})[0].get("factorised_text", DEFAULTS["factorised_text"]))]
        self.set_params(p0)
        for d in self.sdevs:
            torch.cuda.synchronize(d)

    def write_a(self, i0, x, peak=0.995):
        """A samples [i0, i0 + len(x)) of the conversation (24 kHz float32, loudness-matched) -> abuf (hard ceiling = training's peak guard)."""
        if len(x):
            x = np.clip(np.ascontiguousarray(x, np.float32), -peak, peak)
            n = min(len(x), self.abuf.shape[1] - self.S0 - i0)
            if n > 0:
                self.abuf[0, self.S0 + i0:self.S0 + i0 + n].copy_(torch.from_numpy(x[:n]))

    @torch.no_grad()
    def step(self, T, graphs=True, force=None):
        """frame f = T - P (module doc). Async: result(T, dec) waits. COLD START: lane_B[P] is forced to PAD (= the live engines' cold_pad:
        the head at the prefix's last position is never trained, and a training row's first lane frame is never a word piece).
        force = (lane token, B codes [32] global or None) (parity tests only): teacher-forces B's lane / codes of position T."""
        P = self.P
        f = T - P
        assert f >= 0
        self.set_T(T)
        d0, dl = self.d0, self.dl
        use = graphs and bool(self.graphs)
        with torch.cuda.device(d0):                                                  # front first: GPU 0 idles while the head GPU samples
            if f == 0 or not use:
                assert self.vx.k == f, ("Voxtral front out of step", self.vx.k, f)
                self._front_eager(f)
            else:
                self.graphs["front"].replay()
                self.vx.k += 1
        full = T >= P + CD
        with torch.cuda.device(dl):
            if use:
                self.graphs["text"].replay()
            else:
                self._text()
            if f == 0 or force is not None:
                self.laneB[T:T + 1].fill_(PAD if force is None else int(force[0]))
                if f == 0:
                    self.ring.reset()                                                 # the ring holds no word for the forced PAD
            if use:
                self.graphs["audio_full" if full else "audio"].replay()
            else:
                self._audio(full)
            if force is not None and force[1] is not None:
                self.depB[T].copy_(torch.as_tensor(np.asarray(force[1], np.int64), device=dl))
                self.pk[dl].copy_(torch.cat([self.laneB[T:T + 1], self.depB[T]]))
        for si, (d, _) in enumerate(self.segs):
            with torch.cuda.device(d):
                if si == 0:
                    if d != dl:
                        self.pk[d].copy_(self.pk[dl])
                else:
                    self.xi[si].copy_(self.xo[si - 1])
                if use:
                    self.graphs[f"seg{si}"].replay()
                else:
                    self._seg(si)
        with torch.cuda.device(d0):
            self.vrec[f].copy_(self.a_vec0[0])
            if full:
                self.cbB.copy_(self.depB[T:T + 1])
                if use:
                    self.graphs["mdec"].replay()
                else:
                    self._mdec()
        return full

    def result(self, T, dec):
        torch.cuda.synchronize(self.dl)
        tokB = int(self.laneB[T].item())
        pp, pn = self.probs.tolist()
        self.last_ptok = self.ptok.tolist()
        self.last_cfgd = self.cfgd.tolist()
        fs = self.fstate.tolist()                                    # one copy: adaptive rep-pen [0:3] + adaptive PAD bias [3:6] (10-24)
        self.last_adapt, self.last_padad = fs[0:3], fs[3:6]
        y = None
        if dec:
            torch.cuda.synchronize(self.d0)
            y = self.ybuf[0].cpu().numpy().copy()
        return tokB, pp, pn, y
