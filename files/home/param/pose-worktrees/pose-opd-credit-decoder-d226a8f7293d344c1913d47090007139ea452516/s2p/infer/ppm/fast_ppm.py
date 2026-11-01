# POSE-S2P BRANCH (ppm/pose-s2p, 10-27): copy of aloe-pose-monoplex-infer/ppm/fast_ppm.py + speech -> pose conditioning (PPM_S2P / the
# checkpoint's tower_p.s_emb; module doc "POSE-S2P"). Flag off (no s_emb: every deployed checkpoint) = the original code path line for line.
# VENDORED for aloe-pose-monoplex-infer. Upstream: ppm/fastinfer/fast_ppm.py. Edits (behaviour-neutral for this model): repo-local imports (ppm_io for infer_ppm); Voxtral front + streaming Mimi decoder imports removed and asserted off; no HF_HUB_OFFLINE / VOXTRAL_DIR defaults.
#!/usr/bin/env python3
"""FAST PPM ENGINE WITH POSE (CUDA graphs, bf16): the live Aloe Voxtral demo engine (ppm/aloe-m1vox-live/live_ppm.py, copied, not imported)
extended with channel-B POSE generation, several instances per engine (rows), channel-A feed modes and pose CFG rows.

MODEL     the trainer's model_ppm (aloe-m1vox-ft/code copy = ppm/code model_ppm + optional vadapter / nopose; the pose maths are identical
          to ppm/code, aloe-pose-ft/code, aloe-uncond-ft/code: diffed). Pose parts (tower_p, dec_p, decp_bb, inp.pose) are built when the
          checkpoint has them; the Voxtral front (vadapter x s on A) when it has vadapter.*. bf16 weights.
DEVICES   backbone layers split over the GPUs (last GPU gets fewer); text head + speech tower + fused 0.6B speech decoder on the LAST GPU;
          pose tower + fused 0.6B pose decoder + decp_bb on the FIRST GPU (pose_dev) so the pose branch runs CONCURRENTLY with the speech
          branch; front (Voxtral + adapter) and streaming Mimi decoder on the first GPU.
ROWS      NI instances (independent conversations or the two sides of a duo), sharing the weights, batched in every graph; with pose CFG
          (cfg_rows) NI more rows = each instance's uncond stream. Each row has its own system-prompt length P_r: row r's position at
          conversation frame f is T_r = P_r + f (per-row KV lengths / RoPE positions).
LAYOUT    = the trainer's micro (data_ppm.speech_micro + pose arrays; model_ppm.PPM.forward): trunk input at position T
            e[T] = sum_c present_c[T] (emb_c(ids_c[T]) + aud_c(depth_c[T])) + pose_emb(pdepth[T]) [+ s x adapter(Voxtral(A frame f))];
          B Mimi codes AND B pose codes at code delay CD = 2 (positions P, P + 1 carry neither); towers at T read text_emb(lane_B[T]) +
          code_emb(codes at T - 1) + cond(hn[T - 1]); the pose decoder's slot 0 = proj(hv_p[T]) + alpha x decp_bb(hn[T - 1]) (D33).
CHANNEL A per instance: 'empty' (PAD / Mimi -1 / present 0: the monoplex pose-only rows), 'file' (words + Mimi from arrays, e.g. the row's
          real partner), 'partner' (another instance's B text + Mimi at the same step: duo, = infer_pose_2side's cross-feed; never its
          pose), 'voxtral' (live 24 kHz audio through the streaming Voxtral front, NI = 1).
PER FRAME f (T_r = P_r + f; every step a CUDA graph except the very first front frame; device-side frame index, no host sync):
  text   (last)   lane_B[T] ~ head(hn[T-1]) (live engine sampler: grammar, penalties, temperature, min-p, top-k, top-p).
  pose   (first)  pose tower step T on all rows (tok + pose codes T - 1 + cond(hn[T-1])) -> cb0 (pose CFG mix when on) -> fused 0.6B pose
                  decoder cb1..31 (slot 0 + decp_bb link; CFG per codebook) for f >= CD; runs while the speech branch runs on the last GPU.
  speech (last)   speech tower step T (instance rows) -> cb0 -> fused 0.6B speech decoder cb1..31 for f >= CD.
  trunk  (all)    e[T] (A per mode; uncond rows: every Mimi code -> the Mimi codes of digital silence) -> 64 layers -> hn[T].
  B out  (first)  streaming Mimi decoder (graphed) on the instances' B codes of T (= B frame f - 2) when mimi_stream.
POSE CFG  (= infer_pose_2side --pose-cfg w, SAM scope 'all'): uncond row = the same text lanes, system-prompt text and pose history, every
          Mimi input of both channels (voice clips, prefill, generated) -> Mimi codes of digital silence (frame index restarting at each
          contiguous coded run); pose cb0 and cb1..31 logits l = lu + w (lc - lu), one code per instance fed to both rows. Off by default.
          Live knobs pose_cfg_on (default 0), pose_cfg_s (default 3 = the SAM released recipe / the duo CFG-3 samples), pose_cfg_cb0 /
          pose_cfg_light (guide cb0 / cb1..31; both on = SAM scope 'all'); the uncond rows run every frame (switchable mid-session) when the
          engine is built with cfg_rows.
POSE REP  CTRL repetition penalty on pose cb0 (pose_rep_cb0) and per codebook on cb1..31 (pose_rep_light; 10-26 split, old pose_rep = both) on codes used in the last pose_rep_win conversation frames (
          default 1 = off; = the audio rep-pen's semantics) or, with pose_rep_win 0 (default; = PPM infer_ppm --pose-rep-win 0 and the
          slow sampler's --pose-rep), on every code GENERATED so far in the session (per codebook); after the CFG mix, before temperature / top-k.
ADAPTIVE  (= aloe-m1vox-live/stage-adaptrep live_ppm, copied, + pose): per frame and instance a SPEAKING flag = B's lane holds a word piece
          (id < TE) at a conversation position in [T - speak_window_frames, T - 1] (computed at the top of the text graph); with
          adaptive_rep_on the frame's audio / text / pose repetition penalties = *_rep_speaking | *_rep_silent instead of the static
          audio_rep_pen / text_rep / pose_rep. Off -> the static values exactly (same kernels, same RNG draws).
TEXT FORCING (10-18, default off = byte-identical output): per instance row, B's lane can be teacher-forced from the session start: start()
          reads inst['text_force'] (a lane array over the row's positions, e.g. the row's real ids[1]) into the device buffer tforce [NI, lmax]
          (-1 = not forced) for positions >= P + f0; inside the text graph the drawn token q at T is replaced by tforce[T] when >= 0, BEFORE it
          is written to lane_B / tok_now / the ring, so the speech tower, pose tower, trunk input (cond and CFG uncond rows) and every later
          position see the forced token exactly as a sampled one (KV / state consistent). The text draw itself still runs (same RNG use).
POSE-S2P  (branch, off unless the model has tower_p.s_emb: PPM_S2P=1|full|cb0, or a checkpoint trained with it) the pose tower input at T
          gains s_emb(B's SPEECH codes sampled at the SAME T) (= the trainer's model_ppm.S2PEmb, zero at init). Same timing (pose frame f and
          speech frame f still leave in the same step); inside the step the order becomes text -> speech -> pose. 'full' (32 codebooks): the
          pose branch waits for the whole speech branch (tower + cb0 + 31-codebook decoder); 'cb0': speech is split into two graphs
          (tower + cb0 draw | light decoder) and pose starts right after the cb0 draw, concurrently with the speech light decoder. Forced
          speech codes (force['codes'], pose-only runs) are written BEFORE pose reads them. Uncond pose-CFG rows: s_emb reads the Mimi codes of
          digital silence (= the trunk's uncond rule: sil_t[T - P - CD] where B has codes at T, -1 elsewhere); prefill: s_emb(depth[1]) of the
          row's own prefix (uncond rows: silence_depth's codes; the voice clip's codes included, as in training). Positions without speech codes
          (system block outside the clip, frames 0, 1) -> -1 -> 0. No audio-CFG rows exist in this engine.
SAMPLING  live device tensors (one capture serves every setting). Standing rules: temperature 1 everywhere, audio top-k 30 (cb0 + light),
          pose top-k 30 (cb0 + light), no repetition penalty, no min-p, CFG off, bf16 only (all new knobs default off)."""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("PPM_CD", "2")
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import model_ppm as M  # noqa: E402
import fused_triton as FO  # noqa: E402
import fa3_op  # noqa: E402
import speech_q as SP  # noqa: E402
import ppm_io as IP  # noqa: E402   (vendored subset of ppm/code/infer_ppm.py: Mimi, wav io, tokenizer, lane words)
assert IP.M is M, "ppm_io must use the same model_ppm"
import light_kernels as LK  # noqa: E402
import light_exact_ppm as LX  # noqa: E402   (10-26: trainer-exact 0.6B decoder kernels; next to this file, HERE is on sys.path)
VS = VoxStreamFast = MimiDecStream = None      # Voxtral front / streaming Mimi decoder: not vendored (unused by the monoplex pose model)

BF = torch.bfloat16
CD, DEPTH, CBS, FPS = SP.CD, SP.DEPTH, SP.CBS, SP.FPS
PAD, NW, EOB, TE = SP.PAD_ID, SP.NW_ID, SP.EOB_ID, 151643
OFF = np.arange(DEPTH, dtype=np.int64) * CBS
SPF = 1920                       # 24 kHz samples per 12.5 Hz frame (= vox_stream.SPF24)
NINF = float("-inf")
assert CD == 2, CD
VADAPT = {"enc_hidden": 1280, "hidden": 4096, "stack": 4, "norm": "none"}
RUN_METRICS = ""                 # vadapter checkpoints only (not this model)
A_MODES = ("empty", "file", "partner", "voxtral")
MAX_ROWS = 64                    # instances per engine (was 8; batched offline generation: bench_batch.py measures up to 64)


def log(*a):
    print(f"[fastppm {time.strftime('%H:%M:%S')}]", *a, flush=True)


S2P_KEY = "tower_p.s_emb.weight"


def s2p_mode(v):
    """= the trainer's model_ppm.s2p_mode: None (off) | 'full' | 'cb0'."""
    v = str(v if v is not None else "0").strip().lower()
    if v in ("", "0", "off", "none", "false"):
        return None
    if v in ("1", "full", "on", "true"):
        return "full"
    assert v == "cb0", ("PPM_S2P must be 0 | 1 | full | cb0", v)
    return "cb0"


class S2PEmb(torch.nn.Module):
    """= the trainer's model_ppm.S2PEmb (pose-s2p): sum of per-code rows of B's speech codes at T ('full': 32 x 2048 rows on global ids,
    'cb0': 2048 rows on cb0); -1 -> 0 (H2.code_feedback)."""

    def __init__(self, H, mode):
        super().__init__()
        assert mode in ("full", "cb0"), mode
        self.mode = mode
        self.weight = torch.nn.Parameter(torch.empty((DEPTH if mode == "full" else 1) * CBS, H))

    def forward(self, depth):
        return M.H2.code_feedback(depth if self.mode == "full" else depth[..., :1], self.weight)


# ================================================================================================ checkpoint
def ckpt_info(path):
    """consolidated bf16 DCP export (every model tensor + train_cfg.json) or a ppm-local-bf16-shards-v1 copy -> dict (sizes, pose parts,
    D33 link + alpha, vadapter)."""
    path = os.path.abspath(path.rstrip("/"))
    if os.path.isfile(f"{path}/meta.json"):
        meta = json.load(open(f"{path}/meta.json"))
        assert meta.get("format") == "ppm-local-bf16-shards-v1", meta.get("format")
        assert os.path.isfile(f"{path}/COMPLETE"), ("local copy not COMPLETE", path)
        c = meta["cfg"]
        if "layers" in c:
            dims = {k: int(c[k]) for k in ("layers", "tower_layers", "dec_layers", "proj_r", "tower_vocab")}
        else:
            dims = {"layers": 64, "tower_layers": 36, "dec_layers": 28, "proj_r": 2048, "tower_vocab": M.TOWER_VOCAB}
        keys = set(torch.load(f"{path}/rep.pt", map_location="cpu", weights_only=True, mmap=True).keys())
        keys |= set(torch.load(f"{path}/rank0.pt", map_location="cpu", weights_only=True, mmap=True).keys())
        bb = "decp_bb.weight" in keys
        s2 = None
        if S2P_KEY in keys:                                                            # pose-s2p checkpoint: mode from the table's rows
            r_ = 0
            for fn in [f"rank{r}.pt" for r in range(int(meta["world"]))] + ["rep.pt"]:
                sd_ = torch.load(f"{path}/{fn}", map_location="cpu", weights_only=True, mmap=True)
                r_ += int(sd_[S2P_KEY].shape[0]) if S2P_KEY in sd_ else 0
            s2 = "full" if r_ == DEPTH * CBS else "cb0"
        return dict(kind="local", path=path, step=int(meta["step"]), meta=meta, pose=any(k.startswith("tower_p.") for k in keys), s2p=s2,
                    vad=any(k.startswith("vadapter.") for k in keys), decp_bb=bb,
                    decp_bb_alpha=IP.decp_bb_alpha_of(c, meta["step"]) if bb else None, **dims)
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
    bb = "decp_bb.weight" in shp
    if bb and "decp_bb_alpha" not in side:
        log(f"WARNING {path}: D33 decp_bb link but no train_cfg.json decp_bb_alpha -> alpha 1.0")
    return dict(kind="dcp", path=path, step=step, shapes=shp, side=side, layers=nlay("layers."), tower_layers=nlay("tower_s.layers."),
                dec_layers=nlay("dec_s.model.layers."), proj_r=shp["head.proj.G"][0], tower_vocab=shp["tower_s.text_emb.weight"][0],
                pose=any(k.startswith("tower_p.") for k in shp), vad=any(k.startswith("vadapter.") for k in shp), decp_bb=bb,
                decp_bb_alpha=float(side.get("decp_bb_alpha", 1.0)) if bb else None,
                s2p=(None if S2P_KEY not in shp else ("full" if shp[S2P_KEY][0] == DEPTH * CBS else "cb0")))


def s_from_metrics(step, path=RUN_METRICS):
    if not os.path.isfile(path):
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


def build_model(ck, devs, s_value=None, last_w=None):
    """-> (model, ldev, cfgs, info). Pose parts / D33 link / vadapter exactly as the checkpoint has them. Layer split: weights 1 per GPU and
    last_w for the last GPU (default 0.75 with pose: the first GPU also holds the pose branch; 0.45 without = the live engine's split)."""
    ci = ckpt_info(ck)
    c32, c4, c06 = M.cfg32(ci["layers"]), M.cfg4b(ci["tower_layers"]), M.cfg06(ci["dec_layers"])
    pose, vad, bb = ci["pose"], ci["vad"], ci["decp_bb"]
    torch.set_default_dtype(BF)
    with torch.device("meta"):
        m = M.PPM(c32, c4, c06, ci["proj_r"], ci["tower_vocab"], vadapter=VADAPT if vad else None, nopose=not pose, decp_bb=bb)
    s2 = ci.get("s2p")                                                                 # pose-s2p: the checkpoint's term, else PPM_S2P (zero init)
    s2e = os.environ.get("PPM_S2P")
    if s2 is not None:
        assert s2e is None or s2p_mode(s2e) == s2, ("PPM_S2P disagrees with the checkpoint's s_emb", s2e, s2)
    else:
        s2 = s2p_mode(s2e)
    if s2:
        assert pose, "pose-s2p needs a pose checkpoint"
        with torch.device("meta"):
            m.tower_p.s_emb = S2PEmb(c4.hidden_size, s2)
    torch.set_default_dtype(torch.float32)
    nL, nd = ci["layers"], len(devs)
    if last_w is None:
        last_w = 0.75 if pose else 0.45
    if nd == 1:
        ldev = [devs[0]] * nL
    else:
        w = [1.0] * (nd - 1) + [float(last_w)]
        cuts = np.cumsum(w) / sum(w) * nL
        ldev = [devs[int(np.searchsorted(cuts, i + 0.5))] for i in range(nL)]
    d0, dl = devs[0], devs[-1]
    m.inp.to_empty(device=d0)
    for L, d in zip(m.layers, ldev):
        L.to_empty(device=d)
    for mod in (m.head, m.tower_s, m.dec_s):
        mod.to_empty(device=dl)
    m.dec_s.model.rotary_emb = type(m.dec_s.model.rotary_emb)(config=c06, device=dl)
    if pose:
        for mod in (m.tower_p, m.dec_p) + ((m.decp_bb,) if bb else ()):
            mod.to_empty(device=d0)
        m.dec_p.model.rotary_emb = type(m.dec_p.model.rotary_emb)(config=c06, device=d0)
        if bb:
            m.decp_bb_alpha = float(ci["decp_bb_alpha"])
    if vad:
        m.vadapter.to_empty(device=d0)
        m.vgain = M.VGain().to(d0)
    load_weights(m, ci)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    info = {"ck": ci["path"], "step": ci["step"], "kind": ci["kind"], "pose": pose, "decp_bb": bb, "decp_bb_alpha": ci["decp_bb_alpha"], "s2p": s2,
            "s2p_zero_init": bool(s2) and ci.get("s2p") is None,
            "vadapter": vad, "ldev": [str(d) for d in sorted(set(ldev), key=str)],
            "layers_per_dev": {str(d): sum(1 for x in ldev if x == d) for d in devs}}
    if vad:
        th = float(m.vgain.theta.float().reshape(-1)[0])
        s_ck = math.exp(th)
        s_met = s_from_metrics(ci["step"])
        if s_value is not None:
            s_use, src = float(s_value), "given"
        elif s_met is not None and abs(math.log(s_met) - th) <= 0.04:
            s_use, src = s_met, f"trainer fp32 s at step {ci['step']} (metrics.jsonl)"
        else:
            s_use, src = s_ck, "exp(checkpoint theta)"
        with torch.no_grad():
            m.vgain.theta.fill_(math.log(s_use))
        info.update(s=s_use, s_src=src)
    log(f"[build] {ci['path']} step {ci['step']} ({ci['kind']}): pose {pose} (D33 alpha {ci['decp_bb_alpha']}), vadapter {vad}; layers "
        f"{info['layers_per_dev']}; head / speech tower / speech decoder on {dl}" + (f", pose tower / decoder on {d0}" if pose else ""))
    return m, ldev, (c32, c4, c06), info


def _raw_chunk(b, dst):
    """b = one chunk's torch.save bytes (zip, stored entries): -> CPU tensor viewing the raw storage record '<x>/data/0' when it holds
    exactly dst.numel() elements of dst.dtype (a plain contiguous tensor), else None (caller falls back to torch.load)."""
    import struct
    pos, n = 0, len(b)
    nb = dst.numel() * dst.element_size()
    while pos + 30 <= n and b[pos:pos + 4] == b"PK\x03\x04":
        flag, comp = struct.unpack_from("<HH", b, pos + 6)
        csz, usz, nl, el = struct.unpack_from("<IIHH", b, pos + 18, )
        name = b[pos + 30:pos + 30 + nl]
        d0 = pos + 30 + nl + el
        if flag & 8 or comp != 0:
            return None
        if name.endswith(b"/data/0"):
            if usz != nb or csz != nb or d0 + nb > n:
                return None
            if name.count(b"/data/") != 1:
                return None
            return torch.frombuffer(b, dtype=dst.dtype, count=dst.numel(), offset=d0).view(dst.shape)
        if name.split(b"/")[-1].isdigit() or b"/data/" in name:
            return None                                                                 # a second storage: not a single plain tensor
        pos = d0 + csz
    return None


@torch.no_grad()
def fast_dcp_load(params, path, threads=32):
    """parallel reader of a DCP checkpoint (FileSystemWriter format, torch.save per chunk, no transforms) straight into the given bf16
    parameters: the same bytes as dcp.load (every chunk = torch.load of its [offset, offset + length) slice, narrowed into the target),
    read by `threads` threads instead of one; asserts full element coverage of every parameter."""
    import io
    from concurrent.futures import ThreadPoolExecutor
    import torch.distributed.checkpoint as dcp
    md = dcp.FileSystemReader(path).read_metadata()
    sdm = md.state_dict_metadata
    by = {}
    for idx, info in md.storage_data.items():
        by.setdefault(idx.fqn, {})[tuple(int(x) for x in idx.offset)] = info
    jobs, cover = [], {}
    for n, p in params.items():
        tm = sdm[n]
        assert tuple(tm.size) == tuple(p.shape) and tm.properties.dtype == p.dtype, (n, tm.size, p.shape, tm.properties.dtype, p.dtype)
        for ch in tm.chunks:
            off, sz = tuple(int(x) for x in ch.offsets), tuple(int(x) for x in ch.sizes)
            info = by[n][off]
            assert not info.transform_descriptors, ("transformed DCP item", n)
            dst = p
            for dim, (o, s) in enumerate(zip(off, sz)):
                dst = dst.narrow(dim, o, s)
            jobs.append((info.relative_path, info.offset, info.length, dst, n))
            cover[n] = cover.get(n, 0) + int(np.prod(sz)) if sz else cover.get(n, 0) + 1
    bad = [n for n, p in params.items() if cover.get(n, 0) != p.numel()]
    assert not bad, ("DCP chunks do not cover", bad[:4])
    jobs.sort(key=lambda j: (j[0], j[1]))
    import threading
    fds, lock = {}, threading.Lock()
    nfall = [0]

    def run(job):
        rel, off, ln, dst, n = job
        fn = os.path.join(path, rel)
        with lock:
            fd = fds.get(fn)
            if fd is None:
                fd = fds[fn] = os.open(fn, os.O_RDONLY)
        b = os.pread(fd, ln, off)
        assert len(b) == ln, (n, rel, off, ln, len(b))
        t = _raw_chunk(b, dst)
        if t is None:                                                                  # not a plain contiguous storage record: torch.load
            t = torch.load(io.BytesIO(b), map_location="cpu", weights_only=True)
            nfall[0] += 1
        assert tuple(t.shape) == tuple(dst.shape) and t.dtype == dst.dtype, (n, t.shape, dst.shape, t.dtype)
        with torch.no_grad(), torch.cuda.device(dst.device):                           # grad mode is thread-local
            dst.copy_(t)
    with ThreadPoolExecutor(threads) as ex:
        for _ in ex.map(run, jobs, chunksize=64):
            pass
    for fd in fds.values():
        os.close(fd)
    if nfall[0]:
        log(f"[load] {nfall[0]} chunks via torch.load (no plain storage record)")
    for d in {p.device for p in params.values()}:
        torch.cuda.synchronize(d)
    return len(jobs)


@torch.no_grad()
def load_weights(m, ci):
    t0 = time.time()
    params = {n: p for n, p in m.named_parameters()}
    if S2P_KEY in params and ci.get("s2p") is None:                                    # pose-s2p on a checkpoint without the term: ZERO (step 0)
        params.pop(S2P_KEY).zero_()
        log(f"[load] pose-s2p: {S2P_KEY} not in the checkpoint -> zero (= the trainer's init: today's model exactly)")
    want = set(params)
    if ci["kind"] == "dcp" and os.environ.get("FASTPPM_LOADER", "dcp") == "fast":          # parallel reader: same bytes, no faster (torch.load-bound)
        have = set(ci["shapes"])
        assert want <= have, ("checkpoint lacks model tensors", sorted(want - have)[:6])
        tgt = {n: (params[n] if params[n].dtype == BF else torch.empty(params[n].shape, dtype=BF, device=params[n].device)) for n in want}
        nj = fast_dcp_load(tgt, ci["path"], int(os.environ.get("FASTPPM_LOAD_THREADS", "32")))
        for n in want:
            if tgt[n] is not params[n]:
                params[n].copy_(tgt[n].float())
        extra = sorted(have - want)
        log(f"[load] DCP (parallel reader, {nj} chunks) {ci['path']}: {len(want)} tensors ({time.time() - t0:.0f}s); not used: {extra[:6]}")
    elif ci["kind"] == "dcp":
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
    nonfin = [n for n, p in params.items() if not torch.isfinite(p).all()]
    assert not nonfin, ("non-finite weights", nonfin[:4])


# ================================================================================================ sampling (= live_ppm / live_engine, row-wise)
def _greedy_or(z, temp, q):
    return torch.where(temp <= 0, z.argmax(-1), q)


def _topk_mask(z, k):
    """z [N, V] fp32, k long [] (0 = off) -> z with everything below the k-th largest value at -inf (ties kept)."""
    V = z.shape[-1]
    vals = z.sort(-1, descending=True).values
    thr = vals.gather(-1, (k.clamp(1, V) - 1).view(1, 1).expand(z.shape[0], 1))
    return torch.where((k > 0) & (z < thr), torch.full_like(z, NINF), z)


def _topp_mask(z, p):
    pr = torch.softmax(z, -1)
    ps = pr.sort(-1, descending=True).values
    n = ((ps.cumsum(-1) - ps) < p).sum(-1, keepdim=True).clamp_min(1)
    thr = ps.gather(-1, n - 1)
    return torch.where((p < 1.0) & (pr < thr), torch.full_like(z, NINF), z)


def t_sample(z, gen, temp, topk, topp=None):
    """temperature / top-k / top-p from device tensors; temp <= 0 -> argmax. z [N, V] -> [N]."""
    zz = z / temp.clamp_min(1e-4)
    zz = _topk_mask(zz, topk)
    if topp is not None:
        zz = _topp_mask(zz, topp)
    q = torch.multinomial(torch.softmax(zz, -1), 1, generator=gen)[:, 0]
    return _greedy_or(z, temp, q)


def t_word_min_p(z, is_word, p, rel):
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
    zf = z.float()
    zp = zf - fpen * cnt
    pres = (cnt if cnt_rep is None else cnt_rep) > 0
    zp = torch.where(pres & (rep != 1.0), torch.where(zp > 0, zp / rep, zp * rep), zp)
    w = is_word[None]
    l0 = torch.logsumexp(zf.masked_fill(~w, NINF), -1, keepdim=True)
    l1 = torch.logsumexp(zp.masked_fill(~w, NINF), -1, keepdim=True)
    out = torch.where(w, zp + (l0 - l1), zf)
    return torch.where(torch.isfinite(l0) & torch.isfinite(l1), out, zf)


def ctrl_penalty(z, pres, pen):
    return torch.where(pres, torch.where(z > 0, z / pen, z * pen), z)


def presence(dep_win, OFF_, CBS_, valid):
    """dep_win [S, W, 32] global codes (-1 none), valid [S, W] -> [S, 32, CBS] bool."""
    S, W, K = dep_win.shape
    ok = (dep_win >= 0) & valid[:, :, None]
    ix = torch.where(ok, dep_win - OFF_, torch.full_like(dep_win, CBS_)).transpose(1, 2)
    pres = torch.zeros(S, K, CBS_ + 1, dtype=torch.float32, device=dep_win.device)
    pres.scatter_(2, ix, 1.0)
    return pres[..., :CBS_] > 0


WMAX_T, WMAX_A, WMAX_S = 256, 256, 256


class LiveRing:
    """= live_engine.LiveRing per instance row: B's last word pieces with their push index (variable window <= WMAX_T)."""

    def __init__(self, n, V, TE_, dev, excl):
        self.V, self.TE, self.excl = V, TE_, excl
        self.ring = torch.full((n, WMAX_T), -1, dtype=torch.long, device=dev)
        self.pidx = torch.full((n, WMAX_T), -1, dtype=torch.long, device=dev)
        self.ptr = torch.zeros(n, dtype=torch.long, device=dev)
        self.cnt = torch.zeros(n, V, dtype=torch.float32, device=dev)
        self.cnt2 = torch.zeros(n, V, dtype=torch.float32, device=dev)

    def counts(self, win, out=None):
        out = self.cnt if out is None else out
        ok = (self.ring >= 0) & (self.pidx >= self.ptr[:, None] - win)
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


PARAMS = {
    "text_temp":     (float, 0.0, 3.0),    # text temperature (0 = greedy)
    "text_topk":     (int, 0, 256),        # text top-k over the grammar-allowed lane tokens (0 = off)
    "text_topp":     (float, 0.05, 1.0),   # text nucleus (1 = off)
    "min_p":         (float, 0.0, 0.5),    # min-p on the WORD choice (0 = off)
    "min_p_rel":     (int, 0, 1),
    "freq_pen":      (float, 0.0, 3.0),
    "freq_win":      (int, 1, 200),
    "text_rep":      (float, 1.0, 2.0),
    "text_rep_win":  (int, 1, 200),
    "cb0_temp":      (float, 0.0, 3.0),    # speech codebook 0 temperature (0 = greedy)
    "cb0_topk":      (int, 0, 2048),
    "light_temp":    (float, 0.0, 3.0),    # speech codebooks 1-31 (0.6B decoder)
    "light_topk":    (int, 0, 2048),
    "audio_rep_pen": (float, 1.0, 2.0),
    "audio_rep_win": (int, 1, 256),
    "allow_eob":     (int, 0, 1),
    "free_text":     (int, 0, 1),
    "word_greedy":   (int, 0, 1),
    "pose_temp":     (float, 0.0, 3.0),    # pose codebook 0 (4B pose tower) temperature (0 = greedy)
    "pose_topk":     (int, 0, 2048),       # pose cb0 top-k
    "pose_light_temp": (float, 0.0, 3.0),  # pose codebooks 1-31 (0.6B pose decoder) temperature
    "pose_light_topk": (int, 0, 2048),     # pose cb1..31 top-k
    "pose_cfg_on":   (int, 0, 1),          # pose CFG (silent-audio uncond rows; needs an engine built with cfg_rows)
    "pose_cfg_s":    (float, 1.0, 6.0),    # pose CFG strength w: l = l_uncond + w (l_cond - l_uncond); 1 = cond exactly
    "pose_cfg_cb0":  (int, 0, 1),          # (with pose_cfg_on) guide pose cb0
    "pose_cfg_light": (int, 0, 1),         # (with pose_cfg_on) guide pose cb1..31 (uncond rows through the pose decoder too)
    "pose_cfg_speaking_only": (int, 0, 1), # 10-26 (user): with pose_cfg_on, guide only while B is SPEAKING (uncond rows still run every frame)
    "pose_rep_cb0":  (float, 1.0, 2.0),    # 10-26 (user: separate them): CTRL repetition penalty on pose codebook 0 (1 = off)
    "pose_rep_light": (float, 1.0, 2.0),   # ... on pose codebooks 1-31, per codebook (1 = off); old 'pose_rep' = both (POSE_REP_ALIAS)
    "pose_rep_win":  (int, 0, 256),        # its window in frames (conversation only); 0 = every pose frame generated so far
    "adaptive_rep_on":     (int, 0, 1),    # adaptive repetition penalty: per-frame SPEAKING / SILENT penalties replace the static ones
    "speak_window_frames": (int, 1, 250),  # SPEAKING = B emitted a word piece in the last N frames (25 = 2 s)
    "audio_rep_speaking":  (float, 1.0, 2.0),
    "audio_rep_silent":    (float, 1.0, 2.0),
    "text_rep_speaking":   (float, 1.0, 2.0),
    "text_rep_silent":     (float, 1.0, 2.0),
    "pose_rep_cb0_speaking":   (float, 1.0, 2.0),
    "pose_rep_cb0_silent":     (float, 1.0, 2.0),
    "pose_rep_light_speaking": (float, 1.0, 2.0),
    "pose_rep_light_silent":   (float, 1.0, 2.0),
}
# 10-26: the pre-split combined pose knobs (old links / saved settings / infer.py presets) set BOTH new ones. The old key WINS over the
# new ones in the same dict: only pre-split producers emit it, and start() merges DEFAULTS (which hold the new keys) under the request.
POSE_REP_ALIAS = {"pose_rep": ("pose_rep_cb0", "pose_rep_light"), "pose_rep_speaking": ("pose_rep_cb0_speaking", "pose_rep_light_speaking"),
                  "pose_rep_silent": ("pose_rep_cb0_silent", "pose_rep_light_silent")}
ADAPT_KEYS = ("adaptive_rep_on", "speak_window_frames", "audio_rep_speaking", "audio_rep_silent", "text_rep_speaking", "text_rep_silent",
              "pose_rep_cb0_speaking", "pose_rep_cb0_silent", "pose_rep_light_speaking", "pose_rep_light_silent")
DEFAULTS = {"text_temp": 1.0, "text_topk": 0, "text_topp": 1.0, "min_p": 0.0, "min_p_rel": 0, "freq_pen": 0.0, "freq_win": 20,
            "text_rep": 1.0, "text_rep_win": 64, "cb0_temp": 1.0, "cb0_topk": 30, "light_temp": 1.0, "light_topk": 30, "audio_rep_pen": 1.0,
            "audio_rep_win": 64, "allow_eob": 0, "free_text": 0, "word_greedy": 0, "pose_temp": 1.0, "pose_topk": 30, "pose_light_temp": 1.0,
            "pose_light_topk": 30, "pose_cfg_on": 0, "pose_cfg_s": 3.0, "pose_cfg_cb0": 1, "pose_cfg_light": 1, "pose_cfg_speaking_only": 0, "pose_rep_cb0": 1.0, "pose_rep_light": 1.0,
            "pose_rep_win": 0, "adaptive_rep_on": 0, "speak_window_frames": 25, "audio_rep_speaking": 1.0, "audio_rep_silent": 1.0,
            "text_rep_speaking": 1.0, "text_rep_silent": 1.0, "pose_rep_cb0_speaking": 1.0, "pose_rep_cb0_silent": 1.0,
            "pose_rep_light_speaking": 1.0, "pose_rep_light_silent": 1.0}
GROUPS = {k: ("adaptive repetition penalty" if k in ADAPT_KEYS else "pose CFG" if k.startswith("pose_cfg") else "pose" if k.startswith("pose")
              else "audio" if k.startswith(("cb0", "light", "audio")) else "text grammar" if k in ("allow_eob", "free_text", "word_greedy")
              else "text") for k in PARAMS}
HINTS = {
    "text_temp": "text temperature (0 = greedy)", "text_topk": "keep the k most likely lane tokens (0 = off)",
    "text_topp": "nucleus: smallest set reaching this mass (1 = off)",
    "min_p": "min-p on the WORD choice (0 = off); PAD / NEW_WORD timing untouched",
    "min_p_rel": "min-p mode: 0 absolute (p_w >= p, top word kept), 1 relative (p_w >= p x top word)",
    "freq_pen": "frequency penalty: word-piece logit -= strength x count in the window (word mass kept)",
    "freq_win": "frequency penalty window: B's last N word pieces",
    "text_rep": "CTRL repetition penalty on word pieces present in the window (1 = off; word mass kept)",
    "text_rep_win": "text repetition penalty window: B's last N word pieces",
    "cb0_temp": "audio codebook 0 (semantic, 4B speech tower) temperature", "cb0_topk": "audio codebook 0 top-k",
    "light_temp": "audio codebooks 1-31 (0.6B decoder) temperature", "light_topk": "audio codebooks 1-31 top-k (0 = off)",
    "audio_rep_pen": "CTRL repetition penalty per audio codebook on codes used in the window (1 = off)",
    "audio_rep_win": "audio repetition window (conversation frames)",
    "allow_eob": "grammar: allow EOB (<|im_end|>) in B's lane (training: never)",
    "free_text": "grammar off: any lane token after any (off-distribution)",
    "word_greedy": "when B starts / continues a word, take the most likely word piece (timing still sampled)",
    "pose_temp": "pose codebook 0 (4B pose tower) temperature", "pose_topk": "pose codebook 0 top-k",
    "pose_light_temp": "pose codebooks 1-31 (0.6B pose decoder) temperature", "pose_light_topk": "pose codebooks 1-31 top-k",
    "pose_cfg_on": "pose CFG: uncond = the same input with every Mimi code (B's and A's) replaced by digital-silence codes (SAM / duo CFG-3 recipe)",
    "pose_cfg_s": "pose CFG strength w: l = l_uncond + w (l_cond - l_uncond); 1 = cond exactly (released recipe: 3)",
    "pose_cfg_cb0": "with pose CFG on: guide pose codebook 0", "pose_cfg_light": "with pose CFG on: guide pose codebooks 1-31",
    "pose_cfg_speaking_only": "with pose CFG on: guide only while B is SPEAKING (a word piece in its last speak_window_frames frames, the adaptive rep-pen's rule); silent frames use the plain cond logits (the uncond rows still run every frame)",
    "pose_rep_cb0": "pose cb0 rep-pen: CTRL repetition penalty on pose codebook 0 codes used in the window (1 = off; PPM's old pose default was 1.2 on all)",
    "pose_rep_light": "pose light rep-pen: CTRL repetition penalty per pose codebook 1-31 on codes used in the window (1 = off)",
    "pose_rep_win": "pose repetition window (conversation frames); 0 = every pose frame generated so far (PPM's default)",
    "adaptive_rep_on": "on: each frame uses the SPEAKING or SILENT penalties below instead of the static audio / text / pose repetition penalties",
    "speak_window_frames": "SPEAKING = B emitted a word piece (not PAD / NEW_WORD / EOB) in its last N frames (80 ms each; 25 = 2 s)",
    "audio_rep_speaking": "audio repetition penalty while B is SPEAKING (1 = off)", "audio_rep_silent": "audio repetition penalty while B is SILENT (1 = off)",
    "text_rep_speaking": "text repetition penalty while B is SPEAKING (1 = off)", "text_rep_silent": "text repetition penalty while B is SILENT (1 = off)",
    "pose_rep_cb0_speaking": "pose cb0 repetition penalty while B is SPEAKING (1 = off)",
    "pose_rep_cb0_silent": "pose cb0 repetition penalty while B is SILENT (1 = off)",
    "pose_rep_light_speaking": "pose cb1-31 repetition penalty while B is SPEAKING (1 = off)",
    "pose_rep_light_silent": "pose cb1-31 repetition penalty while B is SILENT (1 = off)"}


def params_schema():
    """every knob with group, type, range, default, live, hint (= the live server's GET /api/params_schema rows)."""
    return [{"name": k, "type": t.__name__, "min": lo, "max": hi, "default": DEFAULTS.get(k), "live": True, "group": GROUPS.get(k, "other"),
             "hint": HINTS.get(k, "")} for k, (t, lo, hi) in PARAMS.items()]
FKEYS = [k for k, v in PARAMS.items() if v[0] is float]
IKEYS = [k for k, v in PARAMS.items() if v[0] is int]


def validate(d):
    out, bad = {}, []
    d = dict(d or {})
    for a, (c, l) in POSE_REP_ALIAS.items():                       # the old combined pose knob -> both new ones (POSE_REP_ALIAS)
        if a in d:
            v = d.pop(a)
            d[c] = v
            d[l] = v
    for k, v in d.items():
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


def silence_depth(depth, sil):
    """= infer_pose_2side.silence_depth: every Mimi code of both channels -> the Mimi codes of digital silence, frame by frame (index restarts
    at each contiguous run of coded positions). depth [2, n, 32] global, sil [32, L] local."""
    du = depth.copy()
    for c in range(depth.shape[0]):
        q = np.flatnonzero(depth[c, :, 0] >= 0)
        if not len(q):
            continue
        run0 = np.concatenate([[True], np.diff(q) > 1])
        idx = np.arange(len(q)) - np.maximum.accumulate(np.where(run0, np.arange(len(q)), 0))
        assert idx.max() < sil.shape[1], (idx.max(), sil.shape)
        du[c, q] = sil[:, idx].T + OFF
    return du


# ================================================================================================ fused 0.6B decoder (S rows, D33 slot-0 link, CFG)
class FusedDec:
    """the 0.6B depth decoder for S rows with a static per-frame KV cache, TRAINER-EXACT dtype flow (10-26, main c7: light_exact_ppm's
    module doc; was the pure-bf16 s3fast FusedLight.step_triton path): fp32 input rows / residual stream / RMSNorm / q-k norm / RoPE, bf16
    GEMMs and SDPA operands (= model_ppm.Decoder.forward: fp32 masters under bf16 autocast). = live_ppm_pose.FusedDecS (aloe-posedup-live,
    same kernels). step(x, j): j = 0 (cb1: the 2 rows slot 0, cb0_emb at positions 0, 1), j >= 1 (cb_{j+1}: one row at position j + 1)."""

    def __init__(self, dec, dev, S, max_pos=64, split=None, share=None):
        assert split is None, "split (per-group GEMVs) was never used by the engine; not carried into the exact decoder"
        cfg = dec.model.config
        self.dec, self.dev, self.S = dec, dev, int(S)
        self.split = None
        self.nh, self.kv = cfg.num_attention_heads, cfg.num_key_value_heads
        self.D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.layers = LX.layer_weights(dec) if share is None else share.layers          # share: reuse another instance's weights
        self.nw = dec.model.norm.weight.float().contiguous() if share is None else share.nw
        self.ne = float(dec.model.norm.variance_epsilon)
        self.K = [torch.zeros(self.S, max_pos, self.kv, self.D, dtype=BF, device=dev) for _ in self.layers]
        self.V = [torch.zeros(self.S, max_pos, self.kv, self.D, dtype=BF, device=dev) for _ in self.layers]
        self.cos, self.sin = LX.rope_f32(dec, dev, max_pos) if share is None else (share.cos, share.sin)
        self.W = dec.heads.weight.view(DEPTH - 1, CBS, -1)

    @staticmethod
    def lin1(x, w):
        return LK.gemv(x, w) if x.numel() // x.shape[-1] <= 16 else F.linear(x, w)

    def lin(self, x, w, L=1):
        return self.lin1(x, w)

    def step(self, x, j):
        """x [S, L, Hl] fp32 -> the final-normed hidden of each row's last position, bf16 [S, Hl]."""
        assert x.shape[0] == self.S, (x.shape, self.S)
        return LX.trunk_step(self.layers, self.nw, self.ne, self.K, self.V, self.cos, self.sin, x, j, self.nh, self.kv, self.D, self.lin1,
                             LK.silu_mul_gu)

    def head(self, hl, d):
        return self.lin1(hl, self.W[d - 1]).float()

    def slot0(self, hv, zb=None):
        """= model_ppm.Decoder.inputs slot 0 under autocast: proj(hv) (bf16 Linear) .to(fp32) (+ zb.to(fp32), D33 pose decoder) -> fp32."""
        p = self.dec.proj(hv).float()
        return p if zb is None else p + zb.float()

    def emb(self, cd, d):
        """input row of code cd of codebook d (1..30): the embed_tokens row (the trainer's fp32 table = the bf16 export upcast)."""
        return self.dec.model.embed_tokens(cd + (d - 1) * CBS)[:, None].float()

    def gen(self, hv, c0, gen, temp, topk, zb=None, pres=None, rp=None, cfg_w=None, cfg_on=None):
        """hv [S, Hv] tower states (S = NI rows, or 2 NI = [cond, uncond] with CFG), c0 [NI] local cb0 (already sampled) -> codes [NI, 32] local.
        CFG (S = 2 NI): per codebook l = lu + w (lc - lu) when cfg_on, one code per instance fed to both rows (= infer_pose_2side.DecoderCFG).
        pres [NI, 32, CBS] / rp: CTRL repetition penalty per codebook (speech decoder; = live_engine._audio_live). Sampling = the previous
        FusedDec.gen line for line (same generator calls in the same order)."""
        dec = self.dec
        NI = c0.shape[0]
        S = hv.shape[0]
        two = S == 2 * NI
        assert S == NI or two, (S, NI)
        cc = torch.cat([c0, c0]) if two else c0
        x = torch.cat([self.slot0(hv, zb)[:, None], dec.cb0_emb(cc)[:, None].float()], 1)
        out = [c0]
        for d in range(1, DEPTH):
            zd = self.head(self.step(x, d - 1), d)
            if two:
                zc, zu = zd[:NI], zd[NI:]
                zd = torch.where(cfg_on, zu + cfg_w * (zc - zu), zc)
            if pres is not None:
                zd = ctrl_penalty(zd, pres[:, d], rp)
            cd = t_sample(zd, gen, temp, topk)
            out.append(cd)
            if d < DEPTH - 1:
                x = self.emb(torch.cat([cd, cd]) if two else cd, d)
        return torch.stack(out, 1)

    @torch.no_grad()
    def teacher_logits(self, hv, codes, zb=None):
        """teacher-forced cb1..31 logits [S, 31, CBS] fp32 (codes [S, 32] LOCAL ids of the frame) through the fused path (parity)."""
        x = torch.cat([self.slot0(hv, zb)[:, None], self.dec.cb0_emb(codes[:, 0])[:, None].float()], 1)
        out = []
        for d in range(1, DEPTH):
            out.append(self.head(self.step(x, d - 1), d))
            if d < DEPTH - 1:
                x = self.emb(codes[:, d], d)
        return torch.stack(out, 1)


# ================================================================================================ the engine
class FastPPM:
    """built once per loaded checkpoint (graphs captured once, static buffers sized for p_max + max_s); start(insts, seed, params, f0) per
    session; step(f) per conversation frame (async); result(f) / read-backs after. See the module doc."""

    def __init__(self, m, ldev, cfgs, devs, n_inst=1, cfg_rows=False, max_s=300.0, p_max=2048, mimi_stream=True, front=None, mimi=None,
                 enc=None, silence=None, s2p=None):
        """s2p (pose-s2p): None = follow the model (on iff m.tower_p.s_emb exists), False = off, or an S2PEmb module to use (tests)."""
        self.m, self.ldev, self.devs = m, ldev, devs
        c32, c4, c06 = cfgs
        self.d0, self.dl = devs[0], devs[-1]
        d0, dl = self.d0, self.dl
        self.pose = hasattr(m, "tower_p")
        self.bblink = hasattr(m, "decp_bb")
        self.dp = d0
        dp = self.dp
        self.front = hasattr(m, "vadapter") if front is None else bool(front)
        assert not self.front and not mimi_stream, "vendored engine: no Voxtral front / streaming Mimi (pass mimi_stream=False)"
        self.NI = NI = int(n_inst)
        self.cfg_rows = bool(cfg_rows) and self.pose
        self.R = R = NI * (2 if self.cfg_rows else 1)
        assert 1 <= NI <= MAX_ROWS and R <= 2 * MAX_ROWS, (NI, R)
        assert not self.front or NI == 1, "the Voxtral front serves one instance"
        self.max_s = float(max_s)
        self.n_fr = int(round(self.max_s * FPS)) + 8
        self.lmax = int(p_max) + self.n_fr + CD + 8
        lmax = self.lmax
        self.bb = IP.Stack(list(m.layers), ldev, c32, lmax)
        self.ts = IP.Stack(list(m.tower_s.layers), [dl] * len(m.tower_s.layers), c4, lmax)
        self.bbK = [torch.zeros(R, *k.shape, dtype=BF, device=k.device) for k in self.bb.k]
        self.bbV = [torch.zeros(R, *v.shape, dtype=BF, device=v.device) for v in self.bb.v]
        self.tsK = [torch.zeros(NI, *k.shape, dtype=BF, device=k.device) for k in self.ts.k]
        self.tsV = [torch.zeros(NI, *v.shape, dtype=BF, device=v.device) for v in self.ts.v]
        self._view(0)
        if self.pose:
            self.tp = IP.Stack(list(m.tower_p.layers), [dp] * len(m.tower_p.layers), c4, lmax)
            self.tpK = [torch.zeros(R, *k.shape, dtype=BF, device=k.device) for k in self.tp.k]
            self.tpV = [torch.zeros(R, *v.shape, dtype=BF, device=v.device) for v in self.tp.v]
            self.tp.k, self.tp.v = [k[0] for k in self.tpK], [v[0] for v in self.tpV]
        torch.cuda.empty_cache()
        self.fa = fa3_op._mod().flash_attn_interface
        h = m.head
        self.R_spec = torch.cat([h.spec_out, h.nw_out, h.eob_out], 0)
        self.spec = torch.tensor(list(M.SPEC), device=dl)
        self.V = h.lm_head.weight.shape[0]
        V = self.V
        gs = [IP.Grammar(V, dl, "nw", e) for e in (False, True)]
        self.gb = torch.stack([g.base for g in gs]); self.gp = torch.stack([g.after_pad for g in gs]); self.gn = gs[0].after_nw
        self.is_word = torch.arange(V, device=dl) < TE
        self.gen_t = torch.Generator(device=dl)
        self.gen_a = torch.Generator(device=dl)
        self.gen_p = torch.Generator(device=dp)
        self.dec_s = FusedDec(m.dec_s, dl, NI)
        # pose decoder: cond rows only (= the no-CFG engine's kernels: byte-identical output while pose CFG is off) and, with cfg_rows, the
        # batched [cond | uncond] decoder used only while pose CFG guides cb1..31 (pose_cfg_on and pose_cfg_light)
        self.dec_p = FusedDec(m.dec_p, dp, NI) if self.pose else None
        self.dec_pu = FusedDec(m.dec_p, dp, R, share=self.dec_p) if self.pose and self.cfg_rows else None
        self.ring = LiveRing(NI, V, TE, dl, (PAD, NW, EOB))
        self.arWA = torch.arange(WMAX_A, device=dl)
        self.segs = []
        for i, d in enumerate(ldev):
            if not self.segs or self.segs[-1][0] != d:
                self.segs.append((d, []))
            self.segs[-1][1].append(i)
        assert self.segs[-1][0] == dl and self.segs[0][0] == d0
        self.sdevs = sorted(set(devs) | {d for d, _ in self.segs}, key=lambda d: d.index)
        # per-row positions: T_r = P_r + f (rows NI.. = the uncond copies of rows 0..NI-1)
        self.Pv = {d: torch.zeros(R, dtype=torch.long, device=d) for d in self.sdevs}
        self.Tv = {d: torch.zeros(R, dtype=torch.long, device=d) for d in self.sdevs}
        self.fd = {d: torch.zeros(1, dtype=torch.long, device=d) for d in self.sdevs}
        self.arR = {d: torch.arange(R, device=d) for d in self.sdevs}
        self.arI = {d: torch.arange(NI, device=d) for d in self.sdevs}
        self.rmap = {d: torch.arange(R, device=d) % NI for d in self.sdevs}
        self.rope = {}
        for d in self.sdevs:
            x = torch.zeros(1, lmax, 1, dtype=BF, device=d)
            self.rope[d] = self.bb.rot[d](x, torch.arange(lmax, device=d)[None]) if d in self.bb.rot else None
        self.rope_t = self.ts.rot[dl](torch.zeros(1, lmax, 1, dtype=BF, device=dl), torch.arange(lmax, device=dl)[None])
        if self.pose:
            self.rope_p = self.tp.rot[dp](torch.zeros(1, lmax, 1, dtype=BF, device=dp), torch.arange(lmax, device=dp)[None])
        H = c32.hidden_size
        self.H = H
        # channel B (instances): lane / speech codes on the head GPU, pose codes on the pose GPU
        self.laneB = torch.full((NI, lmax), PAD, dtype=torch.long, device=dl)
        self.depB = torch.full((NI, lmax, DEPTH), -1, dtype=torch.long, device=dl)
        self.tok_now = torch.full((NI,), PAD, dtype=torch.long, device=dl)
        self.tforce = torch.full((NI, lmax), -1, dtype=torch.long, device=dl)               # TEXT FORCING: lane token forced at T (>= 0) or -1
        self.hn_prev = torch.zeros(R, H, dtype=BF, device=dl)
        self.z0 = torch.zeros(NI, CBS, dtype=torch.float32, device=dl)
        self.zt = torch.zeros(NI, V, dtype=torch.float32, device=dl)
        self.probs = torch.zeros(NI, 2, dtype=torch.float32, device=dl)
        self.ptok = torch.zeros(NI, 2, dtype=torch.float32, device=dl)
        self.adapt = torch.ones(NI, 5, dtype=torch.float32, device=dl)      # this frame: SPEAKING, audio / text / pose cb0 / pose light rep-pen
        self.arWS = torch.arange(WMAX_S, device=dl)
        self.hv_s = torch.zeros(NI, c4.hidden_size, dtype=BF, device=dl)
        self.pk = {d: torch.zeros(NI, 1 + DEPTH, dtype=torch.long, device=d) for d in self.sdevs}
        if self.pose:
            self.pdepB = torch.full((NI, lmax, DEPTH), -1, dtype=torch.long, device=dp)
            self.tokp = torch.full((NI,), PAD, dtype=torch.long, device=dp)
            self.hn_p = torch.zeros(R, H, dtype=BF, device=dp)
            self.zp0 = torch.zeros(R, CBS, dtype=torch.float32, device=dp)
            self.hv_p = torch.zeros(R, c4.hidden_size, dtype=BF, device=dp)
            self.zb_p = torch.zeros(R, c06.hidden_size, dtype=BF, device=dp)
            self.adapt_p = torch.ones(NI, 5, dtype=torch.float32, device=dp)
            self.pseen = torch.zeros(NI, DEPTH, CBS, dtype=torch.bool, device=dp)               # pose codes generated so far (pose_rep_win 0)
            self.pcfg_guided = torch.zeros(NI, dtype=torch.long, device=dp)                      # 10-26: frames with the pose CFG mix applied
            self.arWAp = torch.arange(WMAX_A, device=dp)
            self.pkp = {d: torch.zeros(NI, DEPTH, dtype=torch.long, device=d) for d in self.sdevs}
        # channel A (instances) on the first GPU: buffers for every position + the per-instance partner (duo) index
        self.idsA = torch.full((NI, lmax), PAD, dtype=torch.long, device=d0)
        self.depA = torch.full((NI, lmax, DEPTH), -1, dtype=torch.long, device=d0)
        self.presA = torch.zeros(NI, lmax, dtype=torch.float32, device=d0)
        self.presB = torch.ones(NI, lmax, dtype=torch.float32, device=d0)
        self.a_src = torch.full((NI,), -1, dtype=torch.long, device=d0)
        self.sil_t = torch.zeros(self.n_fr, DEPTH, dtype=torch.long, device=d0)
        self.sil = None
        if self.cfg_rows:
            self.sil = silence if silence is not None else IP.Mimi(dl).encode(np.zeros((p_max + self.n_fr) * 1920, np.float32))[:, :p_max + self.n_fr]
            assert self.sil.shape[0] == DEPTH and self.sil.min() >= 0, self.sil.shape
            self.sil_t.copy_(torch.as_tensor(self.sil[:, :self.n_fr].T + OFF))
        self.xo = [torch.zeros(1, R, H, dtype=BF, device=d) for d, _ in self.segs]
        self.xi = [None] + [torch.zeros(1, R, H, dtype=BF, device=d) for d, _ in self.segs[1:]]
        self.off_t = torch.as_tensor(OFF, device=dl)
        self.OFF0 = torch.as_tensor(OFF, device=d0)
        self.offp = torch.as_tensor(OFF, device=dp)
        self.pf = {d: torch.zeros(len(FKEYS), dtype=torch.float32, device=d) for d in self.sdevs}
        self.pi = {d: torch.zeros(len(IKEYS), dtype=torch.long, device=d) for d in self.sdevs}
        self.params = {}
        self.set_params(DEFAULTS)
        self.a_vec0 = torch.zeros(R, H, dtype=BF, device=d0)
        if self.front:
            with torch.cuda.device(d0):
                self.enc = enc if enc is not None else VS.load_audio_tower(d0)
                self.vx = VoxStreamFast(self.enc, 1, self.n_fr, d0, self.fa, FO)
            self.S0 = VS.CTX24
            self.abuf = torch.zeros(1, self.S0 + self.n_fr * SPF + VS.LOOK24 + 2 * SPF, device=d0)
            self.arWin = torch.arange(VS.CTX24 + SPF + VS.LOOK24, device=d0)
            self.s_t = torch.exp(m.vgain.theta.float()).to(BF)
            self.vrec = torch.zeros(self.n_fr, H, dtype=BF, device=d0)
            self.utab = self._zero_table() if self.cfg_rows else None
        self.mimi_stream = bool(mimi_stream)
        if self.mimi_stream:
            with torch.cuda.device(d0):
                mm = mimi if mimi is not None else VS.MimiStream(d0).mimi
                self.mdec = MimiDecStream(mm, NI, d0)
            self.cbB = torch.zeros(NI, DEPTH, dtype=torch.long, device=d0)
            self.ybuf = torch.zeros(NI, SPF, device=d0)
        # pose-s2p (module doc): the term, its mode and buffers (flag off: none of these exist)
        se = getattr(m.tower_p, "s_emb", None) if (s2p is None and self.pose) else (None if s2p is False else s2p)
        assert se is None or self.pose, "pose-s2p needs the pose branch"
        self.s_emb, self.s2p = se, (se.mode if se is not None else None)
        if self.s2p:
            nc = DEPTH if self.s2p == "full" else 1
            self.scB_p = torch.full((NI, nc), -1, dtype=torch.long, device=dp)          # B's speech codes at T (global ids) on the pose GPU
            self.sc_used = torch.full((R, nc), -1, dtype=torch.long, device=dp)         # what s_emb read this frame, all R rows (tests)
            if self.s2p == "cb0":
                self.c0_s = torch.zeros(NI, dtype=torch.long, device=dl)               # the speech cb0 draw (head graph -> light graph)
                self.pres_s = torch.zeros(NI, DEPTH, CBS, dtype=torch.bool, device=dl)
                self.sc0_l = torch.full((NI, 1), -1, dtype=torch.long, device=dl)
        self.cs_side = torch.cuda.Stream(dl)                                                   # dl -> pose GPU copies (never blocks dl)
        self.ev_text = torch.cuda.Event()
        self.graphs = {}
        self.P = None
        self.f0 = 0

    def _view(self, r):
        """point the eager Stacks' KV views at row r (prefill writes through them)."""
        self.bb.k, self.bb.v = [k[r] for k in self.bbK], [v[r] for v in self.bbV]
        if r < self.NI:
            self.ts.k, self.ts.v = [k[r] for k in self.tsK], [v[r] for v in self.tsV]
        if self.pose and hasattr(self, "tpK"):
            self.tp.k, self.tp.v = [k[r] for k in self.tpK], [v[r] for v in self.tpV]

    @torch.no_grad()
    def _zero_table(self):
        """the uncond rows' A input for the Voxtral front: Voxtral(digital zero) -> adapter x s per frame (= live_ppm digital_zero)."""
        d0, nfr = self.d0, self.n_fr
        tab = torch.zeros(nfr, self.H, dtype=BF, device=d0)
        with torch.cuda.device(d0):
            wav = torch.zeros(1, nfr * SPF, dtype=torch.float32, device=d0)
            vx = VS.VoxStream(self.enc, 1, nfr + 8, d0, self.fa)
            f = 0
            while f < nfr:
                n = min(125, nfr - f)
                lat = vx.encode(VS.window(wav, f, n).to(d0), n)
                tab[f:f + n] = (self.m.vadapter(lat)[0] * self.s_t).to(BF)
                f += n
            del vx, wav
        return tab

    # ---------------------------------------------------------------------------------------------- params
    def prm(self, k, d=None):
        d = self.dl if d is None else d
        return self.pf[d][FKEYS.index(k)] if k in FKEYS else self.pi[d][IKEYS.index(k)]

    def set_params(self, d):
        clean, bad = validate(d)
        new = dict(self.params, **clean)
        fv = torch.tensor([float(new.get(k, 0.0)) for k in FKEYS], dtype=torch.float32)
        iv = torch.tensor([int(new.get(k, 0)) for k in IKEYS], dtype=torch.long)
        for dv in self.sdevs:
            self.pf[dv].copy_(fv); self.pi[dv].copy_(iv)
        self.params = new
        if new.get("pose_cfg_on", 0) and not self.cfg_rows:
            bad.append("pose_cfg_on needs an engine built with cfg_rows (ignored: no uncond rows)")
        return clean, bad

    # ---------------------------------------------------------------------------------------------- pieces
    def _layer_step(self, L, h, cos, sin, kc, vc, sl):
        """one Qwen3 layer for n rows at ONE position each (h [1, n, H]); per-row KV lengths sl [n] int32 (= live_ppm._layer_step2)."""
        n = h.shape[1]
        at, ln1, ln2, mlp = L.self_attn, L.input_layernorm, L.post_attention_layernorm, L.mlp
        x = FO.rms(h, ln1.weight, float(ln1.variance_epsilon))
        q, k = FO.qk_prep(at.q_proj(x), at.k_proj(x), at.q_norm.weight, at.k_norm.weight, float(at.q_norm.variance_epsilon), cos, sin)
        hd = at.head_dim
        v = at.v_proj(x).view(n, 1, -1, hd)
        a = self.fa.flash_attn_with_kvcache(q.view(n, 1, -1, hd), kc, vc, k=k.view(n, 1, -1, hd), v=v, cache_seqlens=sl,
                                            softmax_scale=at.scaling, causal=True, num_splits=1)
        h, x = FO.add_rms(h, at.o_proj(a.reshape(1, n, -1)), ln2.weight, float(ln2.variance_epsilon))
        return h + mlp.down_proj(FO.silu_mul(mlp.gate_proj(x), mlp.up_proj(x)))

    def text_logits(self, hn):
        x = self.m.head.proj(hn)
        z = (x @ self.m.head.lm_head.weight.t()).float()
        z[:, self.spec] = (x @ self.R_spec.t()).float()
        return z

    def _speaking(self, T):
        """adaptive rep-pen: -> bool [NI] = B's lane holds a word piece at a conversation position in [T - N, T - 1]."""
        NI = self.NI
        idx = T[:, None] - WMAX_S + self.arWS[None]
        lw = self.laneB.gather(1, idx.clamp_min(0))
        ok = (idx >= self.Pv[self.dl][:NI, None]) & (idx >= T[:, None] - self.prm("speak_window_frames"))
        return ((lw < TE) & (lw != PAD) & (lw != NW) & (lw != EOB) & ok).any(-1)

    def _rep_eff(self, spk, static, speaking, silent):
        """the repetition penalty in effect this frame per instance: adaptive_rep_on ? (SPEAKING ? speaking : silent) : static."""
        return torch.where(self.prm("adaptive_rep_on") > 0, torch.where(spk, self.prm(speaking), self.prm(silent)), self.prm(static))

    def _text(self):
        """= live_ppm._text per instance row (no text CFG): grammar -> penalties -> temperature -> word min-p -> top-k -> top-p -> draw.
        First: the frame's SPEAKING flag and the repetition penalties in effect (self.adapt: read by the speech / pose graphs)."""
        dl = self.dl
        T = self.Tv[dl][:self.NI]
        spk = self._speaking(T)
        self.adapt[:, 0].copy_(spk.float())
        self.adapt[:, 1].copy_(self._rep_eff(spk, "audio_rep_pen", "audio_rep_speaking", "audio_rep_silent"))
        self.adapt[:, 2].copy_(self._rep_eff(spk, "text_rep", "text_rep_speaking", "text_rep_silent"))
        self.adapt[:, 3].copy_(self._rep_eff(spk, "pose_rep_cb0", "pose_rep_cb0_speaking", "pose_rep_cb0_silent"))
        self.adapt[:, 4].copy_(self._rep_eff(spk, "pose_rep_light", "pose_rep_light_speaking", "pose_rep_light_silent"))
        z = self.text_logits(self.hn_prev[:self.NI])
        self.zt.copy_(z)
        pr = torch.softmax(z, -1)
        self.probs[:, 0].copy_(pr[:, PAD]); self.probs[:, 1].copy_(pr[:, NW])
        tt, tk, tp = self.prm("text_temp"), self.prm("text_topk"), self.prm("text_topp")
        mp, mrel = self.prm("min_p"), self.prm("min_p_rel")
        fp, fw, rp = self.prm("freq_pen"), self.prm("freq_win"), self.adapt[:, 2:3]
        cnt = self.ring.counts(fw)
        cnt2 = self.ring.counts(self.prm("text_rep_win"), self.ring.cnt2)
        prev = self.laneB.gather(1, (T - 1)[:, None])[:, 0]
        ae = self.prm("allow_eob").clamp(0, 1)
        gm = torch.where((prev == NW)[:, None], self.gn[None],
                         torch.where(((prev == PAD) | (prev == EOB))[:, None], self.gp.index_select(0, ae.view(1)), self.gb.index_select(0, ae.view(1))))
        gm = gm | (self.prm("free_text") > 0)
        z = z.masked_fill(~gm, NINF)
        z = t_penalties(z, cnt, fp, rp, self.is_word, cnt2)
        zt_ = t_word_min_p(z / tt.clamp_min(1e-4), self.is_word, mp, mrel)
        zt_ = _topp_mask(_topk_mask(zt_, tk), tp)
        q = torch.multinomial(torch.softmax(zt_, -1), 1, generator=self.gen_t)[:, 0]
        q = _greedy_or(z, tt, q)
        qw = z.masked_fill(~self.is_word[None], NINF).argmax(-1)
        q = torch.where((self.prm("word_greedy") > 0) & (q < TE), qw, q)
        qf = self.tforce.gather(1, T[:, None])[:, 0]                                           # TEXT FORCING (module doc; all -1 = off)
        q = torch.where(qf >= 0, qf, q)
        self.laneB.scatter_(1, T[:, None], q[:, None])
        self.tok_now.copy_(q)
        self.ring.push(q)
        self.ptok[:, 0].copy_(pr.gather(1, q[:, None])[:, 0])
        self.ptok[:, 1].copy_(torch.softmax(zt_, -1).gather(1, q[:, None])[:, 0])

    def _presence_live(self):
        """codes of the last audio_rep_win frames (<= WMAX_A), conversation only -> [NI, 32, CBS]."""
        dl = self.dl
        T = self.Tv[dl][:self.NI]
        P = self.Pv[dl][:self.NI]
        idx = T[:, None] - WMAX_A + self.arWA[None]
        win = self.depB[self.arI[dl][:, None], idx.clamp_min(0)]
        return presence(win, self.off_t, CBS, (idx >= (P + CD)[:, None]) & (idx >= T[:, None] - self.prm("audio_rep_win")))

    def _speech(self, full):
        m, dl, NI = self.m, self.dl, self.NI
        tw = m.tower_s
        T = self.Tv[dl][:NI]
        ar = self.arI[dl]
        sl = T.to(torch.int32)
        tok = self.tok_now
        dprev = self.depB[ar, T - 1]
        e = tw.text_emb(tok) + tw.code_emb(dprev).to(BF) + tw.cond(self.hn_prev[:NI]).to(BF)
        h = e[None]
        cos, sin = self.rope_t[0].index_select(1, T), self.rope_t[1].index_select(1, T)
        for i, L in enumerate(self.ts.L):
            h = self._layer_step(L, h, cos, sin, self.tsK[i], self.tsV[i], sl)
        hv = FO.rms(h[0], tw.norm.weight, float(tw.norm.variance_epsilon))
        self.hv_s.copy_(hv)
        z0 = (hv @ tw.cb0_head.weight.t()).float()
        self.z0.copy_(z0)
        if full:
            pres = self._presence_live()
            rp = self.adapt[:, 1:2]                                                            # = audio_rep_pen unless adaptive_rep_on
            c0 = t_sample(ctrl_penalty(z0, pres[:, 0], rp), self.gen_a, self.prm("cb0_temp"), self.prm("cb0_topk"))
            codes = self.dec_s.gen(hv, c0, self.gen_a, self.prm("light_temp"), self.prm("light_topk"), pres=pres, rp=rp)
            self.depB[ar, T] = codes + self.off_t
        self.pk[dl][:, 0].copy_(self.tok_now)
        self.pk[dl][:, 1:].copy_(self.depB[ar, T])

    def _speech_head(self):
        """pose-s2p 'cb0', full frames: the first half of _speech(True) op for op (tower step, cb0 logits, presence, cb0 draw); the draw
        and its presence go to buffers for _speech_light; sc0_l = the cb0 global id the pose branch reads."""
        m, dl, NI = self.m, self.dl, self.NI
        tw = m.tower_s
        T = self.Tv[dl][:NI]
        ar = self.arI[dl]
        sl = T.to(torch.int32)
        tok = self.tok_now
        dprev = self.depB[ar, T - 1]
        e = tw.text_emb(tok) + tw.code_emb(dprev).to(BF) + tw.cond(self.hn_prev[:NI]).to(BF)
        h = e[None]
        cos, sin = self.rope_t[0].index_select(1, T), self.rope_t[1].index_select(1, T)
        for i, L in enumerate(self.ts.L):
            h = self._layer_step(L, h, cos, sin, self.tsK[i], self.tsV[i], sl)
        hv = FO.rms(h[0], tw.norm.weight, float(tw.norm.variance_epsilon))
        self.hv_s.copy_(hv)
        z0 = (hv @ tw.cb0_head.weight.t()).float()
        self.z0.copy_(z0)
        pres = self._presence_live()
        rp = self.adapt[:, 1:2]
        c0 = t_sample(ctrl_penalty(z0, pres[:, 0], rp), self.gen_a, self.prm("cb0_temp"), self.prm("cb0_topk"))
        self.c0_s.copy_(c0)
        self.pres_s.copy_(pres)
        self.sc0_l.copy_((c0 + self.off_t[0])[:, None])

    def _speech_light(self):
        """pose-s2p 'cb0', full frames: the second half of _speech(True) (fused 0.6B speech decoder cb1..31 from the head's draw)."""
        dl, NI = self.dl, self.NI
        T = self.Tv[dl][:NI]
        ar = self.arI[dl]
        codes = self.dec_s.gen(self.hv_s, self.c0_s, self.gen_a, self.prm("light_temp"), self.prm("light_topk"), pres=self.pres_s,
                               rp=self.adapt[:, 1:2])
        self.depB[ar, T] = codes + self.off_t
        self.pk[dl][:, 0].copy_(self.tok_now)
        self.pk[dl][:, 1:].copy_(self.depB[ar, T])

    def _s2p_codes(self):
        """pose-s2p: the codes s_emb reads for all R rows: instance rows = scB_p (B's speech codes at T), uncond rows = the Mimi codes of
        digital silence where B has codes at T (= _seg's uncond rule), -1 elsewhere."""
        dp, NI = self.dp, self.NI
        sc = self.scB_p
        if self.cfg_rows:
            TI = self.Tv[dp][:NI]
            fr = (TI - self.Pv[dp][:NI] - CD).clamp(0, self.n_fr - 1)
            st = self.sil_t.index_select(0, fr)[:, :sc.shape[1]]
            sc = torch.cat([sc, torch.where(sc[:, :1] >= 0, st, torch.full_like(sc, -1))], 0)
        self.sc_used.copy_(sc)
        return sc

    def _pose(self, full, cfgdec=False):
        """pose branch on the pose GPU for all R rows (uncond rows: own hn, the instance's lane + pose history)."""
        m, dp, NI = self.m, self.dp, self.NI
        tw = m.tower_p
        T = self.Tv[dp]
        sl = T.to(torch.int32)
        rm = self.rmap[dp]
        tok = self.tokp.index_select(0, rm)
        pprev = self.pdepB[rm, T - 1]
        e = tw.text_emb(tok) + tw.code_emb(pprev).to(BF) + tw.cond(self.hn_p).to(BF)
        if self.s2p:                                                                       # pose-s2p (flag off: never runs)
            e = e + self.s_emb(self._s2p_codes()).to(BF)
        h = e[None]
        cos, sin = self.rope_p[0].index_select(1, T), self.rope_p[1].index_select(1, T)
        for i, L in enumerate(self.tp.L):
            h = self._layer_step(L, h, cos, sin, self.tpK[i], self.tpV[i], sl)
        hv = FO.rms(h[0], tw.norm.weight, float(tw.norm.variance_epsilon))
        self.hv_p.copy_(hv)
        z0 = (hv @ tw.cb0_head.weight.t()).float()
        self.zp0.copy_(z0)
        TI = T[:NI]
        ar = self.arI[dp]
        if full:
            zb = (m.decp_bb(self.hn_p) * float(m.decp_bb_alpha)) if self.bblink else None
            if zb is not None:
                self.zb_p.copy_(zb)
            w = self.prm("pose_cfg_s", dp)
            on = self.prm("pose_cfg_on", dp) > 0
            gate = (self.prm("pose_cfg_speaking_only", dp) == 0) | (self.adapt_p[:, 0] > 0)     # [NI] 10-26 speaking-only (off: all True)
            on0, onl = (on & (self.prm("pose_cfg_cb0", dp) > 0) & gate)[:, None], (on & (self.prm("pose_cfg_light", dp) > 0) & gate)[:, None]
            self.pcfg_guided.add_((on0 | onl)[:, 0].long() if self.cfg_rows else torch.zeros_like(self.pcfg_guided))
            zc = z0[:NI]
            zmix = torch.where(on0, z0[NI:] + w * (zc - z0[NI:]), zc) if self.cfg_rows else zc
            P = self.Pv[dp][:NI]
            idx = TI[:, None] - WMAX_A + self.arWAp[None]
            win = self.pdepB[ar[:, None], idx.clamp_min(0)]
            wv = self.prm("pose_rep_win", dp)
            pres = torch.where(wv > 0, presence(win, self.offp, CBS, (idx >= (P + CD)[:, None]) & (idx >= TI[:, None] - wv)), self.pseen)
            prp = self.adapt_p[:, 3:4]                                                         # = pose_rep_cb0 unless adaptive_rep_on
            prl = self.adapt_p[:, 4:5]                                                         # = pose_rep_light unless adaptive_rep_on
            p0 = t_sample(ctrl_penalty(zmix, pres[:, 0], prp), self.gen_p, self.prm("pose_temp", dp), self.prm("pose_topk", dp))
            if cfgdec:
                codes = self.dec_pu.gen(hv, p0, self.gen_p, self.prm("pose_light_temp", dp), self.prm("pose_light_topk", dp), zb=zb,
                                        pres=pres, rp=prl, cfg_w=w, cfg_on=onl)
            else:
                codes = self.dec_p.gen(hv[:NI], p0, self.gen_p, self.prm("pose_light_temp", dp), self.prm("pose_light_topk", dp),
                                       zb=None if zb is None else zb[:NI], pres=pres, rp=prl)
            self.pdepB[ar, TI] = codes + self.offp
            self.pseen.scatter_(2, codes[:, :, None], True)
        self.pkp[dp].copy_(self.pdepB[ar, TI])

    def _seg(self, si):
        d, idx = self.segs[si]
        NI, R = self.NI, self.R
        T = self.Tv[d]
        sl = T.to(torch.int32)
        if si == 0:
            pk = self.pk[d]
            TI = T[:NI]
            ar = self.arI[d]
            lane, cod = pk[:, 0], pk[:, 1:]
            src = self.a_src.clamp_min(0)
            part = self.a_src >= 0
            ida = torch.where(part, lane.index_select(0, src), self.idsA[ar, TI])
            dpa = torch.where(part[:, None], cod.index_select(0, src), self.depA[ar, TI])
            self.idsA[ar, TI] = ida
            self.depA[ar, TI] = dpa
            ids = torch.stack([ida, lane])                                                     # [2, NI]
            dep = torch.stack([dpa, cod])                                                      # [2, NI, 32]
            pres = torch.stack([self.presA[ar, TI], self.presB[ar, TI]])                       # [2, NI]
            pdep = self.pkp[d] if self.pose else None
            if self.cfg_rows:                                                                  # uncond rows: Mimi -> silence codes
                fr = (TI - self.Pv[d][:NI] - CD).clamp(0, self.n_fr - 1)
                st = self.sil_t.index_select(0, fr)                                            # [NI, 32]
                du = torch.where(dep[..., :1] >= 0, st[None].expand(2, -1, -1), torch.full_like(dep, -1))
                ids, dep, pres = torch.cat([ids, ids], 1), torch.cat([dep, du], 1), torch.cat([pres, pres], 1)
                pdep = torch.cat([pdep, pdep], 0)
            h = self.m.inp(ids, dep, pres, pdep)
            if self.front:
                h = h + self.a_vec0
            h = h[None]
        else:
            h = self.xi[si]
        cos, sin = self.rope[d][0].index_select(1, T), self.rope[d][1].index_select(1, T)
        for i in idx:
            h = self._layer_step(self.bb.L[i], h, cos, sin, self.bbK[i], self.bbV[i], sl)
        if si == len(self.segs) - 1:
            n = self.m.head.norm
            self.hn_prev.copy_(FO.rms(h[0], n.weight, float(n.variance_epsilon)))
        else:
            self.xo[si].copy_(h)

    def _a_rows(self, lat, f):
        self.a_vec0[0:1].copy_((self.m.vadapter(lat).reshape(1, -1) * self.s_t).to(BF))
        if self.cfg_rows:
            self.a_vec0[1:2].copy_(self.utab.index_select(0, f.view(-1)))

    def _front_eager(self, f):
        idx = self.S0 + f * SPF - VS.CTX24 + self.arWin
        lat = self.vx.encode(self.abuf.index_select(1, idx), 1)
        self._a_rows(lat, torch.tensor([f], device=self.d0))

    def _front_graphable(self):
        f = self.fd[self.d0]
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
    def set_f(self, f):
        for d in self.sdevs:
            with torch.cuda.device(d):
                self.fd[d].fill_(int(f))
                torch.add(self.Pv[d], int(f), out=self.Tv[d])

    @torch.no_grad()
    def capture(self, f=40, P=32):
        """capture every graph once (warm-ups / capture write KV slots of the capture position and scratch state; every slot is rewritten
        before it is read in a session, and start() resets every stream state + re-seeds the generators)."""
        t0 = time.time()
        for d in self.sdevs:
            self.Pv[d].fill_(P)
        self.set_f(f)
        self.laneB.fill_(PAD); self.depB.fill_(-1)
        if self.pose:
            self.pdepB.fill_(-1)

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
        cap("text", self.dl, self._text, (self.gen_t,))
        cap("speech", self.dl, lambda: self._speech(False))
        cap("speech_full", self.dl, lambda: self._speech(True), (self.gen_a,))
        if self.s2p == "cb0":                                                              # pose-s2p cb0: speech split at the cb0 draw
            cap("speech_head_full", self.dl, self._speech_head, (self.gen_a,))
            cap("speech_light_full", self.dl, self._speech_light, (self.gen_a,))
        if self.pose:
            cap("pose", self.dp, lambda: self._pose(False))
            cap("pose_full", self.dp, lambda: self._pose(True), (self.gen_p,))
            if self.cfg_rows:
                cap("pose_full_cfg", self.dp, lambda: self._pose(True, True), (self.gen_p,))
        for si, (d, _) in enumerate(self.segs):
            cap(f"seg{si}", d, lambda si=si: self._seg(si))
        if self.front:
            vx = self.vx
            sv = (vx.mel_prev.clone(), vx.c1_prev.clone(), vx.kd.clone(), self.a_vec0.clone())
            vx.kd.fill_(1)
            cap("front", self.d0, self._front_graphable)
            vx.mel_prev.copy_(sv[0]); vx.c1_prev.copy_(sv[1]); vx.kd.copy_(sv[2]); self.a_vec0.copy_(sv[3])
        if self.mimi_stream:
            st = [t.clone() for t in self._mdec_state()]
            cap("mdec", self.d0, self._mdec)
            for t, s_ in zip(self._mdec_state(), st):
                t.copy_(s_)
        self.ring.reset()
        if self.pose:
            self.pseen.zero_()
        for d in self.sdevs:
            torch.cuda.synchronize(d)
        log(f"captured {len(self.graphs)} graphs ({', '.join(self.graphs)}) in {time.time() - t0:.1f}s; NI {self.NI} R {self.R}; " +
            ", ".join(f"cuda:{d.index} {torch.cuda.memory_allocated(d) / 2 ** 30:.1f} GiB" for d in self.sdevs))

    # ---------------------------------------------------------------------------------------------- session
    @torch.no_grad()
    def _prefill_row(self, r, ids, depth, present, pdepth, tower_s):
        """eager teacher-forced positions [0, n) of row r (the trainer's fused layers, FA3 varlen) -> hn [n, H] (head GPU)."""
        d0, dl, dp = self.d0, self.dl, self.dp
        self._view(r)
        t = lambda x, d: None if x is None else torch.as_tensor(np.ascontiguousarray(x), device=d)  # noqa: E731
        e = self.m.inp(t(ids, d0), t(depth, d0), t(present, d0), t(pdepth, d0) if self.pose else None)[None]
        h = self.bb.run(e, 0)
        n = self.m.head.norm
        with torch.cuda.device(dl):
            hn = FO.rms(h[0].to(dl), n.weight, float(n.variance_epsilon))
            hprev = torch.cat([torch.zeros_like(hn[:1]), hn[:-1]], 0)
            if tower_s:
                tw = self.m.tower_s
                dprev = np.concatenate([np.full((1, DEPTH), -1, np.int64), depth[1, :-1]], 0)
                et = tw.text_emb(t(ids[1], dl)) + tw.code_emb(t(dprev, dl)).to(BF) + tw.cond(hprev).to(BF)
                self.ts.run(et[None], 0)
        if self.pose:
            with torch.cuda.device(dp):
                tw = self.m.tower_p
                pprev = np.concatenate([np.full((1, DEPTH), -1, np.int64), pdepth[:-1]], 0)
                hp = hprev.to(dp)
                ep = tw.text_emb(t(ids[1], dp)) + tw.code_emb(t(pprev, dp)).to(BF) + tw.cond(hp).to(BF)
                if self.s2p:                                                               # pose-s2p: B's speech codes at the same positions
                    ep = ep + self.s_emb(t(depth[1], dp)).to(BF)
                self.tp.run(ep[None], 0)
        return hn

    @torch.no_grad()
    def start(self, insts, seed, params=None, f0=0):
        """insts: NI dicts {P, ids [2, L], depth [2, L, 32], present [2, L], pdepth [L, 32] (pose models), a_mode, partner (a_mode 'partner')}
        with L >= P + f0: positions [0, P + f0) are prefilled teacher-forced (system prompt + the first f0 conversation frames); generation
        starts at frame f0. a_mode 'file': channel A of positions >= P + f0 comes from the same arrays (ids[0] / depth[0] / present[0]);
        'empty': PAD / -1 / present 0; 'partner': instance `partner`'s B text + Mimi at each step (present 1); 'voxtral': the live front."""
        assert len(insts) == self.NI, (len(insts), self.NI)
        dl, d0 = self.dl, self.d0
        lmax = self.lmax
        self.f0 = int(f0)
        self.laneB.fill_(PAD); self.depB.fill_(-1)
        self.idsA.fill_(PAD); self.depA.fill_(-1); self.presA.zero_(); self.presB.fill_(1.0)
        a_src = []
        Ps = []
        self.prefill_hn = []
        if self.pose:
            self.pdepB.fill_(-1)
            self.pseen.zero_()
            self.pcfg_guided.zero_()
        for r, it in enumerate(insts):
            P = int(it["P"])
            Np = P + self.f0
            Ps.append(P)
            ids, depth, present = (np.asarray(it[k]) for k in ("ids", "depth", "present"))
            assert ids.shape[0] == 2 and ids.shape[1] >= Np and depth.shape[1] >= Np and present.shape[1] >= Np, (ids.shape, Np)
            assert P >= 5 and P + self.n_fr + CD <= lmax, ("prefix too long for the static buffers", P, lmax)
            pdepth = np.asarray(it["pdepth"]) if self.pose else None
            if self.pose:
                assert pdepth is not None and pdepth.shape[0] >= Np, "pose model: pdepth needed for the prefill"
            mode = it.get("a_mode", "empty")
            assert mode in A_MODES, mode
            assert mode != "voxtral" or self.front, "a_mode voxtral needs the Voxtral front (vadapter checkpoint)"
            self.laneB[r, :Np].copy_(torch.as_tensor(ids[1, :Np], device=dl))
            self.depB[r, :Np].copy_(torch.as_tensor(depth[1, :Np], device=dl))
            if self.pose:
                self.pdepB[r, :Np].copy_(torch.as_tensor(pdepth[:Np], device=self.dp))
            self.idsA[r, :Np].copy_(torch.as_tensor(ids[0, :Np], device=d0))
            self.depA[r, :Np].copy_(torch.as_tensor(depth[0, :Np], device=d0))
            self.presA[r, :Np].copy_(torch.as_tensor(present[0, :Np], device=d0))
            self.presB[r, :Np].copy_(torch.as_tensor(present[1, :Np], device=d0))
            if mode == "file":
                n = min(ids.shape[1], lmax)
                self.idsA[r, Np:n].copy_(torch.as_tensor(ids[0, Np:n], device=d0))
                self.depA[r, Np:n].copy_(torch.as_tensor(depth[0, Np:n], device=d0))
                self.presA[r, Np:n].copy_(torch.as_tensor(present[0, Np:n], device=d0))
            elif mode == "partner":
                self.presA[r, Np:].fill_(1.0)
            a_src.append(int(it["partner"]) if mode == "partner" else -1)
            self.tforce[r].fill_(-1)
            if it.get("text_force") is not None:                                               # TEXT FORCING from the generation start on
                tf = np.asarray(it["text_force"], np.int64)
                n = min(len(tf), lmax)
                assert n > Np and (tf[Np:n] >= 0).all() and (tf[Np:n] < self.V).all(), ("text_force: lane ids over [P + f0, len)", n, Np)
                self.tforce[r, Np:n].copy_(torch.as_tensor(tf[Np:n], device=dl))
            hn = self._prefill_row(r, ids[:, :Np], depth[:, :Np], present[:, :Np], None if pdepth is None else pdepth[:Np], True)
            self.hn_prev[r].copy_(hn[-1])
            self.prefill_hn.append(hn)
        if self.cfg_rows:                                                                      # uncond rows: silence codes everywhere
            for r, it in enumerate(insts):
                P = int(it["P"]); Np = P + self.f0
                depth = np.asarray(it["depth"])[:, :Np]
                for c in range(2):                                                             # conversation run must start at P + CD
                    q = np.flatnonzero(depth[c, P:, 0] >= 0)
                    assert not len(q) or (q[0] == CD and (np.diff(q) == 1).all()), ("conversation codes not one run from P + CD", r, c)
                du = silence_depth(depth, self.sil)
                hn = self._prefill_row(self.NI + r, np.asarray(it["ids"])[:, :Np], du, np.asarray(it["present"])[:, :Np],
                                       np.asarray(it["pdepth"])[:Np], False)
                self.hn_prev[self.NI + r].copy_(hn[-1])
        self._view(0)
        self.P = Ps
        Pv = torch.tensor(Ps * (self.R // self.NI), dtype=torch.long)
        for d in self.sdevs:
            self.Pv[d].copy_(Pv)
        self.a_src.copy_(torch.tensor(a_src, dtype=torch.long))
        for r, s in enumerate(a_src):
            assert s < 0 or (0 <= s < self.NI and s != r and a_src[s] == r), ("partner instances must pair up", a_src)
        with torch.cuda.device(dl):
            self.ring.reset()
        with torch.cuda.device(d0):
            self.a_vec0.zero_()
            if self.front:
                self.vx.reset(); self.abuf.zero_()
            if self.mimi_stream:
                self.mdec.reset(); self.ybuf.zero_()
        self.gen_t.manual_seed(int(seed) + 1)
        self.gen_a.manual_seed(int(seed))
        self.gen_p.manual_seed(int(seed) + 2)
        self.params = {}
        self.set_params(dict(DEFAULTS, **(params or {})))
        for d in self.sdevs:
            torch.cuda.synchronize(d)

    def write_a(self, i0, x, peak=0.995):
        """A samples [i0, i0 + len(x)) of the conversation (24 kHz float32) -> abuf (Voxtral front)."""
        if len(x):
            x = np.clip(np.ascontiguousarray(x, np.float32), -peak, peak)
            n = min(len(x), self.abuf.shape[1] - self.S0 - i0)
            if n > 0:
                self.abuf[0, self.S0 + i0:self.S0 + i0 + n].copy_(torch.from_numpy(x[:n]))

    @torch.no_grad()
    def step(self, f, graphs=True, force=None, cold_pad=True):
        """conversation frame f (positions T_r = P_r + f). Async (no host sync). Frame 0 (generation right after the system prompt): B's lane
        is forced to PAD (cold_pad; = the live engines' rule). force (parity only): dict with 'lane' [NI] and/or 'codes' [NI, 32] global and/or
        'pose' [NI, 32] global: teacher-forces B at T."""
        self.set_f(f)
        d0, dl, dp = self.d0, self.dl, self.dp
        use = graphs and bool(self.graphs)
        full = f >= CD
        TI = None
        if self.front:
            with torch.cuda.device(d0):
                if f == self.f0 or not use:
                    assert self.vx.k == f - self.f0, ("Voxtral front out of step", self.vx.k, f)
                    self._front_eager(f)
                else:
                    self.graphs["front"].replay()
                    self.vx.k += 1
        with torch.cuda.device(dl):
            if use:
                self.graphs["text"].replay()
            else:
                self._text()
            lane_f = None
            if force is not None and force.get("lane") is not None:
                lane_f = torch.as_tensor(np.asarray(force["lane"], np.int64), device=dl)
            elif f == 0 and cold_pad:
                lane_f = torch.full((self.NI,), PAD, dtype=torch.long, device=dl)
            if lane_f is not None:
                TI = self.Tv[dl][:self.NI]
                self.laneB[self.arI[dl], TI] = lane_f
                self.tok_now.copy_(lane_f)
                if f == 0 and force is None:
                    self.ring.reset()
            self.ev_text.record(torch.cuda.current_stream(dl))
        if self.s2p:                                                                           # pose-s2p: text -> speech -> pose (module doc)
            self._step_s2p(f, use, full, force)
        else:                                                                                  # flag off: today's order (pose || speech)
            if self.pose:
                if dp != dl:
                    with torch.cuda.device(dl), torch.cuda.stream(self.cs_side):
                        self.cs_side.wait_event(self.ev_text)
                        self.tokp.copy_(self.tok_now)
                        self.hn_p.copy_(self.hn_prev)
                        self.adapt_p.copy_(self.adapt)
                else:
                    self.tokp.copy_(self.tok_now); self.hn_p.copy_(self.hn_prev); self.adapt_p.copy_(self.adapt)
                with torch.cuda.device(dp):
                    cfgdec = self.cfg_rows and bool(self.params.get("pose_cfg_on")) and bool(self.params.get("pose_cfg_light"))
                    if use:
                        self.graphs[("pose_full_cfg" if cfgdec else "pose_full") if full else "pose"].replay()
                    else:
                        self._pose(full, cfgdec)
                    if force is not None and force.get("pose") is not None:
                        pf = torch.as_tensor(np.asarray(force["pose"], np.int64), device=dp)
                        self.pdepB[self.arI[dp], self.Tv[dp][:self.NI]] = pf
                        self.pkp[dp].copy_(pf)
            with torch.cuda.device(dl):
                if use:
                    self.graphs["speech_full" if full else "speech"].replay()
                else:
                    self._speech(full)
                if force is not None and force.get("codes") is not None:
                    cf = torch.as_tensor(np.asarray(force["codes"], np.int64), device=dl)
                    self.depB[self.arI[dl], self.Tv[dl][:self.NI]] = cf
                    self.pk[dl][:, 1:].copy_(cf)
        for si, (d, _) in enumerate(self.segs):
            with torch.cuda.device(d):
                if si == 0:
                    if d != dl:
                        self.pk[d].copy_(self.pk[dl])
                    if self.pose and d != dp:
                        self.pkp[d].copy_(self.pkp[dp])
                else:
                    self.xi[si].copy_(self.xo[si - 1])
                if use:
                    self.graphs[f"seg{si}"].replay()
                else:
                    self._seg(si)
        if self.front:
            with torch.cuda.device(d0):
                self.vrec[f - self.f0].copy_(self.a_vec0[0])
        if self.mimi_stream and full:
            with torch.cuda.device(d0):
                self.cbB.copy_(self.pk[d0][:, 1:] if d0 in self.pk else self.pk[dl][:, 1:])
                if use:
                    self.graphs["mdec"].replay()
                else:
                    self._mdec()
        return full

    def _step_s2p(self, f, use, full, force):
        """pose-s2p frame body after the text draw: speech (full: whole branch; cb0: tower + cb0 draw only) -> forced speech codes -> B's
        speech codes at T to the pose GPU -> pose (+ forced pose) -> (cb0: the speech light decoder, concurrent with pose) -> forced codes."""
        dl, dp, NI = self.dl, self.dp, self.NI
        if dp != dl:                                                                           # pose inputs: = today's copies
            with torch.cuda.device(dl), torch.cuda.stream(self.cs_side):
                self.cs_side.wait_event(self.ev_text)
                self.tokp.copy_(self.tok_now)
                self.hn_p.copy_(self.hn_prev)
                self.adapt_p.copy_(self.adapt)
        else:
            self.tokp.copy_(self.tok_now); self.hn_p.copy_(self.hn_prev); self.adapt_p.copy_(self.adapt)
        cf = None
        if force is not None and force.get("codes") is not None:
            cf = torch.as_tensor(np.asarray(force["codes"], np.int64), device=dl)
        split = self.s2p == "cb0" and full
        with torch.cuda.device(dl):
            if split:
                if use:
                    self.graphs["speech_head_full"].replay()
                else:
                    self._speech_head()
                if cf is not None:                                                             # forced cb0 BEFORE pose reads it
                    self.sc0_l.copy_(cf[:, :1])
                self.scB_p.copy_(self.sc0_l)
            else:
                if use:
                    self.graphs["speech_full" if full else "speech"].replay()
                else:
                    self._speech(full)
                if cf is not None:                                                             # forced codes BEFORE pose reads them
                    self.depB[self.arI[dl], self.Tv[dl][:NI]] = cf
                    self.pk[dl][:, 1:].copy_(cf)
                self.scB_p.copy_(self.pk[dl][:, 1:1 + self.scB_p.shape[1]])                    # dl -> dp (the pose stream waits for it)
        with torch.cuda.device(dp):
            cfgdec = self.cfg_rows and bool(self.params.get("pose_cfg_on")) and bool(self.params.get("pose_cfg_light"))
            if use:
                self.graphs[("pose_full_cfg" if cfgdec else "pose_full") if full else "pose"].replay()
            else:
                self._pose(full, cfgdec)
            if force is not None and force.get("pose") is not None:
                pf = torch.as_tensor(np.asarray(force["pose"], np.int64), device=dp)
                self.pdepB[self.arI[dp], self.Tv[dp][:NI]] = pf
                self.pkp[dp].copy_(pf)
        if split:
            with torch.cuda.device(dl):
                if use:
                    self.graphs["speech_light_full"].replay()
                else:
                    self._speech_light()
                if cf is not None:
                    self.depB[self.arI[dl], self.Tv[dl][:NI]] = cf
                    self.pk[dl][:, 1:].copy_(cf)

    def result(self, f, dec):
        """-> (B tokens [NI], P(PAD) [NI], P(NEW_WORD) [NI], audio [NI, 1920] or None). Syncs."""
        torch.cuda.synchronize(self.dl)
        toks = self.tok_now.tolist()
        pr = self.probs.tolist()
        y = None
        if dec and self.mimi_stream:
            torch.cuda.synchronize(self.d0)
            y = self.ybuf.cpu().numpy().copy()
        return toks, [p[0] for p in pr], [p[1] for p in pr], y

    def readback(self, r, n_end):
        """instance r's B lane [n_end], speech codes [n_end, 32], pose codes [n_end, 32] (global; positions 0..n_end-1) and A lane / codes."""
        for d in self.sdevs:
            torch.cuda.synchronize(d)
        out = {"lane_b": self.laneB[r, :n_end].cpu().numpy().copy(), "dep_b": self.depB[r, :n_end].cpu().numpy().copy(),
               "lane_a": self.idsA[r, :n_end].cpu().numpy().copy(), "dep_a": self.depA[r, :n_end].cpu().numpy().copy()}
        if self.pose:
            out["pdep"] = self.pdepB[r, :n_end].cpu().numpy().copy()
        return out
