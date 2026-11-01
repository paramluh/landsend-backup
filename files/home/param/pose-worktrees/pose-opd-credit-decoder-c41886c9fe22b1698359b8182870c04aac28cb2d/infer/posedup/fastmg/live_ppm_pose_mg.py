#!/usr/bin/env python3
"""MULTI-GPU LIVE ENGINE (fastmg, 10-25): the live pose-duplex engine aloe-posedup-live/live_ppm_pose.py (imported, never modified: every
sampling / text / speech / pose-branch method, PARAMS, DEFAULTS, profiles, hints are ITS code) re-laid out over G GPUs for latency:

DEVICES   dS = devs[0]: trunk input embedding (inp), text head, speech tower + fused 0.6B speech decoder (the speech branch).
          dP = devs[1]: pose tower + fused 0.6B pose decoder + D33 link (the pose branch, alone on its GPU: it no longer queues behind
                        the Voxtral front, and pose CFG's 2-row tower no longer shares a GPU with the trunk's first segment).
          dF = devs[2] (G >= 3, else dP): Voxtral front + adapter x s, streaming Mimi decoder (on a side stream: it overlaps the trunk
                        instead of trailing it).
          TRUNK (64 Qwen3-32B layers) TENSOR-PARALLEL over all G GPUs (G in {2, 4, 8}: the 8 kv heads split evenly): q / k / v heads and
                        gate / up columns column-parallel; o_proj / down_proj row-parallel with a ONE-SHOT all-reduce (each rank's fp32
                        partial stored straight into every rank's receive buffer over NVLink, flags, fixed rank-order fp32 sum, ONE bf16
                        rounding = the unsharded projection's single rounding of an fp32-accumulated dot product) fused with the residual
                        add + RMSNorm (fastmg/tpk.py). The residual stream is replicated (bitwise identical on every rank), so dS (text
                        head) and dP (pose cond / D33 link) read hn[T] locally.
SYNC      every cross-GPU dependency is device-side (tpk: peer stores + release flags, consumer spin with acquire): per frame one CUDA graph
          per GPU per stage, no host-side cross-device events. Hand-offs: lane token dS -> dP (pose tower input), pose codes dP -> dS and
          a_vec dF -> dS (trunk input), B codes dS -> dF (Mimi decoder).
MATHS     = the live engine's, op for op, except the trunk's row-parallel sums (fp32 partial sums in rank order instead of one GEMM's
          internal order: the same fp32-accumulated dot product, bf16-rounded once) and its GEMVs (fastmg tpk.gemv, fp32 accumulation, in
          place of cuBLAS). Attention: FA3 kvcache per head shard (bitwise = unsharded FA3 per head at num_splits 1, measured); attn_splits
          (engine arg) > 1 = flash-decoding for the trunk (faster at long context; another fp32 summation order).
          Sampling, RNG streams, grammar, CFG rows, pose CFG, adaptive knobs: the live engine's code itself.
PREFILL   eager tensor-parallel (cuBLAS shard GEMMs, FA3 varlen per head shard, fp32 partials summed in rank order)."""
import os
import sys
import time

LIVE = "/data/home/amartyavarma/ppm/aloe-posedup-live"
FM = os.path.dirname(os.path.abspath(__file__))
if LIVE not in sys.path:
    sys.path.insert(0, LIVE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import live_ppm_pose as LP  # noqa: E402   (the live engine module, read-only)
sys.path.insert(0, FM)
import tpk  # noqa: E402

M, FO, IP, LK, VS = LP.M, LP.FO, LP.IP, LP.LK, LP.VS
fa3_op = LP.fa3_op
BF = LP.BF
CD, DEPTH, CBS, FPS = LP.CD, LP.DEPTH, LP.CBS, LP.FPS
PAD, NW, EOB, TE = LP.PAD, LP.NW, LP.EOB, LP.TE
OFF, SPF = LP.OFF, LP.SPF
log = LP.log
# re-exports the server uses
PARAMS, DEFAULTS, POSE_CFG_KEYS, POSE_KEYS, ADAPT_KEYS, PADAD_KEYS = LP.PARAMS, LP.DEFAULTS, LP.POSE_CFG_KEYS, LP.POSE_KEYS, LP.ADAPT_KEYS, LP.PADAD_KEYS
UNCOND_SRC, GROUPS, HINTS, PROFILE, validate, nopersona_defaults = LP.UNCOND_SRC, LP.GROUPS, LP.HINTS, LP.PROFILE, LP.validate, LP.nopersona_defaults
HF_TOK, HF_PKP, HF_AVEC, HF_CODES = 0, 1, 2, 3                  # mailbox slots (on the consumer's GPU)


def roles(devs):
    assert len(devs) >= 2, devs
    return devs[0], devs[1], (devs[2] if len(devs) >= 3 else devs[1])


# ================================================================================================ model
def build_model(ck, devs, adapter=None, s_value=None, tp=None):
    """-> (model, shards, cfgs, info). Trunk layers loaded whole onto the TP GPUs (round robin), then re-sharded layer by layer (shards[r] = rank r's tensors);
    everything else placed per DEVICES. s (= exp(theta)) chosen exactly as live_ppm_pose.build_model."""
    import math
    ci = LP.ckpt_info(ck)
    c32, c4, c06 = M.cfg32(ci["layers"]), M.cfg4b(ci["tower_layers"]), M.cfg06(ci["dec_layers"])
    pose, bb = bool(ci.get("pose")), bool(ci.get("decp_bb"))
    assert pose, "fastmg engine: pose checkpoints only (the pose-duplex demo)"
    tpd = list(devs[:tp]) if tp else list(devs)
    G = len(tpd)
    assert c32.num_key_value_heads % G == 0 and c32.num_attention_heads % G == 0 and c32.intermediate_size % (G * 128) == 0, G
    dS, dP, dF = roles(devs)
    torch.set_default_dtype(BF)
    with torch.device("meta"):
        m = M.PPM(c32, c4, c06, ci["proj_r"], ci["tower_vocab"], vadapter=LP.VADAPT, nopose=not pose, decp_bb=bb)
    torch.set_default_dtype(torch.float32)
    m.inp.to_empty(device=dS)
    for i, L in enumerate(m.layers):                             # whole layers spread over the TP GPUs (DCP loads straight to GPU),
        L.to_empty(device=tpd[i % G])                            # then re-sharded layer by layer (flat memory)
    for mod in (m.head, m.tower_s, m.dec_s):
        mod.to_empty(device=dS)
    m.dec_s.model.rotary_emb = type(m.dec_s.model.rotary_emb)(config=c06, device=dS)
    for mod in (m.tower_p, m.dec_p) + ((m.decp_bb,) if bb else ()):
        mod.to_empty(device=dP)
    m.dec_p.model.rotary_emb = type(m.dec_p.model.rotary_emb)(config=c06, device=dP)
    if bb:
        m.decp_bb_alpha = float(ci["decp_bb_alpha"])
    m.vadapter.to_empty(device=dF)
    m.vgain = M.VGain().to(dF)
    t0 = time.time()
    LP.load_weights(m, ci, adapter)
    log(f"[mg build] weights loaded in {time.time() - t0:.0f}s")
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    th = float(m.vgain.theta.float().reshape(-1)[0])
    s_ck = math.exp(th)
    mpath = LP.run_metrics_for(ci) if adapter is None else None
    s_met = LP.s_from_metrics(ci["step"], mpath) if (ci["kind"] == "dcp" or ci["meta"].get("run_tag")) and adapter is None else None
    th_meta = (ci.get("meta") or {}).get("vgain_theta_fp32") if ci["kind"] == "local" and adapter is None else None
    if s_value is not None:
        s_use, src = float(s_value), "given"
    elif th_meta is not None and abs(float(th_meta) - th) <= 0.04:
        s_use, src = math.exp(float(th_meta)), f"exp(meta.json vgain_theta_fp32 {float(th_meta):.10g})"
    elif s_met is not None and abs(math.log(s_met) - th) <= 0.04:
        s_use, src = s_met, f"trainer fp32 s at step {ci['step']} ({mpath})"
    else:
        s_use, src = s_ck, "exp(checkpoint theta)"
    with torch.no_grad():
        m.vgain.theta.fill_(math.log(s_use))
    t0 = time.time()
    shards = shard_trunk(m, tpd, c32)
    log(f"[mg build] trunk sharded over {G} GPUs in {time.time() - t0:.0f}s")
    info = {"ck": ci["path"], "step": ci["step"], "kind": ci["kind"], "s": s_use, "s_src": src, "s_ckpt_theta": s_ck, "s_metrics": s_met, "metrics": mpath,
            "run_tag": ((ci.get("side") or {}).get("cfg") or (ci.get("meta") or {}).get("cfg") or {}).get("run_tag"),
            "total_steps": ((ci.get("side") or {}).get("cfg") or (ci.get("meta") or {}).get("cfg") or {}).get("steps"),
            "adapter": adapter or "checkpoint", "ldev": [str(d) for d in tpd], "tp": G,
            "layers_per_dev": {str(d): f"1/{G} of each of {ci['layers']}" for d in tpd},
            "roles": {"speech+head+inp": str(dS), "pose": str(dP), "front+mimi": str(dF)},
            "pose": pose, "decp_bb": bb, "decp_bb_alpha": ci.get("decp_bb_alpha")}
    log(f"[mg build] {ci['path']} step {ci['step']}: s = {s_use:.8g} ({src}); TP {G} over {[str(d) for d in tpd]}; roles {info['roles']}")
    return m, shards, (c32, c4, c06), info


@torch.no_grad()
def shard_trunk(m, tpd, c32):
    G = len(tpd)
    D, H = c32.head_dim, c32.hidden_size
    nq, nkv, Ig = c32.num_attention_heads // G, c32.num_key_value_heads // G, c32.intermediate_size // G
    shards = [dict(dev=d, nq=nq, nkv=nkv, I=Ig, L=[]) for d in tpd]
    for L in m.layers:
        Ly = L.layer
        at, mlp = Ly.self_attn, Ly.mlp
        assert at.q_proj.bias is None and at.o_proj.bias is None and at.head_dim == D
        q, k, v, o = at.q_proj.weight, at.k_proj.weight, at.v_proj.weight, at.o_proj.weight
        g_, u_, dn = mlp.gate_proj.weight, mlp.up_proj.weight, mlp.down_proj.weight
        src = q.device
        for r, d in enumerate(tpd):
            sq, sk = slice(r * nq * D, (r + 1) * nq * D), slice(r * nkv * D, (r + 1) * nkv * D)
            si = slice(r * Ig, (r + 1) * Ig)
            with torch.cuda.device(src):
                shards[r]["L"].append(dict(
                    ln1=Ly.input_layernorm.weight.to(d), e1=float(Ly.input_layernorm.variance_epsilon),
                    wqkv=torch.cat([q[sq], k[sk], v[sk]], 0).contiguous().to(d),
                    qn=at.q_norm.weight.to(d), kn=at.k_norm.weight.to(d), eqk=float(at.q_norm.variance_epsilon), sc=float(at.scaling),
                    wo=o[:, sq].contiguous().to(d),
                    ln2=Ly.post_attention_layernorm.weight.to(d), e2=float(Ly.post_attention_layernorm.variance_epsilon),
                    wgu=torch.cat([g_[si], u_[si]], 0).contiguous().to(d),
                    wd=dn[:, si].contiguous().to(d)))
        for p in Ly.parameters():                                # free the whole layer
            p.data = torch.empty(0, dtype=p.dtype, device=p.device)
        torch.cuda.empty_cache()
    for r, d in enumerate(tpd):
        shards[r]["nw"] = m.head.norm.weight.to(d)
        shards[r]["ne"] = float(m.head.norm.variance_epsilon)
        shards[r]["rot"] = M.Rot(c32, d)
    return shards


def mm_f32(a, w):
    """a [n, K] bf16 @ w[N, K]^T -> fp32 [n, N] (fp32 accumulation, no bf16 rounding of the partial)."""
    return torch.mm(a.float(), w.float().t())


# ================================================================================================ the engine
class LivePPM(LP.LivePPM):
    """= live_ppm_pose.LivePPM's API (start / set_params / write_a / step / result / capture, last_* fields), multi-GPU layout (module doc).
    Constructed from build_model's (m, shards, cfgs): LivePPM(m, shards, cfgs, devs, ...)."""

    def __init__(self, m, shards, cfgs, devs, max_s=300.0, p_max=2000, light_fused=True, mimi=None, enc=None, pose_cfg_rows=None,
                 attn_splits=1, tower_splits=1):
        self.m, self.devs, self.shards = m, list(devs), shards
        c32, c4, c06 = cfgs
        self.c32 = c32
        self.dS, self.dP, self.dF = roles(devs)
        dS, dP, dF = self.dS, self.dP, self.dF
        self.dl, self.d0 = dS, dF                                    # live-engine names: dl = head / speech GPU, d0 = front / Mimi GPU
        self.tpd = [s["dev"] for s in shards]
        self.G = len(self.tpd)
        assert dS == self.tpd[0] and dP in self.tpd and dF in self.tpd, "the branch GPUs must be TP ranks (local hn)"
        self.rP = self.tpd.index(dP)
        self.max_s = float(max_s)
        self.n_fr = int(round(self.max_s * FPS)) + 8
        self.lmax = int(p_max) + self.n_fr + CD + 8
        self.pose = hasattr(m, "tower_p")
        self.bblink = hasattr(m, "decp_bb")
        assert self.pose
        if pose_cfg_rows is None:
            pose_cfg_rows = os.environ.get("PPMLIVE_POSE_CFG_ROWS", "1") != "0"
        self.pcfg = bool(pose_cfg_rows)
        self.NR = 3 if self.pcfg else 2
        self.attn_splits, self.tower_splits = int(attn_splits), int(tower_splits)
        NR, G = self.NR, self.G
        self._init_trunk()
        H = self.H
        # ---- speech tower (dS) and pose tower (dP): the live engine's Stack + batch-2 KV
        self.ts = IP.Stack(list(m.tower_s.layers), [dS] * len(m.tower_s.layers), c4, self.lmax)
        self.tsK = [torch.zeros(2, *k.shape, dtype=BF, device=k.device) for k in self.ts.k]
        self.tsV = [torch.zeros(2, *v.shape, dtype=BF, device=v.device) for v in self.ts.v]
        self.ts.k, self.ts.v = [k[0] for k in self.tsK], [v[0] for v in self.tsV]
        self.tp = IP.Stack(list(m.tower_p.layers), [dP] * len(m.tower_p.layers), c4, self.lmax)
        if self.pcfg:
            self.tpK = [torch.zeros(2, *k.shape, dtype=BF, device=k.device) for k in self.tp.k]
            self.tpV = [torch.zeros(2, *v.shape, dtype=BF, device=v.device) for v in self.tp.v]
            self.tp.k, self.tp.v = [k[0] for k in self.tpK], [v[0] for v in self.tpV]
        else:
            self.tpK = [k[None] for k in self.tp.k]
            self.tpV = [v[None] for v in self.tp.v]
        torch.cuda.empty_cache()
        self.fa = fa3_op._mod().flash_attn_interface
        self.TE, self.PAD, self.NW, self.EOB, self.CD = TE, PAD, NW, EOB, CD
        # ---- text head + speech (dS): = live engine
        h = m.head
        self.R = torch.cat([h.spec_out, h.nw_out, h.eob_out], 0)
        self.spec = torch.tensor(list(M.SPEC), device=dS)
        self.V = h.lm_head.weight.shape[0]
        V = self.V
        gs = [IP.Grammar(V, dS, "nw", e) for e in (False, True)]
        self.gb = torch.stack([g.base for g in gs]); self.gp = torch.stack([g.after_pad for g in gs]); self.gn = gs[0].after_nw
        self.is_word = torch.arange(V, device=dS) < TE
        self.is_pad = torch.arange(V, device=dS) == PAD
        self.n_in = min(int(m.tower_s.text_emb.weight.shape[0]), int(m.inp.emb[1].weight.shape[0]), V)
        self.in_vocab = torch.arange(V, device=dS) < self.n_in
        self.gen_t = torch.Generator(device=dS)
        self.gen_a = torch.Generator(device=dS)
        self.gen_c = torch.Generator(device=dS)
        self.light = LP.FusedDec(m.dec_s, dS) if light_fused else None
        self.ring = LP.LiveRing(V, TE, dS, (PAD, NW, EOB))
        self.arWA = torch.arange(LP.WMAX_A, device=dS)
        self.arWS = torch.arange(LP.WMAX_S, device=dS)
        self.fstate = torch.zeros(6, dtype=torch.float32, device=dS)
        self.adapt = self.fstate[0:3]
        self.padad = self.fstate[3:6]
        self.lastspk = torch.zeros(1, dtype=torch.long, device=dS)
        self.rope_t = self.ts.rot[dS](torch.zeros(1, self.lmax, 1, dtype=BF, device=dS), torch.arange(self.lmax, device=dS)[None])
        self.laneB = torch.full((self.lmax,), PAD, dtype=torch.long, device=dS)
        self.depB = torch.full((self.lmax, DEPTH), -1, dtype=torch.long, device=dS)
        self.hn_prev = torch.zeros(2, H, dtype=BF, device=dS)
        self.hn_u = torch.zeros(1, H, dtype=BF, device=dS) if self.pcfg else None
        self.z0 = torch.zeros(2, CBS, dtype=torch.float32, device=dS)
        self.zt = torch.zeros(2, V, dtype=torch.float32, device=dS)
        self.probs = torch.zeros(2, dtype=torch.float32, device=dS)
        self.ptok = torch.zeros(2, dtype=torch.float32, device=dS)
        self.cfgd = torch.zeros(4, dtype=torch.float32, device=dS)
        self.Pd = torch.zeros(1, dtype=torch.long, device=dS)
        self.pk = {dS: torch.zeros(1 + DEPTH, dtype=torch.long, device=dS)}
        # trunk input (dS): channel A = PAD / no Mimi / present 0 + a_vec rows (pushed by dF), B = lane + codes (+ pose codes pushed by dP)
        self.pres = torch.tensor([[0.0], [1.0]], dtype=torch.float32, device=dS)
        self.idsA = torch.full((1, 1), PAD, dtype=torch.long, device=dS)
        self.depA = torch.full((1, 1, DEPTH), -1, dtype=torch.long, device=dS)
        self.a_vec_s = torch.zeros(NR, H, dtype=BF, device=dS)
        self.pkp_s = torch.full((DEPTH,), -1, dtype=torch.long, device=dS)
        self.h0 = torch.zeros(NR, H, dtype=BF, device=dS)
        self.off_t = torch.as_tensor(OFF, device=dS)
        self.pf = torch.zeros(len(LP.FKEYS), dtype=torch.float32, device=dS)
        self.pi = torch.zeros(len(LP.IKEYS), dtype=torch.long, device=dS)
        self.params = {}
        self.set_params(DEFAULTS)
        # ---- front + Mimi (dF)
        with torch.cuda.device(dF):
            self.enc = enc if enc is not None else VS.load_audio_tower(dF)
            self.vx = LP.VoxStreamFast(self.enc, 1, self.n_fr, dF, self.fa, FO)
            mm = mimi if mimi is not None else VS.MimiStream(dF).mimi
            self.mimi = mm
            self.mdec = LP.MimiDecStream(mm, 1, dF)
        self.S0 = VS.CTX24
        self.abuf = torch.zeros(1, self.S0 + self.n_fr * SPF + VS.LOOK24 + 2 * SPF, device=dF)
        self.arWin = torch.arange(VS.CTX24 + SPF + VS.LOOK24, device=dF)
        self.Pd0 = torch.zeros(1, dtype=torch.long, device=dF)
        self.a_vec0 = torch.zeros(NR, H, dtype=BF, device=dF)
        self.s_t = torch.exp(m.vgain.theta.float()).to(BF)
        self.usrc = torch.zeros(1, dtype=torch.long, device=dF)
        self.utab = self._uncond_tables()
        self.cbB = torch.zeros(1, DEPTH, dtype=torch.long, device=dF)
        self.ybuf = torch.zeros(1, SPF, device=dF)
        self.vrec = torch.zeros(self.n_fr, H, dtype=BF, device=dF)
        self.sideF = torch.cuda.Stream(dF)
        # ---- pose (dP)
        self.pdepB = torch.full((self.lmax, DEPTH), -1, dtype=torch.long, device=dP)
        self.tokp = torch.full((1,), PAD, dtype=torch.long, device=dP)
        PR = 2 if self.pcfg else 1
        self.hn_p = torch.zeros(PR, H, dtype=BF, device=dP)
        self.hv_p = torch.zeros(PR, c4.hidden_size, dtype=BF, device=dP)
        self.zp0 = torch.zeros(PR, CBS, dtype=torch.float32, device=dP)
        self.pkp = torch.full((DEPTH,), -1, dtype=torch.long, device=dP)
        self.offp = torch.as_tensor(OFF, device=dP)
        self.gen_p = torch.Generator(device=dP)
        self.pf0 = torch.zeros(3, dtype=torch.float32, device=dP)
        self.pi0 = torch.zeros(5, dtype=torch.long, device=dP)
        self.dec_p = LP.FusedDec(m.dec_p, dP)
        self.rope_p = self.tp.rot[dP](torch.zeros(1, self.lmax, 1, dtype=BF, device=dP), torch.arange(self.lmax, device=dP)[None])
        if self.pcfg:
            self.dec_pu = LP.FusedDecS(m.dec_p, dP, 2, share=self.dec_p)
            self.Tp2 = self.Tv[dP][0:2]
            self.sil = LP.silence_codes(int(p_max) + self.n_fr, dS)
            assert self.sil.shape[0] == DEPTH and self.sil.min() >= 0, self.sil.shape
            self.sil_t = torch.as_tensor(self.sil[:, :self.n_fr].T + OFF, device=dS)        # trunk input of the pose-uncond row (dS)
            self.utab_u = self.utab[UNCOND_SRC.index("digital_zero")]
        self.set_params({})
        # ---- hand-off tables [dst ptr, flag ptr] (device tensors: their CONTENT switches between warm (local scratch) and live)
        self.scratch = {d: torch.zeros(1 << 16, dtype=torch.int64, device=d) for d in self.devs}
        self.ho = {}
        self.tok_s = torch.full((1,), PAD, dtype=torch.long, device=dS)
        self.codes_s = torch.full((DEPTH,), -1, dtype=torch.long, device=dS)
        self._handoff("tok", dS, self.tok_s, dP, self.tokp, HF_TOK)
        self._handoff("pkp", dP, self.pkp, dS, self.pkp_s, HF_PKP)
        self._handoff("avec", dF, self.a_vec0, dS, self.a_vec_s, HF_AVEC)
        self._handoff("codes", dS, self.codes_s, dF, self.cbB.view(-1), HF_CODES)
        self.last_codes = [-1] * DEPTH
        self.last_pose = [-1] * DEPTH
        self.graphs = {}
        self.P = None
        self.last_ptok = [0.0, 0.0]
        self.last_cfgd = None
        self.last_adapt = [0.0, 1.0, 1.0]
        self.last_padad = [0.0, 0.0, 0.0]
        self.uncond_src = UNCOND_SRC[0]
        self.errs_checked = 0
        log(f"[mg engine] G {G} rows {NR} lmax {self.lmax}; " + ", ".join(f"{d} {torch.cuda.memory_allocated(d) / 2 ** 30:.1f} GiB" for d in self.devs))

    def _init_trunk(self):
        """peer access, all-reduce state, TP KV caches / work buffers / RoPE tables, mailboxes, per-GPU frame state."""
        c32, shards, NR, G = self.c32, self.shards, self.NR, self.G
        tpk.enable_peer_access(self.devs)
        # ---- trunk shards: KV caches [NR, lmax, nkv, D] per layer per rank, work buffers, all-reduce state
        H, D = c32.hidden_size, c32.head_dim
        self.H = H
        nl = len(shards[0]["L"])
        self.nsite = 1 + 2 * nl
        self.ars = [tpk.AllReduce(r, G, d, H, NR, self.nsite) for r, d in enumerate(self.tpd)]
        tpk.AllReduce.link(self.ars)
        for r, (s, d) in enumerate(zip(shards, self.tpd)):
            nq, nkv, Ig = s["nq"], s["nkv"], s["I"]
            s["K"] = [torch.zeros(NR, self.lmax, nkv, D, dtype=BF, device=d) for _ in range(nl)]
            s["V"] = [torch.zeros(NR, self.lmax, nkv, D, dtype=BF, device=d) for _ in range(nl)]
            s["res"] = torch.zeros(NR, H, dtype=BF, device=d)
            s["x"] = torch.zeros(NR, H, dtype=BF, device=d)
            s["x2"] = torch.zeros(NR, H, dtype=BF, device=d)
            s["qkv"] = torch.zeros(NR, (nq + 2 * nkv) * D, dtype=BF, device=d)
            s["q"] = torch.zeros(NR, nq, D, dtype=BF, device=d)
            s["gu"] = torch.zeros(NR, 2 * Ig, dtype=BF, device=d)
            s["act"] = torch.zeros(NR, Ig, dtype=BF, device=d)
            x = torch.zeros(1, self.lmax, 1, dtype=BF, device=d)
            cs, sn = s["rot"](x, torch.arange(self.lmax, device=d)[None])
            s["cos"], s["sin"] = cs[0].contiguous(), sn[0].contiguous()      # [lmax, D]
        self.mb = {d: tpk.Mailbox(d) for d in self.devs}
        # ---- per-GPU frame state: Tv = [T x 8 (Td / T2 / T3 / Tp2 views), ep] (one H2D copy per GPU per frame)
        self.Tv = {d: torch.zeros(9, dtype=torch.long, device=d) for d in self.devs}
        self.Tv_h = {d: [torch.zeros(9, dtype=torch.long).pin_memory() for _ in range(2)] for d in self.devs}
        self.ev_T = {d: torch.cuda.Event() for d in self.devs}
        for r, d in enumerate(self.tpd):
            self.ars[r].ep = self.Tv[d][8:9]
        self.Td = {d: self.Tv[d][0:1] for d in self.devs}
        self.T2 = {d: self.Tv[d][0:2] for d in self.devs}
        self.T3 = {d: self.Tv[d][0:NR] for d in self.devs}
        self.ep = {d: self.Tv[d][8:9] for d in self.devs}
        self.serial = 0

    # ---------------------------------------------------------------------------------------------- hand-offs
    def _handoff(self, name, dsrc, src, ddst, dst, slot):
        live = torch.tensor([dst.data_ptr(), self.mb[ddst].fptr(slot)], dtype=torch.int64)
        warm = torch.tensor([self.scratch[dsrc].data_ptr(), self.scratch[dsrc].data_ptr() + (1 << 15) * 8], dtype=torch.int64)
        self.ho[name] = dict(dsrc=dsrc, src=src, ddst=ddst, dst=dst, slot=slot, tab=torch.zeros(2, dtype=torch.int64, device=dsrc),
                             live=live, warm=warm, n=dst.numel())

    def _push(self, name):
        h = self.ho[name]
        assert h["src"].numel() == h["n"] and h["src"].dtype == h["dst"].dtype
        tpk.push(h["src"], h["tab"], self.ep[h["dsrc"]])

    def _push_tok(self):
        """B's lane token of T as committed (after the cold-start PAD / forcing) -> dP."""
        self.tok_s.copy_(self.laneB.index_select(0, self.Td[self.dS]))
        self._push("tok")

    def _push_codes(self):
        """B's Mimi codes of T (global ids) -> dF's Mimi decoder input."""
        self.codes_s.copy_(self.depB.index_select(0, self.Td[self.dS])[0])
        self._push("codes")

    def _wait(self, d, slot):
        tpk.wait(self.mb[d], slot, self.ep[d])

    def _mode(self, warm):
        """warm: every hand-off / all-reduce push targets local scratch and every flag reads 'ready' (capture warm-ups run per GPU with
        no cross-GPU effect); live: the real peer tables, flags zeroed."""
        big = 1 << 60
        for h in self.ho.values():
            h["tab"].copy_(h["warm"] if warm else h["live"])
        for r, a in enumerate(self.ars):
            if warm:
                if not hasattr(a, "dst_live"):
                    a.dst_live, a.flg_live = a.dst.clone(), a.flg.clone()
                    sc = self.scratch[a.dev]
                    a.dst_warm = torch.full_like(a.dst, a.recv.data_ptr() + (2 * a.G * a.BMAX * a.N) * 4 * 0)   # own buffer (warm-up junk)
                    a.flg_warm = torch.full_like(a.flg, sc.data_ptr() + (1 << 15) * 8 + 64)
                a.dst.copy_(a.dst_warm); a.flg.copy_(a.flg_warm)
                a.flags.fill_(big)
            else:
                a.dst.copy_(a.dst_live); a.flg.copy_(a.flg_live)
                a.flags.zero_(); a.err.zero_()
        for mb in self.mb.values():
            mb.flags.fill_(big if warm else 0)
            if not warm:
                mb.err.zero_()
        for d in self.devs:
            torch.cuda.synchronize(d)

    # ---------------------------------------------------------------------------------------------- frame state
    def set_T(self, T):
        self.serial += 1
        for d in self.devs:
            th = self.Tv_h[d][self.serial % 2]
            th[:8] = int(T)
            th[8] = self.serial
            with torch.cuda.device(d):
                self.Tv[d].copy_(th, non_blocking=True)
                self.ev_T[d].record()

    # ---------------------------------------------------------------------------------------------- pose branch (dP)
    def _pose(self, full, cfgdec=False):
        """live engine's pose branch on dP (the pose-CFG path is its _pose_cfg, unchanged; this is its 1-row path with dP's frame index)."""
        if self.pcfg:
            return self._pose_cfg(full, cfgdec)
        m, dP = self.m, self.dP
        tw = m.tower_p
        T = self.Td[dP]
        sl = T.to(torch.int32)
        pprev = self.pdepB.index_select(0, T - 1)
        e = tw.text_emb(self.tokp) + tw.code_emb(pprev).to(BF) + tw.cond(self.hn_p).to(BF)
        h = e[None]
        cos, sin = self.rope_p[0].index_select(1, T), self.rope_p[1].index_select(1, T)
        for i, L in enumerate(self.tp.L):
            h = self._layer_step1(L, h, cos, sin, self.tpK[i], self.tpV[i], sl)
        hv = FO.rms(h[0], tw.norm.weight, float(tw.norm.variance_epsilon))
        self.hv_p.copy_(hv)
        z0 = (hv @ tw.cb0_head.weight.t()).float()
        self.zp0.copy_(z0)
        if full:
            zb = (m.decp_bb(self.hn_p) * float(m.decp_bb_alpha)) if self.bblink else None
            p0 = LP.t_sample(z0, self.gen_p, self.pf0[0], self.pi0[0])
            codes = self.dec_p.gen(hv, p0, self.gen_p, self.pf0[1], self.pi0[1], zb=zb)[0]
            self.pdepB.index_copy_(0, T, (codes + self.offp)[None])
        self.pkp.copy_(self.pdepB.index_select(0, T)[0])

    def _pose_g(self, full, cfgdec):
        self._wait(self.dP, HF_TOK)                                   # B's lane token of T (pushed by dS after the text graph)
        self._pose(full, cfgdec)

    # ---------------------------------------------------------------------------------------------- front / Mimi (dF)
    def _mdec_g(self):
        self._wait(self.dF, HF_CODES)
        self._mdec()

    # ---------------------------------------------------------------------------------------------- trunk (tensor parallel)
    def _trunk_in(self):
        """rank 0: the frame's trunk input rows (= live _seg / _seg3 segment-0 input) -> broadcast (all-reduce site 0, slot 0)."""
        dS = self.dS
        self._wait(dS, HF_PKP)
        self._wait(dS, HF_AVEC)
        pk = self.pk[dS]
        ids = torch.cat([self.idsA, pk[0:1].view(1, 1)], 0)
        dep = torch.cat([self.depA, pk[1:].view(1, 1, DEPTH)], 0)
        pdep = self.pkp_s.view(1, DEPTH)
        e01 = self.m.inp(ids, dep, self.pres, pdep) + self.a_vec_s[0:2]
        if self.pcfg:
            T = self.Td[dS]
            fr = (T - self.Pd - CD).clamp(0, self.n_fr - 1)
            st = self.sil_t.index_select(0, fr)
            du = torch.where(dep[..., :1] >= 0, st[None].expand(2, -1, -1), torch.full_like(dep, -1))
            eu = self.m.inp(ids, du, self.pres, pdep) + self.a_vec_s[2:3]
            self.h0.copy_(torch.cat([e01, eu], 0))
        else:
            self.h0.copy_(e01)
        self.ars[0].part[:self.NR].copy_(self.h0)                     # fp32 (exact) -> broadcast at site 0

    def _trunk(self, r):
        s, ar, d = self.shards[r], self.ars[r], self.tpd[r]
        NR, nq, nkv = self.NR, s["nq"], s["nkv"]
        D = self.c32.head_dim
        if r == 0:
            self._trunk_in()
        T3 = self.T3[d]
        sl = (T3 + 1).to(torch.int32)
        cos, sin = s["cos"].index_select(0, T3), s["sin"].index_select(0, T3)
        Ls = s["L"]
        part = ar.part[:NR]
        ar.reduce(0, s["res"], Ls[0]["ln1"], Ls[0]["e1"], s["x"], NR, src0=True, has_res=False)
        for i, Ly in enumerate(Ls):
            tpk.gemv(s["x"], Ly["wqkv"], s["qkv"])
            tpk.qkv_post(s["qkv"], Ly["qn"], Ly["kn"], Ly["eqk"], cos, sin, T3, s["q"], s["K"][i], s["V"][i], nq, nkv, D)
            a = self.fa.flash_attn_with_kvcache(s["q"].view(NR, 1, nq, D), s["K"][i], s["V"][i], cache_seqlens=sl, softmax_scale=Ly["sc"],
                                                causal=True, num_splits=self.attn_splits)
            tpk.gemv(a.reshape(NR, nq * D), Ly["wo"], part, f32=True)
            ar.reduce(1 + 2 * i, s["res"], Ly["ln2"], Ly["e2"], s["x2"], NR)
            tpk.gemv(s["x2"], Ly["wgu"], s["gu"])
            LK.silu_mul_gu(s["gu"], s["act"])
            tpk.gemv(s["act"], Ly["wd"], part, f32=True)
            last = i == len(Ls) - 1
            ar.reduce(2 + 2 * i, s["res"], s["nw"] if last else Ls[i + 1]["ln1"], s["ne"] if last else Ls[i + 1]["e1"], s["x"], NR)
        if d == self.dS:
            self.hn_prev.copy_(s["x"][0:2])
            if self.pcfg:
                self.hn_u.copy_(s["x"][2:3])
        if d == self.dP:
            if self.pcfg:
                self.hn_p[0:1].copy_(s["x"][0:1]); self.hn_p[1:2].copy_(s["x"][2:3])
            else:
                self.hn_p.copy_(s["x"][0:1])

    # ---------------------------------------------------------------------------------------------- tensor-parallel prefill (eager)
    @torch.no_grad()
    def _prefill(self, e, row):
        """e [P, H] (dS) -> KV rows `row` of every rank for positions [0, P); returns the final hidden h [P, H] per rank (pre-norm)."""
        P = e.shape[0]
        NR, D = self.NR, self.c32.head_dim
        hs = [e.to(d) for d in self.tpd]
        cu = {d: (torch.tensor([0, P], dtype=torch.int32, device=d)) for d in self.tpd}
        nl = len(self.shards[0]["L"])
        for i in range(nl):
            parts = []
            for r, (s, d) in enumerate(zip(self.shards, self.tpd)):
                with torch.cuda.device(d):
                    Ly = s["L"][i]
                    nq, nkv = s["nq"], s["nkv"]
                    x = FO.rms(hs[r][None], Ly["ln1"], Ly["e1"])
                    qkv = F.linear(x, Ly["wqkv"])
                    q, k = FO.qk_prep(qkv[..., :nq * D].contiguous(), qkv[..., nq * D:(nq + nkv) * D].contiguous(), Ly["qn"], Ly["kn"], Ly["eqk"],
                                      s["cos"][None, :P], s["sin"][None, :P])
                    v = qkv[0, :, (nq + nkv) * D:].reshape(P, nkv, D)
                    s["K"][i][row, :P] = k[0]
                    s["V"][i][row, :P] = v
                    a = fa3_op.fa3_varlen_func(q[0], s["K"][i][row, :P], s["V"][i][row, :P], cu[d], cu[d], P, P, Ly["sc"], True)
                    parts.append(mm_f32(a.reshape(P, nq * D), Ly["wo"]))
            xs = []
            for r, (s, d) in enumerate(zip(self.shards, self.tpd)):
                with torch.cuda.device(d):
                    acc = parts[0].to(d)
                    for g in range(1, self.G):
                        acc = acc + parts[g].to(d)
                    hs[r], x2 = FO.add_rms(hs[r][None], acc.to(BF)[None], s["L"][i]["ln2"], s["L"][i]["e2"])
                    hs[r] = hs[r][0]
                    xs.append(x2[0])
            parts = []
            for r, (s, d) in enumerate(zip(self.shards, self.tpd)):
                with torch.cuda.device(d):
                    gu = F.linear(xs[r], s["L"][i]["wgu"])
                    act = LK.silu_mul_gu(gu.contiguous())
                    parts.append(mm_f32(act, s["L"][i]["wd"]))
            for r, (s, d) in enumerate(zip(self.shards, self.tpd)):
                with torch.cuda.device(d):
                    acc = parts[0].to(d)
                    for g in range(1, self.G):
                        acc = acc + parts[g].to(d)
                    hs[r] = hs[r] + acc.to(BF)
        return hs

    # ---------------------------------------------------------------------------------------------- graphs
    @torch.no_grad()
    def capture(self, T=64, P=32):
        """warm (every push -> local scratch, every flag 'ready'): each graph's function run twice per GPU (Triton autotune / FA3 / allocator
        warm-up), then captured; then live tables + zeroed flags."""
        t0 = time.time()
        self._mode(True)
        self.set_T(T)
        for d in self.devs:
            torch.cuda.synchronize(d)
        self.Pd0.fill_(P); self.Pd.fill_(P)
        self.laneB.fill_(PAD); self.depB.fill_(-1); self.pdepB.fill_(-1)
        dS, dP, dF = self.dS, self.dP, self.dF
        jobs = [("text", dS, self._text, (self.gen_t, self.gen_c), None),
                ("tokpush", dS, self._push_tok, (), None),
                ("audio", dS, lambda: self._audio(False), (), None),
                ("audio_full", dS, lambda: self._audio(True), (self.gen_a,), None),
                ("codespush", dS, self._push_codes, (), None),
                ("pose", dP, lambda: self._pose_g(False, False), (), None),
                ("pose_full", dP, lambda: self._pose_g(True, False), (self.gen_p,), None),
                ("pkppush", dP, lambda: self._push("pkp"), (), None)]
        if self.pcfg:
            jobs.append(("pose_full_cfg", dP, lambda: self._pose_g(True, True), (self.gen_p,), None))
        for r, d in enumerate(self.tpd):
            jobs.append((f"trunk{r}", d, lambda r=r: self._trunk(r), (), None))
        jobs.append(("avecpush", dF, lambda: self._push("avec"), (), None))
        vx = self.vx
        sv = (vx.mel_prev.clone(), vx.c1_prev.clone(), vx.kd.clone(), self.a_vec0.clone())
        vx.kd.fill_(1)
        jobs.append(("front", dF, self._front_graphable, (), None))
        st = [t.clone() for t in self._mdec_state()]
        jobs.append(("mdec", dF, self._mdec_g, (), self.sideF))
        for name, dev, fn, gens, stream in jobs:
            with torch.cuda.device(dev):
                s = stream if stream is not None else torch.cuda.Stream(dev)
                s.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(s):
                    for _ in range(2):
                        fn()
                torch.cuda.current_stream(dev).wait_stream(s)
                torch.cuda.synchronize(dev)
                if os.environ.get("FASTMG_DEBUG"):
                    for d_ in self.devs:
                        torch.cuda.synchronize(d_)
                    log(f"[mg capture] warm-up ok: {name}")
                g = torch.cuda.CUDAGraph()
                for gen in gens:
                    g.register_generator_state(gen)
                with torch.cuda.graph(g, stream=s):
                    fn()
                torch.cuda.synchronize(dev)
            self.graphs[name] = g
            if name == "front":
                vx.mel_prev.copy_(sv[0]); vx.c1_prev.copy_(sv[1]); vx.kd.copy_(sv[2]); self.a_vec0.copy_(sv[3])
            if name == "mdec":
                for t, s_ in zip(self._mdec_state(), st):
                    t.copy_(s_)
        self.ring.reset()
        self._mode(False)
        log(f"[mg] captured {len(self.graphs)} graphs in {time.time() - t0:.1f}s; " +
            ", ".join(f"{d} {torch.cuda.memory_allocated(d) / 2 ** 30:.1f} GiB" for d in self.devs))

    # ---------------------------------------------------------------------------------------------- session
    @torch.no_grad()
    def start(self, pre, seed, params=None, uncond_src="quiet_frames"):
        """= live engine start(): prefill [0, P) (trunk tensor-parallel; speech tower on dS, pose tower on dP), row 1 = row 0, pose-CFG row 2 =
        the silenced prefix, reset every stream state, seed the generators, params."""
        P = int(pre["P"])
        assert 5 <= P and P + self.n_fr + CD <= self.lmax, ("prefix too long for the static buffers", P, self.lmax)
        assert uncond_src in UNCOND_SRC, uncond_src
        ids, depth, present = (np.asarray(pre[k]) for k in ("ids", "depth", "present"))
        assert ids.shape == (2, P) and depth.shape == (2, P, DEPTH) and present.shape == (2, P)
        dS, dP, dF = self.dS, self.dP, self.dF
        for d in self.devs:
            torch.cuda.synchronize(d)
        self.P = P
        self.laneB.fill_(PAD); self.depB.fill_(-1)
        self.laneB[:P].copy_(torch.as_tensor(ids[1], device=dS)); self.depB[:P].copy_(torch.as_tensor(depth[1], device=dS))
        pd0 = torch.full((P, DEPTH), -1, dtype=torch.long, device=dS)
        e = self.m.inp(torch.as_tensor(ids, device=dS), torch.as_tensor(depth, device=dS), torch.as_tensor(present, device=dS), pd0)
        hs = self._prefill(e, 0)
        n = self.m.head.norm
        with torch.cuda.device(dS):
            hn = FO.rms(hs[0], n.weight, float(n.variance_epsilon))
            tw = self.m.tower_s
            hprev = torch.cat([torch.zeros_like(hn[:1]), hn[:-1]], 0)
            dprev = np.concatenate([np.full((1, DEPTH), -1, np.int64), depth[1, :-1]], 0)
            et = tw.text_emb(torch.as_tensor(ids[1], device=dS)) + tw.code_emb(torch.as_tensor(dprev, device=dS)).to(BF) + tw.cond(hprev).to(BF)
            self.ts.run(et[None], 0)
            self.hn_prev.copy_(hn[-1:].expand(2, -1))
            self.Pd.fill_(P)
            self.ring.reset()
            self.lastspk.fill_(P)
            self.fstate.zero_()
        rp = self.rP
        with torch.cuda.device(dP):
            hnP = FO.rms(hs[rp], self.shards[rp]["nw"], self.shards[rp]["ne"])
            hprevP = torch.cat([torch.zeros_like(hnP[:1]), hnP[:-1]], 0)
            self.pdepB.fill_(-1)
            tw = self.m.tower_p
            pprev = torch.full((P, DEPTH), -1, dtype=torch.long, device=dP)
            ep_ = tw.text_emb(torch.as_tensor(ids[1], device=dP)) + tw.code_emb(pprev).to(BF) + tw.cond(hprevP).to(BF)
            self.tp.run(ep_[None], 0)
            self.pkp.fill_(-1)
            self.hn_p[0:1].copy_(hnP[-1:])
        for s, d in zip(self.shards, self.tpd):
            with torch.cuda.device(d):
                for t in s["K"] + s["V"]:
                    t[1, :P].copy_(t[0, :P])
        for K_ in (self.tsK, self.tsV):
            for t in K_:
                with torch.cuda.device(t.device):
                    t[1, :P].copy_(t[0, :P])
        if self.pcfg:
            du = LP.silence_depth(depth, self.sil)
            eu = self.m.inp(torch.as_tensor(ids, device=dS), torch.as_tensor(du, device=dS), torch.as_tensor(present, device=dS), pd0)
            hu = self._prefill(eu, 2)
            with torch.cuda.device(dS):
                hnu = FO.rms(hu[0], n.weight, float(n.variance_epsilon))
                self.hn_u.copy_(hnu[-1:])
            with torch.cuda.device(dP):
                hnuP = FO.rms(hu[rp], self.shards[rp]["nw"], self.shards[rp]["ne"])
                hprev_u = torch.cat([torch.zeros_like(hnuP[:1]), hnuP[:-1]], 0)
                self.hn_p[1:2].copy_(hnuP[-1:])
                tw = self.m.tower_p
                self.tp.k, self.tp.v = [k[1] for k in self.tpK], [v[1] for v in self.tpV]
                try:
                    pprev = torch.full((P, DEPTH), -1, dtype=torch.long, device=dP)
                    epu = tw.text_emb(torch.as_tensor(ids[1], device=dP)) + tw.code_emb(pprev).to(BF) + tw.cond(hprev_u).to(BF)
                    self.tp.run(epu[None], 0)
                finally:
                    self.tp.k, self.tp.v = [k[0] for k in self.tpK], [v[0] for v in self.tpV]
            self.prefill_uncond = {"silenced_codes": int(((depth[:, :, 0] >= 0)).sum()), "P": P}
        self.prefill_hn = hn
        with torch.cuda.device(dF):
            self.vx.reset(); self.abuf.zero_(); self.a_vec0.zero_(); self.mdec.reset(); self.ybuf.zero_()
            self.Pd0.fill_(P)
            self.usrc.fill_(UNCOND_SRC.index(uncond_src))
        self.uncond_src = uncond_src
        self.gen_t.manual_seed(int(seed) + 1)
        self.gen_a.manual_seed(int(seed))
        self.gen_c.manual_seed(int(seed) + 2)
        self.gen_p.manual_seed(int(seed) + 3)
        self.params = {}
        p0 = dict(DEFAULTS, **(params or {}))
        if "min_p" not in (params or {}):
            p0["min_p"] = LP.MINP_DEFAULT[int(validate({"factorised_text": p0["factorised_text"]})[0].get("factorised_text", DEFAULTS["factorised_text"]))]
        self.set_params(p0)
        for d in self.devs:
            torch.cuda.synchronize(d)

    @torch.no_grad()
    def step(self, T, graphs=True, force=None):
        """frame f = T - P (live engine step(); same cold-start PAD, same force semantics for parity tests). graphs=False is not
        supported (the tensor-parallel trunk exists only as graphs)."""
        assert graphs and self.graphs, "fastmg engine: graphs only"
        P = self.P
        f = T - P
        assert f >= 0
        self.set_T(T)
        dS, dP, dF = self.dS, self.dP, self.dF
        full = T >= P + CD
        G = self.graphs
        with torch.cuda.device(dF):                                   # front -> a_vec (dS)
            if f == 0:
                assert self.vx.k == f, ("Voxtral front out of step", self.vx.k, f)
                self._front_eager(f)
            else:
                G["front"].replay()
                self.vx.k += 1
            G["avecpush"].replay()
        with torch.cuda.device(dS):                                   # text -> lane token (dP)
            G["text"].replay()
            if f == 0 or force is not None:
                self.laneB[T:T + 1].fill_(PAD if force is None else int(force[0]))
                if f == 0:
                    self.ring.reset()
            G["tokpush"].replay()
        with torch.cuda.device(dP):                                   # pose branch -> pose codes (dS)
            cfgdec = self.pcfg and bool(self.params.get("pose_cfg_on")) and bool(self.params.get("pose_cfg_light"))
            G[("pose_full_cfg" if cfgdec else "pose_full") if full else "pose"].replay()
            if force is not None and len(force) > 2 and force[2] is not None:
                self.pdepB[T].copy_(torch.as_tensor(np.asarray(force[2], np.int64), device=dP))
                self.pkp.copy_(self.pdepB[T])
            G["pkppush"].replay()
        with torch.cuda.device(dS):                                   # speech branch -> B codes (dF Mimi)
            G["audio_full" if full else "audio"].replay()
            if force is not None and force[1] is not None:
                self.depB[T].copy_(torch.as_tensor(np.asarray(force[1], np.int64), device=dS))
                self.pk[dS].copy_(torch.cat([self.laneB[T:T + 1], self.depB[T]]))
            if full:
                G["codespush"].replay()
        for r, d in enumerate(self.tpd):                              # the tensor-parallel trunk (rank 0 first waits pose codes + a_vec)
            with torch.cuda.device(d):
                G[f"trunk{r}"].replay()
        with torch.cuda.device(dF):
            self.vrec[f].copy_(self.a_vec0[0])
            if full:
                self.sideF.wait_event(self.ev_T[dF])                  # the frame serial (ep) on dF is in place
                with torch.cuda.stream(self.sideF):
                    G["mdec"].replay()
        return full

    def result(self, T, dec):
        torch.cuda.synchronize(self.dS)
        tokB = int(self.laneB[T].item())
        pp, pn = self.probs.tolist()
        self.last_ptok = self.ptok.tolist()
        self.last_cfgd = self.cfgd.tolist()
        fs = self.fstate.tolist()
        self.last_adapt, self.last_padad = fs[0:3], fs[3:6]
        pk = self.pk[self.dS].tolist()
        self.last_codes = [c - o if c >= 0 else -1 for c, o in zip(pk[1:], OFF.tolist())]
        pp_ = self.pkp.tolist()
        self.last_pose = [c - o if c >= 0 else -1 for c, o in zip(pp_, OFF.tolist())]
        y = None
        if dec:
            self.sideF.synchronize()
            y = self.ybuf[0].cpu().numpy().copy()
        self.errs_checked += 1
        if self.errs_checked % 25 == 0:
            self.check_errors()
        return tokB, pp, pn, y

    def check_errors(self):
        bad = {}
        for a in self.ars:
            v = int(a.err.item())
            if v:
                bad[f"allreduce@{a.dev}"] = v
        for d, mb in self.mb.items():
            v = int(mb.err.item())
            if v:
                bad[f"handoff@{d}"] = v
        if bad:
            raise RuntimeError(f"fastmg: a cross-GPU wait timed out (site / slot codes): {bad}")
        return True
