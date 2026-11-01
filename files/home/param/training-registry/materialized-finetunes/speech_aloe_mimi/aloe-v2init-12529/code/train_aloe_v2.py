"""aloe-v2init (10-17): = aloe-v2-ft job 22383 EXACTLY, only QQA_INIT + run dirs differ. Code changes vs 22383 (no maths change):
  (1) cfg diff vs 22383 resolved cfg (ref_cfg_22383.json), allowed = init + run dirs; (2) load_init accepts a consolidated bf16 DCP
  (no meta.json -> load_init_dcp, = aloe-vox-ft load_init_dcp); (3) init checksum (exact int64 sums of bf16 bits) -> LOGDIR/init_checksum.json;
  (4) per-rank step-0/1 train items + eval items -> LOGDIR/rank_items.json.
aloe-v2-ft TRAINER (10-16 eve, user spec) = aloe-ft/code/train_aloe.py (job 21980) with ONLY these changes:
  DATA      data_aloe_v2: FIRST chunk of every call only (both speakers as B), held-out = 21980's eval rows that are first chunks; system
            prompt with NO text (no persona / context / memory), B's own voice clip (PPM voice_clips rule) on a seeded 30 % of (row, orientation)
            items, the EMPTY system block otherwise; A no clip. Realised clip fraction logged (cumulative) every step.
  DECODER   0.6B speech decoder: K = 1 nanostep per micro (QQA_DEC_K 1; was 8) on a seeded random ~10 % of B's coded frames per micro (data
            light_frac 0.1; was 1.0). Decoder LR schedule in update units = warmup x K, decay to 0 at steps x K (K = 1 -> = the main schedule).
            The "every selected light frame exactly once" check stays (decoder frames == the step's light frames) and is ADAPTED: the step's
            light frames / its B cb0 frames must lie in [0.09, 0.11] (21980: == 1).
  MODEL     model_ppm nopose (no inp.pose / tower_p / dec_p / pose_head / decp_bb built; 21980 had them frozen + never run: same maths);
            compile_decoder ON (= 21980).
  CONFIG    the resolved CFG is diffed against 21980's (ref_cfg_21980.json, from its log) at start; any difference outside the allowed keys
            (items 1-3 of the spec + run dirs) aborts.
Original aloe-ft doc follows.
aloe-ft TRAINER (10-16): PPM full fine-tune on Model 1's Aloe handover rows, BOTH sides as B (data_aloe: every train row twice per
epoch, once per orientation; loss on B only; PPM's sysprompt rule with both speakers' persona text), with qqa-ft's recipe (qqa.env).
= qqa-ft/code/train_qqa.py (job 21823 / 21832) with ONLY these changes:
  DATA      data_aloe plan (one 7,500-frame row = one micro per rank per step, identical plan on all ranks, node-local copies of all 4 stores
            on every node); the last step's spare ranks run DUMMY micros (zero loss weights, zero counts, zero decoder chunk weights: they add
            nothing to any loss / gradient / statistic; asserted zero).
  EVAL      the stores' 240 eval rows x both orientations (480 micros, round-robin over ranks, dummies on the short ranks), reported per
            orientation (eval_asstB = data assistant on B, eval_userB = data user on B) and combined (eval), at step 0, every
            QQA_EVAL_EVERY steps, every export step and the final step.
  EXPORTS   intermediate weights-only exports at QQA_EXPORT_STEPS -> NODE-LOCAL bf16 shards (ppm-local-bf16-shards-v1, world = this run's,
            rank<R>.pt on each node's ALOE_LOCAL_EXPORT_ROOT/export-<step>, rep.pt + meta.json on rank 0's node; = ckpt_local_export's
            format, readable by train_qqa.load_init / infer_ppm load_weights), so /data never holds them; final = qqa's consolidated DCP.
  LOGGING   train/loss (the optimised per-step token-mean loss, text + cb0, mean over ranks) emitted.
Original qqa-ft doc follows.
qqa-ft TRAINER: run C's speech-duplex quick-QA fine-tune (s4-qqa job 21169) on a PPM checkpoint, in PPM's own model / speech-row maths.
A trimmed copy of ppm/code/train_ppm.py (job 21765's code, snapshot in qqa-ft/code): same model (model_ppm), same per-micro forward / loss
terms (B text CE with PPM's factorised weights, speech-tower cb0 CE, 0.6B speech decoder by NANOSTEPS), FSDP2, bf16 compute / fp32 masters.
Differences from train_ppm (all forced by the run-C recipe, each listed in qqa-ft/STATUS.md's mapping table):
  DATA      speech micros only: ONE packed micro per rank per step = run C's conversations of that step (qqa-ft/plan/PLAN-<host>.json, run C's
            order), each conversation a PPM speech row (data_qqa: swap 0, empty system block, CD 2) in its own attention segment.
  LOSS NORM run C's S4_PACK_NORM=tokens_step: every target token of the step weighs the same. Per loss term the normaliser = the STEP's total
            over all ranks / WORLD (FSDP averages over ranks): text (B text targets), cb0 (B cb0 labels), and for the 0.6B decoder update j
            the step's total chunk-j frames over ranks / WORLD (decoder all-reduce = mean over ranks); its input gradient into the tower is the
            chunk gradients weighted M_j / N (M_j = chunk j frames over all ranks, N = all light frames) = d(token mean)/d(hv).
            Step-0 PACKED-LOSS GATE (run C's S4_PACK_LOSSCHECK): with micro-level token normalisers, each packed loss term (text / cb0 / light)
            == the token-weighted mean of the conversations' standalone losses (max rel err over ranks < 2e-3, asserted).
  INIT      every weight from a PPM local inference copy (ckpt_local_export.py: bf16 dim-0 FSDP shards of world 128 + rep.pt), resharded to
            this world; fresh AdamW (run C: fresh Adam from its init export).
  LR        run C's groups / peaks mapped (env QQA_LR_*), run C's schedule: linear warmup over QQA_WARMUP steps then linear decay to 0 at the
            last step (sched_s3q wsd, floor 0); decoder nanostep updates on the same shape in update units (warmup x K, decay to 0 at steps x K).
  FROZEN    pose parts (tower_p, dec_p, pose_head, decp_bb, inp.pose) are in no optimizer, never run, asserted grad-free and bit-identical.
  EVAL      B normalised word CE (+ text / cb0 / light CE) on run C's 96 held-out conversations (4 per rank, unpacked) at step 0, every
            QQA_EVAL_EVERY steps and the final step.
  SAVE      no optimizer checkpoint; weights-only consolidated bf16 DCP export at the final step (QQA_EXPORT, readable by infer_ppm) and on
            STOP_AFTER_SAVE (control file / SIGUSR1: export at the next step boundary, then exit cleanly)."""
import datetime
import hashlib
import json
import math
import os
import shutil
import signal
import socket
import sys
import time
import zoneinfo

import numpy as np
import torch
import torch.distributed as dist

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_ppm as M  # noqa: E402
import data_qqa as Q  # noqa: E402
import data_aloe_v2 as A  # noqa: E402
import sched_s3q as SCH  # noqa: E402

D = Q.D


def E(k, d=None):
    v = os.environ.get(k, d)
    assert v is not None, ("missing env", k)
    return v


CFG = {
    "run_tag": E("QQA_RUN"), "log_root": E("QQA_LOG_ROOT"), "init": E("QQA_INIT"), "plan_dir": E("QQA_PLAN_DIR"),
    "export": E("QQA_EXPORT"), "control_dir": E("QQA_CONTROL_DIR"), "export_min_free_gb": float(E("QQA_EXPORT_MIN_FREE_GB", "300")),
    # 10-16 scaling (export keys only): weights-only exports at these optimizer steps -> <dirname(export)>/export-<step>; an export waits
    # (polling every 15 s, up to QQA_EXPORT_WAIT_MIN) for /data space above the floor instead of being refused (training maths unchanged)
    "export_steps": [int(x) for x in E("QQA_EXPORT_STEPS", "").split(",") if x], "export_wait_min": float(E("QQA_EXPORT_WAIT_MIN", "0")),
    "steps": int(E("QQA_STEPS")), "warmup": int(E("QQA_WARMUP")), "eval_every": int(E("QQA_EVAL_EVERY")),
    "w_text": float(E("QQA_W_TEXT_CE")), "w_cat_word": float(E("QQA_W_CAT_WORD")), "w_tim": float(E("QQA_W_TIM_CE")),
    "w_nw": float(E("QQA_W_NW_CE")), "w_eob": float(E("QQA_W_EOB_CE")), "w_cb0": float(E("QQA_W_CB0")), "w_cb": float(E("QQA_W_CB")),
    "lr": {"trunk": float(E("QQA_LR_TRUNK")), "spec": float(E("QQA_LR_SPEC")), "proj": float(E("QQA_LR_PROJ")),
           "audio": float(E("QQA_LR_AUDIO")), "tower_s": float(E("QQA_LR_TOWER"))},
    "clip": {"trunk": float(E("QQA_CLIP_TRUNK")), "spec": float(E("QQA_CLIP_SPEC")), "proj": float(E("QQA_CLIP_PROJ")),
             "audio": float(E("QQA_CLIP_AUDIO")), "tower_s": float(E("QQA_CLIP_TOWER"))},
    "adam_beta2": float(E("QQA_ADAM_BETA2")),
    "dec_k": int(E("QQA_DEC_K")), "dec_lr": float(E("QQA_DEC_LR")), "dec_clip": float(E("QQA_DEC_CLIP")), "dec_seed": int(E("QQA_DEC_SEED")),
    "dec_micro_frames": int(E("QQA_DEC_MICRO_FRAMES", "300")), "dec_mb_max": int(E("QQA_DEC_MB_MAX", "470")),
    "dec_even": E("QQA_DEC_EVEN", "1") == "1", "dec_dyn": E("QQA_DEC_DYN", "1") == "1", "dec_fast": E("QQA_DEC_FAST", "1") == "1",
    "dec_compile_mode": E("QQA_DEC_COMPILE_MODE", "default"), "eval_light_max": int(E("QQA_EVAL_LIGHT_MAX", "2048")),
    "losscheck": int(E("QQA_LOSSCHECK", "1")), "losscheck_tol": float(E("QQA_LOSSCHECK_TOL", "2e-3")),
    "dl_workers": int(E("QQA_DL_WORKERS", "3")), "data": A.aloe_cfg(), "cd": D.CD,
    "local_export_root": E("ALOE_LOCAL_EXPORT_ROOT"), "stores": A.STORES,
    "eval_data": A.eval_cfg(), "model_nopose": True, "first_chunks_only": True, "ref_cfg": E("ALOE_REF_CFG"),
    "light_ratio_lo": float(E("ALOE_LIGHT_RATIO_LO", "0.09")), "light_ratio_hi": float(E("ALOE_LIGHT_RATIO_HI", "0.11")),
}
CFG["sched"] = {"warmup": CFG["warmup"], "horizon": CFG["steps"], "min_lr_rate": 0.0, "s3_sched": "wsd", "wsd_decay_start": CFG["warmup"],
                "wsd_decay_end": CFG["steps"], "wsd_floor": 0.0, "wsd_shape": "linear"}
CFG["dec_sched"] = {"warmup": CFG["warmup"] * CFG["dec_k"], "horizon": CFG["steps"] * CFG["dec_k"], "min_lr_rate": 0.0, "s3_sched": "wsd",
                    "wsd_decay_start": CFG["warmup"] * CFG["dec_k"], "wsd_decay_end": CFG["steps"] * CFG["dec_k"], "wsd_floor": 0.0, "wsd_shape": "linear"}
TRAIN_GROUPS = ("trunk", "spec", "proj", "audio", "tower_s")
FROZEN_PREFIX = ()                                                # v2 NOPOSE: no pose parts built -> nothing frozen
RANK, WORLD, LOCAL = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"]), int(os.environ["LOCAL_RANK"])
LW = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
HOST = socket.gethostname().split(".")[0]
PT = zoneinfo.ZoneInfo("America/Los_Angeles")
LOGDIR = f"{CFG['log_root']}/{CFG['run_tag']}"
GiB = 2 ** 30
assert CFG["cd"] == 2, ("PPM code delay must be 2 (PPM_CD)", CFG["cd"])


def now_pt():
    return datetime.datetime.now(PT).strftime("%m-%d %H:%M:%S PT")


def log(*a):
    if RANK == 0:
        print(f"[{now_pt()}]", *a, flush=True)


def allsum(v, dev):
    t = torch.tensor(v, dtype=torch.float64, device=dev)
    dist.all_reduce(t)
    return t.tolist()


def frozen_name(n):
    n = n.replace("_checkpoint_wrapped_module.", "")
    return n.startswith(FROZEN_PREFIX)


def group_of(n):
    return None if frozen_name(n) else M.group_of(n)


# ------------------------------------------------------------------------------------------------ micro -> device + weights
ARR = ("ids", "depth", "present", "pid", "sel", "y", "word", "pad", "eob", "nw", "cb0", "light")


def prep(b, dev, S_text, S_cb0):
    """= train_ppm.prep with the normalisers given: S_text (text targets) and S_cb0 (cb0 labels). tokens_step: the step totals / WORLD;
    micro-level ('tokens', loss check): this micro's own totals."""
    t = {"kind": "speech", "src": "speech", "segs": [int(x) for x in b["segs"]]}
    for k in ARR:
        t[k] = torch.as_tensor(np.asarray(b[k])).to(dev, non_blocking=True)
    t["pid"] = t["pid"][None]
    w = {k: t[k].float() for k in ("word", "pad", "eob", "nw")}
    S = float(S_text)
    t["wt"] = ((CFG["w_cat_word"] * w["word"] + CFG["w_tim"] * w["pad"] + CFG["w_eob"] * w["eob"] + CFG["w_nw"] * w["nw"]) / S).float()
    t["wn"] = (CFG["w_text"] * w["word"] / S).float()
    m = (t["cb0"] >= 0).float()
    t["cb0_w"] = m / float(S_cb0)
    t["dummy"] = bool(b.get("dummy", 0))
    if t["dummy"]:                                                # aloe: a dummy micro carries zero weight in every loss term
        t["wt"], t["wn"], t["cb0_w"] = t["wt"] * 0.0, t["wn"] * 0.0, t["cb0_w"] * 0.0
    return t


TK = ("word_ce", "word_norm", "n_word", "word_hit", "pad_ce", "n_pad", "nw_ce", "n_nw", "eob_ce", "n_eob")
CB_S = (8, 16, 31)


def text_vec(t, st):
    w, p, n_, e = (t[k].float() for k in ("word", "pad", "nw", "eob"))
    return [((st[0] * w).sum()), ((st[3] * w).sum()), (w.sum()), ((st[1] * w).sum()), ((st[0] * p).sum()), (p.sum()),
            ((st[0] * n_).sum()), (n_.sum()), ((st[0] * e).sum()), (e.sum())]


def summarize(v, pre):
    d = dict(zip(TK, v[:len(TK)]))
    m = lambda a, b: d[a] / max(d[b], 1.0)  # noqa: E731
    k = f"{pre}_speech"
    out = {f"{k}/word_ce_norm": m("word_norm", "n_word"), f"{k}/word_ce": m("word_ce", "n_word"), f"{k}/word_top1": m("word_hit", "n_word"),
           f"{k}/pad_ce": m("pad_ce", "n_pad"), f"{k}/nw_ce": m("nw_ce", "n_nw"),
           f"{k}/text_ce": (d["word_ce"] + d["pad_ce"] + d["nw_ce"] + d["eob_ce"]) / max(d["n_word"] + d["n_pad"] + d["n_nw"] + d["n_eob"], 1.0),
           f"{k}/n_word": d["n_word"]}
    x = v[len(TK):]
    out[f"{k}/cb0_ce"] = x[0] / max(x[1], 1.0); out[f"{k}/cb0_top1"] = x[2] / max(x[1], 1.0)
    out[f"{k}/light_ce"] = x[3] / max(x[4], 1.0)
    for j, c in enumerate(CB_S):
        out[f"{k}/light_ce_cb{c}"] = x[5 + j] / max(x[4] / 31.0, 1.0)
    return out


# ------------------------------------------------------------------------------------------------ nanosteps (token-weighted over the step)
def dec_lam():
    return SCH.make_lambda(True, CFG["dec_sched"])


class DecOpt:
    """= train_ppm.DecOpt (replicated decoder: grads = views of ONE flat fp32 buffer, one all-reduce per update), LR = dec_lr x the run-C
    shaped schedule over decoder updates (micros x K + j)."""

    def __init__(self, dec):
        self.ps = list(dec.parameters())
        assert not hasattr(self.ps[0], "to_local"), "replicated decoder expected (PPM_DEC_MODE=replicate)"
        self.flat = torch.zeros(sum(p.numel() for p in self.ps), dtype=torch.float32, device=self.ps[0].device)
        o = 0
        for p in self.ps:
            p.grad = self.flat[o:o + p.numel()].view_as(p); o += p.numel()
        self.opt = torch.optim.AdamW([{"params": self.ps, "lr": 0.0}], betas=(0.9, CFG["adam_beta2"]), eps=1e-8, weight_decay=0.0, fused=True)
        self.micros = 0
        self.lam = dec_lam()

    def lr(self, s):
        return CFG["dec_lr"] * self.lam(int(s))


def _mb_plan(m):
    """= train_ppm._mb_plan (dec_even / dec_dyn speed path; same chunk gradient)."""
    MB = CFG["dec_micro_frames"]
    mm = max(m, 1)
    if not CFG["dec_even"]:
        return [(a0, MB) for a0 in range(0, -(-mm // MB) * MB, MB)]
    cap = CFG["dec_mb_max"] or MB
    nmb = -(-mm // cap)
    sz = -(-mm // nmb)
    if not CFG["dec_dyn"]:
        sz = -(-sz // 32) * 32
    plan = [(i * sz, sz) for i in range(nmb) if i * sz < mm]
    return [(a0, max(z, 16)) for a0, z in plan]


def nanosteps_tw(dec, do, hv, depth, light, dummy=False):
    """train_ppm.nanosteps (replicated, one decoder) with tokens_step weights: chunk j's frames weigh 1 / (31 x M_j / WORLD) (M_j = chunk-j
    frames summed over ranks) -> after the all-reduce mean the update's gradient = the mean CE over ALL ranks' chunk-j frames; the input
    gradient accumulates chunk gradients x M_j / N (N = all ranks' light frames) = d(step token mean x WORLD / ...)/d hv, i.e. the tower /
    backbone receive the token-weighted decoder loss. -> (acc, stats, lrs, (M, N))."""
    K, dev = CFG["dec_k"], hv.device
    hd = hv.detach().requires_grad_()
    n = int(light.numel())
    perm = torch.randperm(max(n, 1), generator=torch.Generator().manual_seed(CFG["dec_seed"] * 1000003 + do.micros * 1009 + RANK))
    sizes = [int(perm[j::K].numel()) if n else 0 for j in range(K)]
    Mg = torch.tensor([0] * K if dummy else sizes, dtype=torch.float64, device=dev)      # aloe: a dummy micro adds no frames to any update
    dist.all_reduce(Mg)
    Mj = Mg.tolist(); N = float(sum(Mj))
    perm_d = perm.pin_memory().to(dev, non_blocking=True)
    acc = torch.zeros(hd.shape, dtype=torch.float32, device=dev)
    st = {"sum": torch.zeros((), device=dev, dtype=torch.float32), "n": 0, "cb": torch.zeros(31, device=dev, dtype=torch.float32)}
    lrs = []
    for j in range(K):
        m = sizes[j]
        idx = perm_d[j::K] if m else perm_d[:0]
        hd.grad = None
        do.flat.zero_()
        den = 31.0 * Mj[j] / WORLD
        for a0, sz in _mb_plan(m):
            mv = max(0, min(sz, m - a0))
            fl = light.index_select(0, idx[a0:a0 + mv]) if mv else light[:1]
            if fl.numel() < sz:
                fl = torch.cat([fl, light[:1].expand(sz - fl.numel())])
            wf = torch.zeros(sz, dtype=torch.float32, device=dev)
            if mv and not dummy:
                wf[:mv] = 1.0 / den
            l_, s_, num_, cb_ = dec(hd, depth, fl, wf, n_valid=mv)
            if mv and not dummy:
                st["sum"] += s_; st["n"] += num_; st["cb"] += cb_.detach().float()
            l_.backward()
        if hd.grad is not None and m and not dummy:
            acc += hd.grad.float() * (Mj[j] / N)
        dist.all_reduce(do.flat)
        do.flat.div_(WORLD)
        lr = do.lr(do.micros * K + j)
        for g in do.opt.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(do.ps, CFG["dec_clip"])
        do.opt.step()
        lrs.append(lr)
    hd.grad = None
    do.micros += 1
    return acc, st, lrs, (Mj, N)


class DecFast:
    def __enter__(self):
        self.prev = M.DEC_FAST["active"]; M.DEC_FAST["active"] = CFG["dec_fast"]

    def __exit__(self, *a):
        M.DEC_FAST["active"] = self.prev


def run_micro(model, b, rot, do, grad, info=None):
    """= train_ppm.run_micro for a speech micro (NM = 1, w_task 1): forward, nanosteps (train) or a seeded <= eval_light_max frame decoder
    eval, ONE backward. -> (stats vector, loss value, extra)."""
    with (torch.enable_grad() if grad else torch.no_grad()):
        out = model(b, rot, rot, info=info)
    loss = out["text"] + CFG["w_cb0"] * out["cb0"]
    v = text_vec(b, out["text_stats"])
    extra = {}
    if grad:
        with DecFast():
            up, acc, lrs, (Mj, N) = nanosteps_tw(model.dec_s, do, out["hv_s"], b["depth"][1], b["light"], dummy=b.get("dummy", False))
        roots, grads = [loss, out["hv_s"]], [None, (up * CFG["w_cb"]).to(out["hv_s"].dtype)]
        dv = (acc["sum"], float(acc["n"]), acc["cb"])
        extra.update(dec_lrs=lrs, dec_M=Mj, dec_N=N)
    else:
        light = b["light"]
        with torch.no_grad():
            fl = light[torch.randperm(light.numel(), generator=torch.Generator().manual_seed(1234))[:CFG["eval_light_max"]].to(light.device)]
            _, s_, num_, cb_ = model.dec_s(out["hv_s"], b["depth"][1], fl, torch.zeros(fl.numel(), device=light.device))
        dv = (float(s_), float(num_), cb_.float())
    v += [out["cb0_sum"], out["cb0_n"], out["cb0_hit"], dv[0], dv[1]]
    v += [dv[2][c - 1] for c in CB_S]
    if grad:
        torch.autograd.backward(roots, grads)
    dvc = loss.device
    v = torch.stack([torch.as_tensor(x, device=dvc).double().reshape(()) for x in v]).tolist()
    if b.get("dummy", False):                                     # aloe: a dummy micro contributes nothing to any statistic
        assert float(loss) == 0.0, ("dummy micro with a non-zero loss", float(loss))
        v = [0.0] * len(v)
    return v, float(loss), extra


@torch.no_grad()
def micro_losses(model, mic, rot, dev):
    """one micro with MICRO-level token normalisers -> (text loss, cb0 loss, light mean CE over ALL its light frames), counts."""
    oc = np.asarray(mic["own"]).sum(0) if "own" in mic else np.asarray(Q.own_counts(mic))
    b = prep(mic, dev, oc[0], oc[1])
    out = model(b, rot, rot)
    hv, light = out["hv_s"], b["light"]
    s_tot, n_tot = 0.0, 0
    for a0 in range(0, light.numel(), 1024):
        fl = light[a0:a0 + 1024]
        _, s_, num_, _ = model.dec_s(hv, b["depth"][1], fl, torch.zeros(fl.numel(), device=dev))
        s_tot += float(s_); n_tot += int(num_)
    return [float(out["text"]), float(out["cb0"]), s_tot / max(n_tot, 1)], [float(x) for x in oc[:2]] + [float(n_tot)]


def pack_loss_check(model, b, rot, dev):
    """run C's S4_PACK_LOSSCHECK: each packed loss term == the token-weighted mean of the standalone conversations' losses. Same number of
    forwards on every rank (FSDP): ranks with fewer conversations repeat conversation 0 (ignored)."""
    model.eval()
    singles = b["singles"]
    Pv, _ = micro_losses(model, b, rot, dev)
    K = len(singles)
    t = torch.tensor([float(K)], device=dev); dist.all_reduce(t, op=dist.ReduceOp.MAX); Kmax = int(t)
    num, den = [0.0] * 3, [0.0] * 3
    for k in range(Kmax):
        L, N = micro_losses(model, singles[k if k < K else 0], rot, dev)
        if k < K:
            for i in range(3):
                num[i] += N[i] * L[i]; den[i] += N[i]
    model.train()
    exp = [num[i] / den[i] for i in range(3)]
    err = [abs(Pv[i] - exp[i]) / max(abs(exp[i]), 1e-12) for i in range(3)]
    return Pv, exp, err, K


# ------------------------------------------------------------------------------------------------ init / export
def load_init_dcp(model, path, dev):
    """v2init: QQA_INIT = a consolidated bf16 DCP export (PPM safety_ppm.export_weights w-N: .metadata + COMPLETE): every model tensor
    loaded in place into this world's FSDP2 shards / replicated tensors (DCP reshards; bf16 -> param dtype by copy) = aloe-vox-ft load_init_dcp."""
    import pickle
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    assert os.path.isfile(f"{path}/COMPLETE") and os.path.isfile(f"{path}/.metadata"), path
    keys = set(pickle.load(open(f"{path}/.metadata", "rb")).state_dict_metadata)
    want = get_model_state_dict(model)
    miss = sorted(set(want) - keys)
    assert not miss, ("init DCP lacks model tensors", miss[:6])
    before = {k: M._loc(v).data_ptr() for k, v in want.items()}
    dcp.load(want, checkpoint_id=path)
    params = {n.replace("_checkpoint_wrapped_module.", ""): p for n, p in model.named_parameters()}
    assert set(params) == set(want), ("state dict != parameters", sorted(set(params) ^ set(want))[:6])
    for k, v in want.items():                                      # get_model_state_dict returns the live parameters: loaded in place
        assert M._loc(params[k]).data_ptr() == M._loc(v).data_ptr() == before[k], ("not in place", k)
    nonfin = torch.tensor([float(sum(int(not torch.isfinite(M._loc(p)).all()) for p in params.values()))], device=dev)
    dist.all_reduce(nonfin)
    assert float(nonfin) == 0, "non-finite init weights"
    step = open(f"{path}/COMPLETE").read().strip()
    return {"step": step, "run_tag": os.path.basename(path), "world": "dcp", "save_time": "?", "format": "dcp"}, \
        sum(v.numel() for v in want.values()) * 2 / 1e9


CK_MOD = 65521


@torch.no_grad()
def init_checksum(model, dev):
    """exact per-tensor checksum of the loaded weights as bf16 bits: (sum of int16 bits, sum over global dim-0 rows of ((row+1) % 65521) x
    row bit-sum, numel), int64, summed over FSDP shards (replicated tensors counted once, from rank 0) -> {name: [s, ws, numel]}."""
    names, vals = [], []
    for n, p in model.named_parameters():
        n = n.replace("_checkpoint_wrapped_module.", "")
        loc = M._loc(p)
        sharded = hasattr(p, "to_local")
        a, r = M._rows(p) if p.dim() else (0, 1)
        if (sharded or RANK == 0) and loc.numel():
            b = loc.detach().to(torch.bfloat16).contiguous().view(torch.int16).to(torch.int64).reshape(max(r, 1) if p.dim() else 1, -1)
            rs = b.sum(1)
            w = (torch.arange(a, a + rs.shape[0], device=dev, dtype=torch.int64) + 1) % CK_MOD
            vals.append(torch.stack([rs.sum(), (rs * w).sum()]))
        else:
            vals.append(torch.zeros(2, dtype=torch.int64, device=dev))
        names.append((n, p.numel()))
    t = torch.stack(vals); dist.all_reduce(t)
    t = t.cpu().tolist()
    return {n: [int(s), int(ws), int(ne)] for (n, ne), (s, ws) in zip(names, t)}


@torch.no_grad()
def load_init(model, path, dev):
    if not os.path.isfile(f"{path}/meta.json"):
        return load_init_dcp(model, path, dev)
    """every parameter <- a ppm-local-bf16-shards-v1 copy (world-W0 dim-0 FSDP2 shards rank<r>.pt, concatenated in rank order = infer_ppm
    load_weights; rep.pt = the replicated decoders), resharded onto this world's FSDP2 shards (each rank reads only the rows it holds)."""
    meta = json.load(open(f"{path}/meta.json"))
    assert meta["format"] == "ppm-local-bf16-shards-v1" and os.path.isfile(f"{path}/COMPLETE"), path
    W0 = int(meta["world"])
    params = {n.replace("_checkpoint_wrapped_module.", ""): p for n, p in model.named_parameters()}
    # global row offsets of every tensor in each source rank file (FSDP2 = torch.chunk along dim 0)
    srcs = {}

    def src(r):
        if r not in srcs:
            srcs[r] = torch.load(f"{path}/rank{r}.pt", map_location="cpu", mmap=True, weights_only=True)
        return srcs[r]
    rep = torch.load(f"{path}/rep.pt", map_location="cpu", mmap=True, weights_only=True)
    done, nbytes = set(), 0
    for n, p in params.items():
        if n in rep:
            p.copy_(rep[n].to(device=dev, dtype=p.dtype)); done.add(n); continue
        R = p.shape[0]
        c0 = -(-R // W0)                                          # source chunk rows (torch.chunk)
        a, r = M._rows(p)
        loc = M._loc(p)
        got = 0
        if r:
            for s in range(a // c0, min(W0, -(-(a + r) // c0))):
                lo, hi = s * c0, min((s + 1) * c0, R)
                t = src(s)[n]
                assert t.shape[0] == hi - lo and tuple(t.shape[1:]) == tuple(p.shape[1:]), (n, s, tuple(t.shape), lo, hi)
                x0, x1 = max(lo, a), min(hi, a + r)
                if x1 > x0:
                    loc[x0 - a:x1 - a].copy_(t[x0 - lo:x1 - lo].to(device=dev, dtype=loc.dtype)); got += x1 - x0
                    nbytes += (x1 - x0) * t[0].numel() * 2
        assert got == r, ("rows not covered", n, got, r)
        done.add(n)
    miss = sorted(set(params) - done)
    assert not miss, ("init: parameters not in the checkpoint", miss[:6])
    srcs.clear()
    nonfin = torch.tensor([float(sum(int(not torch.isfinite(M._loc(p)).all()) for p in params.values()))], device=dev)
    dist.all_reduce(nonfin)
    assert float(nonfin) == 0, "non-finite init weights"
    t = torch.tensor([float(nbytes)], device=dev); dist.all_reduce(t)
    return meta, float(t) / 1e9


def tensor_digest(model, names):
    """sha256 over THIS rank's local values (FSDP shard / replicated tensor) of the named parameters -> hex (compared per rank)."""
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        if n not in names:
            continue
        h.update(n.encode()); h.update(M._loc(p).detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def export_weights(model, step, path, min_free_gb):
    """consolidated bf16 DCP of every model tensor (= safety_ppm.export_weights' format, infer_ppm 'dcp' kind) at `path`; never deletes."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    msd = get_model_state_dict(model)
    size_gb = sum(v.numel() * 2 for v in msd.values()) / 1e9
    ok = [None]
    if RANK == 0:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tw = time.time()
        while True:
            free = shutil.disk_usage(os.path.dirname(path)).free / 1e9
            if free - size_gb >= min_free_gb or time.time() - tw > 60.0 * CFG["export_wait_min"] or os.path.exists(path):
                break
            time.sleep(15.0)
        if time.time() - tw > 1.0:
            log(f"[export] waited {time.time() - tw:.0f}s for /data space ({free:.0f} GB free)")
        ok = [(free - size_gb >= min_free_gb and not os.path.exists(path), free)]
    dist.broadcast_object_list(ok, src=0)
    good, free = ok[0]
    if not good:
        log(f"!!! [export] REFUSED {path}: {size_gb:.1f} GB with {free:.0f} GB free (floor {min_free_gb}) or the path exists")
        return None
    t0 = time.time()
    dcp.save({k: v.to(torch.bfloat16) for k, v in msd.items()}, checkpoint_id=path)
    dist.barrier()
    if RANK == 0:
        json.dump({"step": int(step), "decp_bb": 0, "pose": 0, "nopose_model": 1, "run": CFG["run_tag"], "init": CFG["init"], "cfg": CFG},
                  open(f"{path}/train_cfg.json", "w"), indent=1, default=str)
        open(f"{path}/COMPLETE", "w").write(str(step))
    dist.barrier()
    log(f"[export] {path}: consolidated bf16 weights ({size_gb:.1f} GB, {time.time() - t0:.0f}s); /data free before {free:.0f} GB")
    return path


def export_local(model, step, path):
    """aloe: weights-only NODE-LOCAL export (ppm-local-bf16-shards-v1 = ckpt_local_export's format, world = WORLD): each rank writes its
    FSDP2 dim-0 shards (names without the checkpoint wrapper) to <path>/rank<R>.pt on its own node; rank 0 also writes rep.pt (the
    replicated tensors) + meta.json; COMPLETE on rank 0's node once every rank is done. Nothing on /data."""
    t0 = time.time()
    os.makedirs(path, exist_ok=True)
    sh, rp = {}, {}
    for n, p in model.named_parameters():
        k = n.replace("_checkpoint_wrapped_module.", "")
        if hasattr(p, "to_local"):
            sh[k] = p.to_local().detach().to(torch.bfloat16).cpu().contiguous().clone()
        elif RANK == 0:
            rp[k] = p.detach().to(torch.bfloat16).cpu().contiguous().clone()
    torch.save(sh, f"{path}/rank{RANK}.pt.tmp"); os.replace(f"{path}/rank{RANK}.pt.tmp", f"{path}/rank{RANK}.pt")
    nb = sum(t.numel() * 2 for t in sh.values()) + sum(t.numel() * 2 for t in rp.values())
    hosts = [None] * WORLD; dist.all_gather_object(hosts, HOST)
    if RANK == 0:
        torch.save(rp, f"{path}/rep.pt")
        json.dump({"format": "ppm-local-bf16-shards-v1", "src": "aloe-ft train_aloe export_local", "step": int(step), "world": WORLD, "hosts": hosts,
                   "run_tag": CFG["run_tag"], "init": CFG["init"], "cfg": CFG, "save_time": now_pt()}, open(f"{path}/meta.json", "w"), indent=1, default=str)
    dist.barrier()
    if RANK == 0:
        open(f"{path}/COMPLETE", "w").write(str(step))
    t = torch.tensor([float(nb)], device=next(model.parameters()).device); dist.all_reduce(t)
    log(f"[export-local] {path} on every node: {float(t) / 1e9:.1f} GB bf16 shards (world {WORLD}), {time.time() - t0:.0f}s")


# ------------------------------------------------------------------------------------------------ resolved-config diff vs 21980
CFG_ALLOWED = (                                                   # v2init: vs 22383 ONLY the init path + run dirs may differ
    "init", "run_tag", "log_root", "plan_dir", "control_dir", "export", "local_export_root", "ref_cfg")


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{pre}{k}."))
        else:
            out[f"{pre}{k}"] = json.loads(json.dumps(v, default=str))
    return out


def cfg_diff():
    ref = _flat(json.load(open(CFG["ref_cfg"])))
    cur = _flat(json.loads(json.dumps(CFG, default=str)))
    rows, bad = [], []
    for k in sorted(set(ref) | set(cur)):
        a, b = ref.get(k, "<absent>"), cur.get(k, "<absent>")
        if a != b:
            ok = any(k == x or (x.endswith(".") and k.startswith(x)) or k.startswith(x + ".") for x in CFG_ALLOWED)
            rows.append(f"  {'ok ' if ok else 'BAD'} {k}: 22383 {a} -> this {b}")
            if not ok:
                bad.append(k)
    log("[cfg-diff] resolved config vs 22383 (" + CFG["ref_cfg"] + f"): {len(rows)} keys differ, {len(bad)} outside init + run dirs:\n" + "\n".join(rows))
    assert not bad, ("config differs from 22383 outside init + run dirs", bad)


# ------------------------------------------------------------------------------------------------ main
STOP = {"flag": False}


def _usr1(signum, frame):
    STOP["flag"] = True


def make_opt(model):
    by = {g: [] for g in TRAIN_GROUPS}
    frozen, dec = [], []
    for n, p in model.named_parameters():
        g = group_of(n)
        if g is None:
            frozen.append(n)
        elif g in ("dec_s",):
            dec.append(n)
        else:
            assert g in by, (n, g)
            by[g].append(p)
    opt = torch.optim.AdamW([{"params": by[g], "lr": CFG["lr"][g], "name": g} for g in TRAIN_GROUPS if by[g]], betas=(0.9, CFG["adam_beta2"]),
                            eps=1e-8, weight_decay=0.0, fused=True)
    lam = SCH.make_lambda(True, CFG["sched"])                     # every trained group warms up (run C: warmup_groups = all its groups)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, [lam for _ in opt.param_groups])
    return opt, sched, by, frozen, dec


def main():
    dev = torch.device("cuda", LOCAL)
    torch.cuda.set_device(dev)
    dist.init_process_group("nccl", device_id=dev, timeout=datetime.timedelta(seconds=int(E("INIT_TIMEOUT", "1800"))))
    os.makedirs(LOGDIR, exist_ok=True)
    signal.signal(signal.SIGUSR1, _usr1)
    log(f"aloe-v2-ft {CFG['run_tag']}: world {WORLD} (lw {LW}); host {HOST}; cfg {json.dumps(CFG, default=str)}")
    cfg_diff()
    # ---- data (aloe plan: identical on every rank; every train row x both orientations, one row per rank per step)
    PL = A.make_plan(WORLD)
    assert PL["steps"] == CFG["steps"], ("QQA_STEPS != the plan's steps (one epoch over the doubled rows)", CFG["steps"], PL["steps"])
    for k in range(4):
        A.Stores().open(k)                                        # staged store digests == Model 1's (asserted)
    rows, ev_items = A.rank_items(PL, RANK, WORLD)
    _ri = [None] * WORLD; dist.all_gather_object(_ri, {"rank": RANK, "host": HOST, "step0": rows[0], "step1": rows[1], "eval": ev_items})
    if RANK == 0:
        json.dump({"plan_world": WORLD, "ranks": _ri}, open(f"{LOGDIR}/rank_items.json", "w"))
    fallback = PL["items"][0]
    ndum = sum(x is None for x in rows)
    nconv = allsum([float(len(rows) - ndum), float(len(ev_items)), float(ndum)], dev)
    assert int(nconv[0]) == len(PL["items"]) and int(nconv[1]) == len(PL["eval"]), nconv
    dg = hashlib.sha256(json.dumps(PL["items"]).encode()).hexdigest()[:16]
    dga = [None] * WORLD; dist.all_gather_object(dga, dg)
    assert len(set(dga)) == 1, "plan differs across ranks"
    log(f"[data] v2 first chunks only: train rows {PL['n_first']['train']:,} of {PL['n_all']['train']:,}, eval rows {PL['n_first']['eval']} of "
        f"{PL['n_all']['eval']} (21980's held-out rows that are first chunks)")
    log(f"[data] aloe plan {dg}: {PL['n_train_rows']:,} train rows x 2 orientations = {int(nconv[0]):,} micros in {CFG['steps']} steps x {WORLD} "
        f"ranks (1 row / rank / step; {int(nconv[2])} dummy micros on the last step); eval {PL['n_eval_rows']} rows x 2 = {int(nconv[1])} micros; "
        f"data cfg {CFG['data']}")
    # ---- model
    c32, c4, c06 = M.cfg32(64), M.cfg4b(36), M.cfg06(28)
    model, _ = M.build(c32, c4, c06, dev, log, partial=False, proj_r=2048, tower_vocab=M.TOWER_VOCAB, init=False, dec_ac=False,
                       dec_mode="replicate", pose_head=False, decp_bb=False, nopose=True)
    assert not any(n.replace("_checkpoint_wrapped_module.", "").startswith(("tower_p.", "dec_p.", "pose_head.", "decp_bb.", "inp.pose."))
                   for n, _ in model.named_parameters()), "v2: no pose part may be built"
    t0 = time.time()
    meta, gb = load_init(model, CFG["init"], dev)
    ck = init_checksum(model, dev)
    if RANK == 0:
        json.dump({"init": CFG["init"], "format": meta.get("format", "ppm-local-bf16-shards-v1"), "mod": CK_MOD, "tensors": ck},
                  open(f"{LOGDIR}/init_checksum.json", "w"))
    log(f"[init-checksum] {len(ck)} tensors, total bit-sum {sum(v[0] for v in ck.values())} -> {LOGDIR}/init_checksum.json")
    log(f"[init] {CFG['init']}: PPM step {meta['step']} (run {meta.get('run_tag')}, saved {meta.get('save_time')}), world {meta['world']} shards "
        f"resharded to {WORLD}; {gb:.1f} GB read; {time.time() - t0:.0f}s")
    if CFG["dec_fast"]:
        M.compile_decoder(model.dec_s, CFG["dec_compile_mode"], whole=True, dynamic=CFG["dec_dyn"])
    rot = M.Rot(c32, dev)
    opt, sched, by, frozen, decn = make_opt(model)
    do = DecOpt(model.dec_s)
    n_all = len(list(model.parameters()))
    assert sum(len(v) for v in by.values()) + len(do.ps) + len(frozen) == n_all, "every parameter in exactly one optimizer or frozen"
    fz_set = set(frozen)
    fz0 = tensor_digest(model, fz_set)
    log(f"[opt] main AdamW groups {[(pg['name'], len(pg['params']), pg['lr']) for pg in opt.param_groups]}; decoder nanostep AdamW "
        f"{len(do.ps)} tensors (K {CFG['dec_k']}); FROZEN {len(frozen)} tensors ({sorted({n.split('.')[0] for n in frozen})}) digest {fz0[:16]}; "
        f"schedule {SCH.describe(dict(CFG['sched'], warmup_groups=list(TRAIN_GROUPS)))}; decoder schedule = the same shape over updates "
        f"(warmup {CFG['dec_sched']['warmup']}, decay to 0 at {CFG['dec_sched']['wsd_decay_end']})")
    # ---- loaders + eval set
    from torch.utils.data import DataLoader
    W_ = CFG["dl_workers"]

    def _wi(_):
        torch.set_num_threads(1); os.environ["OMP_NUM_THREADS"] = "1"
    ld = iter(DataLoader(A.AloeDS(rows, CFG["data"], RANK, fallback, singles_at=0 if CFG["losscheck"] else -1), batch_size=None,
                         shuffle=False, num_workers=W_, prefetch_factor=2 if W_ else None, persistent_workers=bool(W_), pin_memory=False,
                         worker_init_fn=_wi if W_ else None, collate_fn=lambda x: x))
    ev = A.eval_micros(ev_items, CFG["eval_data"], RANK)
    evc = allsum([float(sum(int(m["voice_clip"]) for m in ev)), float(sum(int(m["voice_drawn"]) for m in ev)), float(len(ev))], dev)
    log(f"[eval] v2 held-out voice clips: drawn {int(evc[1])} / {int(evc[2])}, realised {int(evc[0])} / {int(evc[2])} = {evc[0] / max(evc[2], 1):.4f}")
    t_ = torch.tensor([float(len(ev))], device=dev); dist.all_reduce(t_, op=dist.ReduceOp.MAX); n_ev = int(t_)
    ev += [dict(ev[0], dummy=1, own=np.zeros_like(ev[0]["own"])) for _ in range(n_ev - len(ev))]   # same forwards on every rank (FSDP)
    log(f"[eval] {n_ev} eval micros per rank (dummies pad the short ranks); P mean {np.mean([int(m['P'][0]) for m in ev]):.0f}")
    jl = open(f"{LOGDIR}/metrics.jsonl", "a") if RANK == 0 else None

    def emit(d, step):
        if RANK == 0:
            jl.write(json.dumps({"step": step, "time_pt": now_pt(), **d}) + "\n"); jl.flush()

    def evaluate(step):
        model.eval()
        t0_ = time.time()
        acc = {0: None, 1: None}
        for m in ev:
            oc = np.asarray(m["own"]).sum(0)
            v, _, _ = run_micro(model, prep(m, dev, max(oc[0], 1.0), max(oc[1], 1.0)), rot, do, grad=False)
            if m.get("dummy"):
                v = [0.0] * len(v)
            sw = int(m["swap"])
            acc[sw] = v if acc[sw] is None else [x + y for x, y in zip(acc[sw], v)]
        nv = len(next(a for a in acc.values() if a is not None))
        acc = {k: (a if a is not None else [0.0] * nv) for k, a in acc.items()}
        g0, g1 = allsum(acc[0], dev), allsum(acc[1], dev)
        d = summarize([x + y for x, y in zip(g0, g1)], "eval")
        d.update(summarize(g0, "eval_asstB")); d.update(summarize(g1, "eval_userB"))
        model.train()
        d["eval/time_s"] = time.time() - t0_
        emit(d, step)
        log(f"EVAL step {step}: " + " | ".join(f"{k} {v:.4f}" for k, v in d.items()))
        for k_, v_ in d.items():
            assert math.isfinite(v_), (k_, v_)
        return d

    evaluate(0)
    import gc
    gc.collect(); gc.freeze(); gc.disable()
    times = []
    stopped = None
    vc_cum = [0.0, 0.0, 0.0]                                      # v2: realised voice clips, drawn, real micros (cumulative over the run)
    for step in range(CFG["steps"]):
        torch.cuda.synchronize(); ts = time.time()
        torch.cuda.reset_peak_memory_stats()
        tw = time.time()
        b = next(ld)
        wait = time.time() - tw
        assert int(b["step"]) == step, ("packed micro order", b["step"], step)
        if step == 0 and CFG["losscheck"]:
            Pv, ex, er, K = pack_loss_check(model, b, rot, dev)
            ea = [None] * WORLD; dist.all_gather_object(ea, er)
            worst = [max(e[i] for e in ea) for i in range(3)]
            log(f"[pack] LOSS CHECK step 0: packed micro loss (micro token-normalised) vs token-weighted standalone conversations, max rel err over "
                f"{WORLD} ranks: text {worst[0]:.2e} cb0 {worst[1]:.2e} light {worst[2]:.2e} (rank 0: {K} conversations, packed {[round(x, 5) for x in Pv]}, "
                f"weighted standalone {[round(x, 5) for x in ex]})")
            assert max(worst) < CFG["losscheck_tol"], ("packed loss != token-weighted standalone losses", worst)
        oc = np.asarray(b["own"]).sum(0)
        tot = allsum([float(oc[0]), float(oc[1]), float(oc[2])], dev)
        bt = prep(b, dev, tot[0] / WORLD, tot[1] / WORLD)                 # tokens_step normalisers
        if step == 0:
            log(f"[pack] tokens_step: step 0 totals over {WORLD} micros: text targets {int(tot[0])}, cb0 labels {int(tot[1])}, light frames "
                f"{int(tot[2])}; per-micro normaliser = total / {WORLD}")
        t_m = time.time()
        v, loss_v, ex_ = run_micro(model, bt, rot, do, grad=True)
        torch.cuda.synchronize(); t_micro = time.time() - t_m
        assert abs(ex_["dec_N"] - tot[2]) < 0.5, ("decoder frames != the step's light frames", ex_["dec_N"], tot[2])
        lr_ratio = tot[2] / max(tot[1], 1.0)                      # v2 (light_frac 0.1): the selected frames = ~10 % of B's coded (cb0) frames
        assert CFG["light_ratio_lo"] <= lr_ratio <= CFG["light_ratio_hi"], ("light frames / B coded frames outside the 10 % band", lr_ratio, tot)
        vc = allsum([float(b.get("voice_clip", 0)), float(b.get("voice_drawn", 0)), 0.0 if b.get("dummy", 0) else 1.0], dev)
        vc_cum = [x + y for x, y in zip(vc_cum, vc)]
        for n_, p_ in model.named_parameters():
            if n_ in fz_set:
                assert p_.grad is None, ("a frozen parameter got a gradient", n_)
        gn = {}
        for pg in opt.param_groups:
            g_ = torch.nn.utils.clip_grad_norm_(pg["params"], CFG["clip"][pg["name"]])
            gn[pg["name"]] = float(g_.full_tensor() if hasattr(g_, "full_tensor") else g_)
        lr_used = {pg["name"]: pg["lr"] for pg in opt.param_groups}
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); dt = time.time() - ts; times.append(dt)
        sv = allsum(list(v) + [loss_v], dev)
        d = summarize(sv[:-1], "train")
        d["train/loss"] = sv[-1] / WORLD
        d["train/dummies"] = float(nconv[2]) if step == CFG["steps"] - 1 else 0.0
        pk = torch.tensor([torch.cuda.max_memory_allocated() / GiB, dt, wait, t_micro], device=dev)
        dist.all_reduce(pk, op=dist.ReduceOp.MAX)
        d.update({f"train/gn_{k}": x for k, x in gn.items()})
        d.update({f"train/lr_{k}": x for k, x in lr_used.items()})
        d.update({"train/dec_lr_first": ex_["dec_lrs"][0], "train/dec_lr_last": ex_["dec_lrs"][-1], "train/step_time": dt, "train/peak_gib": float(pk[0]),
                  "train/data_wait": float(pk[2]), "train/micro_s": float(pk[3]), "train/text_targets": tot[0], "train/cb0_labels": tot[1],
                  "train/light_frames": tot[2], "train/light_ratio": lr_ratio, "train/voice_clip_frac_cum": vc_cum[0] / max(vc_cum[2], 1.0),
                  "train/voice_drawn_frac_cum": vc_cum[1] / max(vc_cum[2], 1.0)})
        for k_, v_ in d.items():
            assert math.isfinite(v_), (k_, v_, "non-finite: fail loudly")
        done = step + 1
        emit(d, done)
        eta = (CFG["steps"] - done) * float(np.mean(times[-20:]))
        log(f"step {done}/{CFG['steps']} {dt:.1f}s (micro {float(pk[3]):.1f} wait {float(pk[2]):.1f}) peak {float(pk[0]):.1f} GiB | "
            + " ".join(f"{k.split('/', 1)[1]}={v:.4f}" for k, v in d.items() if k.startswith("train_speech/") and k.endswith(("word_ce_norm", "cb0_ce", "light_ce", "pad_ce")))
            + " " + " ".join(f"{k.split('/')[1]}={v:.4g}" for k, v in d.items() if "/gn_" in k or "/lr_trunk" in k or "dec_lr_last" in k)
            + f" | light {lr_ratio:.4f} clip {vc_cum[0] / max(vc_cum[2], 1.0):.4f} | ETA {eta / 60:.1f} min")
        gc.collect(1)
        if (CFG["eval_every"] and done % CFG["eval_every"] == 0) or done == CFG["steps"] or done in CFG["export_steps"]:
            evaluate(done)
        if done in CFG["export_steps"] and done < CFG["steps"]:
            export_local(model, done, f"{CFG['local_export_root']}/export-{done}")
        st_ = torch.tensor([float(STOP["flag"] or os.path.exists(f"{CFG['control_dir']}/STOP_AFTER_SAVE"))], device=dev)
        dist.all_reduce(st_, op=dist.ReduceOp.MAX)
        if float(st_) > 0 and done < CFG["steps"]:
            log(f"[control] STOP_AFTER_SAVE at step {done}: weights-only export, then exit")
            export_weights(model, done, f"{CFG['export']}-stop{done}", CFG["export_min_free_gb"])
            stopped = done
            break
    fz1 = tensor_digest(model, fz_set)
    same = torch.tensor([float(fz1 == fz0)], device=dev); dist.all_reduce(same, op=dist.ReduceOp.MIN)
    log(f"[frozen] nothing frozen (v2 nopose: no pose parts built); empty-set digest equal on every rank: {bool(float(same))}")
    assert float(same) == 1.0, "a frozen parameter changed"
    if stopped is None:
        export_local(model, CFG["steps"], f"{CFG['local_export_root']}/export-{CFG['steps']}")   # aloe: node-local copy first (never refused)
        export_weights(model, CFG["steps"], CFG["export"], CFG["export_min_free_gb"])
    log(f"done {stopped or CFG['steps']} steps; mean step {np.mean(times):.2f}s; step times {[round(x, 1) for x in times]}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
