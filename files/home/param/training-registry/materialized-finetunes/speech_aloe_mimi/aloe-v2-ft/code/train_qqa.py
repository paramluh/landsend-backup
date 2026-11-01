"""qqa-ft TRAINER: run C's speech-duplex quick-QA fine-tune (s4-qqa job 21169) on a PPM checkpoint, in PPM's own model / speech-row maths.
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
    "dl_workers": int(E("QQA_DL_WORKERS", "3")), "data": Q.qqa_cfg(), "cd": D.CD,
}
CFG["sched"] = {"warmup": CFG["warmup"], "horizon": CFG["steps"], "min_lr_rate": 0.0, "s3_sched": "wsd", "wsd_decay_start": CFG["warmup"],
                "wsd_decay_end": CFG["steps"], "wsd_floor": 0.0, "wsd_shape": "linear"}
CFG["dec_sched"] = {"warmup": CFG["warmup"] * CFG["dec_k"], "horizon": CFG["steps"] * CFG["dec_k"], "min_lr_rate": 0.0, "s3_sched": "wsd",
                    "wsd_decay_start": CFG["warmup"] * CFG["dec_k"], "wsd_decay_end": CFG["steps"] * CFG["dec_k"], "wsd_floor": 0.0, "wsd_shape": "linear"}
TRAIN_GROUPS = ("trunk", "spec", "proj", "audio", "tower_s")
FROZEN_PREFIX = ("tower_p.", "dec_p.", "pose_head.", "decp_bb.", "inp.pose.")
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


def nanosteps_tw(dec, do, hv, depth, light):
    """train_ppm.nanosteps (replicated, one decoder) with tokens_step weights: chunk j's frames weigh 1 / (31 x M_j / WORLD) (M_j = chunk-j
    frames summed over ranks) -> after the all-reduce mean the update's gradient = the mean CE over ALL ranks' chunk-j frames; the input
    gradient accumulates chunk gradients x M_j / N (N = all ranks' light frames) = d(step token mean x WORLD / ...)/d hv, i.e. the tower /
    backbone receive the token-weighted decoder loss. -> (acc, stats, lrs, (M, N))."""
    K, dev = CFG["dec_k"], hv.device
    hd = hv.detach().requires_grad_()
    n = int(light.numel())
    perm = torch.randperm(max(n, 1), generator=torch.Generator().manual_seed(CFG["dec_seed"] * 1000003 + do.micros * 1009 + RANK))
    sizes = [int(perm[j::K].numel()) if n else 0 for j in range(K)]
    Mg = torch.tensor(sizes, dtype=torch.float64, device=dev)
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
            if mv:
                wf[:mv] = 1.0 / den
            l_, s_, num_, cb_ = dec(hd, depth, fl, wf, n_valid=mv)
            if mv:
                st["sum"] += s_; st["n"] += num_; st["cb"] += cb_.detach().float()
            l_.backward()
        if hd.grad is not None and m:
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
            up, acc, lrs, (Mj, N) = nanosteps_tw(model.dec_s, do, out["hv_s"], b["depth"][1], b["light"])
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
@torch.no_grad()
def load_init(model, path, dev):
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
        json.dump({"step": int(step), "decp_bb": 1, "decp_bb_alpha": 1.0, "run": CFG["run_tag"], "init": CFG["init"], "cfg": CFG},
                  open(f"{path}/train_cfg.json", "w"), indent=1, default=str)
        open(f"{path}/COMPLETE", "w").write(str(step))
    dist.barrier()
    log(f"[export] {path}: consolidated bf16 weights ({size_gb:.1f} GB, {time.time() - t0:.0f}s); /data free before {free:.0f} GB")
    return path


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
    log(f"qqa-ft {CFG['run_tag']}: world {WORLD} (lw {LW}); host {HOST}; cfg {json.dumps(CFG, default=str)}")
    # ---- data (run C's plan for this host / local rank)
    PL = json.load(open(f"{CFG['plan_dir']}/PLAN-{HOST}.json"))
    assert PL["format"] == "ppm-qqa-plan-v1" and PL["lw"] == LW and PL["steps"] >= CFG["steps"], (PL["format"], PL["lw"], PL["steps"])
    node = D.R3.load_node(PL["store"])
    assert node["node"] == HOST and node["digest"] == PL["node_digest"], ("store != the plan's", node["node"], node["digest"])
    rows = PL["ranks"][str(LOCAL)][:CFG["steps"]]
    ev_rows = PL["eval"][str(LOCAL)]
    nconv = allsum([float(sum(len(r) for r in rows)), float(len(ev_rows))], dev)
    log(f"[data] run C's plan: {int(nconv[0]):,} conversations in {CFG['steps']} steps x {WORLD} packed micros ({nconv[0] / CFG['steps'] / WORLD:.2f} "
        f"conversations / micro), {int(nconv[1])} held-out eval conversations (4 / rank, unpacked); data cfg {CFG['data']}")
    # ---- model
    c32, c4, c06 = M.cfg32(64), M.cfg4b(36), M.cfg06(28)
    model, _ = M.build(c32, c4, c06, dev, log, partial=False, proj_r=2048, tower_vocab=M.TOWER_VOCAB, init=False, dec_ac=False,
                       dec_mode="replicate", pose_head=True, decp_bb=True)
    model.pose_head_on = False                                    # pose parts never run (speech micros only)
    t0 = time.time()
    meta, gb = load_init(model, CFG["init"], dev)
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
    ld = iter(DataLoader(Q.PackDS(PL["store"], rows, CFG["data"], RANK, 0, singles_at=0 if CFG["losscheck"] else -1), batch_size=None,
                         shuffle=False, num_workers=W_, prefetch_factor=2 if W_ else None, persistent_workers=bool(W_), pin_memory=False,
                         worker_init_fn=_wi if W_ else None, collate_fn=lambda x: x))
    ev = Q.eval_micros(PL["store"], ev_rows, CFG["data"], RANK)
    jl = open(f"{LOGDIR}/metrics.jsonl", "a") if RANK == 0 else None

    def emit(d, step):
        if RANK == 0:
            jl.write(json.dumps({"step": step, "time_pt": now_pt(), **d}) + "\n"); jl.flush()

    def evaluate(step):
        model.eval()
        t0_ = time.time()
        acc = None
        for m in ev:
            oc = Q.own_counts(m)
            v, _, _ = run_micro(model, prep(m, dev, oc[0], oc[1]), rot, do, grad=False)
            acc = v if acc is None else [x + y for x, y in zip(acc, v)]
        d = summarize(allsum(acc, dev), "eval")
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
        v, _, ex_ = run_micro(model, bt, rot, do, grad=True)
        torch.cuda.synchronize(); t_micro = time.time() - t_m
        assert abs(ex_["dec_N"] - tot[2]) < 0.5, ("decoder frames != the step's light frames", ex_["dec_N"], tot[2])
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
        d = summarize(allsum(v, dev), "train")
        pk = torch.tensor([torch.cuda.max_memory_allocated() / GiB, dt, wait, t_micro], device=dev)
        dist.all_reduce(pk, op=dist.ReduceOp.MAX)
        d.update({f"train/gn_{k}": x for k, x in gn.items()})
        d.update({f"train/lr_{k}": x for k, x in lr_used.items()})
        d.update({"train/dec_lr_first": ex_["dec_lrs"][0], "train/dec_lr_last": ex_["dec_lrs"][-1], "train/step_time": dt, "train/peak_gib": float(pk[0]),
                  "train/data_wait": float(pk[2]), "train/micro_s": float(pk[3]), "train/text_targets": tot[0], "train/cb0_labels": tot[1],
                  "train/light_frames": tot[2]})
        for k_, v_ in d.items():
            assert math.isfinite(v_), (k_, v_, "non-finite: fail loudly")
        done = step + 1
        emit(d, done)
        eta = (CFG["steps"] - done) * float(np.mean(times[-20:]))
        log(f"step {done}/{CFG['steps']} {dt:.1f}s (micro {float(pk[3]):.1f} wait {float(pk[2]):.1f}) peak {float(pk[0]):.1f} GiB | "
            + " ".join(f"{k.split('/', 1)[1]}={v:.4f}" for k, v in d.items() if k.startswith("train_speech/") and k.endswith(("word_ce_norm", "cb0_ce", "light_ce", "pad_ce")))
            + " " + " ".join(f"{k.split('/')[1]}={v:.4g}" for k, v in d.items() if "/gn_" in k or "/lr_trunk" in k or "dec_lr_last" in k)
            + f" | ETA {eta / 60:.1f} min")
        gc.collect(1)
        if (CFG["eval_every"] and done % CFG["eval_every"] == 0) or done == CFG["steps"]:
            evaluate(done)
        if done in CFG["export_steps"] and done < CFG["steps"]:
            export_weights(model, done, f"{os.path.dirname(CFG['export'])}/export-{done}", CFG["export_min_free_gb"])
        st_ = torch.tensor([float(STOP["flag"] or os.path.exists(f"{CFG['control_dir']}/STOP_AFTER_SAVE"))], device=dev)
        dist.all_reduce(st_, op=dist.ReduceOp.MAX)
        if float(st_) > 0 and done < CFG["steps"]:
            log(f"[control] STOP_AFTER_SAVE at step {done}: weights-only export, then exit")
            export_weights(model, done, f"{CFG['export']}-stop{done}", CFG["export_min_free_gb"])
            stopped = done
            break
    fz1 = tensor_digest(model, fz_set)
    same = torch.tensor([float(fz1 == fz0)], device=dev); dist.all_reduce(same, op=dist.ReduceOp.MIN)
    log(f"[frozen] pose parts bit-identical to the init on every rank: {bool(float(same))} (rank 0 {fz1[:16]})")
    assert float(same) == 1.0, "a frozen parameter changed"
    if stopped is None:
        export_weights(model, CFG["steps"], CFG["export"], CFG["export_min_free_gb"])
    log(f"done {stopped or CFG['steps']} steps; mean step {np.mean(times):.2f}s; step times {[round(x, 1) for x in times]}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
