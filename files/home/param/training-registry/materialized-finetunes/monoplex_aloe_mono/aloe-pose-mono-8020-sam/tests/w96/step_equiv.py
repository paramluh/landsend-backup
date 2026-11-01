"""w96 STEP-EQUIVALENCE TEST (GPU; aloe-pose-mono-8020-sam, 10-26). The question: does world 96 (64 real rows + 32 zero-weight dummy ranks
per step) give the SAME update as world 64 (one row per rank)? Small-scale analogue: world W with rows_per_step = W (every rank real) vs world
W' > W with the same rows_per_step = W (ranks W..W'-1 dummies at every step), same seed / data / init, the trainer's OWN code:
  data      data_aloe_pose.make_plan_v2 (rows digest asserted == 3681cc26ea97cad7: the real plan) + StepDS (steps 0, 1 = the plan's first
            2 x W rows) + eval_micros_v2 (the first EQ_NEVAL eval micros, round-robin over ranks, dummies pad)
  model     model_ppm.build (FSDP2, bf16 compute / fp32 masters / fp32 reduce, activation checkpointing, replicated decoders, decp_bb link,
            pose head not built) at FULL WIDTH, reduced depth EQ_LAYERS / EQ_TOWER_LAYERS / EQ_DEC_LAYERS, initialised from the run's init
            (PPM local-11488, world-128 bf16 shards resharded to this world by a copy of train_aloe_pose.load_init: only the unloaded-key
            assertion relaxed for the deeper layers)
  step      = train_aloe_pose.main's step body line for line: tokens_step normalisers (step totals / NORM_W = WORLD unless
            ALOE_NORM_WORLD is set; then also the FSDP divide factor, main's set_norm_world), train_step_multi (pass A,
            joint decoder nanosteps K 8, pass B + backward), per-group clip_grad_norm_ (gn), fused AdamW + LambdaLR warmup / decay
            (QQA_STEPS / QQA_WARMUP = the run's 357 / 12, so step 1's main LR is 0 exactly as in job 24272 and step 2's is peak / 12),
            stats allsum + summarize; step 0 runs pack_loss_check (dummy ranks via the env-gated ALOE_LOSSCHECK_DUMMY=1 fix)
  memory    the shared test GPUs host idle demo workers: the vocab-sized tables inp.emb.{0,1}, head.lm_head, tower_{s,p}.text_emb and the
            channel-A audio table inp.aud.0 (its gradient is exactly 0 in this run: channel A empty, present_A = 0) are FROZEN in BOTH
            layouts (requires_grad False: no grad / no AdamW state); this process is capped with set_per_process_memory_fraction.
Every repeat (EQ_REPEATS, same process) restores the init weights and builds a fresh optimizer / scheduler / decoder AdamW (noise floor:
FA3's backward and the decoder kernels are not bit-deterministic). Recorded per repeat (rank 0 -> torch.save EQ_OUT/<tag>_r<k>.pt):
  eval sums, loss-check numbers, per step: summarize dict + train/loss, gn per group, decoder M_j / N / LRs, and FINGERPRINTS of
  (a) every trainable main parameter's gradient before clipping, (b) every parameter after the optimizer step, (c) both decoders after the
  step: per tensor float64 sum / sum of squares over the GLOBAL tensor and an fp32 subsample at the global flat indices i = off (mod 997)
  (off = crc32(name) % 997): world-invariant, so W and W' runs compare element for element."""
import json
import os
import sys
import time
import zlib

import numpy as np
import torch
import torch.distributed as dist

CODE = os.environ.get("EQ_CODE", "/data/home/amartyavarma/ppm/aloe-pose-mono-8020-sam/code")
sys.path.insert(0, CODE)
import train_aloe_pose as TR  # noqa: E402  (CFG from the env; prep / train_step_multi / DecOpt / make_opt / summarize / pack_loss_check)

M, A = TR.M, TR.A
RANK, WORLD, LOCAL = TR.RANK, TR.WORLD, TR.LOCAL
CFG = TR.CFG
GiB = 2 ** 30
EQ_L, EQ_L4, EQ_L06 = int(os.environ.get("EQ_LAYERS", "1")), int(os.environ.get("EQ_TOWER_LAYERS", "1")), int(os.environ.get("EQ_DEC_LAYERS", "1"))
EQ_STEPS, EQ_REPEATS, EQ_NEVAL = int(os.environ.get("EQ_STEPS", "2")), int(os.environ.get("EQ_REPEATS", "2")), int(os.environ.get("EQ_NEVAL", "8"))
EQ_CAP_GB, EQ_KEEP_GB = float(os.environ.get("EQ_CAP_GB", "36")), float(os.environ.get("EQ_KEEP_GB", "12"))
OUT, TAG = os.environ["EQ_OUT"], os.environ["EQ_TAG"]
STRIDE = 997
FREEZE = ("inp.emb.0.weight", "inp.emb.1.weight", "inp.aud.0.weight", "head.lm_head.weight", "tower_s.text_emb.weight", "tower_p.text_emb.weight")


def log(*a):
    if RANK == 0:
        print(f"[{TR.now_pt()}] [{TAG}]", *a, flush=True)


def pname(n):
    return n.replace("_checkpoint_wrapped_module.", "")


@torch.no_grad()
def load_init_partial(model, path, dev):
    """= train_aloe_pose.load_init (same resharding of the world-W0 dim-0 shards onto this world's FSDP2 shards) for a REDUCED-depth model:
    the checkpoint's deeper layers (backbone / towers / decoders) are not loaded (asserted to be the only unloaded keys besides pose_head)."""
    import re
    meta = json.load(open(f"{path}/meta.json"))
    assert meta["format"] == "ppm-local-bf16-shards-v1" and os.path.isfile(f"{path}/COMPLETE"), path
    W0 = int(meta["world"])
    params = {pname(n): p for n, p in model.named_parameters()}
    srcs = {}

    def src(r):
        if r not in srcs:
            srcs[r] = torch.load(f"{path}/rank{r}.pt", map_location="cpu", mmap=True, weights_only=True)
        return srcs[r]
    rep = torch.load(f"{path}/rep.pt", map_location="cpu", mmap=True, weights_only=True)
    done = set()
    for n, p in params.items():
        if n in rep:
            p.copy_(rep[n].to(device=dev, dtype=p.dtype)); done.add(n); continue
        R = p.shape[0]
        c0 = -(-R // W0)
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
        assert got == r, ("rows not covered", n, got, r)
        done.add(n)
    assert not sorted(set(params) - done)
    unl = sorted((set(rep) | set(src(0))) - set(params))
    deep = re.compile(r"^(layers|tower_[sp]\.layers|dec_[sp]\.model\.layers)\.(\d+)\.")
    lim = {"layers": EQ_L, "tower_s.layers": EQ_L4, "tower_p.layers": EQ_L4, "dec_s.model.layers": EQ_L06, "dec_p.model.layers": EQ_L06}
    odd = [k for k in unl if not k.startswith("pose_head.") and not ((m := deep.match(k)) and int(m.group(2)) >= lim[m.group(1)])]
    assert not odd, ("unloaded checkpoint keys other than pose_head / deeper layers", odd[:8])
    srcs.clear()
    return meta, len(unl)


def fingerprint(model, names, kind, dos=None):
    """world-invariant fingerprint (module doc). kind 'grad' (main params' gradients, sharded), 'param' (main params), 'dec' (decoders)."""
    loc_out, sums = {}, []
    if kind == "dec":
        items = [(f"{nm}.{k}", v) for nm, d in (("dec_s", model.dec_s), ("dec_p", model.dec_p)) for k, v in d.state_dict().items()]
    else:
        items = [(pname(n), p) for n, p in model.named_parameters() if pname(n) in names]
    for n, p in items:
        if kind == "dec":
            x, g0, mine = p.detach(), 0, RANK == 0                 # replicated: rank 0's copy
        else:
            t = p.grad if kind == "grad" else p
            if t is None:
                x, g0, mine = None, 0, False
            else:
                x = M._loc(t).detach()
                a, _ = M._rows(p)
                g0, mine = a * int(np.prod(p.shape[1:])), True
        if x is None or not mine or x.numel() == 0:
            sums.append([0.0, 0.0]); continue
        x = x.float().reshape(-1)
        sums.append([float(x.double().sum()), float((x.double() ** 2).sum())])
        off = zlib.crc32(n.encode()) % STRIDE
        st = (off - g0) % STRIDE
        idx = torch.arange(st, max(st, x.numel()), STRIDE, device=x.device)
        loc_out[n] = (np.arange(st, x.numel(), STRIDE, dtype=np.int64) + g0, x.index_select(0, idx).cpu().numpy())
    s = torch.tensor(sums, dtype=torch.float64, device=f"cuda:{LOCAL}")
    dist.all_reduce(s)
    allp = [None] * WORLD if RANK == 0 else None
    dist.gather_object(loc_out, allp, dst=0)
    if RANK != 0:
        return None
    out = {}
    for i, (n, _) in enumerate(items):
        parts = [d[n] for d in allp if n in d]
        if parts:
            gi = np.concatenate([q[0] for q in parts]); v = np.concatenate([q[1] for q in parts])
            o = np.argsort(gi, kind="stable")
            gi, v = gi[o], v[o]
        else:
            gi, v = np.zeros(0, np.int64), np.zeros(0, np.float32)
        out[n] = {"sum": float(s[i, 0]), "sumsq": float(s[i, 1]), "idx": gi, "val": v}
    return out


def main():
    dev = torch.device("cuda", LOCAL)
    torch.cuda.set_device(dev)
    free, total = torch.cuda.mem_get_info(dev)
    cap = min(EQ_CAP_GB * GiB, free - EQ_KEEP_GB * GiB)
    assert cap >= 20 * GiB, ("not enough free GPU memory on this shared GPU", free / GiB, cap / GiB)
    torch.cuda.set_per_process_memory_fraction(cap / total, dev)
    dist.init_process_group("nccl", device_id=dev)
    caps = [None] * WORLD; dist.all_gather_object(caps, (os.environ.get("CUDA_VISIBLE_DEVICES"), LOCAL, round(free / GiB, 1), round(cap / GiB, 1)))
    log(f"world {WORLD}; rows/step {A.ROWS_PER_STEP}; layers {EQ_L} / towers {EQ_L4} / decoders {EQ_L06}; steps {EQ_STEPS}; repeats {EQ_REPEATS}; "
        f"GPU (visible, local, free GiB, cap GiB) per rank {caps}; micro_layout {CFG.get('micro_layout')}")
    t0 = time.time()
    PL = A.make_plan_v2(WORLD, CFG["data"])
    rdg = A.rows_digest(PL)
    assert rdg == os.environ["ALOE_ROWS_DIGEST"], ("rows digest", rdg)
    my_steps = [PL["step_micros"][t][RANK] for t in range(EQ_STEPS)]
    ds = A.StepDS(PL["items"], my_steps, CFG["data"], RANK, PL["items"][0], start=0, singles_at=0 if CFG["losscheck"] else -1)
    bls = [ds[t] for t in range(EQ_STEPS)]
    roles = [None] * WORLD
    dist.all_gather_object(roles, [("dummy" if b_[0].get("dummy") else b_[0]["item"]) for b_ in bls])
    ev_groups = PL["eval_micros"][:EQ_NEVAL][RANK::WORLD]
    ev = A.eval_micros_v2(PL["eval_items"], ev_groups, CFG["data"], RANK) if ev_groups else []
    t_ = torch.tensor([float(len(ev))], device=dev); dist.all_reduce(t_, op=dist.ReduceOp.MAX); n_ev = int(t_)
    fb_ev = ev[0] if ev else A.eval_micros_v2(PL["eval_items"], [PL["eval_micros"][0]], CFG["data"], RANK)[0]
    ev += [dict(fb_ev, dummy=1, own=np.zeros_like(fb_ev["own"])) for _ in range(n_ev - len(ev))]
    log(f"plan rows digest {rdg} (== the run's); plan {A.plan_digest(PL)}; data in {time.time() - t0:.0f}s; per-rank step items "
        f"{roles}; eval micros {EQ_NEVAL} -> {n_ev} per rank")
    # ---- model (= main's build / init / compile, reduced depth)
    c32, c4, c06 = M.cfg32(EQ_L), M.cfg4b(EQ_L4), M.cfg06(EQ_L06)
    model, _ = M.build(c32, c4, c06, dev, log, partial=True, proj_r=2048, tower_vocab=M.TOWER_VOCAB, init=False, dec_ac=False,
                       dec_mode="replicate", pose_head=CFG["pose_head_build"], decp_bb=True)
    model.decp_bb_alpha = CFG["decp_bb_alpha"]
    n_norm = TR.set_norm_world(model)                             # = main (ALOE_NORM_WORLD; default no-op)
    log(f"NORM_W {TR.NORM_W} (world {WORLD}); FSDP units with divide factor set: {n_norm}")
    t0 = time.time()
    meta, nunl = load_init_partial(model, CFG["init"], dev)
    log(f"init {CFG['init']} (PPM step {meta['step']}, world {meta['world']} -> {WORLD}) in {time.time() - t0:.0f}s; {nunl} deeper / pose-head keys not loaded")
    for n, p in model.named_parameters():
        if pname(n) in FREEZE:
            p.requires_grad_(False)
    if CFG["dec_fast"]:
        for dec_ in (model.dec_s, model.dec_p):
            M.compile_decoder(dec_, CFG["dec_compile_mode"], whole=True, dynamic=CFG["dec_dyn"])
    rot = M.Rot(c32, dev)
    names_tr = {pname(n) for n, p in model.named_parameters() if p.requires_grad and not pname(n).startswith(("dec_s.", "dec_p."))}
    names_all = {pname(n) for n, p in model.named_parameters() if not pname(n).startswith(("dec_s.", "dec_p."))}
    fp_init = fingerprint(model, names_tr, "param")
    fp_dec0 = fingerprint(model, None, "dec")
    for rep_k in range(EQ_REPEATS):
        if rep_k:                                                 # restore the init by re-reading it (host memory on the shared node is tight)
            for n, p in model.named_parameters():
                p.grad = None
            load_init_partial(model, CFG["init"], dev)
            f1, f2 = fingerprint(model, names_tr, "param"), fingerprint(model, None, "dec")
            if RANK == 0:
                same = all(f1[n]["sum"] == fp_init[n]["sum"] and np.array_equal(f1[n]["val"], fp_init[n]["val"]) for n in fp_init) and \
                    all(f2[n]["sum"] == fp_dec0[n]["sum"] and np.array_equal(f2[n]["val"], fp_dec0[n]["val"]) for n in fp_dec0)
                log(f"r{rep_k} init restored by re-reading the init: identical to repeat 0's init fingerprint: {same}")
                assert same
            del f1, f2
        torch.cuda.reset_peak_memory_stats(dev)
        opt, sched, by, frozen, decn = TR.make_opt(model)
        dos = {"dec_s": TR.DecOpt(model.dec_s, CFG["dec_lr"], CFG["dec_clip"]), "dec_p": TR.DecOpt(model.dec_p, CFG["dec_p_lr"], CFG["dec_p_clip"])}
        rec = {"tag": TAG, "repeat": rep_k, "world": WORLD, "norm_w": TR.NORM_W, "rows_per_step": A.ROWS_PER_STEP, "layers": [EQ_L, EQ_L4, EQ_L06], "roles": roles,
               "frozen": list(FREEZE), "steps": []}
        if rep_k == 0:
            rec["fp_init"], rec["fp_dec0"] = fp_init, fp_dec0
        # ---- eval at step 0 (= main's evaluate over the eval subset)
        model.eval()
        acc = {0: None, 1: None}
        for m in ev:
            oc = np.asarray(m["own"]).sum(0)
            v, _, _ = TR.run_micro(model, TR.prep(m, dev, max(oc[0], 1.0), max(oc[1], 1.0), max(oc[3], 1.0)), rot, dos, grad=False)
            if m.get("dummy"):
                v = [0.0] * len(v)
            sw = int(m["swap"])
            acc[sw] = v if acc[sw] is None else [x + y for x, y in zip(acc[sw], v)]
        nv = len(next(a for a in acc.values() if a is not None))
        acc = {k: (a if a is not None else [0.0] * nv) for k, a in acc.items()}
        g0_, g1_ = TR.allsum(acc[0], dev), TR.allsum(acc[1], dev)
        model.train()
        rec["eval_raw"] = [g0_, g1_]
        rec["eval"] = TR.summarize([x + y for x, y in zip(g0_, g1_)], "eval")
        log(f"r{rep_k} EVAL step 0 ({EQ_NEVAL} micros): " + " | ".join(f"{k} {v:.6f}" for k, v in rec["eval"].items() if k.endswith(("word_ce_norm", "cb0_ce", "light_ce", "pose_cb0_ce", "pose_light_ce"))))
        # ---- steps (= main's loop body)
        for step in range(EQ_STEPS):
            torch.cuda.synchronize(); ts = time.time()
            bl = bls[step]
            b = bl[0]
            if step == 0 and CFG["losscheck"]:
                Pv, ex, er, K = TR.pack_loss_check(model, b, rot, dev)
                ea = [None] * WORLD; dist.all_gather_object(ea, er)
                worst = [max(e[i] for e in ea) for i in range(len(er))]
                rec["losscheck"] = {"worst": worst, "rank0_packed": Pv, "rank0_standalone": ex, "rank0_K": K}
                log(f"r{rep_k} [pack] LOSS CHECK step 0 max rel err over {WORLD} ranks {['%.2e' % w for w in worst]}; rank 0 packed "
                    f"{[round(x, 6) for x in Pv]} standalone {[round(x, 6) for x in ex]} K {K}")
                assert max(worst) < CFG["losscheck_tol"], worst
            oc = np.sum([np.asarray(b_["own"]).sum(0) for b_ in bl], 0)
            tot = TR.allsum([float(x) for x in oc[:5]], dev)
            bts = [TR.prep(b_, dev, tot[0] / TR.NORM_W, tot[1] / TR.NORM_W, tot[3] / TR.NORM_W) for b_ in bl]
            v, loss_v, ex_ = TR.train_step_multi(model, bts, rot, dos)
            torch.cuda.synchronize()
            assert abs(ex_["dec_s_N"] - tot[2]) < 0.5 and abs(ex_["dec_p_N"] - tot[4]) < 0.5
            fp_g = fingerprint(model, names_tr, "grad")
            gn = {}
            for pg in opt.param_groups:
                g_ = torch.nn.utils.clip_grad_norm_(pg["params"], CFG["clip"][pg["name"]])
                gn[pg["name"]] = float(g_.full_tensor() if hasattr(g_, "full_tensor") else g_)
            lr_used = {pg["name"]: pg["lr"] for pg in opt.param_groups}
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(); dt = time.time() - ts
            sv = TR.allsum(list(v) + [loss_v], dev)
            d = TR.summarize(sv[:-1], "train")
            d["train/loss"] = sv[-1] / TR.NORM_W
            ul = [x / TR.NORM_W for x in sv[-1 - TR.NSTREAM:-1]]
            d.update({"train/uloss_word": ul[0], "train/uloss_cb0": ul[1], "train/uloss_pose_cb0": ul[2]})
            fp_p = fingerprint(model, names_tr, "param")
            fp_d = fingerprint(model, None, "dec")
            pk = torch.tensor([torch.cuda.max_memory_allocated(dev) / GiB], device=dev); dist.all_reduce(pk, op=dist.ReduceOp.MAX)
            rec["steps"].append({"step": step + 1, "d": d, "gn": gn, "lr": lr_used, "tot": tot, "dec_s_M": ex_["dec_s_M"], "dec_p_M": ex_["dec_p_M"],
                                 "dec_s_lrs": ex_["dec_s_lrs"], "dec_p_lrs": ex_["dec_p_lrs"], "passab": [ex_["passab_loss_rel"], ex_["passab_hv_abs"]],
                                 "time_s": dt, "peak_gib": float(pk), "fp_grad": fp_g, "fp_param": fp_p, "fp_dec": fp_d})
            log(f"r{rep_k} step {step + 1}: {dt:.1f}s peak {float(pk):.1f} GiB | loss {d['train/loss']:.6f} word_ce_norm {d['train_speech/word_ce_norm']:.6f} "
                f"cb0 {d['train_speech/cb0_ce']:.6f} light {d['train_speech/light_ce']:.6f} pose_cb0 {d['train_speech/pose_cb0_ce']:.6f} "
                f"pose_light {d['train_speech/pose_light_ce']:.6f} | gn " + " ".join(f"{k}={x:.6g}" for k, x in gn.items())
                + f" | lr_trunk {lr_used['trunk']:.3g} dec_lr_last {ex_['dec_s_lrs'][-1]:.3g}")
        if RANK == 0:
            os.makedirs(OUT, exist_ok=True)
            torch.save(rec, f"{OUT}/{TAG}_r{rep_k}.pt")
        del opt, sched, dos, by, frozen, decn
        torch.cuda.empty_cache()
    log("done")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
