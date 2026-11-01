"""PPM RUN SAFETY (T16): node-local per-rank checkpoints (keep last N), save/stop control files, SIGUSR1, consolidated Weka export with a
/data free-space floor. Used by train_ppm.py (needs an initialised process group).

NODE-LOCAL CHECKPOINT  <save_root>/<run_base>/dcp-N/  (node-local /scratch only; asserted)
  rank<R>.pt         this rank's LOCAL shards: every parameter's fp32 shard, every optimizer state tensor's shard (main AdamW + both decoder
                     AdamWs; FSDP2 shards = dim-0 slices), keyed by parameter FQN. Plain torch.save, no DCP: a DCP save deduplicates replicated
                     entries onto arbitrary ranks and writes .metadata on rank 0's node only, so a node-local DCP cannot be loaded on the
                     other nodes (stage 3's dcp_resume workaround). Per-rank files need the SAME world size and rank -> host map on resume
                     (asserted from train_state.json); a world change goes through the consolidated export (weights only).
  digest-<R>.txt     sha256 of this rank's saved state (resume re-computes it after loading: exact restore, asserted)
  train_state.json   step, cfg, sched, decoder counters, hosts (one copy per node, written by its local rank 0)
  COMPLETE           written per node only after EVERY rank's file is on disk (barrier); written into dcp-N.tmp first, renamed after
  Keep-last-N (PPM_KEEP_CKPT, default 2): after a COMPLETE save, each node deletes its own older COMPLETE dcp-* beyond the newest N (a newer
  complete save supersedes them) and stale *.tmp dirs; the resume source is never deleted by the run that resumed from it until superseded.
CONTROL  <control_dir>/SAVE_NOW (save at the next step boundary, continue) | STOP_AFTER_SAVE (save at the next step boundary, exit 0);
  consumed (renamed *.done-<step>) by rank 0 once handled. SIGUSR1 on any rank = STOP_AFTER_SAVE (ppm.sbatch traps the batch shell's
  USR1 -- Slurm --signal=B:USR1@300 -- and touches STOP_AFTER_SAVE). Flags are all-reduced (MAX) at every step boundary.
EXPORT  consolidated bf16 weights (model only, no optimizer) as a DCP on the shared /data (<export_root>/<run_base>/w-N, COMPLETE marker):
  loadable at any world size. BEFORE writing: free space of /data - export size >= PPM_EXPORT_MIN_FREE_GB (300): else REFUSED (logged,
  nothing written). PPM_EXPORT_PARTS all (default) | new (every tensor except the 64 backbone layers)."""
import glob
import hashlib
import json
import os
import shutil
import signal
import time

import torch
import torch.distributed as dist

USR1 = {"hit": False}
PROTECT_STEPS = set()        # 8,1,1 (code-811): node-local dcp-N with N in this set (the switch checkpoint) are never deleted by prune /
                             #   space_check (empty = production)


def install_usr1(log):
    def h(signum, frame):
        USR1["hit"] = True
    signal.signal(signal.SIGUSR1, h)
    log("[safety] SIGUSR1 handler installed (= STOP_AFTER_SAVE at the next step boundary)")


def _loc(t):
    return t.to_local() if hasattr(t, "to_local") else t


def _keep(p, which):
    return which == "all" or (hasattr(p, "to_local") == (which == "sharded"))


def opt_named_states(model, opt, which="all"):
    name = {id(p): n.replace("_checkpoint_wrapped_module.", "") for n, p in model.named_parameters()}
    out = {}
    for g in opt.param_groups:
        for p in g["params"]:
            st = opt.state.get(p)
            if st and _keep(p, which):
                out[name[id(p)]] = {k: _loc(v) for k, v in st.items()}
    return out


def local_state(model, opts, which="all", skip=()):
    """which: sharded (FSDP2 DTensor params: this rank's shards) | rep (replicated plain params, e.g. the T14 decoders: identical on every
    rank -> saved ONCE per node by local rank 0) | all. skip: parameter names left out (load_local's allow_missing digest check)."""
    sd = {"model": {n.replace("_checkpoint_wrapped_module.", ""): _loc(p).detach() for n, p in model.named_parameters() if _keep(p, which)},
          **{k: opt_named_states(model, o, which) for k, o in opts.items()}}
    return {sec: {n: v for n, v in d.items() if n not in skip} for sec, d in sd.items()} if skip else sd


def digest_state(sd):
    h = hashlib.sha256()
    for sec in sorted(sd):
        for n in sorted(sd[sec]):
            v = sd[sec][n]
            for k, t in (sorted(v.items()) if isinstance(v, dict) else [("", v)]):
                h.update(f"{sec}|{n}|{k}".encode()); h.update(t.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def state_bytes(model, opts, which):
    n = sum(_loc(p).numel() * 4 for p in model.parameters() if _keep(p, which))
    for o in opts.values():
        for g in o.param_groups:
            for p in g["params"]:
                if _keep(p, which):
                    n += sum(_loc(t).numel() * t.element_size() for t in o.state.get(p, {}).values())
    return n


def space_check(model, opts, root, keep, local, log, margin=1.15):
    """before a save: this node needs (its ranks' shard bytes + the replicated file) x margin free on /scratch. Short -> prune this node's
    oldest COMPLETE saves down to keep - 1 (the save about to be written makes keep again); still short -> False on every rank (the save is
    SKIPPED, loudly: a full /scratch must never crash or corrupt the run). -> bool (agreed by all ranks)."""
    dev = f"cuda:{torch.cuda.current_device()}"
    mine = torch.tensor([float(state_bytes(model, opts, "sharded"))], device=dev)
    allb = [torch.zeros_like(mine) for _ in range(dist.get_world_size())]
    dist.all_gather(allb, mine)
    lw = int(os.environ.get("LOCAL_WORLD_SIZE", "8"))
    node = dist.get_rank() // lw
    need = (sum(float(x) for x in allb[node * lw:(node + 1) * lw]) + state_bytes(model, opts, "rep")) * margin
    if os.environ.get("PPM_TEST_SCRATCH_NEED_GB"):                # smokes only: pretend a save needs this much (exercises the skip path)
        need = float(os.environ["PPM_TEST_SCRATCH_NEED_GB"]) * 1e9
    ok = True
    if local == 0:
        os.makedirs(root, exist_ok=True)
        free = shutil.disk_usage(root).free
        if free < need:
            s = [x for x in complete_steps(root) if x not in PROTECT_STEPS]   # (8,1,1: the protected switch save is never pruned)
            while free < need and len(s) > max(keep - 1, 0):
                shutil.rmtree(f"{root}/dcp-{s[0]}", ignore_errors=True)
                log(f"!!! [save] /scratch short on {os.uname().nodename}: need {need / 1e9:.0f} GB, free {free / 1e9:.0f} GB -> deleted older dcp-{s[0]} first")
                s = s[1:]
                free = shutil.disk_usage(root).free
            ok = free >= need
            if not ok:
                print(f"!!! [save] {os.uname().nodename}: /scratch free {free / 1e9:.0f} GB < need {need / 1e9:.0f} GB even after pruning", flush=True)
    t = torch.tensor([float(ok)], device=dev); dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(float(t))


def save_local(model, opts, sched, extra, path, rank, local, log, keep=2):
    """all ranks: space check, write <path>.tmp/rank<R>.pt (+ digest), barrier, local rank 0 writes train_state + COMPLETE and renames to
    <path>. -> True, or False if SKIPPED for lack of /scratch space (logged loudly; nothing half-written)."""
    assert path.startswith("/scratch/"), ("node-local checkpoints go to /scratch only", path)
    if not space_check(model, opts, os.path.dirname(path), keep, local, log):
        log(f"!!! [save] SKIPPED {path}: some node's /scratch cannot hold it (see the per-node lines); training continues, previous saves kept")
        return False
    t0 = time.time()
    tmp = path + ".tmp"
    if local == 0:
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp, exist_ok=True)
    dist.barrier()
    sd = local_state(model, opts, "sharded")
    torch.save(sd, f"{tmp}/rank{rank}.pt")
    open(f"{tmp}/digest-{rank}.txt", "w").write(digest_state(sd))
    del sd
    if local == 0:                                             # replicated tensors (decoders + their AdamW states): once per node
        sr = local_state(model, opts, "rep")
        torch.save(sr, f"{tmp}/rep.pt")
        open(f"{tmp}/digest-rep.txt", "w").write(digest_state(sr))
        del sr
    dist.barrier()
    if local == 0:
        json.dump({**extra, "sched": sched.state_dict()}, open(f"{tmp}/train_state.json", "w"))
        open(f"{tmp}/COMPLETE", "w").write(str(extra["step"]))
        if os.path.isdir(path):
            shutil.rmtree(path)
        os.rename(tmp, path)
    dist.barrier()
    log(f"[save] {path} node-local per-rank state ({time.time() - t0:.1f}s)")
    return True


@torch.no_grad()
def load_local(model, opts, sched, path, rank, log, allow_missing=(), drop_opt=(), drop_model=()):
    """restore weights, optimizer states (exact, via a zero-gradient step that creates the state tensors, then in-place copies), sched;
    asserts the recomputed digest == the saved one. -> train_state dict. allow_missing (D32): parameter names that a checkpoint made before
    they existed lacks (they must be missing from it, all of them): they keep their current (init) values and get NO optimizer state (fresh
    AdamW); the digest check covers every other tensor. drop_opt (D36): parameter names whose MAIN-AdamW ('opt') state the checkpoint holds
    (every one of them, in the per-rank sharded file) but which are deliberately NOT in the optimizer now (frozen, unused): their WEIGHTS are
    restored as usual, their saved AdamW state is set aside (not restored). The digest check is NOT loosened: the recomputed state + the
    set-aside entries (exactly as read from the file) must reproduce the saved per-rank digest. drop_model (D37, 10-16): parameter names the
    checkpoint holds (every one, in the per-rank sharded file, weights ONLY: no state in any optimizer section) but which the model no longer
    has (module removed); their saved weights are set aside, every other tensor restored as usual; the digest check is NOT loosened: the
    restored state + the set-aside weights (exactly as read) must reproduce the saved per-rank digest."""
    assert os.path.isfile(f"{path}/COMPLETE"), ("resume source not COMPLETE on this node", path)
    ts = json.load(open(f"{path}/train_state.json"))
    dv = f"cuda:{torch.cuda.current_device()}"
    if os.environ.get("PPM_TEST_LOAD_CPU") == "1":             # 8,1,1 TEST-ONLY (2-GPU reduced-depth tests: a rank's whole state would not fit
        dv = "cpu"                                             #   twice on its GPU): stage the saved tensors in host RAM; copies are exact
    sd = torch.load(f"{path}/rank{rank}.pt", map_location=dv)
    rep_names = set()
    if os.path.isfile(f"{path}/rep.pt"):                       # replicated tensors (same file for every rank of the node)
        sr = torch.load(f"{path}/rep.pt", map_location=dv)
        rep_names = {(sec, n) for sec in sr for n in sr[sec]}
        for sec in sr:
            sd.setdefault(sec, {}).update(sr[sec])
        del sr
    params = {n.replace("_checkpoint_wrapped_module.", ""): p for n, p in model.named_parameters()}
    miss = set(allow_missing)
    dropm = set(drop_model)
    dropped_m = {}
    if dropm:                                                  # D37: every check explicit; only these names' saved WEIGHTS are set aside
        assert not (dropm & set(params)), ("drop_model: names must be absent from the model", sorted(dropm & set(params)))
        assert not (dropm & (miss | set(drop_opt))), ("drop_model: overlaps allow_missing / drop_opt", sorted(dropm & (miss | set(drop_opt))))
        assert dropm <= set(sd["model"]), ("drop_model: the checkpoint must hold weights for every name", sorted(dropm - set(sd["model"])))
        assert not any(("model", n) in rep_names for n in dropm), "drop_model: weights expected in the per-rank (sharded) file, found in rep.pt"
        assert not any(n in sd[k] for k in opts for n in dropm), ("drop_model: the checkpoint holds optimizer state for a dropped name",
                                                                  sorted((k, n) for k in opts for n in dropm if n in sd[k]))
        dropped_m = {n: sd["model"].pop(n) for n in sorted(dropm)}
    if miss:
        assert miss <= set(params) and not (miss & set(sd["model"])), ("allow_missing: names must be model parameters absent from the checkpoint",
                                                                       sorted(miss - set(params)), sorted(miss & set(sd["model"])))
        assert not any(n in sd[k] for k in opts for n in miss), "allow_missing: optimizer state for a missing parameter"
    assert set(sd["model"]) | miss == set(params), ("checkpoint parameter names differ", sorted((set(sd["model"]) | miss) ^ set(params))[:4])
    drop = set(drop_opt)
    dropped = {}
    if drop:                                                   # D36: every check explicit; only these names' 'opt' state is set aside
        assert not (drop & miss) and drop <= set(sd["model"]), ("drop_opt: names must be model parameters present in the checkpoint", sorted(drop))
        assert "opt" in opts and all(n in sd["opt"] for n in drop), ("drop_opt: the checkpoint must hold main-AdamW state for every name",
                                                                     sorted(n for n in drop if n not in sd.get("opt", {})))
        assert not any(("opt", n) in rep_names for n in drop), "drop_opt: state expected in the per-rank (sharded) file, found in rep.pt"
        assert not any(n in sd[k] for k in opts if k != "opt" for n in drop), "drop_opt: state in a decoder optimizer"
        in_opt = {id(p) for g in opts["opt"].param_groups for p in g["params"]}
        assert not any(id(params[n]) in in_opt for n in drop), ("drop_opt: names still in the main optimizer", sorted(drop))
        dropped = {n: sd["opt"].pop(n) for n in sorted(drop)}
    for n, p in params.items():
        if n not in miss:
            _loc(p).copy_(sd["model"][n])
    for k, o in opts.items():
        saved = sd[k]
        keep_views = set()
        for g in o.param_groups:                               # create the state tensors with a zero-gradient step (update exactly 0)
            for p in g["params"]:
                if p.grad is not None:                         # replicated decoders: grads are views of a flat buffer -> zero in place, keep
                    p.grad.zero_(); keep_views.add(p)
                else:
                    p.grad = torch.zeros_like(p)
        lrs = [g["lr"] for g in o.param_groups]
        for g in o.param_groups:
            g["lr"] = 0.0
        o.step()
        for g in o.param_groups:
            for p in g["params"]:
                if p in keep_views:
                    p.grad.zero_()
                else:
                    p.grad = None
        for g, lr in zip(o.param_groups, lrs):
            g["lr"] = lr
        cur = opt_named_states(model, o)
        for n, st in cur.items():
            if n not in saved:                                  # a parameter that had no state in the saved run: drop it again
                o.state.pop(params[n])
                continue
            for kk, t in st.items():
                t.copy_(saved[n][kk])
        assert set(saved) <= set(cur), (k, sorted(set(saved) - set(cur))[:4])
    for n, p in params.items():
        if n not in miss:
            _loc(p).copy_(sd["model"][n])                      # (unchanged by the zero steps; re-copied defensively)
    sched.load_state_dict(ts["sched"])
    del sd
    want = open(f"{path}/digest-{rank}.txt").read().strip()
    loc = local_state(model, opts, "sharded", skip=miss)
    if dropped_m:                                              # D37: the set-aside saved weights back in -> must equal the saved digest
        assert not (set(dropped_m) & set(loc["model"])), "drop_model: restored model has a dropped name"
        loc["model"] = {**loc["model"], **dropped_m}
    if dropped:                                                # D36: the set-aside saved entries back in -> must equal the saved digest
        assert not (set(dropped) & set(loc["opt"])), "drop_opt: restored optimizer has state for a dropped name"
        loc["opt"] = {**loc["opt"], **dropped}
    got = digest_state(loc)
    if os.path.isfile(f"{path}/digest-rep.txt"):
        want += open(f"{path}/digest-rep.txt").read().strip()
        got += digest_state(local_state(model, opts, "rep", skip=miss))
    ok = torch.tensor([float(want == got)], device=f"cuda:{torch.cuda.current_device()}")
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    assert float(ok) == 1.0, ("resume digest mismatch (state not restored exactly)", path)
    log(f"[resume] {path}: step {ts['step']}, weights + 3 optimizers + sched restored, per-rank digests == saved on every rank"
        + (f"; NOT in the checkpoint (kept at init, no optimizer state): {sorted(miss)}" if miss else "")
        + (f"; weights restored but saved main-AdamW state SET ASIDE (not in the optimizer, D36; digest incl. it == saved): {sorted(drop)}" if drop else "")
        + (f"; saved WEIGHTS SET ASIDE, module removed from the model (D37; no optimizer state in the checkpoint; digest incl. them == saved): "
           f"{sorted(dropm)}" if dropm else ""))
    return ts


def complete_steps(root):
    out = []
    for d in glob.glob(f"{root}/dcp-*"):
        b = os.path.basename(d)
        if b[4:].isdigit() and os.path.isfile(f"{d}/COMPLETE"):
            out.append(int(b[4:]))
    return sorted(out)


def newest_common(root, dev):
    """the newest step COMPLETE on every node (-1 if none)."""
    s = complete_steps(root)
    mine = s[-1] if s else -1
    t = torch.tensor([mine], device=dev); dist.all_reduce(t, op=dist.ReduceOp.MIN)
    m = int(t)
    if m >= 0:                                                 # every node must actually hold that step
        ok = torch.tensor([float(m in s)], device=dev); dist.all_reduce(ok, op=dist.ReduceOp.MIN)
        assert float(ok) == 1.0, ("newest common checkpoint missing on some node", m)
    return m


def prune(root, keep, local, log, protect=()):
    """each node: delete its COMPLETE dcp-* older than the newest `keep` (superseded) + stale .tmp dirs. Called after a COMPLETE save."""
    if local != 0:
        return []
    s = complete_steps(root)
    gone = []
    for st in s[:-keep] if keep > 0 else []:
        d = f"{root}/dcp-{st}"
        if os.path.abspath(d) in {os.path.abspath(p) for p in protect if p} or st in PROTECT_STEPS:
            continue
        shutil.rmtree(d, ignore_errors=True); gone.append(st)
    for d in glob.glob(f"{root}/dcp-*.tmp"):
        shutil.rmtree(d, ignore_errors=True)
    if gone:
        log(f"[save] keep-last-{keep}: deleted superseded node-local dcp-{gone} (newest {s[-keep:]})")
    return gone


class Control:
    def __init__(self, cdir, rank, dev, log):
        self.cdir, self.rank, self.dev, self.log = cdir, rank, dev, log
        if rank == 0:
            os.makedirs(cdir, exist_ok=True)

    def poll(self):
        """-> (save_now, stop) agreed by every rank (control files on rank 0, USR1 on any rank)."""
        f = [0.0, 0.0]
        if self.rank == 0:
            f = [float(os.path.exists(f"{self.cdir}/SAVE_NOW")), float(os.path.exists(f"{self.cdir}/STOP_AFTER_SAVE"))]
        f[1] = max(f[1], float(USR1["hit"]))
        t = torch.tensor(f, device=self.dev); dist.all_reduce(t, op=dist.ReduceOp.MAX)
        return bool(t[0]), bool(t[1])

    def consume(self, step, stop):
        if self.rank == 0:
            for nm in ("SAVE_NOW",) + (("STOP_AFTER_SAVE",) if stop else ()):
                p = f"{self.cdir}/{nm}"
                if os.path.exists(p):
                    os.rename(p, f"{p}.done-{step}")


def export_weights(model, step, root, run_base, min_free_gb, parts, rank, log, side=None):
    """consolidated bf16 DCP on /data (module doc). -> path or None (refused). side (D33): a small dict written as <path>/train_cfg.json
    (inference settings that are not tensors, e.g. the decp_bb link's alpha at this step)."""
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    msd = get_model_state_dict(model)
    if parts == "new":
        msd = {k: v for k, v in msd.items() if not k.startswith("layers.")}
    size_gb = sum(v.numel() * 2 for v in msd.values()) / 1e9
    path = f"{root}/{run_base}/w-{step}"
    ok = [None]
    if rank == 0:
        os.makedirs(root, exist_ok=True)
        free = shutil.disk_usage(root).free / 1e9
        ok = [(free - size_gb >= float(min_free_gb), free)]
    dist.broadcast_object_list(ok, src=0)
    good, free = ok[0]
    if not good:
        log(f"!!! [export] REFUSED w-{step} ({parts}): {size_gb:.1f} GB would leave {free - size_gb:.0f} GB free on {root} "
            f"(< floor {min_free_gb} GB; free now {free:.0f} GB). Nothing written; node-local checkpoints kept.")
        return None
    t0 = time.time()
    dcp.save({k: v.to(torch.bfloat16) for k, v in msd.items()}, checkpoint_id=path)
    dist.barrier()
    gone = []
    if rank == 0:
        if side:
            json.dump(side, open(f"{path}/train_cfg.json", "w"))
        open(f"{path}/COMPLETE", "w").write(str(step))
        keep = int(os.environ.get("PPM_EXPORT_KEEP", "2"))          # superseded COMPLETE exports beyond the newest `keep` are deleted
        ws = sorted(int(os.path.basename(d)[2:]) for d in glob.glob(f"{root}/{run_base}/w-*")
                    if os.path.basename(d)[2:].isdigit() and os.path.isfile(f"{d}/COMPLETE"))
        for w in ws[:-keep] if keep > 0 else []:
            shutil.rmtree(f"{root}/{run_base}/w-{w}", ignore_errors=True); gone.append(w)
    log(f"[export] {path}: consolidated bf16 weights ({parts}, {size_gb:.1f} GB, {time.time() - t0:.0f}s); /data free before {free:.0f} GB"
        + (f"; deleted superseded exports w-{gone}" if gone else ""))
    return path
