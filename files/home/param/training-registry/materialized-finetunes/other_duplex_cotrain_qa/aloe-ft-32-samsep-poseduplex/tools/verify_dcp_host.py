"""verify_dcp_host.py <dcp dir> <step> <host index> <ranks per host> <out json> [workers]   (READ ONLY)
One host's node-local full-state save (safety_ppm.save_local layout): COMPLETE == step; exactly rank<R>.pt + digest-<R>.txt for this host's
ranks (R = idx*n .. idx*n+n-1, task_samsep.sh rank rule) + rep.pt + digest-rep.txt + train_state.json + COMPLETE; train_state step == step;
every rank<R>.pt (and rep.pt) re-digested with the FT's OWN safety_ppm.digest_state (sha256 over sorted sections / names / keys of each
tensor's float32 bytes) == its saved digest file. Also records each file's plain sha256 + size (for the later HF check)."""
import concurrent.futures as cf
import hashlib
import importlib.util
import json
import os
import socket
import sys
import time

SAFE = os.environ.get("VERIFY_SAFETY_PY", "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/code/safety_ppm.py")   # the saving run's own copy
D, STEP, IDX, NPH, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
NW = int(sys.argv[6]) if len(sys.argv) > 6 else 4
H = os.environ.get("VERIFY_HOST_LABEL") or socket.gethostname().split(".")[0]   # label = the host whose ranks these are (a /data copy may be read elsewhere)
ALLOW_EXTRA = {x for x in os.environ.get("VERIFY_ALLOW_EXTRA", "").split(",") if x}   # known extra files (e.g. RESHARD_VERIFIED.json)


CH = 32 << 20                                                  # streaming chunk (bytes): RSS stays small (overlap steps share a job cgroup)
LOWMEM = os.environ.get("VERIFY_LOWMEM", "1") == "1"            # 1 = stream tensor bytes with pread + FADV_DONTNEED (never fault the mmap in)


def sha_file(p):
    m = hashlib.sha256()
    fd = os.open(p, os.O_RDONLY)
    try:
        off = 0
        while True:
            b = os.pread(fd, CH, off)
            if not b:
                break
            m.update(b)
            os.posix_fadvise(fd, off, len(b), os.POSIX_FADV_DONTNEED)
            off += len(b)
    finally:
        os.close(fd)
    return m.hexdigest()


def _map_base(path):
    """start address of the whole-file mapping torch.load(mmap=True) made of `path` (offset 0), from /proc/self/maps."""
    rp = os.path.realpath(path)
    for ln in open("/proc/self/maps"):
        f = ln.split(maxsplit=5)
        if len(f) == 6 and f[5].strip() == rp and int(f[2], 16) == 0:
            return int(f[0].split("-")[0], 16)
    raise RuntimeError(f"no mapping of {rp} in /proc/self/maps")


def digest_stream(path, torch):
    """== safety_ppm.digest_state(torch.load(path)) (same order, same bytes: each tensor's float32 C-order bytes), but tensor bytes are read with
    pread from the file in CH chunks (+ FADV_DONTNEED) instead of faulting in the mmap: tiny RSS. Non-contiguous tensors (rare) go through the
    mmap in dim-0 row blocks (same C-order bytes). -> (hexdigest, n_tensors, n_noncontig)."""
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    base, size = _map_base(path), os.path.getsize(path)
    fd = os.open(path, os.O_RDONLY)
    h = hashlib.sha256(); nt = nnc = 0
    try:
        for sec in sorted(sd):
            for n in sorted(sd[sec]):
                v = sd[sec][n]
                for k, t in (sorted(v.items()) if isinstance(v, dict) else [("", v)]):
                    h.update(f"{sec}|{n}|{k}".encode()); nt += 1
                    t = t.detach()
                    es, nb = t.element_size(), t.numel() * t.element_size()
                    if nb == 0:
                        continue
                    if t.is_contiguous():
                        off = t.data_ptr() - base
                        assert 0 <= off and off + nb <= size, ("tensor outside the file mapping", sec, n, k, off, nb, size)
                        q = 0
                        while q < nb:
                            ln = min(CH - CH % es, nb - q)
                            b = os.pread(fd, ln, off + q)
                            assert len(b) == ln, ("short read", sec, n, k)
                            if q == 0:                           # offset sanity: the first bytes equal the tensor's own (mmap) bytes
                                m0 = min(64, nb)
                                assert b[:m0] == t.reshape(-1)[: max(1, m0 // es)].contiguous().view(torch.uint8).numpy().tobytes()[:m0], \
                                    ("offset mismatch", sec, n, k)
                            if t.dtype == torch.float32:
                                h.update(b)
                            else:
                                h.update(torch.frombuffer(bytearray(b), dtype=t.dtype).float().numpy().tobytes())
                            os.posix_fadvise(fd, off + q, ln, os.POSIX_FADV_DONTNEED)
                            q += ln
                    else:
                        nnc += 1
                        if t.dim() == 0 or t.shape[0] <= 1:
                            h.update(t.float().cpu().numpy().tobytes())
                        else:
                            rows = max(1, CH // max(1, (t.numel() // t.shape[0]) * 4))
                            for i in range(0, t.shape[0], rows):
                                h.update(t[i:i + rows].float().contiguous().numpy().tobytes())
    finally:
        os.close(fd)
    secs = sorted(sd)
    del sd
    return h.hexdigest(), nt, nnc, secs


def selftest(mod, torch):
    """digest_stream == safety_ppm.digest_state on a small file with every awkward case (bf16, int, bool, 0-dim, empty, non-contiguous, views)."""
    import tempfile
    g = torch.Generator().manual_seed(0)
    big = torch.randn(300, 7, generator=g)
    sd = {"model": {"a": torch.randn(5, 3, generator=g), "b": torch.randn(4, 6, generator=g).bfloat16(), "c": big.t(), "d": big[10:20],
                    "e": torch.tensor(3.5), "f": torch.empty(0, 4), "g": torch.arange(10, dtype=torch.int64)},
          "opt": {"a": {"exp_avg": torch.randn(5, 3, generator=g), "step": torch.tensor(7.0)}, "z": {"m": torch.rand(9, generator=g) > 0.5}}}
    d = tempfile.mkdtemp(prefix="vdh-selftest-", dir="/dev/shm")
    try:
        p = f"{d}/t.pt"; torch.save(sd, p)
        ref = mod.digest_state(torch.load(p, map_location="cpu", weights_only=True))
        global CH
        ch0, CH = CH, 64                                           # tiny chunks: exercise the chunk loop
        got = digest_stream(p, torch)[0]
        CH = ch0
        assert got == ref, ("SELFTEST: digest_stream != safety_ppm.digest_state", got, ref)
    finally:
        import shutil; shutil.rmtree(d, ignore_errors=True)
    return True


def one(name, dg_name):
    import torch
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location("safety_ppm_ft", SAFE)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    p = f"{D}/{name}"
    pz = [x for x in os.environ.get("VERIFY_PAUSE_FILE", "").split(",") if x]
    if any(os.path.exists(x) for x in pz):                       # cooperative pause (the backup driver: MoT slower than its idle baseline)
        print(f"[{H}] PAUSED (digest check, before {name})", flush=True)
        while any(os.path.exists(x) for x in pz):
            time.sleep(20)
        print(f"[{H}] resumed (digest check, {name})", flush=True)
    t0 = time.time()
    fsha = sha_file(p)
    if LOWMEM:
        selftest(mod, torch)
        got, nt, nnc, secs = digest_stream(p, torch)
    else:
        sd = torch.load(p, map_location="cpu", mmap=True, weights_only=True)
        got = mod.digest_state(sd); nt = sum(len(v) for v in sd.values()); nnc = None; secs = sorted(sd)
        del sd
    want = open(f"{D}/{dg_name}").read().strip()
    return {"file": name, "size": os.path.getsize(p), "sha256": fsha, "digest_saved": want, "digest_recomputed": got, "match": got == want,
            "sections": secs, "n_entries": nt, "n_noncontig": nnc, "lowmem": LOWMEM, "sec": round(time.time() - t0, 1)}


ranks = list(range(IDX * NPH, IDX * NPH + NPH))
res = {"host": H, "ran_on": socket.gethostname().split(".")[0], "safety_py": SAFE, "dir": D, "step": STEP, "host_index": IDX, "ranks": ranks, "time": time.strftime("%F %T %Z"), "errors": []}
err = res["errors"]
if not os.path.isdir(D):
    err.append("dir missing")
else:
    have = set(os.listdir(D))
    want = {f"rank{r}.pt" for r in ranks} | {f"digest-{r}.txt" for r in ranks} | {"rep.pt", "digest-rep.txt", "train_state.json", "COMPLETE"}
    res["missing"] = sorted(want - have); res["extra"] = sorted(have - want - ALLOW_EXTRA); res["allowed_extra"] = sorted(have & ALLOW_EXTRA)
    if res["missing"]:
        err.append(f"missing {res['missing']}")
    if res["extra"]:
        err.append(f"extra files {res['extra']}")
    c = open(f"{D}/COMPLETE").read().strip() if "COMPLETE" in have else None
    res["COMPLETE"] = c
    if c != str(STEP):
        err.append(f"COMPLETE {c!r} != {STEP}")
    if "train_state.json" in have:
        ts = json.load(open(f"{D}/train_state.json"))
        res["train_state_step"] = ts.get("step")
        res["train_state_keys"] = sorted(ts)
        for k in ("hosts", "host_ranks", "world"):
            if k in ts:
                res[f"train_state_{k}"] = ts[k]
        if ts.get("step") != STEP:
            err.append(f"train_state step {ts.get('step')} != {STEP}")
        hs = ts.get("hosts")                                   # per-rank host list (PPMVOX) or per-host list (aloe FT): must name this host
        if isinstance(hs, list) and hs:
            mine = [hs[r] for r in ranks] if len(hs) == ts.get("world", -1) and len(hs) > IDX * NPH + NPH - 1 else [hs[IDX]] if len(hs) > IDX else []
            res["train_state_hosts_for_ranks"] = sorted(set(mine))
            if set(mine) != {H}:
                err.append(f"train_state hosts for ranks {ranks[0]}..{ranks[-1]} = {sorted(set(mine))} != {H}")
    jobs = [(f"rank{r}.pt", f"digest-{r}.txt") for r in ranks if f"rank{r}.pt" in have and f"digest-{r}.txt" in have]
    if "rep.pt" in have and "digest-rep.txt" in have:
        jobs.append(("rep.pt", "digest-rep.txt"))
    t0 = time.time()
    with cf.ProcessPoolExecutor(NW) as ex:
        rows = list(ex.map(one, *zip(*jobs)))
    res["files"] = rows
    res["bytes"] = sum(r["size"] for r in rows)
    res["sec"] = round(time.time() - t0, 1)
    for r in rows:
        if not r["match"]:
            err.append(f"DIGEST MISMATCH {r['file']}")
    if len(rows) != len(ranks) + 1:
        err.append(f"digested {len(rows)} files != {len(ranks) + 1}")
    res["small_files_sha256"] = {f: sha_file(f"{D}/{f}") for f in sorted(have) if not f.endswith(".pt")}
res["PASS"] = not err
json.dump(res, open(OUT, "w"), indent=1)
print(f"[{H}] {'PASS' if res['PASS'] else 'FAIL'} {D} COMPLETE={res.get('COMPLETE')} ranks {ranks[0]}..{ranks[-1]} "
      f"files {len(res.get('files', []))} {res.get('bytes', 0) / 1e9:.1f} GB in {res.get('sec')} s; errors {err}", flush=True)
sys.exit(0 if res["PASS"] else 3)
