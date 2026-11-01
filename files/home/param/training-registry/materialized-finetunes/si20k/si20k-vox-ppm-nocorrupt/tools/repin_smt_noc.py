"""si20k-tin copy (match train_si20k_tin.py) of the si20k-txt copy of si20k-vox-ppm/tools/repin_smt.py (the layout that took the vox runs to ~11 s / step: main + NCCL threads on whole cores, corruption / collator on other physical cores, no shared SMT siblings); state file per Slurm job, matches train_si20k_txt.py.
si20k-vox OPERATIONAL CPU re-pin (no maths change), run on each training node inside the job (srun --overlap): the collator's qqa-corrupt-v2
servers were sharing physical cores (SMT siblings: logical CPUs 2k / 2k+1) with the ranks' main / NCCL threads, slowing every kernel launch.
Per socket: each rank's main process (all threads) -> 2 logical CPUs on 2 DIFFERENT physical cores (first siblings; the second siblings are
left idle), its DataLoader workers + corruption servers (+ any other descendants) -> the socket's pool of the remaining whole cores.
Prints the layout. usage: python3 repin_smt.py [--dry]"""
import os
import sys


def siblings(c):
    return tuple(int(x) for x in open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read().strip().replace("-", ",").split(","))


def pkg(c):
    return int(open(f"/sys/devices/system/cpu/cpu{c}/topology/physical_package_id").read())


def descendants(pid):
    kids, out = {}, []
    for p in os.listdir("/proc"):
        if p.isdigit():
            try:
                pp = int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1])
            except (OSError, IndexError, ValueError):
                continue
            kids.setdefault(pp, []).append(int(p))
    st = [pid]
    while st:
        x = st.pop()
        for k in kids.get(x, []):
            out.append(k); st.append(k)
    return out


def setaff(pid, cpus):
    n = 0
    for t in os.listdir(f"/proc/{pid}/task"):
        try:
            os.sched_setaffinity(int(t), cpus); n += 1
        except OSError:
            pass
    return n


def main():
    dry = "--dry" in sys.argv
    mains = []
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            cmd = open(f"/proc/{p}/cmdline").read()
            pp = int(open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1])
            pcmd = open(f"/proc/{pp}/cmdline").read()
        except OSError:
            continue
        if "code/train_si20k_vox_nocorrupt.py" in cmd and "train_si20k_vox_nocorrupt.py" not in pcmd and "python" in cmd:
            mains.append(int(p))
    U, R = set(), {}
    for m in mains:
        lst = open(f"/proc/{m}/status").read().split("Cpus_allowed_list:")[1].split("\n")[0].strip()
        cs = set()
        for part in lst.split(","):
            a, _, b = part.partition("-")
            cs |= set(range(int(a), int(b or a) + 1))
        R[m] = cs
    # the job's CPUs = union of the ranks' ORIGINAL task sets (saved on first run; a re-run reuses them)
    state = f"/tmp/si20k_noc_repin_{os.getuid()}_{os.environ.get('SLURM_JOB_ID', 'nojob')}.txt"
    if os.path.exists(state):
        U = set(int(x) for x in open(state).read().split())
    else:
        for cs in R.values():
            U |= cs
        open(state, "w").write(" ".join(map(str, sorted(U))))
    # 10-19 (si20k-vox-ppm11488, coordinator: step time): ONE shared corruption / collator pool = the union of every socket's free whole cores
    # (socket 0's pool measured saturated while socket 1's idled); mains keep their 2 whole cores each (no SMT sibling shared with the pool)
    POOL_ALL = set()
    for sock in sorted({pkg(c) for c in U}):
        cores_ = sorted({siblings(c) for c in U if pkg(c) == sock and all(x in U for x in siblings(c))})
        loose_ = sorted(c for c in U if pkg(c) == sock and not all(x in U for x in siblings(c)))
        need_ = 2 * len([m for m in mains if pkg(min(R[m])) == sock])
        POOL_ALL |= set(x for c in cores_[need_:] for x in c) | set(loose_)
    for sock in sorted({pkg(c) for c in U}):
        cores = sorted({siblings(c) for c in U if pkg(c) == sock and all(x in U for x in siblings(c))})
        loose = sorted(c for c in U if pkg(c) == sock and not all(x in U for x in siblings(c)))
        ranks = [m for m in mains if pkg(min(R[m])) == sock]
        ranks.sort(key=lambda m: min(R[m]))
        need = 2 * len(ranks)
        assert len(cores) >= need + 2, ("not enough whole cores on socket", sock, len(cores), need)
        pool = set(x for c in cores[need:] for x in c) | set(loose)
        pool = POOL_ALL                                           # shared over both sockets
        for i, m in enumerate(ranks):
            mc = set(cores[2 * i]) | set(cores[2 * i + 1])               # both siblings of 2 whole cores: no corruption thread on them
            d = descendants(m)
            nt = 0 if dry else setaff(m, mc)
            nd = 0 if dry else sum(setaff(k, pool) for k in d if os.path.exists(f"/proc/{k}"))
            print(f"{os.uname().nodename} socket {sock} rank-proc {m}: main -> {sorted(mc)} ({nt} threads); {len(d)} descendants -> pool {min(pool)}..{max(pool)} "
                  f"({len(pool)} logical CPUs, {nd} threads)", flush=True)


if __name__ == "__main__":
    main()
