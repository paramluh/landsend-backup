"""aloe-v2vox-raw LAUNCHER (login node: squeue / scontrol / sbatch only, no compute). Polls every 30 s:
 1. waits for the shared A-track staging (all 8 logical nodes' ledgers DONE_v2vox_node<l>.json in the shared /data copy);
 2. spare nodes = slinky-4, slinky-7, slinky-18 (never PPM's 16 nodes). A node is FREE when no GPU job holds it, >= 540 GB memory and
    >= 32 CPUs are unallocated, continuously for GRACE s (gives the pose agent time to submit after its servers stop);
 3. launch rule: >= 3 free -> now on 3; else if a pose (other GPU) job holds spare nodes -> if >= 2 free: now on the free ones, else chain
    sbatch --dependency=afterany:<that job> -w <its spare nodes + free ones> (>= 2 nodes); no other GPU job and 2 free for GRACE -> now on 2;
 4. fallback: our job PENDING for a non-dependency reason > 15 min while >= 2 of the spare nodes are free -> cancel + resubmit on the free ones;
    our job PENDING on a dependency whose job ended > 10 min ago -> same.
 cpus/task = min over the chosen nodes of (64 - CPUs held by non-GPU jobs) // 8, in [4, 8]. Writes logs/launcher.log and LAUNCHED_JOB.
usage: nohup python3 launcher.py &"""
import json
import os
import re
import subprocess
import time

R = "/data/home/amartyavarma/ppm/aloe-v2vox-raw"
SPARE = ["slinky-4", "slinky-7", "slinky-15", "slinky-18"]   # user's spares 4 / 7 / 18 + slinky-15 (the pose FT 23051's node: "same nodes")
GRACE = 300
INIT = "/data/home/amartyavarma/ppm/ckpt-infer/ppm-v1/local-11488"
LOG = f"{R}/logs/launcher.log"
STATE = f"{R}/LAUNCHED_JOB"
MINE = ("aloev2vraw",)
SERVERS = ("m1vox-live", "m1vox-live2", "s4live-demo", "t2v-server")   # inference servers being stopped: block nodes, never a chain target


def log(*a):
    s = time.strftime("%m-%d %H:%M:%S") + " " + " ".join(str(x) for x in a)
    open(LOG, "a").write(s + "\n")


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout


def staged():
    return all(os.path.exists(f"{R}/flac/_ledger/DONE_v2vox_node{l}.json") for l in range(8))


def jobs():
    out = []
    for ln in sh(["squeue", "-h", "-o", "%i|%j|%T|%N|%C|%b|%r|%E"]).splitlines():
        p = ln.split("|")
        if len(p) < 8:
            continue
        jid, name, st, nl, cpus, tres, reason, dep = p[:8]
        nodes = sh(["scontrol", "show", "hostnames", nl]).split() if nl and nl != "(null)" else []
        out.append({"id": jid, "name": name, "state": st, "nodes": nodes, "cpus": int(cpus) if cpus.isdigit() else 0,
                    "gpu": "gpu" in tres, "reason": reason, "dep": dep})
    return out


def node_info(n):
    t = sh(["scontrol", "show", "node", n])
    g = lambda k: (re.search(rf"{k}=(\S+)", t) or [None, ""])[1]  # noqa: E731
    alloc = g("AllocTRES")
    gpu = int((re.search(r"gres/gpu=(\d+)", alloc) or [None, "0"])[1])
    mem = int(g("RealMemory") or 0) - int(g("AllocMem") or 0)
    cpu = int(g("CPUTot") or 0) - int(g("CPUAlloc") or 0)
    st = g("State")
    return {"gpu_alloc": gpu, "mem_free_mb": mem, "cpu_free": cpu, "state": st}


def cpus_per_task(nodes, js):
    c = 8
    for n in nodes:
        held = sum(j["cpus"] // max(len(j["nodes"]), 1) for j in js if n in j["nodes"] and not j["gpu"] and j["state"] == "RUNNING"
                   and not j["name"].startswith("v2vraw"))
        c = min(c, (64 - held) // 8)
    return max(4, c)


def submit(nodes, cpt, dep=None):
    nodes = sorted(nodes, key=lambda x: int(x.split("-")[1]))
    cmd = ["sbatch", "-N", str(len(nodes)), "-w", ",".join(nodes), f"--cpus-per-task={cpt}", "--mem=540G",
           f"--export=ALL,QQA_INIT={INIT}"]
    if dep:
        cmd += [f"--dependency=afterany:{dep}"]
    cmd += [f"{R}/aloe_v2vox_raw.sbatch"]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    log("SUBMIT", " ".join(cmd), "->", out.stdout.strip(), out.stderr.strip())
    m = re.search(r"Submitted batch job (\d+)", out.stdout)
    if m:
        json.dump({"job": m.group(1), "nodes": nodes, "cpt": cpt, "dep": dep, "time": time.strftime("%F %T")}, open(STATE, "w"))
        return m.group(1)
    return None


def main():
    free_since = {}
    pend_since = {}
    dep_end = {}
    log("launcher start")
    while True:
        try:
            js = jobs()
            mine = [j for j in js if j["name"] in MINE]
            now = time.time()
            infos = {n: node_info(n) for n in SPARE}
            for n in SPARE:
                i = infos[n]
                busy_gpu = any(n in j["nodes"] and j["gpu"] and j["state"] == "RUNNING" for j in js)
                ok = (not busy_gpu and i["gpu_alloc"] == 0 and i["mem_free_mb"] >= 540 * 1024 and i["cpu_free"] >= 32
                      and "DRAIN" not in i["state"] and "DOWN" not in i["state"])
                if ok:
                    free_since.setdefault(n, now)
                else:
                    free_since.pop(n, None)
            free = [n for n in SPARE if n in free_since and now - free_since[n] >= GRACE]
            others = [j for j in js if j["gpu"] and j["name"] not in MINE and j["name"] not in SERVERS and set(j["nodes"]) & set(SPARE)]
            if mine:
                j = mine[0]
                if j["state"] == "RUNNING":
                    log("our job", j["id"], "RUNNING on", j["nodes"], "-> launcher exits")
                    return
                if j["state"] == "PENDING":
                    dep_live = any(o["id"] in j["dep"] for o in js if o["id"] != j["id"]) if j["dep"] and j["dep"] != "(null)" else False
                    if "Dependency" in j["reason"] and dep_live:
                        pend_since.pop(j["id"], None)
                        dep_end.pop(j["id"], None)
                    else:
                        pend_since.setdefault(j["id"], now)
                        lim = 600 if j["dep"] and j["dep"] != "(null)" else 900
                        if now - pend_since[j["id"]] > lim and len(free) >= 2:
                            log("fallback: job", j["id"], "pending", j["reason"], f"{(now - pend_since[j['id']]) / 60:.0f} min; free", free, "-> cancel + resubmit")
                            subprocess.run(["scancel", j["id"]], timeout=60)
                            time.sleep(5)
                            submit(free[:3], cpus_per_task(free[:3], js))
                time.sleep(30)
                continue
            if not staged():
                time.sleep(30)
                continue
            if os.path.exists(STATE):                               # a job was already launched (and is gone): never relaunch blindly
                log("our job is gone (finished / failed / cancelled): launcher exits; see", STATE)
                return
            if len(free) >= 3:
                submit(free[:3], cpus_per_task(free[:3], js))
            elif others:
                o = max(others, key=lambda x: len(set(x["nodes"]) & set(SPARE)))
                held = [n for n in SPARE if n in o["nodes"]]
                if len(free) >= 2:
                    log("pose/other job", o["id"], o["name"], "holds", held, "; 2 spare nodes free -> run now on", free)
                    submit(free, cpus_per_task(free, js))
                else:
                    tgt = sorted(set(held) | set(free), key=lambda x: int(x.split("-")[1]))
                    if len(tgt) >= 2:
                        log("chain after", o["id"], o["name"], o["state"], "on", tgt)
                        submit(tgt, cpus_per_task(tgt, js), dep=o["id"])
                    else:
                        log("waiting: other job", o["id"], "holds", held, "free", free)
            elif len(free) >= 2:
                submit(free, cpus_per_task(free, js))
            else:
                log("waiting for spare nodes: free", free, {n: infos[n] for n in SPARE})
        except Exception as e:  # noqa: BLE001
            log("ERROR", repr(e))
        time.sleep(30)


if __name__ == "__main__":
    main()
