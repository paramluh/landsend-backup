"""si20k-vox CPU check of the co-training podcast path (run on a host holding a pretraining slice, e.g. slinky-9): for each trainer, import it
with its run env (+ PLAN values of a store set), build this host's podcast Source for local ranks 0..7 (train cfg, light 0.1) and eval cfg,
pack with podcast_rows.as_micro, run the trainer's own prep_pod on CPU (INTL: il_check with native A words), and report T, P, swap, light
ratio, A-lane word ids, the forward path and the cfg diff. usage: python -I pod_check.py --stores DIR [--runs ppm,intl]"""
import argparse
import json
import os
import subprocess
import sys

P = "/data/home/amartyavarma/ppm"


def env_file(path):
    out = {}
    for ln in open(path):
        ln = ln.strip()
        if ln and not ln.startswith("#") and "=" in ln:
            k, v = ln.split("=", 1); out[k] = v
    return out


CHILD = r'''
import sys, json, socket, numpy as np, torch
Q = sys.argv[1]
sys.path.insert(0, Q + "/code"); sys.path.insert(0, Q + "/code/lib")
import train_si20k_vox as T
import podcast_rows as PR
host = socket.gethostname().split(".")[0]
pl = PR.load_plan(T.CFG["podcast"]["plan"])
tr, ev = PR.cfgs(pl)
out = {"host": host, "mode": PR.host_entry(pl, host)[1], "ranks": []}
for j in range(8):
    src = PR.Source(pl, host, j, tr)
    m = src.micro(src.codes[0]); b = PR.as_micro(m, src.codes[0])
    t = T.prep_pod(b, torch.device("cpu"), float(len(b["sel"])), float((np.asarray(b["cb0"]) >= 0).sum()))
    ida = np.asarray(b["ids"])[0]
    r = {"j": j, "prod_rank": src.prod_rank, "n_codes": len(src.codes), "T": len(b["pid"]), "P": int(b["P"][0]), "swap": int(b["swap"]),
         "light_ratio": len(b["light"]) / max(int((np.asarray(b["cb0"]) >= 0).sum()), 1), "A_lane_word_ids": int((ida < 151643).sum()),
         "A_present_sum": float(np.asarray(b["present"])[0].sum()), "fwd": t["fwd"], "wt_sum": float(t["wt"].sum()), "n_eval": len(src.eval_codes)}
    if src.eval_codes:
        me = PR.Source(pl, host, j, ev).micro(src.eval_codes[0]); be = PR.as_micro(me, src.eval_codes[0])
        T.prep_pod(be, torch.device("cpu"), 1.0, 1.0); r["eval_T"] = len(be["pid"]); r["eval_light_ratio"] = len(be["light"]) / max(int((np.asarray(be["cb0"]) >= 0).sum()), 1)
    out["ranks"].append(r)
n, bad = T.cfg_diff_si20k()
out["cfg_diff"] = {"n": n, "bad": bad}
print("PODCHECK " + json.dumps(out))
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stores", required=True); ap.add_argument("--runs", default="ppm,intl")
    a = ap.parse_args()
    sj = json.load(open(f"{a.stores}/STORES.json"))
    for run in a.runs.split(","):
        Q = f"{P}/si20k-vox-{run}"
        env = dict(os.environ)
        env.update(env_file(f"{Q}/si20k_vox_{run}.env"))
        env.update(QQA_STEPS=str(sj["plan"]["steps"]), QQA_EXPORT_STEPS=",".join(map(str, sj["plan"]["export_steps"])), QQA_EXPORT=f"{Q}/export-{sj['plan']['steps']}",
                   QQA_REF_PLAN_DIGEST=sj["plan"]["digest"], ALOE_SI20K_SRC=a.stores, ALOE_SI20K_STORE_ROOT=a.stores, ALOE_SI20K_STORES_JSON=f"{a.stores}/STORES.json",
                   RANK="0", WORLD_SIZE="64", LOCAL_RANK="0", LOCAL_WORLD_SIZE="8", ALOE_LNODES="0;1;2;3;4;5;6;7", QQA_RUN=f"si20k-vox-{run}-jPODCHECK",
                   QQA_LOG_ROOT=f"/tmp/si20k-podcheck-{run}", PPM_CD="2",
                   QQA_INIT={"ppm": f"{P}/ckpt-infer/ppm-v1/local-12829", "intl": f"{P}/next-arch/ckpt-weka/ppm-next/w-6242"}[run])
        rr = subprocess.run([sys.executable, "-I", "-c", CHILD, Q], env=env, capture_output=True, text=True, cwd="/tmp")
        line = [ln for ln in rr.stdout.splitlines() if ln.startswith("PODCHECK ")]
        print(run, "rc", rr.returncode, line[0][9:] if line else rr.stderr[-3000:], flush=True)


if __name__ == "__main__":
    main()
