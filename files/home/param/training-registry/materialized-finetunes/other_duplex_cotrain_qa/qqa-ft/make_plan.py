"""qqa-ft PLAN (CPU, run on each spare node that holds the run-C store): run C's EXACT packed data order, per local rank, as node rows.
Re-derives run C's node sequence with run C's own code (stage4/pack/packed_plan.node_sequence + data_s3q.plan with run C's resolved config
runs/cfg-21169.json), asserts it equals the offline plan run C trained on (s4-qqa/prep/PACKS-<host>.json: node digest, sequence digest),
then maps every pack (sequence positions) to node rows of this host's store. Eval rows = run C's per-rank eval rows (data_s3q.plan).
-> qqa-ft/plan/PLAN-<host>.json {host, node_digest, steps, ranks: {l: [[node rows of step s's conversations], ...]}, eval: {l: [rows]},
   L_runC: {l: [[run C segment lengths]]}}
usage: python -I make_plan.py --host H"""
import argparse
import json
import os
import socket
import sys

import numpy as np

S4 = "/data/home/amartyavarma/qwen-1ch/stage4"
sys.path.insert(0, f"{S4}/pack")
import packed_plan as PPL  # noqa: E402  (puts stage4 + stage3 rows_build + stage2 on sys.path)

STORE = "/scratch/amartyavarma/s4qqa/prep/store"
CFG = f"{S4}/runs/cfg-21169.json"
PREP = "/data/home/amartyavarma/s4-qqa/prep"
OUT = "/data/home/amartyavarma/ppm/qqa-ft/plan"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    a = ap.parse_args()
    assert socket.gethostname().split(".")[0] == a.host, ("run on the host that holds the store", a.host)
    os.environ.setdefault("S3_MIN_FRAMES", "125")
    import data_s4q as D4
    DS = D4.DS
    C = json.load(open(CFG))
    PK = json.load(open(f"{PREP}/PACKS-{a.host}.json"))
    node = DS.R3.load_node(STORE); idx = DS.load_index(STORE)
    assert node["node"] == a.host and node["digest"] == PK["node_digest"], ("store != run C's", node["node"], node["digest"], PK["node_digest"])
    seq, plans = PPL.node_sequence(DS, node, idx, C)
    assert PPL.seq_digest(node, seq[:PK["n_rows"]]) == PK["seq_digest"], "node sequence != run C's offline plan"
    LW = int(PK["lw"]); steps = int(open(f"{PREP}/STEPS").read())
    ranks, Ls, ev = {}, {}, {}
    for l in range(LW):
        pk = PK["packs"][str(l)]
        assert len(pk) >= steps, (l, len(pk), steps)
        ranks[str(l)] = [[int(seq[i]) for i in p] for p in pk[:steps]]
        Ls[str(l)] = [[int(PK["L"][i]) for i in p] for p in pk[:steps]]
        ev[str(l)] = [int(r) for r in plans[l]["eval_rows"]]
        assert len(ev[str(l)]) == int(C["s3_eval_rows_per_rank"]), ev[str(l)]
    allr = [r for l in ranks for p in ranks[l] for r in p]
    assert len(set(allr)) == len(allr), "a conversation twice"
    assert not set(allr) & {r for l in ev for r in ev[l]}, "an eval row in training"
    out = {"format": "ppm-qqa-plan-v1", "host": a.host, "store": STORE, "node_digest": node["digest"], "seq_digest": PK["seq_digest"],
           "runc_cfg": CFG, "runc_packs": f"{PREP}/PACKS-{a.host}.json", "steps": steps, "lw": LW, "ranks": ranks, "eval": ev, "L_runC": Ls,
           "n_conv": len(allr), "conv_per_step_mean": round(len(allr) / (steps * LW), 3)}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(f"{OUT}/PLAN-{a.host}.json.tmp", "w")); os.replace(f"{OUT}/PLAN-{a.host}.json.tmp", f"{OUT}/PLAN-{a.host}.json")
    print(json.dumps({k: v for k, v in out.items() if k not in ("ranks", "eval", "L_runC")}))


if __name__ == "__main__":
    main()
