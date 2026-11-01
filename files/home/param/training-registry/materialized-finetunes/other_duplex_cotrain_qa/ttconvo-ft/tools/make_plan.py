"""ttconvo-ft PLAN: one epoch over every tt-v3 TRAIN row on disk, laid over the 16 PPM hosts x 8 ranks (1 row / rank / step).
Stores (staged by tools/, logs/staging.log): slot stores slinky-4 / -7 / -11 / -13 (4 of the 16 tt-v3 node stores; the other 12 + all raw chunks
are gone, never re-downloaded) + the cross-slot partial store slinky-18 (659 episodes), on the hosts below.
Excluded from training: (1) the tt-v3 eval split (eval_episodes.txt, 281 episodes); (2) every PPM held-out speech episode: salt ppm-v1 and
s3eval-v1 at PPM_SPEECH_EVAL_FRAC 0.0005 (PPM's speech eval + its D2 holdout) and the episodes of the 103 speech_s3 eval rows; (3) slinky-18 rows
duplicated in a slot store (same row_key; the slot store's copy is used). Short final chunks (>= 60 s) are kept (one epoch of tt-v3 train).
Layout: host h's 8 ranks read its slot store + the partial store. Slot rows: seeded shuffle, dealt round-robin over the slot's 32 ranks; partial
rows: seeded shuffle, each to the rank with the fewest rows so far; then each rank's list is shuffled (seed, rank). steps = min over ranks; the
few rows beyond it are dropped (counted). Eval: the 512 rows of the Qwen tt3 run's eval set (j11717 eval_manifest.json, 4 per rank, its rank order).
usage: python -I make_plan.py OUT.json"""
import hashlib, json, sys
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "/data/home/amartyavarma/ppm/code-811/lib")
import data_s3q as DS  # noqa: E402

P = "/data/home/amartyavarma/ppm/ttconvo-ft/plan"
HOSTS = ["slinky-0", "slinky-1", "slinky-2", "slinky-3", "slinky-5", "slinky-6", "slinky-8", "slinky-9", "slinky-10", "slinky-11", "slinky-12",
         "slinky-13", "slinky-14", "slinky-16", "slinky-17", "slinky-19"]
SLOT_OF = {**{h: "slinky-4" for h in HOSTS[0:4]}, **{h: "slinky-7" for h in HOSTS[4:8]},
           **{h: "slinky-11" for h in ("slinky-10", "slinky-11", "slinky-12", "slinky-14")},
           **{h: "slinky-13" for h in ("slinky-13", "slinky-16", "slinky-17", "slinky-19")}}
SLOTS = ["slinky-4", "slinky-7", "slinky-11", "slinky-13", "slinky-18"]
PARTIAL = "slinky-18"
SEED = 20261021
LW, WORLD = 8, 128
EVAL_FRAC = 0.0005
HOLD_SALTS = ("ppm-v1", "s3eval-v1")


def ek(d, p, e):
    return DS.episode_key(d, p, e)


def main():
    out = sys.argv[1]
    ev_eps = {l.strip() for l in open("/data/home/amartyavarma/qwen-1ch-data/tt-v3-rows/eval_episodes.txt") if l.strip() and not l.startswith("#")}
    s3m = json.load(open("/data/home/amartyavarma/ppm/evalsets/speech_s3/MANIFEST.json"))
    s3_eps = {"/".join(k.split("/")[:3]) for k in s3m["row_keys"]}
    T = {s: pq.read_table(f"{P}/rows-{s}.parquet", columns=["row", "row_key", "dataset", "podcast_id", "episode_id", "chunk_index", "n_frames"]).to_pydict()
         for s in SLOTS}
    for s in SLOTS:
        assert T[s]["row"] == list(range(len(T[s]["row"]))), s
        nd = json.load(open(f"{P}/NODE-{s}.json")); assert nd["node"] == s and nd["total_rows"] == len(T[s]["row"]), s
    slot_keys = {k for s in SLOTS if s != PARTIAL for k in T[s]["row_key"]}
    st = {"rows_on_disk": {}, "eval_split": {}, "ppm_holdout": {}, "dup_partial": 0, "train": {}, "short_rows": {}, "hours": {}}
    train = {}
    for s in SLOTS:
        t, keep = T[s], []
        st["rows_on_disk"][s] = len(t["row"]); st["eval_split"][s] = 0; st["ppm_holdout"][s] = 0
        for i in range(len(t["row"])):
            e = ek(t["dataset"][i], t["podcast_id"][i], t["episode_id"][i])
            if e in ev_eps:
                st["eval_split"][s] += 1; continue
            if e in s3_eps or any(DS.is_eval_episode(e, EVAL_FRAC, sl) for sl in HOLD_SALTS):
                st["ppm_holdout"][s] += 1; continue
            if s == PARTIAL and t["row_key"][i] in slot_keys:
                st["dup_partial"] += 1; continue
            keep.append(i)
        train[s] = keep
        st["train"][s] = len(keep); st["short_rows"][s] = int(sum(1 for i in keep if t["n_frames"][i] < 7500))
        st["hours"][s] = round(sum(t["n_frames"][i] for i in keep) / 12.5 / 3600, 1)
    ranks = [[] for _ in range(WORLD)]
    rng = np.random.default_rng(SEED)
    for s in SLOTS[:4]:
        rk = [h_ * LW + l for h_, h in enumerate(HOSTS) if SLOT_OF[h] == s for l in range(LW)]
        assert len(rk) == 32, (s, rk)
        rows = np.asarray(train[s], np.int64)[rng.permutation(len(train[s]))]
        for j, r in enumerate(rows.tolist()):
            ranks[rk[j % 32]].append((SLOTS.index(s), int(r)))
    pr = np.asarray(train[PARTIAL], np.int64)[rng.permutation(len(train[PARTIAL]))]
    for r in pr.tolist():
        k = min(range(WORLD), key=lambda q: (len(ranks[q]), q))
        ranks[k].append((SLOTS.index(PARTIAL), int(r)))
    for q in range(WORLD):
        g = np.random.default_rng(np.random.SeedSequence([SEED, q]))
        ranks[q] = [ranks[q][i] for i in g.permutation(len(ranks[q]))]
    steps = min(len(x) for x in ranks)
    dropped = sum(len(x) - steps for x in ranks)
    ranks = [[list(c) for c in x[:steps]] for x in ranks]
    # eval: the Qwen tt3 run's 512 eval rows (j11717), 4 / rank in its rank order, mapped to the evalv3 store's node rows
    man = json.load(open("/data/home/amartyavarma/qwen-1ch/stage3/runs/q32-s3-tt3-from2096-j11717/eval_manifest.json"))
    et = pq.read_table("/data/home/amartyavarma/qwen-1ch-data/tt-v3-rows/evalstore/rows/rows.parquet", columns=["row", "row_key"]).to_pydict()
    pos = {k: i for i, k in enumerate(et["row_key"])}
    byr = {}
    for r in man["rows"]:
        byr.setdefault(int(r["rank"]), []).append(r["row_key"])
    assert sorted(byr) == list(range(WORLD)) and all(len(v) == 4 for v in byr.values()), (len(byr), {k: len(v) for k, v in byr.items() if len(v) != 4})
    ev = [[pos[k] for k in byr[q]] for q in range(WORLD)]
    ev_keys = [k for q in range(WORLD) for k in byr[q]]
    h = hashlib.blake2b(digest_size=16)
    for q in range(WORLD):
        for s_, r_ in ranks[q]:
            h.update(f"{q}|{s_}|{r_}\0".encode())
    tot = sum(st["train"].values())
    used_frames = sum(T[SLOTS[s_]]["n_frames"][r_] for x in ranks for s_, r_ in x)
    st.update(train_rows_total=tot, rows_per_rank=steps, steps=steps, dropped_to_even=dropped, rows_used=steps * WORLD,
              hours_used=round(used_frames / 12.5 / 3600, 1), eval_rows=len(ev_keys), eval_src=man.get("eval_rows_sha", ""),
              tt_v3_train_rows_full_dataset=143981, frac_of_full=round(steps * WORLD / 143981, 4))
    P_ = {"format": "ttconvo-ft-plan-v1", "seed": SEED, "hosts": HOSTS, "slot_of_host": SLOT_OF, "slots": SLOTS, "world": WORLD, "lw": LW,
          "steps": steps, "ranks": ranks, "eval_rows": ev, "eval_keys": ev_keys, "digest": h.hexdigest(), "stats": st,
          "excl": {"eval_split": "qwen-1ch-data/tt-v3-rows/eval_episodes.txt", "ppm_holdout_salts": list(HOLD_SALTS), "eval_frac": EVAL_FRAC,
                   "speech_s3_episodes": len(s3_eps)}}
    json.dump(P_, open(out, "w"))
    print(json.dumps(st, indent=1))
    print("digest", P_["digest"], "steps", steps)


if __name__ == "__main__":
    main()
