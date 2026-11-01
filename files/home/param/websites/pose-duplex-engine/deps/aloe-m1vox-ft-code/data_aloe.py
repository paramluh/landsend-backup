"""aloe-ft DATA: Model 1's Aloe handover rows (stage-2 handover 'none' crossbleed bucket, s3-duplex-rows-v1 stores m1n0..m1n3 =
/home/param/websites/pose-duplex-engine/deps/qwen-1ch-data/s2h-m1/stores, staged UNCHANGED to every training node's /scratch/amartyavarma/aloe-ft/stores/m1nK)
in PPM's speech-row format. CPU only (DataLoader workers, CPU tests).

BOTH SIDES AS B (user spec 10-16): both speakers' tracks are clean, so every row is used TWICE per epoch:
  swap 0: model channel B = data 'assistant' (SPEAKER_00), A = data 'user' (conditioning only, no loss)
  swap 1: model channel B = data 'user' (SPEAKER_01),     A = data 'assistant'
PPM's loss is on B only. Each orientation = data_ppm.speech_micro(row, node_row, sib, pers, cfg, rank, swap) UNCHANGED = PPM's own
speech-row maths and PPM's system-prompt rule (data_ppm.podcast_prefix): persona-sidecar text of BOTH speakers (persona_sides both),
each block on its own speaker's model channel (follows the swap), voice clips of both speakers on 1/3 of rows (later chunk in the
recording's second half), cap 2,000 positions. Data cfg = qqa-ft's (data_qqa.qqa_cfg = PPM prod's data cfg with crop_conv False:
the conversation is never cropped by the prefix: T = P + n + CD).

PLAN (deterministic, identical on every rank): items = (store k, node row r, swap s) for every TRAIN row (rows.parquet split == 'train')
of the 4 stores x s in (0, 1) = 23,400 items, permuted with seed 20261011. ONE item per rank per step (a 7,500-frame row fills a micro;
qqa's ~17.8 conversations per micro is not meaningful here): step t, rank R -> item t x WORLD + R. The last step is partly empty: those
ranks run a DUMMY micro (item 0's data, zero loss weight, zero counts: contributes nothing to any loss, gradient or statistic).
EVAL: every EVAL row (split == 'eval', 240) x both orientations = 480 micros, round-robin over ranks (item i -> rank i % WORLD); ranks
with fewer run dummies (ignored)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "lib"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import data_qqa as Q  # noqa: E402

D = Q.D
STORES = [f"/scratch/amartyavarma/aloe-ft/stores/m1n{k}" for k in range(4)]
SRC_STORES = [f"/home/param/websites/pose-duplex-engine/deps/qwen-1ch-data/s2h-m1/stores/m1n{k}" for k in range(4)]
DIGESTS = ["c1c3b2517152741137cc4bb72dd6bdbb", "09bdd43fef84574175caacf4abdddd1e", "ecb1c731e8989b3f2e1b5266b51ea0fd",
           "3b16fb8dd2660a2e8c1fe32c0d361d7b"]
PLAN_SEED = 20261011
BUDGET = 2000 + 7500 + D.CD                       # sp_cap + the longest row + CD


def aloe_cfg():
    return Q.qqa_cfg()


def splits(roots=STORES):
    """-> {'train': [(k, r)], 'eval': [(k, r)]} from each store's rows.parquet (row == node row asserted)."""
    import pyarrow.parquet as pq
    out = {"train": [], "eval": []}
    for k, root in enumerate(roots):
        t = pq.read_table(f"{root}/rows.parquet", columns=["row", "split", "n_frames"]).to_pydict()
        assert t["row"] == list(range(len(t["row"]))), root
        for r, s in zip(t["row"], t["split"]):
            out[s].append((k, int(r)))
    return out


def make_plan(world, roots=STORES):
    sp = splits(roots)
    items = [(k, r, s) for (k, r) in sp["train"] for s in (0, 1)]
    perm = np.random.default_rng(PLAN_SEED).permutation(len(items))
    items = [items[i] for i in perm]
    steps = -(-len(items) // world)
    ev = [(k, r, s) for (k, r) in sp["eval"] for s in (0, 1)]
    return {"items": items, "steps": steps, "eval": ev, "n_train_rows": len(sp["train"]), "n_eval_rows": len(sp["eval"])}


def rank_items(plan, rank, world):
    """-> (train: [item or None (dummy)] per step, eval: [item] of this rank)."""
    it = plan["items"]
    tr = [it[t * world + rank] if t * world + rank < len(it) else None for t in range(plan["steps"])]
    return tr, plan["eval"][rank::world]


class Stores:
    """the 4 node-local stores, opened lazily (per DataLoader worker)."""

    def __init__(self, roots=STORES):
        self.roots, self.s = list(roots), {}

    def open(self, k):
        if k not in self.s:
            node = D.R3.load_node(self.roots[k])
            assert node["digest"] == DIGESTS[k], ("staged store differs from Model 1's", self.roots[k], node["digest"])
            st = D.R3.Rows(node)
            self.s[k] = (node, st, D.SPR.SibIndex(node, st), D.PER.PersonaSidecar(node))
        return self.s[k]

    def conv(self, item, cfg, rank):
        k, r, sw = item
        node, st, sib, ps = self.open(int(k))
        row = st[int(r)]
        D.R3.check_row(row)
        try:
            pers = ps.get(int(r), row["row_key"])
        except (FileNotFoundError, OSError):
            pers = None
        m = D.speech_micro(row, int(r), sib, pers, cfg, rank, swap=int(sw))
        T = len(m["pid"])
        assert m["segs"] == [T] and np.array_equal(m["pid"], np.arange(T)) and m["swap"] == int(sw)
        assert T == int(m["P"]) + int(row["n_frames"]) + D.CD <= BUDGET, ("segment length", T, m["P"], row["n_frames"])
        m["row_key"] = row["row_key"]
        m["item"] = [int(k), int(r), int(sw)]
        return m


def as_micro(m, dummy=False):
    """one conversation -> the trainer's micro (data_qqa.pack of one = its own segment); dummy: zero counts (trainer zeroes its weights)."""
    out = Q.pack([m], budget=BUDGET)
    out["swap"] = int(m["swap"])
    out["item"] = json.dumps(m["item"])
    if dummy:
        out["dummy"] = 1
        out["own"] = np.zeros_like(out["own"])
    return out


class AloeDS:
    """one rank's training stream: item t = the micro of step t (dummy where the plan is exhausted)."""

    def __init__(self, items, cfg, rank, fallback, singles_at=0, roots=STORES):
        self.st, self.items, self.cfg, self.rank, self.fb, self.singles_at = Stores(roots), list(items), dict(cfg), int(rank), fallback, singles_at

    def __len__(self):
        return len(self.items)

    def __getitem__(self, t):
        it = self.items[int(t)]
        m = self.st.conv(it if it is not None else self.fb, self.cfg, self.rank)
        out = as_micro(m, dummy=it is None)
        out["step"] = int(t)
        if int(t) == self.singles_at:
            out["singles"] = [m]
        return out


def eval_micros(items, cfg, rank, roots=STORES):
    st = Stores(roots)
    return [as_micro(st.conv(it, cfg, rank)) for it in items]
