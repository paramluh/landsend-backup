"""aloe-nopose-ft (10-16, the no-pose ablation of job 22091): env QQA_POSE_B (default 1 = 22091 unchanged). QQA_POSE_B=0: SAME stores,
SAME items (the qualifying B tracks: pose_index b_ok selection unchanged), SAME plan / split / rank sharding, SAME speech micro; the ONLY change:
the micro carries NO pose (no pdepth / pcb0 / plight), kind 'speech' (= 21980's data_aloe micro: B = Mimi + words, A = partner's words + Mimi),
own counts = (text targets, cb0 labels, light frames).
aloe-pose-ft DATA (10-16): 21980's data_aloe with POSE ON CHANNEL B. CPU only (DataLoader workers, CPU tests).

STORES: the 8 aloe-pose-ft stores p0..p7 (build_poseB_rows.py: s3-duplex-rows-v1 + pose add-on), staged unchanged to every training node's
/scratch/amartyavarma/aloe-pose-ft/stores/pK (digests asserted against STORES.json written after the build).
ITEMS (user spec 10-16): one item per QUALIFYING B track (pose ok + no crossbleed into B) per row: (store k, node row r, swap s) with
  swap 0: model channel B = data 'assistant' (SPEAKER_00), A = data 'user'      (iff pose_index b_ok[r, 1])
  swap 1: model channel B = data 'user' (SPEAKER_01),     A = data 'assistant'  (iff pose_index b_ok[r, 0])
  A conversation whose two tracks both qualify gives both orientations; A is conditioning only (its words + Mimi, no pose, no loss).
MICRO = data_ppm.speech_micro(row, r, sib, pers, cfg, rank, swap) UNCHANGED (21980's speech-row maths: PPM sysprompt rule with both
  speakers' personas each on its own model channel, voice clips on 1/3 of rows, NEW_WORD timing targets, CD = 2, no conversation crop)
  + channel B's POSE exactly as PPM's pose rows carry it (lib/pose_s3q.segment_arrays / pose_micro):
    pdepth [N, 32]  B's pose codes of row frame f at position P + f + pose_delay (pose_delay = PPM's cfg 'pose_delay' = CD = 2: the same
                    positions as B's Mimi frame f), codebook offsets cb x 2048, -1 elsewhere (system prompt, the first CD positions)
    pcb0 [N]        pose cb0 label where pdepth >= 0 (-100 none)
    plight [Lp]     the pose 0.6B frames: pose_s3q.light_pick over those positions with PPM's pose_light_frac 1.0 (= all of them) and
                    PPM's seed words (light_seed, h64(row_key), 1, 0x9053)
  kind = 'pose' (model_ppm.PPM.forward: + inp.pose(pdepth) summed into the backbone input, speech tower on depth[1], pose tower on pdepth,
  decp_bb link; the trainer adds the pose cb0 loss and the pose decoder nanosteps).
PLAN (identical on every rank): train items permuted with seed 20261011; one item per rank per step (step t, rank R -> item t x WORLD + R);
  the last step's spare ranks run DUMMY micros (zero weights). EVAL: every eval item, round-robin over ranks (item i -> rank i % WORLD)."""
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
PS = D.PS
NST = 8
STORES = [f"/scratch/amartyavarma/aloe-pose-ft/stores/p{k}" for k in range(NST)]
SRC_STORES = [f"/data/home/amartyavarma/ppm/aloe-pose-ft/stores/p{k}" for k in range(NST)]
DIGESTS_FILE = "/data/home/amartyavarma/ppm/aloe-pose-ft/stores/STORES.json"
PLAN_SEED = 20261011
BUDGET = 2000 + 7500 + D.CD                       # sp_cap + the longest row + CD
POSE_SEED_TAG = 0x9053
POSE_B = os.environ.get("QQA_POSE_B", "1") == "1"   # aloe-nopose-ft: 0 = no pose anywhere (module doc)
assert os.environ.get("QQA_POSE_B", "1") in ("0", "1")


def aloe_cfg():
    return Q.qqa_cfg()


def store_digests():
    return json.load(open(DIGESTS_FILE))


def splits(roots=STORES):
    """-> {'train': [(k, r, swap)], 'eval': [...]} from each store's rows.parquet + pose_index.npz (b_ok per side)."""
    import pyarrow.parquet as pq
    out = {"train": [], "eval": []}
    nrows = {"train": 0, "eval": 0}
    for k, root in enumerate(roots):
        t = pq.read_table(f"{root}/rows.parquet", columns=["row", "split", "n_frames"]).to_pydict()
        assert t["row"] == list(range(len(t["row"]))), root
        z = np.load(f"{root}/pose_index.npz")
        b_ok, sp = z["b_ok"], z["split"]
        assert len(b_ok) == len(t["row"]) and [("eval" if x else "train") for x in sp] == t["split"], root
        for r, s in zip(t["row"], t["split"]):
            nrows[s] += 1
            for sw in (0, 1):                                     # swap 0 -> B = assistant (b_ok col 1); swap 1 -> B = user (col 0)
                if b_ok[r, 1 - sw]:
                    out[s].append((k, int(r), sw))
    return out, nrows


def make_plan(world, roots=STORES):
    sp, nrows = splits(roots)
    items = list(sp["train"])
    perm = np.random.default_rng(PLAN_SEED).permutation(len(items))
    items = [items[i] for i in perm]
    steps = -(-len(items) // world)
    return {"items": items, "steps": steps, "eval": sp["eval"], "n_train_rows": nrows["train"], "n_eval_rows": nrows["eval"]}


def rank_items(plan, rank, world):
    it = plan["items"]
    tr = [it[t * world + rank] if t * world + rank < len(it) else None for t in range(plan["steps"])]
    return tr, plan["eval"][rank::world]


def pose_arrays(m, row, pose_b, cfg):
    """B's pose (store frame grid [32, 7500], -1 outside the row) -> pdepth / pcb0 / plight on the micro's positions (module doc)."""
    P, n, N = int(m["P"]), int(row["n_frames"]), len(m["pid"])
    pd = int(cfg["pose_delay"])
    assert pd == D.CD, ("pose delay != the speech code delay (PPM: both CD)", pd, D.CD)
    off = np.arange(D.DEPTH, dtype=np.int64) * D.CBS
    pdepth = np.full((N, D.DEPTH), -1, np.int64)
    pos = P + pd + np.arange(n)
    p = np.asarray(pose_b[:, :n]).T.astype(np.int64)
    assert (p >= 0).all() and (p < D.CBS).all(), ("B pose codes missing / out of range on a qualifying track", row["row_key"])
    pdepth[pos] = p + off
    # alignment with B's Mimi: frame f of the conversation sits at P + CD + f in depth[1] (the conversation part of the micro)
    db = np.asarray(m["depth"][1])
    conv_m = db[P:, 0] >= 0
    assert np.array_equal(np.flatnonzero(conv_m) + P, pos), ("B Mimi / pose positions differ", row["row_key"])
    assert np.array_equal(db[pos] - off, np.asarray(row[f"codes_{'user' if m['swap'] else 'assistant'}"])[:, :n].T.astype(np.int64)), "B side"
    pcb0 = np.where(pdepth[:, 0] >= 0, pdepth[:, 0], -100).astype(np.int64)
    plight = PS.light_pick(np.flatnonzero(pdepth[:, 0] >= 0), float(cfg["pose_light_frac"]),
                           (cfg["light_seed"], PS.h64(row["row_key"]), 1, POSE_SEED_TAG)).astype(np.int64)
    return pdepth, pcb0, plight


class Stores:
    def __init__(self, roots=STORES):
        self.roots, self.s = list(roots), {}

    def open(self, k):
        if k not in self.s:
            root = self.roots[k]
            node = D.R3.load_node(root)
            dg = store_digests()[f"p{k}"]
            assert node["digest"] == dg["node_digest"], ("staged store differs from the build", root, node["digest"])
            st = D.R3.Rows(node)
            name = node["shards"][0]
            assert len(node["shards"]) == 1
            pz = {s: np.load(f"{root}/{name}.pose_{s}.npy", mmap_mode="r") for s in ("user", "assistant")}
            z = np.load(f"{root}/pose_index.npz")
            self.s[k] = (node, st, D.SPR.SibIndex(node, st), D.PER.PersonaSidecar(node), pz, z["b_ok"])
        return self.s[k]

    def conv(self, item, cfg, rank):
        k, r, sw = item
        node, st, sib, ps, pz, b_ok = self.open(int(k))
        row = st[int(r)]
        D.R3.check_row(row)
        bside = "user" if int(sw) else "assistant"
        assert b_ok[int(r), 0 if bside == "user" else 1], ("item's B side does not qualify", item)
        try:
            pers = ps.get(int(r), row["row_key"])
        except (FileNotFoundError, OSError):
            pers = None
        m = D.speech_micro(row, int(r), sib, pers, cfg, rank, swap=int(sw))
        T = len(m["pid"])
        assert m["segs"] == [T] and np.array_equal(m["pid"], np.arange(T)) and m["swap"] == int(sw)
        assert T == int(m["P"]) + int(row["n_frames"]) + D.CD <= BUDGET, ("segment length", T, m["P"], row["n_frames"])
        if POSE_B:
            m["pdepth"], m["pcb0"], m["plight"] = pose_arrays(m, row, pz[bside][int(r)], cfg)
        m["row_key"] = row["row_key"]
        m["item"] = [int(k), int(r), int(sw)]
        return m


def own_counts(m):
    """(text targets, cb0 labels, light frames, pose cb0 labels, pose light frames) of one conversation."""
    if not POSE_B:                                                # no-pose ablation: 21980's (text, cb0, light) counts
        return Q.own_counts(m)
    return Q.own_counts(m) + [float((np.asarray(m["pcb0"]) >= 0).sum()), float(len(m["plight"]))]


def as_micro(m, dummy=False):
    """one conversation -> the trainer's micro (data_qqa.pack of one = its own segment) + the pose arrays; dummy: zero counts."""
    out = Q.pack([m], budget=BUDGET)
    if POSE_B:
        out["kind"] = "pose"
        out["pdepth"], out["pcb0"], out["plight"] = m["pdepth"], m["pcb0"], m["plight"]
    else:
        assert out["kind"] == "speech" and "pdepth" not in m, "no-pose micro must be a plain speech micro"
    out["own"] = np.asarray([own_counts(m)], np.float64)
    out["swap"] = int(m["swap"])
    out["item"] = json.dumps(m["item"])
    if dummy:
        out["dummy"] = 1
        out["own"] = np.zeros_like(out["own"])
    return out


class AloeDS:
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
            out["singles"] = [as_micro(m)]
        return out


def eval_micros(items, cfg, rank, roots=STORES):
    st = Stores(roots)
    return [as_micro(st.conv(it, cfg, rank)) for it in items]
