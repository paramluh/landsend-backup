"""aloe-voxall-ft (10-16): = aloe-vox-ft's data_aloe_vox with env QQA_POSE_B=0 support (data_aloe_pose.POSE_B, the no-pose run 22291's
switch): no pose arrays loaded / produced, micro kind 'speech'; stores = build_voxall_rows.py (ALL Aloe handover conversations, every
speaker track a B candidate except bleed into B). Everything else unchanged.
aloe-vox-ft DATA (10-16): the aloe-pose-ft micro (data_aloe_pose: channel B = that speaker's words + Mimi + POSE, B losses) with channel A
REPLACED ENTIRELY by the other speaker's Voxtral latents. CPU only (DataLoader workers, CPU tests).

STORES: 8 conversation-sharded stores p0..p7 (build_voxB_rows.py: the pose FT's rows, re-partitioned balanced by B items) + vox_index.json
  (channel-A audio reference per row / side). Node i of N holds stores {k : k % N == i} (prod N = 8: node i = store i only) on its
  $ALOE_SCRATCH/aloe-voxall-ft/stores/pK, digests asserted against plan/STORES.json; A's FLACs of those stores on its ALOE_FLAC_ROOT.
ITEMS (= the pose FT's): one per QUALIFYING B track per row: (store k, row r, swap s); swap 0: B = data 'assistant' (b_ok[r, 1]), swap 1:
  B = data 'user' (b_ok[r, 0]). A = the other data side.
MICRO = data_ppm.speech_micro maths (21980 / pose FT) with these channel-A changes ONLY:
  * system prompt: channel A's VOICE CLIP is dropped (it is A's Mimi codes; user 10-13 'no side-A voice clips' + this run 'no A Mimi'),
    B's voice clip kept with the same draw (data_ppm.voice_clips unchanged, only A's segment skipped); both personas kept ('personas per
    speaker as before'; A's persona block stays on channel A's text table in the prefix, as stage-4 run F kept A's prefix text).
  * conversation: channel A has NO text (ids PAD), NO Mimi (depth -1 everywhere on A, prefix included) and present_A = 0 at every
    conversation position p >= P (A's tables contribute nothing there, = run F's tmask[0, P:] = 0 with S4_A_MIMI=0); A has no loss.
  * vin = (P, n): the trainer adds vadapter(Voxtral(A audio of row frames 0..n-1)) at positions P..P+n-1 (row frame f at P + f = where
    A's word lane sat; run F data_s4q.row_arrays_s4 'vin': frame f at P + f, NO delay; the ASR adapter run: clip frame f at position f).
  B (lane, Mimi at CD 2, pose at CD 2, targets, light / pose-light frames) is bit-identical to the pose FT's micro for the same row
  except for the shift by the shorter prefix on rows whose A voice clip is dropped.
A AUDIO (worker side): audio_s4q.ffmpeg_row_audio (stage-4 handover rows, the Mimi encoder's preprocessing): ffmpeg -f f32le -ac 1 -ar 24000
  of the speaker's WLT FLAC (clip-relative, = Mimi frame 0), samples [1920 frame0, 1920 (frame0 + n)) zero-padded past the end (shortfall
  <= 0.25 s asserted), x 10^(gain_db / 20) with the RECORDED Mimi LUFS gain of that speaker (w00.mimi.json). The trainer's frozen Voxtral
  front (audio_s4q.VoxFront = voxtral_front.VoxtralFront) resamples 24 -> 16 kHz, log-mel on GPU, encodes -> [4n, 1280] (50 Hz).
PLAN: node i's train items permuted with seed (20261011, i); rank (i, j) at step t gets node item 8 t + j; steps = max_i ceil(n_i / 8)
  (= 430 for the prod partition); spare slots run DUMMY micros (zero weights). EVAL: node i's eval items round-robin over its 8 ranks."""
_ALOE_ROOT = __import__("os").environ["ALOE_ROOT"]        # release: every path comes from aloe.env (ALOE_ROOT)
_ALOE_SCRATCH = __import__("os").environ["ALOE_SCRATCH"]  # release: node-local caches (aloe.env)
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "lib"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import data_qqa as Q  # noqa: E402
import data_aloe_pose as AP  # noqa: E402   (pose_arrays / own_counts: the pose FT's, unchanged)

D = Q.D
SPR = D.SPR
NST = 8
STORE_ROOT = os.environ.get("ALOE_STORE_ROOT", f"{_ALOE_SCRATCH}/aloe-voxall-ft/stores")
FLAC_ROOT = os.environ.get("ALOE_FLAC_ROOT", f"{_ALOE_SCRATCH}/aloe-voxall-ft/flac")
DIGESTS_FILE = os.environ.get("ALOE_STORES_JSON", f"{_ALOE_ROOT}/ppm/aloe-voxall-ft/plan/STORES.json")
PLAN_SEED = 20261011
BUDGET = AP.BUDGET
LW = 8


def aloe_cfg():
    return Q.qqa_cfg()


def store_root(k):
    return f"{STORE_ROOT}/p{k}"


def node_stores(node, nnodes):
    assert NST % nnodes == 0, (NST, nnodes)
    return [k for k in range(NST) if k % nnodes == node]


def store_meta():
    return json.load(open(DIGESTS_FILE))


def make_plan(world):
    """-> per node: train item list (permuted) and eval items; steps. From STORES.json (identical on every rank; each rank later asserts
    its own stores' digests / item lists against the staged files)."""
    sm = store_meta()
    nn = world // LW
    assert nn * LW == world, world
    tc = os.environ.get("ALOE_TEST_CONVS", "")                    # TEST ONLY (GPU test on 2 spare nodes): train items restricted to these
    tc = set(json.load(open(tc))) if tc else None                 # conversations (the ones whose A FLACs the test fetched)
    nodes = []
    for i in range(nn):
        tr, ev = [], []
        for k in node_stores(i, nn):
            s = sm[f"p{k}"]
            tr += [(k, int(r), int(sw)) for r, sw in s["items_train"] if tc is None or s["row_convs"][int(r)] in tc]
            ev += [(k, int(r), int(sw)) for r, sw in s["items_eval"]]
        perm = np.random.default_rng([PLAN_SEED, i]).permutation(len(tr))
        nodes.append({"items": [tr[j] for j in perm], "eval": ev})
    steps = max(-(-len(n_["items"]) // LW) for n_ in nodes)
    return {"nodes": nodes, "steps": steps, "n_train": sum(len(n_["items"]) for n_ in nodes), "n_eval": sum(len(n_["eval"]) for n_ in nodes)}


def rank_items(plan, rank):
    node, j = rank // LW, rank % LW
    it = plan["nodes"][node]["items"]
    tr = [it[t * LW + j] if t * LW + j < len(it) else None for t in range(plan["steps"])]
    return tr, plan["nodes"][node]["eval"][j::LW], it[0]


def local_items(k):
    """(train, eval) item lists of store k from its staged files (rows.parquet split + pose_index b_ok) = data_aloe_pose.splits' rule."""
    import pyarrow.parquet as pq
    root = store_root(k)
    t = pq.read_table(f"{root}/rows.parquet", columns=["row", "split"]).to_pydict()
    assert t["row"] == list(range(len(t["row"]))), root
    z = np.load(f"{root}/pose_index.npz")
    b_ok, sp = z["b_ok"], z["split"]
    assert len(b_ok) == len(t["row"]) and [("eval" if x else "train") for x in sp] == t["split"], root
    out = {"train": [], "eval": []}
    for r, s in zip(t["row"], t["split"]):
        for sw in (0, 1):
            if b_ok[r, 1 - sw]:
                out[s].append([int(r), sw])
    return out["train"], out["eval"]


# ------------------------------------------------------------------------------------------------ system prompt without A's voice clip
def podcast_prefix_vox(row, swap, sib, pers, cfg):
    """= data_ppm.podcast_prefix with channel A's voice-clip segment dropped (B's clip: same draw, same window). -> (prefix, provenance)."""
    sw = int(swap)
    want_voice = D.h64(f"ppm-voice|{int(cfg['seed'])}|{row['row_key']}") % int(cfg["voice_mod"]) == 0
    segs, pv = [], {"voice_drawn": int(want_voice)}
    clips = D.voice_clips(row, sib, cfg) if want_voice else {0: None, 1: None}
    pv["a_clip_dropped"] = int(clips[0 ^ sw] is not None)
    for c in (1,):                                                   # VOX: model channel B only (A's clip = A Mimi codes: dropped)
        cl = clips[c ^ sw]
        if cl is not None:
            segs.append({"kind": "speech", "track": c, **cl})
    pv["clips"] = [{"track": "AB"[s["track"]], "data_side": s["side"], "src_key": s["src_key"], "src_chunk": s["src_chunk"],
                    "frames": [s["w0"], s["w0"] + s["L"]], "cov": round(s["cov"], 3)} for s in segs]
    fixed = 5 + sum(int(s["L"]) + 5 for s in segs)
    assert fixed <= int(cfg["sp_cap"]), ("voice clips alone exceed the system-prompt cap", fixed)
    allow = {"both": (0, 1), "B": (1,), "A": (0,), "none": ()}[cfg["persona_sides"]]
    blocks, chans = [], []
    if pers is not None and allow:
        D._tok_init()
        for c in (0, 1):
            if c not in allow:
                continue
            av = D.PT.avail(pers.get(D.PER.SIDES[c ^ sw]))
            keys = [k for k in D.PT.SECTIONS if k in av]
            if keys:
                try:
                    ids = D.PT.encode(D.PT.block_text(av, keys))
                except AssertionError as e:
                    print(f"[data] persona block skipped row {row['row_key']} side {D.PER.SIDES[c ^ sw]}: {e}", file=sys.stderr, flush=True)
                    pv["persona_skipped"] = pv.get("persona_skipped", 0) + 1
                    continue
                blocks.append([D.PT.SOP] + ids + [D.PT.EOP]); chans.append(c)
    blocks2, cut = D.crop_blocks(blocks, int(cfg["sp_cap"]) - fixed)
    segs_t = [{"kind": "text", "track": c, "ids": b2} for c, b2 in zip(chans, blocks2) if b2 is not None]
    segs += segs_t
    pv.update(persona=[{"track": "AB"[s["track"]], "tokens": len(s["ids"])} for s in segs_t], persona_cut=int(cut))
    if not segs:
        segs = [{"kind": "text", "track": 1, "ids": []}]
    spx = SPR.build_prefix(row, segs, {"new_word": cfg["new_word"]})
    assert 5 <= spx["P"] <= int(cfg["sp_cap"]), ("system prompt length", spx["P"])
    pv["P"] = int(spx["P"])
    return spx, pv


def speech_micro_vox(row, node_row, sib, pers, cfg, rank, swap):
    """data_ppm.speech_micro with podcast_prefix_vox, then channel A emptied in the conversation (module doc)."""
    _orig = D.podcast_prefix
    D.podcast_prefix = podcast_prefix_vox
    try:
        m = D.speech_micro(row, node_row, sib, pers, cfg, rank, swap=swap)
    finally:
        D.podcast_prefix = _orig
    P, n = int(m["P"]), int(row["n_frames"])
    assert (m["depth"][0, :P] < 0).all(), "an A code in the prefix (A's voice clip must be dropped)"
    m["ids"] = m["ids"].copy(); m["depth"] = m["depth"].copy(); m["present"] = m["present"].copy()
    m["ids"][0, P:] = D.PAD_ID
    m["depth"][0] = -1
    m["present"][0, P:] = 0.0
    m["vin"] = (P, n)
    pv = json.loads(m["prov"]); pv["chan_A"] = "voxtral_latents_only"; m["prov"] = json.dumps(pv)
    return m


# ------------------------------------------------------------------------------------------------ stores + audio
class Stores:
    def __init__(self, ks):
        self.ks, self.s = list(ks), {}
        self.dg = store_meta()

    def open(self, k):
        assert k in self.ks, ("store not on this node", k, self.ks)
        if k not in self.s:
            root = store_root(k)
            node = D.R3.load_node(root)
            dg = self.dg[f"p{k}"]
            assert node["digest"] == dg["node_digest"], ("staged store differs from the build", root, node["digest"])
            pj = json.load(open(f"{root}/POSE.json"))
            assert pj["pose_digest"] == dg["pose_digest"], ("pose add-on differs", root)
            st = D.R3.Rows(node)
            name = node["shards"][0]
            assert len(node["shards"]) == 1
            pz = ({s: np.load(f"{root}/{name}.pose_{s}.npy", mmap_mode="r") for s in ("user", "assistant")} if AP.POSE_B
                  else None)                                         # aloe-voxall-ft QQA_POSE_B=0: stores carry no pose arrays
            z = np.load(f"{root}/pose_index.npz")
            vi = json.load(open(f"{root}/vox_index.json"))["rows"]
            assert len(vi) == len(z["b_ok"])
            self.s[k] = (node, st, D.SPR.SibIndex(node, st), D.PER.PersonaSidecar(node), pz, z["b_ok"], vi)
        return self.s[k]

    def conv(self, item, cfg, rank, audio=True):
        import audio_s4q as AU
        import rowcache_vox as RC                                   # SPEED FIX 10-16: node-local decoded-row cache (bit-identical)
        k, r, sw = item
        node, st, sib, ps, pz, b_ok, vi = self.open(int(k))
        row = st[int(r)]
        D.R3.check_row(row)
        bside = "user" if int(sw) else "assistant"
        aside = "assistant" if int(sw) else "user"
        assert b_ok[int(r), 0 if bside == "user" else 1], ("item's B side does not qualify", item)
        try:
            pers = ps.get(int(r), row["row_key"])
        except (FileNotFoundError, OSError):
            pers = None
        m = speech_micro_vox(row, int(r), sib, pers, cfg, rank, swap=int(sw))
        T = len(m["pid"])
        assert m["segs"] == [T] and np.array_equal(m["pid"], np.arange(T)) and m["swap"] == int(sw)
        assert T == int(m["P"]) + int(row["n_frames"]) + D.CD <= BUDGET, ("segment length", T, m["P"], row["n_frames"])
        if AP.POSE_B:                                                # aloe-voxall-ft: QQA_POSE_B=0 -> plain speech micro (no pose arrays)
            m["pdepth"], m["pcb0"], m["plight"] = AP.pose_arrays(m, row, pz[bside][int(r)], cfg)
        e = vi[int(r)]
        assert e["row_key"] == row["row_key"] and aside in e["A"], ("vox_index row / side", item, e["row_key"], row["row_key"])
        a = e["A"][aside]
        assert int(a["n_frames"]) == int(row["n_frames"]) and int(a["frame0"]) == int(row["frame_start"]), (item, a, row["frame_start"])
        assert a["a_spk"] == row[f"{aside}_speaker_dir"], ("A audio speaker != the row's A side", a["a_spk"], row[f"{aside}_speaker_dir"])
        m["aref"] = {"row_key": row["row_key"], "side": aside, "flac": a["flac"], "gain_db": float(a["gain_db"]), "frame0": int(a["frame0"]),
                     "n": int(a["n_frames"])}
        if audio:
            path = f"{FLAC_ROOT}/{a['flac']}"
            assert os.path.getsize(path) == int(a["flac_bytes"]), ("A FLAC missing / wrong size on this node", path)
            m["wav"] = RC.row_audio({"flac": path, "frame0": int(a["frame0"]), "n_frames": int(a["n_frames"]), "row_key": row["row_key"]},
                                           float(a["gain_db"]))
            assert m["wav"].dtype == np.float32 and m["wav"].shape == (int(a["n_frames"]) * AU.SAMPLES_PER_FRAME_24K,)
        m["row_key"] = row["row_key"]
        m["item"] = [int(k), int(r), int(sw)]
        return m


def as_micro(m, dummy=False):
    out = AP.as_micro(m, dummy=dummy)
    assert out["segs"] == [len(m["pid"])]
    out["present"] = m["present"]
    out["vin"] = np.asarray(m["vin"], np.int64)
    out["aref"] = json.dumps(m["aref"])
    if "wav" in m:
        out["wav"] = m["wav"]
    return out


class VoxDS:
    def __init__(self, items, ks, cfg, rank, fallback, singles_at=0):
        self.items, self.ks, self.cfg, self.rank, self.fb, self.singles_at = list(items), list(ks), dict(cfg), int(rank), fallback, singles_at
        self.st = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, t):
        if self.st is None:
            self.st = Stores(self.ks)
        it = self.items[int(t)]
        m = self.st.conv(it if it is not None else self.fb, self.cfg, self.rank)
        out = as_micro(m, dummy=it is None)
        out["step"] = int(t)
        if int(t) == self.singles_at:
            s1 = as_micro(m)
            s1.pop("wav", None)                                      # the loss check reuses the micro's own wav / latents
            out["singles"] = [s1]
        return out


def eval_micros(items, ks, cfg, rank):
    st = Stores(ks)
    return [as_micro(st.conv(it, cfg, rank)) for it in items]
