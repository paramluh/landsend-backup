"""aloe-ft-32-samsep POSE-DUPLEX DATA (10-25, user via main: "the same as the SAM speech duplex with pose summed in on the assistant side",
"train for a bit longer") = data_aloe_v2vox.py (job 29345's data path, imported UNCHANGED; data_aloe_v2 / data_ppm / data_qqa / data_aloe stay
byte-identical and are sha-asserted by the trainer) + channel B's POSE exactly as job 26089 (aloe-v2pure-16610/code/data_aloe_pose.py
pose_arrays, PPM's pose rows) adds it, + a multi-epoch plan.

POSE ON B (POSEDUP_POSE=1): for item (store k, row r, swap s) the channel-B speaker is data 'assistant' (SPEAKER_00) on swap 0, data 'user'
  (SPEAKER_01) on swap 1 (data_aloe_v2 rule). Its pose tokens are read IN PLACE from the handover copy (aloe-handover-full/DATA/files/none/
  <session>/<seg>/w00.SPEAKER_0N.pose_tokens.npy, int16 [32, T], 12.5 Hz, pose tokenizer noshift@1d6ff4f0 = PPM's / v5's), pinned by
  POSE_INDEX.json (tools/build_pose_index.py: sha256 == the pose json's, T == mimi_T == the row's src_frames; the file's sha256 is re-checked
  on EVERY read). First chunks start at frame 0, so row frame f = pose frame f. The micro gets (26089's pose_arrays, verbatim maths):
    pdepth [N, 32]  B's pose codes of row frame f at position P + pose_delay + f (pose_delay = CD = 2: the SAME positions as B's Mimi frame f,
                    asserted per micro), + cb x 2048 offsets, -1 elsewhere (system prompt, first CD positions)
    pcb0 [N]        pose cb0 label where pdepth >= 0 (-100 none)
    plight [Lp]     the pose 0.6B decoder frames: pose_s3q.light_pick over those positions with pose_light_frac (cfg, default 1.0 = PPM's /
                    26089's: all of them), seed words (light_seed, h64(row_key), 1, 0x9053)
  kind 'pose' (model_ppm.PPM.forward: e += inp.pose(pdepth) summed into the backbone input with B's lane / Mimi and A's adapter latents;
  speech tower on depth[1], pose tower on pdepth, decp_bb link; the trainer adds the pose cb0 loss and the pose decoder nanosteps).
  Items whose B side has NO pose (pose_status flagged / error: 14.6 % of train micros) under POSEDUP_ITEMS=all: the SAME micro with
  pdepth -1 everywhere (inp.pose adds exactly 0: code tables map -1 to 0), pcb0 -100, plight empty, pose_ok 0 (zero pose input, zero pose loss;
  the trainer joins the pose decoder's per-step collective with zeros). Every speech field (ids / depth / present / targets / light / A audio /
  voice clip) is the 29345 micro, bit for bit (tests/identity.py).
POSEDUP_POSE=0: every micro = data_aloe_v2vox's (job 29345), bit-identical (the module only re-exports it).

KNOBS (env, read at import; identical on every rank / DataLoader worker):
  POSEDUP_POSE            1 = pose on B (default 0 = job 29345 exactly)
  POSEDUP_ITEMS           all (default: the 15,898 train micros of 29345's plan; pose where the B side has it) | pose_ok (only items whose B
                          side has pose: 29345's order filtered; 13,578 micros)
  POSEDUP_POSE_INDEX      POSE_INDEX.json path; POSEDUP_POSE_INDEX_SHA its sha256 (asserted)
  POSEDUP_POSE_LIGHT_FRAC pose decoder frame fraction (default 1.0 = 26089 / PPM). Speech light stays the data cfg's 0.1 unless
  POSEDUP_SPEECH_LIGHT_FRAC is set (default 0.1 = 29345; the trainer's 10 % band env must then follow).
  POSEDUP_VOICE_RULE      v2_30 (default: 29345's B voice clip on 30 %, h64('aloe-v2-voice|seed|row_key|swap') % 10 < 3) | ppm_third (26089's
                          draw: h64('ppm-voice|seed|row_key') % voice_mod(3) == 0, B's clip only, A none; the clip itself = the same
                          data_ppm.voice_clips rule)
  POSEDUP_EPOCH_SEED      later-epoch permutation seed base (default 20261011): epoch 0 = 29345's plan order EXACTLY; epoch e >= 1 = a fresh
                          permutation np.random.default_rng([seed, e]) of the same items. Each epoch is padded to whole steps with dummies
                          (= 29345's last-step dummies), so epoch e occupies steps [e x E, (e + 1) x E), E = ceil(items / 64) (249 for 'all').
FRAMING_COND (10-29, framingprep for main -1e; prep only, NOT launched; env FRAMING_COND=1, needs POSEDUP_POSE=1; default 0 = every micro and
cfg_view() byte-identical to before): every pose micro additionally carries frpos / fridx / frf (framing.py): ONE standardised framing row =
channel B's track (b_side(item): the data side the item puts on B; only B carries pose in this model, so each pose position gets its own
channel's framing) from the side arrays FRAMING_DIR/m1n<k>/framing_<b side>.npy (store digest == this run's store, asserted), stats
FRAMING_DIR/FRAMING_STATS.json (sha FRAMING_STATS_SHA if set). A B side with pose: status 0 asserted (fail loud). A pose-less B (pose_ok 0,
or a bandmask DROP lane): a zero feature row and no positions (the MLP still runs: every rank runs every FSDP unit). frpos = the positions
with pdepth[:, 0] >= 0 after the band mask."""
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "lib"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import data_aloe_v2vox as V  # noqa: E402   (job 29345's data module, unchanged copy; sha asserted by the trainer)

A2, D, Q = V.A2, V.D, V.Q
PS = D.PS
STORES, SRC_STORES, BUDGET = V.STORES, V.SRC_STORES, V.BUDGET
FLAC_ROOT, INDEX, SAMSEP, RAWGAIN_PLAN_DIGEST, GAIN_MODE, RAWGAIN = V.FLAC_ROOT, V.INDEX, V.SAMSEP, V.RAWGAIN_PLAN_DIGEST, V.GAIN_MODE, V.RAWGAIN
splits, rank_items = V.splits, V.rank_items
rawgain, a_gain, vox_index, a_entry, node_items = V.rawgain, V.a_gain, V.vox_index, V.a_entry, V.node_items


def _env(k, d):
    v = os.environ.get(k, "")
    return d if v == "" else v


POSE = _env("POSEDUP_POSE", "0") == "1"
assert _env("POSEDUP_POSE", "0") in ("0", "1")
ITEMS = _env("POSEDUP_ITEMS", "all")
assert ITEMS in ("all", "pose_ok"), ITEMS
VOICE_RULE = _env("POSEDUP_VOICE_RULE", "v2_30")
assert VOICE_RULE in ("v2_30", "ppm_third"), VOICE_RULE
POSE_LIGHT_FRAC = float(_env("POSEDUP_POSE_LIGHT_FRAC", "1.0"))
SPEECH_LIGHT_FRAC = float(_env("POSEDUP_SPEECH_LIGHT_FRAC", "0.1"))
EPOCH_SEED = int(_env("POSEDUP_EPOCH_SEED", str(A2.PLAN_SEED)))
POSE_INDEX = _env("POSEDUP_POSE_INDEX", "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/data/POSE_INDEX.json")
POSE_INDEX_SHA = _env("POSEDUP_POSE_INDEX_SHA", "")
POSE_SHA_EVERY_READ = _env("POSEDUP_POSE_SHA_EVERY_READ", "1") == "1"
POSE_SEED_TAG = 0x9053                                            # = 26089 / PPM's pose light seed word
assert 0.0 < POSE_LIGHT_FRAC <= 1.0 and 0.0 < SPEECH_LIGHT_FRAC <= 1.0
assert ITEMS == "all" or POSE, "POSEDUP_ITEMS=pose_ok needs POSEDUP_POSE=1"
_PIX = [None]
# bandmask (10-29, prep only, NOT launched): black-band (no-detection) handling of channel B's pose. Default OFF: every micro, the plan and
# cfg_view() are byte-identical to job 30074's (tests: bandmask/tests/test_posedup.py).
#   POSE_BAND_MASK=1        (needs POSEDUP_POSE=1) (a) MASK: B pose frames inside a band (index no-detection intervals of B's speaker, 1 token
#                           before entry .. 3 after exit, sub-second blips too, + frozen-token runs; POSE_BAND_MASK_DIR/m1n<k>/pose_band_mask_
#                           <side>.npy, bool [R, 7500] on the row frame grid) carry NO pose cb0 label (pcb0 -100) and NO pose decoder frame
#                           (dropped from plight); the step normalisers follow (own counts). (b) DROP: lanes whose clip starts inside a band
#                           (frame-0 junk skeleton -> wrong normalisation for the whole lane; DROP_LANES.json) = pose-less B (pose_ok 0: zero
#                           pose input, zero pose loss), exactly as flagged / error tracks; under POSEDUP_ITEMS=pose_ok they leave the plan.
#   POSE_BAND_MASK_INPUT=1  (needs POSE_BAND_MASK=1; default 0) also blanks the masked frames' pose INPUT (pdepth -1 -> inp.pose adds 0, the
#                           tower sees 'absent'), instead of the frozen junk-skeleton codes. Proposed only; not needed for the loss mask.
POSE_BAND_MASK = _env("POSE_BAND_MASK", "0") == "1"
assert _env("POSE_BAND_MASK", "0") in ("0", "1")
POSE_BAND_MASK_INPUT = _env("POSE_BAND_MASK_INPUT", "0") == "1"
assert _env("POSE_BAND_MASK_INPUT", "0") in ("0", "1")
POSE_BAND_MASK_DIR = _env("POSE_BAND_MASK_DIR", "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/data/bandmask")
assert not POSE_BAND_MASK or POSE, "POSE_BAND_MASK=1 needs POSEDUP_POSE=1"
assert not POSE_BAND_MASK_INPUT or POSE_BAND_MASK, "POSE_BAND_MASK_INPUT=1 needs POSE_BAND_MASK=1"
_BM = {}
import framing as FR  # noqa: E402   (FRAMING_COND: the shared spec, byte-identical copy)
assert _env("FRAMING_COND", "0") in ("0", "1"), ("FRAMING_COND must be 0 | 1", _env("FRAMING_COND", "0"))
FRAMING = _env("FRAMING_COND", "0") == "1"                       # FRAMING_COND (module doc); 0 = no framing key anywhere
FRAMING_DIR = _env("FRAMING_DIR", "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/poseduplex/data/framing")
FRAMING_STATS_SHA = _env("FRAMING_STATS_SHA", "")
assert not FRAMING or POSE, "FRAMING_COND=1 needs POSEDUP_POSE=1"
_FR = {}
# RAW CODES (10-30, user-approved via main -a8 / duplexplan: "job 30074's recipe VERBATIM with RAW Mimi codes instead of SAM-separated codes").
#   POSEDUP_RAW_CODES=1 (needs ALOE_SAMSEP_STORES): every row read from the sam-sep stores (the item's own row AND the later-chunk sibling rows
#     B's voice clip is cut from) carries the codes_user / codes_assistant arrays of the MAPPED raw s2h-m1 row (ALOE_SAMSEP_ROW_MAP, new -> old:
#     the raw-lane mimi-stream-v5 codes job 26232 trained on), read IN PLACE from POSEDUP_RAW_STORES/m1n<k> (store digests pinned by
#     POSEDUP_RAW_DIGESTS = data_aloe_v2's own s2h-m1 DIGESTS). Rows, words, plan, pose, band masks, A audio, voice-clip choice: the sam-sep
#     row's, unchanged (A2.DIGESTS stay the sam-sep ones, so POSE_INDEX / BANDMASK / framing keep matching). Only code VALUES change.
#     Asserted at open: the map is a bijection kept-old-rows <-> ALL sam-sep rows, mapped rows have equal row_key / n_frames / chunk_index /
#     frame_start / split / speaker dirs; per row read: equal code shape / dtype and identical valid-frame mask. run_check (the trainer's
#     [same-as-v2] startup check) then also compares the micro FULLY (code values incl.) with data_aloe_v2's (26232's) micro of the mapped row.
#   POSEDUP_CLEAN_TARGET=<CLEAN_TARGET.json> (tools/clean_target_check.py; default '' = off): make_plan asserts every used item's B side
#     (train items of every epoch + eval) is a clean lane (handover xbleed.json: acoustic_crossbleed false, transcript_crossbleed false).
#   Defaults (0 / ''): every micro, the plan and cfg_view() byte-identical to before (no key, nothing read).
assert _env("POSEDUP_RAW_CODES", "0") in ("0", "1")
RAW_CODES = _env("POSEDUP_RAW_CODES", "0") == "1"
RAW_STORES = _env("POSEDUP_RAW_STORES", "/data/home/amartyavarma/qwen-1ch-data/s2h-m1/stores")
RAW_ROW_MAP = _env("ALOE_SAMSEP_ROW_MAP", "/data/home/amartyavarma/ppm/aloe-ft-32-samsep/data/row_map.json")
RAW_DIGESTS = _env("POSEDUP_RAW_DIGESTS", "").split(",") if RAW_CODES else []
assert not RAW_CODES or SAMSEP, "POSEDUP_RAW_CODES=1 needs ALOE_SAMSEP_STORES (rows stay the sam-sep stores'; only code values are swapped)"
assert not RAW_CODES or (len(RAW_DIGESTS) == 4 and all(len(x) == 32 for x in RAW_DIGESTS)), ("POSEDUP_RAW_DIGESTS: 4 x 32 hex", RAW_DIGESTS)
CLEAN_TARGET = _env("POSEDUP_CLEAN_TARGET", "")
_RAW = {}


def framing_stats():
    if "stats" not in _FR:
        st = FR.load_stats(f"{FRAMING_DIR}/FRAMING_STATS.json", FRAMING_STATS_SHA or None)
        assert st["dataset"] == "posedup", ("FRAMING_STATS.json is not the pose-duplex stats", st["dataset"])
        _FR["stats"] = st
    return _FR["stats"]


def framing_item(item, row_key):
    """FRAMING_COND: channel B's track of the item -> standardised features [F] float32 (status 0 asserted: FAIL LOUD)."""
    k, r, _ = (int(x) for x in item)
    if k not in _FR:
        _FR[k] = FR.SideStore(FRAMING_DIR, f"m1n{k}", "posedup", list(A2.DIGESTS)[k])
    rec, st = _FR[k].get(r, b_side(item))
    assert st == 0, ("FRAMING_COND: B's pose track has no usable frame-0 anchor (fail loud)", item, row_key, FR.STATUS.get(st))
    return FR.track_features(rec, framing_stats())


def cfg_view():
    """the resolved knobs (CFG['posedup'] in the trainer)."""
    v = {"pose": int(POSE), "items": ITEMS, "voice_rule": VOICE_RULE, "pose_light_frac": POSE_LIGHT_FRAC, "speech_light_frac": SPEECH_LIGHT_FRAC,
         "epoch_seed": EPOCH_SEED, "pose_index": POSE_INDEX if POSE or ITEMS != "all" else "", "pose_index_sha": POSE_INDEX_SHA if POSE else "",
         "pose_sha_every_read": int(POSE_SHA_EVERY_READ), "pose_seed_tag": POSE_SEED_TAG,
         "pose_src": "aloe-handover-full/DATA/files/none/<session>/<seg>/w00.SPEAKER_0N.pose_tokens.npy (B side, in place)"}
    if POSE_BAND_MASK:                                            # bandmask: keys exist only when on (off: 30074's dict exactly)
        v.update(pose_band_mask=1, pose_band_mask_input=int(POSE_BAND_MASK_INPUT), pose_band_mask_dir=POSE_BAND_MASK_DIR,
                 pose_band_drop_items=len(band_drops()))
    if RAW_CODES:                                                 # raw codes: keys exist only when on
        v.update(raw_codes=1, raw_stores=RAW_STORES, raw_digests=list(RAW_DIGESTS), raw_row_map=RAW_ROW_MAP,
                 raw_src="s2h-m1 raw-lane mimi-stream-v5 codes of the mapped row (= job 26232's); rows / pose / plan = the sam-sep stores'")
    if CLEAN_TARGET:
        v.update(clean_target=CLEAN_TARGET, clean_target_sha=hashlib.sha256(open(CLEAN_TARGET, "rb").read()).hexdigest())
    return v


def aloe_cfg():
    """= data_aloe_v2vox.aloe_cfg() (29345) with ONLY the knobs that are off their default applied (defaults: the dict is 29345's exactly)."""
    c = V.aloe_cfg()
    if SPEECH_LIGHT_FRAC != 0.1:
        c["light_frac"] = SPEECH_LIGHT_FRAC
    if POSE_LIGHT_FRAC != float(c["pose_light_frac"]):
        c["pose_light_frac"] = POSE_LIGHT_FRAC
    if VOICE_RULE == "ppm_third":
        c["sysprompt"] = "ppm_third_voiceB_only"
    return c


def eval_cfg():
    """= data_aloe_v2vox.eval_cfg(): aloe_cfg() with light_frac 1.0 (eval decoder frames)."""
    return dict(aloe_cfg(), light_frac=1.0)


def pose_index():
    if _PIX[0] is None:
        raw = open(POSE_INDEX, "rb").read()
        if POSE_INDEX_SHA:
            assert hashlib.sha256(raw).hexdigest() == POSE_INDEX_SHA, ("POSE_INDEX.json != POSEDUP_POSE_INDEX_SHA", POSE_INDEX)
        j = json.loads(raw)
        assert j["format"] == "posedup-pose-index-v1" and j["store_digests"] == list(A2.DIGESTS), ("pose index built for other stores", j["store_digests"])
        _PIX[0] = j
    return _PIX[0]


def b_side(item):
    return "user" if int(item[2]) else "assistant"


def pose_entry(item):
    k, r, _ = (int(x) for x in item)
    return pose_index()["rows"][str(k)][str(r)]["sides"][b_side(item)]


def pose_ok_item(item):
    return bool(pose_entry(item)["ok"]) and not band_dropped(item)


# ------------------------------------------------------------------------------------------------ bandmask (module knobs above)
def band_store(k):
    """store k's band masks {side: bool [R, 7500] (mmap)}; BANDMASK.json format / dataset / store digest (== this run's store k) / sha256."""
    if k not in _BM:
        d = f"{POSE_BAND_MASK_DIR}/m1n{k}"
        meta = json.load(open(f"{d}/BANDMASK.json"))
        assert meta["format"] == "pose-band-mask-v1" and meta["dataset"] == "posedup" and int(meta["store"]) == k, ("band mask meta", d)
        assert meta["store_digest"] == list(A2.DIGESTS)[k], ("band mask built for another store", d, meta["store_digest"], A2.DIGESTS[k])
        arr = {}
        for sd in ("user", "assistant"):
            f = f"{d}/pose_band_mask_{sd}.npy"
            assert hashlib.sha256(open(f, "rb").read()).hexdigest() == meta["sha256"][sd], ("band mask sha256 != BANDMASK.json", f)
            a = np.load(f, mmap_mode="r")
            assert a.dtype == np.bool_ and a.shape == (int(meta["rows"]), 7500), ("band mask shape", f, a.shape)
            arr[sd] = a
        _BM[k] = arr
    return _BM[k]


def band_drops():
    """{(store, row, B side)} of lanes whose clip starts inside a band (DROP_LANES.json)."""
    if "drop" not in _BM:
        j = json.load(open(f"{POSE_BAND_MASK_DIR}/DROP_LANES.json"))
        assert j["format"] == "pose-band-drop-v1" and j["dataset"] == "posedup", ("DROP_LANES.json", POSE_BAND_MASK_DIR)
        _BM["drop"] = {(int(x[0]), int(x[1]), str(x[2])) for x in j["items"]}
    return _BM["drop"]


def band_dropped(item):
    """flag on: B's lane starts inside a band -> no B pose for this item. Flag off: always False (nothing read)."""
    if not POSE_BAND_MASK:
        return False
    k, r, _ = (int(x) for x in item)
    return (k, r, b_side(item)) in band_drops()


def band_positions(item, n, P, cfg):
    """micro positions of B's masked pose frames: P + pose_delay + f for every masked row frame f < n (the pose label positions)."""
    k, r, _ = (int(x) for x in item)
    msk = np.asarray(band_store(k)[b_side(item)][r])
    assert not msk[n:].any(), ("band mask beyond the row's frames", item, n)
    return int(P) + int(cfg["pose_delay"]) + np.flatnonzero(msk[:n]).astype(np.int64)


def band_apply(pdepth, pcb0, plight, pos):
    """masked positions: pose cb0 label -> -100, pose decoder frame removed; POSE_BAND_MASK_INPUT: pose input -> -1 (absent)."""
    assert (pcb0[pos] >= 0).all(), "band position without a pose label"
    pcb0 = pcb0.copy()
    pcb0[pos] = -100
    plight = plight[~np.isin(plight, pos)]
    if POSE_BAND_MASK_INPUT:
        pdepth = pdepth.copy()
        pdepth[pos] = -1
    return pdepth, pcb0, plight


# ------------------------------------------------------------------------------------------------ raw codes (module knobs above)
RAW_ROW_COLS = ("row_key", "n_frames", "chunk_index", "frame_start", "split", "user_speaker_dir", "assistant_speaker_dir", "src_conversation")


def raw_map():
    """ALOE_SAMSEP_ROW_MAP -> {k: (old_of_new int64 [R_new], old rows)}. Asserted: built FROM the pinned raw stores (old digests ==
    POSEDUP_RAW_DIGESTS) FOR this run's sam-sep stores (new digests == A2.DIGESTS); new_of_old injective and onto range(R_new) (a bijection
    between the kept raw rows and every sam-sep row), old rows in range."""
    if "map" not in _RAW:
        j = json.load(open(RAW_ROW_MAP))
        assert j["format"] == "aloe-ft-32-samsep-row-map-v1", ("row map format", RAW_ROW_MAP, j.get("format"))
        assert [j["old_digest"][f"m1n{k}"] for k in range(4)] == list(RAW_DIGESTS), ("row map built from other raw stores", j["old_digest"])
        assert [j["new_digest"][f"m1n{k}"] for k in range(4)] == list(A2.DIGESTS), ("row map built for other sam-sep stores", j["new_digest"])
        out = {}
        for k in range(4):
            mp = j["map"][f"m1n{k}"]
            no = {int(o): int(n) for o, n in mp["new_of_old"].items()}
            R, n_old = len(no), int(mp["old_rows"])
            assert sorted(no.values()) == list(range(R)), ("row map is not a bijection onto the sam-sep rows", k)
            assert all(0 <= o < n_old for o in no), ("row map old row out of range", k)
            oon = np.full(R, -1, np.int64)
            for o, n in no.items():
                oon[n] = o
            assert (oon >= 0).all() and len(np.unique(oon)) == R, ("row map inverse", k)
            out[k] = (oon, n_old)
        _RAW["map"] = out
    return _RAW["map"]


class RawRows:
    """POSEDUP_RAW_CODES: sam-sep store k's s3rows.Rows whose rows carry the MAPPED raw s2h-m1 row's codes_user / codes_assistant (mmap,
    read in place); every other field = the sam-sep row's. Open: raw store digest pinned, every mapped row's provenance == the sam-sep row's."""

    def __init__(self, k, node, rows):
        import pyarrow.parquet as pq
        self.k, self.rows = int(k), rows
        self.oon, n_old = raw_map()[self.k]
        assert len(self.oon) == len(rows) == len(node["length"]), ("row map rows != sam-sep store rows", k, len(self.oon), len(rows))
        self.root = f"{RAW_STORES}/m1n{self.k}"
        self.raw = D.R3.load_node(self.root)
        assert self.raw["digest"] == RAW_DIGESTS[self.k], ("raw store digest != POSEDUP_RAW_DIGESTS", self.root, self.raw["digest"])
        assert len(self.raw["length"]) == n_old, ("raw store rows != row map old_rows", self.root, len(self.raw["length"]), n_old)
        cols = ["row", *RAW_ROW_COLS]
        tn = pq.read_table(f"{node['flat_root']}/rows.parquet", columns=cols).to_pydict()
        to = pq.read_table(f"{self.root}/rows.parquet", columns=cols).to_pydict()
        assert tn["row"] == list(range(len(self.oon))) and to["row"] == list(range(n_old)), "rows.parquet row ids"
        ol = self.oon.tolist()
        for c in RAW_ROW_COLS:
            assert [to[c][o] for o in ol] == tn[c], ("mapped raw row != sam-sep row", self.k, c)
        assert np.array_equal(self.raw["length"][self.oon], node["length"]), ("mapped raw n_frames != sam-sep", self.k)
        self.cu, self.ca = {}, {}

    def __len__(self):
        return len(self.rows)

    def raw_codes(self, i):
        """sam-sep row i -> (raw row, raw codes_user [32, 7500], raw codes_assistant) (mmap views)."""
        o = int(self.oon[int(i)])
        s, r = int(self.raw["shard_of"][o]), int(self.raw["row_of"][o])
        if s not in self.cu:
            sh = f"{self.root}/{self.raw['shards'][s]}"
            self.cu[s] = np.load(f"{sh}.codes_user.npy", mmap_mode="r")
            self.ca[s] = np.load(f"{sh}.codes_assistant.npy", mmap_mode="r")
        return o, self.cu[s][r], self.ca[s][r]

    def __getitem__(self, i):
        row = self.rows[i]
        o, cu, ca = self.raw_codes(i)
        n = int(row["n_frames"])
        assert n == int(self.raw["length"][o]), ("raw n_frames", row["row_key"])
        for sd, c in (("user", cu), ("assistant", ca)):
            s = row[f"codes_{sd}"]
            assert c.shape == s.shape == (D.DEPTH, D.R3.ROW_FRAMES) and c.dtype == s.dtype, ("raw codes shape / dtype", row["row_key"], sd, c.shape)
            assert np.array_equal(np.asarray(c) >= 0, np.asarray(s) >= 0), ("raw / sam-sep valid-frame mask differs", row["row_key"], sd)
        out = dict(row)
        out["codes_user"], out["codes_assistant"] = cu, ca
        return out


def raw_ref_check(m):
    """POSEDUP_RAW_CODES: micro m (this module's 29345 code path, raw codes) vs data_aloe_v2's micro (a FRESH module instance of the same
    file = 26232's data path, its own s2h-m1 DIGESTS) of the MAPPED raw row, same cfg / rank: B channel (prefix + voice clip incl.), targets
    and light frames FULLY (code VALUES included), channel-A prefix; A in the conversation is Voxtral-only here (checked by the caller)."""
    if "ref" not in _RAW:
        import importlib.util
        spec = importlib.util.spec_from_file_location("data_aloe_v2_rawref", A2.__file__)
        R = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(R)
        assert list(R.DIGESTS) == list(RAW_DIGESTS), ("POSEDUP_RAW_DIGESTS != data_aloe_v2's own s2h-m1 digests", R.DIGESTS)
        _RAW["ref"] = R.Stores(roots=[f"{RAW_STORES}/m1n{k}" for k in range(4)])
    k, r, sw = (int(x) for x in m["item"])
    cfg, rank = m["raw_ctx"]
    mr = _RAW["ref"].conv((k, int(raw_map()[k][0][r]), sw), cfg, rank)
    P = int(m["P"])
    eq = lambda a, b: bool(np.array_equal(np.asarray(a), np.asarray(b)))  # noqa: E731
    return {"P": int(mr["P"]) == P, "row_key": mr["row_key"] == m["row_key"], "ids_B": eq(mr["ids"][1], m["ids"][1]),
            "depth_B_values": eq(mr["depth"][1], m["depth"][1]), "present_B": eq(mr["present"][1], m["present"][1]),
            "prefix_A": eq(mr["ids"][0][:P], m["ids"][0][:P]) and eq(mr["present"][0][:P], m["present"][0][:P]) and eq(mr["depth"][0][:P], m["depth"][0][:P]),
            "targets_values": all(eq(mr[x], m[x]) for x in ("sel", "y", "word", "pad", "eob", "nw", "light", "cb0")),
            "voice": (mr["voice_clip"], mr["voice_drawn"]) == (m["voice_clip"], m["voice_drawn"]) and mr["swap"] == m["swap"]}


# ------------------------------------------------------------------------------------------------ clean target (POSEDUP_CLEAN_TARGET)
def clean_lanes():
    """CLEAN_TARGET.json (tools/clean_target_check.py) -> {(store, row, side)} of clean lanes; built for this run's stores (asserted)."""
    if "clean" not in _RAW:
        j = json.load(open(CLEAN_TARGET))
        assert j["format"] == "posedup-clean-target-v1" and j["store_digests"] == list(A2.DIGESTS), ("CLEAN_TARGET.json built for other stores", CLEAN_TARGET)
        _RAW["clean"] = {(int(x[0]), int(x[1]), str(x[2])) for x in j["clean_lanes"]}
    return _RAW["clean"]


def assert_clean_target(items):
    """every item's B side is a clean lane (acoustic_crossbleed false and transcript_crossbleed false in the handover xbleed.json)."""
    cl = clean_lanes()
    bad = [tuple(x) for x in items if x is not None and (int(x[0]), int(x[1]), b_side(x)) not in cl]
    assert not bad, ("CLEAN-TARGET: B side not a clean lane (crossbleed flagged / missing)", len(bad), bad[:5])
    return len([x for x in items if x is not None])


# ------------------------------------------------------------------------------------------------ plan (multi-epoch; epoch 0 = 29345's)
def make_plan(world, roots=STORES, steps=None):
    """-> 29345's plan dict (V.make_plan: items, steps, eval, counts) with items = the concatenated per-epoch orders, each padded with None
    (dummy) to whole steps; steps = `steps` (None = one epoch). 'items0' = epoch 0's order (29345's, or its pose-ok filter)."""
    base = V.make_plan(world, roots)
    it0 = [tuple(x) for x in base["items"]]
    if ITEMS == "pose_ok":
        it0 = [x for x in it0 if pose_ok_item(x)]
    E = -(-len(it0) // world)
    S = int(steps) if steps else E
    items, e = [], 0
    while len(items) < S * world:
        if e == 0:
            order = list(it0)
        else:
            perm = np.random.default_rng([EPOCH_SEED, e]).permutation(len(it0))
            order = [it0[i] for i in perm]
        items += order + [None] * (E * world - len(order))
        e += 1
    items = items[:S * world]
    out = dict(base)
    out.update(items=items, steps=S, epoch_steps=E, n_epochs=S / E, items0=it0, n_items0=len(it0),
               n_train_micros=sum(x is not None for x in items), n_dummy=sum(x is None for x in items))
    if CLEAN_TARGET:                                              # clean-target startup assert (module knobs): train items + eval list
        out["clean_target_checked"] = assert_clean_target(list(it0) + [tuple(x) for x in base["eval"]])
    return out


def plan_digest(pl):
    """full-plan digest (every step's items, dummies as null) and epoch-0 digest (= 29345's d1a3e3721328da0a for ITEMS=all)."""
    h = hashlib.sha256(json.dumps([list(x) if x is not None else None for x in pl["items"]]).encode()).hexdigest()[:16]
    h0 = hashlib.sha256(json.dumps([list(x) for x in pl["items0"]]).encode()).hexdigest()[:16]
    return h, h0


# ------------------------------------------------------------------------------------------------ micro
def pose_arrays(m, row, pose_b, cfg):
    """= aloe-v2pure-16610 (job 26089) data_aloe_pose.pose_arrays (verbatim maths). pose_b None (no B pose): pdepth -1, pcb0 -100, no frames."""
    P, n, N = int(m["P"]), int(row["n_frames"]), len(m["pid"])
    pd = int(cfg["pose_delay"])
    assert pd == D.CD, ("pose delay != the speech code delay (PPM: both CD)", pd, D.CD)
    off = np.arange(D.DEPTH, dtype=np.int64) * D.CBS
    pdepth = np.full((N, D.DEPTH), -1, np.int64)
    if pose_b is None:
        return pdepth, np.full(N, -100, np.int64), np.zeros(0, np.int64)
    pos = P + pd + np.arange(n)
    p = np.asarray(pose_b[:, :n]).T.astype(np.int64)
    assert (p >= 0).all() and (p < D.CBS).all(), ("B pose codes missing / out of range on a qualifying track", row["row_key"])
    pdepth[pos] = p + off
    db = np.asarray(m["depth"][1])
    conv_m = db[P:, 0] >= 0
    assert np.array_equal(np.flatnonzero(conv_m) + P, pos), ("B Mimi / pose positions differ", row["row_key"])
    assert np.array_equal(db[pos] - off, np.asarray(row[f"codes_{'user' if m['swap'] else 'assistant'}"])[:, :n].T.astype(np.int64)), "B side"
    pcb0 = np.where(pdepth[:, 0] >= 0, pdepth[:, 0], -100).astype(np.int64)
    plight = PS.light_pick(np.flatnonzero(pdepth[:, 0] >= 0), float(cfg["pose_light_frac"]),
                           (cfg["light_seed"], PS.h64(row["row_key"]), 1, POSE_SEED_TAG)).astype(np.int64)
    return pdepth, pcb0, plight


def read_pose(item, row):
    """B's pose tokens of the item's conversation (mmap) or None; file size + sha256 (every read) + shape asserted against the index."""
    e = pose_entry(item)
    if not e["ok"]:
        return None, e
    if band_dropped(item):                                        # bandmask DROP: the lane's frame 0 is the junk skeleton (module knobs)
        return None, {**{k: v for k, v in e.items() if k != "path"}, "why": "bandmask drop: clip starts inside a black band"}
    f = f"{pose_index()['data_root']}/{e['path']}"
    assert os.path.getsize(f) == int(e["bytes"]), ("pose file size != index", f)
    if POSE_SHA_EVERY_READ:
        assert hashlib.sha256(open(f, "rb").read()).hexdigest() == e["sha256"], ("pose file sha256 != index", f)
    a = np.load(f, mmap_mode="r")
    assert a.shape == (D.DEPTH, int(e["T"])) and a.dtype == np.int16 and int(row["frame_start"]) == 0 and int(row["n_frames"]) <= int(e["T"]), \
        ("pose tokens / row mismatch", f, a.shape, row["row_key"])
    return a, e


def want_voice_third(row_key, cfg):
    """26089's draw (data_ppm.podcast_prefix): h64('ppm-voice|seed|row_key') % voice_mod == 0 (per row: both orientations draw together)."""
    return D.h64(f"ppm-voice|{int(cfg['seed'])}|{row_key}") % int(cfg["voice_mod"]) == 0


def prefix_third(row, swap, sib, cfg):
    """= data_aloe_v2.prefix_v2 line for line with the draw replaced by want_voice_third (POSEDUP_VOICE_RULE=ppm_third)."""
    sw = int(swap)
    wv = want_voice_third(row["row_key"], cfg)
    pv = {"voice_drawn": int(wv)}
    segs = []
    if wv:
        cl = D.voice_clips(row, sib, cfg)[1 ^ sw]
        if cl is not None:
            segs.append({"kind": "speech", "track": 1, **cl})
    pv["clips"] = [{"track": "AB"[s["track"]], "data_side": s["side"], "src_key": s["src_key"], "src_chunk": s["src_chunk"],
                    "frames": [s["w0"], s["w0"] + s["L"]], "cov": round(s["cov"], 3)} for s in segs]
    pv["voice_clip"] = int(bool(segs))
    fixed = 5 + sum(int(s["L"]) + 5 for s in segs)
    assert fixed <= int(cfg["sp_cap"]), ("voice clip exceeds the system-prompt cap", fixed)
    if not segs:
        segs = [{"kind": "text", "track": 1, "ids": []}]
    spx = D.SPR.build_prefix(row, segs, {"new_word": cfg["new_word"]})
    assert 5 <= spx["P"] <= int(cfg["sp_cap"]), ("system prompt length", spx["P"])
    assert pv["voice_clip"] or spx["P"] == 5, ("empty system block must be 5 positions", spx["P"])
    assert (spx["ids"][0] == D.PAD_ID).all() and not (np.asarray(spx["depth"])[0] >= 0).any(), "channel A must carry nothing in the prefix"
    pv["P"] = int(spx["P"])
    return spx, pv


def speech_micro_third(row, node_row, sib, cfg, rank, swap):
    """= data_aloe_v2.speech_micro_v2 with prefix_third, then channel A emptied exactly as data_aloe_v2vox.speech_micro_v2vox."""
    sw = int(swap)
    spx, pv = prefix_third(row, sw, sib, cfg)
    P, n = int(spx["P"]), int(row["n_frames"])
    rf = P + n if not cfg["crop_conv"] else 7500
    dc = {"new_word": cfg["new_word"], "eob_gap_s": cfg["eob_gap_s"], "wpid": cfg["wpid"], "light_seed": cfg["light_seed"], "row_frames": rf,
          "frame_light": True, "no_crop": not cfg["crop_conv"]}
    a = D.DS.row_arrays(row, dc, rank, int(node_row), "ppm", float(cfg["light_frac"]), sp=spx, swap=sw)
    sel, y, word, pad, eob, nw, cb0, light = D._b_only(a)
    T = len(a["pid"])
    m = {"kind": "speech", "ids": a["ids"], "depth": a["depth"], "present": a["present"], "pid": a["pid"], "segs": [T],
         "sel": sel, "y": y, "word": word, "pad": pad, "eob": eob, "nw": nw, "n_text": float(len(sel)), "cb0": cb0, "light": light,
         "P": P, "swap": sw}
    assert (cb0[:P + D.CD] < 0).all() and (light >= P + D.CD).all() and (sel >= P).all()
    assert T == (7500 if cfg["crop_conv"] else P + n) + D.CD, ("row length", T, P, n)
    pv.update(row_key=row["row_key"], swap=sw, chan_B_data_side=("user" if sw else "assistant"), n_frames=n, T=T, n_light=int(len(light)),
              voice_rule="ppm_third")
    m["prov"] = json.dumps(pv)
    m["voice_clip"] = pv["voice_clip"]
    m["voice_drawn"] = pv["voice_drawn"]
    assert (m["depth"][0, :P] < 0).all() and (m["ids"][0, :P] == D.PAD_ID).all(), "v2 prefix must carry nothing on channel A"
    m["ids"] = m["ids"].copy(); m["depth"] = m["depth"].copy(); m["present"] = m["present"].copy()
    m["ids"][0, P:] = D.PAD_ID
    m["depth"][0] = -1
    m["present"][0, P:] = 0.0
    m["vin"] = (P, n)
    pv = json.loads(m["prov"]); pv["chan_A"] = "voxtral_latents_only"; m["prov"] = json.dumps(pv)
    return m


class Stores(V.Stores):
    """= data_aloe_v2vox.Stores (29345) + B's pose. conv(pose=None): the module default; False: no pose arrays (the 29345 micro)."""

    def open(self, k):
        """= data_aloe_v2.Stores.open; POSEDUP_RAW_CODES: the store's rows (and the sibling index's, for voice clips) become RawRows."""
        if not RAW_CODES:
            return super().open(k)
        if k not in self.s:
            node, st, sib = super().open(k)
            rr = RawRows(k, node, st)
            sib.rows = rr
            self.s[k] = (node, rr, sib)
        return self.s[k]

    def conv(self, item, cfg, rank, audio=True, pose=None, rule=None):
        use = POSE if pose is None else (bool(pose) and POSE)
        if (rule or VOICE_RULE) == "v2_30":
            m = super().conv(item, cfg, rank, audio=audio)          # = job 29345's micro, its own code path
        else:
            m = self.conv_third(item, cfg, rank, audio)
        if RAW_CODES:                                             # raw codes: run_check's reference build needs the cfg / rank (not in the micro)
            m["raw_ctx"] = (dict(cfg), int(rank))
        if not use:
            return m
        k, r, sw = (int(x) for x in item)
        node, st, sib = self.open(k)
        row = st[r]
        assert row["row_key"] == m["row_key"]
        pb, e = read_pose(item, row)
        m["pdepth"], m["pcb0"], m["plight"] = pose_arrays(m, row, pb, cfg)
        if POSE_BAND_MASK and pb is not None:                     # bandmask MASK (module knobs); m['pband'] stays out of the micro
            m["pband"] = band_positions(item, int(row["n_frames"]), int(m["P"]), cfg)
            m["pdepth"], m["pcb0"], m["plight"] = band_apply(m["pdepth"], m["pcb0"], m["plight"], m["pband"])
        m["pose_ok"] = int(pb is not None)
        m["pose_src"] = e.get("path", "none: " + e.get("why", "?"))
        if FRAMING:                                               # FRAMING_COND: B's track framing (None = pose-less B: zero row, no positions)
            m["frf1"] = framing_item(item, row["row_key"]) if pb is not None else None
        return m

    def conv_third(self, item, cfg, rank, audio):
        """= data_aloe_v2vox.Stores.conv with speech_micro_third in place of speech_micro_v2vox (everything else line for line)."""
        import audio_s4q as AU
        import rowcache_vox as RC
        k, r, sw = item
        node, st, sib = self.open(int(k))
        row = st[int(r)]
        D.R3.check_row(row)
        assert int(row["chunk_index"]) == 0, ("v2: first chunks only", row["row_key"], row["chunk_index"])
        m = speech_micro_third(row, int(r), sib, cfg, rank, int(sw))
        T = len(m["pid"])
        assert m["segs"] == [T] and np.array_equal(m["pid"], np.arange(T)) and m["swap"] == int(sw)
        assert T == int(m["P"]) + int(row["n_frames"]) + D.CD <= BUDGET, ("segment length", T, m["P"], row["n_frames"])
        aside, a = a_entry(item, row)
        g_db = a_gain(a)
        m["aref"] = {"row_key": row["row_key"], "side": aside, "flac": a["flac"], "gain_db": g_db, "frame0": int(a["frame0"]),
                     "n": int(a["n_frames"]), "kind": a["kind"], "gain_mode": GAIN_MODE, "gain_mimi_db": float(a["gain_db"])}
        if audio:
            path = f"{FLAC_ROOT}/{a['flac']}"
            if a["kind"] == "gcs":
                assert os.path.getsize(path) == int(a["flac_bytes"]), ("A FLAC missing / wrong size on this node", path)
            else:
                assert a["kind"] == "mimidec" and int(a["audio_frame0"]) == 0 and os.path.getsize(path) > 0, ("A mimidec file", path)
            m["wav"] = RC.row_audio({"flac": path, "frame0": int(a["audio_frame0"]), "n_frames": int(a["n_frames"]), "row_key": row["row_key"]},
                                    g_db)
            assert m["wav"].dtype == np.float32 and m["wav"].shape == (int(a["n_frames"]) * AU.SAMPLES_PER_FRAME_24K,)
        m["row_key"] = row["row_key"]
        m["item"] = [int(k), int(r), int(sw)]
        return m


def own_pose(m):
    """(pose cb0 labels, pose decoder frames) of one conversation."""
    return [float((np.asarray(m["pcb0"]) >= 0).sum()), float(len(m["plight"]))]


def as_micro(m, dummy=False):
    """= data_aloe_v2vox.as_micro (29345's micro) + (pose) kind 'pose', pdepth / pcb0 / plight, own counts (text, cb0, light, pose cb0, pose
    frames), pose_ok. Without pose arrays: the 29345 micro unchanged."""
    out = V.as_micro(m, dummy=dummy)
    if "pdepth" not in m:
        return out
    assert out["kind"] == "speech" and out["segs"] == [len(m["pid"])]
    out["kind"] = "pose"
    out["pdepth"], out["pcb0"], out["plight"] = m["pdepth"], m["pcb0"], m["plight"]
    own = np.asarray(out["own"], np.float64)
    assert own.shape == (1, 3), own.shape
    out["own"] = np.concatenate([own, np.asarray([own_pose(m)], np.float64)], 1)
    out["pose_ok"] = int(m["pose_ok"])
    out["pose_src"] = m["pose_src"]
    if FRAMING:                                                   # FRAMING_COND: one track (B), positions = where pose_emb adds
        out["frpos"], out["fridx"], out["frf"] = FR.micro_arrays([m["pdepth"]], [m["frf1"]], len(framing_stats()["feat"]))
    if dummy:
        out["own"] = np.zeros_like(out["own"])
        out["pose_ok"] = 0
    return out


class AccDS(V.AccDS):
    """= data_aloe_v2vox.AccDS (one physical rank's stream of logical ranks' micros) with this module's Stores / as_micro."""

    def __getitem__(self, i):
        if self.st is None:
            self.st = Stores()
        t, g = divmod(int(i), self.G)
        L = self.lr[g]
        pad = L is None
        it = None if pad else self.rows[g][t]
        rk = self.lr[0] if pad else L
        m = self.st.conv(it if it is not None else self.fb, self.cfg, rk)
        out = as_micro(m, dummy=it is None)
        out["step"], out["slot"], out["lrank"], out["padslot"] = t, g, (-1 if pad else int(L)), int(pad)
        if t == self.singles_at:
            s1 = as_micro(m)
            s1.pop("wav", None)
            out["singles"] = [s1]
        return out


def eval_micros(items, cfg, rank):
    st = Stores()
    return [as_micro(st.conv(it, cfg, rank)) for it in items]


# ------------------------------------------------------------------------------------------------ checks
SPEECH_KEYS = ("ids", "depth", "present", "pid", "segs", "sel", "y", "word", "pad", "eob", "nw", "cb0", "light", "P", "swap", "vin", "row_key",
               "item", "voice_clip", "voice_drawn")


def run_check(m0, m1):
    """m0 = job 29345's micro of an item, m1 = this run's micro of the same item (same cfg). Default voice rule: every speech field identical
    (the pose fields are additions); ppm_third: the conversation part identical after the prefix shift d = P1 - P0 (ids / depth / present
    from P on, targets / light shifted by d), A empty. Pose (if any): pdepth >= 0 exactly at B's conversation Mimi positions."""
    ok = True
    if VOICE_RULE == "v2_30":
        for k in SPEECH_KEYS:
            a, b = m0.get(k), m1.get(k)
            ok &= bool(np.array_equal(np.asarray(a), np.asarray(b))) if isinstance(a, np.ndarray) or isinstance(b, np.ndarray) else (a == b)
    else:
        P0, P1 = int(m0["P"]), int(m1["P"])
        d = P1 - P0
        for k in ("ids", "depth", "present"):
            ok &= bool(np.array_equal(np.asarray(m0[k])[:, P0:], np.asarray(m1[k])[:, P1:]))
        ok &= bool(np.array_equal(np.asarray(m0["sel"]) + d, np.asarray(m1["sel"])) and np.array_equal(np.asarray(m0["light"]) + d, np.asarray(m1["light"])))
        for k in ("y", "word", "pad", "eob", "nw"):
            ok &= bool(np.array_equal(np.asarray(m0[k]), np.asarray(m1[k])))
        ok &= bool(np.array_equal(np.asarray(m0["cb0"])[P0:], np.asarray(m1["cb0"])[P1:])) and m0["row_key"] == m1["row_key"]
    if RAW_CODES:                                                 # raw codes: m0 == 26232's v2 micro of the mapped raw row, code VALUES incl.
        ok &= "raw_ctx" in m0 and all(raw_ref_check(m0).values())
    if "pdepth" in m1:
        P1 = int(m1["P"])
        pp = np.flatnonzero(np.asarray(m1["pdepth"])[:, 0] >= 0)
        if int(m1["pose_ok"]) and "pband" in m1:                  # bandmask MASK: masked frames = no label / no decoder frame (+ no input)
            db = np.flatnonzero(np.asarray(m1["depth"][1])[:, 0] >= 0)
            conv, bp = db[db >= P1], np.asarray(m1["pband"])
            ppf = np.union1d(pp, bp) if POSE_BAND_MASK_INPUT else pp
            pl = np.flatnonzero(np.asarray(m1["pcb0"]) >= 0)
            ok &= bool(np.isin(bp, conv).all() and np.array_equal(ppf, conv) and np.array_equal(pl, np.setdiff1d(conv, bp))
                       and not np.isin(np.asarray(m1["plight"]), bp).any())
            if POSE_LIGHT_FRAC == 1.0:
                ok &= len(m1["plight"]) == len(conv) - len(bp)
        elif int(m1["pose_ok"]):
            db = np.flatnonzero(np.asarray(m1["depth"][1])[:, 0] >= 0)
            ok &= bool(np.array_equal(pp, db[db >= P1])) and len(m1["plight"]) == round(len(pp) * POSE_LIGHT_FRAC) if POSE_LIGHT_FRAC == 1.0 \
                else bool(np.array_equal(pp, db[db >= P1]))
        else:
            ok &= len(pp) == 0 and len(m1["plight"]) == 0 and bool((np.asarray(m1["pcb0"]) < 0).all())
    return bool(ok)


# ------------------------------------------------------------------------------------------------ reports (trainer start, tests, plan tool)
def plan_report(pl):
    """pose coverage / hours of the plan's train items (per epoch) and of the eval list."""
    pix = pose_index()["rows"]

    def n_of(it):
        return int(pix[str(int(it[0]))][str(int(it[1]))]["n"])
    rep = {"steps": pl["steps"], "epoch_steps": pl["epoch_steps"], "n_epochs": round(pl["n_epochs"], 4), "items_per_epoch": pl["n_items0"],
           "train_micros": pl["n_train_micros"], "dummy_micros": pl["n_dummy"]}
    ok = [x for x in pl["items0"] if pose_ok_item(x)]
    rep["epoch0"] = {"micros": len(pl["items0"]), "pose_ok": len(ok), "frac_pose_ok": round(len(ok) / max(len(pl["items0"]), 1), 4),
                     "hours": round(sum(n_of(x) for x in pl["items0"]) / 12.5 / 3600, 1), "pose_hours": round(sum(n_of(x) for x in ok) / 12.5 / 3600, 1)}
    tr = [x for x in pl["items"] if x is not None]
    okt = [x for x in tr if pose_ok_item(x)]
    rep["run"] = {"micros": len(tr), "pose_ok": len(okt), "hours": round(sum(n_of(x) for x in tr) / 12.5 / 3600, 1),
                  "pose_hours": round(sum(n_of(x) for x in okt) / 12.5 / 3600, 1)}
    ev = [tuple(x) for x in pl["eval"]]
    oke = [x for x in ev if pose_ok_item(x)]
    rep["eval"] = {"micros": len(ev), "pose_ok": len(oke), "hours": round(sum(n_of(x) for x in ev) / 12.5 / 3600, 1),
                   "pose_hours": round(sum(n_of(x) for x in oke) / 12.5 / 3600, 1)}
    return rep


def render_dump(items, cfg, path, n=3, conv_pos=6):
    """n rendered train micros (pose-OK items) as text: per position near the system-prompt end and into the conversation, channel A / B
    lanes, B's Mimi code (cb0) and B's POSE code (cb0) summed into the same position, plus the pose / Mimi alignment and counts."""
    import transformers
    tok = transformers.AutoTokenizer.from_pretrained("/data/home/amartyavarma/models/Qwen3-32B")
    st = Stores()
    out, done = [], 0
    for it in items:
        if it is None or not pose_ok_item(it):
            continue
        m = st.conv(it, cfg, 0, audio=False)
        P, N = int(m["P"]), len(m["pid"])
        pos = np.flatnonzero(m["pdepth"][:, 0] >= 0)
        dpos = np.flatnonzero(np.asarray(m["depth"][1])[:, 0] >= 0)
        dpos = dpos[dpos >= P]
        out.append(f"=== item {list(it)} row_key {m['row_key']} swap {m['swap']} (B = data {'user' if m['swap'] else 'assistant'}) P {P} T {N} "
                   f"voice_clip {m['voice_clip']} pose_src {m['pose_src']}")
        out.append(f"    pose frames {len(pos)} at positions [{pos[0]}..{pos[-1]}]; B Mimi frames in the conversation {len(dpos)} at "
                   f"[{dpos[0]}..{dpos[-1]}]; identical positions: {bool(np.array_equal(pos, dpos))}; pose decoder frames (plight) {len(m['plight'])}; "
                   f"pose cb0 labels {int((m['pcb0'] >= 0).sum())}; A = Voxtral latents at positions {tuple(int(x) for x in m['vin'])} (start, n)")
        out.append("    pos | A id (present) | B lane token | B Mimi cb0 (depth[1][:,0]) | B POSE cb0 (pdepth[:,0] - 0) | pose cb1 (pdepth[:,1] - 2048) | pcb0 label")
        for p in list(range(max(0, P - 3), min(N, P + D.CD + conv_pos))) + list(range(P + D.CD + 200, P + D.CD + 203)):
            if p >= N:
                continue
            bt = tok.decode([int(m["ids"][1, p])]) if int(m["ids"][1, p]) < 151643 or int(m["ids"][1, p]) > 151645 else f"<{int(m['ids'][1, p])}>"
            out.append(f"    {p:5d} | {int(m['ids'][0, p]):6d} ({float(m['present'][0, p]):.0f}) | {bt!r:>14} ({int(m['ids'][1, p])}) | "
                       f"{int(m['depth'][1][p, 0]):5d} | {int(m['pdepth'][p, 0]):5d} | {int(m['pdepth'][p, 1]) - 2048 if m['pdepth'][p, 1] >= 0 else -1:5d} | "
                       f"{int(m['pcb0'][p]):5d}")
        out.append("    model input at each position p: e[p] = present_A(emb_A(ids_A) + aud_A(depth_A)) + present_B(emb_B(ids_B) + aud_B(depth_B)) "
                   "+ inp.pose(pdepth[p])   (-1 codes -> 0), + s x vadapter(Voxtral(A waveform)) at A's conversation positions")
        done += 1
        if done >= n:
            break
    txt = "\n".join(out) + "\n"
    open(path, "w").write(txt)
    return txt
