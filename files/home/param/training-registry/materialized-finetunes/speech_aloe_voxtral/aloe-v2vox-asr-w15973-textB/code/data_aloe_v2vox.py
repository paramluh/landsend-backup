"""aloe-v2vox-ft DATA (10-17, user): = aloe-v2-ft's data (data_aloe_v2.py, job 22383, byte-identical copy: FIRST chunk of every call only,
both speakers as B, system prompt with NO text, B's own voice clip on the seeded 30 % rule (h64 'aloe-v2-voice|20261011|row_key|swap'),
A no clip, light_frac 0.1 in training / 1.0 in eval, plan seed 20261011, item t x WORLD + R, dummies on the last step, held-out = 21980's
eval rows that are first chunks x 2 = 324 micros) with ONLY channel A changed exactly as aloe-m1vox-ft (job 22335, data_aloe_m1vox /
data_aloe_vox.speech_micro_vox): in the conversation A has NO text (PAD), NO Mimi (depth -1 everywhere on A; the v2 prefix already carries
nothing on A, asserted by data_aloe_v2.prefix_v2), present_A = 0 at p >= P; vin = (P, n): the trainer adds s x vadapter(Voxtral(A audio of
row frames 0..n-1)) at positions P..P+n-1.
A AUDIO = aloe-m1vox-ft's: the other speaker's WLT FLAC from plan/M1VOX_INDEX.json (m1vox's index, built for these 4 stores; v2's rows are a
subset of m1vox's rows), node-local under ALOE_FLAC_ROOT (each node holds the A tracks ITS ranks read under the v2 plan:
tools/stage_v2vox.py, ledger _ledger/DONE_v2vox_node<i>.json), decoded in the DataLoader workers (audio_s4q.ffmpeg_row_audio via rowcache_vox,
recorded Mimi LUFS gain); purged conversations: the Mimi decode of the row's own A codes (m1vox's rule).
aloe-v2vox-raw (10-17, user): ONE change, the A gain before Voxtral, switch VOX_A_GAIN (default 'mimi' = the path above, unchanged):
  VOX_A_GAIN=raw   'gcs' (WLT FLAC) entries: gain 0 dB -> the waveform = the raw FLAC samples (ffmpeg f32le mono 24 kHz slice, no scaling);
                   'mimidec' entries (Mimi decode of the NORMALISED track): gain = MINUS that track's recorded Mimi LUFS gain -> raw level.
                   Gains from plan/RAWGAIN.json (ALOE_RAWGAIN; tools/build_rawgain_mimidec.py); asserted: gcs 0.0, mimidec = -recorded.
                   Training AND eval micros (same Stores.conv). No other change (no augmentation, no clipping / limiting).
  AccDS            one PHYSICAL rank's stream when it emulates several LOGICAL ranks of 22420's 64 (gradient accumulation, same rows per
                   step): element i = (step t, slot g) -> the micro logical rank L_g reads at step t (= VoxDS of rank L_g, item t), built with
                   rank L_g (bit-identical micro); slots without a logical rank (a node emulating fewer logical nodes) = dummy micros.
aloe-v2vox-asr-w15973-textB (10-20 23:25 PT, user: "user speech in, no assistant speech, just text"): env ALOE_B_TEXT_ONLY (default 0 = the
  path above, unchanged). ALOE_B_TEXT_ONLY=1 changes ONLY channel B (assistant = model channel B, both orientations as before):
    * no B speech INPUT anywhere: depth[1] = -1 at every position (no B Mimi codes in the conversation -> aud_B contributes exactly 0), and
      NO B voice clip in the system block: every item gets the EMPTY system block (header + footer, P = 5) = what the template's items
      without a realised clip already get (the seeded 30 % draw is still evaluated and logged as voice_drawn; voice_clip = 0 always);
    * no B speech OUTPUT targets: cb0 = -100 everywhere, no light frames (len(light) = 0);
    * B's TEXT lane unchanged: ids[1] in the conversation = data_s3q.channel_stream of B's data side = the row's stored word timings
      (words_<side> start/end from the forced alignment made at row-build time, piece_ids / piece_offsets): each word's pieces at their
      frames, PAD elsewhere, NEW_WORD inserted before each word (insert_new_word), EOB by the eob_gap_s rule; targets sel / y / word / pad /
      eob / nw = the template's (only shifted by -(clip length + 5) on the items whose template prefix carried a clip). No audio is read
      for B at train time (the Mimi codes in the row are simply not used).
    Channel A (Voxtral latents of the user's speech through the adapter), the plan, items, order, seeds = unchanged."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "lib"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import data_aloe_v2 as A2  # noqa: E402   (v2's data module, unchanged copy; sha asserted against aloe-v2-ft's by the trainer)

D = A2.D
Q = A2.Q
STORES = A2.STORES
SRC_STORES = A2.SRC_STORES
DIGESTS = A2.DIGESTS
BUDGET = A2.BUDGET
FLAC_ROOT = os.environ.get("ALOE_FLAC_ROOT", "/scratch/amartyavarma/aloe-m1vox-ft/flac")
INDEX = os.environ.get("ALOE_M1VOX_INDEX", "/data/home/amartyavarma/ppm/aloe-m1vox-ft/plan/M1VOX_INDEX.json")
aloe_cfg, eval_cfg, splits, make_plan, rank_items = A2.aloe_cfg, A2.eval_cfg, A2.splits, A2.make_plan, A2.rank_items
_IDX = [None]
GAIN_MODE = os.environ.get("VOX_A_GAIN", "mimi")
RAWGAIN = os.environ.get("ALOE_RAWGAIN", "/data/home/amartyavarma/ppm/aloe-v2vox-raw/plan/RAWGAIN.json")
assert GAIN_MODE in ("mimi", "raw"), ("VOX_A_GAIN must be 'mimi' (recorded Mimi LUFS gain, 22420) or 'raw' (no loudness normalisation)", GAIN_MODE)
_RG = [None]


def rawgain():
    if _RG[0] is None:
        j = json.load(open(RAWGAIN))
        assert j["format"] == "aloe-v2vox-rawgain-v1" and j["plan_digest"] == "ba2d70729ea437ab", (RAWGAIN, j.get("format"), j.get("plan_digest"))
        _RG[0] = j["rows"]
    return _RG[0]


def a_gain(a):
    """the gain (dB) applied to channel A's waveform before Voxtral: 'mimi' = the index's recorded gain (22420); 'raw' = 0 dB for the WLT
    FLAC, minus the recorded Mimi LUFS gain for a mimidec track (its audio was decoded from codes of the normalised track)."""
    if GAIN_MODE == "mimi":
        return float(a["gain_db"])
    e = rawgain()[a["flac"]]
    assert e["kind"] == a["kind"] and abs(float(e["gain_mimi_db"]) - float(a["gain_db"])) < 1e-12, ("RAWGAIN entry != index entry", a["flac"])
    if a["kind"] == "gcs":
        assert float(e["gain_raw_db"]) == 0.0, (a["flac"], e)
    else:
        assert a["kind"] == "mimidec" and e.get("codes_checked") and float(e["gain_raw_db"]) == -float(e["recorded_lufs_gain_db"]), (a["flac"], e)
    return float(e["gain_raw_db"])


def vox_index():
    if _IDX[0] is None:
        j = json.load(open(INDEX))
        assert j["digests"] == A2.DIGESTS, "index built for other stores"
        _IDX[0] = {int(k): v for k, v in j["rows"].items()}
    return _IDX[0]


def a_entry(item, row):
    """item (k, r, swap) -> m1vox index entry of channel A's audio (the OTHER data side than B), checked against the row."""
    k, r, sw = item
    aside = "assistant" if int(sw) else "user"                      # swap 0: B = data assistant, A = data user (data_aloe doc)
    e = vox_index()[int(k)][int(r)]
    assert e["row_key"] == row["row_key"] and e["row"] == int(r), ("index row", item, e["row_key"], row["row_key"])
    a = e["A"][aside]
    assert int(a["n_frames"]) == int(row["n_frames"]) and int(a["frame0"]) == int(row["frame_start"]), (item, a, row["frame_start"])
    assert a["a_spk"] == row[f"{aside}_speaker_dir"], ("A audio speaker != the row's A side", a["a_spk"], row[f"{aside}_speaker_dir"])
    return aside, a


B_TEXT_ONLY = os.environ.get("ALOE_B_TEXT_ONLY", "0") == "1"     # aloe textB (module doc): 1 = channel B text lane only, no B speech in / out
assert os.environ.get("ALOE_B_TEXT_ONLY", "0") in ("0", "1"), ("ALOE_B_TEXT_ONLY must be 0 or 1", os.environ.get("ALOE_B_TEXT_ONLY"))


def prefix_textB(row, swap, cfg):
    """textB: the EMPTY system block for every item (no B voice clip = no B speech in the prefix); the v2 30 % draw kept as provenance."""
    sw = int(swap)
    wv = A2.want_voice(row["row_key"], sw, cfg["voice_seed"])
    spx = D.SPR.build_prefix(row, [{"kind": "text", "track": 1, "ids": []}], {"new_word": cfg["new_word"]})
    assert spx["P"] == 5, ("textB: empty system block must be 5 positions", spx["P"])
    assert (spx["ids"][0] == D.PAD_ID).all() and not (np.asarray(spx["depth"]) >= 0).any(), "textB prefix must carry no codes on either channel"
    return spx, {"voice_drawn": int(wv), "clips": [], "voice_clip": 0, "P": 5, "textB_clip_dropped_if_drawn": int(wv)}


def speech_micro_textB(row, node_row, sib, cfg, rank, swap):
    """= data_aloe_v2.speech_micro_v2 line for line with prefix_textB in place of prefix_v2, then B's speech blanked: depth[1] = -1
    everywhere, cb0 = -100 everywhere, no light frames. B's text lane / text targets = row_arrays' (unchanged maths)."""
    sw = int(swap)
    spx, pv = prefix_textB(row, sw, cfg)
    P, n = int(spx["P"]), int(row["n_frames"])
    rf = P + n if not cfg["crop_conv"] else 7500
    dc = {"new_word": cfg["new_word"], "eob_gap_s": cfg["eob_gap_s"], "wpid": cfg["wpid"], "light_seed": cfg["light_seed"], "row_frames": rf,
          "frame_light": True, "no_crop": not cfg["crop_conv"]}
    a = D.DS.row_arrays(row, dc, rank, int(node_row), "ppm", float(cfg["light_frac"]), sp=spx, swap=sw)
    sel, y, word, pad, eob, nw, cb0, light = D._b_only(a)
    T = len(a["pid"])
    assert (cb0[:P + D.CD] < 0).all() and (light >= P + D.CD).all() and (sel >= P).all()
    depth = a["depth"].copy()
    depth[1] = -1                                                    # textB: no B Mimi codes at any position
    cb0 = np.full_like(cb0, -100)                                    # textB: no B cb0 targets
    light = light[:0].copy()                                         # textB: no B light-decoder frames
    out = {"kind": "speech", "ids": a["ids"], "depth": depth, "present": a["present"], "pid": a["pid"], "segs": [T],
           "sel": sel, "y": y, "word": word, "pad": pad, "eob": eob, "nw": nw, "n_text": float(len(sel)), "cb0": cb0, "light": light,
           "P": P, "swap": sw}
    assert T == (7500 if cfg["crop_conv"] else P + n) + D.CD, ("row length", T, P, n)
    pv.update(row_key=row["row_key"], swap=sw, chan_B_data_side=("user" if sw else "assistant"), n_frames=n, T=T, n_light=0, chan_B="text_lane_only")
    out["prov"] = json.dumps(pv)
    out["voice_clip"] = 0
    out["voice_drawn"] = pv["voice_drawn"]
    return out


def speech_micro_v2vox(row, node_row, sib, cfg, rank, swap):
    """= data_aloe_v2.speech_micro_v2 (v2 prefix, PPM speech-row maths), then channel A emptied in the conversation (= speech_micro_vox).
    textB (ALOE_B_TEXT_ONLY=1): speech_micro_textB instead (empty system block, B speech blanked), then the SAME channel-A edit."""
    m = speech_micro_textB(row, node_row, sib, cfg, rank, swap) if B_TEXT_ONLY else A2.speech_micro_v2(row, node_row, sib, cfg, rank, swap)
    if B_TEXT_ONLY:
        assert (m["depth"][1] < 0).all() and (np.asarray(m["cb0"]) < 0).all() and len(m["light"]) == 0 and int(m["P"]) == 5, "textB: B speech must be absent"
    P, n = int(m["P"]), int(row["n_frames"])
    assert (m["depth"][0, :P] < 0).all() and (m["ids"][0, :P] == D.PAD_ID).all(), "v2 prefix must carry nothing on channel A"
    m["ids"] = m["ids"].copy(); m["depth"] = m["depth"].copy(); m["present"] = m["present"].copy()
    m["ids"][0, P:] = D.PAD_ID
    m["depth"][0] = -1
    m["present"][0, P:] = 0.0
    m["vin"] = (P, n)
    pv = json.loads(m["prov"]); pv["chan_A"] = "voxtral_latents_only"; m["prov"] = json.dumps(pv)
    return m


class Stores(A2.Stores):
    """= data_aloe_v2.Stores (no persona sidecar) with the vox micro + A waveform (= data_aloe_m1vox.Stores.conv's audio path)."""

    def conv(self, item, cfg, rank, audio=True):
        import audio_s4q as AU
        import rowcache_vox as RC                                   # SPEED FIX 10-16: node-local decoded-row cache (bit-identical)
        k, r, sw = item
        node, st, sib = self.open(int(k))
        row = st[int(r)]
        D.R3.check_row(row)
        assert int(row["chunk_index"]) == 0, ("v2: first chunks only", row["row_key"], row["chunk_index"])
        m = speech_micro_v2vox(row, int(r), sib, cfg, rank, int(sw))
        T = len(m["pid"])
        assert m["segs"] == [T] and np.array_equal(m["pid"], np.arange(T)) and m["swap"] == int(sw)
        assert T == int(m["P"]) + int(row["n_frames"]) + D.CD <= BUDGET, ("segment length", T, m["P"], row["n_frames"])
        aside, a = a_entry(item, row)
        g_db = a_gain(a)                                             # aloe-v2vox-raw: VOX_A_GAIN ('mimi' = 22420's recorded gain)
        m["aref"] = {"row_key": row["row_key"], "side": aside, "flac": a["flac"], "gain_db": g_db, "frame0": int(a["frame0"]),
                     "n": int(a["n_frames"]), "kind": a["kind"], "gain_mode": GAIN_MODE, "gain_mimi_db": float(a["gain_db"])}
        if audio:
            path = f"{FLAC_ROOT}/{a['flac']}"
            if a["kind"] == "gcs":
                assert os.path.getsize(path) == int(a["flac_bytes"]), ("A FLAC missing / wrong size on this node", path)
            else:                                                    # purged conversation: Mimi decode of the row's A codes (staged per node)
                assert a["kind"] == "mimidec" and int(a["audio_frame0"]) == 0 and os.path.getsize(path) > 0, ("A mimidec file", path)
            m["wav"] = RC.row_audio({"flac": path, "frame0": int(a["audio_frame0"]), "n_frames": int(a["n_frames"]), "row_key": row["row_key"]},
                                    g_db)
            assert m["wav"].dtype == np.float32 and m["wav"].shape == (int(a["n_frames"]) * AU.SAMPLES_PER_FRAME_24K,)
        m["row_key"] = row["row_key"]
        m["item"] = [int(k), int(r), int(sw)]
        return m


def as_micro(m, dummy=False):
    out = A2.as_micro(m, dummy=dummy)                               # v2's (adds voice_clip / voice_drawn)
    assert out["segs"] == [len(m["pid"])]
    out["present"] = m["present"]
    out["vin"] = np.asarray(m["vin"], np.int64)
    out["aref"] = json.dumps(m["aref"])
    if "wav" in m:
        out["wav"] = m["wav"]
    return out


class VoxDS:
    """= data_aloe_m1vox.VoxDS (one rank's stream, dummy where the plan is exhausted) with the v2vox micro; singles = the micro without its wav."""

    def __init__(self, items, cfg, rank, fallback, singles_at=0):
        self.items, self.cfg, self.rank, self.fb, self.singles_at = list(items), dict(cfg), int(rank), fallback, singles_at
        self.st = None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, t):
        if self.st is None:
            self.st = Stores()
        it = self.items[int(t)]
        m = self.st.conv(it if it is not None else self.fb, self.cfg, self.rank)
        out = as_micro(m, dummy=it is None)
        out["step"] = int(t)
        if int(t) == self.singles_at:
            s1 = as_micro(m)
            s1.pop("wav", None)
            out["singles"] = [s1]
        return out


class AccDS:
    """aloe-v2vox-raw: one PHYSICAL rank's stream emulating logical ranks `lranks` (slot g -> logical rank lranks[g]; None = no logical
    rank: dummy) of 22420's 64. Element i = step i // G, slot i % G: = VoxDS(rank L_g).__getitem__(t) exactly (same item, same rank arg,
    dummy where the plan is exhausted), + 'slot' / 'lrank' / 'padslot' (padslot = a slot without a logical rank: dummy micro of the fallback item)."""

    def __init__(self, rows_by_slot, lranks, cfg, fallback, singles_at=0):
        assert len(rows_by_slot) == len(lranks) and lranks[0] is not None
        self.rows, self.lr, self.cfg, self.fb, self.singles_at = [list(r) if r is not None else None for r in rows_by_slot], list(lranks), dict(cfg), fallback, singles_at
        self.G = len(lranks)
        self.T = len(self.rows[0])
        assert all(r is None or len(r) == self.T for r in self.rows)
        self.st = None

    def __len__(self):
        return self.T * self.G

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
        out["step"], out["slot"], out["lrank"], out["padslot"] = t, g, (-1 if pad else int(L)), int(pad)   # NOT "pad" (= the timing-CE target array)
        if t == self.singles_at:                                     # pad slots too (their loss check runs for FSDP call parity, ignored)
            s1 = as_micro(m)
            s1.pop("wav", None)
            out["singles"] = [s1]
        return out


def eval_micros(items, cfg, rank):
    st = Stores()
    return [as_micro(st.conv(it, cfg, rank)) for it in items]


def node_items(plan, node, world, lw=8):
    """every item node `node`'s ranks read under this plan (train incl. the dummy fallback, eval) -> sorted unique list."""
    out = set()
    for R in range(node * lw, (node + 1) * lw):
        tr, ev = rank_items(plan, R, world)
        out |= {tuple(x) for x in tr if x is not None} | {tuple(x) for x in ev}
        if any(x is None for x in tr):
            out.add(tuple(plan["items"][0]))
    return sorted(out)
