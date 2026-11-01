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
recorded Mimi LUFS gain); purged conversations: the Mimi decode of the row's own A codes (m1vox's rule)."""
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


def speech_micro_v2vox(row, node_row, sib, cfg, rank, swap):
    """= data_aloe_v2.speech_micro_v2 (v2 prefix, PPM speech-row maths), then channel A emptied in the conversation (= speech_micro_vox)."""
    m = A2.speech_micro_v2(row, node_row, sib, cfg, rank, swap)
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
        m["aref"] = {"row_key": row["row_key"], "side": aside, "flac": a["flac"], "gain_db": float(a["gain_db"]), "frame0": int(a["frame0"]),
                     "n": int(a["n_frames"]), "kind": a["kind"]}
        if audio:
            path = f"{FLAC_ROOT}/{a['flac']}"
            if a["kind"] == "gcs":
                assert os.path.getsize(path) == int(a["flac_bytes"]), ("A FLAC missing / wrong size on this node", path)
            else:                                                    # purged conversation: Mimi decode of the row's A codes (staged per node)
                assert a["kind"] == "mimidec" and int(a["audio_frame0"]) == 0 and os.path.getsize(path) > 0, ("A mimidec file", path)
            m["wav"] = RC.row_audio({"flac": path, "frame0": int(a["audio_frame0"]), "n_frames": int(a["n_frames"]), "row_key": row["row_key"]},
                                    float(a["gain_db"]))
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
