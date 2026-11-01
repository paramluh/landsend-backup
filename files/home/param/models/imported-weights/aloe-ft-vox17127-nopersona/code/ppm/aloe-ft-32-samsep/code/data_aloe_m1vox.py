"""aloe-m1vox-ft DATA (10-16, user): job 21980's data EXACTLY (data_aloe.py, unchanged copy: Model 1's Aloe handover 'none' stores m1n0..m1n3
staged to every node's $ALOE_SCRATCH/aloe-ft/stores, digests asserted; plan = every train row x both orientations permuted with seed
20261011, item t x WORLD + R for rank R at step t, dummies on the last step; eval = 240 rows x 2 round-robin over ranks) with ONLY channel A
changed (= aloe-vox-ft's data_aloe_vox.speech_micro_vox): A's voice clip dropped from the prefix, A has no text (PAD), no Mimi, present_A = 0
in the conversation; vin = (P, n): the trainer adds s x vadapter(Voxtral(A audio of row frames 0..n-1)) at positions P..P+n-1.
A AUDIO: the other speaker's WLT FLAC (plan/M1VOX_INDEX.json: build_m1vox_index.py, code alignment vs aloe-handover-full verified on every
row / side), node-local under ALOE_FLAC_ROOT (each node holds the A tracks its ranks read: plan/NEEDS.json), decoded in the DataLoader
workers exactly as aloe-vox-ft (audio_s4q.ffmpeg_row_audio, recorded Mimi LUFS gain). 163 rows of 122 conversations purged from the handover
after 21980 (noface exclusion; no audio left in GCS): A audio = Mimi decode of the row's own A codes (build_m1vox_index doc)."""
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
import data_aloe as AL  # noqa: E402   (21980's data module, unchanged)
import data_aloe_vox as V  # noqa: E402   (speech_micro_vox: channel A = Voxtral latents only)

D = AL.D
Q = AL.Q
STORES = AL.STORES
BUDGET = AL.BUDGET
FLAC_ROOT = os.environ.get("ALOE_FLAC_ROOT", f"{_ALOE_SCRATCH}/aloe-m1vox-ft/flac")
INDEX = os.environ.get("ALOE_M1VOX_INDEX", f"{_ALOE_ROOT}/ppm/aloe-m1vox-ft/plan/M1VOX_INDEX.json")
aloe_cfg, splits, make_plan, rank_items = AL.aloe_cfg, AL.splits, AL.make_plan, AL.rank_items
_IDX = [None]


def vox_index():
    if _IDX[0] is None:
        j = json.load(open(INDEX))
        assert j["digests"] == AL.DIGESTS, "index built for other stores"
        _IDX[0] = {int(k): v for k, v in j["rows"].items()}
    return _IDX[0]


class Stores(AL.Stores):
    def conv(self, item, cfg, rank, audio=True):
        import audio_s4q as AU
        import rowcache_vox as RC                                   # SPEED FIX 10-16: node-local decoded-row cache (bit-identical)
        k, r, sw = item
        node, st, sib, ps = self.open(int(k))
        row = st[int(r)]
        D.R3.check_row(row)
        try:
            pers = ps.get(int(r), row["row_key"])
        except (FileNotFoundError, OSError):
            pers = None
        m = V.speech_micro_vox(row, int(r), sib, pers, cfg, rank, swap=int(sw))       # = D.speech_micro with channel A emptied
        T = len(m["pid"])
        assert m["segs"] == [T] and np.array_equal(m["pid"], np.arange(T)) and m["swap"] == int(sw)
        assert T == int(m["P"]) + int(row["n_frames"]) + D.CD <= BUDGET, ("segment length", T, m["P"], row["n_frames"])
        aside = "assistant" if int(sw) else "user"                  # swap 0: B = data assistant, A = data user (data_aloe doc)
        e = vox_index()[int(k)][int(r)]
        assert e["row_key"] == row["row_key"] and e["row"] == int(r), ("index row", item, e["row_key"], row["row_key"])
        a = e["A"][aside]
        assert int(a["n_frames"]) == int(row["n_frames"]) and int(a["frame0"]) == int(row["frame_start"]), (item, a, row["frame_start"])
        assert a["a_spk"] == row[f"{aside}_speaker_dir"], ("A audio speaker != the row's A side", a["a_spk"], row[f"{aside}_speaker_dir"])
        m["aref"] = {"row_key": row["row_key"], "side": aside, "flac": a["flac"], "gain_db": float(a["gain_db"]), "frame0": int(a["frame0"]),
                     "n": int(a["n_frames"])}
        m["aref"]["kind"] = a["kind"]
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
    out = AL.as_micro(m, dummy=dummy)
    assert out["segs"] == [len(m["pid"])]
    out["present"] = m["present"]
    out["vin"] = np.asarray(m["vin"], np.int64)
    out["aref"] = json.dumps(m["aref"])
    if "wav" in m:
        out["wav"] = m["wav"]
    return out


class VoxDS:
    """= data_aloe.AloeDS (one rank's stream, dummy where the plan is exhausted) with the vox micro; singles = the micro without its wav."""

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
