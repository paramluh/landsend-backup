"""ROSIE STAMP speaker fine-tune DATA (export-253 lineage: train_cotrain.py -> train_samsep_vox.py), CPU only (DataLoader workers, tests).

SOURCE   HF InternalCan/rosie-stamp-conversations-v1 (216 conversations, 35.0 h), prepared once by /workspace/rosie-ft/prep_rosie.py into
         <root>/prepared/ (index.json, rows/<conv>.json, codes/<conv>.{rosie,user}.npy, wav/<conv>.user.f32.npy) + <root>/flac/.
ROWS     every conversation (session_id/seg) is cut into 7,500-frame (600 s) chunks; ALL chunks are used; a chunk < 125 frames is dropped
         (S3_MIN_FRAMES=125). One chunk = one in-memory s3rows-format row (lib/s3rows.py: codes_{user,assistant} int16 [32, 7500] with -1
         past n, words_<side> {text, start, end (s, chunk-relative), start_frame, end_frame}, piece_ids_<side> / piece_offsets_<side> =
         speech_q.word_pieces on the Qwen3-32B tokenizer over the chunk's word list), checked with s3rows.check_row.
         data 'assistant' = ROSIE (her rosie_lane, whichever SPEAKER_0x it is), data 'user' = the partner. swap = 0 ALWAYS -> model channel
         B = Rosie, her ORIGINAL Mimi codes (never SAM), her words only on the B lane.
MICRO    = the trainer's own builders, unchanged: data_aloe_v2vox.speech_micro_v2vox (-> data_aloe_v2.speech_micro_v2 -> prefix_v2 ->
         data_ppm.voice_clips / sysprompt.build_prefix -> data_s3q.row_arrays) and data_aloe_v2vox.as_micro (-> data_qqa.pack). The only
         Rosie-specific inputs are:
           * VOICE CLIP on 100 % of items (v2's want_voice drawn with h64(..) % 1 < 1: VOICE_NUM / VOICE_DEN set to 1 / 1 for the call,
             under a lock) from a DIFFERENT TRAIN CALL (different session_id): a ClipSib stands in for sysprompt.SibIndex and offers the
             candidate chunks as "later chunks in the second half" so data_ppm.voice_clips runs unchanged (its L0 / u draws seeded by
             cfg['seed'] and the row_key, 38..125 frames, sysprompt.window: stride 4, coverage >= 0.5, top_frac 0.9, lane BLANKED).
             Candidate order is deterministic per item: train calls != the item's session, rotated by h64('rosie-clip|<seed>|<row_key>'),
             each call's clip-feasible chunks rotated by the same hash; the first feasible window wins (index-time feasibility check at
             L = 38 guarantees the first candidate succeeds). Eval items also take clips from train calls.
           * channel A: no codes / no text in the micro (speech_micro_v2vox: present_A = 0 at p >= P, ids PAD, depth -1); wav = the
             PARTNER's original FLAC decoded exactly as audio_s4q.ffmpeg_row_audio (ffmpeg -f f32le -ac 1 -ar 24000; slice
             [1920 frame0, +1920 n), zero pad) at gain 0 dB (raw level, NO loudness normalisation; for an identical-channel stereo FLAC
             ffmpeg's mono downmix = sqrt(2) x one channel = +3.01 dB). Served from the prepared per-conversation float32 decode
             (prep stage 'wav', bit-identical to ffmpeg_row_audio, asserted by the tests) or decoded on the fly when absent.
           * light: data_s3q's per-frame rule (u < light_frac, seed (light_seed, h64(row_key), track)); light_frac is a parameter
             (prior Rosie run: 1.0). light_max (default None = off) optionally keeps a seeded subset of <= light_max light frames
             (the trainer itself caps EVAL light frames at QQA_EVAL_LIGHT_MAX = 2048).
SPLIT    eval = the 4 held-out calls 09bb73ac, 4cc4c9fc, 88e8e0f4, d65a446f (session_id prefix, seg00); everything else = train.

INTERFACE
  RosieData(root='/workspace/data/rosie', split='train'|'eval'|'all', light_frac=1.0, seed=CLIP_SEED, light_max=None, cfg=None)
    .items()            -> list of item keys (conv_idx, chunk_index, 0)   (tuple; the last element = swap, always 0)
    .n_items            -> len(items())
    .micro(item, dummy=False, audio=True) -> the trainer micro dict (data_aloe_v2vox.as_micro output):
          kind 'speech', ids int64 [2, T], depth int64 [2, T, 32] (code + 2048 k, -1 none), present f32 [2, T], pid [T], segs [T],
          sel / y / word / pad / eob / nw (channel B targets), n_text, cb0 [T], light [L], own [1, 3], P int64 [1], swap 0,
          item json '[c, ci, 0]', voice_clip 1, voice_drawn 1, prov json [row_key], vin int64 [2] = (P, n), aref json
          {row_key, side 'user', flac, gain_db 0.0, frame0, n, kind 'gcs', gain_mode 'raw', gain_mimi_db, lufs_in}, wav f32 [n x 1920]
          (only with audio=True), rprov json (Rosie provenance: conv, chunk, split, clip source); dummy=True: + dummy 1, own zeroed.
          T = P + n + CD (CD = 2), P = L + 9 (clip of L frames).
    .conv(item, audio=True) -> the pre-pack conversation dict (speech_micro_v2vox output + wav / aref / row_key / item)
    .row(item)          -> the in-memory s3rows row
  make_plan(world, epochs=1, seed=PLAN_SEED, steps_per_epoch=None, root=...) -> {'train': [step][rank] item | None (dummy),
          'eval': [items], 'steps', 'steps_per_epoch', 'epochs', 'world', 'fallback', 'n_train', 'n_eval'} (= data_aloe_v2.make_plan:
          seeded permutation per epoch, item t x world + r, dummies (None) only on the last step of each epoch).
  rank_items(plan, rank) -> (train items per step (None = dummy), eval items of this rank (eval[rank::world]))
  RosieDS(data, items, fallback, singles_at=0): one rank's stream (= data_aloe_v2vox.VoxDS)."""
import json
import math
import os
import shutil
import sys
import threading

import numpy as np

os.environ.setdefault("S3_MIN_FRAMES", "125")                    # check_row's floor (read at s3rows import)
HERE = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.join(HERE, "lib"), HERE):
    if p not in sys.path:
        sys.path.insert(0, p)
import data_aloe_v2vox as AV  # noqa: E402
import audio_s4q as AU  # noqa: E402

A2, D = AV.A2, AV.D
DS, SPR, SP, R3 = D.DS, D.SPR, D.SP, D.R3
assert R3.MIN_FRAMES <= 125, ("s3rows was imported with S3_MIN_FRAMES > 125; set S3_MIN_FRAMES=125 before importing the data modules", R3.MIN_FRAMES)
if not os.path.exists(AU.FFMPEG_EXE):                             # the cluster's imageio ffmpeg is absent here: the system ffmpeg
    AU.FFMPEG_EXE = os.environ.get("S4_FFMPEG") or shutil.which("ffmpeg") or AU.FFMPEG_EXE

ROOT = "/workspace/data/rosie"
TOKENIZER = os.path.join(os.path.dirname(HERE), "tokenizer")    # Qwen3-32B tokenizer files (copy of the trainer's, REVISION 9216db57)
CHUNK = 7500
MIN_FRAMES = 125
EVAL_PREFIXES = ("09bb73ac", "4cc4c9fc", "88e8e0f4", "d65a446f")
CLIP_SEED = 20261030
PLAN_SEED = 20261030
FAKE_CHUNK0 = 1_000_000                                           # ClipSib: candidates appear as later chunks FAKE_CHUNK0 + j
FORMAT = "rosie-prepared-v1"
_VOICE_LOCK = threading.Lock()


def h64(s):
    return D.h64(s)


def conv_name(sid, seg):
    return f"{sid}_{seg}"


def row_key_of(sid, seg, ci):
    return f"rosie/{sid}/{seg}/c{int(ci):02d}"


def rosie_cfg(light_frac=1.0):
    """= data_aloe_v2.aloe_cfg() (PPM / v2 cfg: clip 38..125, stride 4, min_cov 0.5, top_frac 0.9, new_word, wpid both, light_seed,
    crop_conv False, sysprompt v2_voiceB_only) with light_frac given."""
    return dict(A2.aloe_cfg(), light_frac=float(light_frac))


# ------------------------------------------------------------------------------------------------ index build (prep_rosie.py 'index')
def _chunk_words(words, frame0, n):
    """the words of a chunk: start in [frame0 / 12.5, frame0 / 12.5 + n / 12.5), re-timed to the chunk; end clipped to the chunk."""
    t0, dur = frame0 / 12.5, n / 12.5
    out = []
    for w in words:
        s = float(w["start"]) - t0
        if not (0.0 <= s < dur):
            continue
        e = min(max(float(w["end"]) - t0, s), dur)
        out.append({"text": w["text"], "start": s, "end": e, "start_frame": R3.frame_of(s), "end_frame": min(R3.frame_of(e), n),
                    "in_overlap": False, "truncated": bool(float(w["end"]) - t0 > dur)})
    return out


_TOKW = [None]


def _tok():
    if _TOKW[0] is None:
        from transformers import AutoTokenizer
        _TOKW[0] = AutoTokenizer.from_pretrained(TOKENIZER)
    return _TOKW[0]


def _prep_conv(args):
    """one conversation: codes -> prepared/codes, chunk rows (words + pieces) -> prepared/rows/<conv>.json. -> index entry."""
    import io
    import pyarrow.parquet as pq
    root, meta = args
    prep = f"{root}/prepared"
    t = pq.read_table(f"{root}/{meta['token_shard']}", columns=["session_id", "seg", "mimi_T", "rosie_original_codes", "user_original_codes",
                                                                 "words_json", "original_mimi_json"])
    r = t.slice(int(meta["token_row"]), 1).to_pylist()[0]
    sid, seg = r["session_id"], r["seg"]
    assert (sid, seg) == (meta["session_id"], meta["seg"])
    T = int(r["mimi_T"])
    name = conv_name(sid, seg)
    codes = {}
    for side, col in (("rosie", "rosie_original_codes"), ("user", "user_original_codes")):
        c = np.load(io.BytesIO(r[col]))
        assert c.shape == (32, T) and c.dtype == np.int16 and int(c.min()) >= 0 and int(c.max()) < 2048, (name, side, c.shape, c.dtype)
        codes[side] = c
        dst = f"{prep}/codes/{name}.{side}.npy"
        tmp = dst[:-4] + ".part.npy"
        np.save(tmp, c)
        os.replace(tmp, dst)
    W = json.loads(r["words_json"])["words"]
    by = {"assistant": [w for w in W if w["speaker"] == "Rosie Stamp"], "user": [w for w in W if w["speaker"] == "user"]}
    assert len(by["assistant"]) + len(by["user"]) == len(W), (name, "speaker labels")
    assert {w["lane"] for w in by["assistant"]} <= {meta["rosie_lane"]} and {w["lane"] for w in by["user"]} <= {meta["user_lane"]}, (name, "lanes")
    mj = json.loads(r["original_mimi_json"]).get("mimi", {})
    lufs = {side: {"lufs_in": mj.get(lane, {}).get("lufs_in"), "lufs_gain_db": mj.get(lane, {}).get("lufs_gain_db")}
            for side, lane in (("rosie", meta["rosie_lane"]), ("user", meta["user_lane"]))}
    tok = _tok()
    chunks, rows = [], []
    for ci in range(math.ceil(T / CHUNK)):
        f0 = ci * CHUNK
        n = min(CHUNK, T - f0)
        ent = {"ci": ci, "frame0": f0, "n": n, "kept": n >= MIN_FRAMES}
        chunks.append(ent)
        rw = {"ci": ci, "frame0": f0, "n": n}
        for side in ("user", "assistant"):
            ws = _chunk_words(by[side], f0, n)
            pcs = SP.word_pieces(tok, ws)
            rw[f"words_{side}"] = ws
            rw[f"piece_ids_{side}"] = [int(x) for p in pcs for x in p]
            rw[f"piece_offsets_{side}"] = [0] + np.cumsum([len(p) for p in pcs]).astype(int).tolist() if pcs else [0]
            ent[f"n_words_{side}"] = len(ws)
        # clip feasibility of this chunk as a voice-clip SOURCE (window at the minimum length: feasible -> any draw succeeds)
        ent["clip_ok"] = bool(n >= MIN_FRAMES and SPR.window(rw["words_assistant"], n, 38, 0.0, {"clip_min": 38, "stride": 4, "min_cov": 0.5,
                                                                                                    "top_frac": 0.9}) is not None)
        rows.append(rw)
    dst = f"{prep}/rows/{name}.json"
    with open(dst + ".part", "w") as f:
        json.dump({"session_id": sid, "seg": seg, "T": T, "rows": rows}, f)
    os.replace(dst + ".part", dst)
    split = "eval" if (sid[:8] in EVAL_PREFIXES and seg == "seg00") else "train"
    return {"name": name, "session_id": sid, "seg": seg, "set": meta["source_dataset"], "split": split, "T": T,
            "dur": float(meta["clip_duration_sec"]), "rosie_lane": meta["rosie_lane"], "user_lane": meta["user_lane"],
            "partner_name": meta["partner_name"] or "user", "has_sam": bool(meta["has_sam"]),
            "flac_user": f"flac/{sid}/{seg}/user.original.flac", "flac_rosie": f"flac/{sid}/{seg}/Rosie_Stamp.original.flac",
            "lufs": lufs, "chunks": chunks}


def _prep_wav(args):
    root, c = args
    dst = f"{root}/prepared/wav/{c['name']}.user.f32.npy"
    src = f"{root}/{c['flac_user']}"
    if os.path.exists(dst):
        return dst, "present"
    x = AU.ffmpeg_decode_24k(src)
    tmp = dst[:-4] + ".part.npy"
    np.save(tmp, x)
    os.replace(tmp, dst)
    return dst, "built"


def build_index(root=ROOT, n_workers=16):
    import pyarrow.parquet as pq
    from multiprocessing import Pool
    prep = f"{root}/prepared"
    for d in ("codes", "rows", "wav"):
        os.makedirs(f"{prep}/{d}", exist_ok=True)
    idx = pq.read_table(f"{root}/index/conversations.parquet").to_pandas().sort_values(["session_id", "seg"]).reset_index(drop=True)
    assert len(idx) == 216, len(idx)
    metas = [dict(r) for _, r in idx.iterrows()]
    for m in metas:
        for k in ("rosie_lane", "user_lane"):
            assert m[k] in ("SPEAKER_00", "SPEAKER_01")
        assert m["rosie_lane"] != m["user_lane"]
        assert os.path.getsize(f"{root}/flac/{m['session_id']}/{m['seg']}/user.original.flac") > 0
    with Pool(n_workers) as pool:
        convs = pool.map(_prep_conv, [(root, m) for m in metas])
    ev = sorted({c["session_id"][:8] for c in convs if c["split"] == "eval"})
    assert tuple(ev) == EVAL_PREFIXES and sum(c["split"] == "eval" for c in convs) == 4, ev
    cands = [[i, ch["ci"]] for i, c in enumerate(convs) if c["split"] == "train" for ch in c["chunks"] if ch["kept"] and ch["clip_ok"]]
    out = {"format": FORMAT, "root": root, "tokenizer": TOKENIZER, "tokenizer_revision": open(f"{TOKENIZER}/REVISION").read().strip(),
           "chunk": CHUNK, "min_frames": MIN_FRAMES, "eval_prefixes": list(EVAL_PREFIXES), "convs": convs, "clip_cands": cands}
    st = {}
    for sp in ("train", "eval"):
        cs = [c for c in convs if c["split"] == sp]
        ks = [ch for c in cs for ch in c["chunks"] if ch["kept"]]
        st[sp] = {"convs": len(cs), "items": len(ks), "frames": int(sum(ch["n"] for ch in ks)), "hours": sum(ch["n"] for ch in ks) / 12.5 / 3600,
                  "dropped_chunks": int(sum(1 for c in cs for ch in c["chunks"] if not ch["kept"])),
                  "dropped_frames": int(sum(ch["n"] for c in cs for ch in c["chunks"] if not ch["kept"]))}
    out["stats"] = st
    with open(f"{prep}/index.json.part", "w") as f:
        json.dump(out, f)
    os.replace(f"{prep}/index.json.part", f"{prep}/index.json")
    print("[index]", json.dumps(st), f"clip candidates {len(cands)}", flush=True)
    return out


def build_wav(root=ROOT, n_workers=16):
    from multiprocessing import Pool
    ix = json.load(open(f"{root}/prepared/index.json"))
    os.makedirs(f"{root}/prepared/wav", exist_ok=True)
    with Pool(n_workers) as pool:
        res = pool.map(_prep_wav, [(root, c) for c in ix["convs"]])
    print(f"[wav] {sum(s == 'built' for _, s in res)} built, {sum(s == 'present' for _, s in res)} present", flush=True)


# ------------------------------------------------------------------------------------------------ runtime
class _LazyRows:
    def __init__(self, data):
        self.data, self.c = data, {}

    def __getitem__(self, nr):
        if nr not in self.c:
            self.c[nr] = self.data.row(self.data.cand_item(nr))
        return self.c[nr]


class ClipSib:
    """stands in for sysprompt.SibIndex inside data_ppm.voice_clips for ONE item: the item's clip-source candidates (chunks of OTHER train
    calls, deterministic order) appear as later chunks FAKE_CHUNK0 + j of the item's recording (so voice_clips' 'later chunk in the second
    half' condition holds for all of them, nearest = j = 0 first); rows[nr] loads candidate nr lazily."""

    def __init__(self, data, row, order):
        self.data, self.order = data, order
        rk = SPR.rec_key(row["row_key"])
        ent = [(int(row["chunk_index"]), -1, row["user_speaker"], row["assistant_speaker"])]
        ent += [(FAKE_CHUNK0 + j, int(nr), "", row["assistant_speaker"]) for j, nr in enumerate(order)]
        self.by_rec = {rk: ent}
        self.rows = _LazyRows(data)

    def siblings(self, row, side):
        assert side in ("user", "assistant")
        if side == "user":                                           # v2: A never gets a clip (voice_clips still asks for both tracks)
            return [], []
        return [], [(FAKE_CHUNK0 + j, int(nr), "assistant") for j, nr in enumerate(self.order)]


class RosieData:
    def __init__(self, root=ROOT, split="train", light_frac=1.0, seed=CLIP_SEED, light_max=None, cfg=None):
        assert split in ("train", "eval", "all"), split
        self.root, self.split, self.seed = root, split, int(seed)
        self.light_max = None if light_max is None else int(light_max)
        self.cfg = dict(cfg) if cfg is not None else rosie_cfg(light_frac)
        self.cfg["light_frac"] = float(light_frac)
        self.ix = json.load(open(f"{root}/prepared/index.json"))
        assert self.ix["format"] == FORMAT and tuple(self.ix["eval_prefixes"]) == EVAL_PREFIXES
        self.convs = self.ix["convs"]
        self._items = [(i, ch["ci"], 0) for i, c in enumerate(self.convs) if split in ("all", c["split"]) for ch in c["chunks"] if ch["kept"]]
        self.cands = [tuple(x) for x in self.ix["clip_cands"]]          # (conv idx, chunk) of TRAIN calls, clip-feasible
        self.by_call = {}
        for nr, (i, ci) in enumerate(self.cands):
            self.by_call.setdefault(self.convs[i]["session_id"], []).append(nr)
        self.calls = sorted(self.by_call)
        self._rows, self._codes = {}, {}

    # ---- keys
    def items(self):
        return list(self._items)

    @property
    def n_items(self):
        return len(self._items)

    def cand_item(self, nr):
        i, ci = self.cands[int(nr)]
        return (i, ci, 0)

    # ---- data
    def _conv_rows(self, i):
        if i not in self._rows:
            if len(self._rows) > 64:
                self._rows.clear()
            self._rows[i] = json.load(open(f"{self.root}/prepared/rows/{self.convs[i]['name']}.json"))
        return self._rows[i]

    def codes(self, i, side):
        k = (i, side)
        if k not in self._codes:
            self._codes[k] = np.load(f"{self.root}/prepared/codes/{self.convs[i]['name']}.{side}.npy", mmap_mode="r")
        return self._codes[k]

    def row(self, item):
        """item -> the s3rows-format row of that chunk (module doc), check_row'd."""
        i, ci = int(item[0]), int(item[1])
        c = self.convs[i]
        rw = self._conv_rows(i)["rows"][ci]
        f0, n = int(rw["frame0"]), int(rw["n"])
        assert rw["ci"] == ci and f0 == CHUNK * ci and n >= MIN_FRAMES
        row = {"row": i * 16 + ci, "row_key": row_key_of(c["session_id"], c["seg"], ci), "dataset": "ALOE",
               "dataset_folder": "rosie-stamp-conversations-v1", "podcast_id": c["session_id"], "episode_id": c["seg"], "chunk_index": ci,
               "frame_start": f0, "frame_end": f0 + n, "n_frames": n,
               "user_speaker_dir": c["user_lane"], "user_speaker": f"partner:{c['partner_name']}",
               "assistant_speaker_dir": c["rosie_lane"], "assistant_speaker": "Rosie Stamp"}
        for side, cs in (("user", "user"), ("assistant", "rosie")):
            full = np.full((32, CHUNK), R3.PAD_CODE, np.int16)
            full[:, :n] = self.codes(i, cs)[:, f0:f0 + n]
            row[f"codes_{side}"] = full
            row[f"words_{side}"] = [dict(w) for w in rw[f"words_{side}"]]
            row[f"piece_ids_{side}"] = np.asarray(rw[f"piece_ids_{side}"], np.int64)
            row[f"piece_offsets_{side}"] = np.asarray(rw[f"piece_offsets_{side}"], np.int32)
        R3.check_row(row)
        return row

    def clip_order(self, row):
        """the item's clip-source candidates (indices into self.cands): train calls != the row's session, rotated by the item hash; each
        call's feasible chunks rotated by the same hash."""
        sid = row["podcast_id"]
        hv = h64(f"rosie-clip|{self.seed}|{row['row_key']}")
        calls = [s for s in self.calls if s != sid]
        k = hv % len(calls)
        out = []
        for s in calls[k:] + calls[:k]:
            ch = self.by_call[s]
            j = (hv >> 20) % len(ch)
            out += ch[j:] + ch[:j]
        return out

    def wav(self, item):
        """channel A: the partner's raw-level 24 kHz mono decode of the chunk (= audio_s4q.ffmpeg_row_audio(e, 0.0))."""
        i, ci = int(item[0]), int(item[1])
        c = self.convs[i]
        ch = c["chunks"][ci]
        s0, n = int(ch["frame0"]) * AU.SAMPLES_PER_FRAME_24K, int(ch["n"]) * AU.SAMPLES_PER_FRAME_24K
        cache = f"{self.root}/prepared/wav/{c['name']}.user.f32.npy"
        if os.path.exists(cache):
            x = np.load(cache, mmap_mode="r")
            assert x.dtype == np.float32 and x.ndim == 1
            assert len(x) > s0 + n - 6000, ("source flac shorter than the row by > 0.25 s", c["name"], len(x), s0, n)
            w = np.zeros(n, np.float32)
            seg = x[s0:s0 + n]
            w[:len(seg)] = seg
            return (w * np.float32(10.0 ** (0.0 / 20.0))).astype(np.float32)
        return AU.ffmpeg_row_audio({"flac": f"{self.root}/{c['flac_user']}", "frame0": int(ch["frame0"]), "n_frames": int(ch["n"]),
                                    "row_key": row_key_of(c["session_id"], c["seg"], ci)}, 0.0)

    def conv(self, item, audio=True):
        """item -> the conversation dict (speech_micro_v2vox output, before packing) + row_key / item / aref / wav / rprov."""
        i, ci, sw = (int(x) for x in item)
        assert sw == 0, "Rosie is always channel B (swap 0)"
        row = self.row(item)
        order = self.clip_order(row)
        sib = ClipSib(self, row, order)
        with _VOICE_LOCK:                                             # want_voice: h64(..) % 1 < 1 (100 % of items)
            old = A2.VOICE_NUM, A2.VOICE_DEN
            A2.VOICE_NUM, A2.VOICE_DEN = 1, 1
            try:
                m = AV.speech_micro_v2vox(row, int(row["row"]), sib, self.cfg, 0, 0)
            finally:
                A2.VOICE_NUM, A2.VOICE_DEN = old
        pv = json.loads(m["prov"])
        assert m["voice_drawn"] == 1 and m["voice_clip"] == 1 and len(pv["clips"]) == 1 and pv["clips"][0]["track"] == "B", ("no clip", row["row_key"])
        cl = pv["clips"][0]
        j = int(cl["src_chunk"]) - FAKE_CHUNK0
        src = self.cand_item(order[j])
        assert cl["src_key"] == row_key_of(self.convs[src[0]]["session_id"], self.convs[src[0]]["seg"], src[1])
        assert self.convs[src[0]]["session_id"] != row["podcast_id"] and self.convs[src[0]]["split"] == "train"
        T = len(m["pid"])
        P, n = int(m["P"]), int(row["n_frames"])
        L = int(cl["frames"][1] - cl["frames"][0])
        assert m["segs"] == [T] and np.array_equal(m["pid"], np.arange(T)) and m["swap"] == 0
        assert T == P + n + D.CD <= A2.BUDGET and P == L + 7 + D.CD and 38 <= L <= 125, ("segment length", T, P, n, L)
        if self.light_max is not None and len(m["light"]) > self.light_max:
            rng = np.random.default_rng(np.random.SeedSequence([int(self.cfg["light_seed"]), h64(row["row_key"]) & 0x7FFFFFFF, 0x11CA]))
            m["light"] = np.sort(rng.permutation(m["light"])[:self.light_max]).astype(np.int64)
        c = self.convs[i]
        ch = c["chunks"][ci]
        m["aref"] = {"row_key": row["row_key"], "side": "user", "flac": f"{self.root}/{c['flac_user']}", "gain_db": 0.0, "frame0": int(ch["frame0"]), "n": n,
                     "kind": "gcs", "gain_mode": "raw", "gain_mimi_db": c["lufs"]["user"]["lufs_gain_db"], "lufs_in": c["lufs"]["user"]["lufs_in"]}
        if audio:
            m["wav"] = self.wav(item)
            assert m["wav"].dtype == np.float32 and m["wav"].shape == (n * AU.SAMPLES_PER_FRAME_24K,)
        m["row_key"] = row["row_key"]
        m["item"] = [i, ci, 0]
        m["rprov"] = {"conv": c["name"], "chunk": ci, "split": c["split"], "set": c["set"], "rosie_lane": c["rosie_lane"], "n": n, "P": P,
                      "clip": {"src_conv": self.convs[src[0]]["name"], "src_chunk": src[1], "w0": cl["frames"][0], "L": L, "cov": cl["cov"],
                               "cand_rank": j}, "light_frac": self.cfg["light_frac"], "n_light": int(len(m["light"]))}
        return m

    def micro(self, item, dummy=False, audio=True):
        m = self.conv(item, audio=audio)
        out = AV.as_micro(m, dummy=dummy)
        out["rprov"] = json.dumps(m["rprov"])
        return out


# ------------------------------------------------------------------------------------------------ plan / dataset
def make_plan(world, epochs=1, seed=PLAN_SEED, steps_per_epoch=None, root=ROOT):
    """= data_aloe_v2.make_plan (seeded permutation, item t x world + r, None = dummy on the last step), per epoch (seed, epoch)."""
    tr = RosieData(root, "train").items()
    ev = RosieData(root, "eval").items()
    auto = -(-len(tr) // int(world))
    spe = auto if steps_per_epoch is None else int(steps_per_epoch)
    assert spe >= auto, ("steps_per_epoch below one pass over the items", spe, auto)
    grid, flat = [], []
    for e in range(int(epochs)):
        perm = np.random.default_rng(np.random.SeedSequence([int(seed), e])).permutation(len(tr))
        its = [tr[k] for k in perm]
        flat += its
        for t in range(spe):
            grid.append([its[t * world + r] if t * world + r < len(its) else None for r in range(int(world))])
    return {"train": grid, "eval": ev, "steps": len(grid), "steps_per_epoch": spe, "epochs": int(epochs), "world": int(world),
            "fallback": flat[0], "items": flat, "n_train": len(tr), "n_eval": len(ev), "seed": int(seed)}


def rank_items(plan, rank):
    return [s[int(rank)] for s in plan["train"]], plan["eval"][int(rank)::plan["world"]]


class RosieDS:
    """one rank's training stream (= data_aloe_v2vox.VoxDS): element t = the micro of step t (dummy micro of `fallback` where None)."""

    def __init__(self, data_kw, items, fallback, singles_at=0):
        self.kw, self.items, self.fb, self.singles_at, self.d = dict(data_kw), list(items), tuple(fallback), singles_at, None

    def __len__(self):
        return len(self.items)

    def __getitem__(self, t):
        if self.d is None:
            self.d = RosieData(**self.kw)
        it = self.items[int(t)]
        m = self.d.conv(tuple(it) if it is not None else self.fb)
        out = AV.as_micro(m, dummy=it is None)
        out["rprov"] = json.dumps(m["rprov"])
        out["step"] = int(t)
        if int(t) == self.singles_at:
            s1 = AV.as_micro(m)
            s1.pop("wav", None)
            out["singles"] = [s1]
        return out
