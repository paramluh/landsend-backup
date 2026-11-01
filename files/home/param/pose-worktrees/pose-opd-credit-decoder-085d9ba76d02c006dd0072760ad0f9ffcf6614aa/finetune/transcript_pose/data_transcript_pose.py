"""Compact production Xabi rows, with a full timed transcript before generation.

The compact cache's ``text.npy`` belongs to SAM and is deliberately never read.
PPM words, Qwen pieces, voice references and the generation lane are reconstructed
with the production helpers. Pose/Mimi codes are sliced verbatim, including when a
long clip is split; a split does not rebase or re-encode frame-0 pose geometry.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np

TRANSCRIPT_START = 151680  # existing SOP embedding, unused by Xabi personas
TRANSCRIPT_END = 151681    # existing EOP embedding
PLAN_SEED = 20261011
PAD, NW, EOB, CTRL_LO = 151670, 151671, 151645, 151643
CD, DEPTH, CBS = 2, 32, 2048
_CJK = re.compile(r"[　-鿿가-힯]")
_BASE = Path(__file__).resolve().parents[1] / "xabi-main-17850" / "code"


@lru_cache(None)
def legacy():
    for p in (_BASE / "lib", _BASE):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import data_aloe_pose as A
    if Path(A.__file__).resolve() != (_BASE / "data_aloe_pose.py").resolve():
        raise RuntimeError(f"Unexpected production data module: {A.__file__}")
    if A.D.CD != CD:
        raise ValueError("The production compact adapter requires PPM_CD=2")
    return A


def clean_words(raw, n):
    """Exact build_v2pure_rows.clip_words -> s2h_to_s3rows.side_words rules."""
    raw = json.loads(raw) if isinstance(raw, str) else raw
    dur, out = int(n) / 12.5, []
    for start, end, text in raw:
        start, end = float(start), float(end)
        if start < -0.5:
            continue
        start = max(0.0, start)
        text = str(text or "").strip()
        if start >= dur or not text or _CJK.search(text):
            continue
        end = min(max(end, start), dur)
        out.append(dict(text=text, start=start, end=end,
                        start_frame=math.floor(start * 12.5 + 1e-6),
                        end_frame=min(math.floor(end * 12.5 + 1e-6), round(dur * 12.5)),
                        in_overlap=False, truncated=False))
    if any(out[i]["start"] < out[i - 1]["start"] for i in range(1, len(out))):
        out.sort(key=lambda w: (w["start"], w["end"]))
    return out


@dataclass(frozen=True)
class Clip:
    shard: str
    cache_index: int
    n: int
    swap: int
    rid: str
    chunk: int
    store: int
    store_row: int
    split: str
    segment_id: str

    @property
    def row_key(self):
        return f"ALOE/aloe-v2pure/{self.rid}/c{self.chunk:05d}"

    @property
    def cache_key(self):
        return f"{self.rid}/speaker_{2 if self.swap else 1}/{self.segment_id}"


class _Rows:
    def __init__(self, owner):
        self.owner = owner

    def __getitem__(self, index):
        return self.owner.row(int(index), tokenize=False)


class CompactXabi:
    """Read ragged arrays directly; retain archived store ordering and row keys.

    ``root`` accepts the release directory or its cache/ child. The archived
    PLAN is mandatory: guessing new row ordering changes deterministic voices.
    Tokenizers and memmaps are opened lazily in DataLoader workers.
    """
    def __init__(self, root, tokenizer_path=None, *, tokenizer=None, cfg=None, shard_cache=8):
        root = Path(root).expanduser().resolve()
        self.cache = root if (root / "plan.json").exists() else root / "cache"
        self.root = self.cache.parent
        self.original = self.root / "provenance" / "original"
        plan_path = self.original / "ppm/aloe-v2pure-ft/stores/PLAN.json"
        self.source_plan_sha256 = hashlib.sha256(plan_path.read_bytes()).hexdigest()
        source = json.loads(plan_path.read_text())
        self.clips, self.split_indices = [], {"train": [], "eval": []}
        self.skipped_lt2 = {"train": 0, "eval": 0}
        for store in range(int(source["stores"])):
            store_row = 0
            for rid, rec in sorted(source["recs"].items()):
                if int(rec["store"]) != store:
                    continue
                for chunk, index, n, speaker, segment_id in rec["items"]:
                    clip = Clip(rec["shard"], int(index), int(n), int(speaker == "speaker_2"), rid,
                                int(chunk), store, store_row, rec["split"], str(segment_id))
                    i = len(self.clips)
                    self.clips.append(clip)
                    if clip.n >= 2:
                        self.split_indices[clip.split].append(i)
                    else:
                        self.skipped_lt2[clip.split] += 1
                    store_row += 1
        self.tokenizer_path = str(tokenizer_path or self.original / "models/Qwen3-32B")
        self._tokenizer = tokenizer
        self.cfg = dict(legacy().aloe_cfg() if cfg is None else cfg)
        self.cfg.update(crop_conv=False, pose_delay=CD, light_frac=1.0, pose_light_frac=1.0)
        self.shard_cache = int(shard_cache)
        self._shards, self._rows = OrderedDict(), OrderedDict()
        self._prefix_lengths = np.full(len(self.clips), -1, np.int16)
        self._compact_lengths = np.full(len(self.clips), -1, np.int32)
        self._compact_tokens_cache = OrderedDict()
        self._silence = None
        self._sib = None

    def __getstate__(self):
        out = self.__dict__.copy()
        out.update(_shards=OrderedDict(), _rows=OrderedDict(), _sib=None, _tokenizer=None,
                   _compact_tokens_cache=OrderedDict())
        return out

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, local_files_only=True)
        return self._tokenizer

    @property
    def sib(self):
        if self._sib is None:
            SPR = legacy().D.SPR
            sib = SPR.SibIndex.__new__(SPR.SibIndex)
            sib.rows, sib.by_rec, sib.keys, sib.recs = _Rows(self), {}, [], []
            for i, c in enumerate(self.clips):
                rk = SPR.rec_key(c.row_key)
                sib.by_rec.setdefault(rk, []).append((c.chunk, i, f"{c.rid}/speaker_2", f"{c.rid}/speaker_1"))
                sib.keys.append(c.row_key)
                sib.recs.append(rk)
            self._sib = sib
        return self._sib

    def _shard(self, name):
        if name not in self._shards:
            import pyarrow.parquet as pq
            p = self.cache / name
            tab = pq.read_table(p / "items.parquet", columns=["key", "n", "words"]).to_pydict()
            self._shards[name] = (tab, np.load(p / "codes_off.npy", mmap_mode="r"),
                                  np.load(p / "codes.npy", mmap_mode="r"),
                                  np.load(p / "pose.npy", mmap_mode="r"))
            while len(self._shards) > self.shard_cache:
                self._shards.popitem(last=False)
        self._shards.move_to_end(name)
        return self._shards[name]

    def row(self, index, tokenize=True):
        index = int(index)
        key = (index, bool(tokenize))
        if key in self._rows:
            self._rows.move_to_end(key)
            return self._rows[key]
        c = self.clips[index]
        tab, off, codes, _ = self._shard(c.shard)
        j = c.cache_index
        assert tab["key"][j] == c.cache_key and int(tab["n"][j]) == c.n
        a, b = int(off[j]), int(off[j + 1])
        assert b - a == c.n
        if self._silence is None:
            p = self.original / "setup-logs/s3hestia/ourrows/silence_codes.json"
            self._silence = np.asarray(json.loads(p.read_text())["seq"], np.int16)
            assert self._silence.shape[0] == DEPTH
        sil = self._silence[:, np.arange(c.n) % self._silence.shape[1]]
        row = dict(row_key=c.row_key, dataset="ALOE", podcast_id="aloe-v2pure", episode_id=c.rid,
                   chunk_index=c.chunk, frame_start=7500 * c.chunk, frame_end=7500 * c.chunk + c.n,
                   n_frames=c.n, user_speaker=f"{c.rid}/speaker_2", assistant_speaker=f"{c.rid}/speaker_1")
        active = "user" if c.swap else "assistant"
        for side in ("user", "assistant"):
            words = clean_words(tab["words"][j], c.n) if side == active else []
            pieces = legacy().D.SP.word_pieces(self.tokenizer, words) if tokenize else [[] for _ in words]
            row[f"words_{side}"] = words
            row[f"piece_ids_{side}"] = np.asarray([x for w in pieces for x in w], np.int64)
            row[f"piece_offsets_{side}"] = np.concatenate([[0], np.cumsum([len(x) for x in pieces])]).astype(np.int64)
            row[f"codes_{side}"] = codes[a:b].T if side == active else sil
        self._rows[key] = row
        while len(self._rows) > 128:
            self._rows.popitem(last=False)
        return row

    def conditioning_length(self, index):
        """Actual deterministic voice prefix length, without Qwen tokenization."""
        index = int(index)
        if self._prefix_lengths[index] < 0:
            c, A = self.clips[index], legacy()
            P = 5
            if A.want_voice(c.row_key, self.cfg):
                row = self.row(index, tokenize=False)
                cl = A.D.voice_clips(row, self.sib, self.cfg)[1 ^ c.swap]
                if cl is not None:
                    P = 9 + int(cl["L"])
            self._prefix_lengths[index] = P
        return int(self._prefix_lengths[index])

    def compact_tokens(self, index, start=0, end=None):
        """Ordinary Qwen text, including words dropped by the timed lane.

        A crop includes a complete word when its timestamp interval overlaps the
        crop. Boundary-crossing words occur in both prefixes. Zero-duration words
        belong to the crop containing their start; word pieces are never sliced.
        """
        index, start = int(index), int(start)
        c = self.clips[index]
        end = c.n if end is None else int(end)
        key = (index, start, end)
        if key not in self._compact_tokens_cache:
            tab, *_ = self._shard(c.shard)
            assert tab["key"][c.cache_index] == c.cache_key
            words = clean_words(tab["words"][c.cache_index], c.n)
            if start != 0 or end != c.n:
                lo, hi = start / 12.5, end / 12.5
                words = [w for w in words if
                         (w["end"] > lo and w["start"] < hi) or
                         (w["end"] == w["start"] and lo <= w["start"] < hi)]
            text = " ".join(w["text"] for w in words)
            # Special-token spellings inside spoken words remain ordinary text,
            # as in the production word_pieces helper.
            ids = np.asarray(self.tokenizer.encode(text, add_special_tokens=False,
                                                   split_special_tokens=True), np.int64)
            if np.any(ids >= CTRL_LO) or np.any(ids < 0):
                raise ValueError(f"Non-text token in compact transcript: {c.row_key}")
            self._compact_tokens_cache[key] = ids
            if start == 0 and end == c.n:
                self._compact_lengths[index] = len(ids)
            while len(self._compact_tokens_cache) > 128:
                self._compact_tokens_cache.popitem(last=False)
        self._compact_tokens_cache.move_to_end(key)
        return self._compact_tokens_cache[key]

    def compact_length(self, index, start=0, end=None):
        end = self.clips[index].n if end is None else int(end)
        if start == 0 and end == self.clips[index].n and self._compact_lengths[index] >= 0:
            return int(self._compact_lengths[index])
        return len(self.compact_tokens(index, start, end))

    def compact_length_cache(self, path, *, save=False):
        """Optional local planning cache, pinned to source rows and tokenizer."""
        path = Path(path)
        tok_sha = hashlib.sha256((Path(self.tokenizer_path) / "tokenizer.json").read_bytes()).hexdigest()
        if save:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            with tmp.open("wb") as f:
                np.savez_compressed(f, lengths=self._compact_lengths,
                                    source_sha=self.source_plan_sha256, tokenizer_sha=tok_sha,
                                    policy="plain-space-overlap-v1")
            tmp.replace(path)
        elif path.exists():
            with np.load(path, allow_pickle=False) as cached:
                if (str(cached["source_sha"]) != self.source_plan_sha256 or
                    str(cached["tokenizer_sha"]) != tok_sha or
                    str(cached["policy"]) != "plain-space-overlap-v1" or
                    cached["lengths"].shape != self._compact_lengths.shape):
                    raise ValueError(f"Compact token-length cache provenance mismatch: {path}")
                self._compact_lengths[:] = cached["lengths"]

    def micro(self, unit, rank=0, mode="transcript"):
        index, start, end = map(int, unit)
        c, A = self.clips[index], legacy()
        if not 0 <= start < end <= c.n or end - start < 2:
            raise ValueError(f"Invalid clip slice {unit}")
        row = self.row(index)
        m = A.speech_micro_a_empty(row, c.store_row, self.sib, None, self.cfg, rank, c.swap)
        _, off, _, pose = self._shard(c.shard)
        a, b = map(int, off[c.cache_index:c.cache_index + 2])
        m["pdepth"], m["pcb0"], m["plight"] = A.pose_arrays(m, row, pose[a:b].T, self.cfg)
        m.update(kind="pose", row_key=c.row_key, item=[c.store, c.store_row, c.swap])
        if mode == "no-text":
            if start != 0 or end != c.n:
                raise ValueError("The no-text ablation keeps complete original clips")
            return remove_text(m)
        if mode == "original" and start == 0 and end == c.n:
            return m
        if mode == "compact-transcript":
            return transform_micro(m, start, end, prefix_ids=self.compact_tokens(index, start, end))
        if mode not in ("original", "transcript"):
            raise ValueError(mode)
        return transform_micro(m, start, end, transcript=(mode == "transcript"))

    def make_plan(self, world, steps=200, ga=2, budget=7500, mode="transcript", oversize_policy="split", **kwargs):
        return make_plan(self, world, steps, ga, budget, mode, oversize_policy, **kwargs)


def transform_micro(m, start, end, *, transcript=True, prefix_ids=None):
    """Slice the already-timed B lane; reset delays without touching codec codes."""
    oldP, n = int(m["P"]), int(end - start)
    lane = m["ids"][1, oldP + start:oldP + end].copy()
    text = lane if prefix_ids is None else np.asarray(prefix_ids, np.int64)
    prefix = np.concatenate([[TRANSCRIPT_START], text, [TRANSCRIPT_END]]) if transcript else np.empty(0, np.int64)
    P = oldP + len(prefix)
    N = P + n + CD
    ids = np.full((2, N), PAD, np.int64)
    depth = np.full((2, N, DEPTH), -1, np.int64)
    present = np.zeros((2, N), np.float32)
    ids[:, :oldP], depth[:, :oldP], present[:, :oldP] = m["ids"][:, :oldP], m["depth"][:, :oldP], m["present"][:, :oldP]
    ids[1, oldP:P], ids[1, P:P + n], present[1, oldP:] = prefix, lane, 1.0
    dst, src = P + CD + np.arange(n), oldP + CD + np.arange(start, end)
    depth[1, dst] = m["depth"][1, src]
    pdepth = np.full((N, DEPTH), -1, np.int64)
    pdepth[dst] = m["pdepth"][src]
    cb0, pcb0 = np.full(N, -100, np.int64), np.full(N, -100, np.int64)
    cb0[dst], pcb0[dst] = depth[1, dst, 0], pdepth[dst, 0]
    sel = P + np.arange(n - 1, dtype=np.int64)
    y = ids[1, sel + 1].copy()
    out = dict(kind="pose", ids=ids, depth=depth, present=present, pid=np.arange(N, dtype=np.int64), segs=[N],
               P=P, sel=sel, y=y, word=y < CTRL_LO, pad=y == PAD, eob=y == EOB, nw=y == NW,
               n_text=float(len(sel)), cb0=cb0, light=dst.copy(), pdepth=pdepth, pcb0=pcb0, plight=dst.copy(),
               swap=m["swap"], row_key=m["row_key"], item=m["item"], prov=m.get("prov", "{}"),
               source_span=[int(start), int(end)], conditioning_P=oldP,
               transcript_span=[oldP, P] if transcript else [oldP, oldP])
    assert np.all(out["word"] | out["pad"] | out["eob"] | out["nw"])
    return out


def remove_text(m):
    """Remove every lexical/timing token, including words in the voice reference.

    Only system/voice delimiters survive inside the original conditioning block;
    all generation positions are PAD. Audio/pose, position/segment boundaries,
    presence masks and the deterministic voice-reference draw remain unchanged.
    """
    out = dict(m)
    ids = np.full_like(m["ids"], PAD)
    P = int(m["P"])
    structural = np.isin(m["ids"][:, :P], [151644, 151645, 151676, 151677, 151678, 151679])
    ids[:, :P][structural] = m["ids"][:, :P][structural]
    out["ids"] = ids
    out["sel"] = np.empty(0, np.int64)
    out["y"] = np.empty(0, np.int64)
    for key in ("word", "pad", "eob", "nw"):
        out[key] = np.empty(0, bool)
    out.update(n_text=0.0, conditioning_P=P, transcript_span=[P, P], text_policy="no-text-pad")
    return out


def split_spans(n, capacity, policy="split"):
    if n < 2:
        raise ValueError("Generation examples need at least two frames")
    if capacity < 2:
        raise ValueError("Budget cannot fit even a two-frame example")
    if n <= capacity:
        return [(0, int(n))]
    if policy == "drop":
        return []
    if policy != "split":
        raise ValueError(f"Oversized clip ({n} > {capacity}), policy={policy}")
    if capacity == 2 and n % 2:
        raise ValueError("An odd-length clip cannot be split into two-frame chunks")
    sizes = [capacity] * (n // capacity)
    if n % capacity:
        sizes.append(n % capacity)
    if sizes[-1] == 1:
        sizes[-2] -= 1
        sizes[-1] += 1
    ends = np.cumsum([0] + sizes).tolist()
    return list(zip(ends[:-1], ends[1:]))


def compact_spans(data, index, budget, policy="split"):
    """Fit media plus its complete overlapping words, without a one-frame tail."""
    n, P = data.clips[index].n, data.conditioning_length(index)
    def cost(a, b):
        return P + 4 + b - a + data.compact_length(index, a, b)
    if cost(0, n) <= budget:
        return [(0, n)]
    if policy == "drop":
        return []
    if policy != "split":
        raise ValueError(f"Oversized compact transcript: {data.clips[index].row_key}")
    spans, start = [], 0
    while start < n:
        end = min(n, start + budget - P - 4)
        while end - start >= 2:
            overflow = cost(start, end) - budget
            if overflow <= 0:
                if n - end == 1:
                    end -= 1
                    continue
                break
            end -= max(1, overflow)
        if end - start < 2 or cost(start, end) > budget:
            raise ValueError(f"Cannot fit two media frames and overlapping words: {data.clips[index].row_key}")
        spans.append((start, end))
        start = end
    return spans


def pack_micro(ms, budget=7500, dummy=False):
    """FA varlen segment boundaries enforce independent causal examples."""
    if not ms:
        raise ValueError("A packed micro needs at least one example")
    A = legacy()
    out = A.Q.pack(ms, budget=budget)
    offsets = np.cumsum([0] + [len(m["pid"]) for m in ms[:-1]])
    out.update(kind="pose", pdepth=np.concatenate([m["pdepth"] for m in ms]),
               pcb0=np.concatenate([m["pcb0"] for m in ms]),
               plight=np.concatenate([m["plight"] + off for m, off in zip(ms, offsets)]),
               own=np.asarray([[len(m["sel"]), np.count_nonzero(m["cb0"] >= 0), len(m["light"]),
                                np.count_nonzero(m["pcb0"] >= 0), len(m["plight"])] for m in ms], np.float64),
               item=json.dumps([m["item"] for m in ms]))
    sw = {int(m["swap"]) for m in ms}
    out["swap"] = sw.pop() if len(sw) == 1 else -1
    if dummy:
        out.update(dummy=1, own=np.zeros_like(out["own"]))
    return out


def _eval_groups(data, units, budget, mode):
    groups = []
    # Keep the production evaluation cap of 2,048 media frames (long singles
    # alone); this permits the existing bounded decoder evaluator.
    for swap in (0, 1):
        cur, total, frames = [], 0, 0
        for i, (index, start, end) in enumerate(units):
            if data.clips[index].swap != swap:
                continue
            n = end - start
            cost = data.conditioning_length(index) + n + CD + (n + 2 if mode == "transcript" else 0)
            if mode == "compact-transcript":
                cost += data.compact_length(index, start, end) + 2
            if cur and (total + cost > budget or frames + n > 2048):
                groups.append(cur)
                cur, total, frames = [], 0, 0
            cur.append(i)
            total += cost
            frames += n
        if cur:
            groups.append(cur)
    return groups


def make_plan(data, world, steps=200, ga=2, budget=7500, mode="transcript", oversize_policy="split", *, seed=PLAN_SEED,
              compact_length_cache=None, micro_batch_rows=1):
    if mode == "no-text":
        return make_no_text_plan(data, world, steps=steps, seed=seed, micro_batch_rows=micro_batch_rows, ga=ga)
    if mode == "original":
        return make_original_plan(data, world, steps=steps, seed=seed)
    if mode not in ("transcript", "compact-transcript") or min(world, steps, ga, budget) <= 0:
        raise ValueError((mode, world, steps, ga, budget))
    compact = mode == "compact-transcript"
    if compact and compact_length_cache is not None:
        data.compact_length_cache(compact_length_cache)
    units, by_item, oversized = [], {}, []
    for i in data.split_indices["train"]:
        c = data.clips[i]
        cap = (budget - data.conditioning_length(i) - 4) // 2
        spans = compact_spans(data, i, budget, oversize_policy) if compact else split_spans(c.n, cap, oversize_policy)
        if (compact and (len(spans) != 1 or spans[0] != (0, c.n))) or (not compact and c.n > cap):
            oversized.append(i)
        by_item[i] = []
        for start, end in spans:
            by_item[i].append(len(units))
            units.append([i, start, end])
    eligible = [i for i in data.split_indices["train"] if by_item[i]]
    if not eligible:
        raise ValueError("No training clips fit the budget")
    costs = [data.conditioning_length(i) + 2 * (b - a) + 4 for i, a, b in units]
    if compact:
        costs = [data.conditioning_length(i) + b - a + data.compact_length(i, a, b) + 4 for i, a, b in units]
    epoch, cursor, order = -1, 0, []
    def next_unit():
        nonlocal epoch, cursor, order
        if cursor == len(order):
            epoch += 1
            perm = np.random.default_rng(seed + epoch).permutation(len(eligible))
            order = [u for j in perm for u in by_item[eligible[int(j)]]]
            cursor = 0
        u = order[cursor]
        cursor += 1
        return u
    pending = next_unit()
    step_micros, frames_step = [], []
    seen, presentations, max_tokens, consumed_epoch = set(), 0, 0, 0
    for _ in range(steps):
        per = [[[] for _ in range(ga)] for _ in range(world)]
        step_frames = 0
        for g in range(ga):
            for rank in range(world):
                pack, tokens = [], 0
                while not pack or tokens + costs[pending] <= budget:
                    u = pending
                    pack.append(u)
                    tokens += costs[u]
                    step_frames += units[u][2] - units[u][1]
                    seen.add(u)
                    presentations += 1
                    consumed_epoch = epoch
                    pending = next_unit()
                per[rank][g] = pack
                max_tokens = max(max_tokens, tokens)
        step_micros.append(per)
        frames_step.append(step_frames)
    eval_units = []
    for i in data.split_indices["eval"]:
        cap = (budget - data.conditioning_length(i) - 4) // 2
        spans = compact_spans(data, i, budget, oversize_policy) if compact else split_spans(data.clips[i].n, cap, oversize_policy)
        eval_units.extend([i, a, b] for a, b in spans)
    unique_frames = sum(units[u][2] - units[u][1] for u in seen)
    stats = dict(source_train_clips=len(data.split_indices["train"]), source_train_frames=sum(data.clips[i].n for i in data.split_indices["train"]),
                 units=len(units), oversized_clips=len(oversized), oversized_source_frames=sum(data.clips[i].n for i in oversized),
                 unique_clips=len({units[u][0] for u in seen}), unique_units=len(seen), unique_media_frames=unique_frames,
                 presented_units=presentations, presented_media_frames=sum(frames_step), repeated_media_frames=sum(frames_step) - unique_frames,
                 epochs_started=consumed_epoch + 1, max_micro_tokens=max_tokens)
    plan = dict(format="transcript-pose-plan-v1", mode=mode, world=int(world), steps=int(steps), ga=int(ga), budget=int(budget),
                seed=int(seed), oversize_policy=oversize_policy, units=units, items=units, step_micros=step_micros,
                R=[ga] * steps, eval_units=eval_units, eval_items=eval_units,
                eval_micros=_eval_groups(data, eval_units, budget, mode), media_frames_per_step=frames_step,
                coverage=stats, source_plan_sha256=data.source_plan_sha256, data_cfg=data.cfg,
                n_train_items=len(data.split_indices["train"]), n_eval_items=len(data.split_indices["eval"]),
                train_frames=stats["source_train_frames"], eval_frames=sum(data.clips[i].n for i in data.split_indices["eval"]),
                skipped_lt2=[data.skipped_lt2["train"], data.skipped_lt2["eval"]])
    plan["digest"] = plan_digest(plan)
    if compact and compact_length_cache is not None:
        data.compact_length_cache(compact_length_cache, save=True)
    return plan


def make_original_plan(data, world, *, steps=None, seed=PLAN_SEED):
    """Production 64 logical rows/step, same permutation, upper bounds and LPT.

    Distribution changes with world size; membership in each optimizer step does
    not. An optional steps limit truncates this one-epoch plan, never cycles it.
    """
    A = legacy()
    train = data.split_indices["train"]
    order = [train[int(j)] for j in np.random.default_rng(seed).permutation(len(train))]
    units = [[i, 0, data.clips[i].n] for i in order]
    ns = [data.clips[i].n for i in order]
    tk = [A.tok_ub(data.clips[i].n, data.clips[i].row_key, data.cfg) for i in order]
    rows, cur, nf, nt = [], [], 0, 0
    for j, (n, t) in enumerate(zip(ns, tk)):
        if n > 7500 or t > 9502:
            raise ValueError(("Original row cap exceeded", j, n, t))
        if cur and (nf + n > 7500 or nt + t > 9502):
            rows.append(cur)
            cur, nf, nt = [], 0, 0
        cur.append(j)
        nf += n
        nt += t
    if cur:
        rows.append(cur)
    count = math.ceil(len(rows) / 64)
    count = count if steps is None else min(int(steps), count)
    step_micros, rounds, frames_step = [], [], []
    for s in range(count):
        cl = [j for row in rows[s * 64:(s + 1) * 64] for j in row]
        R = max(1, math.ceil(sum(tk[j] for j in cl) / (world * 6000)))
        while True:
            bins, loads = A._lpt([tk[j] for j in cl], world * R)
            if max(loads) <= 9502:
                break
            R += 1
        per = [[[] for _ in range(R)] for _ in range(world)]
        for j, b in enumerate(sorted(range(world * R), key=lambda b: (-loads[b], b))):
            per[j % world][j // world] = sorted(cl[x] for x in bins[b])
        step_micros.append(per)
        rounds.append(R)
        frames_step.append(sum(ns[j] for j in cl))
    ev = [[i, 0, data.clips[i].n] for i in data.split_indices["eval"]]
    # Production uses conservative voice bounds for eval grouping too.
    groups = []
    for sw in (0, 1):
        cur, nf, nt = [], 0, 0
        for j, (i, _, n) in enumerate(ev):
            if data.clips[i].swap != sw:
                continue
            t = A.tok_ub(n, data.clips[i].row_key, data.cfg)
            if cur and (nf + n > 2048 or nt + t > 9502):
                groups.append(cur)
                cur, nf, nt = [], 0, 0
            cur.append(j)
            nf += n
            nt += t
        if cur:
            groups.append(cur)
    plan = dict(format="transcript-pose-plan-v1", mode="original", world=world, steps=count, ga=None, budget=9502, seed=seed,
                units=units, items=units, n_frames=ns, tok_ub=tk, rows=rows, n_rows=len(rows), step_micros=step_micros, R=rounds,
                eval_units=ev, eval_items=ev, eval_micros=groups, media_frames_per_step=frames_step,
                source_plan_sha256=data.source_plan_sha256, data_cfg=data.cfg, n_train_items=len(units), n_eval_items=len(ev),
                train_frames=sum(ns), eval_frames=sum(u[2] for u in ev),
                skipped_lt2=[data.skipped_lt2["train"], data.skipped_lt2["eval"]])
    plan["digest"] = plan_digest(plan)
    return plan


def make_no_text_plan(data, world, *, steps=200, seed=PLAN_SEED, micro_batch_rows=2, ga=1):
    """Redistribute the original 64 logical rows without resampling any clips.

    A logical row holds <=7500 media frames and <=9502 total positions including
    conditioning. MB2 concatenates two such rows in one varlen forward; GA2 runs
    them separately while retaining the same joint eight decoder updates.
    """
    world, micro_batch_rows, ga = map(int, (world, micro_batch_rows, ga))
    if min(world, micro_batch_rows, ga) < 1 or world * micro_batch_rows * ga != 64:
        raise ValueError("no-text requires world * micro_batch_rows * gradient_accumulation == 64")
    plan = make_original_plan(data, world, steps=steps, seed=seed)
    micros = []
    for step in range(plan["steps"]):
        rows = plan["rows"][step * 64:(step + 1) * 64]
        per = [[[] for _ in range(ga)] for _ in range(world)]
        for logical_row, group in enumerate(rows):
            rank = logical_row % world
            round_ = logical_row // (world * micro_batch_rows)
            per[rank][round_].extend(group)
        expected = sorted(u for group in rows for u in group)
        actual = sorted(u for rounds in per for group in rounds for u in group)
        assert actual == expected
        micros.append(per)
    plan.update(mode="no-text", ga=ga, micro_batch_rows=micro_batch_rows, global_batch_rows=64,
                logical_row_media_budget=7500, logical_row_token_budget=9502,
                budget=9502 * micro_batch_rows, step_micros=micros, R=[ga] * plan["steps"],
                text_policy="all generation PAD; lexical and timing tokens removed from conditioning; structural delimiters retained")
    plan["coverage"] = dict(logical_rows=plan["n_rows"], global_rows_per_step=64,
                            presented_media_frames=sum(plan["media_frames_per_step"]),
                            unique_media_frames=sum(plan["media_frames_per_step"]), epochs_started=1,
                            max_micro_token_upper_bound=max(sum(plan["tok_ub"][u] for u in group)
                                                           for per in micros for rounds in per for group in rounds))
    plan["digest"] = plan_digest(plan)
    return plan


def plan_digest(plan):
    payload = [plan["mode"], plan["world"], plan["budget"], plan["source_plan_sha256"], plan.get("data_cfg"),
               plan["units"], plan["step_micros"], plan["eval_units"], plan["eval_micros"]]
    if plan["mode"] == "no-text":
        payload.extend([plan["micro_batch_rows"], plan["ga"], plan["global_batch_rows"], plan.get("benchmark_stress", False)])
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def stress_no_text_plan(plan):
    """Benchmark only: full 7500-frame source rows, padded to token memory cap."""
    assert plan["mode"] == "no-text"
    rows = [r for r in plan["rows"] if sum(plan["n_frames"][u] for u in r) == 7500]
    if not rows:
        raise ValueError("No complete 7500-media-frame source row for stress benchmark")
    row = max(rows, key=lambda r: sum(plan["tok_ub"][u] for u in r))
    group = row * plan["micro_batch_rows"]
    plan["step_micros"] = [[[list(group) for _ in range(plan["ga"])] for _ in range(plan["world"])] for _ in range(plan["steps"])]
    plan["benchmark_stress"] = True
    plan["stress_source_row"] = row
    plan["media_frames_per_step"] = [64 * 7500] * plan["steps"]
    plan["coverage"] = dict(benchmark_only=True, padded_tokens_per_micro=plan["budget"],
                            media_frames_per_rank=7500 * plan["micro_batch_rows"] * plan["ga"],
                            repeated_source_row=row)
    plan["digest"] = plan_digest(plan)
    return plan


def pad_packed_micro(b, budget):
    """Add an independent, unsupervised PAD segment for a bounded memory probe."""
    extra = int(budget) - len(b["pid"])
    if extra < 0:
        raise ValueError("Stress micro exceeds its token cap")
    if not extra:
        return b
    for key, fill in (("ids", PAD), ("depth", -1), ("present", 0)):
        shape = list(b[key].shape)
        shape[1] = extra
        b[key] = np.concatenate([b[key], np.full(shape, fill, b[key].dtype)], axis=1)
    for key, fill in (("pdepth", -1), ("cb0", -100), ("pcb0", -100)):
        shape = list(b[key].shape)
        shape[0] = extra
        b[key] = np.concatenate([b[key], np.full(shape, fill, b[key].dtype)])
    b["pid"] = np.concatenate([b["pid"], np.arange(extra, dtype=np.int64)])
    b["segs"].append(extra)
    b["P"] = np.concatenate([b["P"], [extra]])
    return b


class StepDataset:
    """DataLoader item -> rank's list of packed micros for one optimizer step."""
    def __init__(self, data, plan, rank, start=0, singles_at=-1):
        if not 0 <= rank < plan["world"]:
            raise ValueError(rank)
        self.data, self.plan, self.rank = data, plan, int(rank)
        self.start, self.singles_at = int(start), int(singles_at)

    def __len__(self):
        return self.plan["steps"] - self.start

    def __getitem__(self, index):
        step = self.start + int(index)
        out = []
        for round_, group in enumerate(self.plan["step_micros"][step][self.rank]):
            # Empty original-mode LPT bins carry a zero-weight real example.
            us = group or [0]
            ms = [self.data.micro(self.plan["units"][u], self.rank, self.plan["mode"]) for u in us]
            b = pack_micro(ms, self.plan["budget"], dummy=not group)
            if self.plan.get("benchmark_stress"):
                b = pad_packed_micro(b, self.plan["budget"])
            b["step"] = step
            if step == self.singles_at and round_ == 0:
                b["singles"] = [pack_micro([m], self.plan["budget"]) for m in ms]
            out.append(b)
        return out


def eval_micros(data, plan, rank, world=None):
    world = plan["world"] if world is None else int(world)
    for group in plan["eval_micros"][rank::world]:
        yield pack_micro([data.micro(plan["eval_units"][u], rank, plan["mode"]) for u in group], plan["budget"])
