"""Full-row rollout inputs and replay views with explicit action provenance.

Only the externally supplied A waveform, an available B conditioning clip and
the initial ground-truth prefill enter a rollout seed. Future ground-truth B
tokens and labels are erased before the seed leaves this module.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import threading

import numpy as np

from .config import TrainingConfig

PRODUCTION = Path(__file__).resolve().parents[1] / "duplex_distill" / "production"
_MODULE_LOCK = threading.RLock()
CODEBOOKS, VOCAB_SIZE = 32, 2048
OFFSETS = np.arange(CODEBOOKS, dtype=np.int64) * VOCAB_SIZE
ROW_IDENTITY_COLUMNS = ("row_key", "n_frames", "chunk_index", "frame_start", "split",
                        "user_speaker_dir", "assistant_speaker_dir", "src_conversation")


class UnavailableRow(ValueError):
    """A row cannot satisfy the configured input/prefill requirements."""


class ExcludedShortRow(UnavailableRow):
    """The explicit minimum-duration selection rule excludes this source row."""


def conditioning_segment(clip, mode):
    """Select audio-only reference content without dropping boundary controls.

    The native prefix builder still writes its system/SOS/EOS markers. Setting
    blank_lane suppresses just the reference's spoken words and NEW_WORD marks;
    the independent two-second conversation prefill is unchanged.
    """
    if mode not in ("audio", "voice", "voice_pose"):
        raise ValueError(f"Unsupported conditioning mode: {mode}")
    segment = {"kind": "speech", "track": 1, **clip}
    if mode == "audio":
        segment["blank_lane"] = True
    return segment


def _prefix(raw):
    return int(np.asarray(raw["P"]).reshape(-1)[0])


def sampling_masks(prefix, frames, prefill_frames=25, speech_delay=2, pose_delay=2):
    """Masks use model positions; prefill is specified in physical row frames."""
    if min(prefix, frames, prefill_frames, speech_delay, pose_delay) < 0:
        raise ValueError("Negative timeline size")
    if frames <= prefill_frames:
        raise UnavailableRow("Row has no frames after the requested prefill")
    length = prefix + frames + max(speech_delay, pose_delay)
    positions = np.arange(length)
    masks = {}
    for name, delay in (("text", 0), ("audio", speech_delay), ("pose", pose_delay)):
        masks[name] = (positions >= prefix + prefill_frames + delay) & (positions < prefix + frames + delay)
    return masks


def row_prefill_frames(config, item, rank=0, *, update=0, slot=0):
    """Reproducible independent row draws, including across epochs/resumes.

    The update cursor, rank and slot distinguish repeated source orientations.
    Whole 80 ms frames keep the requested 3--5 second interval inside its
    bounds: the default inclusive integer range is 38--62 (3.04--4.96 s).
    """
    if config.rollout_mode != "teacher_partner":
        return config.prefill_frames
    entropy = [config.seed, update, rank, slot, *map(int, item)]
    if any(type(value) is not int or value < 0 for value in entropy):
        raise ValueError("Prefill seed coordinates must be nonnegative integers")
    rng = np.random.default_rng(np.random.SeedSequence(entropy))
    return int(rng.integers(config.partner_prefill_min_frames, config.partner_prefill_max_frames + 1))


def output_gain_metadata(entry, raw_gains, *, sha256):
    """Undo the speaker's ORIGINAL raw Mimi encoding gain, never SAM gain."""
    flac = entry["flac"]
    recorded = raw_gains[flac]
    if recorded.get("a_spk") != entry.get("a_spk") or recorded.get("kind") != entry.get("kind"):
        raise ValueError(f"Original output gain speaker/source mismatch: {flac}")
    # A mimidec source FLAC is already normalized: its index gain_mimi_db is
    # zero. The gain originally applied before encoding lives separately.
    if recorded["kind"] == "mimidec":
        if not recorded.get("codes_checked"):
            raise ValueError(f"Unverified Mimi fallback gain: {flac}")
        gain_db = float(recorded["recorded_lufs_gain_db"])
    else:
        gain_db = float(recorded["gain_mimi_db"])
    inverse = float(10.0 ** (-gain_db / 20.0))
    if not np.isfinite(gain_db) or not np.isfinite(inverse) or inverse <= 0:
        raise ValueError(f"Invalid original output gain: {flac}")
    return {"flac": flac, "speaker": recorded["a_spk"], "gain_mimi_db": gain_db,
            "inverse_gain": inverse, "raw_gain_sha256": sha256,
            "source": "original/provenance/RAWGAIN.json"}


def prepare_rollout_seed(raw, *, pad_id, prefill_frames=25, speech_delay=2, pose_delay=2,
                         require_pose_prefill=True, missing_clip_policy="error"):
    """Copy only causal inputs; erase all future B data and discard source labels."""
    prefix = _prefix(raw)
    n = int(raw.get("conversation_frames", np.asarray(raw["vin"])[1]))
    if missing_clip_policy not in ("error", "skip", "empty"):
        raise ValueError("missing_clip_policy must be error, skip or empty")
    has_clip = bool(raw.get("voice_clip", 0))
    if not has_clip and missing_clip_policy != "empty":
        raise UnavailableRow("The configured policy requires an available, nonempty conditioning clip")
    masks = sampling_masks(prefix, n, prefill_frames, speech_delay, pose_delay)
    length = len(masks["text"])
    required = {"ids": (2, length), "depth": (2, length, CODEBOOKS),
                "present": (2, length), "pdepth": (length, CODEBOOKS)}
    for key, shape in required.items():
        if np.asarray(raw[key]).shape != shape:
            raise ValueError(f"{key} has shape {np.asarray(raw[key]).shape}, expected {shape}")
    if has_clip and not np.any(np.asarray(raw["depth"])[1, :prefix, 0] >= 0):
        raise UnavailableRow("Conditioning metadata says clip present but prefix has no audio codes")
    if not has_clip:
        if (prefix != 5 or not np.array_equal(np.asarray(raw["ids"])[1, :prefix], [151644, 8948, 198, 151645, 198])
                or np.any(np.asarray(raw["depth"])[1, :prefix] >= 0)
                or np.any(np.asarray(raw["pdepth"])[:prefix] >= 0)):
            raise ValueError("Unavailable clips require the native empty system prefix with no reference audio or pose")
    for name, values, delay in (("audio", raw["depth"][1], speech_delay), ("pose", raw["pdepth"], pose_delay)):
        if name == "pose" and not require_pose_prefill:
            continue
        before = np.asarray(values)[prefix + delay:prefix + delay + prefill_frames] - OFFSETS
        if np.any(before < 0) or np.any(before >= VOCAB_SIZE):
            raise UnavailableRow(f"Missing or invalid ground-truth {name} in the required prefill")
    keys = ("ids", "depth", "present", "pdepth", "pid", "wav", "vin")
    seed = {key: np.array(raw[key], copy=True) for key in keys if key in raw}
    for key in ("row_key", "item", "aref", "conditioning", "pose_src", "prov"):
        if key in raw:
            seed[key] = copy.deepcopy(raw[key])
    seed.update(kind="pose", P=np.asarray([prefix], np.int64), segs=[length],
                conversation_frames=n, prefill_frames=int(prefill_frames),
                speech_delay=int(speech_delay), pose_delay=int(pose_delay), pad_id=int(pad_id),
                voice_clip=int(has_clip), voice_drawn=int(raw.get("voice_drawn", 1)), pose_ok=1, sampled_masks=masks)
    seed["ids"][1, prefix + prefill_frames:] = pad_id
    seed["depth"][1, masks["audio"]] = -1
    seed["pdepth"][masks["pose"]] = -1
    # Native input is audio-only: do not leak the dataset A transcript/codes.
    seed["ids"][0] = pad_id
    seed["depth"][0] = -1
    seed["present"][0] = 0
    return seed


def teacher_views(student, pad_id):
    """One generated B history; empty-A monoplex and same-waveform duplex."""
    pose = dict(student)
    for key in ("ids", "depth", "present"):
        pose[key] = np.array(student[key], copy=True)
    pose["ids"][0] = pad_id
    pose["depth"][0] = -1
    pose["present"][0] = 0
    for key in ("vin", "wav", "aref"):
        pose.pop(key, None)
    audio = dict(student)
    for key in ("pdepth", "pcb0", "plight", "pose_ok", "pose_src"):
        audio.pop(key, None)
    audio["kind"] = "speech"
    if "own" in audio:
        audio["own"] = np.asarray(audio["own"])[:, :3]
    return {"student": student, "pose_teacher": pose, "audio_teacher": audio}


def prefix_seed(seed, frames):
    """Keep the deterministic first frames of an already scrubbed row.

    Conditioning and the physical prefill remain exact. Short rows remain
    short; no random start, tail windows, or additional epoch examples arise.
    """
    if type(frames) is not int or frames <= int(seed['prefill_frames']):
        raise ValueError('Conversation prefix must exceed the forced prefill')
    current = int(seed['conversation_frames'])
    original = int(seed.get('original_conversation_frames', current))
    n = min(frames, current)
    prefix = int(np.asarray(seed['P']).reshape(-1)[0])
    delay = max(int(seed['speech_delay']), int(seed['pose_delay']))
    total = prefix + n + delay
    out = dict(seed)
    for key in ('ids', 'depth', 'present'):
        out[key] = np.array(seed[key][:, :total], copy=True)
    out['pdepth'] = np.array(seed['pdepth'][:total], copy=True)
    out['pid'] = np.arange(total, dtype=np.int64)
    out['vin'] = np.asarray([prefix, n], np.int64)
    out['segs'] = [total]
    out['conversation_frames'] = n
    out['original_conversation_frames'] = original
    if 'wav' in seed:
        minimum = int(seed['prefill_frames']) if seed.get('rollout_mode') == 'teacher_partner' else n
        if len(seed['wav']) < minimum * 1920:
            raise ValueError('A waveform does not cover the selected conversation prefix')
        out['wav'] = np.array(seed['wav'][:n * 1920], copy=True)
    if 'aref' in seed:
        reference = json.loads(seed['aref'])
        reference.update(n=n, original_n=original)
        out['aref'] = json.dumps(reference)
    out['sampled_masks'] = sampling_masks(prefix, n, int(seed['prefill_frames']),
                                         int(seed['speech_delay']), int(seed['pose_delay']))
    if seed.get('kind') == 'speech':
        out['sampled_masks']['pose'].fill(False)
    if 'partner' in seed:
        out['partner'] = prefix_seed(seed['partner'], n)
    return out


def replay_from_rollout(seed, text, audio, pose, *, word_end=151643, new_word_id=151671,
                        eob_id=151645, text_vocab_size=151682):
    """Build scored production micros from complete generated timeline arrays.

    Media use production global code offsets. Forced prefix/prefill/tail values
    must remain exact; all sampled media codebooks are scored, without thinning.
    """
    text, audio, pose = (np.asarray(x, np.int64) for x in (text, audio, pose))
    original = (seed["ids"][1], seed["depth"][1], seed["pdepth"])
    for name, values, reference in zip(("text", "audio", "pose"), (text, audio, pose), original):
        mask = seed["sampled_masks"][name]
        if values.shape != reference.shape:
            raise ValueError(f"Generated {name} has incorrect shape")
        if not np.array_equal(values[~mask], reference[~mask]):
            raise ValueError(f"Generated {name} changed forced prefix/prefill/tail values")
        local = values[mask] if name == "text" else values[mask] - OFFSETS
        maximum = text_vocab_size if name == "text" else VOCAB_SIZE
        if np.any(local < 0) or np.any(local >= maximum):
            raise ValueError(f"Generated {name} includes absent/out-of-vocabulary sampled actions")
    out = dict(seed)
    out["ids"] = np.array(seed["ids"], copy=True)
    out["depth"] = np.array(seed["depth"], copy=True)
    out["ids"][1] = text
    out["depth"][1] = audio
    out["pdepth"] = np.array(pose, copy=True)
    tm, am, pm = (seed["sampled_masks"][key] for key in ("text", "audio", "pose"))
    target = np.flatnonzero(tm)
    y = text[target]
    out.update(text_positions=target, sel=target - 1, y=y,
               word=(y < word_end), pad=(y == seed["pad_id"]), nw=(y == new_word_id), eob=(y == eob_id),
               cb0=np.where(am, audio[:, 0], -100), pcb0=np.where(pm, pose[:, 0], -100),
               light=np.flatnonzero(am), plight=np.flatnonzero(pm), n_text=float(tm.sum()),
               own=np.asarray([[tm.sum(), am.sum(), am.sum(), pm.sum(), pm.sum()]], np.float64))
    return teacher_views(out, seed["pad_id"])


class _ProductionArena:
    """Isolate the archived loader's flat module names and environment settings.

    All imports, including imports executed inside loader methods, occur under
    this serialized context. Existing module objects and env values are restored
    after every call; no global want_voice monkeypatch is needed.
    """
    def __init__(self, data_root):
        self.data_root = Path(data_root).resolve()
        self.modules = {}
        self.names = {p.stem for root in (PRODUCTION, PRODUCTION / "lib") for p in root.glob("*.py")}

    @contextmanager
    def enter(self):
        with _MODULE_LOCK:
            old_env, old_path = dict(os.environ), list(sys.path)
            old_modules = {name: sys.modules[name] for name in self.names if name in sys.modules}
            for name in self.names:
                sys.modules.pop(name, None)
            sys.modules.update(self.modules)
            try:
                from finetune.duplex_distill.recipe import configure_environment
                configure_environment(self.data_root)
                for path in (PRODUCTION / "lib", PRODUCTION):
                    sys.path.insert(0, str(path))
                module = importlib.import_module("data_aloe_posedup")
                yield module
            finally:
                self.modules = {name: sys.modules[name] for name in self.names if name in sys.modules}
                for name in self.names:
                    sys.modules.pop(name, None)
                sys.modules.update(old_modules)
                sys.path[:] = old_path
                os.environ.clear()
                os.environ.update(old_env)


def invert_row_map(mapping, new_count, old_count):
    """Validate and invert the published kept-old -> new row bijection."""
    pairs = [(int(old), int(new)) for old, new in mapping.items()]
    if (len(pairs) != new_count or len({old for old, _ in pairs}) != new_count
            or sorted(new for _, new in pairs) != list(range(new_count))
            or any(old < 0 or old >= old_count for old, _ in pairs)):
        raise ValueError("Raw row map must bijectively cover all selected rows with in-range source rows")
    inverse = np.empty(new_count, np.int64)
    for old, new in pairs:
        inverse[new] = old
    return inverse


class _RawMappedRows:
    """Preserve selected row metadata and replace both speakers' Mimi codes.

    The same wrapper backs the sibling index, so reference clips use the chosen
    raw policy as well as conversation prefill. Row identities and valid-frame
    masks are verified; mappings never join rows by a guessed name or order.
    """
    def __init__(self, module, selected_node, selected_rows, raw_node, mapping):
        import pyarrow.parquet as pq
        self.rows, self.raw_node = selected_rows, raw_node
        old_count = int(mapping["old_rows"])
        if old_count != len(raw_node["length"]):
            raise ValueError("Raw store row count differs from the row map")
        self.inverse = invert_row_map(mapping["new_of_old"], len(selected_rows), old_count)
        if len(selected_node["length"]) != len(self.inverse):
            raise ValueError("Selected store row count differs from the row map")
        columns = ["row", *ROW_IDENTITY_COLUMNS]
        selected = pq.read_table(Path(selected_node["flat_root"]) / "rows.parquet", columns=columns).to_pydict()
        original = pq.read_table(Path(raw_node["flat_root"]) / "rows.parquet", columns=columns).to_pydict()
        if selected["row"] != list(range(len(self.inverse))) or original["row"] != list(range(old_count)):
            raise ValueError("Store metadata row IDs are not contiguous source indices")
        indices = self.inverse.tolist()
        for name in ROW_IDENTITY_COLUMNS:
            if [original[name][i] for i in indices] != selected[name]:
                raise ValueError(f"Mapped raw/selected row identity differs: {name}")
        if not np.array_equal(np.asarray(raw_node["length"])[self.inverse], selected_node["length"]):
            raise ValueError("Mapped raw/selected row lengths differ")
        self.raw_rows = module.D.R3.Rows(raw_node)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        selected = self.rows[int(index)]
        original = self.raw_rows[int(self.inverse[int(index)])]
        out = dict(selected)
        for side in ("user", "assistant"):
            key = "codes_" + side
            before, after = np.asarray(selected[key]), np.asarray(original[key])
            if before.shape != after.shape or before.dtype != after.dtype or before.shape[0] != CODEBOOKS:
                raise ValueError(f"Mapped raw/selected {side} code shapes or dtypes differ")
            if not np.array_equal(before >= 0, after >= 0):
                raise ValueError(f"Mapped raw/selected {side} valid-frame masks differ")
            if np.any(after[after >= 0] >= VOCAB_SIZE):
                raise ValueError(f"Raw {side} contains out-of-vocabulary codes")
            out[key] = original[key]
        return out


class ProductionRows:
    """Full-row loader that uses every available clip and an explicit missing policy."""
    def __init__(self, config: TrainingConfig):
        self.config = config.validate()
        self.arena = _ProductionArena(config.data_root)
        self.audit = []
        self._stores = None
        self._raw_open = {}
        self._raw_nodes = {}
        self._original_gains = None
        if config.rollout_mode == "teacher_partner":
            gain_path = Path(config.data_root) / "original/provenance/RAWGAIN.json"
            gain_bytes = gain_path.read_bytes()
            gain_doc = json.loads(gain_bytes)
            if gain_doc.get("format") != "aloe-v2vox-rawgain-v1":
                raise ValueError("Unsupported original raw audio gain metadata")
            self._original_gains = gain_doc["rows"]
            self._original_gain_sha256 = hashlib.sha256(gain_bytes).hexdigest()
        with self.arena.enter() as module:
            self.cfg = dict(module.aloe_cfg(), **config.checkpoint_data_config())
            self.cfg.update(crop_conv=False, light_frac=1.0, pose_light_frac=1.0)
            self.pad_id = int(module.D.PAD_ID)
            self.metadata = {
                "variant": config.data_variant, "selected_store_digests": list(module.A2.DIGESTS),
                "pose_index_sha256": hashlib.sha256(Path(module.POSE_INDEX).read_bytes()).hexdigest(),
                "audio_index_sha256": hashlib.sha256(Path(module.INDEX).read_bytes()).hexdigest(),
                "raw_gain_sha256": hashlib.sha256(Path(module.RAWGAIN).read_bytes()).hexdigest(),
                "a_waveform": "same raw waveform and gain in both variants",
            }
            if self._original_gains is not None:
                self.metadata["partner_output_gain_sha256"] = self._original_gain_sha256
                self.metadata["partner_checkpoint"] = config.resolved_partner_checkpoint
                self.metadata["partner_prefill_frames"] = [config.partner_prefill_min_frames,
                                                            config.partner_prefill_max_frames]
            drop_bytes = (Path(__file__).parent / "assets/DROP_LANES.json").read_bytes()
            drops = json.loads(drop_bytes)
            if (drops.get("format") != "pose-opd-frame0-drop-v1" or drops.get("dataset") != "posedup"
                    or drops.get("store_digests") != list(module.A2.DIGESTS)):
                raise ValueError("Source frame-zero quality exclusions describe another dataset")
            self._frame0_drops = {}
            for entry in drops["items"]:
                if entry["side"] not in ("user", "assistant"):
                    raise ValueError("Invalid speaker side in frame-zero quality exclusions")
                item = (int(entry["store"]), int(entry["row"]), int(entry["side"] == "user"))
                if item in self._frame0_drops:
                    raise ValueError("Duplicate item in frame-zero quality exclusions")
                self._frame0_drops[item] = entry
            self.metadata["pose_frame0_filter"] = {
                "enabled": config.exclude_invalid_pose_frame0,
                "manifest_sha256": hashlib.sha256(drop_bytes).hexdigest(),
                "source_annotation_release": drops["source_annotation_release"],
                "source_annotation_sha256": drops["source_annotation_sha256"],
                "source_drop_count": len(self._frame0_drops),
            }
            if config.data_variant == "raw":
                root = Path(config.data_root)
                map_path = root / "pose/hf/provenance/row_map.original.json"
                map_bytes = map_path.read_bytes()
                self._raw_map = json.loads(map_bytes)
                if self._raw_map.get("format") != "aloe-ft-32-samsep-row-map-v1":
                    raise ValueError("Unsupported raw-to-selected row map format")
                if [self._raw_map["new_digest"][f"m1n{k}"] for k in range(4)] != list(module.A2.DIGESTS):
                    raise ValueError("Row map describes different selected stores")
                digests = []
                for k in range(4):
                    node = module.D.R3.load_node(str(root / f"original/stores/m1n{k}"))
                    if node["digest"] != self._raw_map["old_digest"][f"m1n{k}"]:
                        raise ValueError(f"Raw store {k} digest differs from the row map")
                    self._raw_nodes[k] = node
                    digests.append(node["digest"])
                self.metadata.update(raw_store_digests=digests, raw_row_map=str(map_path),
                                     raw_row_map_sha256=hashlib.sha256(map_bytes).hexdigest())

    def _store(self, module):
        if self._stores is None:
            self._stores = module.Stores()
        return self._stores

    def _open(self, module, k):
        node, rows, siblings = self._store(module).open(k)
        if self.config.data_variant == "samsep":
            return node, rows, siblings
        if k not in self._raw_open:
            raw_rows = _RawMappedRows(module, node, rows, self._raw_nodes[k], self._raw_map["map"][f"m1n{k}"])
            self._raw_open[k] = (node, raw_rows, module.D.SPR.SibIndex(node, raw_rows))
        return self._raw_open[k]

    def _source(self, module, item, *, load_pose=True):
        k, index, swap = map(int, item)
        _, store, siblings = self._open(module, k)
        row = store[index]
        n = int(row["n_frames"])
        selected_n = min(n, self.config.conversation_prefix_frames or n)
        if self.config.exclude_short_rows and n < self.config.conversation_prefix_frames:
            raise ExcludedShortRow(f"Source row has {n} frames; minimum required is {self.config.conversation_prefix_frames}")
        if int(row["chunk_index"]) != 0:
            raise UnavailableRow("Pose-duplex index supports first-chunk dataset rows")
        required_prefill = (self.config.partner_prefill_max_frames if self.config.rollout_mode == "teacher_partner"
                            else self.config.prefill_frames)
        if n <= required_prefill:
            raise UnavailableRow("No generated frames remain after prefill")
        if self.config.max_row_frames is not None and selected_n > self.config.max_row_frames:
            raise UnavailableRow(f"Selected conversation has {selected_n} frames (source {n}); configured capacity is {self.config.max_row_frames}")
        clip = module.D.voice_clips(row, siblings, self.cfg)[1 ^ swap]
        if clip is None and self.config.missing_clip_policy != "empty":
            raise UnavailableRow("No supported later-sibling B conditioning clip is available")
        entry = module.pose_entry(item)
        if not entry["ok"] and (self.config.require_pose_prefill or self.config.conditioning_mode == "voice_pose"):
            raise UnavailableRow("Row lacks B pose needed for mandatory pose prefill/conditioning")
        if self.config.exclude_invalid_pose_frame0 and (k, index, swap) in self._frame0_drops:
            dropped = self._frame0_drops[(k, index, swap)]
            if dropped["row_key"] != row["row_key"] or dropped["speaker"] != entry["spk"]:
                raise ValueError("Source frame-zero exclusion has a different row/speaker identity")
            raise UnavailableRow("Source frame-zero pose normalization is invalid (source quality exclusion)")
        pose = module.read_pose(item, row)[0] if load_pose else None
        if clip is not None and self.config.conditioning_mode == "voice_pose":
            src = clip.get("src_row", row)
            side = "user" if swap else "assistant"
            source_side = clip.get("src_side", side)
            if src[f"{source_side}_speaker_dir"] != row[f"{side}_speaker_dir"]:
                raise UnavailableRow("Conditioning clip speaker differs from the indexed pose speaker")
            if int(src["frame_start"]) + int(clip["w0"]) + int(clip["L"]) > int(entry["T"]):
                raise UnavailableRow("Paired conditioning pose does not cover the complete audio clip")
        return row, clip, pose, entry

    def items(self, split="train"):
        if split not in ("train", "eval"):
            raise ValueError("split must be train or eval")
        result = []
        self.audit = []
        self._item_shapes = {}
        conditioning_counts = {"split": split, "with_clip": 0, "without_clip": 0}
        with self.arena.enter() as module:
            candidates = [(k, row, side) for k, row in module.splits()[split] for side in (0, 1)]
            for item in candidates:
                try:
                    row, clip, _, _ = self._source(module, item, load_pose=False)
                except UnavailableRow as exc:
                    if self.config.missing_clip_policy == "error" and not isinstance(exc, ExcludedShortRow):
                        raise UnavailableRow(f"{item}: {exc}") from exc
                    self.audit.append({"item": list(item), "reason": str(exc)})
                else:
                    result.append(item)
                    # Native single-reference prefix: three system tokens,
                    # SOS + L frames + two media-delay slots + EOS, then two
                    # closing tokens. Empty references use five tokens.
                    prefix = 5 if clip is None else int(clip['L']) + 7 + self.config.speech_delay
                    source_n = int(row['n_frames'])
                    selected_n = min(source_n, self.config.conversation_prefix_frames or source_n)
                    self._item_shapes[tuple(item)] = {
                        'item': list(item), 'conversation_frames': selected_n,
                        'original_conversation_frames': source_n,
                        'prefix_frames': prefix,
                        'total_frames': prefix + selected_n + self.config.speech_delay}
                    conditioning_counts["with_clip" if clip is not None else "without_clip"] += 1
        self.metadata["conditioning_counts"] = conditioning_counts
        self.metadata['conversation_selection'] = {
            'prefix_frames': self.config.conversation_prefix_frames,
            'exclude_short_rows': self.config.exclude_short_rows,
            'source_frames': sum(value['original_conversation_frames'] for value in self._item_shapes.values()),
            'selected_frames': sum(value['conversation_frames'] for value in self._item_shapes.values()),
            'selection': 'first_frames' if self.config.conversation_prefix_frames is not None else 'complete_row',
            'one_example_per_source_orientation': True}
        if not result:
            raise UnavailableRow("No rows meet the configured conditioning and prefill requirements")
        order = np.random.default_rng(self.config.seed).permutation(len(result))
        return [result[i] for i in order]

    def frame_lengths(self, items):
        """Read only row metadata for balancing already selected orientations."""
        if all(tuple(item) in getattr(self, '_item_shapes', {}) for item in items):
            return {tuple(item): self._item_shapes[tuple(item)]['conversation_frames'] for item in items}
        result = {}
        with self.arena.enter() as module:
            for item in items:
                key = tuple(map(int, item))
                _, rows, _ = self._open(module, key[0])
                n = int(rows[key[1]]['n_frames'])
                result[key] = min(n, self.config.conversation_prefix_frames or n)
        return result

    def shape_specs(self, items):
        """Exact metadata dimensions of the selected single-reference rows.

        ``items`` must come from the most recent eligibility audit. Benchmark
        callers additionally compare these bounds to the loaded native prefix.
        """
        return [dict(self._item_shapes[tuple(item)]) for item in items]

    def _partner_seed(self, module, row, item, rank, prefill):
        """Build the opposite speech lane without requiring its pose export."""
        k, index, swap = map(int, item)
        partner_swap = 1 - swap
        _, _, siblings = self._open(module, k)
        clip = module.D.voice_clips(row, siblings, self.cfg)[1 ^ partner_swap]
        segment = (conditioning_segment(clip, "audio") if clip is not None
                   else {"kind": "text", "track": 1, "ids": []})
        prefix = module.D.SPR.build_prefix(row, [segment], {"new_word": self.cfg["new_word"]})
        p, n = int(prefix["P"]), int(row["n_frames"])
        if p > int(self.cfg["sp_cap"]):
            raise UnavailableRow("Partner conditioning clip exceeds configured prefix capacity")
        dc = {"new_word": self.cfg["new_word"], "eob_gap_s": self.cfg["eob_gap_s"],
              "wpid": self.cfg["wpid"], "light_seed": self.cfg["light_seed"],
              "row_frames": p + n, "frame_light": True, "no_crop": True}
        arrays = module.D.DS.row_arrays(row, dc, int(rank), index, "ppm", 1.0,
                                       sp=prefix, swap=partner_swap)
        length = len(arrays["pid"])
        raw = {key: arrays[key] for key in ("ids", "depth", "present", "pid")}
        raw.update(P=p, segs=[length], row_key=row["row_key"], item=[k, index, partner_swap],
                   voice_clip=int(clip is not None), voice_drawn=1,
                   vin=np.asarray([p, n]), conversation_frames=n,
                   pdepth=np.full((length, CODEBOOKS), -1, dtype=np.int64),
                   conditioning={"mode": "audio", "available": clip is not None,
                       "missing_clip_policy": "empty", "prefix_frames": p,
                       "clip_frames": int(clip["L"]) if clip is not None else 0,
                       "source_row": clip["src_key"] if clip is not None else None,
                       "source_start": int(clip["w0"]) if clip is not None else None,
                       "reference_text": False, "reference_pose": False, "data_variant": "raw"})
        seed = prepare_rollout_seed(raw, pad_id=self.pad_id, prefill_frames=prefill,
            speech_delay=self.config.speech_delay, pose_delay=self.config.pose_delay,
            require_pose_prefill=False, missing_clip_policy="empty")
        if self.config.conversation_prefix_frames is not None:
            seed = prefix_seed(seed, self.config.conversation_prefix_frames)
        seed.update(kind="speech", pose_ok=0, rollout_mode="teacher_partner")
        seed["sampled_masks"]["pose"].fill(False)
        return seed

    def _attach_partner_audio(self, module, row, item, seed, partner, *, load_audio):
        """Attach only real prefix PCM; generated histories are filled by the actor."""
        k, index, swap = map(int, item)
        for destination, orientation in ((seed, swap), (partner, 1 - swap)):
            aside, input_entry = module.a_entry((k, index, orientation), row)
            _, output_entry = module.a_entry((k, index, 1 - orientation), row)
            output_gain = output_gain_metadata(output_entry, self._original_gains,
                                               sha256=self._original_gain_sha256)
            input_gain = output_gain_metadata(input_entry, self._original_gains,
                                              sha256=self._original_gain_sha256)
            recorded = self._original_gains[input_entry["flac"]]
            gain = float(recorded["gain_raw_db"])
            expected = 0.0 if input_entry["kind"] == "gcs" else -input_gain["gain_mimi_db"]
            if not np.isfinite(gain) or abs(gain - expected) > 1e-9:
                raise ValueError("Original input PCM gain disagrees with raw-audio convention")
            prefill = int(destination["prefill_frames"])
            destination["output_audio_gain"] = output_gain
            destination["rollout_mode"] = "teacher_partner"
            destination["aref"] = json.dumps({"row_key": row["row_key"], "side": aside,
                "n": destination["conversation_frames"], "flac": input_entry["flac"],
                "gain_db": gain, "frame0": int(input_entry["frame0"]),
                "prefill_frames": prefill, "source_usage": "prefill_only",
                "generated_after_prefill": True, "partner_checkpoint": self.config.resolved_partner_checkpoint})
            if load_audio:
                rc = importlib.import_module("rowcache_vox")
                wave = rc.row_audio({"flac": str(Path(module.FLAC_ROOT) / input_entry["flac"]),
                    "frame0": int(input_entry["audio_frame0"]), "n_frames": prefill,
                    "row_key": row["row_key"]}, gain)
                if wave.shape != (prefill * 1920,) or not np.isfinite(wave).all():
                    raise ValueError("Raw audio does not cover the complete conversation prefill")
                destination["wav"] = wave
        seed["partner"] = partner

    def load(self, item, rank=0, *, load_audio=True, update=0, slot=0):
        with self.arena.enter() as module:
            row, clip, pose, pose_entry = self._source(module, item)
            prefill = row_prefill_frames(self.config, item, rank, update=update, slot=slot)
            k, index, swap = map(int, item)
            d, cfg = module.D, self.cfg
            segment = (conditioning_segment(clip, self.config.conditioning_mode) if clip is not None
                       else {"kind": "text", "track": 1, "ids": []})
            prefix = d.SPR.build_prefix(row, [segment], {"new_word": cfg["new_word"]})
            p, n = int(prefix["P"]), int(row["n_frames"])
            if p > int(cfg["sp_cap"]):
                raise UnavailableRow("Conditioning clip exceeds configured prefix capacity")
            dc = {"new_word": cfg["new_word"], "eob_gap_s": cfg["eob_gap_s"], "wpid": cfg["wpid"],
                  "light_seed": cfg["light_seed"], "row_frames": p + n, "frame_light": True, "no_crop": True}
            arrays = d.DS.row_arrays(row, dc, int(rank), index, "ppm", 1.0, sp=prefix, swap=swap)
            sel, y, word, pad, eob, nw, cb0, light = d._b_only(arrays)
            length = len(arrays["pid"])
            if length != p + n + self.config.speech_delay:
                raise ValueError("Source loader truncated or padded the conversation")
            raw = {key: arrays[key] for key in ("ids", "depth", "present", "pid")}
            raw.update(P=p, segs=[length], swap=swap, sel=sel, y=y, word=word, pad=pad, eob=eob, nw=nw,
                       cb0=cb0, light=light, row_key=row["row_key"], item=list(item),
                       voice_clip=int(clip is not None), voice_drawn=1, vin=np.asarray([p, n]), conversation_frames=n)
            raw["pdepth"], raw["pcb0"], raw["plight"] = module.pose_arrays(raw, row, pose, cfg)
            if clip is not None and self.config.conditioning_mode == "voice_pose":
                src = clip.get("src_row", row)
                side = "user" if swap else "assistant"
                source_side = clip.get("src_side", side)
                if src[f"{source_side}_speaker_dir"] != row[f"{side}_speaker_dir"]:
                    raise UnavailableRow("Conditioning clip speaker differs from the indexed pose speaker")
                start = int(src["frame_start"]) + int(clip["w0"])
                count = int(clip["L"])
                paired = np.asarray(pose[:, start:start + count]).T.astype(np.int64)
                positions = np.flatnonzero(prefix["depth"][1, :, 0] >= 0)
                if (paired.shape != (count, CODEBOOKS) or len(positions) != count
                        or np.any(paired < 0) or np.any(paired >= VOCAB_SIZE)):
                    raise UnavailableRow("Paired conditioning pose does not cover the complete audio clip")
                raw["pdepth"][positions] = paired + OFFSETS
            aside, entry = module.a_entry(item, row)
            gain = module.a_gain(entry)
            raw["aref"] = json.dumps({"row_key": row["row_key"], "side": aside, "n": n,
                                      "flac": entry["flac"], "gain_db": gain, "frame0": int(entry["frame0"])})
            if load_audio and self.config.rollout_mode != "teacher_partner":
                rc = importlib.import_module("rowcache_vox")
                audio_n = min(n, self.config.conversation_prefix_frames or n)
                raw["wav"] = rc.row_audio({"flac": str(Path(module.FLAC_ROOT) / entry["flac"]),
                    "frame0": int(entry["audio_frame0"]), "n_frames": audio_n, "row_key": row["row_key"]}, gain)
                if raw["wav"].shape != (audio_n * 1920,):
                    raise ValueError("A waveform does not cover the selected row prefix")
            raw["pose_src"] = pose_entry.get("path", "missing")
            raw["conditioning"] = {"mode": self.config.conditioning_mode, "probability": 1.0,
                "available": clip is not None, "missing_clip_policy": self.config.missing_clip_policy,
                "prefix_frames": p, "clip_frames": int(clip["L"]) if clip is not None else 0,
                "source_row": clip["src_key"] if clip is not None else None,
                "source_start": int(clip["w0"]) if clip is not None else None,
                "source_chunk": clip.get("src_chunk") if clip is not None else None,
                "reference_text": clip is not None and not bool(segment.get("blank_lane", False)),
                "reference_pose": clip is not None and self.config.conditioning_mode == "voice_pose",
                "data_variant": self.config.data_variant,
                "pose_teacher_a": "native_empty", "audio_teacher_a": "same_as_student"}
            seed = prepare_rollout_seed(raw, pad_id=self.pad_id, prefill_frames=prefill,
                speech_delay=self.config.speech_delay, pose_delay=self.config.pose_delay,
                require_pose_prefill=self.config.require_pose_prefill, missing_clip_policy=self.config.missing_clip_policy)
            if self.config.conversation_prefix_frames is not None:
                seed = prefix_seed(seed, self.config.conversation_prefix_frames)
            if self.config.rollout_mode == "teacher_partner":
                partner = self._partner_seed(module, row, item, rank, prefill)
                self._attach_partner_audio(module, row, item, seed, partner, load_audio=load_audio)
            return seed
