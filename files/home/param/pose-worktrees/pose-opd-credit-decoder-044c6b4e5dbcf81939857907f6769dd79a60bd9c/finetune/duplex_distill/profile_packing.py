"""Pack one or two prepared conversations with independent attention segments.

Encode each waveform separately before preparing either row. pack_prepared
retains supplied weights; a backward benchmark must first apply_global_weights
using counts summed over all local rows and all ranks. This implements bounded
one/two-example microbatches, not arbitrary packing into a fixed token budget.
"""
from __future__ import annotations

import torch


_AXIS1 = {"ids", "depth", "present", "pid", "vlat"}
_INDICES = {"sel", "light", "plight", "vpos"}
_AXIS0 = {"y", "word", "pad", "eob", "nw", "cb0", "pdepth", "pcb0",
          "own", "P", "text_mask", "speech_mask", "pose_mask",
          "text_weight", "speech_weight", "pose_weight"}
_META = {"kind", "segs", "dummy", "vin"}
_BASE = {"ids", "depth", "present", "pid", "sel", "light", "y", "word",
         "pad", "eob", "nw", "cb0", "own", "P", "text_mask", "speech_mask",
         "text_weight", "speech_weight", "kind", "segs", "dummy"}
_POSE = {"pdepth", "pcb0", "plight", "pose_mask", "pose_weight"}
_VOX = {"vlat", "vpos", "vin"}


def apply_global_weights(rows, global_counts, world):
    """Return row copies normalized over all examples in the distributed step.

    Counts are text, speech cb0, pose cb0, speech depth, pose depth. The caller
    supplies the sum of both rows' all-reduced counts, not a per-row count or a
    mean. WORLD compensates for FSDP's gradient averaging. Depth normalization
    remains with the decoder caller, using these same combined counts.
    """
    if global_counts.shape != (5,) or int(world) != world or world < 1:
        raise ValueError("Expected five combined global counts and positive world size")
    result = []
    for row in rows:
        item = dict(row)
        for name, index in (("text", 0), ("speech", 1), ("pose", 2)):
            mask_key = name + "_mask"
            if mask_key not in row:
                if name == "pose":
                    continue  # Audio-only teacher has no pose tensors.
                raise ValueError(f"Missing prepared mask: {mask_key}")
            mask = row[mask_key]
            if mask.device != global_counts.device:
                raise ValueError("Masks and global counts must be on the same device")
            item[name + "_weight"] = (torch.zeros_like(mask, dtype=torch.float32)
                                      if row.get("dummy", False) else
                                      mask.float() * (world / global_counts[index].clamp_min(1)))
        result.append(item)
    return result


def pack_prepared(rows):
    """Return ``(packed, info)`` for same-role outputs of runtime.prepare.

    Every prefix and delayed tail stays within its original attention segment.
    ``vin`` becomes [rows, 2] with global injection offsets; forward uses vpos.
    No input is modified. Unknown fields are errors, never silently dropped.
    """
    if len(rows) not in (1, 2):
        raise ValueError("Supported microbatches contain one or two rows")
    keys = set(rows[0])
    unknown = keys - (_AXIS1 | _INDICES | _AXIS0 | _META)
    if unknown:
        raise ValueError(f"Unknown prepared fields: {sorted(unknown)}")
    if not _BASE <= keys:
        raise ValueError(f"Missing prepared fields: {sorted(_BASE - keys)}")
    for optional in (_POSE, _VOX):
        if keys & optional and not optional <= keys:
            raise ValueError(f"Incomplete prepared field group: {sorted(optional - keys)}")
    lengths, offsets, offset = [], [], 0
    original_rows, rows = rows, []
    for source in original_rows:
        row = dict(source)
        if set(row) != keys:
            raise ValueError("Both rows must have exactly the same prepared schema")
        if row["kind"] != original_rows[0]["kind"]:
            raise ValueError("Cannot mix model roles/kinds in one packed forward")
        t = row["ids"].shape[1]
        if row["segs"] != [t] or row["pid"].shape != (1, t):
            raise ValueError("Inputs must be standalone conversations")
        if row["P"].shape != (1,) or row["own"].shape[0] != 1:
            raise ValueError("Expected one prefix/count row per conversation")
        for key in keys - {"kind", "segs", "dummy"}:
            value = row[key]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Prepared field {key} must be a tensor")
            reference = original_rows[0][key]
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError(f"Inconsistent device/dtype for {key}")
        if _VOX <= keys:
            if row["vin"].shape != (2,):
                raise ValueError("Expected standalone vin=(prefix, frames)")
            n = row["vpos"].numel()
            if row["vlat"].ndim != 3 or row["vlat"].shape[:2] != (1, 4 * n):
                raise ValueError("Latents must contain four encoder frames per vpos")
        lengths.append(t)
        offsets.append(offset)
        offset += t
        if row["dummy"]:
            # A packed batch with a real row has dummy=False overall, so the
            # depth loss cannot rely on that flag to skip the padded row. Its
            # selected depth indices must be removed before concatenation.
            for key in ("light", "plight"):
                if key in row:
                    row[key] = row[key][:0]
            for key in ("text_mask", "speech_mask", "pose_mask", "text_weight",
                        "speech_weight", "pose_weight", "own"):
                if key in row:
                    row[key] = torch.zeros_like(row[key])
            for key in ("y", "cb0", "pcb0"):
                if key in row:
                    row[key] = torch.full_like(row[key], -100)
        rows.append(row)
    packed = {"kind": rows[0]["kind"], "segs": lengths.copy(),
              "dummy": all(row["dummy"] for row in rows)}
    for key in keys & _AXIS1:
        packed[key] = torch.cat([row[key] for row in rows], dim=1)
    for key in keys & _AXIS0:
        packed[key] = torch.cat([row[key] for row in rows], dim=0)
    for key in keys & _INDICES:
        packed[key] = torch.cat([row[key] + start for row, start in zip(rows, offsets)])
    if "vin" in keys:
        packed["vin"] = torch.stack([
            row["vin"] + row["vin"].new_tensor([start, 0])
            for row, start in zip(rows, offsets)])
    info = {"row_count": len(rows), "timeline_frames": offset,
            "real_rows": sum(not row["dummy"] for row in rows),
            "dummy_rows": sum(bool(row["dummy"]) for row in rows),
            "row_timeline_frames": lengths, "segment_lengths": lengths.copy(),
            "row_offsets": offsets, "text_positions": packed["sel"].numel(),
            "row_text_positions": [row["sel"].numel() for row in rows],
            "speech_depth_frames": packed["light"].numel(),
            "pose_depth_frames": packed["plight"].numel() if "plight" in packed else 0,
            "audio_frames": packed["vpos"].numel() if "vpos" in packed else 0}
    return packed, info
