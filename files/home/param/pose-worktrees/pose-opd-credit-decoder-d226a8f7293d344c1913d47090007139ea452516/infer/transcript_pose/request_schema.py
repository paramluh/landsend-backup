"""Shared, standard-library-only contract for the prepared-clip inference UI.

No request can supply a filesystem path, checkpoint, transcript or command.
Both the HTTP API and the GPU worker must call normalize_request independently.
"""
from __future__ import annotations

import json
import math

RECIPES = ("xabi-original", "aligned-media", "compact-all")
TEXT_MODES = ("forced", "native")
NATIVE_TEXT_RECIPES = ("xabi-original", "compact-all")
CLIP_IDS = tuple(f"clip-{i:02d}" for i in range(1, 9))
MAX_SEED = 2**31 - 1
FIELDS = frozenset(("recipe", "clip_id", "duration_seconds", "seed", "video_seed",
                    "audio_temp", "pose_temp", "audio_topk", "pose_topk",
                    "text_mode", "text_temp", "text_topk"))


def normalize_request(payload: dict) -> dict:
    """Validate a request; legacy requests default to forced framewise text."""
    if not isinstance(payload, dict):
        raise ValueError("Request must be a JSON object")
    if payload.keys() - FIELDS:
        raise ValueError("Unknown request field(s): " + ", ".join(sorted(map(str, payload.keys() - FIELDS))))
    for name, choices in (("recipe", RECIPES), ("clip_id", CLIP_IDS)):
        if payload.get(name) not in choices:
            raise ValueError(f"{name} must be one of: {', '.join(choices)}")
    result = dict(recipe=payload["recipe"], clip_id=payload["clip_id"])
    result["text_mode"] = payload.get("text_mode", "forced")
    if result["text_mode"] not in TEXT_MODES:
        raise ValueError("text_mode must be forced or native")
    if result["text_mode"] == "native" and result["recipe"] not in NATIVE_TEXT_RECIPES:
        raise ValueError("aligned-media requires forced framewise text; native text is available for compact-all and xabi-original")
    integers = {
        "duration_seconds": (60, 4, 60),
        "seed": (20261028 + CLIP_IDS.index(result["clip_id"]), 0, MAX_SEED),
        "video_seed": (2026, 0, MAX_SEED),
        "audio_topk": (0, 0, 2048), "pose_topk": (0, 0, 2048), "text_topk": (0, 0, 256),
    }
    for name, (default, low, high) in integers.items():
        value = payload.get(name, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        result[name] = value
    for name in ("audio_temp", "pose_temp", "text_temp"):
        value = payload.get(name, 1.0)
        if type(value) not in (int, float) or not 0 <= value <= 3 or not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number in [0, 3]")
        result[name] = float(value)
    return result


def request_options() -> dict:
    """Public UI choices; server-side validation remains authoritative."""
    return {
        "recipes": list(RECIPES), "clip_ids": list(CLIP_IDS),
        "text_modes": list(TEXT_MODES), "default_text_mode": "forced",
        "native_text_recipes": list(NATIVE_TEXT_RECIPES),
        "duration_seconds": {"min": 4, "max": 60, "default": 60},
        "seed": {"min": 0, "max": MAX_SEED,
                 "defaults": {clip: 20261028 + i for i, clip in enumerate(CLIP_IDS)}},
        "video_seed": {"min": 0, "max": MAX_SEED, "default": 2026},
        "audio_temp": {"min": 0, "max": 3, "default": 1},
        "pose_temp": {"min": 0, "max": 3, "default": 1},
        "text_temp": {"min": 0, "max": 3, "default": 1},
        "audio_topk": {"min": 0, "max": 2048, "default": 0},
        "pose_topk": {"min": 0, "max": 2048, "default": 0},
        "text_topk": {"min": 0, "max": 256, "default": 0},
        "voice_reference_seconds": 8,
        "topk_zero_means": "Unfiltered", "temperature_zero_means": "Greedy",
    }


def load_request(text: str) -> dict:
    """Reject duplicate keys and non-standard JSON numbers before validation."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate request field: {key}")
            result[key] = value
        return result

    def invalid_number(value):
        raise ValueError(f"Invalid JSON number: {value}")

    return normalize_request(json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_number))


def media_frames(duration_seconds: int) -> int:
    """Even 12.5 Hz frames sufficient for 25 Hz video: decode yields 2*n-3."""
    if type(duration_seconds) is not int or not 4 <= duration_seconds <= 60:
        raise ValueError("duration_seconds must be an integer in [4, 60]")
    return 2 * ((25 * duration_seconds + 6) // 4)
