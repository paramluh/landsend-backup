"""Required Rosie conditioning shared by every live session and the startup smoke.

Assets are loaded and validated before GPU model construction. The configuration
contains Mimi codes and provenance; there is deliberately no empty fallback.
"""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


class FixedConditioning:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        cfg = json.loads(self.config_path.read_text())
        if cfg.get("speaker") != "Rosie Stamp":
            raise ValueError("Fixed conditioning must identify speaker Rosie Stamp")
        self.preset_id = cfg.get("preset_id", "rosie-reference")
        if self.preset_id != "rosie-reference":
            raise ValueError("The fixed Rosie preset ID must be rosie-reference")
        self.prompt_mode = cfg.get("prompt_mode", "voice_clip_only")
        if self.prompt_mode != "voice_clip_only" or cfg.get("persona_path") or cfg.get("persona"):
            raise ValueError("Rosie conditioning is audio-only; persona text is not allowed")

        def asset(name):
            p = Path(cfg[name])
            return p if p.is_absolute() else self.config_path.parent / p

        code_path = asset("codes_path")
        self.codes = np.load(code_path, allow_pickle=False)
        if (self.codes.ndim != 2 or self.codes.shape[0] != 32
                or not 38 <= self.codes.shape[1] <= 125
                or self.codes.dtype.kind not in "iu"
                or self.codes.min() < 0 or self.codes.max() >= 2048):
            raise ValueError("Rosie reference must be integer Mimi codes [32, 38..125] in 0..2047")
        self.frames = int(self.codes.shape[1])
        self.source = cfg.get("source")
        if not isinstance(self.source, dict) or not self.source.get("row_id"):
            raise ValueError("Rosie reference requires source row_id provenance")
        if int(self.source.get("frames", self.frames)) != self.frames:
            raise ValueError("Rosie reference provenance does not match the code length")
        self.codes_sha256 = sha256(code_path.read_bytes()).hexdigest()
        if cfg.get("codes_sha256") is not None and cfg["codes_sha256"] != self.codes_sha256:
            raise ValueError("Rosie conditioning codes_sha256 mismatch")
        self.clip_id = "preset:" + self.preset_id

    def start_fields(self):
        return {"preset": self.preset_id, "persona": "", "clip": self.clip_id}

    def build_prefix(self, prompts):
        clip = prompts.upload_clip(self.codes)
        pre, info = prompts.build_prefix("", clip)
        if info.get("clip_frames") != self.frames or sum(info.get("persona_tokens", [])):
            raise ValueError("Rosie prefix must contain the complete reference and no persona tokens")
        info.update(
            preset=self.preset_id,
            persona_edited=False,
            clip={"src": "fixed_rosie_reference", "lane": False, "speaker": "Rosie Stamp",
                  "seconds": self.frames / 12.5, "src_key": self.source["row_id"],
                  "source": self.source, "codes_sha256": self.codes_sha256},
            prompt_mode=self.prompt_mode,
            route="Rosie voice reference only; no persona text",
            conditioning_locked=True,
        )
        return pre, info, ""

    def public(self):
        label = f"Rosie Stamp's voice ({self.frames / 12.5:.2f} s)"
        return {
            "presets": [{"id": self.preset_id, "label": "Rosie Stamp", "name": "Rosie Stamp",
                         "persona": "", "row": None, "group": "Rosie", "default": True}],
            "clips": [{"id": self.clip_id, "label": label, "row": None}],
            "default_clip": {self.preset_id: self.clip_id},
            "default_preset": self.preset_id,
            "prompt_default": self.prompt_mode,
            "conditioning_locked": True,
            "conditioning": {**self.start_fields(), "speaker": "Rosie Stamp",
                             "codes_sha256": self.codes_sha256},
            "upload_ok": False,
        }
