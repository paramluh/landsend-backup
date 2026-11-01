"""Explicit, serializable configuration for two-teacher on-policy credit.

Paths identify inputs, not a pinned historical export recipe.  All distribution,
window and weighting choices are recorded in checkpoints by the trainer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    # JSON run configs still supply all checkpoint paths explicitly, even for
    # the selected student/pose teacher and historical audio fixture below.
    student: str = "/home/param/models/imported-weights/ppm-aloe-poseduplex-raw-final/export-424"
    pose_teacher: str = "/home/param/models/imported-weights/ppm-aloe-monoplex-pose-raw-bandfix-355/weights/export-355"
    audio_teacher: str = "/home/param/models/imported-weights/ppm-aloe-samsep-vox17127-final/export-249-dcp"
    audio_teacher_gain_policy: str = "require_fp32"
    pose_teacher_cfg_scale: float = 1.0
    pose_teacher_silence_path: str = ""
    pose_teacher_silence_sha256: str = ""
    teacher_depth_cache: str = "none"
    teacher_batch_per_rank: int | None = None
    pose_cfg_batch: bool = False
    fixed_sequence_frames: int | None = None
    data_root: str = "/home/param/datasets/aloe-duplex"
    data_variant: str = "samsep"
    offload_audio_front: bool = False
    offload_optimizer_during_rollout: bool = False
    prefetch_rows: bool = False
    rollout_samples_per_update: int = 0
    lookback_frames: int = 8
    window_mode: str = "previous_frames"
    prefill_frames: int = 25
    frames_per_second: float = 12.5
    speech_delay: int = 2
    pose_delay: int = 2
    modality_weights: dict[str, float] = field(default_factory=lambda: {"text": 1 / 3, "audio": 1 / 3, "pose": 1 / 3})
    codebook_weights: tuple[float, ...] = (0.5,) + (0.5 / 31,) * 31
    reduction: str = "row_mean"
    conditioning_mode: str = "audio"
    missing_clip_policy: str = "empty"
    exclude_invalid_pose_frame0: bool = True
    require_pose_prefill: bool = True
    sampler: str = "categorical"
    temperature: float = 1.0
    rollout_mode: str = "continuous_row"
    partner_checkpoint: str = ""
    partner_replay_backend: str = "sequential"
    partner_prefill_min_frames: int = 38
    partner_prefill_max_frames: int = 62
    conversation_prefix_frames: int | None = None
    exclude_short_rows: bool = False
    max_row_frames: int | None = None
    seed: int = 20261031
    learning_rate: float = 3e-6
    learning_rate_schedule: str = "constant"
    learning_rate_warmup_steps: int | None = None
    epochs: int = 1
    decoder_chunk_frames: int = 128
    decoder_gradient_sync: str = "per_chunk"
    checkpoint_depth_decoder: bool = True
    text_chunk_frames: int = 32
    physical_microbatch: int = 1
    effective_batch_per_rank: int = 1
    balance_rows_by_length: bool = False
    rollout_batch_per_rank: int | None = None
    frozen_teacher_bf16: bool = True
    actor_backbone_cut: int | None = None
    partner_actor_backbone_cut: int | None = None
    actor_sampler_backend: str = "multinomial"
    actor_depth_cache_positions: int = 64
    actor_attention_precision: str = "fp32"
    actor_linear_backend: str = "native"
    checkpoint_interval_seconds: float | None = 7200.0
    checkpoint_interval_steps: int | None = None
    keep_latest_optimizer_state: bool = False
    data_config: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> "TrainingConfig":
        if type(self.rollout_samples_per_update) is not int or self.rollout_samples_per_update < 0:
            raise ValueError("rollout_samples_per_update must be a nonnegative integer")
        for name in ("student", "pose_teacher", "audio_teacher", "data_root"):
            if not str(getattr(self, name)):
                raise ValueError(f"{name} must identify an input path")
        if self.data_variant not in ("samsep", "raw"):
            raise ValueError("data_variant must explicitly select samsep or raw")
        if type(self.offload_audio_front) is not bool:
            raise ValueError("offload_audio_front must be a boolean")
        if type(self.offload_optimizer_during_rollout) is not bool:
            raise ValueError("offload_optimizer_during_rollout must be a boolean")
        if type(self.prefetch_rows) is not bool:
            raise ValueError("prefetch_rows must be a boolean")
        if type(self.balance_rows_by_length) is not bool:
            raise ValueError("balance_rows_by_length must be a boolean")
        if self.audio_teacher_gain_policy not in ("require_fp32", "exported"):
            raise ValueError("audio_teacher_gain_policy must be require_fp32 or exported")
        if self.teacher_depth_cache not in ("none", "cpu"):
            raise ValueError("teacher_depth_cache must be none or cpu")
        if type(self.pose_cfg_batch) is not bool:
            raise ValueError("pose_cfg_batch must be a boolean")
        if self.fixed_sequence_frames is not None and (type(self.fixed_sequence_frames) is not int
                or self.fixed_sequence_frames <= self.prefill_frames):
            raise ValueError("fixed_sequence_frames must exceed prefill_frames or be None")
        if self.decoder_gradient_sync not in ("per_chunk", "per_replay"):
            raise ValueError("decoder_gradient_sync must be per_chunk or per_replay")
        if type(self.checkpoint_depth_decoder) is not bool:
            raise ValueError("checkpoint_depth_decoder must be a boolean")
        if self.actor_attention_precision not in ("native_bf16", "fp32"):
            raise ValueError("actor_attention_precision must be native_bf16 or fp32")
        if self.actor_linear_backend not in ("native", "tuned_m16"):
            raise ValueError("actor_linear_backend must be native or tuned_m16")
        if not math.isfinite(self.pose_teacher_cfg_scale) or self.pose_teacher_cfg_scale < 1:
            raise ValueError("pose_teacher_cfg_scale must be finite and at least one")
        if self.pose_teacher_cfg_scale != 1:
            if not self.pose_teacher_silence_path:
                raise ValueError("Pose CFG requires encoded digital-silence tokens")
            if len(self.pose_teacher_silence_sha256) != 64 or any(
                    c not in '0123456789abcdef' for c in self.pose_teacher_silence_sha256):
                raise ValueError("Pose CFG requires the silence file SHA256")
        if type(self.exclude_invalid_pose_frame0) is not bool:
            raise ValueError("exclude_invalid_pose_frame0 must be a boolean")
        for name in ("lookback_frames", "prefill_frames", "speech_delay", "pose_delay"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.window_mode not in ("previous_frames", "including_current"):
            raise ValueError("window_mode must be previous_frames or including_current")
        if self.window_mode == "including_current" and self.lookback_frames == 0:
            raise ValueError("An inclusive frame window needs at least one frame")
        if self.frames_per_second != 12.5 or self.speech_delay != 2 or self.pose_delay != 2:
            raise ValueError("The production PPM data uses 12.5 Hz and two-frame speech/pose delays")
        if set(self.modality_weights) != {"text", "audio", "pose"}:
            raise ValueError("Specify all three modality weights")
        weights = (*self.modality_weights.values(), *self.codebook_weights)
        if any(not math.isfinite(float(x)) or float(x) < 0 for x in weights):
            raise ValueError("Loss weights must be finite and nonnegative")
        if not any(self.modality_weights.values()) or len(self.codebook_weights) != 32 or not any(self.codebook_weights):
            raise ValueError("Require a nonzero modality weight and 32 nonzero-total codebook weights")
        if self.reduction not in ("row_mean", "position_mean"):
            raise ValueError("reduction must be row_mean or position_mean")
        if self.conditioning_mode not in ("audio", "voice", "voice_pose"):
            raise ValueError("conditioning_mode must be audio, voice or voice_pose")
        if self.missing_clip_policy not in ("error", "skip", "empty"):
            raise ValueError("missing_clip_policy must be error, skip or empty")
        if self.sampler != "categorical" or self.temperature != 1.0:
            raise ValueError("Only the untruncated temperature-one categorical sampling policy is implemented")
        if self.rollout_mode not in ("continuous_row", "teacher_partner"):
            raise ValueError("rollout_mode must be continuous_row or teacher_partner")
        if not isinstance(self.partner_checkpoint, str):
            raise ValueError("partner_checkpoint must be a path string or empty for audio_teacher")
        if self.partner_replay_backend not in ("sequential", "block"):
            raise ValueError("partner_replay_backend must be sequential or block")
        for name in ("partner_prefill_min_frames", "partner_prefill_max_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.partner_prefill_min_frames > self.partner_prefill_max_frames:
            raise ValueError("Partner prefill bounds must be ordered")
        if self.rollout_mode == "teacher_partner":
            if self.data_variant != "raw" or self.conditioning_mode != "audio":
                raise ValueError("Teacher-partner rollouts require raw audio and audio-only reference clips")
            for name in ("conversation_prefix_frames", "fixed_sequence_frames", "max_row_frames"):
                value = getattr(self, name)
                if value is not None and value <= self.partner_prefill_max_frames:
                    raise ValueError(f"{name} must exceed the maximum partner prefill")
        if self.conversation_prefix_frames is not None and (type(self.conversation_prefix_frames) is not int
                or self.conversation_prefix_frames <= self.prefill_frames):
            raise ValueError("conversation_prefix_frames must exceed prefill_frames or be None")
        if type(self.exclude_short_rows) is not bool:
            raise ValueError("exclude_short_rows must be boolean")
        if self.exclude_short_rows and self.conversation_prefix_frames is None:
            raise ValueError("exclude_short_rows requires conversation_prefix_frames")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if self.learning_rate_schedule not in ("constant", "amu_opd", "source_linear"):
            raise ValueError("learning_rate_schedule must be constant, amu_opd or source_linear")
        if self.learning_rate_warmup_steps is not None and (type(self.learning_rate_warmup_steps) is not int
                or self.learning_rate_warmup_steps < 0):
            raise ValueError("learning_rate_warmup_steps must be a nonnegative integer or None")
        if self.learning_rate_schedule == "constant" and self.learning_rate_warmup_steps is not None:
            raise ValueError("A constant learning rate has no warmup steps")
        for name in ("epochs", "decoder_chunk_frames", "text_chunk_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("physical_microbatch", "effective_batch_per_rank"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.effective_batch_per_rank % self.physical_microbatch:
            raise ValueError("effective_batch_per_rank must be divisible by physical_microbatch")
        if self.teacher_batch_per_rank is not None:
            if type(self.teacher_batch_per_rank) is not int or self.teacher_batch_per_rank <= 0:
                raise ValueError("teacher_batch_per_rank must be a positive integer or None")
            if self.effective_batch_per_rank % self.teacher_batch_per_rank:
                raise ValueError("teacher_batch_per_rank must divide effective_batch_per_rank")
        if self.rollout_batch_per_rank is not None:
            if type(self.rollout_batch_per_rank) is not int or self.rollout_batch_per_rank <= 0:
                raise ValueError("rollout_batch_per_rank must be a positive integer or None")
            if self.effective_batch_per_rank % self.rollout_batch_per_rank:
                raise ValueError("rollout_batch_per_rank must divide effective_batch_per_rank")
        if type(self.frozen_teacher_bf16) is not bool:
            raise ValueError("frozen_teacher_bf16 must be a boolean")
        if self.actor_backbone_cut is not None and (type(self.actor_backbone_cut) is not int or self.actor_backbone_cut < 1):
            raise ValueError("actor_backbone_cut must be a positive layer count or null")
        if self.partner_actor_backbone_cut is not None and (type(self.partner_actor_backbone_cut) is not int
                or self.partner_actor_backbone_cut < 1):
            raise ValueError("partner_actor_backbone_cut must be a positive layer count or null")
        if self.actor_sampler_backend not in ("multinomial", "inverse_cdf"):
            raise ValueError("actor_sampler_backend must be multinomial or inverse_cdf")
        if type(self.actor_depth_cache_positions) is not int or self.actor_depth_cache_positions not in (32, 64):
            raise ValueError("actor_depth_cache_positions must be 32 or 64")
        if self.checkpoint_interval_seconds is not None and (type(self.checkpoint_interval_seconds) not in (int, float)
                or not math.isfinite(self.checkpoint_interval_seconds) or self.checkpoint_interval_seconds <= 0):
            raise ValueError("checkpoint_interval_seconds must be finite and positive or None")
        if self.checkpoint_interval_steps is not None and (type(self.checkpoint_interval_steps) is not int
                or self.checkpoint_interval_steps <= 0):
            raise ValueError("checkpoint_interval_steps must be a positive integer or None")
        if self.checkpoint_interval_seconds is None and self.checkpoint_interval_steps is None:
            raise ValueError("At least one checkpoint interval must be configured")
        if type(self.keep_latest_optimizer_state) is not bool:
            raise ValueError("keep_latest_optimizer_state must be a boolean")
        if self.max_row_frames is not None and (type(self.max_row_frames) is not int or self.max_row_frames <= self.prefill_frames):
            raise ValueError("max_row_frames must exceed prefill_frames; overlong rows are rejected, never cropped")
        if "crop_conv" in self.data_config and self.data_config["crop_conv"]:
            raise ValueError("Full-row OPD must not crop conversations")
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.validate())

    @property
    def resolved_partner_checkpoint(self) -> str:
        return self.partner_checkpoint or self.audio_teacher

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainingConfig":
        value = dict(value)
        missing = sorted({"student", "pose_teacher", "audio_teacher", "data_variant"} - value.keys())
        if missing:
            raise ValueError(f"JSON run configuration must explicitly select {', '.join(missing)}")
        if "codebook_weights" in value:
            value["codebook_weights"] = tuple(value["codebook_weights"])
        return cls(**value).validate()

    def checkpoint_data_config(self) -> dict[str, Any]:
        """Optional checkpoint data defaults, without requiring any run/export ID."""
        import json
        path = Path(self.student) / "train_cfg.json"
        if not path.is_file():
            path = Path(self.student) / "meta.json"
        if path.is_file():
            source = json.loads(path.read_text())
            cfg = source.get("cfg", source)
            base = dict(cfg.get("data", {}))
        else:
            base = {}
        base.update(self.data_config)
        base.update(crop_conv=False, light_frac=1.0, pose_light_frac=1.0,
                    pose_delay=self.pose_delay, voice_frac=1.0, persona_sides="none")
        return base
