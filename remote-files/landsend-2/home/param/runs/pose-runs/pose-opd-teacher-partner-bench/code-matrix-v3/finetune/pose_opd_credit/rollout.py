"""Cached student rollouts with explicit sampling provenance.

The numpy orchestration is CPU-importable. ``make_fastppm_class`` specializes
the existing cached inference engine without changing its shared source. Its
input hook injects the student's own Voxtral-adapter output in both prefill and
incremental backbone calls. Sampling is the untruncated, temperature-one
categorical policy on executable text IDs and every audio/pose codebook.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import time
from types import MethodType
from typing import Any

import numpy as np


FPS = 12.5
CODE_DELAY = 2
CODEBOOKS = 32
CODEBOOK_SIZE = 2048
PAD = 151670
EXECUTABLE_VOCAB_SIZE = 151682

# Fixed H100 M16 launch shapes measured against the native kernel on32 distinct
# BF16 matrices per shape. The kernel body and arithmetic remain unchanged.
_TUNED_M16 = {
    (4096, 1024): (16, 128, 2, 4),
    (2048, 1024): (16, 128, 2, 4),
    (6144, 1024): (16, 128, 2, 4),
    (1024, 2048): (16, 256, 4, 3),
    (1024, 3072): (16, 256, 4, 3),
}


def tuned_m16_linear(native, fallback, x, weight):
    """Change only known M16 native GEMV launch configurations, per actor."""
    import torch
    rows = x.numel() // x.shape[-1]
    parameters = _TUNED_M16.get(tuple(weight.shape)) if rows == 16 else None
    if (parameters is None or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or not weight.is_contiguous()):
        return fallback(x, weight)
    bn, bk, warps, stages = parameters
    outputs, inputs = weight.shape
    flat = x.reshape(rows, inputs).contiguous()
    result = torch.empty(rows, outputs, dtype=torch.bfloat16, device=x.device)
    with torch.cuda.device(x.device):
        # Bypass only the one-config autotune wrapper. The exact same JIT
        # function runs, with no environment or shared-module mutation.
        native.LK._gemv_k.fn[((outputs + bn - 1) // bn,)](
            flat, weight, result, rows, outputs, inputs, BM=16, BN=bn, BK=bk,
            num_warps=warps, num_stages=stages)
    return result.view(*x.shape[:-1], outputs)

def inverse_cdf_sample(probability, uniform):
    """Draw from existing FP32 weights using float64 cumulative sums/uniforms.

    Scaling by the accumulated total accepts multinomial's unnormalized
    weights. A right search skips zero mass even when the uniform is zero.
    """
    import torch
    if probability.ndim != 2 or uniform.shape != probability.shape[:1]:
        raise ValueError('Categorical weights/uniforms must be [rows,vocab] and [rows]')
    cdf = probability.cumsum(-1, dtype=torch.float64)
    threshold = (uniform.to(torch.float64) * cdf[:, -1])[:, None].contiguous()
    return torch.searchsorted(cdf, threshold, right=True).squeeze(-1).clamp_max(probability.shape[-1] - 1)


def resize_depth_cache(decoder, positions):
    """Reduce unused depth capacity before capture; cb31 reads position31."""
    if type(positions) is not int or positions not in (32, 64):
        raise ValueError('Depth cache capacity must be 32 or 64 positions')
    if len(decoder.K) != len(decoder.V) or not decoder.K:
        raise ValueError('Depth caches must have paired nonempty K/V layers')
    for key in ('K', 'V'):
        if any(value.ndim != 4 or value.shape[1] < positions for value in getattr(decoder, key)):
            raise ValueError('Existing depth cache cannot cover requested positions')
    decoder.K = [value[:, :positions].contiguous() for value in decoder.K]
    decoder.V = [value[:, :positions].contiguous() for value in decoder.V]


NATURAL_SAMPLING = {
    "text_temp": 1.0, "text_topk": 0, "text_topp": 1.0,
    "min_p": 0.0, "min_p_rel": 0, "freq_pen": 0.0,
    "text_rep": 1.0, "word_greedy": 0, "free_text": 1,
    "allow_eob": 1, "cb0_temp": 1.0, "cb0_topk": 0,
    "light_temp": 1.0, "light_topk": 0, "audio_rep_pen": 1.0,
    "pose_temp": 1.0, "pose_topk": 0, "pose_light_temp": 1.0,
    "pose_light_topk": 0, "pose_rep_cb0": 1.0,
    "pose_rep_light": 1.0, "pose_cfg_on": 0, "adaptive_rep_on": 0,
}


@dataclass
class Rollout:
    """Global-offset media arrays on the native, delayed model timeline."""

    text: np.ndarray
    audio: np.ndarray
    pose: np.ndarray
    sampled_masks: dict[str, np.ndarray]
    behavior_logprobs: dict[str, np.ndarray]
    prefix: int
    conversation_frames: int
    prefill_frames: int
    policy_probes: tuple[dict[str, Any], ...] = ()


class _StageProfile:
    """Temporary replay wrappers; remove themselves after the sampled frames."""
    def __init__(self, graphs, devices, frames):
        import torch
        if type(frames) is not int or frames < 1:
            raise ValueError('Stage profiling needs a positive frame count')
        if not devices or any(name not in graphs for name in devices):
            raise ValueError('Capture all actor graphs before enabling stage profiling')
        self.graphs, self.devices, self.frames = graphs, dict(devices), frames
        self.originals = {name: graphs[name] for name in devices}
        self.counts = {name: 0 for name in devices}
        self.submit_seconds = {name: 0. for name in devices}
        self.events = {}
        self.started, self.ended, self.result = None, None, None
        for name, device in devices.items():
            with torch.cuda.device(device):
                pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                         for _ in range(frames)]
                # Materialize CUDA event handles before the measured frames.
                for start, end in pairs:
                    start.record()
                    end.record()
            self.events[name] = pairs
        for name in devices:
            graphs[name] = _ProfiledGraph(self, name)

    def replay(self, name):
        start, end = self.events[name][self.counts[name]]
        before = time.perf_counter()
        if self.started is None:
            self.started = before
        start.record()
        self.originals[name].replay()
        end.record()
        after = time.perf_counter()
        self.submit_seconds[name] += after - before
        self.ended = after
        self.counts[name] += 1
        if self.counts[name] == self.frames:
            self.graphs[name] = self.originals[name]

    def stop(self):
        for name, graph in self.originals.items():
            self.graphs[name] = graph

    def collect(self):
        import torch
        if self.result is not None:
            return self.result
        self.stop()
        # One bounded collection point; never synchronize in frame submission.
        for device in sorted(set(self.devices.values()), key=str):
            torch.cuda.synchronize(device)
        stages = {}
        for name, count in self.counts.items():
            values = [start.elapsed_time(end) for start, end in self.events[name][:count]]
            stages[name] = {'samples': count, 'gpu_ms_total': sum(values),
                            'gpu_ms_mean': sum(values) / count if count else None,
                            'gpu_ms_min': min(values) if count else None,
                            'gpu_ms_max': max(values) if count else None,
                            'cpu_submit_ms_total': 1000 * self.submit_seconds[name]}
        self.result = {'frames_requested': self.frames,
                       'completed': all(count == self.frames for count in self.counts.values()),
                       'cpu_submission_span_seconds': 0. if self.started is None else self.ended - self.started,
                       'stages': stages}
        self.events.clear()
        return self.result


class _ProfiledGraph:
    def __init__(self, profile, name):
        self.profile, self.name = profile, name

    def replay(self):
        self.profile.replay(self.name)

    def __getattr__(self, name):
        return getattr(self.profile.originals[self.name], name)


def _scalar(value: Any, name: str) -> int:
    raw = np.asarray(value)
    if raw.size != 1 or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must be one integer")
    return int(raw.reshape(-1)[0])


def rollout_masks(prefix: int, frames: int, prefill: int, delay: int = CODE_DELAY):
    """Forced prefix, physical prefill and terminal text never become actions."""
    if prefix < 1 or not 0 <= prefill < frames or delay != CODE_DELAY:
        raise ValueError("Invalid prefix, prefill, row length or native code delay")
    total = prefix + frames + delay
    text = np.zeros(total, dtype=bool)
    media = np.zeros(total, dtype=bool)
    text[prefix + prefill:prefix + frames] = True
    media[prefix + prefill + delay:total] = True
    return {"text": text, "audio": media.copy(), "pose": media.copy()}


def _validate_seed(row, prefill):
    prefix = _scalar(row.get("P", np.asarray(row["vin"])[0]), "P")
    frames = _scalar(row.get("conversation_frames", np.asarray(row["vin"])[1]), "conversation_frames")
    if _scalar(row.get("speech_delay", CODE_DELAY), "speech_delay") != CODE_DELAY:
        raise ValueError("Speech generation requires native CD=2")
    if _scalar(row.get("pose_delay", CODE_DELAY), "pose_delay") != CODE_DELAY:
        raise ValueError("Pose generation requires native CD=2")
    masks = rollout_masks(prefix, frames, prefill)
    total = prefix + frames + CODE_DELAY
    arrays = {key: np.asarray(row[key]) for key in ("ids", "depth", "present", "pdepth")}
    shapes = {"ids": (2, total), "depth": (2, total, CODEBOOKS),
              "present": (2, total), "pdepth": (total, CODEBOOKS)}
    for key, shape in shapes.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{key} must have shape {shape}, got {arrays[key].shape}")
    for key in ("ids", "depth", "pdepth"):
        if not np.issubdtype(arrays[key].dtype, np.integer):
            raise ValueError(f"{key} must contain integer IDs")
    supplied = row.get("sampled_masks")
    if supplied is not None:
        for stream, expected in masks.items():
            if not np.array_equal(np.asarray(supplied[stream]), expected):
                raise ValueError(f"{stream} sampling mask does not match the prefill/delay contract")
    if not np.all(arrays["ids"][1, prefix + prefill:] == PAD):
        raise ValueError("Future B text must be removed before student generation")
    if not np.all(arrays["depth"][1, masks["audio"]] == -1):
        raise ValueError("Future B audio must be removed before student generation")
    if not np.all(arrays["pdepth"][masks["pose"]] == -1):
        raise ValueError("Future B pose must be removed before student generation")
    offsets = np.arange(CODEBOOKS, dtype=np.int64) * CODEBOOK_SIZE
    known = slice(prefix + CODE_DELAY, prefix + CODE_DELAY + prefill)
    for key, value in (("audio", arrays["depth"][1, known]), ("pose", arrays["pdepth"][known])):
        local = value - offsets
        if np.any((local < 0) | (local >= CODEBOOK_SIZE)):
            raise ValueError(f"The full physical prefill requires valid {key} codes")
    return prefix, frames, arrays, masks


def rollout_row(engine, row, *, seed: int, prefill_frames: int | None = None, graphs: bool = True) -> Rollout:
    """Generate one entire row with KV caches, including every conversation turn.

    ``row`` must already have future B values scrubbed. A remains fixed. EOB and
    silence do not stop generation. At the two final model positions text is
    forced to PAD while the final physical media frames are generated.
    """
    prefill = _scalar(row.get("prefill_frames", 25) if prefill_frames is None else prefill_frames,
                      "prefill_frames")
    prefix, frames, arrays, masks = _validate_seed(row, prefill)
    total = prefix + frames + CODE_DELAY
    instance = {key: value.copy() for key, value in arrays.items()}
    instance.update(P=prefix, a_mode="file")
    if hasattr(engine, "check_audio_features"):
        engine.check_audio_features(prefix, frames)
    engine.start([instance], int(seed), params=dict(NATURAL_SAMPLING), f0=prefill)
    for frame in range(prefill, frames + CODE_DELAY):
        force = {}
        if frame < prefill + CODE_DELAY:
            force["codes"] = arrays["depth"][1, prefix + frame][None]
            force["pose"] = arrays["pdepth"][prefix + frame][None]
        if frame >= frames:
            force["lane"] = np.full(1, PAD, dtype=np.int64)
        engine.step(frame, graphs=bool(graphs), force=force or None, cold_pad=False)
    return _collect_rollout(engine, row, prefill)


def rollout_rows(engine, rows, *, seeds, prefill_frames=None, graphs=True, progress=None):
    """Batch independent full-row conversations in the same cached actor.

    Prefixes and durations may differ. Finished rows are discarded after their
    two delayed media frames; their later batch slots cannot affect other rows.
    Every instance has its own RNG streams, including across repeated starts.
    """
    rows, seeds = list(rows), list(seeds)
    if not rows or len(seeds) != len(rows) or getattr(engine, "NI", len(rows)) != len(rows):
        raise ValueError("Provide one row and seed per actor instance")
    prefills = [_scalar(row.get("prefill_frames", 25) if prefill_frames is None else prefill_frames,
                        "prefill_frames") for row in rows]
    if len(set(prefills)) != 1:
        raise ValueError("A cached batch requires the same physical prefill length")
    prefill = prefills[0]
    validated = [_validate_seed(row, prefill) for row in rows]
    instances = []
    for i, (prefix, frames, arrays, _masks) in enumerate(validated):
        instance = {key: value.copy() for key, value in arrays.items()}
        instance.update(P=prefix, a_mode="file")
        instances.append(instance)
        if hasattr(engine, "check_audio_features"):
            engine.check_audio_features(prefix, frames, instance=i)
    engine.start(instances, list(map(int, seeds)), params=dict(NATURAL_SAMPLING), f0=prefill)
    lengths = np.asarray([value[1] for value in validated])
    total_frames = int(lengths.max()) + CODE_DELAY
    started = time.monotonic()
    for frame in range(prefill, total_frames):
        force = {}
        if frame < prefill + CODE_DELAY:
            force["codes"] = np.stack([arrays["depth"][1, prefix + frame]
                                       for prefix, _, arrays, _ in validated])
            force["pose"] = np.stack([arrays["pdepth"][prefix + frame]
                                      for prefix, _, arrays, _ in validated])
        if np.any(frame >= lengths):
            force["lane"] = np.where(frame >= lengths, PAD, -1).astype(np.int64)
        engine.step(frame, graphs=bool(graphs), force=force or None, cold_pad=False)
        if progress is not None and ((frame + 1 - prefill) % 25 == 0 or frame + 1 == total_frames):
            # Submission progress only: never force a device/host synchronization.
            progress(current_frame=frame + 1, total_frames=total_frames,
                     seconds_elapsed=time.monotonic() - started)
    return [_collect_rollout(engine, row, prefill, instance=i) for i, row in enumerate(rows)]


def _collect_rollout(engine, row, prefill, instance=0):
    prefix, frames, arrays, masks = _validate_seed(row, prefill)
    total = prefix + frames + CODE_DELAY
    generated = engine.readback(instance, total)
    text = np.asarray(generated["lane_b"]).copy()
    audio = np.asarray(generated["dep_b"]).copy()
    pose = np.asarray(generated["pdep"]).copy()
    for stream, actual, expected in (("text", text, arrays["ids"][1]),
                                     ("audio", audio, arrays["depth"][1]),
                                     ("pose", pose, arrays["pdepth"])):
        if actual.shape != expected.shape or not np.array_equal(actual[~masks[stream]], expected[~masks[stream]]):
            raise RuntimeError(f"Actor changed forced {stream} prefix/prefill/tail values")
    if not np.array_equal(generated["lane_a"], arrays["ids"][0]):
        raise RuntimeError("Actor changed fixed A text conditioning")
    if not np.array_equal(generated["dep_a"], arrays["depth"][0]):
        raise RuntimeError("Actor changed fixed A audio conditioning")
    offsets = np.arange(CODEBOOKS, dtype=np.int64) * CODEBOOK_SIZE
    if np.any((text[masks["text"]] < 0) | (text[masks["text"]] >= EXECUTABLE_VOCAB_SIZE)):
        raise RuntimeError("Actor sampled a padded/non-executable text ID")
    for stream, value in (("audio", audio), ("pose", pose)):
        local = value[masks[stream]] - offsets
        if np.any((local < 0) | (local >= CODEBOOK_SIZE)):
            raise RuntimeError(f"Actor did not generate every {stream} codebook")
    behavior = (engine.behavior_logprobs(total) if getattr(engine, "NI", 1) == 1
                else engine.behavior_logprobs(total, instance=instance))
    for stream, mask in masks.items():
        expected_shape = (total,) if stream == "text" else (total, CODEBOOKS)
        lp = np.asarray(behavior[stream], dtype=np.float32).copy()
        if lp.shape != expected_shape or not np.isfinite(lp[mask]).all() or (lp[mask] > 1e-6).any():
            raise RuntimeError(f"Missing/invalid sampled {stream} behavior likelihoods")
        lp[~mask] = 0.0
        behavior[stream] = lp
    probes = tuple(engine.policy_probes(instance)) if hasattr(engine, 'policy_probes') else ()
    return Rollout(text, audio, pose, masks, behavior, prefix, frames, prefill, probes)


def audio_features_from_latents(engine, latents):
    """Apply THIS student's adapter/gain to shared frozen causal Voxtral latents."""
    import torch
    if not hasattr(engine.m, "vadapter") or not hasattr(engine.m, "vgain"):
        raise ValueError("The duplex student must have its trained audio adapter and gain")
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        latents = torch.as_tensor(latents, device=engine.d0, dtype=torch.bfloat16)
        if latents.ndim != 3 or latents.shape[0] != 1 or latents.shape[1] % 4:
            raise ValueError("Voxtral latents must be [1, 4 * conversation_frames, hidden]")
        return engine.m.vgain(engine.m.vadapter(latents)[0]).detach()


def make_fastppm_class(native, *, require_pose=True):
    """Create the concrete cached actor class after the caller loads native code.

    Imports are deliberately lazy so CPU orchestration/tests need neither CUDA
    extensions nor the production model's global import names. This class uses
    native trunk/tower caches and decoder kernels; it never replays a growing
    full sequence at each generation frame.
    """
    import torch
    require_pose_default = require_pose

    def device_scope(device):
        return torch.cuda.device(device) if torch.device(device).type == 'cuda' else nullcontext()

    class NaturalCachedFastPPM(native.FastPPM):
        def __init__(self, *args, executable_vocab_size=EXECUTABLE_VOCAB_SIZE,
                     sampler_backend='multinomial', depth_cache_positions=64, policy_probe_frames=(),
                     attention_precision='native_bf16', linear_backend='native', require_pose=None, **kwargs):
            if sampler_backend not in ('multinomial', 'inverse_cdf'):
                raise ValueError('Unknown natural categorical sampler backend')
            if type(depth_cache_positions) is not int or depth_cache_positions not in (32, 64):
                raise ValueError('Depth cache capacity must be 32 or 64 positions')
            if attention_precision not in ('native_bf16', 'fp32'):
                raise ValueError('Unknown actor depth attention precision')
            if linear_backend not in ('native', 'tuned_m16'):
                raise ValueError('Unknown actor depth linear backend')
            policy_probe_frames = tuple(policy_probe_frames)
            if any(type(frame) is not int or frame < 0 for frame in policy_probe_frames):
                raise ValueError('Policy probe frames must be nonnegative integers')
            for key, required in (("cfg_rows", False), ("front", False), ("mimi_stream", False)):
                if key in kwargs and kwargs[key] != required:
                    raise ValueError(f"OPD actor requires {key}={required!r}")
                kwargs[key] = required
            kwargs.setdefault("n_inst", 1)
            if int(kwargs["n_inst"]) > 1 and kwargs.get("per_instance_rng") is False:
                raise ValueError("Batched OPD requires independent per-instance RNG streams")
            kwargs.setdefault("per_instance_rng", True)
            super().__init__(*args, **kwargs)
            if (require_pose_default if require_pose is None else require_pose) and not self.pose:
                raise ValueError("Pose OPD needs a student with both speech and pose branches")
            self._opd_media = (('audio', self.dl), ('pose', self.dp)) if self.pose else (('audio', self.dl),)
            self._opd_executable = int(executable_vocab_size)
            self._opd_sampler_backend = sampler_backend
            self._opd_policy_probe_frames = frozenset(policy_probe_frames)
            self._opd_active_probe_frames = self._opd_policy_probe_frames
            self._opd_policy_probes = [[] for _ in range(self.NI)]
            self._opd_depth_probe = None
            if self._opd_policy_probe_frames:
                self._opd_depth_probe = {
                    stream: torch.empty(self.NI, 2, CODEBOOK_SIZE, dtype=torch.float32, device=device)
                    for stream, device in self._opd_media}
            if not 0 < self._opd_executable <= self.V:
                raise ValueError("Executable text vocabulary exceeds the actor head")
            self._opd_audio = torch.zeros(self.NI, self.lmax, self.H, dtype=torch.bfloat16, device=self.d0)
            self._opd_audio_span = [None] * self.NI
            self._opd_prefill_instance = 0
            self._opd_stage_profile = None
            self._opd_input_mode = None
            self._opd_force = None
            self._opd_history = {"text": torch.full((self.NI, self.lmax), float("nan"), device=self.dl)}
            self._opd_history.update({stream: torch.full((self.NI, self.lmax, CODEBOOKS), float("nan"), device=device)
                                     for stream, device in self._opd_media})
            self._opd_now = {stream: torch.zeros(self.NI, CODEBOOKS, device=device)
                             for stream, device in self._opd_media}
            self._opd_lane_force = torch.full((self.NI,), -1, dtype=torch.long, device=self.dl)
            self._opd_uniforms = None
            if sampler_backend == 'inverse_cdf':
                self._opd_uniforms = {'text': torch.full((self.NI, self.lmax), .5, dtype=torch.float64, device=self.dl)}
                self._opd_uniforms.update({stream: torch.full((self.NI, self.lmax, CODEBOOKS), .5,
                                                            dtype=torch.float64, device=device)
                                          for stream, device in self._opd_media})
            self._opd_next_frame = None
            self._opd_pending_frame = None
            self._opd_rewind_window = 0
            self._opd_rewind_frames = []
            self._opd_rewind_states = {}
            self._opd_hook = self.m.inp.register_forward_hook(self._add_audio)
            for stream, _ in self._opd_media:
                decoder = self.dec_s if stream == 'audio' else self.dec_p
                if depth_cache_positions != 64:
                    resize_depth_cache(decoder, depth_cache_positions)
                decoder._opd_logprobs = self._opd_now[stream]
                decoder._opd_uniforms = None if self._opd_uniforms is None else self._opd_uniforms[stream]
                decoder._opd_probe_logits = (None if self._opd_depth_probe is None else
                                            {14: self._opd_depth_probe[stream][:, 0],
                                             31: self._opd_depth_probe[stream][:, 1]})
                if self._opd_uniforms is not None:
                    device = self.dl if stream == 'audio' else self.dp
                    decoder._opd_positions = self.Tv[device][:self.NI]
                    decoder._opd_rows = self.arI[device]
                decoder.gen = MethodType(natural_depth_gen, decoder)
                if attention_precision == 'fp32':
                    decoder.step = MethodType(fp32_attention_depth_step, decoder)
                if linear_backend == 'tuned_m16':
                    original = decoder.lin1
                    decoder.lin1 = lambda x, w, original=original: tuned_m16_linear(native, original, x, w)

        def _add_audio(self, _module, _inputs, output):
            if self._opd_input_mode == "prefill":
                return output + self._opd_audio[self._opd_prefill_instance, :output.shape[0]].to(output.dtype)
            if self._opd_input_mode == "step":
                selected = self._opd_audio[self.arI[self.d0], self.Tv[self.d0][:self.NI]]
                return output + selected.to(output.dtype)
            return output

        def set_audio_features(self, features, *, prefix, instance=0):
            features = torch.as_tensor(features, device=self.d0)
            if features.ndim != 2 or features.shape[1] != self.H or not torch.isfinite(features).all():
                raise ValueError("Audio features must be finite [conversation_frames, backbone_hidden]")
            prefix = int(prefix)
            instance = int(instance)
            if not 0 <= instance < self.NI or prefix < 1 or prefix + len(features) + CODE_DELAY > self.lmax:
                raise ValueError("Audio feature timeline exceeds the actor cache")
            self._opd_audio[instance].zero_()
            self._opd_audio[instance, prefix:prefix + len(features)].copy_(features)
            self._opd_audio_span[instance] = (prefix, len(features))

        def check_audio_features(self, prefix, frames, instance=0):
            if self._opd_audio_span[int(instance)] != (int(prefix), int(frames)):
                raise ValueError("Set the student's exact full-row A audio features before rollout")

        @torch.no_grad()
        def set_audio_frame(self, f, features):
            """Write one batched adapter-output frame at each row's own prefix.

            Features stay on the actor device; this method adds no CPU readback
            or synchronization. The streaming frontend owns waveform/latent
            validation and must supply this model's adapter and gain output.
            """
            f = int(f)
            if self.P is None or f < 0 or any(prefix + f >= self.lmax for prefix in self.P):
                raise ValueError('Streaming audio frame is outside the started actor timeline')
            features = torch.as_tensor(features, device=self.d0, dtype=self._opd_audio.dtype)
            if features.shape != (self.NI, self.H):
                raise ValueError('Streaming audio features must have shape [instances,backbone_hidden]')
            positions = self.Pv[self.d0][:self.NI] + f
            self._opd_audio[self.arI[self.d0], positions] = features

        @torch.no_grad()
        def _prefill_row(self, r, *args, **kwargs):
            previous = self._opd_input_mode
            self._opd_input_mode = "prefill"
            self._opd_prefill_instance = int(r)
            try:
                return super()._prefill_row(r, *args, **kwargs)
            finally:
                self._opd_input_mode = previous

        def _seg(self, si):
            previous = self._opd_input_mode
            self._opd_input_mode = "step"
            try:
                return super()._seg(si)
            finally:
                self._opd_input_mode = previous

        @torch.no_grad()
        def start(self, insts, seed, params=None, f0=0):
            settings = dict(NATURAL_SAMPLING)
            for key, value in (params or {}).items():
                if key not in settings or value != settings[key]:
                    raise ValueError(f"Unsupported OPD sampling override: {key}={value!r}")
            for history in self._opd_history.values():
                history.fill_(float("nan"))
            self._opd_force = None
            self._opd_lane_force.fill_(-1)
            super().start(insts, seed, params=settings, f0=f0)
            self._opd_next_frame = int(f0)
            self._opd_pending_frame = None
            self._opd_rewind_frames = [None] * self._opd_rewind_window
            self._remember_frame(int(f0))
            if self._opd_policy_probe_frames:
                self._opd_policy_probes = [[] for _ in range(self.NI)]
                self._opd_probe_prefill = int(f0)
                self._opd_probe_prefixes = [int(instance['P']) for instance in insts]
                self._opd_probe_lengths = [np.asarray(instance['ids']).shape[1] - prefix - CODE_DELAY
                                          for instance, prefix in zip(insts, self._opd_probe_prefixes)]
                self._opd_active_probe_frames = self._opd_policy_probe_frames | frozenset(
                    frame for length in self._opd_probe_lengths for frame in (length - 1, length + CODE_DELAY - 1))
            if self._opd_uniforms is not None:
                # Own lengths keep each RNG stream independent of padding/NI.
                for stream, generators in (('text', self.gen_t), ('audio', self.gen_a), ('pose', self.gen_p)):
                    if stream not in self._opd_uniforms:
                        continue
                    table = self._opd_uniforms[stream]
                    table.fill_(0.5)
                    for index, instance in enumerate(insts):
                        generator = generators[index] if isinstance(generators, (tuple, list)) else generators
                        total = np.asarray(instance['ids']).shape[1]
                        table[index, :total].uniform_(generator=generator)

        def _draw(self, probability, stream, codebook=None):
            if self._opd_uniforms is None:
                generator = self.gen_t if stream == 'text' else self.gen_a if stream == 'audio' else self.gen_p
                return native.row_multinomial(probability, generator)
            device = self.dp if stream == 'pose' else self.dl
            uniform = self._opd_uniforms[stream][self.arI[device], self.Tv[device][:self.NI]]
            if codebook is not None:
                uniform = uniform[:, codebook]
            return inverse_cdf_sample(probability, uniform)

        def _text(self):
            # Own PAD/NW/EOB output rows are already replaced by text_logits.
            # Exclude ONLY padded head rows, as in Amu's executable-vocab scorer.
            positions = self.Tv[self.dl][:self.NI]
            self.adapt[:, 0].zero_()
            self.adapt[:, 1:].fill_(1.0)
            logits = self.text_logits(self.hn_prev[:self.NI])
            logits[:, self._opd_executable:] = float("-inf")
            self.zt.copy_(logits)
            logp = logits.log_softmax(-1)
            probability = logp.exp()
            sampled = self._draw(probability, 'text')
            force = self._opd_lane_force >= 0
            token = torch.where(force, self._opd_lane_force, sampled)
            lp = torch.where(force, torch.zeros_like(token, dtype=torch.float32),
                             logp.gather(1, sampled[:, None])[:, 0])
            self.probs[:, 0].copy_(probability[:, native.PAD])
            self.probs[:, 1].copy_(probability[:, native.NW])
            self.ptok[:, 0].copy_(lp.exp()); self.ptok[:, 1].copy_(lp.exp())
            self.laneB.scatter_(1, positions[:, None], token[:, None])
            self.tok_now.copy_(token)
            self.ring.push(token)
            self._opd_history["text"][self.arI[self.dl], positions] = lp

        def _speech(self, full):
            forced = self._opd_force is not None and self._opd_force.get("codes") is not None
            active = full and not forced
            if active and self._opd_uniforms is not None:
                super()._speech(False)
                cb0 = self._draw(self.z0.softmax(-1), 'audio', 0)
                codes = self.dec_s.gen(self.hv_s, cb0, self.gen_a, None, None)
                positions = self.Tv[self.dl][:self.NI]
                self.depB[self.arI[self.dl], positions] = codes + self.off_t
                self.pk[self.dl][:, 1:].copy_(self.depB[self.arI[self.dl], positions])
            else:
                super()._speech(active)
            if active:
                positions = self.Tv[self.dl][:self.NI]
                cb0 = self.depB[self.arI[self.dl], positions, 0]
                self._opd_now["audio"][:, 0].copy_(self.z0.log_softmax(-1).gather(1, cb0[:, None])[:, 0])
                self._opd_history["audio"][self.arI[self.dl], positions] = self._opd_now["audio"]

        def _pose(self, full, cfgdec=False):
            if cfgdec:
                raise ValueError("CFG is outside the OPD natural sampling policy")
            forced = self._opd_force is not None and self._opd_force.get("pose") is not None
            active = full and not forced
            if active and self._opd_uniforms is not None:
                super()._pose(False, False)
                cb0 = self._draw(self.zp0[:self.NI].softmax(-1), 'pose', 0)
                zb = (self.m.decp_bb(self.hn_p) * float(self.m.decp_bb_alpha)) if self.bblink else None
                if zb is not None:
                    self.zb_p.copy_(zb)
                codes = self.dec_p.gen(self.hv_p[:self.NI], cb0, self.gen_p, None, None,
                                       zb=None if zb is None else zb[:self.NI])
                positions = self.Tv[self.dp][:self.NI]
                self.pdepB[self.arI[self.dp], positions] = codes + self.offp
                self.pkp[self.dp].copy_(self.pdepB[self.arI[self.dp], positions])
            else:
                super()._pose(active, False)
            if active:
                positions = self.Tv[self.dp][:self.NI]
                cb0 = self.pdepB[self.arI[self.dp], positions, 0]
                self._opd_now["pose"][:, 0].copy_(self.zp0[:self.NI].log_softmax(-1).gather(1, cb0[:, None])[:, 0])
                self._opd_history["pose"][self.arI[self.dp], positions] = self._opd_now["pose"]

        def _rewind_tensors(self):
            # Temporal KV suffixes are intentionally NOT copied. Native attention
            # writes at T and reads only <=T via cache_seqlens=T, so replay simply
            # overwrites them. Depth KV is restarted independently on each draw.
            values = {'hn_prev': self.hn_prev}
            for name in ('ring', 'pidx', 'ptr'):
                value = getattr(self.ring, name, None)
                if value is not None:
                    values['ring.' + name] = value
            if self.pose:
                for name in ('pseen', 'pcfg_guided'):
                    if hasattr(self, name):
                        values[name] = getattr(self, name)
            return values

        @torch.no_grad()
        def enable_rewind(self, window=8):
            """Keep bounded states immediately BEFORE each conversation frame.

            Only the position-indexed inverse-CDF sampler supports rewind: its
            random uniforms are fixed by row/position/codebook, independent of
            replay count. This does not snapshot external Voxtral/Mimi state.
            Call before start(), or at a completed backbone boundary. Delivered
            outputs remain in laneB/depB and can be forced when replaying history.
            """
            if type(window) is not int or window < 1:
                raise ValueError('Rewind window must be a positive integer')
            if self._opd_sampler_backend != 'inverse_cdf':
                raise ValueError('Rewind requires position-indexed inverse_cdf sampling')
            if self._opd_pending_frame is not None:
                raise ValueError('Enable rewind at a completed backbone boundary')
            self._opd_rewind_window = window
            self._opd_rewind_frames = [None] * window
            self._opd_rewind_states = {name: value.new_empty((window, *value.shape))
                                       for name, value in self._rewind_tensors().items()}
            if self._opd_next_frame is not None:
                self._remember_frame(self._opd_next_frame)

        def _remember_frame(self, f):
            if not self._opd_rewind_window:
                return
            slot = f % self._opd_rewind_window
            for name, value in self._rewind_tensors().items():
                self._opd_rewind_states[name][slot].copy_(value)
            self._opd_rewind_frames[slot] = f

        @torch.no_grad()
        def rewind(self, f):
            """Restore state before f, retaining stored tokens for forced replay.

            Callers must replace changed input features and replay every frame
            from f onward. Jumping directly to a later frame would expose stale
            KV entries and is rejected by sample_outputs(). A pending sampled
            frame may be discarded by rewinding, without advancing its backbone.
            """
            f = int(f)
            if (not self._opd_rewind_window or self._opd_next_frame is None or f > self._opd_next_frame
                    or self._opd_rewind_frames[f % self._opd_rewind_window] != f):
                raise ValueError(f'No retained actor state before frame {f}')
            slot = f % self._opd_rewind_window
            for name, value in self._rewind_tensors().items():
                value.copy_(self._opd_rewind_states[name][slot])
            self._opd_rewind_frames = [value if value is None or value <= f else None
                                       for value in self._opd_rewind_frames]
            self._opd_next_frame, self._opd_pending_frame = f, None
            self._opd_force = None
            self._opd_lane_force.fill_(-1)
            self.set_f(f)

        @torch.no_grad()
        def sample_outputs(self, f, graphs=True, force=None, cold_pad=False, sample_text=True):
            """Sample text/media at f from h[f-1], without consuming input A[f].

            ``force`` has lane [NI], codes [NI,32], pose [NI,32], in native
            global-offset IDs. Values may already be CUDA tensors. A forced
            media lane refreshes its tower KV but skips its entire light decoder.
            This permits replay of committed partner outputs without redrawing
            them. Call advance_backbone(f) after making A[f] available.
            With sample_text=False, every lane value must be a committed token;
            it is supplied directly without evaluating the text head.
            """
            f = int(f)
            if self._opd_pending_frame is not None or self._opd_next_frame != f:
                raise ValueError(f'Expected unsampled frame {self._opd_next_frame}, got {f}; '
                                 f'pending backbone={self._opd_pending_frame}')
            force = dict(force or {})
            if set(force) - {'lane', 'codes', 'pose'}:
                raise ValueError('Unknown forced output lane')
            if not self.pose and force.get('pose') is not None:
                raise ValueError('Audio-only actor has no pose output')
            lane = force.pop('lane', None)
            if not sample_text and lane is None:
                raise ValueError('Skipping the text head requires committed lane tokens')
            if lane is None and cold_pad and f == 0:
                lane = torch.full((self.NI,), PAD, dtype=torch.long, device=self.dl)
            if lane is None:
                self._opd_lane_force.fill_(-1)
            else:
                lane = torch.as_tensor(lane, device=self.dl, dtype=torch.long)
                if lane.shape != (self.NI,):
                    raise ValueError('Forced text must have shape [instances]')
                self._opd_lane_force.copy_(lane)
            for name, device in (('codes', self.dl), ('pose', self.dp)):
                if force.get(name) is not None:
                    value = torch.as_tensor(force[name], device=device, dtype=torch.long)
                    if value.shape != (self.NI, CODEBOOKS):
                        raise ValueError(f'Forced {name} must have shape [instances,32]')
                    force[name] = value
            self.set_f(f)
            use, full = graphs and bool(self.graphs), f >= CODE_DELAY
            dl, dp = self.dl, self.dp
            self._opd_force = force or None
            try:
                with device_scope(dl):
                    if not sample_text:
                        self._supply_text(lane)
                    elif use:
                        self.graphs['text'].replay()
                    else:
                        self._text()
                    if self.pose:
                        self.ev_text.record(torch.cuda.current_stream(dl))
                if self.pose:
                    if dp != dl:
                        with device_scope(dl), torch.cuda.stream(self.cs_side):
                            self.cs_side.wait_event(self.ev_text)
                            self.tokp.copy_(self.tok_now)
                            self.hn_p.copy_(self.hn_prev)
                            self.adapt_p.copy_(self.adapt)
                    else:
                        self.tokp.copy_(self.tok_now)
                        self.hn_p.copy_(self.hn_prev)
                        self.adapt_p.copy_(self.adapt)
                    with device_scope(dp):
                        active = full and force.get('pose') is None
                        if use:
                            self.graphs['pose_full' if active else 'pose'].replay()
                        else:
                            self._pose(active, False)
                        if force.get('pose') is not None:
                            self.pdepB[self.arI[dp], self.Tv[dp][:self.NI]] = force['pose']
                            self.pkp[dp].copy_(force['pose'])
                with device_scope(dl):
                    active = full and force.get('codes') is None
                    if use:
                        self.graphs['speech_full' if active else 'speech'].replay()
                    else:
                        self._speech(active)
                    if force.get('codes') is not None:
                        self.depB[self.arI[dl], self.Tv[dl][:self.NI]] = force['codes']
                        self.pk[dl][:, 1:].copy_(force['codes'])
                self._opd_pending_frame = f
                if sample_text and f in self._opd_active_probe_frames:
                    self._record_policy_probe(f)
                return full
            finally:
                self._opd_force = None

        @torch.no_grad()
        def advance_backbone(self, f, graphs=True, force=None):
            """Consume current outputs and exact input features A[f].

            After rewind, force={lane,codes[,pose]} can supply already committed
            outputs without refreshing either tower. This is valid only for the
            first changed input position: its towers still use unchanged h[f-1].
            Subsequent replay positions must use sample_outputs with forcing so
            their tower KV incorporates the corrected backbone history.
            """
            f = int(f)
            if force is not None:
                if self._opd_pending_frame is not None or self._opd_next_frame != f:
                    raise ValueError('Backbone-only forcing requires an unsampled current frame')
                required = {'lane', 'codes'} | ({'pose'} if self.pose else set())
                if set(force) != required:
                    raise ValueError(f'Backbone-only forcing needs exactly {sorted(required)}')
                self.set_f(f)
                lane = torch.as_tensor(force['lane'], dtype=torch.long, device=self.dl)
                codes = torch.as_tensor(force['codes'], dtype=torch.long, device=self.dl)
                if lane.shape != (self.NI,) or codes.shape != (self.NI, CODEBOOKS):
                    raise ValueError('Committed text/audio output shapes differ from actor batch')
                self._supply_text(lane)
                self.depB[self.arI[self.dl], self.Tv[self.dl][:self.NI]] = codes
                self.pk[self.dl][:, 0].copy_(lane)
                self.pk[self.dl][:, 1:].copy_(codes)
                if self.pose:
                    pose = torch.as_tensor(force['pose'], dtype=torch.long, device=self.dp)
                    if pose.shape != (self.NI, CODEBOOKS):
                        raise ValueError('Committed pose shape differs from actor batch')
                    self.pdepB[self.arI[self.dp], self.Tv[self.dp][:self.NI]] = pose
                    self.pkp[self.dp].copy_(pose)
                self._opd_pending_frame = f
            if self._opd_pending_frame != f or self._opd_next_frame != f:
                raise ValueError(f'Backbone frame {f} has no matching sampled outputs')
            use = graphs and bool(self.graphs)
            for si, (device, _) in enumerate(self.segs):
                with device_scope(device):
                    if si == 0:
                        if device != self.dl:
                            self.pk[device].copy_(self.pk[self.dl])
                        if self.pose and device != self.dp:
                            self.pkp[device].copy_(self.pkp[self.dp])
                    else:
                        self.xi[si].copy_(self.xo[si - 1])
                    if use:
                        self.graphs[f'seg{si}'].replay()
                    else:
                        self._seg(si)
            self._opd_pending_frame = None
            self._opd_next_frame = f + 1
            self._remember_frame(f + 1)

        def _supply_text(self, lane):
            """Force an entire committed lane without sampling or a head forward."""
            self.laneB[self.arI[self.dl], self.Tv[self.dl][:self.NI]] = lane
            self.tok_now.copy_(lane)
            self.ring.push(lane)
            self.adapt[:, 0].zero_()
            self.adapt[:, 1:].fill_(1.)

        @torch.no_grad()
        def force_media_rows(self, f, *, mask, codes=None, pose=None):
            """Replace sampled media for rows still inside their own prefill.

            Call after sample_outputs and before advance_backbone. The towers
            depend on previous media only; their current KV stays valid. The
            per-frame light caches have no persistent temporal state. Replacing
            the codes and backbone-input pack therefore preserves the exact
            teacher-forced path while mixed batches keep static sampling shapes.
            Forced rows get zero behavior log-probabilities, never action credit.
            """
            f = int(f)
            if self._opd_pending_frame != f or self._opd_next_frame != f:
                raise ValueError('Media forcing requires sampled outputs awaiting their backbone')
            if codes is None and pose is None:
                raise ValueError('Provide audio or pose codes to force')
            if pose is not None and not self.pose:
                raise ValueError('Audio-only actor has no pose output')
            for stream, value, device in (('audio', codes, self.dl), ('pose', pose, self.dp)):
                if value is None:
                    continue
                row_mask = torch.as_tensor(mask, dtype=torch.bool, device=device)
                value = torch.as_tensor(value, dtype=torch.long, device=device)
                if row_mask.shape != (self.NI,) or value.shape != (self.NI, CODEBOOKS):
                    raise ValueError('Forced media/mask must be [instances,32] and [instances]')
                positions, rows = self.Tv[device][:self.NI], self.arI[device]
                target = self.depB if stream == 'audio' else self.pdepB
                selected = torch.where(row_mask[:, None], value, target[rows, positions])
                target[rows, positions] = selected
                if stream == 'audio':
                    self.pk[self.dl][:, 1:].copy_(selected)
                else:
                    self.pkp[self.dp].copy_(selected)
                history = self._opd_history[stream]
                history[rows, positions] = torch.where(row_mask[:, None], 0., history[rows, positions])

        @torch.no_grad()
        def step(self, f, graphs=True, force=None, cold_pad=False):
            full = self.sample_outputs(f, graphs=graphs, force=force, cold_pad=cold_pad)
            self.advance_backbone(f, graphs=graphs)
            return full

        def _record_policy_probe(self, frame):
            # Explicit diagnostic only: collect a handful of full categorical
            # distributions from their actual sampled contexts, never histories.
            text = self.zt[:, :self._opd_executable].detach().float().cpu().numpy()
            media = {stream: (cb0[:self.NI].detach().float().cpu().numpy(),
                              self._opd_depth_probe[stream].detach().float().cpu().numpy())
                     for stream, cb0 in ([('audio', self.z0)] + ([('pose', self.zp0)] if self.pose else []))
                     if frame >= self._opd_probe_prefill + CODE_DELAY}
            for row, (prefix, frames) in enumerate(zip(self._opd_probe_prefixes, self._opd_probe_lengths)):
                if frame not in self._opd_policy_probe_frames and frame not in (frames - 1, frames + CODE_DELAY - 1):
                    continue
                position = prefix + int(frame)
                if self._opd_probe_prefill <= frame < frames:
                    self._opd_policy_probes[row].append({'position': position, 'stream': 'text', 'codebook': -1,
                                                        'logits': text[row].copy()})
                if self._opd_probe_prefill + CODE_DELAY <= frame < frames + CODE_DELAY:
                    for stream, (cb0, depth) in media.items():
                        for codebook, logits in ((0, cb0[row]), (14, depth[row, 0]), (31, depth[row, 1])):
                            self._opd_policy_probes[row].append({'position': position, 'stream': stream,
                                                                'codebook': codebook, 'logits': logits.copy()})

        def policy_probes(self, instance=0):
            return tuple(self._opd_policy_probes[int(instance)])

        def behavior_logprobs(self, total, instance=0):
            return {key: value[int(instance), :int(total)].detach().float().cpu().numpy().copy()
                    for key, value in self._opd_history.items()}

        def enable_stage_profile(self, frames=32):
            if self._opd_stage_profile is not None:
                self._opd_stage_profile.stop()
            devices = {'text': self.dl, 'speech_full': self.dl}
            if self.pose:
                devices['pose_full'] = self.dp
            devices.update({f'seg{i}': device for i, (device, _) in enumerate(self.segs)})
            self._opd_stage_profile = _StageProfile(self.graphs, devices, frames)

        def stage_profile_result(self):
            return None if self._opd_stage_profile is None else self._opd_stage_profile.collect()

        def close(self):
            self._opd_hook.remove()
            if self._opd_stage_profile is not None:
                self._opd_stage_profile.stop()
                self._opd_stage_profile = None
            # Remove instance-bound overrides, which otherwise form reference
            # cycles retaining actor weights until Python's next GC pass.
            for decoder in [self.dec_s] + ([self.dec_p] if self.pose else []):
                if "gen" in decoder.__dict__:
                    del decoder.gen
                if "step" in decoder.__dict__:
                    del decoder.step
                if "lin1" in decoder.__dict__:
                    del decoder.lin1
                if hasattr(decoder, "_opd_logprobs"):
                    del decoder._opd_logprobs
                for name in ('_opd_uniforms', '_opd_positions', '_opd_rows', '_opd_probe_logits'):
                    if hasattr(decoder, name):
                        delattr(decoder, name)

    def fp32_attention_depth_step(decoder, x, j):
        """Native exact-dtype decoder, with FP32 P×V as replay actually uses.

        The vendored replay attention uses IEEE FP32 P×V despite an older
        BF16-P docstring. Keep every norm, residual and GEMM unchanged here;
        select the native attention kernel's existing unrounded-P branch.
        """
        rows, length, hidden = x.shape
        if rows != decoder.S:
            raise ValueError('Depth decoder batch does not match its cache')
        start = 0 if j == 0 else j + 1
        cos, sin = decoder.cos[start:start + length], decoder.sin[start:start + length]
        lx = native.LX
        residual, normalized = lx.add_rms_f32(x.reshape(rows * length, hidden).float().contiguous(),
                                             None, decoder.layers[0]['ln1'], decoder.layers[0]['e1'])
        mlp = None
        for index, layer in enumerate(decoder.layers):
            if index:
                residual, normalized = lx.add_rms_f32(residual, mlp, layer['ln1'], layer['e1'])
            qkv = decoder.lin1(normalized, layer['wqkv'])
            attended = lx.attn_exact(qkv, layer['qn'], layer['kn'], layer['eqk'], cos, sin,
                                     decoder.K[index], decoder.V[index], start, length,
                                     decoder.nh, decoder.kv, decoder.D, layer['sc'], round_p=0)
            attended = decoder.lin1(attended, layer['wo'])
            residual, normalized = lx.add_rms_f32(residual, attended, layer['ln2'], layer['e2'])
            mlp = decoder.lin1(native.LK.silu_mul_gu(decoder.lin1(normalized, layer['wgu'])), layer['wd'])
        output = lx.final_norm_f32(residual, mlp, decoder.nw, decoder.ne)
        return output.view(rows, length, hidden)[:, -1]

    def natural_depth_gen(decoder, hv, c0, gen, temp, topk, zb=None, pres=None,
                          rp=None, cfg_w=None, cfg_on=None):
        if hv.shape[0] != c0.shape[0] or cfg_on is not None:
            raise ValueError("Natural OPD depth sampling does not accept CFG rows")
        x = torch.cat([decoder.slot0(hv, zb)[:, None], decoder.dec.cb0_emb(c0)[:, None].float()], 1)
        codes = [c0]
        for codebook in range(1, CODEBOOKS):
            logits = decoder.head(decoder.step(x, codebook - 1), codebook)
            if decoder._opd_probe_logits is not None and codebook in decoder._opd_probe_logits:
                decoder._opd_probe_logits[codebook].copy_(logits)
            logp = logits.log_softmax(-1)
            if decoder._opd_uniforms is None:
                code = native.row_multinomial(logp.exp(), gen)
            else:
                uniform = decoder._opd_uniforms[decoder._opd_rows, decoder._opd_positions, codebook]
                code = inverse_cdf_sample(logp.exp(), uniform)
            decoder._opd_logprobs[:, codebook].copy_(logp.gather(1, code[:, None])[:, 0])
            codes.append(code)
            if codebook < CODEBOOKS - 1:
                x = decoder.emb(code, codebook)
        return torch.stack(codes, 1)

    return NaturalCachedFastPPM
