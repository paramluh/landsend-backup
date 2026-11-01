"""The waveform clock for a frozen conversational partner with delivery delay.

All frame numbers here are physical conversation frames, except explicitly
named ``model_frame`` fields. Prefix tokens are deliberately not included.
An output audio frame j lives at model frame j+CD and reads h[j+CD-1].
Its incoming waveform therefore ends at (j+CD)*SPF+LOOK, exclusively, on
the receiver's clock. A delivery delay d maps this to source endpoint
(j+CD)*SPF+LOOK-d. The native frontend and output frame grid are unchanged.

The student never hears placeholders. The teacher hears actual student audio
where available and digital silence in its unknown future. Once any part of a
teacher Mimi frame has been heard, its WHOLE token frame/waveform is fixed.
Reconditioning the teacher forces those old text/audio tokens while rebuilding
its conditioning states. It does not redraw old speech or rewind Mimi.

With CD=2, LOOK=72 and no delivery delay, each student frame requires six teacher
frontend/backbone positions, five forced tower positions and one new tower /
light draw. Forced positions can be processed as causal blocks. This count is
not six full light samplings. Delaying delivered audio by 72 samples in both
directions, including the real prefill, reduces the count to four backbone and
three forced tower positions without removing the native frontend lookahead.

At rollback, restore hn *before* the first affected model frame, the Voxtral
conv histories (mel_prev/c1_prev), and position counters. Cached KV prefixes
remain valid; stale suffix slots are overwritten with rewound cache lengths.
Snapshotting whole KV arrays or rolling back committed Mimi state is neither
necessary nor correct. The CPU reference below checks this clock and output
commitment, not GPU numerical parity of a particular model implementation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class AudioTiming:
    frame_samples: int = 1920
    lookahead_samples: int = 72
    code_delay: int = 2
    delivery_delay_samples: int = 0

    def __post_init__(self):
        if type(self.frame_samples) is not int or self.frame_samples <= 0:
            raise ValueError("frame_samples must be a positive integer")
        if (type(self.lookahead_samples) is not int
                or not 0 <= self.lookahead_samples < self.frame_samples):
            raise ValueError("lookahead_samples must lie within one frame")
        if type(self.code_delay) is not int or self.code_delay < 1:
            raise ValueError("code_delay must be a positive integer")
        if (type(self.delivery_delay_samples) is not int
                or not 0 <= self.delivery_delay_samples < self.frame_samples):
            raise ValueError("delivery_delay_samples must lie within one frame")

    def input_stop_for_output(self, output_frame: int) -> int:
        """Exclusive receiver-clock endpoint needed to predict output_frame."""
        _frame(output_frame)
        return (output_frame + self.code_delay) * self.frame_samples + self.lookahead_samples

    def source_stop_for_output(self, output_frame: int) -> int:
        """Exclusive source-waveform endpoint, before delivery delay is added."""
        return self.input_stop_for_output(output_frame) - self.delivery_delay_samples

    def partner_frame_for_output(self, output_frame: int) -> int:
        """Last whole partner Mimi frame needed, including frontend lookahead."""
        return (self.source_stop_for_output(output_frame) - 1) // self.frame_samples

    def earliest_frontend_changed(self, changed_audio_frame: int) -> int:
        """First frontend frame affected when this whole audio frame changes."""
        _frame(changed_audio_frame)
        return max(0, (changed_audio_frame * self.frame_samples + self.delivery_delay_samples
                       - self.lookahead_samples)
                   // self.frame_samples)


DEFAULT_TIMING = AudioTiming()


def _frame(value: int):
    if type(value) is not int or value < 0:
        raise ValueError("Frame indices must be nonnegative integers")


@dataclass(frozen=True)
class FrameInterval:
    """A half-open interval; use .frames() for executable frame indices."""
    start: int
    stop: int

    def __post_init__(self):
        _frame(self.start)
        _frame(self.stop)
        if self.stop < self.start:
            raise ValueError("Frame interval ends before its start")

    def frames(self):
        return range(self.start, self.stop)

    def __len__(self):
        return self.stop - self.start


@dataclass(frozen=True)
class TeacherRefreshPlan:
    student_frame: int
    newly_real_student_frame: int
    teacher_new_audio_frame: int
    teacher_sample_model_frame: int
    teacher_backbone: FrameInterval
    teacher_forced_tower: FrameInterval
    student_heard_stop: int
    teacher_input_stop: int
    student_real_stop: int
    student_source_stop: int
    teacher_source_stop: int
    teacher_received_real_stop: int

    @property
    def teacher_tower(self):
        return FrameInterval(self.teacher_forced_tower.start, self.teacher_sample_model_frame + 1)


def plan_for_student(student_frame: int, prefill_frames: int,
                     timing: AudioTiming = DEFAULT_TIMING) -> TeacherRefreshPlan:
    """Refresh after S[student_frame-1], before predicting S[student_frame].

    The first generated student frame is initialized by bootstrap_plan instead.
    Only the final teacher tower position samples new text/audio; every earlier
    replayed output is forced to its already-committed value.
    """
    _frame(student_frame)
    _frame(prefill_frames)
    if student_frame <= prefill_frames:
        raise ValueError("Use bootstrap_plan for the first generated student frame")
    changed = student_frame - 1
    first = timing.earliest_frontend_changed(changed)
    teacher_audio = timing.partner_frame_for_output(student_frame)
    sample_model = teacher_audio + timing.code_delay
    return TeacherRefreshPlan(
        student_frame=student_frame, newly_real_student_frame=changed,
        teacher_new_audio_frame=teacher_audio, teacher_sample_model_frame=sample_model,
        teacher_backbone=FrameInterval(first, sample_model),
        teacher_forced_tower=FrameInterval(first + 1, sample_model),
        student_heard_stop=timing.input_stop_for_output(student_frame),
        teacher_input_stop=timing.input_stop_for_output(teacher_audio),
        student_real_stop=student_frame * timing.frame_samples,
        student_source_stop=timing.source_stop_for_output(student_frame),
        teacher_source_stop=timing.source_stop_for_output(teacher_audio),
        teacher_received_real_stop=student_frame * timing.frame_samples + timing.delivery_delay_samples,
    )


@dataclass(frozen=True)
class TeacherBootstrapPlan:
    prefill_frames: int
    teacher_prefill_stop_model_frame: int
    teacher_new_audio: FrameInterval
    teacher_backbone_stop_model_frame: int
    student_forced_media_model_frames: FrameInterval
    student_heard_stop: int
    teacher_input_stop: int
    student_source_stop: int
    teacher_source_stop: int
    teacher_received_real_stop: int


def bootstrap_plan(prefill_frames: int, timing: AudioTiming = DEFAULT_TIMING) -> TeacherBootstrapPlan:
    """Initialize teacher first so the student's prefill gets real lookahead.

    Teacher input contains delivery-delay zeros from conversation sample 0,
    followed by the student's real opening and unknown-future silence. Without
    delivery delay, teacher new audio is N,N+1,N+2 and frontend N-1 already sees
    teacher frame N's first 72 samples. With delivery delay equal to lookahead,
    both prefills end at N and only teacher frames N,N+1 are newly required.
    Student model frames N,N+1 retain real delayed media but sample fresh text.
    """
    _frame(prefill_frames)
    if prefill_frames < timing.code_delay:
        raise ValueError("Prefill must cover the code delay")
    last = timing.partner_frame_for_output(prefill_frames)
    return TeacherBootstrapPlan(
        prefill_frames=prefill_frames,
        teacher_prefill_stop_model_frame=timing.earliest_frontend_changed(prefill_frames),
        teacher_new_audio=FrameInterval(prefill_frames, last + 1),
        teacher_backbone_stop_model_frame=last + timing.code_delay,
        student_forced_media_model_frames=FrameInterval(prefill_frames, prefill_frames + timing.code_delay),
        student_heard_stop=timing.input_stop_for_output(prefill_frames),
        teacher_input_stop=timing.input_stop_for_output(last),
        student_source_stop=timing.source_stop_for_output(prefill_frames),
        teacher_source_stop=timing.source_stop_for_output(last),
        teacher_received_real_stop=prefill_frames * timing.frame_samples + timing.delivery_delay_samples,
    )


class CommittedAudioTimeline:
    """An append-only physical waveform; reading unknown future is an error."""

    def __init__(self, prefill_audio, *, frame_samples: int = 1920, delivery_delay_samples: int = 0):
        if type(frame_samples) is not int or frame_samples <= 0:
            raise ValueError("frame_samples must be a positive integer")
        if type(delivery_delay_samples) is not int or not 0 <= delivery_delay_samples < frame_samples:
            raise ValueError("delivery_delay_samples must lie within one frame")
        audio = np.asarray(prefill_audio, dtype=np.float32)
        if audio.ndim != 1 or len(audio) % frame_samples or not np.isfinite(audio).all():
            raise ValueError("Prefill must be finite mono audio with whole frames")
        self.frame_samples = frame_samples
        self.delivery_delay_samples = delivery_delay_samples
        self._frames = [part.copy() for part in audio.reshape(-1, frame_samples)]
        self.heard_stop = 0

    @property
    def next_frame(self):
        return len(self._frames)

    @property
    def available_stop(self):
        return self.next_frame * self.frame_samples

    @property
    def delivered_available_stop(self):
        return self.available_stop + self.delivery_delay_samples

    def append(self, frame: int, samples):
        _frame(frame)
        if frame != self.next_frame:
            raise ValueError("Audio is append-only; committed teacher frames cannot be redrawn")
        samples = np.asarray(samples, dtype=np.float32)
        if samples.shape != (self.frame_samples,) or not np.isfinite(samples).all():
            raise ValueError("A committed audio frame must be finite mono samples")
        self._frames.append(samples.copy())

    def read(self, start: int, stop: int):
        if type(start) is not int or type(stop) is not int or not 0 <= start <= stop <= self.available_stop:
            raise ValueError("Requested audio is not yet committed")
        if start == stop:
            return np.empty(0, dtype=np.float32)
        first, last = start // self.frame_samples, (stop - 1) // self.frame_samples
        wave = np.concatenate(self._frames[first:last + 1])
        return wave[start - first * self.frame_samples:stop - first * self.frame_samples].copy()

    def hear_through(self, stop: int):
        """Record exact waveform exposure; this never permits partial-frame rewrite."""
        result = self.delivered_read(0, stop)
        self.heard_stop = max(self.heard_stop, stop)
        return result

    def delivered_read(self, start: int, stop: int):
        """Read the receiver stream, including the same delay from sample 0."""
        if (type(start) is not int or type(stop) is not int
                or not 0 <= start <= stop <= self.delivered_available_stop):
            raise ValueError("Requested audio is not yet committed")
        result = np.zeros(stop - start, dtype=np.float32)
        source_start = max(0, start - self.delivery_delay_samples)
        source_stop = max(0, stop - self.delivery_delay_samples)
        if source_stop > source_start:
            offset = max(0, self.delivery_delay_samples - start)
            result[offset:offset + source_stop - source_start] = self.read(source_start, source_stop)
        return result

    def silence_future(self, stop: int):
        """Teacher-only view: known real audio followed by digital zeros."""
        if type(stop) is not int or stop < self.delivered_available_stop:
            raise ValueError("Speculative view must cover all committed student audio")
        out = np.zeros(stop, dtype=np.float32)
        out[:self.delivered_available_stop] = self.delivered_read(0, self.delivered_available_stop)
        return out


@dataclass(frozen=True)
class ReferenceRollout:
    student_audio: np.ndarray
    teacher_audio: np.ndarray
    teacher_heard_stops: tuple[int, ...]
    student_heard_stops: tuple[int, ...]


def reference_rollout(student_prefill, teacher_prefill, frames: int,
                      student_generate: Callable, teacher_generate: Callable,
                      timing: AudioTiming = DEFAULT_TIMING) -> ReferenceRollout:
    """CPU oracle for timing/commitment, with waveform-valued model callbacks.

    callback(physical_output_frame, exact_incoming_waveform, committed_own_waveform)
    returns one frame. A production teacher callback can recompute from scratch
    with its own committed tokens forced; the optimized engine must match that
    oracle after its short rollback. No post-prefill dataset audio is accepted.
    """
    _frame(frames)
    student = CommittedAudioTimeline(student_prefill, frame_samples=timing.frame_samples,
                                    delivery_delay_samples=timing.delivery_delay_samples)
    teacher = CommittedAudioTimeline(teacher_prefill, frame_samples=timing.frame_samples,
                                    delivery_delay_samples=timing.delivery_delay_samples)
    n = student.next_frame
    if teacher.next_frame != n or frames <= n:
        raise ValueError("Equal-length prefills and at least one generated frame are required")
    initial = bootstrap_plan(n, timing)
    teacher_reads, student_reads = [], []

    def append_teacher(frame):
        stop = timing.input_stop_for_output(frame)
        incoming = student.silence_future(stop)
        own = teacher.read(0, teacher.available_stop)
        teacher.append(frame, teacher_generate(frame, incoming, own))
        teacher_reads.append(stop)

    for frame in initial.teacher_new_audio.frames():
        append_teacher(frame)
    for frame in range(n, frames):
        if frame > n:
            plan = plan_for_student(frame, n, timing)
            append_teacher(plan.teacher_new_audio_frame)
        stop = timing.input_stop_for_output(frame)
        incoming = teacher.hear_through(stop)
        student_reads.append(stop)
        own = student.read(0, student.available_stop)
        student.append(frame, student_generate(frame, incoming, own))
    return ReferenceRollout(
        student.read(0, student.available_stop), teacher.read(0, teacher.available_stop),
        tuple(teacher_reads), tuple(student_reads),
    )
