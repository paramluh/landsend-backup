"""One pose student batch with independently scheduled audio-partner replicas.

Partners own disjoint contiguous rows and GPU pairs. Their committed-token
refreshes run concurrently on persistent host workers; the student consumes a
frame only after every partner's decoded PCM is ready. The waveform clock,
forced prefixes, inverse gains and delivery-delay replay match partner_rollout.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
import threading
import time

import numpy as np
import torch

from .partner_audio import AudioTimeline, StreamingFront, StreamingDecoder
from .partner_rollout import ForcedPrefix, _AudioProfile, _codes, _committed, _instances
from .rollout import CODE_DELAY, NATURAL_SAMPLING, _collect_rollout, _validate_seed
from .teacher_partner_schedule import AudioTiming, bootstrap_plan, plan_for_student


# CUDA graph capture and its lazy allocations are serialized during setup.
# Independent baseline teams can rendezvous after setup before timed generation.
_SETUP_LOCK = threading.RLock()


def partner_partitions(student_count, partner_counts):
    """Stable row assignment; seeds and learner ownership never depend on fanout."""
    counts = list(partner_counts)
    if (type(student_count) is not int or student_count < 1 or not counts
            or any(type(count) is not int or count < 1 for count in counts)
            or sum(counts) != student_count):
        raise ValueError('Positive partner batch sizes must sum to the student batch')
    start, parts = 0, []
    for count in counts:
        parts.append(slice(start, start + count))
        start += count
    return parts


def _dispatch(pool, function, values):
    """Submit every independent partner before waiting; retain input row order."""
    if pool is None:
        return [function(value) for value in values]
    futures = [pool.submit(function, value) for value in values]
    return [future.result() for future in futures]


class _Profile:
    def __init__(self, calls):
        self.profile = _AudioProfile(calls) if calls else None

    def run(self, name, device, function):
        if self.profile is None:
            return function()
        return self.profile.run(name, device, function)

    def result(self):
        return {} if self.profile is None else self.profile.result()


class _Partner:
    def __init__(self, engine, rows, seeds, front, mimi_path, *, frames, capacity,
                 graphs, f0, fused_ops, profile_calls, delivery_delay_samples=0):
        from transformers import MimiModel
        self.engine, self.rows = engine, rows
        self.graphs, self.frames = graphs, frames
        self.profile = _Profile(profile_calls)
        self.streams = {device: torch.cuda.Stream(device=device) for device in engine.sdevs}
        # Imported actor weights/caches were created on the caller's streams.
        for device, stream in self.streams.items():
            stream.wait_stream(torch.cuda.current_stream(device))
        with self.context():
            self.instances = _instances(engine, rows, extra=8)
            self.forcing = ForcedPrefix(engine, rows, frames)
            self.timeline = AudioTimeline([row['wav'] for row in rows], capacity, engine.d0,
                inverse_gains=[row['_student_inverse_gain'] for row in rows],
                delivery_delay_samples=delivery_delay_samples)
            self.front = StreamingFront(front, engine, self.timeline, fa=engine.fa,
                fused_ops=fused_ops, frames=capacity, graphs=graphs)
            mimi = MimiModel.from_pretrained(str(mimi_path), local_files_only=True).to(engine.d0).eval()
            self.decoder = StreamingDecoder(mimi, len(rows), engine.d0, graphs=graphs)
            for physical in range(f0 - CODE_DELAY):
                codes = np.stack([row['depth'][1, int(np.asarray(row['P']).reshape(-1)[0])
                                                + physical + CODE_DELAY] for row in rows])
                self.decoder.decode(physical, torch.as_tensor(codes, device=engine.dl))
            for frame in range(f0):
                self.front.step(frame, speculative=True)
            engine.start(self.instances, [seed ^ 0x5A17B39D for seed in seeds],
                         params=dict(NATURAL_SAMPLING), f0=f0)
            if engine._opd_rewind_window < 8:
                engine.enable_rewind(8)

    @contextmanager
    def context(self):
        with ExitStack() as stack:
            for device in self.engine.sdevs:
                stack.enter_context(torch.cuda.stream(self.streams[device]))
            yield

    def ready(self):
        event = torch.cuda.Event()
        event.record(self.streams[self.engine.d0])
        return event

    @torch.no_grad()
    def bootstrap_frame(self, frame, last):
        with self.context():
            self.forcing.sample(frame, self.graphs, tail_pad=False)
            wave = self.decoder.decode(frame - CODE_DELAY, _codes(self.engine, frame)).clone()
            if frame < last:
                self.front.step(frame, speculative=True)
                self.engine.advance_backbone(frame, graphs=self.graphs)
            return wave, self.ready()

    @torch.no_grad()
    def ingest(self, frame, wave, input_ready):
        with self.context():
            for stream in self.streams.values():
                stream.wait_event(input_ready)
            self.timeline.append(frame, wave)
            return self.ready()

    @torch.inference_mode()
    def refresh(self, frame, wave, input_ready, plan, backend):
        with self.context():
            for stream in self.streams.values():
                stream.wait_event(input_ready)
            engine, front, profile = self.engine, self.front, self.profile
            profile.run('student_to_partner', engine.d0, lambda: self.timeline.append(frame, wave))
            if plan is None:
                return None, self.ready()
            start = plan.teacher_backbone.start
            engine.rewind(start)
            front.rewind(start)
            if backend == 'block':
                from .partner_block import replay_committed_block
                profile.run('partner_voxtral_block', front.device,
                    lambda: front.step_block(start, len(plan.teacher_backbone), speculative=True))
                profile.run('partner_committed_block', engine.dl,
                    lambda: replay_committed_block(engine, start, plan.teacher_backbone.stop,
                                                   graphs=self.graphs))
            else:
                for position in plan.teacher_backbone.frames():
                    old = _committed(engine, position)
                    if position != start:
                        engine.sample_outputs(position, graphs=self.graphs, force=old, sample_text=False)
                    front.step(position, speculative=True)
                    engine.advance_backbone(position, graphs=self.graphs,
                                            force=old if position == start else None)
            self.forcing.sample(plan.teacher_sample_model_frame, self.graphs, tail_pad=False)
            wave = profile.run('partner_mimi', self.decoder.device,
                lambda: self.decoder.decode(plan.teacher_new_audio_frame,
                                           _codes(engine, plan.teacher_sample_model_frame)))
            return wave, self.ready()


def _merge_audio(target, parts, outputs):
    """A caller stream must first wait for each output's producer event."""
    for part, (wave, _) in zip(parts, outputs, strict=True):
        target[part].copy_(wave, non_blocking=True)
    return target


@torch.no_grad()
def rollout_teacher_partners(student_engine, partner_engines, rows, seeds, student_front,
                             partner_fronts, mimi_path, *, graphs=True, progress=None,
                             teacher_replay='block', concurrent=True, capacity_frames=None,
                             startup_barrier=None, profile_calls=0, delivery_delay_samples=0):
    """Generate one student batch with one or more row-partitioned partners.

    ``capacity_frames`` sizes PCM and recorded frontend buffers independently
    of measured rollout length, in addition to the caller's full actor caches.
    For two independent baseline teams, pass the same two-party startup_barrier
    to both calls: setup/capture is serial, then their generation loops overlap.
    """
    from transformers import MimiModel
    from . import runtime

    timing = AudioTiming(delivery_delay_samples=delivery_delay_samples)
    started = time.monotonic()
    rows, seeds = list(rows), list(map(int, seeds))
    engines, fronts = list(partner_engines), list(partner_fronts)
    parts = partner_partitions(student_engine.NI, [engine.NI for engine in engines])
    if len(rows) != student_engine.NI or len(seeds) != len(rows) or len(fronts) != len(engines):
        raise ValueError('Student rows, seeds and partner frontends must cover every actor row')
    if not student_engine.pose or any(engine.pose for engine in engines):
        raise ValueError('Use a pose student and audio-only conversational partners')
    if teacher_replay not in ('block', 'sequential') or profile_calls < 0:
        raise ValueError('Unknown replay backend or negative profile count')
    devices = list(student_engine.sdevs)
    for engine in engines:
        if set(devices).intersection(engine.sdevs):
            raise ValueError('Student and partner replicas must own disjoint GPU sets')
        devices.extend(engine.sdevs)
    frame_counts = {int(row['conversation_frames']) for row in rows}
    if len(frame_counts) != 1:
        raise ValueError('Partner fanout requires a fixed physical row length')
    frames = frame_counts.pop()
    capacity = frames if capacity_frames is None else int(capacity_frames)
    if capacity < frames:
        raise ValueError('Recording capacity cannot be shorter than the measured row')
    prefills = [int(row['prefill_frames']) for row in rows]
    if any(int(row['partner']['prefill_frames']) != count for row, count in zip(rows, prefills)):
        raise ValueError('Both directions must share each row prefill boundary')
    for row, count in zip(rows, prefills):
        _validate_seed(row, count)
    first = min(prefills)
    if first + 8 >= frames:
        raise ValueError('Rollout length must leave room for the partner bootstrap')
    bootstrap = bootstrap_plan(first, timing)
    steady_plan = plan_for_student(first + 1, first, timing)
    f0 = bootstrap.teacher_prefill_stop_model_frame
    student_profile = _Profile(profile_calls)
    pool, partners = None, []
    setup_locked = False
    try:
        _SETUP_LOCK.acquire()
        setup_locked = True
        instances = _instances(student_engine, rows)
        forcing = ForcedPrefix(student_engine, rows, frames)
        heard = AudioTimeline([row['wav'] for row in rows], capacity, student_engine.d0,
            inverse_gains=[row['partner']['output_audio_gain']['inverse_gain'] for row in rows],
            delivery_delay_samples=delivery_delay_samples)
        sf = StreamingFront(student_front, student_engine, heard, fa=student_engine.fa,
            fused_ops=runtime.M.FO, frames=capacity, graphs=graphs)
        mimi = MimiModel.from_pretrained(str(mimi_path), local_files_only=True).to(student_engine.d0).eval()
        sd = StreamingDecoder(mimi, len(rows), student_engine.d0, graphs=graphs)
        combined = torch.empty(len(rows), 1920, device=student_engine.d0, dtype=torch.float32)
        student_stream = torch.cuda.current_stream(student_engine.d0)
        for physical in range(f0 - CODE_DELAY):
            codes = np.stack([row['depth'][1, int(np.asarray(row['P']).reshape(-1)[0])
                                            + physical + CODE_DELAY] for row in rows])
            sd.decode(physical, torch.as_tensor(codes, device=student_engine.dl))
        for engine, front, part in zip(engines, fronts, parts, strict=True):
            selected = [dict(row['partner'],
                             _student_inverse_gain=row['output_audio_gain']['inverse_gain'])
                        for row in rows[part]]
            partners.append(_Partner(engine, selected, seeds[part], front, mimi_path,
                frames=frames, capacity=capacity, graphs=graphs, f0=f0,
                fused_ops=runtime.M.FO, profile_calls=profile_calls,
                delivery_delay_samples=delivery_delay_samples))
        # Clones retain every bootstrap frame until copies finish; steady state
        # instead protects the reusable decoder outputs with input-ready events.
        bootstrap_waves = []
        for frame in range(f0, bootstrap.teacher_backbone_stop_model_frame + 1):
            outputs = [partner.bootstrap_frame(frame, bootstrap.teacher_backbone_stop_model_frame)
                       for partner in partners]
            bootstrap_waves.extend(wave for wave, _ in outputs)
            for _, ready in outputs:
                student_stream.wait_event(ready)
            heard.append(frame - CODE_DELAY, _merge_audio(combined, parts, outputs))
        for frame in range(f0):
            sf.step(frame)
        student_engine.start(instances, seeds, params=dict(NATURAL_SAMPLING), f0=f0)
        for frame in range(f0, first + CODE_DELAY):
            forcing.sample(frame, graphs)
            wave = sd.decode(frame - CODE_DELAY, _codes(student_engine, frame))
            ready = torch.cuda.Event()
            ready.record(student_stream)
            for partner, part in zip(partners, parts, strict=True):
                consumed = partner.ingest(frame - CODE_DELAY, wave[part], ready)
                student_stream.wait_event(consumed)
            sf.step(frame)
            student_engine.advance_backbone(frame, graphs=graphs)
        for device in devices:
            torch.cuda.synchronize(device)
        del bootstrap_waves
        bootstrap_seconds = time.monotonic() - started
        generation_started = time.monotonic()
        warm_started = None
        if concurrent and len(partners) > 1:
            pool = ThreadPoolExecutor(max_workers=len(partners), thread_name_prefix='opd-partner')
        for index in range(first, frames):
            frame = index + CODE_DELAY
            forcing.sample(frame, graphs)
            wave = student_profile.run('student_mimi', sd.device,
                                       lambda: sd.decode(index, _codes(student_engine, frame)))
            ready = torch.cuda.Event()
            ready.record(student_stream)
            plan = plan_for_student(index + 1, first, timing) if frame < frames else None
            def refresh(value):
                partner, part = value
                return partner.refresh(index, wave[part], ready, plan, teacher_replay)
            # First real refresh builds lazy frontend/block graphs serially.
            outputs = _dispatch(None if index == first else pool, refresh, zip(partners, parts))
            for _, done in outputs:
                student_stream.wait_event(done)
            if plan is not None:
                heard.append(plan.teacher_new_audio_frame, _merge_audio(combined, parts, outputs))
                student_profile.run('student_voxtral', sf.device, lambda: sf.step(frame))
            student_engine.advance_backbone(frame, graphs=graphs)
            if index == first:
                for device in devices:
                    torch.cuda.synchronize(device)
                _SETUP_LOCK.release()
                setup_locked = False
                if startup_barrier is not None:
                    startup_barrier.wait(timeout=180)
                warm_started = time.monotonic()
            if progress and ((index - first + 1) % 25 == 0 or index + 1 == frames):
                progress(current_frame=index + 1, total_frames=frames,
                    seconds_elapsed=time.monotonic() - generation_started,
                    frame_clock='physical', generated_start_frame=first,
                    actor_group_gpus=len(devices), partner_replicas=len(partners))
        for device in devices:
            torch.cuda.synchronize(device)
        finished = time.monotonic()
        outputs = [_collect_rollout(student_engine, row, int(row['prefill_frames']), instance=index)
                   for index, row in enumerate(rows)]
        partner_stop = timing.partner_frame_for_output(frames - CODE_DELAY) + CODE_DELAY + 1
        partner_outputs = [partner.engine.readback(index,
            int(np.asarray(row['P']).reshape(-1)[0]) + partner_stop)
            for partner in partners for index, row in enumerate(partner.rows)]
        return {'rollouts': outputs, 'latents': [sf.recorded(index, frames) for index in range(len(rows))],
            'waveforms': [heard.waveform(index, frames) for index in range(len(rows))],
            'partner_rollouts': partner_outputs, 'delivery_delay_samples': delivery_delay_samples,
            'timings': {'bootstrap_seconds': bootstrap_seconds,
                'generation_seconds': finished - generation_started,
                'warm_generation_seconds': finished - warm_started,
                'warm_student_frames': frames - first - 1,
                'total_seconds': time.monotonic() - started,
                'audio_stages': student_profile.result(),
                'partner_audio_stages': [partner.profile.result() for partner in partners],
                'student_frames': frames - first,
                'teacher_backbone_per_refresh': len(steady_plan.teacher_backbone),
                'teacher_forced_tower_per_refresh': len(steady_plan.teacher_forced_tower),
                'teacher_replay_backend': teacher_replay,
                'student_front_cache': sf.cache_report(),
                'partner_front_caches': [partner.front.cache_report() for partner in partners],
                'teacher_fresh_light_per_refresh': 1, 'added_audio_delay_samples': delivery_delay_samples,
                'delay_applies_from_sample': 0,
                'partner_replicas': len(partners), 'partner_batches': [engine.NI for engine in engines],
                'concurrent_partners': concurrent, 'recording_capacity_frames': capacity}}
    except BaseException:
        if startup_barrier is not None:
            startup_barrier.abort()
        raise
    finally:
        if setup_locked:
            _SETUP_LOCK.release()
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
