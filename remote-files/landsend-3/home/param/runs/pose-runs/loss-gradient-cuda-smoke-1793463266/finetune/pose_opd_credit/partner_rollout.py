"""Explicit PCM delivery timing with a frozen, silence-speculating partner.

Published partner audio is irrevocable. As each real student chunk arrives, the
partner reconditions its bounded suffix on that chunk, forcing old outputs.
Only the newly needed teacher frame samples its light decoder. Student audio
features are recorded exactly as consumed, and are the learner's replay input.
"""
from __future__ import annotations

import time
import numpy as np
import torch

from .partner_audio import AudioTimeline, StreamingFront, StreamingDecoder
from .rollout import CODE_DELAY, NATURAL_SAMPLING, PAD, _collect_rollout, _validate_seed
from .teacher_partner_schedule import AudioTiming, bootstrap_plan, plan_for_student


def _instances(engine, rows, extra=0):
    out = []
    for row in rows:
        inst = {key: np.asarray(row[key]).copy() for key in ('ids', 'depth', 'present', 'pdepth')}
        if extra:
            for key, fill in (('ids', PAD), ('depth', -1), ('present', 0), ('pdepth', -1)):
                value = inst[key]
                axis = 0 if key == 'pdepth' else 1
                widths = [(0, 0)] * value.ndim
                widths[axis] = (0, extra)
                inst[key] = np.pad(value, widths, constant_values=fill)
            inst['present'][1, -extra:] = 1
        inst.update(P=int(np.asarray(row['P']).reshape(-1)[0]), a_mode='file')
        out.append(inst)
    prefixes = torch.as_tensor([i['P'] for i in out], dtype=torch.long)
    for device in engine.sdevs:
        engine.Pv[device].copy_(prefixes)
    engine._opd_audio.zero_()
    return out


class ForcedPrefix:
    """GPU copies of only the permitted forced values, with per-row cutoffs."""
    def __init__(self, engine, rows, frames):
        self.engine, self.frames = engine, frames
        self.prefills = np.asarray([int(r['prefill_frames']) for r in rows])
        self.max_prefill = int(self.prefills.max())
        values = {'lane': [], 'codes': [], 'pose': []}
        for row in rows:
            p = int(np.asarray(row['P']).reshape(-1)[0])
            stop = p + self.max_prefill + CODE_DELAY
            values['lane'].append(np.asarray(row['ids'])[1, p:stop])
            values['codes'].append(np.asarray(row['depth'])[1, p:stop])
            if engine.pose:
                values['pose'].append(np.asarray(row['pdepth'])[p:stop])
        self.values = {k: torch.as_tensor(np.stack(v), device=engine.dp if k == 'pose' else engine.dl)
                       for k, v in values.items() if v}
        self.prefills_gpu = torch.as_tensor(self.prefills, device=engine.dl)

    def sample(self, frame, graphs, *, tail_pad=True):
        engine = self.engine
        force = {}
        if frame < self.max_prefill:
            force['lane'] = torch.where(frame < self.prefills_gpu, self.values['lane'][:, frame], -1)
        elif tail_pad and frame >= self.frames:
            force['lane'] = torch.full((engine.NI,), PAD, device=engine.dl, dtype=torch.long)
        forced_media = frame < self.prefills + CODE_DELAY
        all_forced = bool(forced_media.all())
        mixed = bool(forced_media.any()) and not all_forced
        if all_forced:
            for key in ('codes', 'pose'):
                if key in self.values:
                    force[key] = self.values[key][:, frame]
        engine.sample_outputs(frame, graphs=graphs, force=force or None)
        if mixed:
            engine.force_media_rows(frame, codes=self.values['codes'][:, frame],
                                    pose=self.values.get('pose', None)[:, frame] if engine.pose else None,
                                    mask=torch.as_tensor(forced_media, device=engine.dl))


def _committed(engine, frame):
    """Read existing tokens without moving them through the host."""
    positions = engine.Pv[engine.dl][:engine.NI] + frame
    rows = engine.arI[engine.dl]
    return {'lane': engine.laneB[rows, positions], 'codes': engine.depB[rows, positions]}


def _codes(engine, frame):
    return engine.depB[engine.arI[engine.dl], engine.Pv[engine.dl][:engine.NI] + frame]


class _AudioProfile:
    def __init__(self, samples=32, warmup=25):
        self.samples, self.warmup, self.events, self.calls = samples, warmup, {}, {}
        self.host_seconds, self.devices, self.warm_host = {}, {}, {}

    def _submit(self, name, fn):
        started = time.perf_counter()
        try:
            return fn()
        finally:
            self.host_seconds[name] = self.host_seconds.get(name, 0.) + time.perf_counter() - started

    def run(self, name, device, fn):
        if not self.samples:
            return fn()
        count = self.calls.get(name, 0)
        self.calls[name] = count + 1
        self.devices[name] = str(device)
        if count < self.warmup or count >= self.warmup + self.samples:
            return self._submit(name, fn)
        with torch.cuda.device(device):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            began = time.perf_counter()
            value = self._submit(name, fn)
            self.warm_host.setdefault(name, []).append(1000 * (time.perf_counter() - began))
            b.record()
        self.events.setdefault(name, []).append((a, b))
        return value

    def result(self):
        return {name: {'calls': self.calls[name], 'timed_calls': len(pairs), 'warmup_calls_excluded': self.warmup,
                       'host_submit_seconds': self.host_seconds[name],
                       'host_submit_ms_mean_all_calls': 1000 * self.host_seconds[name] / self.calls[name],
                       'host_warm_submit_ms': {'mean': float(np.mean(self.warm_host[name])),
                           'median': float(np.median(self.warm_host[name])),
                           'p90': float(np.percentile(self.warm_host[name], 90))},
                       'gpu_event_device': self.devices[name],
                       'gpu_timing_scope': 'primary-stream span including cross-device dependency waits; '
                                           'not exclusive kernel time or additive across stages',
                       'gpu_ms_mean': sum(a.elapsed_time(b) for a, b in pairs) / len(pairs),
                       'gpu_ms_median': float(np.median([a.elapsed_time(b) for a, b in pairs])),
                       'gpu_ms_p90': float(np.percentile([a.elapsed_time(b) for a, b in pairs], 90))}
                for name, pairs in self.events.items()}


@torch.no_grad()
def rollout_teacher_partner(student_engine, partner_engine, rows, seeds, student_front, partner_front,
                            mimi_path, *, graphs=True, progress=None, teacher_replay='sequential', verify_block=False,
                            profile_calls=32, profile_warmup=25, mimi_placement='frontend',
                            delivery_delay_samples=0):
    """Generate a static batch; each row independently chooses its prefill.

    student_front/partner_front are native VoxtralFront objects. Both actors have
    already loaded their weights and optionally captured their standard graphs.
    The fixed learner timeline retains the existing two terminal flush slots:
    input features there are zero, exactly as in ordinary OPD replay.
    """
    from transformers import MimiModel
    from . import runtime

    if teacher_replay not in ('sequential', 'block'):
        raise ValueError('Unknown partner replay backend')
    if mimi_placement not in ('frontend', 'audio', 'student-audio'):
        raise ValueError('Unknown streaming Mimi device placement')
    timing = AudioTiming(delivery_delay_samples=delivery_delay_samples)

    started = time.monotonic()
    rows, seeds = list(rows), list(map(int, seeds))
    if not rows or len(rows) != len(seeds) or student_engine.NI != len(rows) or partner_engine.NI != len(rows):
        raise ValueError('Student and partner batches must correspond one-to-one')
    if not student_engine.pose or partner_engine.pose:
        raise ValueError('Use the pose student and the selected audio-only conversational teacher')
    frames = {int(r['conversation_frames']) for r in rows}
    if len(frames) != 1:
        raise ValueError('Teacher-partner rollouts require a fixed physical row length')
    frames = frames.pop()
    partners = [r['partner'] for r in rows]
    prefills = [int(r['prefill_frames']) for r in rows]
    if any(int(p['prefill_frames']) != n for p, n in zip(partners, prefills)):
        raise ValueError('Both directions must share each row\'s prefill boundary')
    for row in rows:
        _validate_seed(row, int(row['prefill_frames']))
    first = min(prefills)
    if first + 8 >= frames:
        raise ValueError('Benchmark/training length must leave room for the partner bootstrap')
    bootstrap = bootstrap_plan(first, timing)
    steady_plan = plan_for_student(first + 1, first, timing)
    f0 = bootstrap.teacher_prefill_stop_model_frame
    student_instances = _instances(student_engine, rows)
    partner_instances = _instances(partner_engine, partners, extra=8)
    student_forcing = ForcedPrefix(student_engine, rows, frames)
    partner_forcing = ForcedPrefix(partner_engine, partners, frames)

    # A row's wav is INCOMING raw PCM. Its opposite row supplies the gain of
    # the speaker generating that stream, rather than the receiver's gain.
    heard_by_student = AudioTimeline([r['wav'] for r in rows], frames, student_engine.d0,
        inverse_gains=[p['output_audio_gain']['inverse_gain'] for p in partners],
        delivery_delay_samples=delivery_delay_samples)
    heard_by_partner = AudioTimeline([p['wav'] for p in partners], frames, partner_engine.d0,
        inverse_gains=[r['output_audio_gain']['inverse_gain'] for r in rows],
        delivery_delay_samples=delivery_delay_samples)
    sf = StreamingFront(student_front, student_engine, heard_by_student, fa=student_engine.fa,
                        fused_ops=runtime.M.FO, frames=frames, graphs=graphs)
    tf = StreamingFront(partner_front, partner_engine, heard_by_partner, fa=partner_engine.fa,
                        fused_ops=runtime.M.FO, frames=frames, graphs=graphs)
    # The student's first device also runs its independent pose decoder. Audio
    # PCM can instead be decoded on the audio device, allowing the partner to
    # begin its refresh while the student's pose branch finishes.
    student_mimi_device = student_engine.d0 if mimi_placement == 'frontend' else student_engine.dl
    partner_mimi_device = partner_engine.dl if mimi_placement == 'audio' else partner_engine.d0
    mimi_student = MimiModel.from_pretrained(str(mimi_path), local_files_only=True).to(student_mimi_device).eval()
    mimi_partner = MimiModel.from_pretrained(str(mimi_path), local_files_only=True).to(partner_mimi_device).eval()
    sd = StreamingDecoder(mimi_student, len(rows), student_mimi_device, graphs=graphs)
    td = StreamingDecoder(mimi_partner, len(rows), partner_mimi_device, graphs=graphs)
    for physical in range(f0 - CODE_DELAY):
        for engine, decoder, selected in ((student_engine, sd, rows), (partner_engine, td, partners)):
            codes = np.stack([r['depth'][1, int(np.asarray(r['P']).reshape(-1)[0]) + physical + CODE_DELAY]
                              for r in selected])
            decoder.decode(physical, torch.as_tensor(codes, device=engine.dl))
    for f in range(f0):
        tf.step(f, speculative=True)
    partner_engine.start(partner_instances, [s ^ 0x5A17B39D for s in seeds], params=dict(NATURAL_SAMPLING), f0=f0)
    # A previous batch deliberately ends with the last committed teacher draw
    # awaiting a backbone update. start() resets that state before configuring
    # the rewind ring for this independent conversation batch.
    if partner_engine._opd_rewind_window < 8:
        partner_engine.enable_rewind(8)
    for f in range(f0, bootstrap.teacher_backbone_stop_model_frame + 1):
        partner_forcing.sample(f, graphs, tail_pad=False)
        wave = td.decode(f - CODE_DELAY, _codes(partner_engine, f))
        heard_by_student.append(f - CODE_DELAY, wave)
        if f < bootstrap.teacher_backbone_stop_model_frame:
            tf.step(f, speculative=True)
            partner_engine.advance_backbone(f, graphs=graphs)
    for f in range(f0):
        sf.step(f)
    student_engine.start(student_instances, seeds, params=dict(NATURAL_SAMPLING), f0=f0)
    for f in range(f0, first + CODE_DELAY):
        student_forcing.sample(f, graphs)
        wave = sd.decode(f - CODE_DELAY, _codes(student_engine, f))
        heard_by_partner.append(f - CODE_DELAY, wave)
        sf.step(f)
        student_engine.advance_backbone(f, graphs=graphs)
    for device in set(student_engine.sdevs + partner_engine.sdevs):
        torch.cuda.synchronize(device)
    bootstrap_seconds = time.monotonic() - started
    generated_started = time.monotonic()
    profile = _AudioProfile(profile_calls, warmup=profile_warmup)
    block_verification = None
    for i in range(first, frames):
        f = i + CODE_DELAY
        student_forcing.sample(f, graphs)
        wave = profile.run('student_mimi', sd.device, lambda: sd.decode(i, _codes(student_engine, f)))
        profile.run('student_to_partner', heard_by_partner.device, lambda: heard_by_partner.append(i, wave))
        if f < frames:
            plan = plan_for_student(i + 1, first, timing)
            start = plan.teacher_backbone.start
            partner_engine.rewind(start)
            tf.rewind(start)
            if teacher_replay == 'block' or (verify_block and block_verification is None):
                from .partner_block import replay_committed_block
                profile.run('partner_voxtral_block', tf.device,
                            lambda: tf.step_block(start, len(plan.teacher_backbone), speculative=True))
                if verify_block and block_verification is None:
                    from .partner_block import verify_block_replay
                    block_verification = verify_block_replay(partner_engine, start,
                                                            plan.teacher_backbone.stop, graphs=graphs)
                if teacher_replay == 'block':
                    profile.run('partner_committed_block', partner_engine.dl,
                                lambda: replay_committed_block(partner_engine, start,
                                                               plan.teacher_backbone.stop, graphs=graphs))
                else:
                    for t in plan.teacher_backbone.frames():
                        old = _committed(partner_engine, t)
                        if t != start:
                            partner_engine.sample_outputs(t, graphs=graphs, force=old, sample_text=False)
                        partner_engine.advance_backbone(t, graphs=graphs, force=old if t == start else None)
            else:
                for t in plan.teacher_backbone.frames():
                    old = _committed(partner_engine, t)
                    if t != start:
                        partner_engine.sample_outputs(t, graphs=graphs, force=old, sample_text=False)
                    profile.run('partner_voxtral', tf.device, lambda t=t: tf.step(t, speculative=True))
                    partner_engine.advance_backbone(t, graphs=graphs, force=old if t == start else None)
            t = plan.teacher_sample_model_frame
            partner_forcing.sample(t, graphs, tail_pad=False)
            wave = profile.run('partner_mimi', td.device,
                               lambda: td.decode(plan.teacher_new_audio_frame, _codes(partner_engine, t)))
            profile.run('partner_to_student', heard_by_student.device,
                        lambda: heard_by_student.append(plan.teacher_new_audio_frame, wave))
            profile.run('student_voxtral', sf.device, lambda: sf.step(f))
        student_engine.advance_backbone(f, graphs=graphs)
        if progress and ((i - first + 1) % 25 == 0 or i + 1 == frames):
            progress(current_frame=i + 1, total_frames=frames,
                     seconds_elapsed=time.monotonic() - generated_started,
                     frame_clock='physical', generated_start_frame=first,
                     actor_group_gpus=4)
    for device in set(student_engine.sdevs + partner_engine.sdevs):
        torch.cuda.synchronize(device)
    generated_seconds = time.monotonic() - generated_started
    outputs = [_collect_rollout(student_engine, row, int(row['prefill_frames']), instance=i)
               for i, row in enumerate(rows)]
    partner_stop = timing.partner_frame_for_output(frames - CODE_DELAY) + CODE_DELAY + 1
    partner_outputs = [partner_engine.readback(i, int(np.asarray(row['P']).reshape(-1)[0]) + partner_stop)
                       for i, row in enumerate(partners)]
    return {'rollouts': outputs, 'latents': [sf.recorded(i, frames) for i in range(len(rows))],
            'waveforms': [heard_by_student.waveform(i, frames) for i in range(len(rows))],
            'partner_rollouts': partner_outputs, 'delivery_delay_samples': delivery_delay_samples,
            'timings': {'bootstrap_seconds': bootstrap_seconds, 'generation_seconds': generated_seconds,
                        'total_seconds': time.monotonic() - started, 'audio_stages': profile.result(),
                        'student_frames': frames - first,
                        'teacher_backbone_per_refresh': len(steady_plan.teacher_backbone),
                        'teacher_forced_tower_per_refresh': len(steady_plan.teacher_forced_tower),
                        'teacher_replay_backend': teacher_replay,
                        'student_front_cache': sf.cache_report(), 'partner_front_cache': tf.cache_report(),
                        'block_verification': block_verification,
                        'teacher_fresh_light_per_refresh': 1,
                        'added_audio_delay_samples': delivery_delay_samples,
                        'delay_applies_from_sample': 0}}
