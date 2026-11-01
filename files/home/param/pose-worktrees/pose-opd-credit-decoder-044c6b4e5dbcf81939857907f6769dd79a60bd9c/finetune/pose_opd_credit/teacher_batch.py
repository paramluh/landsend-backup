"""Frozen teacher batches reused by independent student replay microbatches."""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from types import MappingProxyType

import torch


def pack_cfg_batch(conditional, unconditional):
    """Pack complete conditional rows followed by complete unconditional rows.

    Segment boundaries and position IDs are duplicated, not extended: no token
    in either branch can attend to the other branch or a neighboring row.
    """
    length = sum(conditional['segs'])
    if unconditional['segs'] != conditional['segs']:
        raise ValueError('CFG branches must have identical row boundaries')
    dim_one = {'ids', 'depth', 'present', 'pid', 'vlat'}
    positions = {'sel', 'text_positions', 'light', 'plight', 'vpos'}
    packed = {}
    for key, value in conditional.items():
        other = unconditional[key]
        if key == 'segs':
            packed[key] = list(value) + list(other)
        elif torch.is_tensor(value):
            packed[key] = torch.cat((value, other + length if key in positions else other),
                                    dim=1 if key in dim_one else 0)
        else:
            packed[key] = value
    return packed


def split_pose_outputs(outputs, length):
    conditional, unconditional = {'text': {}}, {'text': {}}
    for key, value in outputs.items():
        if key != 'text':
            conditional[key], unconditional[key] = value[:length], value[length:]
    return conditional, unconditional


def pose_temporal(teacher, batch, rot, config, stage):
    from . import runtime as R
    if config.pose_teacher_cfg_scale == 1:
        with stage('pose_teacher_conditional_temporal'):
            return teacher(batch, rot, role='pose_teacher'), None
    silence = R.load_silence_codes(config.pose_teacher_silence_path, config.pose_teacher_silence_sha256)
    silent = R.silent_pose_view(batch, silence)
    if config.pose_cfg_batch:
        with stage('pose_teacher_cfg_packed_temporal'):
            both = teacher(pack_cfg_batch(batch, silent), rot, role='pose_teacher')
            pose, unconditional = split_pose_outputs(both, sum(batch['segs']))
    else:
        with stage('pose_teacher_conditional_temporal'):
            pose = teacher(batch, rot, role='pose_teacher')
        with stage('pose_teacher_unconditional_temporal'):
            unconditional = teacher(silent, rot, role='pose_teacher')
    pose['logits_p'] = R.guided_pose_logits(pose['logits_p'], unconditional['logits_p'], config.pose_teacher_cfg_scale)
    del unconditional['logits_p']
    return pose, unconditional


def teacher_depth_logits(key, decoder, conditional, unconditional, codes, flat, config):
    from . import runtime as R
    if key == 'p' and unconditional is not None and config.pose_cfg_batch:
        # Gather only this bounded chunk before concatenating branches. Both
        # branches consume the exact same already sampled pose code prefix.
        hv = torch.cat((conditional['hv_p'][flat], unconditional['hv_p'][flat]))
        zb = torch.cat((conditional['zb_p'][flat], unconditional['zb_p'][flat]))
        selected = codes[flat]
        packed_codes = torch.cat((selected, selected))
        logits, _ = decoder(hv, packed_codes, torch.arange(len(hv), device=flat.device), zb=zb)
        conditional_logits, unconditional_logits = logits.chunk(2)
        return R.guided_pose_logits(conditional_logits, unconditional_logits, config.pose_teacher_cfg_scale)
    logits, _ = decoder(conditional['hv_' + key], codes, flat, zb=conditional.get('zb_' + key))
    if key == 'p' and unconditional is not None:
        other, _ = decoder(unconditional['hv_p'], codes, flat, zb=unconditional.get('zb_p'))
        logits = R.guided_pose_logits(logits, other, config.pose_teacher_cfg_scale)
    return logits


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    return value.detach() if torch.is_tensor(value) else value


def _slice_outputs(value, start, end, text_start, text_end):
    return _freeze({key: ({name: tensor if name in ('W', 'R') else tensor[text_start:text_end]
                          for name, tensor in item.items()} if key == 'text' else item[start:end])
                    for key, item in value.items()})


def _merge_outputs(values):
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return {key: ({name: items[0] if name in ('W', 'R') else torch.cat(items)
                   for name in values[0][key]
                   for items in [[item[key][name] for item in values]]} if key == 'text'
                  else torch.cat([item[key] for item in values])) for key in values[0]}


@dataclass(frozen=True)
class TeacherSlice:
    audio: object
    pose: object
    unconditional: object
    cache: object
    cache_ranges: object
    segments: tuple[int, ...]

    def depth_chunk(self, key, index, size, device):
        """Materialize only one FP32 chunk, repeating the valid first frame for padding."""
        start, end = self.cache_ranges[key]
        cache = self.cache[key]
        begin = min(start + index * size, end)
        count = min(size, end - begin)
        if count == size:
            return cache[begin:begin + size].to(device, non_blocking=device.type == 'cuda')
        chunk = torch.empty((size, *cache.shape[1:]), dtype=cache.dtype,
                            device='cpu', pin_memory=device.type == 'cuda')
        chunk.copy_(cache[start:start + 1].expand_as(chunk))
        if count:
            chunk[:count].copy_(cache[begin:begin + count])
        return chunk.to(device, non_blocking=device.type == 'cuda')


@dataclass(frozen=True)
class PreparedTeachers:
    audio: object
    pose: object
    unconditional: object
    cache: object
    segments: tuple[int, ...]
    text_counts: tuple[int, ...]
    media_counts: object

    def slice_rows(self, begin, end):
        if not 0 <= begin < end <= len(self.segments):
            raise ValueError('Teacher slice must select complete contiguous rows')
        start, stop = sum(self.segments[:begin]), sum(self.segments[:end])
        ts, te = sum(self.text_counts[:begin]), sum(self.text_counts[:end])
        slice_value = lambda value: _slice_outputs(value, start, stop, ts, te) if value is not None else None
        ranges = {key: (sum(counts[:begin]), sum(counts[:end])) for key, counts in self.media_counts.items()}
        return TeacherSlice(slice_value(self.audio), slice_value(self.pose), slice_value(self.unconditional),
                            self.cache, MappingProxyType(ranges), self.segments[begin:end])


def prepare_teacher_update(audio_teacher, pose_teacher, bundles, latents, config, rot, device, *,
                           world=1, normalization=None, stage_profiler=None):
    """Score frozen teachers once, before any student temporal graph exists.

    Teacher row batches are independent of learner physical microbatches. With
    CPU caching, all guided depth logits are computed once and stored FP32;
    subsequent student passes do not gather or execute a teacher decoder.
    """
    from . import runtime as R
    if not bundles or len(bundles) != len(latents):
        raise ValueError('Teacher preparation requires matching nonempty rows/features')
    bundles = [R.pad_replay_bundle(bundle, config.fixed_sequence_frames) for bundle in bundles]
    batch_size = config.teacher_batch_per_rank or config.effective_batch_per_rank
    if len(bundles) % batch_size:
        raise ValueError('Teacher batch must divide this complete effective batch')
    stage = stage_profiler.span if stage_profiler is not None else lambda _: contextlib.nullcontext()
    segments = tuple(len(row['student']['pid']) for row in bundles)
    text_counts = tuple(len(row['student']['text_positions']) for row in bundles)
    media_counts = {key: tuple(len(row['student'][positions]) for row in bundles)
                    for key, positions in (('s', 'light'), ('p', 'plight'))}
    cached = {} if config.teacher_depth_cache == 'cpu' else None
    outputs = {'audio': [], 'pose': [], 'unconditional': []}
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == 'cuda', dtype=torch.bfloat16):
        for begin in range(0, len(bundles), batch_size):
            end = begin + batch_size
            with stage('teacher_prepare'):
                batch, _, _ = R.prepare(bundles[begin:end], latents[begin:end], config, device,
                                        world=world, normalization=normalization)
            with stage('audio_teacher_temporal'):
                audio = audio_teacher(batch['audio_teacher'], rot, role='audio_teacher')
            pose, unconditional = pose_temporal(pose_teacher, batch['pose_teacher'], rot, config, stage)
            if cached is not None:
                s = batch['student']
                for key, model, out, codes, positions in (
                        ('s', audio_teacher, audio, s['depth'][1], s['light']),
                        ('p', pose_teacher, pose, s['pdepth'], s['plight'])):
                    decoder = getattr(model, 'dec_' + key)
                    cursor = sum(media_counts[key][:begin])
                    with stage('teacher_depth_audio' if key == 's' else 'teacher_depth_pose'), R.resident_decoders(decoder):
                        for flat, valid in R.chunk_indices(positions, config.decoder_chunk_frames):
                            logits = teacher_depth_logits(key, decoder, out, unconditional, codes, flat, config)
                            if logits.dtype != torch.float32:
                                raise RuntimeError('Teacher cache requires exact FP32 logits')
                            if key not in cached:
                                cached[key] = torch.empty((sum(media_counts[key]), *logits.shape[1:]),
                                    dtype=torch.float32, device='cpu', pin_memory=device.type == 'cuda')
                            if valid:
                                cached[key][cursor:cursor + valid].copy_(logits[:valid], non_blocking=logits.is_cuda)
                                cursor += valid
                # Depth contexts are no longer needed after their exact guided
                # logits have been cached. Retain only CB0/text scoring data.
                del audio['hv_s'], pose['hv_p'], pose['zb_p']
                unconditional = None
            outputs['audio'].append(audio)
            outputs['pose'].append(pose)
            if unconditional is not None:
                outputs['unconditional'].append(unconditional)
            del batch
        # Host cache reads must not race the final asynchronous device-to-host
        # copy. One synchronization per teacher update, never per depth chunk.
        if cached is not None and device.type == 'cuda':
            torch.cuda.current_stream(device).synchronize()
    return PreparedTeachers(_freeze(_merge_outputs(outputs['audio'])), _freeze(_merge_outputs(outputs['pose'])),
                            _freeze(_merge_outputs(outputs['unconditional'])),
                            None if cached is None else MappingProxyType(cached), segments, text_counts,
                            MappingProxyType(media_counts))
