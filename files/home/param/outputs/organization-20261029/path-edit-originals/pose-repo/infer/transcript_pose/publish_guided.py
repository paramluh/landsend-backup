"""Publish validated audio-guided comparisons without replacing older galleries."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
import time

import comparison_video as video
from request_worker import validate_video_receipt

ROOT = Path('/home/param/pose-runs/audio-guided-native-comparison-20261028')
PREFILL_ROOT = Path('/home/param/pose-runs/audio-guided-prefill2s-comparison-20261028')
HOST = 'landsend-1'
RECIPES = ('aligned-media', 'compact-all', 'xabi-original')
_STATUS_LOCK = threading.Lock()
_TRANSFER_SLOTS = threading.BoundedSemaphore(2)


def locations(job):
    root = Path(job.get('root', job.get('run_root', ROOT))).resolve()
    directory = Path(job.get('directory', job.get('out', ''))).resolve()
    if root not in (ROOT, PREFILL_ROOT):
        raise ValueError('Guided publication must use its separate comparison root')
    cid = job['clip']['id'] if 'clip' in job else job['clip_id']
    clip = next(c for c in video.read(root / 'selection.json')['clips'] if c['id'] == cid)
    recipe = job['recipe']
    if recipe not in RECIPES or directory != root / 'outputs' / clip['gallery_id'] / recipe:
        raise ValueError('Guided publication clip/model path mismatch')
    return root, directory, clip, recipe


def validate_guidance(ppm, recipe, clip):
    """Check the actual inference receipt, rather than the planned settings."""
    if ppm['recipe'] != recipe or ppm.get('force_text') is not (recipe == 'aligned-media'):
        raise ValueError('Framewise text conditioning differs from the model native format')
    if recipe != 'aligned-media' and ppm.get('text_mode') != 'native':
        raise ValueError('Compact/Xabi framewise text must be sampled')
    expected = dict(text_temp=1, cb0_temp=1, light_temp=1, pose_temp=1, pose_light_temp=1,
                    text_topk=0, cb0_topk=30, light_topk=30, pose_topk=30, pose_light_topk=50,
                    pose_cfg_on=1, pose_cfg_s=4, pose_cfg_cb0=1, pose_cfg_light=1, pose_cfg_speaking_only=0)
    if any(ppm['sampling'].get(k) != v for k, v in expected.items()):
        raise ValueError('Sampling or pose-guidance parameters differ from this experiment')
    guidance = ppm.get('audio_guidance', {})
    expected_guidance = dict(enabled=True, strength=3.0, engine_scale=4.0,
                            formula='audio + strength * (audio - silence)', scope='all pose codebooks',
                            silence_source='Mimi encoding of digital silence',
                            full_model_silence_pass=True, shared_text_and_pose=True)
    if any(guidance.get(k) != v for k, v in expected_guidance.items()):
        raise ValueError('Missing or incorrect independent-silence guidance provenance')
    if not re.fullmatch(r'[a-f0-9]{64}', guidance.get('silence_codes_sha256', '')):
        raise ValueError('Missing encoded-silence hash')
    audit = guidance.get('audit', {})
    prefill = int(clip.get('prefill_frames', 0))
    expected_audit = dict(enabled=True, independent_cache_rows=True, d33_uses_separate_backbone_hidden=True,
                          codebooks=list(range(32)), guided_frames=ppm['frames'] - prefill,
                          conditional_weight=4, silence_weight=-3, speaking_only=False,
                          silence_codes_sha256=guidance['silence_codes_sha256'])
    if any(audit.get(k) != v for k, v in expected_audit.items()):
        raise ValueError('Full-model independent guidance audit is incomplete or inconsistent')
    if (ppm['ground_truth_generation_media'] or ppm['voice_clip_frames'] != 100 or ppm['prefill_frames'] != prefill
            or ppm['voice_codes_sha256'] != clip['voice_provenance']['codes_sha256']):
        raise ValueError('Unexpected generation media or voice conditioning')
    if prefill:
        if (prefill != 25 or ppm.get('f0') != 25 or ppm.get('ground_truth_prefill_media') is not True
                or set(ppm.get('prefill_modalities', [])) != {'audio', 'pose', 'text'}
                or ppm.get('prefill_duration_seconds') != 2 or ppm.get('prefill_sha256') != clip['prefill_sha256']):
            raise ValueError('Ground-truth prefill does not match the selected source audio, pose and text')
    supplied = ppm.get('supplied_transcript')
    if ((recipe == 'xabi-original' and supplied is not None)
            or (recipe != 'xabi-original' and (not isinstance(supplied, str) or not supplied.strip()))):
        raise ValueError('Upfront transcript does not match the model native format')
    return guidance


def initialize_gallery(root=ROOT):
    """Publish the common source conditioning and initialize 24 independent cells."""
    import numpy as np

    root = Path(root).resolve()
    if root not in (ROOT, PREFILL_ROOT):
        raise ValueError('Unknown guided comparison root')
    prefilling = root == PREFILL_ROOT
    namespace = 'guided-prefill' if prefilling else 'guided'
    gallery = root / 'gallery'
    selection = video.read(root / 'selection.json')
    video.execute([sys.executable, str(video.PUBLISH), '--data-root', str(gallery), 'init',
                   '--clips', str(root / 'selection.json'), '--mode', 'guided-native',
                   '--title', 'Audio-guided comparison · 2-second prefill' if prefilling else 'Audio-guided native comparison',
                   '--note', ('The first 2 seconds use source audio, pose and text. A brief white flash and beep mark the continuation at 2 seconds. ' if prefilling else '') +
                   'Eight clips × three models × 60 seconds. Each model uses its native text format and the same 8-second voice. Pose guidance is audio + 3 × (audio − silence). All temperatures 1; audio top-k 30, pose cb0 top-k 30, pose cb1–31 top-k 50.'])
    prior = video.read(Path('/home/param/pose-runs/controlled-comparison-20261028/gallery/manifest.json'))
    prior_clips = {c['id']: c for c in prior['clips']}
    extras = {}
    for clip in selection['clips']:
        cid = clip['id']
        lane = np.load(clip['lane'], allow_pickle=False)
        voice = np.load(clip['voice_clip'], allow_pickle=False)
        references = video.read(Path(clip['input_assets']) / 'reference-selection.json')
        public_refs = []
        media = gallery / 'media' / cid / 'conditioning'
        media.mkdir(parents=True, exist_ok=True)
        for row in references['rows']:
            source = Path(row['image_path'])
            digest = hashlib.sha256(source.read_bytes()).hexdigest()
            target = media / (digest[:20] + source.suffix.lower())
            shutil.copy2(source, target)
            public_refs.append(dict(property=row['property'], slot=row.get('slot'),
                                    image=f'/{namespace}-' + str(target.relative_to(gallery)), sha256=digest,
                                    source_timestamp_seconds=row['source_timestamp_s'],
                                    inside_evaluation_window=row['inside_evaluation_window']))
        payload = dict(clip_id=cid, source_transcript=clip['transcript'], fps=12.5, frames=752,
                       timed_words=video.read(clip['words']), framewise_text_token_ids=lane.tolist(),
                       framewise_lane_sha256=hashlib.sha256(np.ascontiguousarray(lane, dtype=np.int64).tobytes()).hexdigest(),
                       voice_reference_seconds=8, voice_codes=voice.tolist(),
                       voice_codes_sha256=clip['voice_provenance']['codes_sha256'],
                       aligned_transcript_prefix_ids=[151680, *lane.tolist(), 151681],
                       model_text_formats={
                           'aligned-media': 'Full aligned prefix plus this exact forced framewise text lane',
                           'compact-all': 'Full plain transcript upfront; generated framewise text',
                           'xabi-original': 'No transcript; generated framewise text'},
                       visual_references=public_refs)
        if prefilling:
            with np.load(clip['prefill_source'], allow_pickle=False) as ground_truth:
                payload['ground_truth_prefill'] = dict(frames=25, seconds=2, modalities=['audio', 'pose', 'text'],
                                                       sha256=clip['prefill_sha256'],
                                                       **{key: ground_truth[key].tolist() for key in ('audio_codes', 'pose_codes', 'text_lane')})
            payload['model_text_formats']['xabi-original'] = 'First 2 seconds of source text as prefill; no upfront transcript for the continuation'
        video.write(gallery / 'conditioning' / f'{cid}.json', payload)
        old_voice = prior_clips[cid]['voice_reference']
        source = Path('/home/param/pose-runs/controlled-comparison-20261028/gallery') / old_voice['audio'].lstrip('/')
        target = gallery / old_voice['audio'].lstrip('/')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        extras[cid] = dict(voice_reference=old_voice, conditioning_url=f'/{namespace}-conditioning/{cid}.json',
                           visual_references=public_refs)
    spec = importlib.util.spec_from_file_location('guided_gallery_publisher', video.PUBLISH)
    publisher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(publisher)
    with publisher.manifest_lock(gallery):
        manifest = video.read(gallery / 'manifest.json')
        for clip in manifest['clips']:
            clip.update(extras[clip['id']])
        if prefilling:
            for model in manifest['models']:
                if model['id'] == 'xabi-original':
                    model['description'] = 'Common voice plus 2-second source audio/pose/text prefill; no future transcript. The continuation is sampled.'
                else:
                    model['description'] += ' First 2 seconds of audio, pose and text are prefilled from the source.'
        manifest.update(updated_at=video.stamp(), request_studio_url='/', guided_comparison_available=True)
        publisher.atomic_json(gallery / 'manifest.json', manifest)
    return gallery


def publish_guided_state(job, stage, message=''):
    # A launch can have sixteen teams reporting together. Serialize progress
    # connections so sshd's unauthenticated-connection cap cannot block GPU work.
    with _STATUS_LOCK:
        return _publish_guided_state(job, stage, message)


def _publish_guided_state(job, stage, message=''):
    root, directory, clip, recipe = locations(job)
    if stage == 'complete':
        if video.complete(directory / 'video_state.json'):
            return
        raise ValueError('Completion must be published through publish_guided')
    state = 'failed' if stage == 'failed' else 'pending' if stage in ('pending', 'waiting') else 'generating'
    video.write(directory / 'video_state.json', dict(status=stage, message=message, updated_at=video.stamp()))
    args = ['--data-root', root / 'gallery', 'status', '--clip-id', clip['id'], '--model', recipe,
            '--status', state, '--message', message]
    video.execute([sys.executable, str(video.PUBLISH), *map(str, args)])
    for attempt in range(3):
        try:
            video.remote(HOST, ['python3', video.PUBLISH, *args])
            break
        except subprocess.CalledProcessError as exc:
            if exc.returncode != 255:
                raise
            if attempt == 2:
                print(f'Gallery progress update delayed for {clip["id"]}/{recipe}: SSH unavailable; local state retained', flush=True)
                return False
            time.sleep(0.5 * (attempt + 1))
    return True


def publish_guided(job):
    import numpy as np

    root, directory, clip, recipe = locations(job)
    namespace = 'guided-prefill' if root == PREFILL_ROOT else 'guided'
    pose_args = video.pose_views(directory)
    receipt = video.read(directory / 'VIDEO_COMPLETE.json')
    protocol = validate_video_receipt(receipt['diffusion_receipt'], 752, 2026)
    ppm = video.read(directory / 'ppm/meta.json')
    words = video.read(directory / 'ppm/words.json')
    complete = video.read(directory / 'ppm/PPM_COMPLETE.json')
    final = directory / 'video.mp4'
    if not (receipt['status'] == complete['status'] == 'complete' and receipt['frames'] == 1500
            and receipt['seconds'] == 60 and ppm['frames'] == 752
            and hashlib.sha256(final.read_bytes()).hexdigest() == receipt['sha256']):
        raise ValueError('Guided video is not a validated sixty-second result')
    guidance = validate_guidance(ppm, recipe, clip)
    prefill = int(ppm['prefill_frames'])
    marker = None
    if prefill:
        from prefill_marker import validate_prefill_marker
        marker = validate_prefill_marker(directory)
        if marker['boundary_seconds'] != 2 or not marker.get('protected_inputs_sha256'):
            raise ValueError('Missing validated presentation-only prefill marker')
    with np.load(directory / 'ppm/inputs.npz', allow_pickle=False) as inputs:
        prefix = int(inputs['P'])
        actual_conditioning = dict(recipe=recipe, clip_id=clip['id'], prefix_positions=prefix,
                                   prefix_text_token_ids=inputs['ids'][:, :prefix].tolist(),
                                   prefix_audio_codes=inputs['depth'][:, :prefix].tolist(),
                                   prefix_audio_presence=inputs['present'][:, :prefix].tolist(),
                                   supplied_transcript=ppm.get('supplied_transcript'),
                                   framewise_text_forced=ppm['force_text'],
                                   common_conditioning=f'/{namespace}-conditioning/{clip["id"]}.json')
    if prefill:
        with np.load(clip['prefill_source'], allow_pickle=False) as ground_truth:
            actual_conditioning['ground_truth_prefill'] = dict(frames=prefill, seconds=2, sha256=clip['prefill_sha256'],
                                                               **{key: ground_truth[key].tolist() for key in ('audio_codes', 'pose_codes', 'text_lane')})
    conditioning_path = root / 'gallery/conditioning' / clip['id'] / f'{recipe}.json'
    video.write(conditioning_path, actual_conditioning)
    sampling = ppm['sampling']
    conditioning = {
        'aligned-media': 'Common 8-second voice; full aligned transcript upfront and forced framewise text.',
        'compact-all': 'Common 8-second voice; full plain transcript upfront, with framewise text sampled.',
        'xabi-original': 'Common 8-second voice only; no transcript supplied, with framewise text sampled.',
    }
    metadata = {
        'recipe': recipe, 'text_mode': 'forced' if ppm['force_text'] else 'native', 'force_text': ppm['force_text'],
        'duration_seconds': 60, 'dataset_split': 'Production Xabi eval',
        'generated_transcript': words['text'], 'supplied_transcript': ppm.get('supplied_transcript'),
        'source_transcript': words.get('source_transcript', ppm.get('transcript')),
        'voice_reference_seconds': 8, 'voice_codes_sha256': ppm['voice_codes_sha256'],
        'voice_reference_nonoverlapping_generation': True,
        'generated_framewise_lane_sha256': ppm['generated_lane_sha256'],
        'ground_truth_generation_media': False, 'seed': ppm['seed'], 'batch_size': ppm['batch_size'],
        'conditioning_control': conditioning[recipe], 'audio_guidance': guidance,
        'conditioning_url': f'/{namespace}-conditioning/{clip["id"]}/{recipe}.json',
        'sampling': {'text_temperature': sampling['text_temp'], 'text_top_k': sampling['text_topk'],
                     'audio_temperature': sampling['cb0_temp'], 'audio_top_k': sampling['cb0_topk'],
                     'audio_residual_temperature': sampling['light_temp'], 'audio_residual_top_k': sampling['light_topk'],
                     'pose_temperature': sampling['pose_temp'], 'pose_top_k': sampling['pose_topk'],
                     'pose_residual_temperature': sampling['pose_light_temp'], 'pose_residual_top_k': sampling['pose_light_topk'],
                     'residual_codebooks_use_same_settings': False, 'top_k_zero_means': 'Unfiltered'},
        'video_model': receipt['video_model'], 'video_protocol': protocol, 'video_seed': 2026,
        'postprocessing': receipt['postprocessing'],
        'evaluation_scope': 'Same held-out source identities and appearance; native text conditioning differs by model. Image-conditioned video demonstration.',
    }
    if prefill:
        metadata.update(prefill_frames=prefill, prefill_duration_seconds=2, prefill_modalities=['audio', 'pose', 'text'],
                        ground_truth_prefill_media=True, prefill_sha256=ppm['prefill_sha256'], prefill_marker=marker,
                        output_transcript=words['text'],
                        continuation_transcript=' '.join(w['text'] for w in words.get('words', []) if not w.get('prompt', False)))
        if recipe == 'xabi-original':
            metadata['conditioning_control'] = 'Common 8-second voice; first 2 seconds of source audio, pose and text, then a sampled continuation with no future transcript.'
        else:
            metadata['conditioning_control'] += ' First 2 seconds: source audio, pose and text; generation continues after the flash and beep.'
    video.add_public_provenance(metadata, video.read(job['assets']))
    references = video.read(Path(job['assets']).parent / 'reference-selection.json')
    metadata['reference_images'] = [{'property': row['property'], 'source_timestamp_seconds': row['source_timestamp_s'],
                                    'inside_evaluation_window': row['inside_evaluation_window'],
                                    'source_sha256': row['source_sha256']} for row in references['rows']]
    path = directory / 'public-metadata.json'
    video.write(path, metadata)
    args = ['--data-root', root / 'gallery', 'publish', '--clip-id', clip['id'], '--model', recipe,
            '--video', final, '--metadata', path, *pose_args]
    video.execute([sys.executable, str(video.PUBLISH), *map(str, args)])
    with _TRANSFER_SLOTS:
        for attempt in range(3):
            try:
                paths = [conditioning_path, final, path, directory / 'pose.mp4', directory / 'overlay.mp4', directory / 'pose-views.json']
                if marker:
                    paths.append(directory / 'prefill-marker.json')
                video.sync_to(HOST, paths)
                video.remote(HOST, ['python3', video.PUBLISH, *args])
                break
            except subprocess.CalledProcessError as exc:
                if exc.returncode not in (12, 255) or attempt == 2:
                    raise
                time.sleep(0.5 * (attempt + 1))
    video.write(directory / 'video_state.json', dict(status='complete', published=True,
                                                    published_host=HOST, updated_at=video.stamp()))
