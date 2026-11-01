"""Publish validated native comparison cells to the CPU gallery on landsend-1."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

import comparison_video as video
from request_worker import validate_video_receipt

ROOT = Path('/home/param/pose-runs/native-comparison-20261028')
HOST = 'landsend-1'


def locations(job):
    root = Path(job.get('root', job.get('run_root', ROOT))).resolve()
    directory = Path(job.get('directory', job.get('out', ''))).resolve()
    if root != ROOT or not directory.is_relative_to(root / 'outputs'):
        raise ValueError('Native publication must stay within the native comparison outputs')
    clip = job.get('clip')
    if clip is None:
        clip = next(c for c in video.read(root / 'selection.json')['clips'] if c['id'] == job['clip_id'])
    recipe = job['recipe']
    if recipe not in ('compact-all', 'xabi-original') or directory != root / 'outputs' / clip['gallery_id'] / recipe:
        raise ValueError('Native publication clip/model mismatch')
    return root, directory, clip, recipe


def publish_native_state(job, stage, message=''):
    root, directory, clip, recipe = locations(job)
    if stage == 'complete':
        current = directory / 'video_state.json'
        if current.is_file() and video.read(current).get('status') == 'complete':
            return
        raise ValueError('Completion must be published through publish_native')
    state = 'failed' if stage == 'failed' else 'pending' if stage in ('pending', 'waiting') else 'generating'
    video.write(directory / 'video_state.json', {'status': stage, 'message': message, 'updated_at': video.stamp()})
    args = ['--data-root', root / 'gallery', 'status', '--clip-id', clip['id'], '--model', recipe,
            '--status', state, '--message', message]
    video.execute([sys.executable, str(video.PUBLISH), *map(str, args)])
    video.remote(HOST, ['python3', video.PUBLISH, *args])


def publish_native(job):
    root, directory, clip, recipe = locations(job)
    receipt = video.read(directory / 'VIDEO_COMPLETE.json')
    video_validation = validate_video_receipt(receipt['diffusion_receipt'], 752, 2026)
    ppm = video.read(directory / 'ppm/meta.json')
    words = video.read(directory / 'ppm/words.json')
    complete = video.read(directory / 'ppm/PPM_COMPLETE.json')
    final = directory / 'video.mp4'
    if not (receipt['status'] == complete['status'] == 'complete' and receipt['frames'] == 1500
            and receipt['seconds'] == 60 and hashlib.sha256(final.read_bytes()).hexdigest() == receipt['sha256']):
        raise ValueError('Native video is not a validated sixty-second result')
    if ppm.get('text_mode') != 'native' or ppm.get('force_text') is not False or ppm['recipe'] != recipe:
        raise ValueError('Cannot label a word-forced result as native')
    sampling = ppm['sampling']
    if not (all(sampling[k] == 1 for k in ('text_temp', 'cb0_temp', 'light_temp', 'pose_temp', 'pose_light_temp'))
            and all(sampling[k] == 0 for k in ('text_topk', 'cb0_topk', 'light_topk', 'pose_topk', 'pose_light_topk'))):
        raise ValueError('Native comparison requires temperature one and unfiltered sampling')
    if (ppm['ground_truth_generation_media'] or ppm['voice_clip_frames'] != 100 or ppm['prefill_frames'] != 0
            or ppm['voice_codes_sha256'] != clip['voice_provenance']['codes_sha256']):
        raise ValueError('Unexpected native conditioning inputs')
    supplied = ppm.get('supplied_transcript')
    if (recipe == 'xabi-original' and supplied is not None) or (recipe == 'compact-all' and not supplied):
        raise ValueError('Native transcript conditioning does not match this model')
    metadata = {
        'recipe': recipe, 'text_mode': 'native', 'force_text': False,
        'duration_seconds': 60, 'dataset_split': 'Production Xabi eval',
        'generated_transcript': words['text'], 'supplied_transcript': supplied,
        'source_transcript': words.get('source_transcript', ppm.get('transcript')),
        'voice_reference_seconds': 8, 'voice_codes_sha256': ppm['voice_codes_sha256'],
        'voice_reference_nonoverlapping_generation': True,
        'generated_framewise_lane_sha256': ppm['generated_lane_sha256'],
        'ground_truth_generation_media': False, 'seed': ppm['seed'], 'batch_size': ppm['batch_size'],
        'conditioning_control': 'Identical original-speaker voice and appearance; framewise words and timing are generated',
        'sampling': {'text_temperature': sampling['text_temp'], 'text_top_k': sampling['text_topk'],
                     'audio_temperature': sampling['cb0_temp'], 'audio_top_k': sampling['cb0_topk'],
                     'pose_temperature': sampling['pose_temp'], 'pose_top_k': sampling['pose_topk'],
                     'residual_codebooks_use_same_settings': True, 'top_k_zero_means': 'Unfiltered'},
        'video_model': receipt['video_model'], 'video_protocol': video_validation, 'video_seed': 2026,
        'postprocessing': receipt['postprocessing'],
        'evaluation_scope': 'Same held-out source identities; native wording and timing differ. Image-conditioned video demonstration, not leakage-free video evaluation',
    }
    assets = video.read(job['assets'])
    video.add_public_provenance(metadata, assets)
    ref_selection = video.read(Path(job['assets']).parent / 'reference-selection.json')
    metadata['reference_images'] = [{'property': row['property'], 'source_timestamp_seconds': row['source_timestamp_s'],
                                    'inside_evaluation_window': row['inside_evaluation_window'],
                                    'source_sha256': row['source_sha256']} for row in ref_selection['rows']]
    path = directory / 'public-metadata.json'
    video.write(path, metadata)
    args = ['--data-root', root / 'gallery', 'publish', '--clip-id', clip['id'], '--model', recipe,
            '--video', final, '--metadata', path, *video.pose_views(directory)]
    video.execute([sys.executable, str(video.PUBLISH), *map(str, args)])
    # Copy only this completed cell. Each host merges its manifest under the publisher lock.
    video.sync_to(HOST, [final, path, directory / 'pose.mp4', directory / 'overlay.mp4', directory / 'pose-views.json'])
    video.remote(HOST, ['python3', video.PUBLISH, *args])
    video.write(directory / 'video_state.json', {'status': 'complete', 'published': True,
                                               'published_host': HOST, 'updated_at': video.stamp()})
