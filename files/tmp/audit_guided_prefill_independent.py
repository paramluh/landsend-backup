import collections
import datetime
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

ROOT = Path('/home/param/pose-runs/audio-guided-prefill2s-comparison-20261028')
sys.path.insert(0, '/home/param/pose-repo/infer/transcript_pose')
from request_schema import normalize_request
sample_request = dict(recipe='compact-all', clip_id='clip-01', text_mode='native')
assert normalize_request(sample_request, new_submission=True)['prefill_frames'] == 25
assert normalize_request(sample_request)['prefill_frames'] == 0
assert normalize_request(dict(sample_request, prefill_frames=0), new_submission=True)['prefill_frames'] == 0
read = lambda p: json.loads(Path(p).read_text())
sha = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
selection = read(ROOT / 'selection.json')
jobs = read(ROOT / 'jobs.json')
teams = read(ROOT / 'execution-plan.json')
assert len(jobs) == 24 and len(teams) == 16 and len(selection['clips']) == 8
slots = [(team['node'], gpu) for team in teams for gpu in team['devs']]
assert len(slots) == len(set(slots)) == 32
assert set(slots) == {(f'landsend-{n}', g) for n in range(1, 5) for g in range(8)}
assert collections.Counter(j['recipe'] for j in jobs) == {'aligned-media': 8, 'compact-all': 8, 'xabi-original': 8}
distribution = read(ROOT / 'gpu-distribution-audit.json') if (ROOT / 'gpu-distribution-audit.json').exists() else {'nodes': []}
actual_slots = []
for node in distribution['nodes']:
    assert len(node['processes']) == 4
    for process in node['processes']:
        command = process['command']
        assert command and '--pose-audio-guidance' in command['command']
        args = command['command']
        assert args[args.index('--pose-audio-guidance') + 1] == '3'
        assert args[args.index('--prefill-frames') + 1] == '25'
        actual_slots.extend((node['node'], gpu) for gpu in command['gpus'])
if actual_slots:
    assert sorted(actual_slots) == sorted(slots)
actual_launch_verified = len(actual_slots) == 32

cache = Path('/home/param/datasets/xabi-v2pure-cache/cache/eval')
source = pq.read_table(cache / 'items.parquet', columns=['key', 'n']).to_pydict()
offsets = np.load(cache / 'codes_off.npy', mmap_mode='r')
codes = np.load(cache / 'codes.npy', mmap_mode='r')
pose_codes = np.load(cache / 'pose.npy', mmap_mode='r')
voice_hashes = {}
prefills = {}
for clip in selection['clips']:
    i = clip['cache_index']
    assert clip['split'] == 'eval'
    assert source['key'][i] == clip['cache_key'] and source['n'][i] == clip['n_source_frames']
    v = clip['voice_provenance']
    a, b = v['source_clip_frames']
    assert a >= clip['start_frame'] + 752 and b <= clip['n_source_frames'] and b - a == 100
    original_voice = codes[int(offsets[i]) + a:int(offsets[i]) + b].T.astype(np.int64)
    voice = np.load(clip['voice_clip'], allow_pickle=False)
    assert np.array_equal(voice, original_voice), (clip['id'], 'Voice not copied from matching eval speaker')
    voice_hashes[clip['id']] = sha(voice.astype(np.int64))
    assert voice_hashes[clip['id']] == v['codes_sha256']
    start = int(offsets[i]) + clip['start_frame']
    expected = dict(audio_codes=codes[start:start+25].T.astype(np.int64),
                    pose_codes=pose_codes[start:start+25].T.astype(np.int64),
                    text_lane=np.load(clip['lane'], allow_pickle=False)[:25].astype(np.int64))
    with np.load(clip['prefill_source'], allow_pickle=False) as saved:
        assert set(saved.files) == set(expected)
        for key in expected:
            assert np.array_equal(saved[key], expected[key]), (clip['id'], key, 'Prepared prefill differs from raw source')
    assert {k: sha(v) for k,v in expected.items()} == clip['prefill_sha256']
    prefills[clip['id']] = expected

tokenizer = None
results, pending, failures = [], [], []
for job in jobs:
    name = f"{job['clip_id']}/{job['recipe']}"
    path = Path(job['ppm'])
    if not (path / 'PPM_COMPLETE.json').exists():
        pending.append(name)
        continue
    try:
        clip, recipe = job['clip'], job['recipe']
        meta, receipt = read(path / 'meta.json'), read(path / 'PPM_COMPLETE.json')
        assert receipt['status'] == 'complete' and receipt['exit_code'] == 0
        assert meta['recipe'] == recipe and meta['clip_index'] == clip['clip_index']
        assert meta['checkpoint']['ck'] == f'/home/param/models/transcript-pose/{recipe}/export-200'
        assert meta['checkpoint']['step'] == 200 and meta['checkpoint']['pose'] and meta['checkpoint']['decp_bb']
        assert meta['split'] == 'eval' and meta['cache_key'] == clip['cache_key']
        assert meta['frames'] == 752 and meta['prefill_frames'] == meta['f0'] == 25
        assert meta['prefill_duration_seconds'] == 2 and meta['ground_truth_prefill_media']
        assert meta['prefill_modalities'] == ['audio', 'pose', 'text']
        assert meta['prefill_sha256'] == clip['prefill_sha256']
        assert meta['source_span'] == [clip['start_frame'], clip['start_frame'] + 752]
        assert meta['batch_size'] == job['batch_size'] and meta['batch_row'] == job['batch_row']
        assert not meta['ground_truth_generation_media']
        assert meta['voice_codes_sha256'] == voice_hashes[clip['id']]
        assert sha(np.load(path / 'voice_clip_mimi.npy').astype(np.int64)) == voice_hashes[clip['id']]
        g, audit = meta['audio_guidance'], meta['audio_guidance']['audit']
        assert g['enabled'] and g['strength'] == 3 and g['engine_scale'] == 4
        assert g['full_model_silence_pass'] and g['shared_text_and_pose'] and g['scope'] == 'all pose codebooks'
        assert audit['enabled'] and audit['independent_cache_rows'] and audit['d33_uses_separate_backbone_hidden']
        assert audit['codebooks'] == list(range(32)) and audit['guided_frames'] == 727
        assert audit['conditional_weight'] == 4 and audit['silence_weight'] == -3 and not audit['speaking_only']
        assert audit['conditional_row'] == job['batch_row'] and audit['silence_row'] == job['batch_size'] + job['batch_row']
        assert audit['backbone_cache_rows'] == audit['pose_tower_cache_rows'] == audit['pose_light_cache_rows'] == 2 * job['batch_size']
        assert audit['prefix_audio_frames_replaced_by_channel'] == [0, 123]
        assert audit['forced_audio_frames_replaced_by_channel'] == [0, 2]
        assert audit['total_supplied_audio_frames_replaced_by_channel'] == [0, 125]
        assert audit['forced_pose_frames'] == 2
        silence = np.load(path / 'guidance_silence_mimi.npy', allow_pickle=False)
        assert silence.dtype == np.int64 and silence.shape == (32, g['silence_frames'])
        assert silence.min() >= 0 and silence.max() < 2048 and np.any(silence != 0)
        assert sha(silence) == g['silence_codes_sha256'] == audit['silence_codes_sha256']
        assert g['silence_codec'] == 'kyutai/mimi@89091b3e466eb6a9d11e537bf26b144f194978f7'
        sampling = meta['sampling']
        for key in ('text_temp', 'cb0_temp', 'light_temp', 'pose_temp', 'pose_light_temp'):
            assert sampling[key] == 1
        for key, value in dict(text_topk=0, cb0_topk=30, light_topk=30, pose_topk=30, pose_light_topk=50,
                               pose_cfg_on=1, pose_cfg_s=4, pose_cfg_cb0=1, pose_cfg_light=1,
                               pose_cfg_speaking_only=0).items():
            assert sampling[key] == value, (key, sampling[key])
        lane = np.load(clip['lane'], allow_pickle=False)
        with np.load(path / 'inputs.npz', allow_pickle=False) as inputs:
            P, C = meta['P'], meta['conditioning_P']
            assert C == 109 and meta['transcript_span'] == [C, P]
            prefix_hashes = {k: sha(inputs[k][:, :C]) for k in ('ids', 'depth', 'present')}
            assert prefix_hashes == clip['voice_provenance']['system_prefix_array_sha256']
            assert np.all(inputs['pdepth'][:P+2] == -1) and np.all(inputs['pdepth'][P+27:] == -1)
            assert np.all(inputs['depth'][1,P:P+2] == -1) and np.all(inputs['depth'][1,P+27:] == -1)
            off = np.arange(32, dtype=np.int64) * 2048
            assert np.array_equal((inputs['depth'][1,P+2:P+27] - off).T, prefills[clip['id']]['audio_codes'])
            assert np.array_equal((inputs['pdepth'][P+2:P+27] - off).T, prefills[clip['id']]['pose_codes'])
            assert np.array_equal(inputs['ids'][1,P:P+25], prefills[clip['id']]['text_lane'])
            assert np.all(inputs['depth'][0] == -1) and np.all(inputs['present'][0] == 0)
            forced = recipe == 'aligned-media'
            assert meta['force_text'] == forced
            assert meta['text_mode'] == ('forced' if forced else 'native')
            if forced:
                assert np.array_equal(inputs['text_force'][P:P+752], lane)
                assert np.array_equal(inputs['ids'][1], inputs['text_force'])
            else:
                assert 'text_force' not in inputs and np.all(inputs['ids'][1, P+25:] == 151670)
            upfront = inputs['ids'][1, C:P]
            if recipe == 'xabi-original':
                assert not meta['upfront_transcript'] and P == C and meta['supplied_transcript'] is None
            else:
                assert meta['upfront_transcript'] and upfront[0] == 151680 and upfront[-1] == 151681
                assert meta['supplied_transcript'] == meta['source_transcript']
                if forced:
                    assert np.array_equal(upfront[1:-1], lane)
                else:
                    if tokenizer is None:
                        tokenizer = AutoTokenizer.from_pretrained('/home/param/models/ppm-811-17850-base-bf16/configs/Qwen3-32B', local_files_only=True)
                    expected = tokenizer.encode(meta['source_transcript'], add_special_tokens=False, split_special_tokens=True)
                    assert np.array_equal(upfront[1:-1], expected)
                    assert np.all(upfront[1:-1] < 151643)
        with np.load(path / 'tokens.npz', allow_pickle=False) as out:
            for key in ('audio_codes', 'pose_codes'):
                x = out[key]
                assert x.shape == (32, 752) and np.issubdtype(x.dtype, np.integer) and x.min() >= 0 and x.max() < 2048
                assert np.array_equal(x[:,:25], prefills[clip['id']][key]), (key, 'Output prefix changed')
            assert sha(out['lane'].astype(np.int64)) == meta['generated_lane_sha256']
            assert np.array_equal(out['lane'][:25], prefills[clip['id']]['text_lane'])
            if forced:
                assert np.array_equal(out['lane'], lane)
        results.append(dict(sample=name, status='passed', node=job['node'], gpu_pair=job['devs'],
                            voice_codes_sha256=voice_hashes[clip['id']], silence_codes_sha256=sha(silence),
                            guided_frames=727, pose_codebooks=32, text_mode=meta['text_mode'],
                            prefill_frames=25, prefill_sha256=meta['prefill_sha256'],
                            upfront_transcript=meta['upfront_transcript'], independent_cache_rows=2*job['batch_size']))
    except Exception as error:
        import traceback
        failures.append(dict(sample=name, error=repr(error), traceback=traceback.format_exc()))

report = dict(status='failed' if failures else 'passed' if len(results) == 24 and actual_launch_verified else 'in-progress',
              checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              source_eval_clips=8, same_nonoverlapping_voice_verified=True, nodes=4, gpu_slots=32, teams=16,
              actual_launch_commands_verified=actual_launch_verified, gpu_distribution_receipt=str(ROOT / 'gpu-distribution-audit.json'),
              prefill_frames=25, prefill_seconds=2.0, prefill_modalities=['audio','pose','text'],
              prefill_source_check='All8 prepared NPZ prefixes match first25 raw eval-cache audio/pose frames and original framewise text',
              future_new_request_default_frames=25, legacy_missing_prefill_frames=0, explicit_zero_still_allowed=True,
              preparation_validation_reference=str(ROOT/'preparation-validation.json'),
              planned_node_gpu_slots=[dict(node=n, gpu=g) for n,g in sorted(slots)],
              completed_samples=len(results), expected_samples=24, failures=failures, pending=pending, samples=results,
              mechanism_review='Full engine code reviewed: independent backbone/pose tower/light KV rows, independent D33 links; same text and pose history; sampled current codebook prefix shared. Actual FusedDec.gen CPU mechanism and B2 CUDA smoke passed before production.')
(ROOT / 'independent-audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k:report[k] for k in ('status','completed_samples','expected_samples','failures','pending')}, indent=2))
