import datetime
import hashlib
import json
import math
from pathlib import Path
import sys

root = Path(sys.argv[1])
health = json.loads((Path('/home/param/pose-runs/duplex-distill-ops-20261029') / (root.name + '-health.json')).read_text())
prof = json.loads((root / 'profile-all-ranks.json').read_text())
rows = [r for rankrows in prof['records'] for r in rankrows]
cfg = json.loads((root / 'config.json').read_text())
plan = json.loads((root / 'plan.json').read_text())
steps = prof['steps']
expected = {(rank, step) for rank in range(48) for step in range(1, steps + 1)}
profile_pairs = [(r['rank'], r['step']) for r in rows]
events = [r for node in health['nodes'].values() for r in node['progress']]
event_pairs = [(r['rank'], r['step']) for r in events]
mismatches = [{k: r[k] for k in ('rank', 'step', 'speech_depth_frames', 'pose_depth_frames')} for r in rows if r['pose_depth_frames'] not in (0, r['speech_depth_frames'])]
checks = {
    'world_48': prof['world'] == 48 and cfg['world'] == 48,
    'all_48_ranks_all_expected_steps': set(profile_pairs) == expected and len(profile_pairs) == 48 * steps,
    'all_48_rank_event_logs_all_expected_steps': set(event_pairs) == expected and len(event_pairs) == 48 * steps,
    'all_six_launchers_exit_zero': all(n['finished'] and n['finished']['exit_code'] == 0 for n in health['nodes'].values()),
    'all_48_gpus_idle': all(len(n['gpu_pids']) == 0 and len(n['gpus']) == 8 and all(row.split(',')[1].strip() == '0' for row in n['gpus']) for n in health['nodes'].values()),
    'no_node_log_errors': all(not n['errors'] for n in health['nodes'].values()),
    'valid_pose_sample_count_matches_speech_every_record': not mismatches,
    'production_not_resumed': prof['production_training_resumed'] is False,
    'no_checkpoint_written': prof['checkpoint_written'] is False,
    'full_architecture': cfg['smoke_depth'] is None,
    'global_batch_48_microbatch_1_accumulation_1': cfg['global_batch_size'] == 48 and cfg['micro_batch_size'] == 1 and cfg['gradient_accumulation'] == 1,
    'shared_ten_percent_training_sample': cfg['depth_frame_sampling']['shared_conversation_frames'] and cfg['depth_frame_sampling']['speech'] == cfg['depth_frame_sampling']['pose'] == 0.1,
    'plan_digest_matches_original': plan['digest'] == '2275f1eab21f3326',
    'all_completed_events_teacher_gradients_false': all(e.get('teacher_gradients') is False for e in events),
    'all_completed_events_kl_and_data_ce_finite': all(math.isfinite(e['kl']) and math.isfinite(e['data_ce']) for e in events),
    'all_completed_events_pose_teacher_native_empty': all(e['conditioning']['pose_teacher_a'] == 'native_empty' for e in events),
    'objective_forward_kl_temperature_one': cfg['objective'] == 'forward_kl_teacher_to_student_only' and cfg['temperature'] == 1.0,
    'equal_modality_weights': all(cfg['modality_weights'][k] == 1/3 for k in ('text', 'audio', 'pose')),
    'audio_and_pose_cb0_quarter_other_mean_three_quarters': cfg['w_cb0'] == cfg['w_pcb0'] == 0.25 and cfg['w_cb'] == cfg['w_pcb'] == 0.75,
}
if 'fasttext' in root.name:
    checks['profile_only_fasttext_config'] = cfg['profiling_text'] == {
        'enabled': True, 'chunk_size': 512, 'direct_addmm': True, 'scope': 'profile_only',
        'implementation': 'benchmark_text_head.direct_addmm'}
source_audit = root / 'verification/source-config-audit.json'
if source_audit.exists():
    source = json.loads(source_audit.read_text())
    for key in ('all_source_hashes_match_controller', 'all_configs_identical', 'all_plans_identical'):
        checks[key] = source[key]
report = {
    'checked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'run': str(root), 'launch': health['launch'], 'all_checks_pass': all(checks.values()), 'checks': checks,
    'profile_records': len(rows), 'valid_pose_records': sum(r['pose_depth_frames'] > 0 for r in rows),
    'missing_pose_records': sum(r['pose_depth_frames'] == 0 for r in rows), 'frame_count_mismatches': mismatches,
    'speech_depth_frames_min_max': [min(r['speech_depth_frames'] for r in rows), max(r['speech_depth_frames'] for r in rows)],
    'pose_depth_frames_nonzero_min_max': [min(r['pose_depth_frames'] for r in rows if r['pose_depth_frames']), max(r['pose_depth_frames'] for r in rows)],
    'max_peak_allocated_GB': max(r['max_memory_GB'] for r in rows),
    'nodes': {host: {'finished': n['finished'], 'gpu_pids': n['gpu_pids'], 'gpus': n['gpus'], 'completed_rank_step_events': len(n['progress']), 'errors': n['errors']} for host, n in health['nodes'].items()},
    'plan': {k: plan[k] for k in ('digest', 'epoch0_digest', 'n_train_micros', 'n_eval_rows', 'n_train_rows', 'n_epochs', 'epoch_steps', 'steps', 'world', 'n_dummy')},
    'fetched_artifacts': {name: {'bytes': (root/name).stat().st_size, 'sha256': hashlib.sha256((root/name).read_bytes()).hexdigest()} for name in ('profile-all-ranks.json', 'config.json', 'metrics.jsonl', 'PROFILE_COMPLETE.json', 'plan.json')},
    'scope_note': 'Read-only GPU monitoring on landsend-1,2,3,4,5,12. No GPU work on landsend-0. Frame counts confirm equality; selected frame identities are additionally enforced by the data path and its CPU tests.',
}
(root / 'verification/completion-audit.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: report[k] for k in ('all_checks_pass', 'checks', 'profile_records', 'valid_pose_records', 'missing_pose_records', 'frame_count_mismatches', 'speech_depth_frames_min_max', 'max_peak_allocated_GB')}, indent=2))
