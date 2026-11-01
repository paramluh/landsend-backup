import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics as st

root = Path('/home/param/outputs/pose-opd-credit-bench/v5-logs')
previous_root = root.parent / 'v4-logs'
hosts = ['landsend-2', 'landsend-3', 'landsend-4', 'landsend-5', 'landsend-12']
identity = json.loads((root / hosts[0] / 'config.json').read_text())
plan = json.loads((root / hosts[0] / 'plan.json').read_text())
previous_plan = json.loads((previous_root / hosts[0] / 'plan.json').read_text())
if plan['updates'][0] != previous_plan['updates'][0]:
    raise ValueError('First160row plan differs; frame accounting must be recomputed')
accounting = json.loads((root.parent / 'v4-comparison/update1-frame-accounting.json').read_text())
previous = json.loads((previous_root / 'summary.json').read_text())
metrics = json.loads((root / hosts[0] / 'metrics.jsonl').read_text().splitlines()[0])
profiles, warmup, stages, intervals, hosts_out, manifest = {}, {}, {}, [], {}, []
cadence_by_context = {'early_to2500': [], 'middle_to5000': [], 'late_after5000': []}
actor_memory = {}

def stats(values):
    return dict(min=min(values), median=st.median(values), max=max(values), mean=st.mean(values), n=len(values))

for host in hosts:
    path = root / host
    finished = json.loads((path / 'supervisor/finished.json').read_text())
    if finished['exit_code'] != 0:
        raise ValueError(f'{host} exited unsuccessfully')
    events = []
    for file in path.glob('events-rank*.jsonl'):
        records = [json.loads(line) for line in file.read_text().splitlines()]
        events.extend(records)
        for row in records:
            if row['event'] == 'profile':
                profiles[str(row['rank'])] = row
            elif row['event'] == 'optimizer_warmup':
                warmup[str(row['rank'])] = row
            elif row['event'] == 'actor_graph_profile':
                stages[str(row['rank'])] = row
            elif row['event'] == 'actor_memory':
                actor_memory[str(row['rank'])] = row
        progress = [r for r in records if r['event'] == 'rollout_progress']
        for before, after in zip(progress, progress[1:]):
            if before['current_frame'] >= 525:
                cadence = 1000 * (after['seconds_elapsed'] - before['seconds_elapsed']) / (after['current_frame'] - before['current_frame'])
                intervals.append(cadence)
                group = ('early_to2500' if after['current_frame'] <= 2500 else
                         'middle_to5000' if after['current_frame'] <= 5000 else 'late_after5000')
                cadence_by_context[group].append(cadence)
    start = min(r['time'] for r in events if r.get('phase') == 'optimizer_warmup_begin')
    end = max(r['time'] for r in events if r['event'] == 'profile')
    samples = []
    for timestamp, index, memory, utilization in csv.reader((path / 'gpu-samples.csv').read_text().splitlines()):
        stamp = dt.datetime.strptime(timestamp.strip(), '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=dt.timezone.utc).timestamp()
        if start <= stamp <= end:
            samples.append((int(index), int(memory), int(utilization)))
    hosts_out[host] = dict(exit_code=finished['exit_code'], gpu_peak_sampled_used_mib={
        str(index): max(sample[1] for sample in samples if sample[0] == index) for index in range(8)})
    for file in sorted(path.rglob('*')):
        if file.is_file():
            manifest.append(dict(path=str(file.relative_to(root)), bytes=file.stat().st_size,
                                 sha256=hashlib.sha256(file.read_bytes()).hexdigest()))

if len(profiles) != 40 or len(warmup) != 40 or len(stages) != 20:
    raise ValueError('Incomplete40rank/20actor evidence')
phase_names = ['preparation_seconds', 'rollout_seconds', 'replay_seconds', 'optimizer_seconds',
               'peak_allocated_gib', 'peak_reserved_gib']
phases = {name: stats([r[name] for r in profiles.values()]) for name in phase_names}
old_events = [json.loads(line) for line in (previous_root / hosts[0] / 'events-rank0.jsonl').read_text().splitlines()]
old_stages = next(row['stages'] for row in old_events if row['event'] == 'actor_graph_profile')
stage_comparison = {key: dict(v4_gpu_ms=old_stages[key]['gpu_ms_mean'],
    v5_gpu_ms=stages['0']['stages'][key]['gpu_ms_mean'],
    speedup=old_stages[key]['gpu_ms_mean'] / stages['0']['stages'][key]['gpu_ms_mean']) for key in old_stages}
rollout = phases['rollout_seconds']['max']
useful = accounting['total_generated_physical_frames'] / (20 * rollout)
report = dict(run='pose-opd-bench-actor8-cdf-mb4-v5', status='one_complete_bounded_update_with_warm_optimizer_all40ranks_exit0',
    source_sha256=identity['source_sha256'], plan_digest=identity['plan_digest'], first160rows_identical_to_v4=True,
    method=identity['method'], phase_summary_across40ranks=phases,
    rank0_total_update_seconds=metrics['seconds'], rank0_profile=profiles['0'],
    max_rank_phase_sum_seconds=max(sum(r[k] for k in phase_names[:4]) for r in profiles.values()),
    warm_optimizer=dict(enabled_in_identity=identity['benchmark_warm_optimizer'], ranks=len(warmup),
        every_rank_zero_gradient_warmup=all(r['zero_gradient_warmup'] and r['optimizer_step'] == 1 for r in warmup.values()),
        description='Fresh AdamW moments allocated before actor publication via one zero-gradient/zero-decay step. Weights unchanged; counter advances to1. Profiling-only, bounded no-save run, not production initialization.'),
    actor_lanes_per_pair=8, actor_pairs=20, learner_physical_microbatch=4, effective_rows=160,
    total_generated_physical_frames=accounting['total_generated_physical_frames'],
    total_conversation_frames=accounting['total_conversation_frames'],
    useful_generated_frames_per_second_per_pair=useful,
    useful_including_prefill_frames_per_second_per_pair=accounting['total_conversation_frames'] / (20 * rollout),
    amu_useful_frames_per_second_per_pair=previous['amu_production_useful_frames_per_second_per_pair'],
    fraction_of_amu_generated_throughput=useful / previous['amu_production_useful_frames_per_second_per_pair'],
    speedup_vs_v4=dict(rollout=previous['phase_summary_across40ranks']['rollout_seconds']['max'] / rollout,
        replay=previous['phase_summary_across40ranks']['replay_seconds']['max'] / phases['replay_seconds']['max'],
        total_update_rank0=previous['rank0_total_update_seconds'] / metrics['seconds']),
    rank0_stage_comparison=stage_comparison,
    steady_cpu_submission_cadence_ms=stats(intervals),
    cpu_submission_cadence_ms_by_context={key: stats(value) for key, value in cadence_by_context.items()},
    gpu_peak_sampled_used_mib=max(n for host in hosts_out.values() for n in host['gpu_peak_sampled_used_mib'].values()),
    aggregate_loss=metrics['loss/direct'], gradient_norm=metrics['gradient_norm'],
    all_metrics_finite=all(math.isfinite(value) for value in metrics.values() if isinstance(value, (float, int))),
    policy_parity=dict(rank_averaged_mean_absolute_selected_action_logprob_difference=metrics['policy/logprob_mean_abs_difference'],
        global_max_absolute_selected_action_logprob_difference=metrics['policy/logprob_max_abs_difference'],
        largest_selected_action_probability_ratio=math.exp(metrics['policy/logprob_max_abs_difference']),
        interpretation='Finite execution is established; selected-action logprob deltas are not complete-distribution KL or bitwise parity.'),
    limitations=['v4 andv5 use identical rowselection and initial student weights but different samplingRNG/trajectories and different learner batching; this is a combinedoptimization benchmark.',
        'CUDAevent graphprofiles are bounded early-context samples; progress events reportCPUsubmission, not completedGPUwork. Fullphase timing includes finalreadback.',
        'No additional colocated GPU microbench ran duringv5.',
        'Amu had28actorlanes and short speechonly question/answer trajectories; fullrows+pose8lane workload differs.',
        'No checkpoint was saved; warmoptimizer initialization is deliberately prohibited for saved/resumed production runs.'],
    hosts=hosts_out, profiles=profiles, optimizer_warmup_receipts=warmup, stage_profiles=stages,
    actor_memory_receipts=actor_memory, provenance=manifest)
(root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: report[key] for key in ('phase_summary_across40ranks','rank0_total_update_seconds',
    'useful_generated_frames_per_second_per_pair','fraction_of_amu_generated_throughput','speedup_vs_v4',
    'gpu_peak_sampled_used_mib','aggregate_loss','gradient_norm','policy_parity')}, indent=2))
