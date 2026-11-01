import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics as st

root = Path('/home/param/outputs/pose-opd-credit-bench/v4-logs')
hosts = ['landsend-2', 'landsend-3', 'landsend-4', 'landsend-5', 'landsend-12']
identity = json.loads((root / hosts[0] / 'config.json').read_text())
accounting = json.loads(Path('/home/param/outputs/pose-opd-credit-bench/v4-comparison/update1-frame-accounting.json').read_text())
metrics = json.loads((root / hosts[0] / 'metrics.jsonl').read_text().splitlines()[0])
amu = json.loads(Path('/home/param/outputs/pose-opd-credit-bench/amu-rollout-baseline.json').read_text())
exclusions = [(1793428232.7880597, 1793428235.2324479), (1793428323.8994434, 1793428327.5925992)]
profiles, intervals, host_reports, provenance = {}, [], {}, []

def stats(values):
    return dict(min=min(values), median=st.median(values), max=max(values), mean=st.mean(values), n=len(values))

for host in hosts:
    path = root / host
    finished = json.loads((path / 'supervisor/finished.json').read_text())
    gpu_samples = []
    events = []
    for file in path.glob('events-rank*.jsonl'):
        records = [json.loads(line) for line in file.read_text().splitlines()]
        events.extend(records)
        for record in records:
            if record['event'] == 'profile':
                profiles[str(record['rank'])] = record
        progress = [r for r in records if r['event'] == 'rollout_progress']
        for before, after in zip(progress, progress[1:]):
            if before['current_frame'] < 525:
                continue
            if before['rank'] == 38 and any(before['time'] < stop and after['time'] > start for start, stop in exclusions):
                continue
            intervals.append(1000 * (after['seconds_elapsed'] - before['seconds_elapsed']) /
                             (after['current_frame'] - before['current_frame']))
    begin = min(r['time'] for r in events if r.get('phase') == 'data_audio_begin')
    end = max(r['time'] for r in events if r['event'] == 'profile')
    for timestamp, index, memory, utilization in csv.reader((path / 'gpu-samples.csv').read_text().splitlines()):
        time = dt.datetime.strptime(timestamp.strip(), '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=dt.timezone.utc).timestamp()
        if begin <= time <= end:
            gpu_samples.append((int(index), int(memory), int(utilization)))
    host_reports[host] = dict(exit_code=finished['exit_code'],
        gpu_peak_sampled_used_mib={str(index): max(row[1] for row in gpu_samples if row[0] == index) for index in range(8)})
    for file in sorted(path.rglob('*')):
        if file.is_file():
            provenance.append(dict(path=str(file.relative_to(root)), bytes=file.stat().st_size,
                sha256=hashlib.sha256(file.read_bytes()).hexdigest()))

phase_names = ['preparation_seconds', 'rollout_seconds', 'replay_seconds', 'optimizer_seconds',
               'peak_allocated_gib', 'peak_reserved_gib']
phase_summary = {name: stats([record[name] for record in profiles.values()]) for name in phase_names}
phase_sums = {rank: sum(record[name] for name in phase_names[:4]) for rank, record in profiles.items()}
full_rollout = phase_summary['rollout_seconds']['max']
useful = accounting['total_generated_physical_frames'] / (20 * full_rollout)
including_prefill = accounting['total_conversation_frames'] / (20 * full_rollout)
amu_useful = amu['measured_production']['median_active_conversation_frames_per_second_per_pair']
current_norm = metrics['gradient_norm']
report = dict(run='pose-opd-bench-actor8-ladder-v4', status='one_complete_actual_update_all40ranks_all5hosts_exit0',
    stop='Intentional STOP_AFTER_SAVE control marker with --no-save: no resumable checkpoint was written.',
    source_sha256=identity['source_sha256'], plan_digest=identity['plan_digest'],
    phase_summary_across40ranks=phase_summary, max_rank_phase_sum_seconds=max(phase_sums.values()),
    rank0_total_update_seconds=metrics['seconds'], rank0_profile=profiles['0'],
    effective_rows=160, actor_pairs=20, actor_lanes_per_pair=8, learner_physical_microbatch=2,
    total_generated_physical_frames=accounting['total_generated_physical_frames'],
    total_conversation_frames=accounting['total_conversation_frames'],
    useful_generated_frames_per_second_per_pair=useful,
    conversation_frames_per_second_per_pair_including_prefill=including_prefill,
    amu_production_useful_frames_per_second_per_pair=amu_useful,
    fraction_of_amu_generated_throughput=useful / amu_useful,
    amu_over_current_generated_throughput=amu_useful / useful,
    steady_batched_frame_ms_excluding_colocated_microprofiles=stats(intervals),
    aggregate_loss=metrics['loss/direct'], gradient_norm=current_norm,
    all_recorded_metrics_finite=all(math.isfinite(value) for value in metrics.values() if isinstance(value, (int, float))),
    policy_parity=dict(mean_absolute_selected_action_logprob_difference=metrics['policy/logprob_mean_abs_difference'],
        max_absolute_selected_action_logprob_difference=metrics['policy/logprob_max_abs_difference'],
        largest_selected_action_probability_ratio=math.exp(metrics['policy/logprob_max_abs_difference']),
        amu_reference='docs/source-readmes/speed-SWITCHES.md:44-64 and evidence/speed/lightparity.json',
        amu_fastlight_vs_original=dict(mean_abs_logit_diff=.0251, max_abs_logit_diff=.75, mean_full_categorical_kl_per_draw=2.8e-4,
            same_uniform_draw_flip_fraction=.133),
        interpretation='Current results establish finite training execution, not bitwise actor/replay equivalence. Selected-action logprob differences cannot be directly equated to Amu full-logit differences or categorical KL.',
        follow_up=['Log actor/replay differences by text/audio/pose and each codebook, with p50/p95/p99/max and offending frame IDs.',
            'On a bounded set of the exact same sampled contexts, compare complete actor/replay categorical logits and compute full KL; include an identical-kernel selfcheck and a batch-geometry baseline.',
            'Check tail differences versus position/context length, executable vocabulary, both media delay offsets and model dtype flow before attributing them solely to roundoff.',
            'If quantified distribution discrepancy exceeds an accepted tolerance, match actor/replay kernels or add explicit policy correction; do not describe current gradients as bitwise sampling-path exact.']),
    limitations=['First update fits with learner microbatch2; warm Adam-state actor memory and learner microbatch4 are tested separately in v5.',
        'Amu uses28lanes/pair, shortvariable question/answer sequences, and speechonly; this test uses8lanes and fullrows up to7500frames pluspose.',
        'Cross-run throughput differences cannot be assigned uniquely to batch, context length, model work or kernels without matched ablations.',
        'Overall rollout and update durations include diagnostic profiling overhead; only the explicitly listed rank38 progress intervals are excluded from steady frame estimates.',
        'Nvidia-smi values are sampled device-used memory acrossprocesses, while profile memory is perprocess PyTorch allocator peaks.'],
    excluded_rank38_wall_intervals_unix=exclusions, hosts=host_reports, profiles=profiles, provenance=provenance)
(root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: report[key] for key in ('phase_summary_across40ranks','max_rank_phase_sum_seconds',
    'rank0_total_update_seconds','useful_generated_frames_per_second_per_pair',
    'conversation_frames_per_second_per_pair_including_prefill','fraction_of_amu_generated_throughput',
    'amu_over_current_generated_throughput','steady_batched_frame_ms_excluding_colocated_microprofiles',
    'aggregate_loss','gradient_norm','all_recorded_metrics_finite')}, indent=2))
