import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics as st

root = Path('/home/param/outputs/pose-opd-credit-bench/v3-logs')
hosts = ['landsend-2', 'landsend-3', 'landsend-4', 'landsend-5', 'landsend-12']
identity = json.loads((root / hosts[0] / 'config.json').read_text())
cancel = dt.datetime(2026, 10, 31, 6, 18, 37, tzinfo=dt.timezone.utc).timestamp()

def stats(values):
    return dict(min=min(values), median=st.median(values), max=max(values), mean=st.mean(values), n=len(values))

per_rank, host_metrics, intervals, manifest = {}, {}, [], []
for host in hosts:
    directory = root / host
    starts = []
    events = []
    for file in sorted(directory.glob('events-rank*.jsonl')):
        records = [json.loads(line) for line in file.read_text().splitlines() if line.strip()]
        events.extend(records)
        progress = [row for row in records if row['event'] == 'rollout_progress']
        if not progress:
            continue
        starts.append(progress[0]['time'] - progress[0]['seconds_elapsed'])
        warm = []
        for first, last in zip(progress, progress[1:]):
            if first['current_frame'] < 525:
                continue
            elapsed = last['seconds_elapsed'] - first['seconds_elapsed']
            frames = last['current_frame'] - first['current_frame']
            warm.append(1000 * elapsed / frames)
        intervals.extend(warm)
        last = progress[-1]
        per_rank[str(last['rank'])] = dict(host=host, last_frame=last['current_frame'],
            total_frames=last['total_frames'], seconds_elapsed=last['seconds_elapsed'],
            generated_steps_observed=last['current_frame'] - 25, lanes=4,
            cumulative_ms_per_step=1000 * last['seconds_elapsed'] / (last['current_frame'] - 25),
            warm_ms_per_step=stats(warm))
    gpu = {}
    for timestamp, index, memory, util in csv.reader((directory / 'gpu-samples.csv').read_text().splitlines()):
        stamp = dt.datetime.strptime(timestamp.strip(), '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=dt.timezone.utc).timestamp()
        if stamp >= cancel:
            continue
        gpu.setdefault(index.strip(), []).append((stamp, int(memory), int(util)))
    host_metrics[host] = dict(gpus={index: dict(sample_count=len(rows),
        peak_sampled_used_mib=max(row[1] for row in rows),
        warm_used_mib=stats([row[1] for row in rows if row[0] >= max(starts) + 40]),
        warm_utilization_percent=stats([row[2] for row in rows if row[0] >= max(starts) + 40]))
        for index, rows in gpu.items()},
        completed_updates=sum(row['event'] == 'profile' for row in events),
        completed_actor_waves=sum(row.get('phase') == 'actor_microbatch_end' for row in events),
        replay_started=any(row.get('phase') == 'replay_microbatch_begin' for row in events))
    for file in sorted(directory.rglob('*')):
        if file.is_file():
            manifest.append(dict(path=str(file.relative_to(root)), bytes=file.stat().st_size,
                                 sha256=hashlib.sha256(file.read_bytes()).hexdigest()))

warm = stats(intervals)
report = dict(run='pose-opd-bench-ladder-cfg3-v3', status='intentionally_cancelled_during_first_rollout_wave',
    reason='User clarified eight concurrent lanes per GPU pair; v3 incorrectly coupled actor lanes to learner microbatch2.',
    source_sha256=identity['source_sha256'], plan_digest=identity['plan_digest'], world=identity['world'],
    method={key: identity['method'][key] for key in ('actor_backbone_cut', 'physical_microbatch',
        'effective_batch_per_rank', 'pose_teacher_cfg_scale', 'teacher_depth_cache')},
    actor_lanes_per_pair=4, actor_pairs=20, planned_actor_waves_per_update=2,
    warm_ms_per_batched_frame_step=warm,
    warm_row_frames_per_second_per_pair=4000 / warm['median'],
    peak_sampled_device_used_mib=max(g['peak_sampled_used_mib'] for host in host_metrics.values() for g in host['gpus'].values()),
    per_actor=per_rank, hosts=host_metrics,
    limitations=['No full row or actor wave completed; no replay, gradient, optimizer step or completed update observed.',
        'GPU figures are periodic nvidia-smi device-used memory samples, not allocator peaks; short peaks can be missed.',
        'Teacher CFG3 and CPU logit caching were configured but their scoring path was not reached.',
        'Warm speed covers observed frames525 through3025 with4 lanes; it does not establish8-lane or full-context performance.',
        'SIGTERM and Gloo connection-closed errors followed the intentional stop; no preceding runtime error was observed.'],
    provenance=manifest)
(root / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({key: value for key, value in report.items() if key not in ('per_actor', 'hosts', 'provenance')}, indent=2))
