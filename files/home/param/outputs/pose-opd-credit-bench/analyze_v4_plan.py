import hashlib
import json
from pathlib import Path
import statistics

import pyarrow.parquet as pq

out = Path('/home/param/outputs/pose-opd-credit-bench/v4-comparison')
root = Path('/home/param/datasets/aloe-duplex')
plan = json.loads((out / 'plan.json').read_text())
identity = json.loads((out / 'config.json').read_text())
mapping_path = root / 'pose/hf/provenance/row_map.original.json'
mapping = json.loads(mapping_path.read_text())
selected, original, inverses, sources = {}, {}, {}, []
for store in range(4):
    name = f'm1n{store}'
    selected_path = root / f's2h-m1-samsep/stores/{name}/rows.parquet'
    original_path = root / f'original/stores/{name}/rows.parquet'
    selected[store] = pq.read_table(selected_path, columns=['row', 'row_key', 'n_frames']).to_pylist()
    original[store] = pq.read_table(original_path, columns=['row', 'row_key', 'n_frames']).to_pylist()
    inverses[store] = {int(new): int(old) for old, new in mapping['map'][name]['new_of_old'].items()}
    for path in (selected_path, original_path):
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

prefill = identity['method']['prefill_frames']
per_rank = plan['rows_per_rank']
hosts = ['landsend-2', 'landsend-3', 'landsend-4', 'landsend-5', 'landsend-12']
rows, pairs = [], []
for slot, item in enumerate(plan['updates'][0]):
    if item is None:
        raise ValueError('This first-update comparison expects all real rows')
    store, row, orientation = item
    record = selected[store][row]
    raw = original[store][inverses[store][row]]
    if record['row_key'] != raw['row_key'] or record['n_frames'] != raw['n_frames']:
        raise ValueError('Mapped original/selected row metadata disagree')
    rows.append(dict(item=item, rank=slot // per_rank, rank_slot=slot % per_rank,
                     original_row=inverses[store][row], row_key=record['row_key'],
                     conversation_frames=int(record['n_frames']),
                     generated_physical_frames=int(record['n_frames']) - prefill))
for first_rank in range(0, plan['world'], 2):
    group = [r for r in rows if first_rank <= r['rank'] < first_rank + 2]
    lengths = [r['conversation_frames'] for r in group]
    generated = sum(r['generated_physical_frames'] for r in group)
    steps = max(lengths) + 2 - prefill
    pairs.append(dict(leader_rank=first_rank, host=hosts[first_rank // 8], rows=len(group),
        lengths=lengths, conversation_frames=sum(lengths), generated_physical_frames=generated,
        max_conversation_frames=max(lengths), max_duration_seconds=max(lengths) / 12.5,
        actor_model_steps=steps, useful_fraction_of_padded_slots=generated / (len(group) * steps)))
records = [json.loads(line) for line in (out / 'events-rank0.jsonl').read_text().splitlines()]
stage = next(row for row in records if row['event'] == 'actor_graph_profile')
progress = [row for row in records if row['event'] == 'rollout_progress']
last = progress[-1]
generated = sum(r['generated_physical_frames'] for r in rows)
conversation = sum(r['conversation_frames'] for r in rows)
amu = json.loads(Path('/home/param/outputs/pose-opd-credit-bench/amu-rollout-baseline.json').read_text())
target = amu['measured_production']['median_active_conversation_frames_per_second_per_pair']
summary = dict(run='pose-opd-bench-actor8-ladder-v4', update=1, source_sha256=identity['source_sha256'],
    plan_digest=plan['digest'], rows=len(rows), actor_pairs=len(pairs), lanes_per_pair=8,
    total_conversation_frames=conversation, total_generated_physical_frames=generated,
    total_conversation_hours=conversation / 12.5 / 3600,
    per_row_frames=dict(min=min(r['conversation_frames'] for r in rows),
                       median=statistics.median(r['conversation_frames'] for r in rows),
                       mean=conversation / len(rows), max=max(r['conversation_frames'] for r in rows)),
    max_frames_per_pair=[p['max_conversation_frames'] for p in pairs],
    mean_useful_fraction_of_padded_slots=statistics.mean(p['useful_fraction_of_padded_slots'] for p in pairs),
    useful_frames_per_second_per_pair_formula=f'{generated} / (20 * measured_full_rollout_seconds)',
    including_prefill_comparison_formula=f'{conversation} / (20 * measured_full_rollout_seconds)',
    amu_useful_frames_per_second_per_pair=target,
    full_rollout_seconds_needed_to_match_amu_generated=generated / (len(pairs) * target),
    full_rollout_seconds_needed_to_match_amu_including_prefill=conversation / (len(pairs) * target),
    completion_status='NOT_COMPLETE_at_snapshot', last_rank0_progress=last,
    rank0_graph_profile=stage,
    comparison_notes=['Frame totals are metadata-only; no Mimi/pose arrays, waveforms, models or CUDA were loaded.',
        'Both original raw and selected row identities/lengths were checked via the exact published inverse row map.',
        'Use completed overall rollout wall time, including slowest pair and prefill, for comparison to Amu production phase throughput.',
        'Current speech and pose graphs overlap on distinct GPUs; their individual50ms stage times must not be summed.',
        'CUDA-event stage measurements cover32 earlyframes, while Amu profiles use pergraph device synchronization and different lane counts/context/thermal state.'],
    pairs=pairs, row_assignments=rows, metadata_sources=sources,
    row_map_sha256=hashlib.sha256(mapping_path.read_bytes()).hexdigest())
(out / 'update1-frame-accounting.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({key: summary[key] for key in ('rows','total_conversation_frames','total_generated_physical_frames',
    'total_conversation_hours','per_row_frames','max_frames_per_pair','mean_useful_fraction_of_padded_slots',
    'useful_frames_per_second_per_pair_formula','amu_useful_frames_per_second_per_pair',
    'full_rollout_seconds_needed_to_match_amu_generated','full_rollout_seconds_needed_to_match_amu_including_prefill')}, indent=2))
