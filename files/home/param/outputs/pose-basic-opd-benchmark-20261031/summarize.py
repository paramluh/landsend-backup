import json,time
from pathlib import Path
out=Path('/home/param/pose-worktrees/pose-basic-opd/finetune/pose_opd_credit/BASIC_OPD_BENCHMARKS.json')
root=Path('/home/param/runs/smokes');attempts=[]
for suffix in ('a','b','d'):
 run=root/('pose-basic-opd-synthetic-20261031-'+suffix)
 events=run/'events-rank0.jsonl'
 if not events.exists():continue
 rows=[json.loads(s) for s in events.read_text().splitlines()]
 trials=[{k:v for k,v in d.items() if k not in ('rank','metrics')} for d in rows if d['event']=='replay_benchmark']
 spec=json.loads((run/'supervisor/spec.json').read_text())
 finished=run/'supervisor/finished.json'
 attempts.append({'run_dir':str(run),'source_archive_sha256':spec['archive_sha256'],'trials':trials,
                  'finished':json.loads(finished.read_text()) if finished.exists() else None,
                  'outcome':'alternate-layer checkpointing OOM after baseline' if suffix=='a' else 'measured candidates'})
budget=json.loads(Path(__file__).with_name('BUDGET.json').read_text())
report={'purpose':'Bounded synthetic forward/backward optimization; no rollout inference',
        'global_batch':336,'world':56,'rows_per_rank':6,'positions_per_rank':9816,
        'budget':budget,'attempts':attempts,'previous_real_student_seconds':27.88435866695363,
        'comparison_limit':'Synthetic maximum-length batches versus previous real batches are not identical workloads.'}
out.write_text(json.dumps(report,indent=2)+'\n')
for a in attempts:
 for t in a['trials']:print(Path(a['run_dir']).name,t['trial'],t['replay_seconds'],t['peak_allocated_gib'],t['decoder_chunk_frames'],t['text_chunk_frames'],t.get('upper_checkpoint_skip_every',0))
