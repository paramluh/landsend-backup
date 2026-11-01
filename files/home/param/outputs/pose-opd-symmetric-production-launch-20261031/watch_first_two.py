"""Read-only first-two-update monitor, polled every120seconds from productionGO."""
import json,subprocess,time
from pathlib import Path
root=Path(__file__).resolve().parent
specdir=root/'snapshot-1793472444394221943'
go=json.loads((specdir/'GO.json').read_text())['go']['time']
next_at=go+120
while True:
 remaining=next_at-time.time()
 if remaining>0:time.sleep(min(remaining,10));continue
 observed=time.time()
 try:
  result=subprocess.run(['/home/param/pose-repo/.venv/bin/python',str(root/'status.py')],capture_output=True,text=True,timeout=90)
  result.check_returncode();v=json.loads(result.stdout)
  (root/f'poll-{int(observed)}.json').write_text(json.dumps(v,indent=2)+'\n')
  nodes=[]
  for row in v['nodes']:
   events=list(row['rank_last_events'].values());last=max(events,key=lambda x:x.get('time',0))if events else{}
   nodes.append({'host':row['host'],'supervisor_running':row['supervisor']['running'],
    'torchrun_running':row['torchrun']['running'],'owned_gpu_processes':len(set(x['pid']for x in row['gpu_compute_processes']if x['owned'])),
    'last':{k:last[k]for k in ('event','phase','update','current_frame')if k in last}})
  metrics=v['nodes'][0]['metrics'];brief={'time':observed,'nodes':nodes,
   'metrics':[{'update':m['update'],'seconds':m['seconds'],'gradient_norm':m['gradient_norm'],'loss':m['loss/direct']}for m in metrics],
   'raw_sample_updates':v['nodes'][0]['raw_sample_updates']}
  print(json.dumps(brief),flush=True)
  if any(not n['supervisor_running'] or not n['torchrun_running'] for n in nodes):
   (root/'FIRST-TWO-FAILED.json').write_text(json.dumps(brief,indent=2)+'\n');raise SystemExit(1)
  if metrics and max(m['update']for m in metrics)>=2:
   (root/'FIRST-TWO-VERIFIED.json').write_text(json.dumps(brief,indent=2)+'\n');raise SystemExit(0)
 except subprocess.SubprocessError as error:
  print(json.dumps({'time':observed,'poll_error':str(error),'action':'No mutation; retry at next scheduled poll'}),flush=True)
 next_at+=120
