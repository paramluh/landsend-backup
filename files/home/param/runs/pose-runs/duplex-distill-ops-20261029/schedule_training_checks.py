import datetime,json,pathlib,subprocess,sys,time
folder=pathlib.Path('/home/param/pose-runs/duplex-distill-7n-shared10-mbs2-20261029/verification/monitor-30min')
minutes=sys.argv[1:] or ['18','23','28','33','38','43']
schedule=[datetime.datetime.fromisoformat('2026-10-29T11:'+minute+':22+00:00') for minute in minutes]
statefile='schedule-state-extension.json' if sys.argv[1:] else 'schedule-state.json'
state={'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':__import__('os').getpid(),'checks':[],'state':'running','training_will_not_be_stopped':True}
def save():
 tmp=folder/(statefile+'.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(folder/statefile)
save()
for due in schedule:
 while (remaining:=(due-datetime.datetime.now(datetime.timezone.utc)).total_seconds())>0:time.sleep(min(20,remaining))
 result=subprocess.run(['/usr/bin/python3','/home/param/pose-runs/duplex-distill-ops-20261029/monitor_live_training.py',due.isoformat()],capture_output=True,text=True,timeout=50)
 entry={'scheduled_utc':due.isoformat(),'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'returncode':result.returncode,'stderr':result.stderr[-2000:]}
 if result.returncode==0:
  try:entry['summary']=json.loads(result.stdout)
  except json.JSONDecodeError:entry['output']=result.stdout[-3000:]
 else:entry['output']=result.stdout[-3000:]
 state['checks'].append(entry);save();print(json.dumps(entry),flush=True)
state['state']='complete';state['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
