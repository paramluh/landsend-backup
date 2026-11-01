import json,pathlib,subprocess,os,time
root=pathlib.Path(__file__).parent/'validation1'; deadline=time.monotonic()+600
while time.monotonic()<deadline:
 p=root/'collected-status.json'
 if p.exists() and len(json.load(open(p))['completed'])==3:break
 time.sleep(5)
else:raise TimeoutError('Collector still pending; inspect exact live process before restarting')
env=os.environ.copy();env['PYTHONPATH']=str(root.parent/'vendor')+':/home/param/pose-worktrees/pose-opd-credit'
subprocess.run(['/home/param/pose-repo/.venv/bin/python','-m','finetune.pose_opd_credit.pad_sweep','report','--root',str(root/'collected'),'--output',str(root/'report'),'--silero-package',str(root.parent/'vendor/silero_vad')],cwd='/home/param/pose-worktrees/pose-opd-credit',env=env,check=True)
r=json.load(open(root/'report/report.json'))
print(json.dumps([{ 'bias':c['partner_pad_bias'], **c['silero_vad']} for c in r['candidates']],indent=2),flush=True)
