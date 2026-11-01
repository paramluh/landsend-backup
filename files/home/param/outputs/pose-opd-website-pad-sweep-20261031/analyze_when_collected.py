import json,pathlib,subprocess,os,time
root=pathlib.Path(__file__).parent; deadline=time.monotonic()+600
while time.monotonic()<deadline:
 p=root/'collected-status.json'
 if p.exists() and len(json.load(open(p))['completed'])==7:break
 time.sleep(5)
else:raise TimeoutError('Collector still pending; inspect exact live process before restarting')
env=os.environ.copy();env['PYTHONPATH']=str(root/'vendor')+':/home/param/pose-worktrees/pose-opd-credit'
subprocess.run(['/home/param/pose-repo/.venv/bin/python','-m','finetune.pose_opd_credit.pad_sweep','report','--root',str(root/'collected'),'--output',str(root/'initial-report'),'--silero-package',str(root/'vendor/silero_vad')],cwd='/home/param/pose-worktrees/pose-opd-credit',env=env,check=True)
r=json.load(open(root/'initial-report/report.json'))
print(json.dumps([{ 'bias':c['partner_pad_bias'], **c['silero_vad']} for c in r['candidates']],indent=2),flush=True)
