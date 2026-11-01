import json,os,re,socket,subprocess,sys
from pathlib import Path
root=Path(sys.argv[1]);job=(root/'cluster/latest').resolve();node=socket.gethostname().split('.')[0]
result={'node':node,'launch':job.name}
for name in ['supervisor','torchrun']:
 p=job/(name+'.json');d=json.loads(p.read_text());pid=d['pid']
 try:
  stat=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split();live=stat[19]==d['start_ticks'] and stat[0]!='Z'
 except FileNotFoundError:live=False
 result[name]={'pid':pid,'live':live}
result['exit_code']=(job/'exit_code').read_text().strip() if (job/'exit_code').exists() else None
result['training_ranks']=0
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  args=p.read_bytes().split(b'\0')
  stat=(p.parent/'stat').read_text().rsplit(')',1)[1].split()
  if int(stat[1])==result['torchrun']['pid'] and b'/home/param/pose-repo/finetune/transcript_pose/train.py' in args and str(root).encode() in args:result['training_ranks']+=1
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
lines=(job/'node.log').read_text(errors='replace').splitlines()
steps=[]
for line in lines:
 m=re.search(r'step (\d+)/200: ([\d.]+)s, peak=([\d.]+)GiB audio=([\d.]+) pose=([\d.]+)',line)
 if m:steps.append({'step':int(m[1]),'seconds':float(m[2]),'peak_gib':float(m[3]),'audio':float(m[4]),'pose':float(m[5])})
result['steps']=steps
result['errors']=[line for line in lines if 'OutOfMemoryError' in line or 'CUDA out of memory' in line or 'Traceback (most recent call last)' in line or 'ChildFailedError' in line][-8:]
result['latest']=(root/'LATEST').read_text().strip() if (root/'LATEST').exists() else None
result['full_checkpoints']=sorted(int(p.name.split('-')[1]) for p in (root/'saves').glob('step-*') if (p/'NODE_COMPLETE').is_file())
result['training_complete']=(root/'TRAINING_COMPLETE').read_text().strip() if (root/'TRAINING_COMPLETE').exists() else None
result['wandb']=(root/'wandb-url').read_text().strip() if (root/'wandb-url').exists() else None
result['save_log']=[line for line in lines if '[save]' in line or '[export-local]' in line or '[eval]' in line][-10:]
print(json.dumps(result))
