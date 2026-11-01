"""Sequential node refresh: ordinary artifacts followed by excluded source tarballs."""
import json,socket,subprocess,time
from pathlib import Path
BASE=Path('/home/param/outputs/landsend-backup-20261031');PY='/home/param/pose-repo/.venv/bin/python'
results=[]
for name in ['remote_orphan_worker.py','backup_source_archives.py']:
 started=time.time();result=subprocess.run([PY,'-u',str(BASE/'orphans'/name),'--delta'])
 results.append({'worker':name,'returncode':result.returncode,'started_unix':started,'finished_unix':time.time()})
 if result.returncode:break
receipt={'state':'complete' if all(x['returncode']==0 for x in results) and len(results)==2 else 'failed','host':socket.gethostname(),'workers':results}
(BASE/'final-artifact-refresh'/(socket.gethostname()+'.finished.json')).write_text(json.dumps(receipt,indent=2)+'\n')
raise SystemExit(0 if receipt['state']=='complete' else 1)
