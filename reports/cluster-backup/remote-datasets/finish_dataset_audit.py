import subprocess,time,sys,json
from pathlib import Path
root=Path('/home/param/outputs/landsend-backup-20261031/remote-datasets')
while True:
 with (root/'finalize.log').open('w') as log:r=subprocess.run([sys.executable,str(root/'finalize_verification.py')],stdout=log,stderr=subprocess.STDOUT)
 if r.returncode==0:break
 if r.returncode!=2:print((root/'finalize.log').read_text());raise SystemExit(r.returncode)
 time.sleep(10)
print((root/'finalize.log').read_text(),flush=True)
with (root/'final-publication.log').open('w') as log:r=subprocess.run([sys.executable,str(root/'publish_final_evidence.py')],stdout=log,stderr=subprocess.STDOUT)
print((root/'final-publication.log').read_text(),flush=True);raise SystemExit(r.returncode)
