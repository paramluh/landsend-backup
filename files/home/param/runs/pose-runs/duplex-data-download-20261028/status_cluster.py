#!/usr/bin/env python3
"""Read compact dataset download/preparation status on landsend-1 through -4."""
import concurrent.futures
import json
from pathlib import Path
import shlex
import subprocess

REMOTE = '''import json,pathlib
r=pathlib.Path('/home/param/pose-runs/duplex-data-download-20261028')
def get(p):
 return json.loads(p.read_text()) if p.exists() else {}
a=get(r/'status.json');p=get(r/'pose-download/status.json');s=get(r/'preparation-status.json');x=get(r/'pose-download/preparation-status.json');ready=get(r/'DATASET_READY.json');err=get(r/'finalize-error.json')
print(json.dumps(dict(host=a.get('hostname'),audio_download=a.get('download_state'),audio_verified_GB=round(a.get('bytes_verified',0)/1e9,2),audio_total_GB=round(a.get('bytes_total',0)/1e9,2),audio_failures=len(a.get('failed',[])),samsep=s.get('phase'),pose_download=p.get('download_state'),pose_verified_GB=round(p.get('bytes_verified',0)/1e9,2),pose_failures=len(p.get('failed',[])),pose_preparation=x.get('phase'),pose_tracks=x.get('tracks'),training_ready=ready.get('training_ready',False),finalize_error=err)))
'''


def read(host):
    command = ['/home/param/pose-repo/.venv/bin/python', '-c', REMOTE]
    p = subprocess.run(['ssh', '-i', '/home/param/.ssh/id_ed25519_cluster', '-o', 'BatchMode=yes',
                        '-o', 'ConnectTimeout=10', host, shlex.join(command)], capture_output=True, text=True, timeout=30)
    if p.returncode:
        return dict(host=host, error=p.stderr.strip())
    return json.loads(p.stdout)


if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        snapshot = list(pool.map(read, [f'landsend-{i}' for i in range(1, 5)]))
    print(json.dumps(snapshot, indent=2))
    Path(__file__).with_name('latest-cluster-status.json').write_text(json.dumps(snapshot, indent=2) + '\n')
