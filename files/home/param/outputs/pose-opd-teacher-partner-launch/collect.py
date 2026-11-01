"""Collect completed benchmark logs and verify reusable trajectories and samples."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import subprocess

from common import HOSTS, RUN, SSH, remote, write_json

BASE = Path(__file__).resolve().parent
EVIDENCE = BASE / 'snapshot-1793450446728000042' / 'runtime'

script = r'''
import hashlib,json,pathlib,socket
root=pathlib.Path('/home/param/runs/pose-runs/teacher-partner-b320-v1');host=socket.gethostname().split('.')[0]
finished=json.loads((root/'supervisor/finished.json').read_text());result={'host':host,'finished':finished}
for name in ['supervisor','torchrun']:
 rec=json.loads((root/'supervisor'/(name+'.json')).read_text());p=pathlib.Path('/proc/%d/stat'%rec['pid'])
 result[name+'_alive']=p.exists() and int(p.read_text().rsplit(')',1)[1].split()[19])==rec['start_ticks']
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
if finished['exit_code']==0:
 cache=root/'rollout-cache';complete=json.loads((cache/'COMPLETE').read_text());result['cache_global_receipts']=len(complete['receipts'])
 assert len(complete['receipts'])==40
 receipts=[]
 for marker in sorted(cache.glob('rank-*.complete.json')):
  entry=json.loads(marker.read_text());file=cache/entry['file']
  assert file.stat().st_size==entry['bytes'] and digest(file)==entry['sha256']
  assert complete['receipts'][entry['rank']]==entry
  receipts.append({'rank':entry['rank'],'bytes':entry['bytes'],'sha256_verified':True})
 assert len(receipts)==8
 result['cache_local_shards']=receipts
 manifest=root/'raw-rollouts/update-000001/manifest.json'
 if host=='landsend-2':
  samples=json.loads(manifest.read_text());assert len(samples['samples'])==4
  for entry in samples['samples']:
   file=manifest.parent/entry['file'];assert file.stat().st_size==entry['bytes'] and digest(file)==entry['sha256']
  result['four_samples_verified']=[{'id':s['id'],'frames':s['frames'],'prefill_frames':s['prefill_frames']} for s in samples['samples']]
  result['result']=json.loads((root/'RESULT.json').read_text())
  assert result['result']['next_update']==1 and result['result']['checkpointing_enabled'] is False
result['old_step89_complete']=pathlib.Path('/home/param/models/trained-weights/pose-opd-credit-2min-b320-attempt2/checkpoints/step-000089/resume/COMPLETE').is_file()
print(json.dumps(result))
'''


def collect(host):
    verified = json.loads(remote(host, ['python3', '-B', '-c', script]).stdout)
    destination = EVIDENCE / host
    destination.mkdir(exist_ok=True)
    includes = ['events-rank*.jsonl', 'metrics.jsonl', 'RESULT.json', 'config.json', 'plan.json',
                'data-selection.json', 'supervisor/', 'supervisor/*.json', 'supervisor/training.log',
                'supervisor/worker.log', 'raw-rollouts/', 'raw-rollouts/update-000001/',
                'raw-rollouts/update-000001/manifest.json', 'rollout-cache/', 'rollout-cache/COMPLETE',
                'rollout-cache/*.complete.json']
    command = ['rsync', '-a', '-e', shlex.join(SSH)]
    for pattern in includes:
        command += ['--include', '/' + pattern]
    command += ['--exclude', '*', host + ':' + RUN + '/', str(destination) + '/']
    subprocess.run(command, check=True, capture_output=True)
    write_json(destination / 'verification.json', verified)
    return verified


if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=5) as pool:
        verified = list(pool.map(collect, HOSTS))
    profiles = []
    for host in HOSTS:
        for path in (EVIDENCE / host).glob('events-rank*.jsonl'):
            for line in path.read_text().splitlines():
                event = json.loads(line)
                if event['event'] in ('profile', 'actor_memory', 'optimizer_parked', 'optimizer_restored',
                                      'rollout_cache_saved', 'rollout_samples_saved', 'update'):
                    profiles.append({'host': host, **event})
    summary = {'hosts': verified, 'profiles': profiles,
               'telemetry_correction': 'The immutable benchmark final profile actor_instances field says16 due to a reporting bug. Actual teacher-partner actor groups contain4GPUs and32 conversations per team, 10teams and320rows globally. The initial step event and actor allocation are correct. Local follow-up source fixes this report to group_width*actor_batch.'}
    write_json(EVIDENCE / 'FINAL-EVIDENCE.json', summary)
    print(json.dumps({'hosts': len(verified), 'profiles': len(profiles), 'evidence': str(EVIDENCE / 'FINAL-EVIDENCE.json')}))
