from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import subprocess
import time

SUITE = Path('/home/param/runs/pose-runs/pose-jump-audit-suite-20261031')
TOOLS = Path('/home/param/pose-repo/tools/pose_jump_audit')

REMOTE = r'''
import json, subprocess, shutil, time, hashlib
from pathlib import Path
m = payload['model']
root, old = Path(m['root']), Path(m['first_cohort_root'])
root.mkdir(parents=True, exist_ok=True)
for name in ('logs', 'runs', 'shards'):
    (root/name).mkdir(exist_ok=True)
shutil.copytree(old/'source', root/'source', dirs_exist_ok=True)
assert hashlib.sha256((root/'source/request.json').read_bytes()).hexdigest() == m['reference_request_sha256']
assert not (root/'STOP').exists(), 'STOP file exists'
for filename, expected in payload['code_sha256'].items():
    path = Path('/home/param/pose-repo/tools/pose_jump_audit')/filename
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, filename
plans = []
for item in m['launches']:
    command = item['command']
    i = command.index('/home/param/pose-repo/.venv/bin/python')
    result = subprocess.run(command[i:]+['--plan-only'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads((root/'shards'/f"{item['shard']:02d}"/'plan.json').read_text())
    assert plan['shard_count'] == 4 and plan['seed_base'] == 8018027 and plan['batch_size'] == 64
    plans.append({'shard': item['shard'], 'fingerprint': plan['fingerprint'], 'samples': len(plan['sample_indices'])})
assert len({p['fingerprint'] for p in plans}) == 1
occupancy = {
    'checked_at': time.time(),
    'gpus': subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu', '--format=csv,noheader,nounits'], text=True),
    'processes': subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'], text=True),
}
assert not occupancy['processes'].strip(), occupancy
assert all(int(line.split(',')[1].strip()) < 100 for line in occupancy['gpus'].splitlines()), occupancy
(root/'prelaunch-gpu-occupancy.json').write_text(json.dumps(occupancy, indent=2)+'\n')
lineage = {'first_cohort': str(old), 'second_seed_base': 8018027, 'first_seed_base': 8018011,
           'total': 1000, 'draws_disjoint': True, 'shard_count': 4, 'expected_step': m['step'],
           'checker_sha256': payload['detector_sha256']}
(root/'cohort-lineage.json').write_text(json.dumps(lineage, indent=2)+'\n')
(root/'START_GENERATION').write_text('Authorized after all three frozen-r3 first-cohort scans returned zero.\n')
launches = []
for item in m['launches']:
    result = subprocess.run(item['command'], capture_output=True, text=True)
    show = subprocess.run(['systemctl', '--user', 'show', item['unit']+'.service',
                           '--property=ActiveState,SubState,MainPID,Result'], capture_output=True, text=True)
    launches.append(dict(item, returncode=result.returncode, stdout=result.stdout,
                         stderr=result.stderr, unit_status=show.stdout))
(root/f"launches-{m['host']}.json").write_text(json.dumps(launches, indent=2)+'\n')
print(json.dumps({'model': m['model'], 'host': m['host'], 'root': str(root), 'plans': plans,
                  'occupancy': occupancy, 'launches': launches}))
assert all(r['returncode'] == 0 for r in launches), launches
'''


def stage_launch(model, plan):
    subprocess.run(['rsync', '-a', *[str(TOOLS/f) for f in plan['code_sha256']],
                    model['host']+':'+str(TOOLS)+'/'], check=True, capture_output=True, text=True)
    payload = {'model': model, 'code_sha256': plan['code_sha256'], 'detector_sha256': plan['detector_sha256']}
    code = 'payload = '+repr(payload)+'\n'+REMOTE
    result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
                             model['host'], 'python3', '-'], input=code, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(model['host']+' '+result.stdout+' '+result.stderr)
    receipt = json.loads(result.stdout)
    (SUITE/f"cohort2-launch-{model['host']}.json").write_text(json.dumps(receipt, indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    path = SUITE/'second-cohort-launch-plan.json'
    plan = json.loads(path.read_text())
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(stage_launch, model, plan) for model in plan['models']]
        for future in futures:
            receipt = future.result()
            print(json.dumps({'model': receipt['model'], 'host': receipt['host'], 'plans': receipt['plans'],
                              'units': [{'unit': r['unit'], 'status': r['unit_status']} for r in receipt['launches']]}), flush=True)
    plan.update(state='launched', launched_at=time.time())
    path.write_text(json.dumps(plan, indent=2)+'\n')
