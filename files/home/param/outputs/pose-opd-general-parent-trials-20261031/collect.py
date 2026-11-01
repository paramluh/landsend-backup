import json
from pathlib import Path
import shlex
import subprocess
import time

root = Path(__file__).parent
spec = json.loads((root / 'spec-landsend-5.json').read_text())
node, run = spec['node'], spec['run']
ssh = ['ssh', '-i', '/home/param/.ssh/id_ed25519_cluster', '-o', 'BatchMode=yes', node]
code = f"import pathlib; p=pathlib.Path({run!r})/'result.json'; print(p.read_text() if p.exists() else 'pending')"
deadline = time.monotonic() + 600
while time.monotonic() < deadline:
    result = subprocess.check_output(ssh + [shlex.join(['python3', '-c', code])], text=True).strip()
    if result != 'pending':
        status = json.loads(result)
        (root / 'completion.json').write_text(json.dumps(status, indent=2) + '\n')
        if status['codes'] != [0, 0]:
            raise RuntimeError(status)
        break
    time.sleep(5)
else:
    raise TimeoutError('Listening jobs have not finished')
for name in ('first', 'second'):
    target = root / 'collected' / name
    target.mkdir(parents=True, exist_ok=True)
    subprocess.run(['rsync', '-a', '-e', 'ssh -i /home/param/.ssh/id_ed25519_cluster -o BatchMode=yes',
                    f'{node}:{run}/{name}/candidates/', str(target) + '/'], check=True)
    subprocess.run(['scp', '-i', '/home/param/.ssh/id_ed25519_cluster', f'{node}:{run}/{name}.log', str(root / (name + '.log'))], check=True)
print('Collected four conversations at', root / 'collected')
