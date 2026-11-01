"""CPU-only launcher regression: SSH is replaced, never contacted."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest


LAUNCHER = Path(__file__).resolve().parents[1] / 'run_dmd_resolver.sh'


class ResolverLauncherTests(unittest.TestCase):
    def launch_arguments(self, optional):
        with tempfile.TemporaryDirectory(prefix='resolver-launcher-test-') as temporary:
            root = Path(temporary)
            binary = root/'bin'
            binary.mkdir()
            capture = root/'capture'
            capture.mkdir()
            ssh = binary/'ssh'
            ssh.write_text(f'#!{sys.executable}\n' + '''import json, os, sys
from pathlib import Path
args = sys.argv[1:]
assert args[:2] == ['-o', 'BatchMode=yes'], args[:2]
assert len(args) == 4, len(args)
Path(os.environ['SSH_CAPTURE_DIR'], args[2] + '.json').write_text(json.dumps(args))
''')
            ssh.chmod(0o755)
            recorder = root/'record_argv.py'
            recorder.write_text('import json, sys\nprint(json.dumps(sys.argv[1:]))\n')
            env = dict(os.environ, PATH=str(binary)+os.pathsep+os.environ['PATH'],
                       SSH_CAPTURE_DIR=str(capture), SLOTS='8', MASTER_ADDR='unit-node-0',
                       MASTER_PORT='29999', IFNAME='unit-interface',
                       WANDB_ENV_FILE=str(root/'absent-credentials'),
                       RESOLVER_RUN_TAG='cpu-regression',
                       LAUNCHER_TEST_PYTHON=sys.executable,
                       LAUNCHER_TEST_ARGV_RECORDER=str(recorder))
            config = str(root/'config with spaces.toml')
            nodes = [f'unit-node-{index}' for index in range(5)]
            subprocess.run(['bash', str(LAUNCHER), ' '.join(nodes), config, *optional],
                           env=env, check=True, capture_output=True, text=True, timeout=10)
            self.assertEqual(sorted(path.stem for path in capture.glob('*.json')), nodes)
            parsed = []
            for index, node in enumerate(nodes):
                captured = json.loads((capture/f'{node}.json').read_text())
                remote = captured[-1]
                match = re.search(r'(?m)^[ \t]*(\S*/torchrun)(?:[ \t]|$)', remote)
                self.assertIsNotNone(match, 'Expected the final torchrun command')
                # Parse only the final argument string with Bash itself. The
                # remote setup, credential sourcing and torchrun are not run.
                arguments = remote[match.end(1):].strip()
                script = ('set -- ' + arguments + '\n'
                          'exec "$LAUNCHER_TEST_PYTHON" "$LAUNCHER_TEST_ARGV_RECORDER" "$@"')
                result = subprocess.run(['bash', '-c', script], env=env, check=True,
                                        capture_output=True, text=True, timeout=10)
                actual = json.loads(result.stdout)
                expected = ['--nnodes=5', f'--node_rank={index}', '--nproc_per_node=8',
                            '--master_addr=unit-node-0', '--master_port=29999',
                            'learn_dmd_resolver.py', '--config', config, *optional]
                self.assertEqual(actual, expected)
                parsed.append(actual)
            return parsed

    def test_zero_optional_arguments_does_not_add_empty_argument(self):
        for arguments in self.launch_arguments([]):
            self.assertNotIn('', arguments)
            self.assertTrue(arguments[-1].endswith('config with spaces.toml'))

    def test_set_value_with_spaces_and_shell_metacharacters_stays_one_argument(self):
        value = 'wandb.run_name=resolver smoke $HOME $(printf unexpected) "quoted"'
        optional = ['--benchmark', '3', '--set', value, '--set', 'history_dropout=[0.1, 0.1, 0.0]']
        for arguments in self.launch_arguments(optional):
            self.assertEqual(arguments[-len(optional):], optional)


if __name__ == '__main__':
    unittest.main()
