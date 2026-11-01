"""CPU-only contract checks. No SSH, GPU calls, or training launches."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import common
import start
import worker

def spec():
    commit = 'a' * 40
    repo = '/home/param/pose-worktrees/pose-opd-credit-production-' + commit
    return {'source_commit': commit, 'repo': repo, 'config': repo + '/finetune/pose_opd_credit/recipe.raw.json',
            'hosts': common.HOSTS, 'run_dir': common.RUN, 'world_size': 40, 'mode': 'production',
            'checkpointing_enabled': True, 'master_addr': '::1', 'port': 29671}

class LaunchTest(unittest.TestCase):
    def test_normal_training_command(self):
        for rank, host in enumerate(common.HOSTS):
            cmd = common.command(spec(), host)
            self.assertIn('--node_rank=' + str(rank), cmd)
            self.assertIn('--nproc_per_node=8', cmd)
            module = cmd.index('finetune.pose_opd_credit.train')
            self.assertEqual(cmd[module + 1:], ['--config', spec()['config'], '--run-dir', common.RUN])
            self.assertFalse(any(value.startswith(('--max-updates', '--no-save', '--benchmark', '--warm-optimizer', '--resume')) for value in cmd))

    def test_rejects_extra_arguments_or_hosts(self):
        for changed in ({'extra_args': ['--no-save']}, {'checkpointing_enabled': False},
                        {'hosts': ['landsend-0']}, {'mode': 'benchmark'}):
            with self.assertRaises(ValueError):
                common.command(dict(spec(), **changed), 'landsend-2')

    def test_any_failed_preflight_starts_nothing(self):
        calls = []
        def remote(host, argv):
            calls.append(argv)
            if argv[0] == 'cat':
                return mock.Mock(stdout=json.dumps(spec()))
            if host == 'landsend-4':
                raise RuntimeError('Existing GPU processes')
            return mock.Mock(stdout=json.dumps({'host': host, 'read_only': True}))
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'spec.json'; path.write_text(json.dumps(spec()))
            with mock.patch('sys.argv', ['start.py', str(path)]), mock.patch.object(start, 'remote', remote):
                with self.assertRaises(RuntimeError):
                    start.main()
            self.assertFalse((Path(root) / 'STARTED.json').exists())
        self.assertTrue(all(argv[0] == 'cat' or '--check' in argv for argv in calls))

    def test_local_dry_command_uses_no_ssh(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'spec.json'; path.write_text(json.dumps(spec()))
            with mock.patch('sys.argv', ['start.py', str(path), '--dry-command']), mock.patch.object(start, 'remote') as remote, contextlib.redirect_stdout(io.StringIO()) as output:
                start.main()
            remote.assert_not_called()
            self.assertEqual(len(json.loads(output.getvalue())), 5)

    def test_gpu_preflight_rejects_compute_processes(self):
        with mock.patch.object(common.subprocess, 'check_output', return_value='12345\n'):
            with self.assertRaisesRegex(RuntimeError, 'Existing GPU'):
                common.gpu_check()

    def test_partial_go_failure_cleans_only_recorded_supervisors(self):
        signals = []
        def remote(host, argv):
            if argv[0] == 'cat':
                return mock.Mock(stdout=json.dumps(spec()))
            if '--check' in argv:
                return mock.Mock(stdout=json.dumps({'host': host, 'read_only': True}))
            code = argv[-1]
            if 'subprocess.Popen' in code:
                return mock.Mock(stdout=json.dumps({'pid': 100 + common.HOSTS.index(host), 'start_ticks': 4321}))
            if 'Supervisor did not become ready' in code:
                return mock.Mock(stdout=json.dumps({'pid': 100 + common.HOSTS.index(host)}))
            if 'os.kill(pid,signal.SIGTERM)' in code:
                self.assertIn('int(fields[19])!=expected', code)
                self.assertIn('expected=4321', code)
                signals.append(host)
                return mock.Mock(stdout='{}')
            if host == 'landsend-4':
                raise RuntimeError('simulated GO network failure')
            return mock.Mock(stdout='')
        value = dict(spec(), config_sha256='test-config-sha')
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'spec.json'; path.write_text(json.dumps(value))
            # The mocked staged specs must be identical to the local launch identity.
            original_remote = remote
            def same_spec(host, argv):
                return mock.Mock(stdout=json.dumps(value)) if argv[0] == 'cat' else original_remote(host, argv)
            with mock.patch('sys.argv', ['start.py', str(path)]), mock.patch.object(start, 'remote', same_spec), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, 'Partial GO delivery failed'):
                    start.main()
            self.assertTrue((Path(root) / 'PARTIAL-GO-FAILURE.json').exists())
            self.assertTrue((Path(root) / 'STARTED.json').exists())
        self.assertCountEqual(signals, common.HOSTS)

if __name__ == '__main__':
    unittest.main()
