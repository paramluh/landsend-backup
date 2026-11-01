"""Verify complete source synchronization without contacting a remote host."""
import unittest
from unittest import mock

from finetune.duplex_distill import cluster_ops


class ClusterSyncTests(unittest.TestCase):
    def test_sync_includes_shared_storage_helper_and_excludes_run_artifacts(self):
        with mock.patch.object(cluster_ops, 'local_host', return_value='landsend-0'), \
                mock.patch.object(cluster_ops, 'remote') as remote, \
                mock.patch.object(cluster_ops.subprocess, 'run') as run:
            self.assertEqual(cluster_ops.sync('landsend-1'), 'landsend-1')
        commands = [call.args[0] for call in run.call_args_list]
        helper = cluster_ops.REPO / 'finetune/storage_paths.py'
        helper_commands = [command for command in commands if str(helper) in command]
        self.assertEqual(len(helper_commands), 1)
        self.assertEqual(helper_commands[0][-2:], [str(helper), f'landsend-1:{helper}'])
        remote.assert_any_call('landsend-1', 'mkdir', '-p', helper.parent)
        for directory in (cluster_ops.HERE, cluster_ops.REPO / 'finetune/transcript_pose/environment'):
            self.assertTrue(any(command[-2:] == [str(directory) + '/', f'landsend-1:{directory}/']
                                for command in commands))
        for command in commands:
            self.assertIn('--exclude=benchmarks/', command)
            self.assertIn('--exclude=__pycache__/', command)
            self.assertNotIn('--delete', command)

    def test_controller_does_not_copy_or_contact_itself(self):
        with mock.patch.object(cluster_ops, 'local_host', return_value='landsend-0'), \
                mock.patch.object(cluster_ops, 'remote') as remote, \
                mock.patch.object(cluster_ops.subprocess, 'run') as run:
            self.assertEqual(cluster_ops.sync('landsend-0'), 'landsend-0')
        remote.assert_not_called()
        run.assert_not_called()
