"""Focused CPU checks for distributed assignments, preflight, receipts, and cleanup."""
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

import distributed_encode as encode
from encode_bank import VAE_SHA256, sha256
from finalize_check import reference


class DistributedEncodeChecks(unittest.TestCase):
    def test_all_forty_shards_partition_the_same_ordered_groups(self):
        jobs = encode.assignments(list(encode.DEFAULT_NODES), 8)
        self.assertEqual([job['shard'] for job in jobs], list(range(40)))
        groups = list(range(103))
        assigned = [groups[job['shard']::job['shards']] for job in jobs]
        self.assertEqual(sorted(value for shard in assigned for value in shard), groups)
        self.assertEqual(len({(job['node'], job['gpu']) for job in jobs}), 40)
        with self.assertRaises(ValueError):
            encode.assignments(['same', 'same'], 8)

    def test_preflight_requires_same_inputs_versions_and_pinned_vae(self):
        fingerprint = {'sha256': 'implementation', 'inputs': {'versions': {'Pillow': 'version'}}}
        valid = {'node': 'node1', 'selection_sha256': 'selection', 'encoder_fingerprint': fingerprint,
                 'vae_sha256': VAE_SHA256}
        encode.check_preflights([valid], 'selection', fingerprint)
        for change in ({'selection_sha256': 'other'}, {'encoder_fingerprint': {'sha256': 'other'}},
                       {'vae_sha256': None}, {'vae_sha256': 'wrong'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                encode.check_preflights([{**valid, **change}], 'selection', fingerprint)
        encode.check_preflights([{**valid, 'vae_sha256': None}], 'selection', fingerprint, require_vae=False)

    def test_aggregate_checks_every_shard_and_group_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection = root / 'selection.parquet'
            pd.DataFrame([reference()]).to_parquet(selection, index=False)
            selected_hash = sha256(selection)
            fingerprint = {'sha256': 'implementation'}
            manifest = {'run_id': 'run', 'nodes': [{'node': node} for node in encode.DEFAULT_NODES],
                        'assignments': encode.assignments(list(encode.DEFAULT_NODES), 8),
                        'selection': str(selection), 'selection_sha256': selected_hash,
                        'output_dir': str(root), 'encoder_fingerprint': fingerprint}
            actor = reference()['uuid']
            latent = root / 'bank' / actor / 'property_1/latents.safetensors'
            latent.parent.mkdir(parents=True)
            latent.write_bytes(b'checksum-checked-placeholder')
            receipt = root / 'encoding-checkpoints' / f'bank__{actor}__1.json'
            encode.write_json(receipt, {'sha256': sha256(latent), 'selection_sha256': selected_hash,
                                       'vae_sha256': VAE_SHA256, 'encoder_fingerprint': fingerprint, 'n': 1})
            for rank, node in enumerate(encode.DEFAULT_NODES):
                encode.write_json(root / f'node-{rank:02d}.json', {
                    'run_id': 'run', 'node_rank': rank, 'node': node, 'complete': True,
                    'selection_sha256': selected_hash, 'encoder_fingerprint': fingerprint,
                    'shards': [{**job, 'returncode': 0} for job in manifest['assignments'] if job['node_rank'] == rank]})
            self.assertEqual(encode.aggregate(manifest, root)['shards'], 40)
            node_path = root / 'node-00.json'
            original = json.loads(node_path.read_text())
            for change in ({'run_id': 'stale'}, {'shards': original['shards'][:-1]},
                           {'shards': original['shards'] + [original['shards'][0]]}):
                encode.write_json(node_path, {**original, **change})
                with self.subTest(change=change), self.assertRaises(ValueError):
                    encode.aggregate(manifest, root)
            encode.write_json(node_path, original)
            latent.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'group checkpoint'):
                encode.aggregate(manifest, root)

    def test_disconnect_stops_owned_cpu_children(self):
        # Exercise the real process-group cleanup without importing/loading a VAE or using CUDA.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / 'encode_bank.py'
            script.write_text('import time\ntime.sleep(60)\n')
            encode.write_json(root / 'source-hashes.json', {script.name: sha256(script)})
            selection = root / 'selection'
            selection.write_text('selection')
            vae = root / 'vae'
            vae.write_text('test')
            fingerprint = {'sha256': 'test'}
            manifest = {'run_id': 'owned-run', 'snapshot': str(root), 'encoder_root': str(root),
                        'encoder_fingerprint': fingerprint, 'selection': str(selection),
                        'selection_sha256': sha256(selection), 'nodes': [{'node': 'test', 'vae_path': str(vae)}],
                        'assignments': encode.assignments(['test'], 2), 'python': sys.executable,
                        'source_root': str(root), 'output_dir': str(root), 'batch_size': 64}
            encode.write_json(root / 'manifest.json', manifest)
            encode.write_json(root / 'lease.json', {'run_id': 'owned-run', 'time': encode.time.time()})
            handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)}
            try:
                with patch.object(encode, 'encoder_fingerprint', return_value=fingerprint), \
                     patch.object(encode, 'sha256', side_effect=lambda path: VAE_SHA256 if Path(path) == vae else sha256(path)), \
                     patch.object(encode, 'gpu_memory', return_value={0: 0, 1: 0}), \
                     patch.object(encode.select, 'select', return_value=([sys.stdin], [], [])), \
                     patch.object(encode.os, 'read', return_value=b''):
                    with self.assertRaisesRegex(RuntimeError, 'disconnected'):
                        encode.node_worker(SimpleNamespace(manifest=root / 'manifest.json', node_rank=0))
            finally:
                for sig, handler in handlers.items():
                    signal.signal(sig, handler)
            report = json.loads((root / 'node-00.json').read_text())
            self.assertFalse(report['complete'])
            self.assertEqual(len(report['shards']), 2)
            for child in report['shards']:
                with self.assertRaises(ProcessLookupError):
                    os.kill(child['pid'], 0)


if __name__ == '__main__':
    unittest.main()
