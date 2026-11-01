import json
import datetime
from pathlib import Path
import tempfile
import unittest

import numpy as np

from finetune.pose_opd_credit.samples import (
    gather_samples, sample_origins, sample_records, sample_slots, save_sample_records, save_samples,
)
from finetune.pose_opd_credit.tests.test_rollout import FakeCachedEngine, seed_row
from finetune.pose_opd_credit.rollout import rollout_row
from finetune.pose_opd_credit.train import make_plan


def _balanced_updates():
    items = [(0, row, row % 2) for row in range(336 + 126)]
    return make_plan(items, world=56, rows_per_rank=6, epochs=1, seed=81,
                     frame_lengths={item: 1500 for item in items})['updates']


def _archive_worker(rank, directory):
    """Exercise real Gloo transport, including a rank with no selected samples."""
    import torch.distributed as dist
    dist.init_process_group('gloo', init_method=f'file://{directory}/rendezvous',
                            rank=rank, world_size=3, timeout=datetime.timedelta(seconds=30))
    try:
        for update, items in enumerate(_balanced_updates(), start=1):
            origins = sample_origins(items, 6, 4)
            slots = sample_slots(items, 6, 4, rank=rank)
            payload, rollouts, waves = {}, {}, {}
            for slot in slots:
                row = seed_row()
                row.update(item=items[rank * 6 + slot], row_key=f'r{rank}-s{slot}',
                           aref='{}', conditioning={}, dummy=False,
                           unused=lambda: None)  # Non-archive data must not be serialized.
                payload[slot] = (row, lambda: None)
                rollouts[slot] = rollout_row(FakeCachedEngine(), row, seed=12 + slot)
                waves[slot] = np.full(30 * 1920, rank + slot / 10, dtype=np.float32)
            records = sample_records(payload, rollouts, waves, slots, rank)
            dist.barrier()
            gathered = gather_samples(records, origins, rank=rank, group=dist.group.WORLD)
            if rank == 0:
                save_sample_records(directory, update=update, records=gathered,
                    origins=origins, fps=12.5, source_identity='distributed-test')
            elif gathered is not None:
                raise AssertionError('Only rank zero should collect the archive')
            # Non-owner ranks can enter the next agreement while transfers run.
            receipts = [None] * 3
            dist.all_gather_object(receipts, rank)
    finally:
        dist.destroy_process_group()


class SampleExportTests(unittest.TestCase):
    def test_full_and_balanced_126_row_tail_choose_four_real_rows(self):
        full, tail = _balanced_updates()
        self.assertEqual(sum(item is not None for item in tail), 126)
        self.assertEqual(sum(item is not None for item in tail[:6]), 3)
        self.assertEqual(sample_origins(full, 6, 4), [(0, 0), (0, 1), (0, 2), (0, 3)])
        self.assertEqual(sample_origins(tail, 6, 4), [(0, 0), (0, 1), (0, 2), (1, 0)])
        self.assertEqual(sample_slots(tail, 6, 4), [0, 1, 2])
        self.assertEqual(sample_slots(tail, 6, 4, rank=1), [0])
        self.assertEqual(sample_slots(tail, 6, 4, rank=2), [])
        self.assertEqual(sample_origins([None] * 6 + [[0, 1, 0]], 6, 1), [(1, 0)])

    def test_remote_tail_sample_is_gathered_and_published_with_exact_origin(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_archive_worker, args=(directory,), nprocs=3, join=True)
            for update, origins in enumerate(([(0, slot) for slot in range(4)],
                                               [(0, 0), (0, 1), (0, 2), (1, 0)]), start=1):
                folder = Path(directory) / 'raw-rollouts' / f'update-{update:06d}'
                manifest = json.loads((folder / 'manifest.json').read_text())
                self.assertEqual(len(manifest['samples']), 4)
                self.assertEqual([(row['rank'], row['slot']) for row in manifest['samples']], origins)
                for entry, (rank, slot) in zip(manifest['samples'], origins, strict=True):
                    self.assertEqual(entry['id'], f'u{update}-r{rank}-s{slot}')
                    self.assertEqual(entry['file'], f'sample-u{update:06d}-r{rank:03d}-s{slot:02d}.npz')
                    self.assertEqual(entry['row_id'], f'r{rank}-s{slot}')
                    with np.load(folder / entry['file'], allow_pickle=False) as arrays:
                        np.testing.assert_array_equal(arrays['input_audio'],
                            np.full(30 * 1920, rank + slot / 10, dtype=np.float32))
                        self.assertEqual(arrays['sampled_audio_mask'].tolist(), [False] * 25 + [True] * 5)

    def test_selection_does_not_sample_or_export_dummies(self):
        state = np.random.get_state()
        self.assertEqual(sample_slots([[1, 2, 0], None, [1, 3, 1], [1, 4, 0]], 4, 2), [0, 2])
        self.assertEqual(np.random.get_state()[0], state[0])
        np.testing.assert_array_equal(np.random.get_state()[1], state[1])
        with self.assertRaises(ValueError):
            sample_slots([[1, 2, 0], None], 2, 2)

    def test_exact_physical_alignment_and_atomic_resume_archive(self):
        row = seed_row()
        row.update(item=[1, 2, 0], row_key='example', aref=json.dumps({'side': 'user'}),
                   conditioning={'mode': 'audio'}, dummy=False)
        rollout = rollout_row(FakeCachedEngine(), row, seed=12)
        wave = np.linspace(-.5, .5, 30 * 1920, dtype=np.float32)
        original_audio = rollout.audio.copy()
        with tempfile.TemporaryDirectory() as directory:
            kwargs = dict(update=1, payload=[(row, None)], rollouts=[rollout], waveforms={0: wave},
                          slots=[0], fps=12.5, source_identity='test')
            receipt = save_samples(directory, **kwargs)
            path = Path(receipt['manifest'])
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest['samples'][0]['frames'], 30)
            with np.load(path.parent / manifest['samples'][0]['file'], allow_pickle=False) as data:
                np.testing.assert_array_equal(data['input_audio'], wave)
                self.assertEqual(data['audio_codes'].shape, (32, 30))
                np.testing.assert_array_equal(data['audio_codes'][:, :25], np.full((32, 25), 4))
                np.testing.assert_array_equal(data['audio_codes'][:, 25:], np.full((32, 5), 11))
                np.testing.assert_array_equal(data['pose_codes'][:, :25], np.full((32, 25), 5))
                np.testing.assert_array_equal(data['pose_codes'][:, 25:], np.full((32, 5), 12))
                self.assertEqual(data['sampled_audio_mask'].tolist(), [False] * 25 + [True] * 5)
                self.assertEqual(data['sampled_text_mask'].tolist(), [False] * 25 + [True] * 5)
            np.testing.assert_array_equal(rollout.audio, original_audio)
            save_samples(directory, **kwargs)
            histories = list((Path(directory) / 'raw-rollouts').glob('history-update-*'))
            self.assertEqual(len(histories), 1)
            self.assertTrue((histories[0] / 'manifest.json').is_file())

    def test_invalid_codes_never_commit_manifest(self):
        row = seed_row()
        row.update(item=[1, 2, 0], row_key='example', aref='{}', conditioning={})
        rollout = rollout_row(FakeCachedEngine(), row, seed=1)
        rollout.pose[-1] = -1
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                save_samples(directory, update=1, payload=[(row, None)], rollouts=[rollout],
                    waveforms={0: np.zeros(30 * 1920, np.float32)}, slots=[0], fps=12.5, source_identity='test')
            self.assertFalse(list((Path(directory) / 'raw-rollouts').glob('update-*/manifest.json')))


if __name__ == '__main__':
    unittest.main()
