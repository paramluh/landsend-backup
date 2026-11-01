import json
from pathlib import Path
import tempfile
import unittest

import numpy as np


def _worker(rank, rendezvous, result_dir):
    import torch
    import torch.distributed as dist
    from finetune.pose_opd_credit.train import gather_actor_rows

    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=4)
    try:
        pairs = [dist.new_group([0, 1], backend='gloo'), dist.new_group([2, 3], backend='gloo')]
        payload = [({'owner': rank, 'slot': slot, 'values': np.arange(257, dtype=np.float32) + 1000 * rank + slot}, None)
                   for slot in range(6)]
        for width, leader, group in ((2, rank // 2 * 2, pairs[rank // 2]), (4, 0, None)):
            gathered = gather_actor_rows(payload, rank=rank, leader=leader, group_width=width, group=group)
            if rank == leader:
                assert len(gathered) == width
                for owner, rows in enumerate(gathered, start=leader):
                    assert len(rows) == 6
                    for slot, (row, latent) in enumerate(rows):
                        assert row['owner'] == owner and row['slot'] == slot and latent is None
                        np.testing.assert_array_equal(row['values'], np.arange(257, dtype=np.float32) + 1000 * owner + slot)
            else:
                assert gathered is None
            received = [None]
            dist.scatter_object_list(received, gathered, src=leader, group=group)
            for (actual, latent), (expected, _) in zip(received[0], payload, strict=True):
                assert latent is None and actual['owner'] == rank and actual['slot'] == expected['slot']
                np.testing.assert_array_equal(actual['values'], expected['values'])
        Path(result_dir, f'rank{rank}.json').write_text(json.dumps({'rank': rank, 'passed': True,
                                                                'cuda_initialized': torch.cuda.is_initialized()}))
    finally:
        dist.destroy_process_group()


class ActorGatherTests(unittest.TestCase):
    def test_real_gloo_preserves_subgroup_rank_order_and_scatter_ownership(self):
        import torch.multiprocessing as mp
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=(str(Path(directory, 'rendezvous')), directory), nprocs=4, join=True)
            records = [json.loads(Path(directory, f'rank{rank}.json').read_text()) for rank in range(4)]
            self.assertTrue(all(record['passed'] and not record['cuda_initialized'] for record in records))


if __name__ == '__main__':
    unittest.main()
