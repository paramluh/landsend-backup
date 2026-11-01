"""Packed-profile integration: combined denominators and truthful row counts."""
import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import torch

from finetune.duplex_distill.train import batch_summary
from finetune.duplex_distill.tests.test_profile_packing import row


class ProfileMicrobatchTests(unittest.TestCase):
    def test_actual_prepare_many_sums_global_counts_before_reweighting(self):
        a, b = row(5, 7), row(7, 13)
        b['pose_mask'] = torch.zeros(7, dtype=torch.bool)
        b['pdepth'].fill_(-1)
        b['pcb0'].fill_(-100)
        b['plight'] = torch.empty(0, dtype=torch.long)
        roles = ('student', 'audio_teacher', 'pose_teacher')
        first = {role: a for role in roles}
        second = {role: b for role in roles}
        prepare = Mock(side_effect=[(first, torch.tensor([7., 9., 4., 2., 2.])),
                                    (second, torch.tensor([13., 11., 0., 3., 0.]))])
        path = Path(__file__).resolve().parents[1] / 'runtime.py'
        function = next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, ast.FunctionDef) and node.name == 'prepare_many')
        namespace = {'torch': torch, 'prepare': prepare,
                     'profile': SimpleNamespace(span=lambda _: nullcontext())}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        packed, counts = namespace['prepare_many'](['first', 'second'], 'cpu', ['latent1', 'latent2'], 2)
        torch.testing.assert_close(counts, torch.tensor([20., 20., 4., 5., 2.]))
        for role in roles:
            torch.testing.assert_close(packed[role]['text_weight'], torch.full((8,), .1))
            torch.testing.assert_close(packed[role]['speech_weight'], torch.full((12,), .1))
            torch.testing.assert_close(packed[role]['pose_weight'], torch.tensor([.5] * 5 + [0.] * 7))
            self.assertEqual(packed[role]['segs'], [5, 7])
        self.assertEqual(prepare.call_args_list[0].args, ('first', 'cpu', 'latent1', 2))
        self.assertEqual(prepare.call_args_list[1].args, ('second', 'cpu', 'latent2', 2))

    def test_metadata_reports_both_rows_and_preserves_single_row_shape(self):
        a, b = row(5, 7), row(7, 13)
        a['conditioning'] = {'voice_clip': 0}
        b['conditioning'] = {'voice_clip': 1}
        b['plight'] = torch.empty(0, dtype=torch.long)
        single = batch_summary({'student': a})
        packed = batch_summary([{'student': a}, {'student': b}])
        self.assertEqual(single['conditioning'], {'voice_clip': 0})
        self.assertEqual(packed['conditioning'], [{'voice_clip': 0}, {'voice_clip': 1}])
        self.assertEqual(packed['row_count'], 2)
        self.assertEqual(packed['timeline_frames'], 12)
        self.assertEqual(packed['per_row_frames'], [5, 7])
        self.assertEqual(packed['text_positions'], 8)
        self.assertEqual(packed['speech_depth_frames'], 2)
        self.assertEqual(packed['pose_depth_frames'], 1)


if __name__ == '__main__':
    unittest.main()
