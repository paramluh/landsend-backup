"""Pose-light LR ownership and exact scheduled Adam continuation after saving."""
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from finetune.duplex_distill.checkpoints import save_checkpoint, resume_checkpoint
from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.schedule import apply_learning_rate, resolve_schedule
from finetune.pose_opd_credit.train import optimizer_parameter_groups


class TinyStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(2, 2)])
        self.tower_s = nn.Linear(2, 2)
        self.tower_p = nn.Linear(2, 2)
        self.dec_s = nn.Linear(2, 2)
        self.dec_p = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
        self.decp_bb = nn.Linear(2, 2)
        self.inp = nn.ModuleDict({'pose': nn.Embedding(2, 2)})


class OptimizerGroupsTests(unittest.TestCase):
    def setUp(self):
        self.config = TrainingConfig(learning_rate=3e-6, pose_light_learning_rate=8e-6,
                                     learning_rate_schedule='amu_opd')

    def test_only_pose_light_receives_higher_lr_and_every_parameter_is_owned_once(self):
        model = TinyStudent().double()
        groups = optimizer_parameter_groups(model, self.config)
        owners = {id(p): group for group in groups for p in group['params']}
        self.assertEqual(sum(len(group['params']) for group in groups), len(owners))
        self.assertEqual(set(owners), {id(p) for p in model.parameters()})
        for name, parameter in model.named_parameters():
            expected = 8e-6 if name.startswith('dec_p.') else 3e-6
            self.assertEqual(owners[id(parameter)]['peak_lr'], expected, name)
        optimizer = torch.optim.AdamW(groups, betas=(.9, .95), eps=1e-8, weight_decay=0.)
        before = {id(p): p.detach().clone() for p in model.parameters()}
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        for name, parameter in model.named_parameters():
            expected = 8e-6 if name.startswith('dec_p.') else 3e-6
            torch.testing.assert_close(before[id(parameter)] - parameter,
                                       torch.full_like(parameter, expected), atol=1e-12, rtol=0.)

    def test_shared_schedule_preserves_ratio_at_every_step_and_resume_cursor(self):
        model = TinyStudent()
        optimizer = torch.optim.AdamW(optimizer_parameter_groups(model, self.config))
        schedule = resolve_schedule(self.config, 123)
        for step in range(124):
            base = apply_learning_rate(optimizer, 3e-6, schedule, step)
            self.assertAlmostEqual(optimizer.param_groups[0]['lr'], base)
            self.assertAlmostEqual(optimizer.param_groups[1]['lr'] / base, 8 / 3)
        self.assertAlmostEqual(optimizer.param_groups[0]['lr'], 3e-7)
        self.assertAlmostEqual(optimizer.param_groups[1]['lr'], 8e-7)

    def test_saved_optimizer_resumes_exactly_with_module_peaks_and_adam_moments(self):
        model = TinyStudent()
        optimizer = torch.optim.AdamW(optimizer_parameter_groups(model, self.config),
                                     betas=(.9, .95), eps=1e-8, weight_decay=0.)
        schedule = resolve_schedule(self.config, 123)

        def update(student, opt, cursor):
            apply_learning_rate(opt, 3e-6, schedule, cursor)
            opt.zero_grad(set_to_none=True)
            sum(p.square().sum() for p in student.parameters()).backward()
            opt.step()

        update(model, optimizer, 59)
        with tempfile.TemporaryDirectory() as directory:
            saved = save_checkpoint(model, {'main': optimizer}, Path(directory), 60,
                                    {'checkpoint_reserve_gb': 0, 'keep_latest_optimizer_state': True})
            restored = TinyStudent()
            restored_optimizer = torch.optim.AdamW(optimizer_parameter_groups(restored, self.config))
            resume_checkpoint(restored, {'main': restored_optimizer}, saved['resume_path'], 'cpu')
            self.assertEqual([g['peak_lr'] for g in restored_optimizer.param_groups], [3e-6, 8e-6])
            self.assertEqual([g['name'] for g in restored_optimizer.param_groups], ['base', 'pose_light'])
            update(model, optimizer, 60)
            update(restored, restored_optimizer, 60)
            for actual, expected in zip(restored.parameters(), model.parameters(), strict=True):
                torch.testing.assert_close(actual, expected, atol=0., rtol=0.)

    def test_legacy_config_keeps_one_group_and_original_parameter_order(self):
        model = TinyStudent()
        groups = optimizer_parameter_groups(model, TrainingConfig())
        self.assertEqual(len(groups), 1)
        self.assertEqual([id(p) for p in groups[0]['params']], [id(p) for p in model.parameters()])
        with self.assertRaisesRegex(ValueError, 'dec_p'):
            optimizer_parameter_groups(nn.Linear(2, 2), self.config)


if __name__ == '__main__':
    unittest.main()
