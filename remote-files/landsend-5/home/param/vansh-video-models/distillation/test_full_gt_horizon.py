"""Student-update limits preserve headstart, LR schedule, resume, and save timing."""
import ast
import copy
import math
from pathlib import Path
import tomllib
import unittest

import torch

from distillation.joint_runtime import (checkpoint_compatibility, full_gt_checkpoint_due,
                                       full_gt_evaluation_due, planned_student_steps, training_indices)
from distillation.resolver_data import learning_rate_factor


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ast.parse((ROOT / 'learn_dmd_joint.py').read_text())
MAIN = next(node for node in SOURCE.body if isinstance(node, ast.FunctionDef) and node.name == 'main')


class FullGTHorizonTests(unittest.TestCase):
    def test_requested_configuration_preserves_learning_rates_and_headstart(self):
        config = tomllib.loads((ROOT / 'experiments/distillation/full-gt-nine-ref.toml').read_text())
        self.assertEqual(config['max_student_steps'], 8000)
        self.assertEqual(config['save_every_n_steps'], 800)
        self.assertEqual(config['checkpoint_keep_last'], 8)
        self.assertEqual(config['critic_warmup_steps'], 150)
        self.assertEqual(config['lr_warmup_steps'], 100)
        self.assertEqual(config['critic_lr_warmup_steps'], 200)
        self.assertEqual(config['optimizer']['lr'], 2e-6)
        self.assertEqual(config['lr_final_ratio'], .1)
        validate = next(node for node in SOURCE.body if isinstance(node, ast.FunctionDef) and node.name == 'validate_config')
        ns = {'math': math}
        exec(compile(ast.Module(body=[validate], type_ignores=[]), '<validate>', 'exec'), ns)
        ns['validate_config'](config, full_gt=True)
        for limit in (0, -1, True, 8000., '8000'):
            with self.subTest(limit=limit), self.assertRaisesRegex(ValueError, 'max_student_steps'):
                ns['validate_config']({**config, 'max_student_steps': limit}, full_gt=True)

    def test_limit_keeps_full_epoch_identity_and_exact_resume_cursor(self):
        epoch_steps = 168284
        self.assertEqual(planned_student_steps(epoch_steps, 8000), 8000)
        self.assertEqual(planned_student_steps(epoch_steps), epoch_steps)
        self.assertEqual(planned_student_steps(5, 8000), 5)
        warm, student = training_indices(epoch_steps, {'warmup_step': 0, 'epoch_step': 0},
                                         150, max_student_steps=8000)
        self.assertEqual(warm, list(range(150)))
        self.assertEqual(student, range(8000))
        warm, student = training_indices(epoch_steps, {'warmup_step': 73, 'epoch_step': 0},
                                         150, max_student_steps=8000)
        self.assertEqual(warm, list(range(73, 150)))
        self.assertEqual(student, range(8000))
        warm, student = training_indices(epoch_steps, {'warmup_step': 150, 'epoch_step': 7998},
                                         150, max_student_steps=8000)
        self.assertEqual(warm, [])
        self.assertEqual(list(student), [7998, 7999])
        self.assertEqual(training_indices(epoch_steps, {'warmup_step': 150, 'epoch_step': 8000},
                                          150, max_student_steps=8000), ([], range(8000, 8000)))
        with self.assertRaisesRegex(ValueError, 'Resume student cursor'):
            training_indices(epoch_steps, {'warmup_step': 150, 'epoch_step': 8001},
                             150, max_student_steps=8000)
        # Benchmark continues to exercise its requested real full-joint rounds.
        self.assertEqual(training_indices(3, {'warmup_step': 0, 'epoch_step': 0}, 150, 6,
                                          max_student_steps=1), ([], [0, 1, 2, 0, 1, 2]))

    def test_actual_trainer_lr_totals_use_student_horizon_and_resume_exactly(self):
        assignments = [node for node in MAIN.body if isinstance(node, ast.Assign)
                       and any(isinstance(target, ast.Name) and target.id in ('warmup', 'total_rounds', 'totals')
                               for target in node.targets)]
        ns = {'config': {'critic_warmup_steps': 150, 'critic_updates': 4},
              'epoch_steps': 168284, 'student_steps': 8000}
        exec(compile(ast.Module(body=assignments, type_ignores=[]), '<totals>', 'exec'), ns)
        self.assertEqual(ns['total_rounds'], 8150)
        self.assertEqual(ns['totals'], {'student': 8000, 'critic': 32600})
        for role, warmup in (('student', 100), ('critic', 200)):
            total = ns['totals'][role]
            self.assertAlmostEqual(learning_rate_factor(0, total, warmup), 1 / warmup)
            self.assertEqual(learning_rate_factor(total - 1, total, warmup), .1)
            def create():
                optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(()))], lr=2e-6)
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                    lambda step: learning_rate_factor(step, total, warmup))
                return optimizer, scheduler
            uninterrupted, schedule = create()
            for _ in range(800):
                uninterrupted.step()
                schedule.step()
            optimizer_state, scheduler_state = copy.deepcopy(uninterrupted.state_dict()), copy.deepcopy(schedule.state_dict())
            resumed, resumed_schedule = create()
            resumed.load_state_dict(optimizer_state)
            resumed_schedule.load_state_dict(scheduler_state)
            for _ in range(3):
                self.assertEqual(uninterrupted.param_groups[0]['lr'], resumed.param_groups[0]['lr'])
                uninterrupted.step(); schedule.step()
                resumed.step(); resumed_schedule.step()

    def test_saves_at_student_800_multiples_plus_final_manual_without_warmup_offset(self):
        for student in (0, 1, 649, 650, 799, 801, 1599, 7999):
            self.assertFalse(full_gt_checkpoint_due({'student_updates': student, 'rounds': student + 150},
                                                   800, 8000, warming=False))
        for student in range(800, 8001, 800):
            self.assertTrue(full_gt_checkpoint_due({'student_updates': student, 'rounds': student + 150},
                                                  800, 8000, warming=False))
        self.assertFalse(full_gt_checkpoint_due({'student_updates': 0, 'rounds': 150}, 800, 8000, warming=True))
        self.assertTrue(full_gt_checkpoint_due({'student_updates': 1750}, 800, 1750, warming=False))
        # Exercise the trainer's actual full-GT override followed by manual-save handling.
        override = next(node for node in ast.walk(MAIN) if isinstance(node, ast.If)
                        and isinstance(node.test, ast.Name) and node.test.id == 'full_gt'
                        and any(isinstance(child, ast.Name) and child.id == 'full_gt_checkpoint_due'
                                for child in ast.walk(node)))
        save_if = next(node for node in ast.walk(MAIN) if isinstance(node, ast.If)
                       and any(isinstance(child, ast.Name) and child.id == 'due' for child in ast.walk(node.test)))
        for student, warming, manual, expected in ((650, False, False, []), (800, False, False, ['save', 'ack']),
                                                  (0, True, False, []), (0, True, True, ['save', 'ack'])):
            calls = []
            ns = {'full_gt': True, 'full_gt_checkpoint_due': full_gt_checkpoint_due,
                  'full_gt_evaluation_due': full_gt_evaluation_due,
                  'counters': {'student_updates': student}, 'config': {'save_every_n_steps': 800},
                  'student_steps': 8000, 'warming': warming, 'due': True, 'request': {'save': manual},
                  'run_dir': '/run', 'save_checkpoint': lambda: calls.append('save'),
                  'root_action': lambda action: action(), 'acknowledge_save_request': lambda *args: calls.append('ack')}
            exec(compile(ast.Module(body=[override, save_if], type_ignores=[]), '<save>', 'exec'), ns)
            self.assertEqual(calls, expected)

    def test_resume_signature_pins_horizon_but_allows_save_cadence_change(self):
        original = {'schema': 'full-gt-nine-ref-v1-single-target', 'epoch_steps': 168284,
                    'training': {'max_student_steps': 8000, 'save_every_n_steps': 800}}
        changed = copy.deepcopy(original)
        changed['training']['save_every_n_steps'] = 400
        self.assertEqual(checkpoint_compatibility(original), checkpoint_compatibility(changed))
        changed['training']['max_student_steps'] = 9000
        self.assertNotEqual(checkpoint_compatibility(original), checkpoint_compatibility(changed))


if __name__ == '__main__':
    unittest.main()
