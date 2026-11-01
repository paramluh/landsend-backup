"""CPU checks for changing save cadence while preserving exact training resume."""
import ast
import copy
import math
import json
from pathlib import Path
import tomllib
import unittest

from distillation.joint_runtime import checkpoint_compatibility as compatibility


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ast.parse((ROOT / 'learn_dmd_joint.py').read_text())
# Exercise the learner's actual pure functions without importing its GPU stack.
FUNCTIONS = [node for node in SOURCE.body if isinstance(node, ast.FunctionDef)
             and node.name == 'validate_config']
NAMESPACE = {'math': math}
exec(compile(ast.Module(body=FUNCTIONS, type_ignores=[]), 'learn_dmd_joint.py', 'exec'), NAMESPACE)
validate_config = NAMESPACE['validate_config']


class CheckpointCadenceTests(unittest.TestCase):
    def setUp(self):
        self.config = tomllib.loads((ROOT / 'experiments/distillation/joint-nine-ref.toml').read_text())

    def signature(self):
        return {'schema': 'joint-nine-ref-v2-paired-batches', 'world_size': 40,
                'teacher_sha256': 'teacher', 'latent_hw': [30, 16], 'epoch_steps': 168284,
                'data': {'csv_sha256': 'studio-csv', 'reference_bank_validation': 'bank'},
                'training': copy.deepcopy(self.config)}

    def test_positive_integer_intervals_and_production_default(self):
        self.assertEqual(self.config['save_every_n_steps'], 1000)
        for interval in (1, 200, 1000):
            with self.subTest(interval=interval):
                validate_config({**self.config, 'save_every_n_steps': interval})
        for interval in (0, -1, True, False, 1000., '1000', None, math.nan, math.inf):
            with self.subTest(interval=interval):
                with self.assertRaisesRegex(ValueError, 'save_every_n_steps must be a positive integer'):
                    validate_config({**self.config, 'save_every_n_steps': interval})

    def test_saved_old_cadence_and_new_runtime_policy_are_compatible(self):
        old = self.signature()
        old['training']['save_every_n_steps'] = 200
        old['training'].pop('checkpoint_keep_last', None)
        old['training']['wandb'] = {'enabled': False}
        new = self.signature()
        new['training']['resume_from'] = '/run/completed-checkpoint'
        before = copy.deepcopy(old)
        self.assertEqual(compatibility(old), compatibility(new))
        self.assertEqual(old, before)  # Old checkpoint metadata is never rewritten.
        self.assertNotIn('save_every_n_steps', compatibility(new)['training'])

    def test_training_and_identity_changes_remain_incompatible(self):
        baseline = self.signature()
        changes = {
            ('training', 'optimizer', 'lr'): 0.125,
            ('training', 'optimizer', 'weight_decay'): 0.5,
            ('training', 'optimizer', 'betas'): [0.5, 0.75],
            ('training', 'stabilizer_history_dropout'): [0.5],
            ('training', 'resolver_history_dropout'): [0.5],
            ('training', 'global_batch_size'): 80,
            ('training', 'micro_batch_size_per_gpu'): 2,
            ('training', 'critic_updates'): 2,
            ('training', 'critic_warmup_steps'): 0,
            ('training', 'lr_warmup_steps'): 999,
            ('training', 'critic_lr_warmup_steps'): 999,
            ('training', 'epochs'): 2,
            ('training', 'epoch'): 1,
            ('training', 'seed'): 43,
            ('training', 'objective_weights'): {'stabilizer': 0.5, 'resolver': 0.5},
            ('training', 'ema_decay'): 0.5,
            ('training', 'teacher_guidance_scale'): 1.,
            ('training', 'train_csv'): 'different.csv',
            ('training', 'root_dir'): '/different-data',
            ('training', 'pose_latents'): 'old-pose.safetensors',
            ('training', 'mimi_codes'): 'different-audio.safetensors',
            ('training', 'model', 'checkpoint_path'): '/different-teacher',
            ('training', 'model', 'audio', 'num_codebooks'): 2,
            ('training', 'refs', 'bank'): '/different-bank',
            ('training', 'eval_weights'): 'both',
            ('training', 'sensitivity_frames'): 16,
            ('world_size',): 8,
            ('teacher_sha256',): 'different-teacher',
            ('latent_hw',): [60, 32],
            ('epoch_steps',): 168285,
            ('schema',): 'different-schema',
            ('data',): {'csv_sha256': 'different-csv'},
        }
        for path, value in changes.items():
            with self.subTest(path=path):
                changed = copy.deepcopy(baseline)
                target = changed
                for key in path[:-1]:
                    target = target[key]
                self.assertNotEqual(target.get(path[-1]), value)
                target[path[-1]] = value
                self.assertNotEqual(compatibility(baseline), compatibility(changed))

    def save_at(self, rounds, *, interval=1000, manual=False, final=False, warming=False):
        # Execute the existing step-boundary trigger and acknowledgement block.
        due = next(node for node in ast.walk(SOURCE) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'due'
                           for target in node.targets))
        save_if = next(node for node in ast.walk(SOURCE) if isinstance(node, ast.If)
                       and any(isinstance(child, ast.Name) and child.id == 'due'
                               for child in ast.walk(node.test)))
        calls = []
        namespace = {'full_gt': False,
                     'counters': {'rounds': rounds, 'epoch_step': 168284 if final else rounds - 150},
                     'config': {'save_every_n_steps': interval}, 'warming': warming,
                     'epoch_steps': 168284, 'request': {'save': manual}, 'run_dir': '/run',
                     'save_checkpoint': lambda: calls.append('save') or '/run/completed-checkpoint',
                     'root_action': lambda action: action(),
                     'acknowledge_save_request': lambda *_: calls.append('acknowledge')}
        exec(compile(ast.Module(body=[due, save_if], type_ignores=[]),
                     'learn_dmd_joint.py', 'exec'), namespace)
        return calls

    def test_total_round_cadence_has_no_student_or_resume_offset(self):
        for interval in (200, 1000):
            with self.subTest(interval=interval):
                self.assertEqual(self.save_at(interval - 1, interval=interval), [])
                self.assertEqual(self.save_at(interval, interval=interval), ['save', 'acknowledge'])
        for rounds in (1200, 1201, 1999, 2150):
            self.assertEqual(self.save_at(rounds), [])
        self.assertEqual(self.save_at(2000), ['save', 'acknowledge'])

    def test_manual_and_final_saves_do_not_require_interval_boundary(self):
        self.assertEqual(self.save_at(1201, manual=True), ['save', 'acknowledge'])
        self.assertEqual(self.save_at(168434, final=True), ['save', 'acknowledge'])
        self.assertEqual(self.save_at(123, manual=True, warming=True), ['save', 'acknowledge'])
        self.assertEqual(self.save_at(123, final=True, warming=True), [])

    def test_recovered_evaluation_event_follows_verified_publication_before_rng_restore(self):
        # Execute the actual startup branch with CPU stubs, including its next
        # RNG-restore statement. A publication failure must prevent both the
        # completion event and entry into the restored training stream.
        main = next(node for node in SOURCE.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        index = next(index for index, node in enumerate(main.body)
                     if isinstance(node, ast.If) and any(isinstance(child, ast.Constant)
                         and child.value == 'checkpoint_evaluation_recovered' for child in ast.walk(node)))
        block = ast.Module(body=main.body[index:index + 2], type_ignores=[])
        counters = {'rounds': 1314, 'student_updates': 1164, 'critic_updates': 5256}
        receipt = {'verified_round': 1314, 'verified_sensitivity_metrics': 22,
                   'sensitivity_report_uploads_verified': True, 'publish_token': 'fresh'}
        for fail_publish in (False, True):
            with self.subTest(fail_publish=fail_publish):
                calls, events = [], []
                class Tracker:
                    def evaluation(self, directory, actual_counters, *, startup):
                        self.assertions = (str(directory), actual_counters, startup)
                        calls.append('publish')
                        if fail_publish:
                            raise RuntimeError('remote verification failed')
                        return receipt
                tracker = Tracker()
                def root_action(action):
                    # First collective checks whether evaluation already exists.
                    if not calls:
                        calls.append('exists')
                        return False
                    return action()
                def emit(text, **kwargs):
                    calls.append('event')
                    events.append(json.loads(text))
                namespace = {'resume': '/run/checkpoint', 'root_action': root_action, 'Path': Path,
                    'evaluate': lambda path: calls.append('evaluate') or [{'file': 'sensitivity-raw.json'}],
                    'write_json_atomic': lambda path, payload: calls.append('write'),
                    'metadata': {'checkpoint_id': 'identity'}, 'root_tracking_action': lambda action: action(),
                    'tracker': tracker, 'counters': counters, 'rank': 0, 'json': json, 'print': emit,
                    'restore_rng': lambda state: calls.append('restore_rng'), 'training_rng': 'restored'}
                if fail_publish:
                    with self.assertRaisesRegex(RuntimeError, 'remote verification failed'):
                        exec(compile(block, 'learn_dmd_joint.py', 'exec'), namespace)
                    self.assertEqual(calls, ['exists', 'evaluate', 'write', 'publish'])
                    self.assertEqual(events, [])
                else:
                    exec(compile(block, 'learn_dmd_joint.py', 'exec'), namespace)
                    self.assertEqual(calls, ['exists', 'evaluate', 'write', 'publish', 'event', 'restore_rng'])
                    self.assertEqual(events, [{'event': 'checkpoint_evaluation_recovered',
                        'path': '/run/checkpoint', 'counters': counters,
                        'checkpoint_id': 'identity', 'tracking': receipt}])
                    self.assertEqual(tracker.assertions, ('/run/checkpoint', counters, True))


if __name__ == '__main__':
    unittest.main()
