"""CPU checks for run coverage, checkpoint identity and native actor integration."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.train import (
    REPO, args_parser, canonical_json, checkpoint_spec, make_plan, source_digest,
    validate_resume, write_training_metadata, validate_benchmark_microbatches, warm_optimizer_state,
)


class PlanTests(unittest.TestCase):
    def test_every_epoch_covers_every_row_once_and_only_pads_tail(self):
        items = [(0, row, row % 2) for row in range(11)]
        plan = make_plan(items, world=4, epochs=3, seed=81)
        self.assertEqual(len(plan['updates']), 9)
        for epoch in range(3):
            updates = plan['updates'][epoch * 3:(epoch + 1) * 3]
            real = [tuple(item) for update in updates for item in update if item is not None]
            self.assertCountEqual(real, items)
            self.assertEqual(len(real), 11)
            self.assertEqual(sum(item is None for update in updates for item in update), 1)
            self.assertIsNone(updates[-1][-1])
            self.assertTrue(all(item is not None for update in updates[:-1] for item in update))
        self.assertEqual(plan, make_plan(items, world=4, epochs=3, seed=81))
        self.assertNotEqual(plan['digest'], make_plan(items, world=4, epochs=3, seed=82)['digest'])

    def test_invalid_and_duplicate_items_rejected(self):
        for kwargs in ({'world': 0}, {'world': 2.0}, {'epochs': 0}, {'seed': -1}):
            options = dict(world=2, epochs=1, seed=81); options.update(kwargs)
            with self.subTest(options=options), self.assertRaises(ValueError):
                make_plan([(0, 1, 0)], **options)
        with self.assertRaisesRegex(ValueError, 'unique'):
            make_plan([(0, 1, 0), (0, 1, 0)], world=2, epochs=1, seed=81)


class IdentityTests(unittest.TestCase):
    def identity(self):
        return {'method': TrainingConfig().to_dict(), 'source_sha256': 'source', 'plan_digest': 'plan'}

    def saved(self):
        return {'cfg': {'opd_identity': canonical_json(self.identity())}, 'step': 4,
                'runtime_state': {'next_update': 4, 'plan_digest': 'plan'},
                'world': 2, 'hosts': ['host', 'host']}

    def test_resume_roundtrip_handles_tuple_weights_and_checks_cursor(self):
        self.assertIsInstance(self.identity()['method']['codebook_weights'], tuple)
        saved = json.loads(json.dumps(self.saved()))
        self.assertEqual(validate_resume(saved, self.identity(), 20, ['host', 'host']), 4)
        for change in ('method', 'source', 'plan', 'cursor', 'host', 'outside'):
            bad = copy.deepcopy(saved)
            if change == 'method':
                bad['cfg']['opd_identity']['method']['lookback_frames'] = 11
            elif change == 'source':
                bad['cfg']['opd_identity']['source_sha256'] = 'changed'
            elif change == 'plan':
                bad['runtime_state']['plan_digest'] = 'changed'
            elif change == 'cursor':
                bad['runtime_state']['next_update'] = 5
            elif change == 'host':
                bad['hosts'][1] = 'other'
            else:
                bad['step'] = 21
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_resume(bad, self.identity(), 20, ['host', 'host'])

    def test_run_identity_never_overwritten_and_updates_require_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = make_plan([(0, 0, 0)], world=2, epochs=1, seed=81)
            identity = self.identity()
            write_training_metadata(root, identity, plan, [])
            before = (root / 'config.json').read_bytes()
            changed = copy.deepcopy(identity); changed['method']['lookback_frames'] = 1
            with self.assertRaisesRegex(ValueError, 'different training identity'):
                write_training_metadata(root, changed, plan, [], resume=True)
            self.assertEqual((root / 'config.json').read_bytes(), before)
            (root / 'metrics.jsonl').write_text('{"update":1}\n')
            with self.assertRaisesRegex(ValueError, 'recorded updates'):
                write_training_metadata(root, identity, plan, [])
            write_training_metadata(root, identity, plan, [], resume=True)

    def test_source_digest_includes_checkpoint_and_engine_dependencies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = ('finetune/pose_opd_credit/train.py', 'finetune/duplex_distill/checkpoints.py',
                     'finetune/duplex_distill/production/model_ppm.py', 'infer/engine/ppm/fast_ppm.py',
                     'infer/engine/ppm/configs/Qwen/config.json', 'finetune/storage_paths.py')
            for name in paths:
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('initial')
            baseline = source_digest(root)
            for name in paths:
                path = root / name; path.write_text('changed')
                self.assertNotEqual(source_digest(root), baseline, name)
                path.write_text('initial')
            (root / 'finetune/pose_opd_credit/README.md').write_text('documentation only')
            self.assertEqual(source_digest(root), baseline)


class MetadataTests(unittest.TestCase):
    def export(self, root, name, cfg, **metadata):
        path = Path(root) / name; path.mkdir()
        (path / 'COMPLETE').write_text('complete\n')
        if cfg.get('vadapter'):
            metadata.setdefault('vgain_theta_fp32', -5.5)
        (path / 'meta.json').write_text(json.dumps(dict(step=50, cfg=cfg, **metadata)))
        return path

    def test_each_teacher_uses_own_alpha_and_adapter_and_local_exports_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            student = self.export(tmp, 'student', {'vadapter': {'norm': 'none'}, 'decp_bb_alpha': 2.})
            audio = self.export(tmp, 'audio', {'vadapter': {'norm': 'gain'}})
            pose = self.export(tmp, 'pose', {'decp_bb_alpha': 2., 'decp_bb_fade_from': 0, 'decp_bb_fade_steps': 100})
            s, a, p = (checkpoint_spec(path, role) for path, role in
                       ((student, 'student'), (audio, 'audio_teacher'), (pose, 'pose_teacher')))
            self.assertEqual(s['decp_bb_alpha'], 2.)
            self.assertEqual(p['decp_bb_alpha'], 1.)
            self.assertEqual(s['vadapter'], {'norm': 'none'})
            self.assertEqual(a['vadapter'], {'norm': 'gain'})
            self.assertIsNone(p['vadapter'])
            self.assertIn('meta.json', p['metadata_sha256'])
            before = p['metadata_sha256']
            doc = json.loads((pose / 'meta.json').read_text())
            doc['decp_bb_alpha'] = .75  # Already-resolved export value wins over fade schedule.
            (pose / 'meta.json').write_text(json.dumps(doc))
            after = checkpoint_spec(pose, 'pose_teacher')
            self.assertEqual(after['decp_bb_alpha'], .75)
            self.assertNotEqual(before, after['metadata_sha256'])

    def test_missing_adapter_or_metadata_does_not_silently_inherit_student(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = self.export(tmp, 'missing', {})
            with self.assertRaisesRegex(ValueError, 'trained Voxtral adapter'):
                checkpoint_spec(missing, 'audio_teacher')
            (missing / 'COMPLETE').unlink()
            with self.assertRaisesRegex(ValueError, 'complete export'):
                checkpoint_spec(missing, 'pose_teacher')

    def test_gain_is_preflighted_and_sidecar_is_part_of_resume_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.export(tmp, 'student', {'vadapter': {'norm': 'none'}})
            doc = json.loads((path / 'meta.json').read_text())
            del doc['vgain_theta_fp32']
            (path / 'meta.json').write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'exact fp32 audio gain'):
                checkpoint_spec(path, 'student')
            exported = checkpoint_spec(path, 'audio_teacher', gain_policy='exported')
            self.assertIsNone(exported['vgain_theta_fp32'])
            self.assertEqual(exported['gain_policy'], 'exported')
            with self.assertRaisesRegex(ValueError, 'Only the audio teacher'):
                checkpoint_spec(path, 'student', gain_policy='exported')
            sidecar = path / 'vgain_theta_fp32.json'
            sidecar.write_text(json.dumps({'vgain_theta_fp32': -5.6}))
            before = checkpoint_spec(path, 'student')
            self.assertEqual(before['vgain_theta_fp32'], -5.6)
            sidecar.write_text(json.dumps({'vgain_theta_fp32': -5.7}))
            self.assertNotEqual(checkpoint_spec(path, 'student')['metadata_sha256'], before['metadata_sha256'])
            doc['cfg']['mot'] = {'model': 'MoTVox'}
            (path / 'meta.json').write_text(json.dumps(doc))
            with self.assertRaisesRegex(ValueError, 'MoT needs its own teacher adapter'):
                checkpoint_spec(path, 'audio_teacher')


class EntrypointTests(unittest.TestCase):
    def test_profiles_are_explicitly_bounded(self):
        required = ['--config', 'recipe.json', '--run-dir', '/tmp/test-pose-opd']
        self.assertEqual(args_parser(required + ['--no-save', '--max-updates', '2']).max_updates, 2)
        ladder = args_parser(required + ['--no-save', '--max-updates', '2', '--benchmark-microbatches', '2,4'])
        self.assertEqual(ladder.benchmark_microbatches, [2, 4])
        self.assertTrue(args_parser(required + ['--no-save', '--max-updates', '1', '--warm-optimizer']).warm_optimizer)
        profile = args_parser(required + ['--no-save', '--max-updates', '1', '--profile-policy-agreement',
                                        '--benchmark-prefix-frames', '128'])
        self.assertEqual(profile.policy_agreement_frames, 128)
        self.assertEqual(profile.benchmark_prefix_frames, 128)
        validate_benchmark_microbatches(TrainingConfig(physical_microbatch=2, effective_batch_per_rank=4), [2, 4])
        for schedule in ([1, 4], [2, 3]):
            with self.subTest(schedule=schedule), self.assertRaises(ValueError):
                validate_benchmark_microbatches(TrainingConfig(physical_microbatch=2, effective_batch_per_rank=4), schedule)
        for extra in (['--no-save'], ['--max-updates', '0'], ['--prepare-only', '--resume', '/tmp/save'],
                      ['--prepare-only', '--max-updates', '2'],
                      ['--max-updates', '2', '--benchmark-microbatches', '2,4'],
                      ['--no-save', '--max-updates', '2', '--benchmark-microbatches', '2'],
                      ['--no-save', '--max-updates', '2', '--benchmark-microbatches', '2,0'],
                      ['--no-save', '--max-updates', '2', '--benchmark-microbatches', '2,x'],
                      ['--max-updates', '1', '--warm-optimizer'], ['--warm-optimizer'],
                      ['--benchmark-prefix-frames', '128'],
                      ['--no-save', '--max-updates', '1', '--benchmark-prefix-frames', '128', '--resume', '/tmp/save'],
                      ['--policy-agreement-frames', '128'],
                      ['--profile-policy-agreement', '--policy-agreement-frames', '0'],
                      ['--profile-policy-agreement', '--prepare-only'],
                      ['--no-save', '--max-updates', '1', '--warm-optimizer', '--resume', '/tmp/save']):
            with contextlib.redirect_stderr(io.StringIO()), self.subTest(extra=extra), self.assertRaises(SystemExit):
                args_parser(required + extra)

    def test_optimizer_warmup_allocates_exact_moments_without_weight_change(self):
        import torch
        model = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.Linear(4, 3))
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6, weight_decay=0., fused=False)
        receipt = warm_optimizer_state(optimizer)
        self.assertEqual(receipt['optimizer_step'], 1)
        for name, parameter in model.named_parameters():
            self.assertTrue(torch.equal(parameter.detach().view(torch.int32), before[name].view(torch.int32)))
            self.assertIsNone(parameter.grad)
            state = optimizer.state[parameter]
            self.assertEqual(float(state['step']), 1.)
            self.assertEqual(torch.count_nonzero(state['exp_avg']).item(), 0)
            self.assertEqual(torch.count_nonzero(state['exp_avg_sq']).item(), 0)
        with self.assertRaisesRegex(ValueError, 'fresh state'):
            warm_optimizer_state(optimizer)
        unsafe = torch.optim.AdamW(model.parameters(), weight_decay=.1, fused=False)
        with self.assertRaisesRegex(ValueError, 'zero weight decay'):
            warm_optimizer_state(unsafe)

    def test_actual_native_engine_shares_model_and_accepts_constructor_contract(self):
        # Fresh process avoids polluting sibling tests' production module arena.
        code = '''
import inspect
from finetune.pose_opd_credit import runtime
from finetune.pose_opd_credit.train import load_actor_native
from finetune.pose_opd_credit.rollout import make_fastppm_class
model = runtime.load_native()
native = load_actor_native(model)
assert native.M is model and native.IP.M is model
inspect.signature(native.FastPPM.__init__).bind(None, None, None, None, None,
    n_inst=1, cfg_rows=False, max_s=600., p_max=128, mimi_stream=False, front=False)
actor = make_fastppm_class(native)
assert issubclass(actor, native.FastPPM)
assert native.CD == 2 and native.DEPTH == 32 and native.CBS == 2048
print("native interface OK")
'''
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='')
        result = subprocess.run([sys.executable, '-c', code], cwd=REPO, env=env,
                                capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('native interface OK', result.stdout)


if __name__ == '__main__':
    unittest.main()
