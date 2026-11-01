"""CPU tests of data-dependent resharding and exact resume integrity."""
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

from finetune.duplex_distill.checkpoints import (
    _check_keys, _safe_checkpoint_root, _validate_resume_directory,
    load_checkpoint, resume_checkpoint, save_checkpoint,
)
from finetune.duplex_distill import checkpoints as checkpoints_module
from finetune import storage_paths


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(3, 5), nn.Linear(5, 2)])
        self.register_buffer('scale', torch.tensor(1.25))

    def forward(self, x):
        return self.layers[1](self.layers[0](x).tanh()) * self.scale


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(32)
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_production_rank_export_reshards_without_source_world_match(self):
        source = TinyModel()
        path = self.root / 'source'; path.mkdir()
        world = 4  # Includes empty final shards for small tensors, like world64.
        shards = [{} for _ in range(world)]
        rep = {}
        for name, tensor in source.state_dict().items():
            if tensor.ndim == 0:
                rep[name] = tensor.to(torch.bfloat16)
            else:
                width = (tensor.shape[0] + world - 1) // world
                for rank in range(world):
                    shards[rank][name] = tensor[rank * width:(rank + 1) * width].to(torch.bfloat16)
        for rank, shard in enumerate(shards):
            torch.save(shard, path / f'rank{rank}.pt')
        torch.save(rep, path / 'rep.pt')
        (path / 'COMPLETE').write_text('10')
        (path / 'meta.json').write_text(json.dumps({'format': 'ppm-local-bf16-shards-v1', 'world': world, 'step': 10}))
        target = TinyModel()
        result = load_checkpoint(target, path, 'cpu')
        self.assertEqual(result['world'], 1)
        self.assertEqual(result['source_world'], 4)
        for name, value in target.state_dict().items():
            torch.testing.assert_close(value, source.state_dict()[name].bfloat16().float(), atol=0, rtol=0)

    def test_dcp_loading_restores_persistent_buffers_and_exact_gain(self):
        import torch.distributed.checkpoint as dcp
        class WithGain(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(3, 2)
                self.vgain = nn.Module()
                self.vgain.theta = nn.Parameter(torch.tensor(-5.625))
                self.vadapter = nn.Module()
                self.vadapter.register_buffer('target_rms', torch.tensor(float('nan')))
                self.register_buffer('persistent', torch.tensor([1.2, 3.4]))
        source = WithGain()
        path = self.root / 'dcp'
        dcp.save({k: v.bfloat16() for k, v in source.state_dict().items()}, checkpoint_id=path)
        (path / 'COMPLETE').write_text('249')
        (path / 'train_cfg.json').write_text(json.dumps({'step': 249}))
        gain = -5.621907711029053
        (path / 'vgain_theta_fp32.json').write_text(json.dumps({'vgain_theta_fp32': gain}))
        target = WithGain()
        target.persistent.zero_()
        result = load_checkpoint(target, path, 'cpu')
        self.assertEqual(float(target.vgain.theta.detach()), gain)
        self.assertEqual(result['exact_gain']['vgain_theta_fp32'], gain)
        torch.testing.assert_close(target.persistent, source.persistent.bfloat16().float(), atol=0, rtol=0)
        (path / 'vgain_theta_fp32.json').unlink()
        with self.assertRaisesRegex(ValueError, 'lacks exact fp32 gain'):
            load_checkpoint(target, path, 'cpu')

    def test_smoke_skips_only_deeper_layer_keys(self):
        expected = {'layers.0.weight', 'layers.1.weight', 'tower_s.layers.0.weight', 'head.weight'}
        source = expected | {'layers.2.weight', 'tower_s.layers.1.weight'}
        with self.assertRaises(ValueError):
            _check_keys(expected, source, smoke=False)
        self.assertEqual(set(_check_keys(expected, source, smoke=True)), source - expected)
        with self.assertRaises(ValueError):
            _check_keys(expected, source | {'head.extra'}, smoke=True)
        with self.assertRaises(ValueError):
            _check_keys(expected, source | {'layers.0.extra'}, smoke=True)
        with self.assertRaises(ValueError):
            _check_keys(expected | {'missing'}, source, smoke=True)

    def test_resume_restores_exact_optimizer_next_step_and_rng(self):
        source = TinyModel()
        opt = torch.optim.AdamW(source.parameters(), lr=0.004, betas=(0.8, 0.97))
        x = torch.randn(4, 3)
        opt.zero_grad(); source(x).square().mean().backward(); opt.step()
        random.seed(17); np.random.seed(18); torch.manual_seed(19)
        saved = save_checkpoint(source, {'main': opt}, self.root, 1, {'run_tag': 'cpu-proof', '_runtime_state': {'decoder_steps': 12}, 'checkpoint_reserve_gb': 0})
        expected_rng = (random.random(), float(np.random.rand()), torch.rand(4))
        expected_state = {k: v.clone() for k, v in source.state_dict().items()}
        target = TinyModel()
        target_opt = torch.optim.AdamW(target.parameters(), lr=0.2)
        meta = resume_checkpoint(target, {'main': target_opt}, saved['resume_path'], 'cpu')
        self.assertEqual(meta['runtime_state']['decoder_steps'], 12)
        actual_rng = (random.random(), float(np.random.rand()), torch.rand(4))
        self.assertEqual(actual_rng[:2], expected_rng[:2])
        torch.testing.assert_close(actual_rng[2], expected_rng[2], atol=0, rtol=0)
        for name, value in target.state_dict().items():
            torch.testing.assert_close(value, expected_state[name], atol=0, rtol=0)
        self.assertEqual(target_opt.param_groups[0]['lr'], 0.004)
        for model, optimizer in ((source, opt), (target, target_opt)):
            optimizer.zero_grad(); model(x).square().mean().backward(); optimizer.step()
        for a, b in zip(source.parameters(), target.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        # A node-local export clearly records its layout, and reloads for the
        # same world/host map without pretending to be a shared-filesystem DCP.
        export = TinyModel()
        load_checkpoint(export, saved['export_path'], 'cpu')
        for name, value in export.state_dict().items():
            torch.testing.assert_close(value, expected_state[name].bfloat16().float(), atol=0, rtol=0)
        self.assertFalse((Path(saved['export_path']) / '.metadata').exists())
        self.assertTrue((Path(saved['export_path']) / 'NODE_COMPLETE').exists())
        with self.assertRaisesRegex(RuntimeError, 'already exists'):
            save_checkpoint(source, {'main': opt}, self.root, 1, {'checkpoint_reserve_gb': 0})

    def test_resume_detects_corrupt_file_and_wrong_world(self):
        model = TinyModel(); opt = torch.optim.AdamW(model.parameters())
        result = save_checkpoint(model, {'main': opt}, self.root, 0, {'checkpoint_reserve_gb': 0})
        path = Path(result['resume_path']); state = json.loads((path / 'train_state.json').read_text())
        state['world'] = 48; (path / 'train_state.json').write_text(json.dumps(state))
        with self.assertRaisesRegex(ValueError, 'world size'):
            resume_checkpoint(model, {'main': opt}, path, 'cpu')
        state['world'] = 1; (path / 'train_state.json').write_text(json.dumps(state))
        with (path / 'rep.pt').open('ab') as f:
            f.write(b'corruption')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            resume_checkpoint(model, {'main': opt}, path, 'cpu')

    def test_latest_resume_retention_preserves_exports_and_exact_optimizer(self):
        model = TinyModel(); opt = torch.optim.AdamW(model.parameters(), lr=.004)
        cfg = {'checkpoint_reserve_gb': 0, 'keep_latest_optimizer_state': True}
        x = torch.randn(4, 3)
        opt.zero_grad(); model(x).square().mean().backward(); opt.step()
        first = save_checkpoint(model, {'main': opt}, self.root, 1, cfg)
        first_weights = {key: value.clone() for key, value in model.state_dict().items()}
        opt.zero_grad(); model(x).square().mean().backward(); opt.step()
        second = save_checkpoint(model, {'main': opt}, self.root, 2, cfg)
        self.assertTrue(second['retention']['committed'])
        self.assertEqual(len(second['retention']['pruned']), 1)
        self.assertFalse(Path(first['resume_path']).exists())
        self.assertTrue((Path(first['export_path']) / 'NODE_COMPLETE').is_file())
        marker = json.loads((self.root / 'checkpoints/LATEST_RESUME.json').read_text())
        self.assertEqual(marker['step'], 2)
        self.assertEqual(marker['resume_path'], second['resume_path'])
        export = TinyModel(); load_checkpoint(export, first['export_path'], 'cpu')
        for key, value in export.state_dict().items():
            torch.testing.assert_close(value, first_weights[key].bfloat16().float(), atol=0, rtol=0)
        restored = TinyModel(); restored_opt = torch.optim.AdamW(restored.parameters(), lr=.9)
        resume_checkpoint(restored, {'main': restored_opt}, second['resume_path'], 'cpu')
        for m, optimizer in ((model, opt), (restored, restored_opt)):
            optimizer.zero_grad(); m(x).square().mean().backward(); optimizer.step()
        for actual, expected in zip(restored.parameters(), model.parameters()):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_managed_storage_preserves_legacy_resume_identity_and_prunes_safely(self):
        runs = self.root / 'runs'
        legacy = self.root / 'legacy-runs'
        models = self.root / 'models/trained-weights'
        run = runs / 'pose-runs/example'
        roots = tuple(runs / name for name in ('pose-runs', 'smokes', 'failed'))
        with mock.patch.multiple(storage_paths, LEGACY_RUNS=legacy, OUTPUT_RUNS=roots[0],
                                 RUN_ROOTS=roots, MODEL_RUNS=models):
            model = TinyModel(); opt = torch.optim.AdamW(model.parameters())
            cfg = {'checkpoint_reserve_gb': 0, 'keep_latest_optimizer_state': True}
            first = save_checkpoint(model, {'main': opt}, run, 1, cfg)
            target = models / 'example/checkpoints'
            self.assertEqual(Path(first['path']).parent, target)
            self.assertTrue((run / 'checkpoints').is_symlink())
            self.assertEqual((run / 'checkpoints').resolve(), target)
            # Old manifests name the original output root; moving the files
            # does not require rewriting provenance or weight checksums.
            for name in ('resume/train_state.json', 'export/meta.json'):
                path = Path(first['path']) / name
                meta = json.loads(path.read_text())
                meta['run_dir'] = str(legacy / 'example')
                path.write_text(json.dumps(meta))
            second = save_checkpoint(model, {'main': opt}, legacy / 'example', 2, cfg)
            self.assertTrue(second['retention']['committed'])
            self.assertEqual(len(second['retention']['pruned']), 1)
            self.assertFalse(Path(first['resume_path']).exists())
            self.assertTrue(Path(first['export_path']).exists())
            restored = TinyModel(); restored_opt = torch.optim.AdamW(restored.parameters())
            resume_checkpoint(restored, {'main': restored_opt}, second['resume_path'], 'cpu')
            # Classification moves keep the same model namespace.
            classified = roots[1] / 'example'
            classified.parent.mkdir(parents=True)
            run.rename(classified)
            run.symlink_to(classified, target_is_directory=True)
            self.assertEqual(_safe_checkpoint_root(legacy / 'example'), target)

    def test_managed_storage_rejects_redirected_links_and_parent_traversal(self):
        runs = self.root / 'runs'
        roots = tuple(runs / name for name in ('pose-runs', 'smokes', 'failed'))
        models = self.root / 'models/trained-weights'
        run = roots[0] / 'example'; run.mkdir(parents=True)
        outside = self.root / 'outside'; outside.mkdir()
        sentinel = outside / 'keep.txt'; sentinel.write_text('preserve')
        with mock.patch.multiple(storage_paths, LEGACY_RUNS=self.root / 'legacy', OUTPUT_RUNS=roots[0],
                                 RUN_ROOTS=roots, MODEL_RUNS=models):
            (run / 'checkpoints').symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'requires migration'):
                _safe_checkpoint_root(run)
            (run / 'checkpoints').unlink()
            models.mkdir(parents=True)
            (models / 'example').symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                _safe_checkpoint_root(run)
            with self.assertRaisesRegex(ValueError, 'parent traversal'):
                _safe_checkpoint_root(run / '../other')
            self.assertEqual(sentinel.read_text(), 'preserve')

    def test_retention_refuses_symlinks_foreign_paths_and_foreign_manifest(self):
        run = self.root / 'run'
        model = TinyModel(); opt = torch.optim.AdamW(model.parameters())
        cfg = {'checkpoint_reserve_gb': 0, 'keep_latest_optimizer_state': True}
        first = save_checkpoint(model, {'main': opt}, run, 1, cfg)
        old_resume = Path(first['resume_path'])
        outside = self.root / 'outside'; (outside / 'resume').mkdir(parents=True)
        sentinel = outside / 'resume/keep.txt'; sentinel.write_text('outside checkpoint root')
        (run / 'checkpoints/step-000000').symlink_to(outside, target_is_directory=True)
        original = old_resume / 'rep.pt'
        original.unlink(); original.symlink_to(sentinel)
        with mock.patch.object(checkpoints_module.warnings, 'warn') as warning:
            second = save_checkpoint(model, {'main': opt}, run, 2, cfg)
        self.assertTrue(warning.called)
        self.assertTrue(old_resume.is_dir())
        self.assertTrue(original.is_symlink())
        self.assertEqual(sentinel.read_text(), 'outside checkpoint root')
        self.assertEqual(second['retention']['pruned'], [])
        with self.assertRaisesRegex(ValueError, 'Unsafe checkpoint step'):
            _validate_resume_directory(run / 'checkpoints', outside, 'host', 1, ['host'])
        linked_run = self.root / 'linked-run'; linked_run.symlink_to(run, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            _safe_checkpoint_root(linked_run)
        # A validly named directory with a foreign run identity is also retained.
        meta_path = Path(second['resume_path']) / 'train_state.json'
        meta = json.loads(meta_path.read_text()); meta['run_dir'] = str(outside)
        meta_path.write_text(json.dumps(meta))
        with mock.patch.object(checkpoints_module.warnings, 'warn') as warning:
            third = save_checkpoint(model, {'main': opt}, run, 3, cfg)
        self.assertTrue(warning.called)
        self.assertTrue(Path(second['resume_path']).is_dir())
        self.assertTrue(third['retention']['committed'])

    def test_new_verification_or_marker_failure_never_prunes_previous_resume(self):
        model = TinyModel(); opt = torch.optim.AdamW(model.parameters())
        cfg = {'checkpoint_reserve_gb': 0, 'keep_latest_optimizer_state': True}
        first = save_checkpoint(model, {'main': opt}, self.root, 1, cfg)
        original_sha = checkpoints_module._sha256
        def corrupt_new_verification(path):
            path = Path(path)
            if path.parent.name == 'resume' and path.parent.parent.name == 'step-000002' and path.name == 'rank0.pt':
                return '0' * 64
            return original_sha(path)
        with mock.patch.object(checkpoints_module, '_sha256', side_effect=corrupt_new_verification):
            with self.assertRaisesRegex(RuntimeError, 'previous resumes retained'):
                save_checkpoint(model, {'main': opt}, self.root, 2, cfg)
        self.assertTrue(Path(first['resume_path']).is_dir())
        self.assertEqual(json.loads((self.root / 'checkpoints/LATEST_RESUME.json').read_text())['step'], 1)
        with mock.patch.object(checkpoints_module, '_write_latest_marker', side_effect=OSError('marker unavailable')):
            with mock.patch.object(checkpoints_module.warnings, 'warn') as warning:
                third = save_checkpoint(model, {'main': opt}, self.root, 3, cfg)
        self.assertTrue(warning.called)
        self.assertFalse(third['retention']['committed'])
        self.assertTrue(Path(first['resume_path']).is_dir())
        self.assertEqual(json.loads((self.root / 'checkpoints/LATEST_RESUME.json').read_text())['step'], 1)

    def test_prune_failure_warns_and_keeps_new_checkpoint_resumable(self):
        model = TinyModel(); opt = torch.optim.AdamW(model.parameters())
        cfg = {'checkpoint_reserve_gb': 0, 'keep_latest_optimizer_state': True}
        first = save_checkpoint(model, {'main': opt}, self.root, 1, cfg)
        with mock.patch.object(checkpoints_module, '_remove_flat_resume', side_effect=PermissionError('read only')):
            with mock.patch.object(checkpoints_module.warnings, 'warn') as warning:
                second = save_checkpoint(model, {'main': opt}, self.root, 2, cfg)
        self.assertTrue(warning.called)
        self.assertTrue(Path(first['resume_path']).is_dir())
        self.assertTrue(second['retention']['warnings'])
        resume_checkpoint(model, {'main': opt}, second['resume_path'], 'cpu')

    def test_failed_same_step_retries_resume_exactly_and_preserve_exports(self):
        model = TinyModel(); opt = torch.optim.AdamW(model.parameters(), lr=.004)
        x = torch.randn(4, 3)
        opt.zero_grad(); model(x).square().mean().backward(); opt.step()
        cfg = {'checkpoint_reserve_gb': 0, 'keep_latest_optimizer_state': True}
        first = save_checkpoint(model, {'main': opt}, self.root, 1, cfg)
        with mock.patch.object(checkpoints_module, '_atomic_json', side_effect=OSError('metadata unavailable')):
            with self.assertRaises(checkpoints_module.CheckpointSaveError):
                save_checkpoint(model, {'main': opt}, self.root, 2, cfg)
        self.assertTrue((self.root / 'checkpoints/step-000002.tmp').is_dir())
        retry = save_checkpoint(model, {'main': opt}, self.root, 2, cfg, attempt=1)
        self.assertEqual(Path(retry['path']).name, 'step-000002-retry-01')
        self.assertTrue((Path(first['export_path']) / 'NODE_COMPLETE').exists())
        self.assertFalse(Path(first['resume_path']).exists())
        restored = TinyModel(); restored_opt = torch.optim.AdamW(restored.parameters(), lr=.9)
        meta = resume_checkpoint(restored, {'main': restored_opt}, retry['resume_path'], 'cpu')
        self.assertEqual(meta['save_attempt'], 1)
        for current, optimizer in ((model, opt), (restored, restored_opt)):
            optimizer.zero_grad(); current(x).square().mean().backward(); optimizer.step()
        for actual, expected in zip(restored.parameters(), model.parameters()):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        last = save_checkpoint(model, {'main': opt}, self.root, 2, cfg, attempt=2)
        self.assertFalse(Path(retry['resume_path']).exists())
        self.assertTrue((Path(retry['export_path']) / 'NODE_COMPLETE').exists())
        self.assertTrue(last['retention']['committed'])
        for malformed in ('step-000002-retry-00', 'step-000002-retry-1', 'step-000002-retry-001', 'step-000002-retry-01.tmp'):
            self.assertIsNone(checkpoints_module._checkpoint_identity(malformed))


if __name__ == '__main__':
    unittest.main()
