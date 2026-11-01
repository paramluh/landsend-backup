import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from safetensors.torch import load_file, save_file

from distillation.compare import VelocityRecorder, common_probes, load_snapshot, metrics, sha256, velocity_comparison
from distillation.export import export_checkpoint
from distillation.lora import inject_lora
from distillation.render_pairs import render_files, sampler_config, verify_render_receipt, write_render_receipt


class CompareTests(unittest.TestCase):
    def test_reuse_requires_identical_inputs_checkpoint_and_output_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {'rows': [8, 1], 'output_dir': str(root), 'seed': 1234}
            inputs = [{'row': 8, 'sha256': {'pose': 'fixed'}}]
            for name in render_files(config['rows']):
                (root / name).write_bytes(name.encode())
            write_render_receipt(config, 'dense350', inputs)
            verify_render_receipt(config, 'dense350', inputs)
            with self.assertRaises(ValueError):
                verify_render_receipt(config, 'another-checkpoint', inputs)
            with self.assertRaises(ValueError):
                verify_render_receipt({**config, 'seed': 9}, 'dense350', inputs)
            (root / 'sample_8.mp4').write_bytes(b'corrupt')
            with self.assertRaises(ValueError):
                verify_render_receipt(config, 'dense350', inputs)

    def test_recorder_covers_real_predictions_and_compact_layout(self):
        from distillation.bimodal_dmd import Rollout, UniformRollout, UniformStabilizerRollout
        from distillation.test_rollout import attention_cpu, small_net

        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, previous)
        torch.manual_seed(13)
        net = small_net().eval().requires_grad_(False)
        net.configure(False, 'none')
        rows = dict(latents=torch.randn(1, 2, 8, 1, 2), pose=torch.randn(1, 2, 8, 1, 2),
                    ofi=torch.arange(8)[None], drop=torch.tensor([False]))
        noise, renoise, tokens = torch.randn_like(rows['latents']), torch.randn_like(rows['latents']), torch.randn(1, 8, 12)
        recordings = []
        with patch('distillation.bimodal_net.flash_attn_func', attention_cpu), \
                patch('models.pose.audio.flash_attn_func', attention_cpu), torch.no_grad():
            for cls in (UniformRollout, UniformStabilizerRollout, Rollout):
                with VelocityRecorder(net, 2, 0, 100) as recorded:
                    cls([2] * 4, 2, .2, .4)(net, rows, tokens, noise, renoise, stab_only=True)
                recordings.append(recorded)
        a, b, trimmed = recordings
        self.assertEqual(a.calls, 16)
        self.assertEqual(len(a.tags), 44)
        self.assertEqual(sum(tag['generated'] for tag in a.tags), 32)
        self.assertEqual({tag['frame'] for tag in a.tags}, set(range(97, 104)))
        self.assertTrue(all(tag['valid'] and tag['slot'] >= 4 for tag in a.tags))
        self.assertEqual(a.tags, b.tags)
        self.assertEqual(a.tags, trimmed.tags)
        torch.testing.assert_close(torch.stack(a.values), torch.stack(b.values), rtol=2e-5, atol=2e-6)
        # Variable pass shapes can change BF16 rounding; the strict fp32 cache/gradient check lives in
        # test_trimmed_rollout. All heads, including seed-frame heads, must still be recorded and aligned.
        self.assertLess(metrics(torch.stack(trimmed.values), torch.stack(a.values))['relative_l2'], 1e-3)
        velocities = {'teacher': torch.stack(a.values), 'dense': torch.stack(a.values), 'speedy': torch.stack(b.values)}
        result = velocity_comparison(velocities, a.tags)
        self.assertEqual(len(result['predictions']), 44)
        self.assertEqual(len(result['calls']), 14)
        self.assertEqual(len(result['frames']), 7)
        self.assertEqual(len(result['times']), 8)
        self.assertLess(result['all']['speedy_vs_dense']['relative_l2'], 2e-5)

    def test_common_input_probes_are_fixed_and_in_band(self):
        raw = {'latents': torch.randn(1, 2, 8, 2, 2)}
        config = dict(history_t=.2, stab_t_min=.22, stab_t_max=.98)
        probes = common_probes(raw, config, 17)
        repeated = common_probes(raw, config, 17)
        for (x, t, history), again in zip(probes, repeated):
            self.assertEqual(tuple(x.shape), (1, 2, 4, 2, 2))
            self.assertEqual(tuple(history.shape), (1, 2, 3, 2, 2))
            self.assertTrue(((t > torch.tensor([.22, .4, .6, .8])) &
                             (t < torch.tensor([.4, .6, .8, .98]))).all())
            for a, b in zip((x, t, history), again):
                self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(probes[0][0], probes[1][0]))

    def test_export_preserves_adapter_function_and_provenance(self):
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.q = nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)

            def original_named_parameters(self):
                return self.named_parameters()

        lora = dict(rank=2, alpha=2, targets=['q'])
        net = Net()
        inject_lora(net, **lora)
        with torch.no_grad():
            net.q.weight.copy_(torch.eye(2))
            net.q.lora_A.copy_(torch.eye(2))
            net.q.lora_B.copy_(torch.eye(2))
        state = {name: value.detach().clone() for name, value in net.original_named_parameters()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / 'checkpoint'
            checkpoint.mkdir()
            saved = checkpoint / 'student.safetensors'
            save_file(state, str(saved))
            (checkpoint / 'critic.safetensors').write_bytes(b'critic')
            (checkpoint / 'metadata.json').write_text(json.dumps({'checkpoint_id': 'test', 'lora': lora}))
            (checkpoint / 'checkpoint.complete.json').write_text(json.dumps(
                {'checkpoint_id': 'test', 'files': [{'path': item.name, 'size': item.stat().st_size}
                                                   for item in checkpoint.iterdir()]}))
            output = root / 'merged.safetensors'
            from distillation.eval_bundle import create_bundle
            bundle = root / 'evaluation'
            create_bundle(checkpoint, bundle, 'test')
            provenance = export_checkpoint(bundle, output)
            merged = Net()
            merged.load_state_dict(load_file(str(output)))
            probe = torch.tensor([[1., 2.]], dtype=torch.bfloat16)
            self.assertTrue(torch.equal(net.q(probe), merged.q(probe)))
            self.assertEqual(provenance['checkpoint_id'], 'test')
            self.assertEqual(provenance['lora'], lora)
            self.assertEqual(provenance['evaluation_bundle']['source']['checkpoint_id'], 'test')
            self.assertTrue(all('lora_' not in key for key in load_file(str(output))))
            restored = Net()
            load_snapshot(restored, state, lora)
            self.assertTrue(torch.equal(net.q(probe), restored.q(probe)))

    def test_render_configs_pair_everything_except_student_and_output(self):
        config = dict(band_steps=[2] * 4, history_t=.2, t_start=.4, pose_latents='pose.safetensors',
                      mimi_codes='codes.safetensors', model={'checkpoint_path': '/teacher/model.safetensors', 'audio': {}})
        manifest = {'seed': 91, 'windows': [{'clip_index': 7}, {'clip_index': 2}, {'clip_index': 7}]}
        a = sampler_config(config, '/eval.csv', '/data', manifest, '/dense.safetensors', '/out/dense', 64, 2)
        b = sampler_config(config, '/eval.csv', '/data', manifest, '/speedy.safetensors', '/out/speedy', 64, 2)
        self.assertEqual(a['rows'], [7, 2])
        self.assertEqual(a['resolving']['checkpoint_path'], '/teacher/model.safetensors')
        a['stabilising']['checkpoint_path'] = b['stabilising']['checkpoint_path']
        a['output_dir'] = b['output_dir']
        self.assertEqual(a, b)

    def test_prepare_renderer_does_not_export_or_run_inference(self):
        from distillation.render_pairs import main

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'config.toml'
            config.write_text("band_steps=[2,2,2,2]\nresolve_steps=2\nhistory_t=0.2\nt_start=0.4\n"
                              "pose_latents='pose.safetensors'\nmimi_codes='codes.safetensors'\n"
                              "[model]\ncheckpoint_path='/teacher.safetensors'\n[model.audio]\nnum_codebooks=16\n")
            csv = root / 'eval.csv'
            csv.write_text('paths,uuid\na,a\n')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({'seed': 17, 'windows': [{'clip_index': 0, 'path': 'a', 'start': 4}],
                'signature': {'csv_sha256': sha256(csv), 'pose_latents': 'pose.safetensors', 'mimi_codes': 'codes.safetensors',
                              'audio': {'num_codebooks': 16}, 'schedule': {'band_steps': [2] * 4, 'resolve_steps': 2, 'history_t': .2, 't_start': .4}}}))
            (root / 'a').mkdir()
            for filename in ('video_latents.safetensors', 'pose.safetensors'):
                save_file({'latents': torch.zeros(64, 48, 1, 1)}, str(root / 'a' / filename))
            save_file({'codes': torch.zeros(32, 750, dtype=torch.int16)}, str(root / 'a' / 'codes.safetensors'))
            for arm in ('dense', 'speedy'):
                checkpoint = root / arm
                checkpoint.mkdir()
                (checkpoint / 'student.safetensors').write_bytes(b'placeholder')
                (checkpoint / 'critic.safetensors').write_bytes(b'critic')
                (checkpoint / 'metadata.json').write_text(json.dumps({'checkpoint_id': arm,
                    'counters': {'warmup_step': 150, 'outer_step': 200},
                    'compatibility': {'initial_checkpoint_sha256': 'teacher', 'audio': {}, 'rollout': {}, 'latent_hw': [1, 1]}}))
                (checkpoint / 'checkpoint.complete.json').write_text(json.dumps({'checkpoint_id': arm,
                    'files': [{'path': item.name, 'size': item.stat().st_size} for item in checkpoint.iterdir()]}))
            args = ['render_pairs', '--config', str(config), '--csv', str(csv), '--root-dir', str(root),
                    '--manifest', str(manifest), '--dense', str(root / 'dense'), '--speedy', str(root / 'speedy'),
                    '--output', str(root / 'render'), '--max-latent-frames', '64', '--resolver-steps', '2']
            with patch('sys.argv', args), patch('distillation.render_pairs.subprocess.run') as run, \
                    patch('distillation.render_pairs.export_checkpoint') as export, patch('builtins.print'):
                main()
            run.assert_not_called()
            export.assert_not_called()
            self.assertTrue((root / 'render' / 'dense.yaml').is_file())
            self.assertTrue((root / 'render' / 'speedy.yaml').is_file())
            self.assertEqual(json.loads((root / 'render' / 'render-plan.json').read_text())['step'], 350)


if __name__ == '__main__':
    unittest.main()
