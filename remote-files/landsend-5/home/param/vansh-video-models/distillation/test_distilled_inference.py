from copy import deepcopy
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch
import zlib

from inference.distilled import FIXED, MODES, build_conditioned_modes, reference_seed, validate_config


def write_tensor_headers(path, shapes, dtype='F32'):
    offset, header = 0, {}
    for key, shape in shapes.items():
        size = math.prod(shape) * (8 if dtype == 'I64' else 4)
        header[key] = {'shape': shape, 'dtype': dtype, 'data_offsets': [offset, offset + size]}
        offset += size
    data = json.dumps(header).encode()
    with path.open('wb') as handle:
        handle.write(len(data).to_bytes(8, 'little'))
        handle.write(data)
        handle.truncate(8 + len(data) + offset)


class DistilledConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        sample = self.root / 'sample'
        sample.mkdir()
        for filename in ('video_latents.safetensors', FIXED['pose_file']):
            write_tensor_headers(sample / filename, {'latents': [1875, 48, 2, 2]})
        write_tensor_headers(sample / FIXED['mimi_file'], {'codes': [16, 3750]}, dtype='I64')
        mimi = self.root / 'mimi'
        mimi.mkdir()
        (mimi / 'config.json').write_text('{}')
        (mimi / 'model.safetensors').touch()
        self.config = {
            **FIXED, 'sample_dir': str(sample), 'output_dir': str(self.root / 'output'), 'seed': 2026,
            'max_latent_frames': 1875, 'minimum_latent_frames': 1875,
            **deepcopy(MODES),
            'refs': {'bank': str(self.root / 'bank'), 'actor_id': 'test-actor', 'clip_key': 'actor/clip'},
            'vae_path': str(self.root / 'Wan2.2_VAE.pth'),
        }
        Path(self.config['vae_path']).touch()
        self.config['audio']['model_name'] = str(mimi)
        checkpoint = self.root / 'student.safetensors'
        write_tensor_headers(checkpoint, {
            'ref_patch_embedding.weight': [3840, 48, 1, 2, 2], 'ref_patch_embedding.bias': [3840],
        })
        for mode in ('stabilising', 'resolving'):
            self.config[mode]['checkpoint_path'] = str(checkpoint)
        for slot in range(1, 10):
            path = self.root / 'bank' / 'test-actor' / f'property_{slot}'
            path.mkdir(parents=True)
            write_tensor_headers(path / 'latents.safetensors', {'video': [3, 48, 2, 2]})

    def test_exact_fast_student_config_is_accepted_without_gpu_imports(self):
        original = deepcopy(self.config)
        result = validate_config(self.config)
        self.assertTrue(result['shared_weights'])
        self.assertEqual(result['reference_keep'], [True] * 9)
        self.assertEqual(self.config, original)
        self.assertFalse(Path(self.config['output_dir']).exists())

    def test_unlimited_length_and_short_explicit_clip_are_accepted(self):
        self.config.update(max_latent_frames=None, minimum_latent_frames=2)
        validate_config(self.config)
        self.config.update(max_latent_frames=17, minimum_latent_frames=17)
        validate_config(self.config)

    def test_every_required_top_level_value_must_be_explicit(self):
        for key in self.config:
            with self.subTest(key=key):
                changed = deepcopy(self.config)
                del changed[key]
                with self.assertRaisesRegex(ValueError, 'missing keys'):
                    validate_config(changed)

    def test_fast_flags_pose_v3_local_mimi_and_mode_are_enforced(self):
        replacements = {
            'compile': False, 'cuda_graph': False, 'wavefront': False, 'renoising': True,
            'pose_file': 'pose_latents.safetensors', 'mimi_file': 'mimi.safetensors',
            'type': 'universal_wide', 'fps': 30, 'batch_size': 2,
            'runtime_audio_guidance_scale': 2.5,
        }
        for key, value in replacements.items():
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                validate_config({**self.config, key: value})

    def test_exact_student_schedule_and_audio_architecture_are_enforced(self):
        for mode, expected in MODES.items():
            for key, value in expected.items():
                with self.subTest(mode=mode, key=key):
                    changed = deepcopy(self.config)
                    changed[mode][key] = value + 1
                    with self.assertRaisesRegex(ValueError, f'{mode}.{key}'):
                        validate_config(changed)

    def test_unsupported_conditioning_or_unrecognized_options_fail(self):
        for key in ('hidden', 'lora', 'audio_guidance_scale', 'typo'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'unexpected keys'):
                validate_config({**self.config, key: {}})
        for mode in MODES:
            with self.subTest(mode=mode):
                changed = deepcopy(self.config)
                changed[mode]['refs'] = {}
                with self.assertRaisesRegex(ValueError, 'unexpected keys'):
                    validate_config(changed)

    def test_relative_paths_and_remote_mimi_identifier_fail(self):
        for key in ('sample_dir', 'output_dir', 'vae_path'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'absolute path'):
                validate_config({**self.config, key: 'relative'})
        for mode in MODES:
            with self.subTest(mode=mode):
                changed = deepcopy(self.config)
                key = 'model_name' if mode == 'audio' else 'checkpoint_path'
                changed[mode][key] = 'kyutai/mimi' if mode == 'audio' else 'student.safetensors'
                with self.assertRaisesRegex(ValueError, 'absolute path'):
                    validate_config(changed)

    def test_different_checkpoints_and_missing_assets_fail(self):
        changed = deepcopy(self.config)
        other = self.root / 'other.safetensors'
        other.touch()
        changed['resolving']['checkpoint_path'] = str(other)
        with self.assertRaisesRegex(ValueError, 'share the same checkpoint_path'):
            validate_config(changed)
        changed['resolving']['checkpoint_path'] = str(self.root / 'absent.safetensors')
        with self.assertRaisesRegex(ValueError, 'does not exist'):
            validate_config(changed)
        (Path(self.config['sample_dir']) / FIXED['mimi_file']).unlink()
        with self.assertRaisesRegex(ValueError, 'Missing input'):
            validate_config(self.config)

    def test_existing_output_and_dangling_symlink_are_never_overwritten(self):
        output = Path(self.config['output_dir'])
        output.mkdir()
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
            validate_config(self.config)
        output.rmdir()
        output.symlink_to(self.root / 'absent')
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
            validate_config(self.config)

    def test_invalid_lengths_and_boolean_numeric_fields_fail(self):
        for changes in (
            {'minimum_latent_frames': 1}, {'minimum_latent_frames': True},
            {'max_latent_frames': 17}, {'max_latent_frames': 0}, {'max_latent_frames': True},
            {'seed': True}, {'seed': 2.5}, {'compile': 1}, {'batch_size': True}, {'batch_size': 1.0},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config({**self.config, **changes})

    def test_both_native_raw_and_ema_files_are_accepted_without_optimizer_metadata(self):
        for name in ('student.safetensors', 'model_ema.safetensors'):
            target = self.root / name
            if not target.exists():
                target.write_bytes(Path(self.config['stabilising']['checkpoint_path']).read_bytes())
            for mode in ('stabilising', 'resolving'):
                self.config[mode]['checkpoint_path'] = str(target)
            validate_config(self.config)

    def test_reference_selection_seed_matches_aloe_eval(self):
        key = self.config['refs']['clip_key']
        self.assertEqual(reference_seed(key), [1234, zlib.crc32(key.encode()) & 0x7FFFFFFF])
        changed = deepcopy(self.config)
        changed['seed'] += 1
        self.assertEqual(validate_config(changed)['reference_seed'], validate_config(self.config)['reference_seed'])

    def test_missing_properties_remain_masked_but_empty_actor_fails(self):
        actor = self.root / 'bank' / 'test-actor'
        (actor / 'property_2' / 'latents.safetensors').unlink()
        self.assertEqual(validate_config(self.config)['reference_keep'], [True, False] + [True] * 7)
        for path in actor.glob('property_*/latents.safetensors'):
            path.unlink()
        with self.assertRaisesRegex(ValueError, 'no reference latents'):
            validate_config(self.config)

    def test_missing_actor_or_invalid_reference_identity_fails(self):
        for key, value, message in (
            ('bank', 'relative', 'absolute path'), ('actor_id', '../actor', 'folder name'),
            ('actor_id', 'absent', 'Missing actor'), ('clip_key', '', 'CSV paths'),
            ('clip_key', ' padded ', 'CSV paths'),
        ):
            changed = deepcopy(self.config)
            changed['refs'][key] = value
            with self.subTest(key=key, value=value), self.assertRaisesRegex(ValueError, message):
                validate_config(changed)

    def test_reference_module_and_reference_geometry_are_checked(self):
        checkpoint = Path(self.config['stabilising']['checkpoint_path'])
        original = checkpoint.read_bytes()
        write_tensor_headers(checkpoint, {'patch_embedding.weight': [3840, 48, 1, 2, 2]})
        with self.assertRaisesRegex(ValueError, 'nine-reference student'):
            validate_config(self.config)
        checkpoint.write_bytes(original)
        ref = self.root / 'bank' / 'test-actor' / 'property_1' / 'latents.safetensors'
        write_tensor_headers(ref, {'video': [3, 48, 4, 2]})
        with self.assertRaisesRegex(ValueError, 'geometry matching the clip'):
            validate_config(self.config)

    def test_missing_or_wrong_vae_fails(self):
        for path in (self.root / 'missing' / 'Wan2.2_VAE.pth', self.root / 'Wan2.1_VAE.pth'):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, 'Wan2.2_VAE.pth'):
                validate_config({**self.config, 'vae_path': str(path)})

    def test_missing_mimi_weights_and_wrong_code_dtype_fail(self):
        weights = Path(self.config['audio']['model_name']) / 'model.safetensors'
        weights.unlink()
        with self.assertRaisesRegex(ValueError, 'Missing local Mimi asset'):
            validate_config(self.config)
        weights.touch()
        codes = Path(self.config['sample_dir']) / FIXED['mimi_file']
        write_tensor_headers(codes, {'codes': [16, 3750]})
        with self.assertRaisesRegex(ValueError, 'Unexpected dtype for codes'):
            validate_config(self.config)

    def test_runtime_installs_the_same_reference_selection_in_both_modes(self):
        sampler, graph, source, numpy, torch = (MagicMock() for _ in range(5))
        stabilizer, resolving, refs, keep, latents = (MagicMock() for _ in range(5))
        latents.shape = (48, 1875, 2, 2)
        sampler.build_modes.return_value = (stabilizer, resolving)
        source.load_actor_refs.return_value = (refs, keep)
        common = Mock(AUTOCAST_DTYPE='test-autocast-dtype')
        modules = {
            'numpy': numpy, 'torch': torch, 'bimodal': Mock(sampler=sampler),
            'bimodal.graph_resolver': graph, 'dataloading.ds_aloe': source, 'utils.common': common,
        }
        with patch.dict(sys.modules, modules):
            models, cached, resolver = build_conditioned_modes(self.config, latents)
        self.assertIs(models, stabilizer)
        sampler.build_modes.assert_called_once_with(
            self.config, Path(__file__).resolve().parents[1] / 'bimodal', 'cuda',
        )
        numpy.random.default_rng.assert_called_once_with(reference_seed(self.config['refs']['clip_key']))
        source.load_actor_refs.assert_called_once_with(
            self.config['refs']['bank'], self.config['refs']['actor_id'],
            latents.__getitem__.return_value, numpy.random.default_rng.return_value,
        )
        sampler.WavefrontNet.assert_called_once_with(stabilizer, (2, 2), 2, 'cuda', True, 1, .2, guidance_scale=1.0)
        self.assertEqual(cached.refs, (refs.transpose.return_value.__getitem__.return_value.cuda.return_value,
                                       keep.__getitem__.return_value.cuda.return_value))
        refs.transpose.assert_called_once_with(0, 1)
        graph.GraphResolver.assert_called_once_with(resolving, num_steps=1)
        resolver.set_refs.assert_called_once_with(refs.__getitem__.return_value.cuda.return_value,
                                                 keep.__getitem__.return_value.cuda.return_value)
        torch.autocast.assert_called_once_with('cuda', dtype='test-autocast-dtype', cache_enabled=False)

    def test_check_cli_does_not_import_torch_or_create_outputs(self):
        config = self.root / 'config.json'
        config.write_text(json.dumps(self.config))
        script = (
            'import runpy,sys; '
            f'sys.argv=["inference.distilled",{str(config)!r},"--check"]; '
            'runpy.run_module("inference.distilled",run_name="__main__"); '
            'assert "torch" not in sys.modules'
        )
        result = subprocess.run([sys.executable, '-c', script], check=True, text=True, capture_output=True)
        self.assertEqual(json.loads(result.stdout)['stabilizer_band_steps'], [2, 2, 2, 2])
        self.assertFalse(Path(self.config['output_dir']).exists())


if __name__ == '__main__':
    unittest.main()
