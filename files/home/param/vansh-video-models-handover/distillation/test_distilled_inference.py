from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from inference.distilled import FIXED, MODES, validate_config


class DistilledConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        sample = self.root / 'sample'
        sample.mkdir()
        for filename in ('video_latents.safetensors', FIXED['pose_file'], FIXED['mimi_file']):
            (sample / filename).touch()
        mimi = self.root / 'mimi'
        mimi.mkdir()
        self.config = {
            **FIXED, 'sample_dir': str(sample), 'output_dir': str(self.root / 'output'), 'seed': 2026,
            'max_latent_frames': 1875, 'minimum_latent_frames': 1875,
            **deepcopy(MODES),
        }
        self.config['audio']['model_name'] = str(mimi)
        for mode in ('stabilising', 'resolving'):
            checkpoint = self.root / f'{mode}.safetensors'
            checkpoint.touch()
            self.config[mode]['checkpoint_path'] = str(checkpoint)

    def test_exact_fast_student_config_is_accepted_without_gpu_imports(self):
        original = deepcopy(self.config)
        validate_config(self.config)
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
        for key in ('refs', 'hidden', 'lora', 'audio_guidance_scale', 'typo'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'unexpected keys'):
                validate_config({**self.config, key: {}})
        for mode in MODES:
            with self.subTest(mode=mode):
                changed = deepcopy(self.config)
                changed[mode]['refs'] = {}
                with self.assertRaisesRegex(ValueError, 'unexpected keys'):
                    validate_config(changed)

    def test_relative_paths_and_remote_mimi_identifier_fail(self):
        for key in ('sample_dir', 'output_dir'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'absolute path'):
                validate_config({**self.config, key: 'relative'})
        for mode in MODES:
            with self.subTest(mode=mode):
                changed = deepcopy(self.config)
                key = 'model_name' if mode == 'audio' else 'checkpoint_path'
                changed[mode][key] = 'kyutai/mimi' if mode == 'audio' else 'student.safetensors'
                with self.assertRaisesRegex(ValueError, 'absolute path'):
                    validate_config(changed)

    def test_same_checkpoint_and_missing_assets_fail(self):
        changed = deepcopy(self.config)
        changed['resolving']['checkpoint_path'] = changed['stabilising']['checkpoint_path']
        with self.assertRaisesRegex(ValueError, 'separate student checkpoints'):
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


if __name__ == '__main__':
    unittest.main()
