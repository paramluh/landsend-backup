"""Teacher-partner source boundaries, random prefills and original audio gains."""
from dataclasses import replace
import importlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.data import ProductionRows, output_gain_metadata, row_prefill_frames, prefix_seed


def partner_config(**kwargs):
    return TrainingConfig(data_variant='raw', rollout_mode='teacher_partner',
                          conversation_prefix_frames=1500, exclude_short_rows=True, **kwargs)


class PartnerConfigurationTests(unittest.TestCase):
    def test_random_prefills_are_bounded_reproducible_and_update_specific(self):
        cfg = partner_config()
        sample = lambda update: [row_prefill_frames(cfg, (0, index, index % 2), 3,
                                                    update=update, slot=index) for index in range(100)]
        first = sample(0)
        self.assertEqual(first, sample(0))
        self.assertNotEqual(first, sample(1))
        self.assertTrue(set(first) <= set(range(38, 63)))
        self.assertGreater(len(set(first)), 20)
        self.assertTrue(all(3 <= value / 12.5 <= 5 for value in first))
        self.assertEqual(row_prefill_frames(TrainingConfig(), (0, 0, 0)), 25)
        with self.assertRaisesRegex(ValueError, 'nonnegative'):
            row_prefill_frames(cfg, (0, 0, 0), update=-1)

    def test_config_preserves_defaults_and_rejects_incompatible_partner_inputs(self):
        cfg = partner_config().validate()
        self.assertEqual(cfg.resolved_partner_checkpoint, cfg.audio_teacher)
        self.assertEqual(replace(cfg, partner_checkpoint='/frozen/partner').resolved_partner_checkpoint,
                         '/frozen/partner')
        self.assertEqual(TrainingConfig.from_dict(cfg.to_dict()), cfg)
        for changes in ({'data_variant': 'samsep'}, {'conditioning_mode': 'voice'},
                        {'partner_prefill_min_frames': 63}, {'partner_prefill_max_frames': 0},
                        {'partner_checkpoint': None}, {'conversation_prefix_frames': 62},
                        {'fixed_sequence_frames': 62}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(cfg, **changes).validate()


class OutputGainTests(unittest.TestCase):
    def test_original_gain_not_sam_index_gain_and_mimidec_normalization(self):
        entry = {'flac': 'speaker.flac', 'a_spk': 'SPEAKER_01', 'kind': 'gcs', 'gain_db': 17.}
        gains = {'speaker.flac': {'a_spk': 'SPEAKER_01', 'kind': 'gcs', 'gain_mimi_db': -6.}}
        value = output_gain_metadata(entry, gains, sha256='original-digest')
        self.assertEqual(value['gain_mimi_db'], -6.)
        self.assertAlmostEqual(value['inverse_gain'], 10 ** .3)
        self.assertEqual(value['raw_gain_sha256'], 'original-digest')
        entry['kind'] = 'mimidec'
        gains['speaker.flac'].update(kind='mimidec', gain_mimi_db=0., recorded_lufs_gain_db=-8.421,
                                    codes_checked=True)
        self.assertEqual(output_gain_metadata(entry, gains, sha256='x')['gain_mimi_db'], -8.421)
        gains['speaker.flac']['codes_checked'] = False
        with self.assertRaisesRegex(ValueError, 'Unverified'):
            output_gain_metadata(entry, gains, sha256='x')

    def test_missing_or_mismatched_gain_is_never_guessed(self):
        entry = {'flac': 'a', 'a_spk': 'SPEAKER_00', 'kind': 'gcs'}
        with self.assertRaises(KeyError):
            output_gain_metadata(entry, {}, sha256='x')
        for row in ({'a_spk': 'SPEAKER_01', 'kind': 'gcs', 'gain_mimi_db': 0},
                    {'a_spk': 'SPEAKER_00', 'kind': 'gcs', 'gain_mimi_db': float('nan')}):
            with self.assertRaises(ValueError):
                output_gain_metadata(entry, {'a': row}, sha256='x')


class PartnerReplayIntegrationTests(unittest.TestCase):
    def test_four_gpu_teams_and_two_gpu_continuous_path(self):
        from finetune.pose_opd_credit.train import actor_group_width
        self.assertEqual(actor_group_width(partner_config(), 40, 8), 4)
        self.assertEqual(actor_group_width(TrainingConfig(), 40, 8), 2)
        self.assertEqual(actor_group_width(TrainingConfig(), 2, 2), 2)
        for world, local in ((2, 2), (8, 2), (6, 6), (40, 16)):
            with self.assertRaisesRegex(ValueError, 'divisible by 4'):
                actor_group_width(partner_config(), world, local)

    def test_replay_uses_recorded_features_and_samples_without_source_future(self):
        import torch
        from finetune.pose_opd_credit.train import partner_replay_payload
        source = {'conversation_frames': 4, 'partner': {'unused': 'partner seed'},
                  'wav': np.ones(1920, np.float32), 'aref': 'preserved provenance'}
        exact = torch.arange(4 * 4 * 1280).reshape(1, 16, 1280).to(torch.bfloat16)
        wave = np.arange(4 * 1920, dtype=np.float32)
        output = {'rollout': 'generated tokens', 'latents': exact, 'waveform': wave,
                  'delivery_delay_samples': 72}
        payload, rollouts, samples = partner_replay_payload([(source, None)], [output], [0])
        self.assertEqual(payload[0][1].data_ptr(), exact.data_ptr())
        torch.testing.assert_close(payload[0][1], exact, rtol=0, atol=0)
        self.assertIs(samples[0], wave)
        self.assertEqual(rollouts, ['generated tokens'])
        self.assertEqual(payload[0][0], {'conversation_frames': 4, 'aref': 'preserved provenance',
                                       'incoming_audio_delay_samples': 72})
        self.assertIn('partner', source)
        self.assertIn('wav', source)
        self.assertEqual(partner_replay_payload([(source, None)], [{**output, 'waveform': None}], [])[2], {})
        for bad in ({**output, 'latents': exact[:, :-1]}, {**output, 'waveform': wave[:-1]},
                    {**output, 'waveform': None}):
            with self.assertRaises(ValueError):
                partner_replay_payload([(source, None)], [bad], [0])
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            partner_replay_payload([(source, None)], [], [])


@unittest.skipUnless(Path('/home/param/datasets/aloe-duplex/pose/POSE_INDEX.json').is_file(),
                     'Production data not installed')
class PartnerProductionTests(unittest.TestCase):
    def test_both_future_lanes_are_scrubbed_and_no_partner_pose_is_read(self):
        rows = ProductionRows(partner_config())
        item = (0, 1837, 0)
        with rows.arena.enter() as module:
            original = module.pose_entry
            def only_student_pose(requested):
                self.assertEqual(tuple(requested), item, 'The audio partner must not require its pose export')
                return original(requested)
            with patch.object(module, 'pose_entry', side_effect=only_student_pose):
                seed = rows.load(item, load_audio=False, update=4, slot=1)
        partner = seed['partner']
        self.assertEqual(partner['item'], [0, 1837, 1])
        self.assertEqual(seed['prefill_frames'], partner['prefill_frames'])
        self.assertEqual(partner['kind'], 'speech')
        self.assertFalse(partner['sampled_masks']['pose'].any())
        self.assertTrue((partner['pdepth'] == -1).all())
        for value in (seed, partner):
            p, n = map(int, value['vin'])
            pref = value['prefill_frames']
            self.assertEqual(n, 1500)
            self.assertEqual(value['ids'].shape, (2, p + 1502))
            self.assertTrue((value['ids'][0] == value['pad_id']).all())
            self.assertTrue((value['depth'][0] == -1).all())
            self.assertTrue((value['ids'][1, p + pref:] == value['pad_id']).all())
            self.assertTrue((value['depth'][1, p + pref + 2:] == -1).all())
            self.assertFalse(value['conditioning']['reference_text'])
            self.assertFalse(value['conditioning']['reference_pose'])
            self.assertNotIn('wav', value)
            source = json.loads(value['aref'])
            self.assertEqual(source['source_usage'], 'prefill_only')
            self.assertNotEqual(source['flac'], value['output_audio_gain']['flac'])
        self.assertEqual(json.loads(seed['aref'])['flac'], partner['output_audio_gain']['flac'])
        self.assertEqual(json.loads(partner['aref'])['flac'], seed['output_audio_gain']['flac'])

    def test_pcm_loader_only_requests_both_prefills_and_keeps_raw_gains(self):
        rows = ProductionRows(partner_config())
        requests = []
        def read_prefix(entry, gain):
            requests.append((dict(entry), gain))
            return np.full(entry['n_frames'] * 1920, len(requests), np.float32)
        original_import = importlib.import_module
        def module(name, *args, **kwargs):
            return SimpleNamespace(row_audio=read_prefix) if name == 'rowcache_vox' else original_import(name, *args, **kwargs)
        with patch('finetune.pose_opd_credit.data.importlib.import_module', side_effect=module):
            seed = rows.load((0, 1837, 0), update=7, slot=1)
        self.assertEqual(len(requests), 2)
        self.assertEqual({value[0]['n_frames'] for value in requests}, {seed['prefill_frames']})
        self.assertTrue(all(value[1] == 0 for value in requests))
        self.assertEqual(seed['wav'].shape, (seed['prefill_frames'] * 1920,))
        self.assertEqual(seed['partner']['wav'].shape, seed['wav'].shape)
        self.assertTrue((seed['wav'] == 1).all())
        self.assertTrue((seed['partner']['wav'] == 2).all())
        gains = json.loads(Path(rows.config.data_root, 'original/provenance/RAWGAIN.json').read_text())['rows']
        for value in (seed, seed['partner']):
            gain = value['output_audio_gain']
            self.assertEqual(gain['gain_mimi_db'], gains[gain['flac']]['gain_mimi_db'])
        shortened = prefix_seed(seed, 125)
        self.assertEqual(shortened['conversation_frames'], 125)
        self.assertEqual(shortened['partner']['conversation_frames'], 125)
        self.assertEqual(shortened['wav'].shape, seed['wav'].shape)
        self.assertFalse(shortened['partner']['sampled_masks']['pose'].any())


if __name__ == '__main__':
    unittest.main()
