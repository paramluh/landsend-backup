"""Explicit exported-gain policy without loading production checkpoints."""
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from finetune.duplex_distill.checkpoints import _restore_exact_gain, load_checkpoint
from finetune.duplex_distill.production.lib.vadapter_s4q import VAdapter


class GainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgain = nn.Module()
        self.vgain.theta = nn.Parameter(torch.zeros(1))


class CheckpointGainTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.saved = torch.tensor([-5.571630477905273]).bfloat16()
        self.write_export()

    def write_export(self):
        torch.save({'vgain.theta': self.saved}, self.path / 'rank0.pt')
        torch.save({}, self.path / 'rep.pt')
        (self.path / 'COMPLETE').write_text('1')
        (self.path / 'meta.json').write_text(json.dumps({
            'format': 'ppm-local-bf16-shards-v1', 'world': 1, 'step': 1}))

    def test_default_requires_original_fp32_metadata(self):
        with self.assertRaisesRegex(ValueError, 'lacks exact fp32 gain metadata'):
            load_checkpoint(GainModel(), self.path, 'cpu')

    def test_exported_policy_preserves_loaded_tensor_and_reports_precision(self):
        model = GainModel()
        receipt = load_checkpoint(model, self.path, 'cpu', gain_policy='exported')
        self.assertEqual(model.vgain.theta.dtype, torch.float32)
        torch.testing.assert_close(model.vgain.theta, self.saved.float(), atol=0, rtol=0)
        gain = receipt['exact_gain']
        self.assertEqual(gain['policy'], 'exported')
        self.assertEqual(gain['vgain_theta_as_stored'], float(self.saved[0]))
        self.assertIs(gain['original_fp32_recovered'], False)
        self.assertNotIn('vgain_theta_fp32', gain)

    def test_exact_sidecar_wins_even_with_exported_policy(self):
        original = -5.571630477905273
        (self.path / 'vgain_theta_fp32.json').write_text(json.dumps({'vgain_theta_fp32': original}))
        for policy in ('require_fp32', 'exported'):
            with self.subTest(policy=policy):
                model = GainModel()
                receipt = load_checkpoint(model, self.path, 'cpu', gain_policy=policy)
                self.assertEqual(float(model.vgain.theta.detach()[0]), original)
                self.assertEqual(receipt['exact_gain']['vgain_theta_fp32'], original)
                self.assertEqual(receipt['exact_gain']['source'], str(self.path / 'vgain_theta_fp32.json'))

    def test_exact_export_metadata_wins_even_with_exported_policy(self):
        original = -5.571630477905273
        meta_path = self.path / 'meta.json'
        meta = json.loads(meta_path.read_text())
        meta['vgain_theta_fp32'] = original
        meta_path.write_text(json.dumps(meta))
        model = GainModel()
        receipt = load_checkpoint(model, self.path, 'cpu', gain_policy='exported')
        self.assertEqual(float(model.vgain.theta.detach()[0]), original)
        self.assertEqual(receipt['exact_gain']['vgain_theta_fp32'], original)

    def test_exported_policy_still_rejects_nonfinite_tensors(self):
        self.saved.fill_(float('nan'))
        self.write_export()
        with self.assertRaisesRegex(ValueError, 'nonfinite tensors'):
            load_checkpoint(GainModel(), self.path, 'cpu', gain_policy='exported')

    def test_exported_policy_does_not_hide_invalid_exact_metadata(self):
        (self.path / 'vgain_theta_fp32.json').write_text(json.dumps({'vgain_theta_fp32': -2.0}))
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            load_checkpoint(GainModel(), self.path, 'cpu', gain_policy='exported')

    def test_invalid_policy_rejected_before_loading_or_skipping_gain(self):
        with self.assertRaisesRegex(ValueError, 'Unknown checkpoint gain policy'):
            load_checkpoint(GainModel(), self.path / 'does-not-exist', 'cpu', gain_policy='anything')
        with self.assertRaisesRegex(ValueError, 'Unknown checkpoint gain policy'):
            _restore_exact_gain({}, self.path, {}, gain_policy='anything')

    def adapter_export(self, norm='none', *, include_rms=False):
        model = GainModel()
        model.vadapter = VAdapter(encoder_hidden=2, output_dim=4, hidden_dim=3, stack=2,
                                 norm=norm, target_rms=.75)
        saved = {name: value.bfloat16() for name, value in model.state_dict().items()
                 if name != 'vadapter.target_rms' or include_rms}
        saved['vgain.theta'] = self.saved
        replicas = {name: value for name, value in saved.items() if not value.ndim}
        torch.save({name: value for name, value in saved.items() if value.ndim}, self.path / 'rank0.pt')
        torch.save(replicas, self.path / 'rep.pt')
        return model, saved

    def test_rank_export_reconstructs_only_unused_native_adapter_buffer(self):
        model, saved = self.adapter_export()
        receipt = load_checkpoint(model, self.path, 'cpu', gain_policy='exported')
        self.assertTrue(torch.isnan(model.vadapter.target_rms))
        self.assertEqual(receipt['tensors_loaded'], len(saved))
        self.assertEqual([item['name'] for item in receipt['reconstructed_unused_buffers']],
                         ['vadapter.target_rms'])
        for name, value in model.named_parameters():
            torch.testing.assert_close(value, saved[name].float(), atol=0, rtol=0)
        latents = torch.randn(1, 6, 2)
        output = model.vadapter(latents)
        self.assertTrue(torch.isfinite(output).all())
        model.vadapter.target_rms.fill_(12345.)
        torch.testing.assert_close(output, model.vadapter(latents), atol=0, rtol=0)

    def test_rank_export_rms_and_gain_modes_still_require_buffer(self):
        for norm in ('rms', 'gain'):
            with self.subTest(norm=norm):
                model, _ = self.adapter_export(norm)
                with self.assertRaisesRegex(ValueError, 'keys differ.*vadapter.target_rms'):
                    load_checkpoint(model, self.path, 'cpu', gain_policy='exported')

    def test_existing_unused_buffer_loads_verbatim_without_reconstruction(self):
        model, _ = self.adapter_export(include_rms=True)
        model.vadapter.target_rms.zero_()
        receipt = load_checkpoint(model, self.path, 'cpu', gain_policy='exported')
        self.assertEqual(float(model.vadapter.target_rms), .75)
        self.assertEqual(receipt['reconstructed_unused_buffers'], [])

    def test_missing_adapter_learned_weight_is_never_reconstructed(self):
        model, saved = self.adapter_export()
        del saved['vadapter.mlp.linear1.weight']
        torch.save(saved, self.path / 'rank0.pt')
        with self.assertRaisesRegex(ValueError, 'keys differ.*vadapter.mlp.linear1.weight'):
            load_checkpoint(model, self.path, 'cpu', gain_policy='exported')

    def test_target_rms_parameter_is_never_treated_as_unused_buffer(self):
        model, _ = self.adapter_export()
        del model.vadapter.target_rms
        model.vadapter.target_rms = nn.Parameter(torch.tensor(.75))
        with self.assertRaisesRegex(ValueError, 'keys differ.*vadapter.target_rms'):
            load_checkpoint(model, self.path, 'cpu', gain_policy='exported')


if __name__ == '__main__':
    unittest.main()
