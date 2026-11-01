"""Frozen frontend transfers preserve values/dtypes and release shared GPUs."""
import contextlib
from types import SimpleNamespace
import unittest
from unittest import mock

import torch
from torch import nn

from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.train import move_audio_front


class AudioOffloadTests(unittest.TestCase):
    def test_offload_preserves_encoder_dtype_buffers_and_wrapper_device_contract(self):
        encoder = nn.Linear(3, 2).bfloat16().requires_grad_(False)
        encoder.register_buffer('inv_freq', torch.tensor([.123456789], dtype=torch.float32), persistent=False)
        front = SimpleNamespace(dev=torch.device('cuda:2'), fe=SimpleNamespace(
            encoder=encoder, device=torch.device('cuda:2'), dtype=torch.bfloat16))
        before = {name: value.clone() for name, value in [*encoder.named_parameters(), *encoder.named_buffers()]}
        with mock.patch('torch.cuda.device', return_value=contextlib.nullcontext()) as device_context, \
             mock.patch('torch.cuda.synchronize') as synchronize, \
             mock.patch('torch.cuda.empty_cache') as release:
            elapsed = move_audio_front(front, 'cpu')
        self.assertGreaterEqual(elapsed, 0.)
        device_context.assert_called_once_with(torch.device('cuda:2'))
        synchronize.assert_called_once_with()
        release.assert_called_once_with()
        self.assertEqual(front.dev, torch.device('cuda:2'))
        self.assertEqual(front.fe.device, torch.device('cuda:2'))
        self.assertEqual(front.fe.dtype, torch.bfloat16)
        for name, value in [*encoder.named_parameters(), *encoder.named_buffers()]:
            self.assertEqual(value.dtype, before[name].dtype)
            self.assertTrue(torch.equal(value, before[name]))
            self.assertFalse(value.requires_grad)

    def test_restore_uses_only_device_and_waits_for_complete_transfer(self):
        encoder = mock.Mock()
        front = SimpleNamespace(dev=torch.device('cuda:1'), fe=SimpleNamespace(encoder=encoder))
        with mock.patch('torch.cuda.synchronize') as synchronize, mock.patch('torch.cuda.empty_cache') as release:
            move_audio_front(front, torch.device('cuda:1'))
        encoder.to.assert_called_once_with(device=torch.device('cuda:1'))
        synchronize.assert_called_once_with(torch.device('cuda:1'))
        release.assert_not_called()

    def test_config_offload_is_explicit_boolean(self):
        self.assertFalse(TrainingConfig().offload_audio_front)
        self.assertTrue(TrainingConfig(offload_audio_front=True).validate().offload_audio_front)
        with self.assertRaisesRegex(ValueError, 'offload_audio_front must be a boolean'):
            TrainingConfig(offload_audio_front=1).validate()


if __name__ == '__main__':
    unittest.main()
