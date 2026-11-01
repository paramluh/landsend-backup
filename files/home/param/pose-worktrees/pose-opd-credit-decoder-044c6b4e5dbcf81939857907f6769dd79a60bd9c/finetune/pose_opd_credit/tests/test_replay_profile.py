"""Replay timers do not add chunk synchronizations or alter gradients."""
import contextlib
import copy
import unittest
from unittest import mock

import torch

from finetune.pose_opd_credit import runtime
from finetune.pose_opd_credit.config import TrainingConfig
from finetune.pose_opd_credit.profiling import ReplayStages
from finetune.pose_opd_credit.tests.test_runtime import ToyPPM, fixture


class ReplayProfileTests(unittest.TestCase):
    def test_cuda_events_have_only_one_collection_sync(self):
        events = []
        class Event:
            def __init__(self, **kwargs):
                self.records = 0
                events.append(self)
            def record(self):
                self.records += 1
            def elapsed_time(self, other):
                return 2.5
        with mock.patch.object(torch.cuda, 'device', side_effect=lambda _: contextlib.nullcontext()), \
             mock.patch.object(torch.cuda, 'Event', Event), \
             mock.patch.object(torch.cuda, 'synchronize') as synchronize:
            profile = ReplayStages('cuda:0')
            for _ in range(3):
                with profile.span('depth'):
                    pass
            synchronize.assert_not_called()
            result = profile.collect()
            synchronize.assert_called_once_with(torch.device('cuda:0'))
            self.assertIs(profile.collect(), result)
            synchronize.assert_called_once()
            self.assertEqual(result['stages']['depth']['calls'], 3)
            self.assertEqual(result['stages']['depth']['cuda_current_stream_ms'], 7.5)
            self.assertTrue(all(event.records == 1 for event in events))

    def test_real_replay_stages_preserve_loss_and_all_student_gradients(self):
        torch.set_num_threads(1)
        torch.manual_seed(912)
        models = [ToyPPM(), ToyPPM().requires_grad_(False), ToyPPM().requires_grad_(False)]
        reference = copy.deepcopy(models)
        bundle, latents = fixture(frames=8)
        config = TrainingConfig(decoder_chunk_frames=3, decoder_gradient_sync='per_replay')
        profile = ReplayStages('cpu')
        actual, _, _ = runtime.replay_update(*models, bundle, latents, config, None, torch.device('cpu'), stage_profiler=profile)
        expected, _, _ = runtime.replay_update(*reference, bundle, latents, config, None, torch.device('cpu'))
        self.assertEqual(actual, expected)
        for observed, wanted in zip(models[0].parameters(), reference[0].parameters()):
            torch.testing.assert_close(observed.grad, wanted.grad, rtol=0, atol=0)
        stages = profile.collect()['stages']
        self.assertEqual(set(stages), {'prepare', 'audio_teacher_temporal', 'pose_teacher_conditional_temporal',
            'student_detached_temporal', 'detached_depth_audio', 'detached_depth_pose', 'credit_assignment',
            'student_gradient_temporal', 'audio_depth_forward_backward', 'pose_depth_forward_backward',
            'final_temporal_backward'})
        self.assertTrue(all(stage['calls'] == 1 and stage['host_submission_seconds'] >= 0 for stage in stages.values()))

    def test_last_chunk_marker_and_config_modes(self):
        self.assertEqual(list(runtime.mark_last_chunk([])), [])
        self.assertEqual(list(runtime.mark_last_chunk([1, 2, 3])), [(1, False), (2, False), (3, True)])
        self.assertEqual(TrainingConfig().decoder_gradient_sync, 'per_chunk')
        with self.assertRaises(ValueError):
            TrainingConfig(decoder_gradient_sync='unknown').validate()


if __name__ == '__main__':
    unittest.main()
