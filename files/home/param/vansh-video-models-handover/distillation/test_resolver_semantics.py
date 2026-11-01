import unittest

import torch

from distillation.resolver_dmd import GTConditionedStabilizer


class RecordingNet:
    def __init__(self, poison_first=0.):
        self.calls = []
        self.poison_first = poison_first

    def audio_tokens(self, audio):
        return audio

    def frame_audio(self, tokens, ofi, drop):
        return ofi.float()[..., None, None]

    def forward_pass(self, frames, plan, cache, audio, mask, present):
        self.calls.append({'frames': [(x.clone(), None if pose is None else pose.clone(), time, slot, valid)
                                      for x, pose, time, slot, valid in frames],
                           'plan': plan, 'audio': audio.clone(), 'mask': mask.clone()})
        batch, channels, height, width = frames[0][0].shape
        trajectory = torch.arange(1, batch+1, dtype=torch.float32)[:, None, None, None, None]
        slots = torch.arange(4, dtype=torch.float32)[None, None, :, None, None]
        velocity = (trajectory + 7*slots + .125*len(self.calls)).expand(batch, channels, 4, height, width).clone()
        velocity[0] += self.poison_first
        return velocity, None, None


def source_rows(batch=2, frames=11):
    frame = torch.arange(frames, dtype=torch.float32)[None, None, :, None, None]
    channel = torch.arange(2, dtype=torch.float32)[None, :, None, None, None]
    row = torch.arange(batch, dtype=torch.float32)[:, None, None, None, None]
    gt = 1000*row + 100*channel + 10*frame
    noise = -100*row + 20*channel - 3*frame + 7
    return {'latents': gt, 'pose': gt+5000, 'audio': torch.zeros(batch, 1),
            'ofi': torch.arange(frames)[None].expand(batch, -1) + 100*torch.arange(batch)[:, None],
            'drop': torch.zeros(batch, dtype=torch.bool)}, noise


class ResolverTemporalSemantics(unittest.TestCase):
    def test_independent_eight_step_trajectories_use_gt_context(self):
        rows, noise = source_rows()
        net = RecordingNet()
        output = GTConditionedStabilizer()(net, rows, noise=noise)
        self.assertEqual(len(net.calls), 8)
        expected = noise[:, :, 7:].transpose(1, 2).reshape(8, 2, 1, 1).clone()
        bands = [3, 3, 2, 2, 1, 1, 0, 0]
        target_times = [1., .9, .8, .7, .6, .5, .4, .3]
        contexts = {
            3: [('f', 7), ('f', 0), ('f', 1), ('f', 2), ('f', 3)],
            2: [('f', 7), ('r', 6, 3), ('f', 0), ('f', 1), ('f', 2)],
            1: [('f', 7), ('r', 5, 2), ('r', 6, 3), ('f', 0), ('f', 1)],
            0: [('f', 7), ('r', 4, 1), ('r', 5, 2), ('r', 6, 3), ('f', 0)],
        }
        for step, (call, band, time) in enumerate(zip(net.calls, bands, target_times)):
            frames = call['frames']
            self.assertEqual(len(frames), 8)
            self.assertEqual(frames[band][0].shape, (8, 2, 1, 1))
            torch.testing.assert_close(frames[band][0], expected)
            self.assertAlmostEqual(frames[band][2], time)
            self.assertEqual(call['plan'][0][1][band], contexts[band])
            self.assertEqual(call['audio'].shape, (8, 4, 1, 1))
            for row in range(2):
                for target in range(4):
                    flat = 4*row + target
                    target_frame = 7+target
                    for slot in range(4):
                        value, pose, slot_time, position, valid = frames[slot]
                        self.assertEqual(position, 4+slot)
                        self.assertEqual(valid, slot <= band)
                        self.assertEqual(bool(call['mask'][flat, slot]), slot <= band)
                        if slot > band:
                            self.assertIsNone(pose)
                            self.assertEqual(slot_time, 0.)
                            self.assertTrue(torch.equal(value[flat], torch.zeros_like(value[flat])))
                            continue
                        frame = target_frame-band+slot
                        self.assertEqual(float(call['audio'][flat, slot, 0, 0]), float(rows['ofi'][row, frame]))
                        torch.testing.assert_close(pose[flat], rows['pose'][row, :, frame])
                        if slot != band:
                            expected_time = [.4, .6, .8, 1.][slot] - .1*(step % 2)
                            self.assertAlmostEqual(slot_time, expected_time)
                            gt_value = (1-expected_time)*rows['latents'][row, :, frame] + expected_time*noise[row, :, frame]
                            torch.testing.assert_close(value[flat], gt_value)
                    for history in range(3):
                        frame = target_frame-band-3+history
                        value, pose, history_time, position, valid = frames[4+history]
                        self.assertEqual((history_time, position, valid), (.2, 3, True))
                        torch.testing.assert_close(value[flat], .8*rows['latents'][row, :, frame] + .2*noise[row, :, frame])
                        torch.testing.assert_close(pose[flat], rows['pose'][row, :, frame])
                    torch.testing.assert_close(frames[7][0][flat], rows['latents'][row, :, 0])
            trajectory = torch.arange(1, 9, dtype=torch.float32)[:, None, None, None]
            expected = expected - .1*(trajectory + 7*band + .125*(step+1))
        torch.testing.assert_close(output, expected.reshape(2, 4, 2, 1, 1).transpose(1, 2))

    def test_generated_target_cannot_enter_another_trajectory(self):
        rows, noise = source_rows()
        ordinary, poisoned = RecordingNet(), RecordingNet(poison_first=1e6)
        first = GTConditionedStabilizer()(ordinary, rows, noise=noise)
        second = GTConditionedStabilizer()(poisoned, rows, noise=noise)
        self.assertFalse(torch.equal(first[0, :, 0], second[0, :, 0]))
        torch.testing.assert_close(first[0, :, 1:], second[0, :, 1:])
        torch.testing.assert_close(first[1], second[1])
        for left, right in zip(ordinary.calls, poisoned.calls):
            for a, b in zip(left['frames'], right['frames']):
                torch.testing.assert_close(a[0][1:], b[0][1:])

    def test_insufficient_context_or_incorrect_fixed_noise_rejected(self):
        rows, noise = source_rows(frames=10)
        with self.assertRaisesRegex(ValueError, 'six preceding GT frames'):
            GTConditionedStabilizer()(RecordingNet(), rows, noise=noise)
        rows, noise = source_rows()
        with self.assertRaisesRegex(ValueError, 'complete source row'):
            GTConditionedStabilizer()(RecordingNet(), rows, noise=noise[:, :, 7:])

    def test_frozen_rollout_builds_no_gradient_graph(self):
        rows, noise = source_rows()
        rows['latents'].requires_grad_()
        noise.requires_grad_()
        result = GTConditionedStabilizer()(RecordingNet(), rows, noise=noise)
        self.assertFalse(result.requires_grad)
        self.assertIsNone(result.grad_fn)


if __name__ == '__main__':
    unittest.main()
