import unittest

import torch

from bimodal.cached_resolver import ResolverNet, context_frames
from bimodal.graph_resolver import GraphResolver
from distillation.test_wavefront_audio_guidance import small_models
from utils.common import AUTOCAST_DTYPE


@unittest.skipUnless(torch.cuda.is_available(), 'reference-conditioned resolver requires CUDA')
class ResolverReferenceTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(841)
        self.gradient_mode = torch.no_grad()
        self.gradient_mode.__enter__()
        self.addCleanup(self.gradient_mode.__exit__, None, None, None)
        self.autocast_mode = torch.autocast('cuda', dtype=AUTOCAST_DTYPE, cache_enabled=False)
        self.autocast_mode.__enter__()
        self.addCleanup(self.autocast_mode.__exit__, None, None, None)

    def make_case(self, num_refs, batch=2):
        models = small_models(num_refs, history=2, present=1)
        resolver = ResolverNet(models)
        references = torch.randn(batch, 2, num_refs, 2, 3, device='cuda')
        keep = torch.ones(batch, num_refs, dtype=torch.bool, device='cuda')
        if num_refs:
            keep[0, 1::2] = False
            if batch > 1:
                keep[1] = False
            resolver.set_refs(references, keep)
        values = {
            'out': torch.randn(batch, 2, 7, 2, 3, device='cuda'),
            'pose': torch.randn(batch, 2, 7, 2, 3, device='cuda'),
            'tokens': torch.randn(batch, 16, models.net.layers[0].dim, device='cuda'),
            'codes': torch.zeros(batch, 1, 750, device='cuda', dtype=torch.long),
            'references': references,
            'keep': keep,
        }
        return models, resolver, values

    def full_velocity(self, models, values, frame, latent, time):
        batch = latent.shape[0]
        positions = context_frames(frame, models.num_history) + [frame]
        indices = [max(position, 0) for position in positions]
        window = torch.cat((values['out'][:, :, indices[:-1]], latent[:, :, None]), dim=2)
        schedule = torch.zeros(batch, len(positions), device='cuda')
        schedule[:, -1] = time
        inputs = (
            window, schedule, values['tokens'], values['pose'][:, :, indices],
            torch.tensor([positions] * batch, device='cuda'), torch.arange(batch, device='cuda'),
            torch.ones(batch, 1, dtype=torch.bool, device='cuda'),
        )
        if models.net.layers[0].num_refs:
            inputs += (values['references'], values['keep'])
        return models.net(inputs)[:, :, -1].float()

    def full_frame(self, models, values, frame, latent, schedule):
        for step in range(schedule.numel() - 1):
            latent = latent + (schedule[step + 1] - schedule[step]) * self.full_velocity(
                models, values, frame, latent, schedule[step] * 1000)
        return latent

    def test_cached_velocity_matches_full_window_with_pads_and_missing_references(self):
        for num_refs in (0, 9):
            with self.subTest(num_refs=num_refs):
                models, resolver, values = self.make_case(num_refs)
                self.assertEqual(models.net.layers[-1].num_refs, num_refs)
                self.assertEqual(resolver.final.num_refs, 0)
                for frame in (1, 2, 4):
                    positions = context_frames(frame, resolver.H)
                    indices = [max(position, 0) for position in positions]
                    context = resolver.context(
                        values['out'][:, :, indices], values['pose'][:, :, indices],
                        torch.tensor([positions] * 2, device='cuda'))
                    self.assertEqual(context[0].shape[2], (num_refs + 1 + resolver.H) * 6)
                    audio = resolver.audio(values['codes'], frame, values['tokens'])
                    latent = torch.randn_like(values['out'][:, :, frame])
                    for time in (200., 70.):
                        with self.subTest(frame=frame, time=time):
                            expected = self.full_velocity(models, values, frame, latent, time)
                            actual = resolver.velocity(latent, values['pose'][:, :, frame], frame,
                                                       torch.tensor(time, device='cuda'), context, audio)
                            torch.testing.assert_close(actual, expected, rtol=.025, atol=.025)

    def test_dropped_reference_values_do_not_change_cache_or_predictions(self):
        models, resolver, values = self.make_case(9)
        original_cache = tuple(value.clone() for value in resolver.reference_context)
        original_pointers = tuple(value.data_ptr() for value in resolver.reference_context)
        frame = 2
        latent = values['out'][:, :, frame].clone()
        schedule = torch.linspace(.2, 0, 11, device='cuda')
        expected = self.full_frame(models, values, frame, latent, schedule)
        dropped = (~values['keep'])[:, None, :, None, None]
        values['references'] = torch.where(dropped, values['references'] + 1000, values['references'])
        resolver.set_refs(values['references'], values['keep'])
        self.assertEqual(tuple(value.data_ptr() for value in resolver.reference_context), original_pointers)
        for actual, previous in zip(resolver.reference_context, original_cache):
            torch.testing.assert_close(actual, previous, rtol=0, atol=0)
        resolver.frame(values['out'], values['pose'], values['codes'], frame, latent, schedule, values['tokens'])
        torch.testing.assert_close(values['out'][:, :, frame], expected, rtol=.025, atol=.025)

    def test_graph_replay_uses_new_references_without_reallocating_cache(self):
        for num_refs in (0, 9):
            with self.subTest(num_refs=num_refs):
                models, _, values = self.make_case(num_refs, batch=1)
                resolver = GraphResolver(models, num_steps=10)
                if num_refs:
                    resolver.set_refs(values['references'], values['keep'])
                    pointers = tuple(value.data_ptr() for value in resolver.reference_context)
                graph = None
                for frame in (1, 2, 4):
                    latent = values['out'][:, :, frame].clone()
                    schedule = torch.linspace(.2 if frame != 2 else .15, 0, 11, device='cuda')
                    if num_refs and frame == 4:
                        old_expected = self.full_frame(models, values, frame, latent, schedule)
                        values['references'].normal_(mean=3.)
                        values['keep'].fill_(True)
                        resolver.set_refs(values['references'], values['keep'])
                        self.assertEqual(tuple(value.data_ptr() for value in resolver.reference_context), pointers)
                    expected = self.full_frame(models, values, frame, latent, schedule)
                    if num_refs and frame == 4:
                        self.assertGreater(float((expected - old_expected).abs().max()), 1e-4)
                    resolver.frame(values['out'], values['pose'], values['codes'], frame,
                                   latent, schedule, values['tokens'])
                    torch.testing.assert_close(values['out'][:, :, frame], expected, rtol=.025, atol=.025)
                    if graph is None:
                        graph = resolver.graph
                    else:
                        self.assertIs(resolver.graph, graph)

    def test_missing_and_invalid_reference_inputs_fail_explicitly(self):
        models, _, values = self.make_case(9)
        resolver = ResolverNet(models)
        positions = torch.tensor([[0, -1, -1]] * 2, device='cuda')
        with self.assertRaisesRegex(ValueError, 'requires set_refs'):
            resolver.context(values['out'][:, :, :3], values['pose'][:, :, :3], positions)
        with self.assertRaisesRegex(ValueError, 'Reference latents'):
            resolver.set_refs(values['references'][:, :, :-1], values['keep'])
        with self.assertRaisesRegex(ValueError, 'Reference keep mask'):
            resolver.set_refs(values['references'], values['keep'].float())
        with self.assertRaisesRegex(ValueError, 'Reference keep mask'):
            resolver.set_refs(values['references'], values['keep'][:, :-1])
        resolver.set_refs(values['references'], values['keep'])
        with self.assertRaisesRegex(ValueError, 'geometry changed'):
            resolver.set_refs(values['references'][:1], values['keep'][:1])
        no_refs = ResolverNet(small_models(0, history=2, present=1))
        with self.assertRaisesRegex(ValueError, 'no reference conditioning'):
            no_refs.set_refs(values['references'], values['keep'])


if __name__ == '__main__':
    unittest.main()
