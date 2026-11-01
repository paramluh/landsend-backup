"""Exact rewind-boundary equivalence when natural sampling ignores text history."""
import copy
from types import SimpleNamespace
import unittest

import torch

from finetune.pose_opd_credit.partner_faststate import (
    disable_partner_text_history, finalize_natural_block,
)


class State:
    def __init__(self):
        self.pose = False
        self._opd_track_text_history = False
        self._opd_sampler_backend = 'inverse_cdf'
        self._opd_rewind_window = 8
        self.hn_prev = torch.zeros(3, 8, dtype=torch.bfloat16)
        self._opd_rewind_states = {'hn_prev': torch.randn(8, 3, 8).bfloat16()}
        self._opd_rewind_frames = [None] * 8
        self.prefixes = torch.tensor([3, 8, 11])
        self.positions = torch.zeros(3, dtype=torch.long)
        self.tok_now = torch.zeros(3, dtype=torch.long)
        self.lane = torch.arange(3 * 100).reshape(3, 100)
        self.adapt = torch.randn(3, 5)
        self._opd_next_frame, self._opd_pending_frame = None, None

    def _rewind_tensors(self):
        return {'hn_prev': self.hn_prev}

    def set_f(self, frame):
        self.positions.copy_(self.prefixes + frame)

    def _supply_text(self, tokens):
        self.lane[torch.arange(3), self.positions] = tokens
        self.tok_now.copy_(tokens)
        self.adapt[:, 0].zero_()
        self.adapt[:, 1:].fill_(1.)


class FastPartnerStateTests(unittest.TestCase):
    def test_every_rewind_hidden_boundary_and_final_state_match_sequential(self):
        torch.manual_seed(139)
        for start in range(16):
            expected = State()
            actual = copy.deepcopy(expected)
            hidden = torch.randn(3, 6, 8).bfloat16()
            tokens = expected.lane[torch.arange(3)[:, None],
                                   expected.prefixes[:, None] + start + torch.arange(6)]
            for offset, frame in enumerate(range(start, start + 6)):
                expected.set_f(frame)
                expected._supply_text(tokens[:, offset])
                expected.hn_prev.copy_(hidden[:, offset])
                expected._opd_rewind_states['hn_prev'][(frame + 1) % 8].copy_(expected.hn_prev)
                expected._opd_rewind_frames[(frame + 1) % 8] = frame + 1
                expected._opd_next_frame, expected._opd_pending_frame = frame + 1, None
            finalize_natural_block(actual, hidden, tokens, start, start + 6, torch.arange(6))
            for field in ('hn_prev', 'positions', 'tok_now', 'lane', 'adapt'):
                torch.testing.assert_close(getattr(actual, field), getattr(expected, field), rtol=0, atol=0)
            torch.testing.assert_close(actual._opd_rewind_states['hn_prev'],
                                       expected._opd_rewind_states['hn_prev'], rtol=0, atol=0)
            self.assertEqual(actual._opd_rewind_frames, expected._opd_rewind_frames)
            self.assertEqual(actual._opd_next_frame, expected._opd_next_frame)
            self.assertIsNone(actual._opd_pending_frame)

    def test_opt_in_only_discards_unused_ring_snapshots(self):
        actor = State()
        actor._opd_track_text_history = True
        hidden = actor._opd_rewind_states['hn_prev']
        actor._opd_rewind_states['ring.ptr'] = torch.zeros(8, 3)
        receipt = disable_partner_text_history(actor)
        self.assertFalse(actor._opd_track_text_history)
        self.assertEqual(receipt['rewind_state_tensors'], ['hn_prev'])
        self.assertIs(actor._opd_rewind_states['hn_prev'], hidden)
        self.assertEqual(list(actor._opd_rewind_states), ['hn_prev'])
        for override in ({'pose': True}, {'_opd_sampler_backend': 'multinomial'}):
            bad = State()
            for name, value in override.items():
                setattr(bad, name, value)
            with self.assertRaises(ValueError):
                disable_partner_text_history(bad)

    def test_rejects_other_required_rewind_state(self):
        actor = State()
        actor._rewind_tensors = lambda: {'hn_prev': actor.hn_prev, 'required': torch.zeros(1)}
        with self.assertRaises(ValueError):
            finalize_natural_block(actor, torch.zeros(3, 6, 8), torch.zeros(3, 6), 5, 11, torch.arange(6))


if __name__ == '__main__':
    unittest.main()
