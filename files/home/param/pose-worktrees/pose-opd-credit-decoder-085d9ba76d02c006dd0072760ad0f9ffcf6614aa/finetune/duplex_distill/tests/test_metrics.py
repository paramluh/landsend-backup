"""Meaningful dashboard and unequal-evaluation-batch aggregation checks."""
import unittest

import torch

from finetune.duplex_distill.metrics import collect, sum_raw, summarize


CFG = dict(lw_word=1., w_cb0=.25, w_cb=.75, w_pcb0=.25, w_pcb=.75,
           modality_weights={'text': 1/3, 'audio': 1/3, 'pose': 1/3})


def batch(count, kl, ce):
    # Every position here is a lexical target; both codebook streams cover all
    # positions. Different counts expose erroneous means-of-batch-means.
    valid = torch.ones(count, dtype=torch.bool)
    kval = torch.full((count,), float(kl), dtype=torch.float64)
    cval = torch.full((count,), float(ce), dtype=torch.float64)
    stats = dict(valid=valid, kl_sum=kval.sum(), full_kl_sum=kval.sum(), data_ce_sum=cval.sum(),
                 data_weighted_ce_sum=5 * cval.sum(), full_kl=kval, data_ce=cval,
                 lexical_kl=kval / 2, data_word_ce_norm=cval)
    cb0 = dict(kl_sum=kval.sum(), data_ce_sum=cval.sum(), count=valid.sum())
    light = dict(kl=torch.full((31,), float(kl * count), dtype=torch.float64),
                 ce=torch.full((31,), float(ce * count), dtype=torch.float64),
                 count=torch.full((31,), count, dtype=torch.float64))
    out = {'text': {'stats': stats}, 'stats_s': cb0, 'stats_p': cb0}
    masks = {key: torch.zeros_like(valid) for key in ('pad', 'nw', 'eob')}
    masks['word'] = valid
    return collect(out, masks, [light, light], 1, 'cpu')


class MetricTests(unittest.TestCase):
    def test_whole_eval_token_weighting_and_dummy_batch(self):
        a, b, dummy = batch(1, 1, 2), batch(9, 3, 6), batch(0, 90, 90)
        metrics = summarize(sum_raw([a, b, dummy]), CFG)
        self.assertAlmostEqual(metrics['text_kl/full'], 2.8)
        self.assertAlmostEqual(metrics['text_kl/word_conditional'], 1.4)
        self.assertAlmostEqual(metrics['data_text_ce/full'], 5.6)
        self.assertAlmostEqual(metrics['data_loss/text_loss'], 28.0)
        for i in range(32):
            self.assertAlmostEqual(metrics[f'audio_codebooks/cb{i}'], 2.8)
            self.assertAlmostEqual(metrics[f'pose_codebooks/cb{i}'], 2.8)
            self.assertAlmostEqual(metrics[f'data_audio_codebooks/cb{i}'], 5.6)
            self.assertAlmostEqual(metrics[f'data_pose_codebooks/cb{i}'], 5.6)
        self.assertAlmostEqual(metrics['train/loss'], 2.8)
        self.assertAlmostEqual(metrics['data_loss/loss'], (28 + 5.6 + 5.6) / 3)
        wrong = (summarize(a, CFG)['text_kl/full'] + summarize(b, CFG)['text_kl/full']) / 2
        self.assertNotAlmostEqual(metrics['text_kl/full'], wrong)
        self.assertEqual(a['text'][4].item(), 1)  # sum_raw never mutates inputs.

    def test_codebook_specific_counts_and_stream_weights(self):
        raw = batch(2, 1, 3)
        raw['audio'][0, 5] = 12; raw['audio'][1, 5] = 20; raw['audio'][2, 5] = 4
        metrics = summarize(raw, {**CFG, 'w_cb0': 2, 'w_cb': 3})
        self.assertEqual(metrics['audio_codebooks/cb5'], 3)
        self.assertEqual(metrics['data_audio_codebooks/cb5'], 5)
        self.assertAlmostEqual(metrics['train/audio_loss'], 2 + 3 * (30 + 3) / 31)
        self.assertAlmostEqual(metrics['train/light_kl'], (30 + 3) / 31)
        self.assertAlmostEqual(metrics['data_loss/light_ce'], (30 * 3 + 5) / 31)

    def test_equal_modality_and_quarter_cb0_three_quarter_higher_mean_weights(self):
        raw = batch(2, 6, 2)
        # Give every codebook a distinct value to prove higher codebooks are
        # averaged across all 31, then scaled once by .75 and modality by 1/3.
        audio_kl = torch.arange(1., 33., dtype=torch.float64)
        pose_kl = torch.arange(101., 133., dtype=torch.float64)
        audio_ce = audio_kl + 40
        pose_ce = pose_kl + 50
        for stream, kl, ce in (('audio', audio_kl, audio_ce), ('pose', pose_kl, pose_ce)):
            raw[stream][0] = kl * raw[stream][2]
            raw[stream][1] = ce * raw[stream][2]
        result = summarize(raw, CFG)
        expected_audio = .25 * audio_kl[0] + .75 * audio_kl[1:].mean()
        expected_pose = .25 * pose_kl[0] + .75 * pose_kl[1:].mean()
        self.assertAlmostEqual(result['train/audio_loss'], float(expected_audio))
        self.assertAlmostEqual(result['train/pose_loss'], float(expected_pose))
        self.assertAlmostEqual(result['train/loss'], (6 + float(expected_audio) + float(expected_pose)) / 3)
        expected_data = (10 + .25 * float(audio_ce[0]) + .75 * float(audio_ce[1:].mean())
                        + .25 * float(pose_ce[0]) + .75 * float(pose_ce[1:].mean())) / 3
        self.assertAlmostEqual(result['data_loss/loss'], expected_data)
        for cb in range(32):
            self.assertEqual(result[f'audio_codebooks/cb{cb}'], float(audio_kl[cb]))
            self.assertEqual(result[f'pose_codebooks/cb{cb}'], float(pose_kl[cb]))
            self.assertEqual(result[f'data_audio_codebooks/cb{cb}'], float(audio_ce[cb]))
            self.assertEqual(result[f'data_pose_codebooks/cb{cb}'], float(pose_ce[cb]))

    def test_exact_metric_layout_and_text_diagnostic_values(self):
        raw = batch(10, 3, 4)
        # Counts differ by dataset category. Preserve each diagnostic's own
        # denominator, including lexical-conditional KL/CE on word positions.
        raw['text'][5:17] = torch.tensor([8, 12, 2, 15, 18, 3, 7, 8, 1, 36, 40, 4])
        raw['text'][17:] = torch.tensor([3., 5.])
        result = summarize(raw, CFG)
        aggregate = {'loss', 'text_loss', 'audio_loss', 'pose_loss'}
        text = {'full', 'word', 'pad', 'nw', 'eob', 'word_conditional'}
        books = {f'cb{cb}' for cb in range(32)}
        groups = {
            'train': aggregate | {'light_kl', 'pose_light_kl'},
            'text_kl': text,
            'audio_codebooks': books,
            'pose_codebooks': books,
            'data_loss': aggregate | {'light_ce', 'pose_light_ce'},
            'data_text_ce': text,
            'data_audio_codebooks': books,
            'data_pose_codebooks': books,
        }
        expected_keys = {f'{prefix}/{key}' for prefix, names in groups.items() for key in names}
        self.assertEqual(set(result), expected_keys)
        self.assertEqual(len(result), 152)
        # These are the values of the old six text diagnostics, with only
        # their namespace/name changed. No category or lexical reweighting.
        for prefix, values in (
            ('text_kl', dict(full=3., word=4., pad=5., nw=7., eob=9., word_conditional=1.5)),
            ('data_text_ce', dict(full=4., word=6., pad=6., nw=8., eob=10., word_conditional=2.5)),
        ):
            for key, expected in values.items():
                self.assertEqual(result[f'{prefix}/{key}'], expected)
        self.assertEqual(result['text_kl/full'], result['train/text_loss'])
        self.assertEqual(result['data_loss/text_loss'], 20.)
        self.assertEqual(result['train/light_kl'], 3.)
        self.assertEqual(result['train/pose_light_kl'], 3.)
        self.assertEqual(result['data_loss/light_ce'], 4.)
        self.assertEqual(result['data_loss/pose_light_ce'], 4.)

    def test_empty_eval_and_mismatched_shapes_rejected(self):
        with self.assertRaises(ValueError): sum_raw([])
        with self.assertRaises(ValueError): sum_raw([batch(1, 1, 1), {'text': torch.zeros(19)}])
        with self.assertRaises(ValueError): summarize({'text': torch.zeros(18), 'audio': torch.zeros(3,32), 'pose': torch.zeros(3,32)}, CFG)


if __name__ == '__main__':
    unittest.main()
