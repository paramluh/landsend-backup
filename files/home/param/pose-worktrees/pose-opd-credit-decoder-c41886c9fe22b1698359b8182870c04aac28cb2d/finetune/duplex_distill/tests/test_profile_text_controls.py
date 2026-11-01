"""Validated text-head and batch controls are shared by training and profiling."""
from contextlib import redirect_stderr
import io
import unittest

from finetune.duplex_distill.train import args_parser


class ProfileTextControlTests(unittest.TestCase):
    def parse(self, *arguments):
        return args_parser(['--run-dir', '/tmp/unused-profile-argument-test', *arguments])

    def test_ordinary_training_keeps_original_defaults(self):
        args = self.parse()
        self.assertEqual(args.text_chunk_size, 32)
        self.assertFalse(args.text_direct_addmm)
        self.assertEqual(args.micro_batch_size, 1)

    def test_validated_candidate_is_available_for_actual_training(self):
        args = self.parse('--text-chunk-size', '512', '--text-direct-addmm', '--micro-batch-size', '2')
        self.assertEqual(args.text_chunk_size, 512)
        self.assertTrue(args.text_direct_addmm)
        self.assertEqual(args.micro_batch_size, 2)
        self.assertEqual(args.profile_steps, 0)

    def test_bounded_profile_accepts_validated_candidate(self):
        args = self.parse('--profile-steps', '6', '--profile-text-chunk-size', '512',
                          '--profile-text-direct-addmm', '--profile-micro-batch-size', '2')
        self.assertEqual(args.text_chunk_size, 512)
        self.assertTrue(args.text_direct_addmm)
        self.assertEqual(args.micro_batch_size, 2)

    def test_nonpositive_chunk_rejected_before_gpu_initialization(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.parse('--profile-steps', '6', '--profile-text-chunk-size', '0')


if __name__ == '__main__':
    unittest.main()
