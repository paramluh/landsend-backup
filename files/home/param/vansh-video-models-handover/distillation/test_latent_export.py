"""CPU regression tests for long-video output publication."""

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file

from distillation.latent_export import persist_latents


class LatentExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_noncontiguous_outputs_roundtrip_short_and_full_geometry(self):
        for frames, dtype in ((5, torch.bfloat16), (5, torch.float32), (1875, torch.float32)):
            with self.subTest(frames=frames, dtype=dtype), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary)
                resolved = torch.randn(frames, 48, 30, 16, dtype=dtype).permute(1, 0, 2, 3)
                generated = resolved.contiguous() * .75
                self.assertFalse(resolved.is_contiguous())
                # This is the original production failure, including its view layout.
                with self.assertRaisesRegex(ValueError, 'contiguous'):
                    save_file({'latents': resolved.cpu()}, str(output/'old.safetensors'))
                receipts = persist_latents(output, [('sample_8', resolved),
                                                     ('sample_8_stabilised', generated)], verify=True)
                self.assertEqual(len(receipts), 2)
                for receipt, expected in zip(receipts, (resolved, generated)):
                    path = Path(receipt['path'])
                    restored = load_file(str(path))['latents']
                    self.assertEqual(restored.dtype, dtype)
                    self.assertEqual(restored.shape, (48, frames, 30, 16))
                    self.assertTrue(torch.equal(restored, expected))
                    self.assertTrue(restored.is_contiguous())
                    with path.open('rb') as handle:
                        self.assertEqual(receipt['sha256'], hashlib.file_digest(handle, 'sha256').hexdigest())
                    self.assertEqual(receipt['shape'], list(expected.shape))
                    self.assertEqual(receipt['dtype'], str(dtype))
                    self.assertTrue(receipt['finite'] and receipt['contiguous'])
                self.assertFalse(list(output.glob('*.partial')))

    def test_nonfinite_output_is_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            for value in (float('nan'), float('inf')):
                with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, 'Nonfinite output'):
                    persist_latents(temporary, [('sample_8', torch.tensor([value]))])
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_failed_write_preserves_existing_final_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            final = output/'sample_8_latents.safetensors'
            original = torch.tensor([1.0])
            save_file({'latents': original}, str(final))
            original_bytes = final.read_bytes()

            def interrupted_write(tensors, filename):
                self.assertEqual(Path(filename).suffix, '.partial')
                self.assertEqual(final.read_bytes(), original_bytes)
                Path(filename).write_bytes(b'incomplete')
                raise OSError('simulated interrupted write')

            with patch('distillation.latent_export.save_file', side_effect=interrupted_write):
                with self.assertRaisesRegex(OSError, 'interrupted'):
                    persist_latents(output, [('sample_8', torch.tensor([2.0]))])
            self.assertEqual(final.read_bytes(), original_bytes)
            self.assertTrue(torch.equal(load_file(str(final))['latents'], original))
            # A retry rewrites the temporary and then atomically replaces the final.
            persist_latents(output, [('sample_8', torch.tensor([2.0]))], verify=True)
            self.assertTrue(torch.equal(load_file(str(final))['latents'], torch.tensor([2.0])))
            self.assertFalse(list(output.glob('*.partial')))


if __name__ == '__main__':
    unittest.main()
