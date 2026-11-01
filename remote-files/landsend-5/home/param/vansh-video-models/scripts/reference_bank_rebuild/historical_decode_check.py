"""Exact lineage checks for historical selection through the frozen decoder."""
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from hf_sources import write_json
from validate_historical_decode import validate
from video_pipeline import digest


class HistoricalDecodeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.selected = pa.table({'uuid': ['actor', 'actor'], 'property': [2, 3], 'index': [0, 0],
                                  'frame_index': [123, 234], 'optional_metric': [None, 1.25]})
        pq.write_table(self.selected.replace_schema_metadata({'original': 'selection'}), self.root / 'video-selected.parquet')
        self.decoded = self.selected.append_column('image_path', pa.array(['first.png', 'second.png']))
        self.decoded = self.decoded.append_column('prepared_image', pa.array([False, False]))
        self.save_decoded(self.decoded)
        write_json(self.root / 'historical-video-policy.json', {'policy_id': 'historical-fixture',
                   'selected_sha256': digest(self.root / 'video-selected.parquet')})

    def save_decoded(self, table):
        pq.write_table(table.replace_schema_metadata({'different': 'decoder metadata'}), self.root / 'video-encoding-input.parquet')

    def test_exact_projection_ignores_only_metadata_and_pins_both_hashes(self):
        result = validate(self.root)
        self.assertTrue(result['passed'])
        self.assertEqual(result['rows'], 2)
        self.assertEqual(result['selected_sha256'], digest(self.root / 'video-selected.parquet'))
        self.assertEqual(result['decoded_sha256'], digest(self.root / 'video-encoding-input.parquet'))

    def test_row_value_order_count_or_columns_change_rejects_and_removes_stale_receipt(self):
        variants = [self.decoded.take(pa.array([1, 0])), self.decoded.slice(0, 1),
                    self.decoded.drop(['frame_index']),
                    self.decoded.set_column(3, 'frame_index', pa.array([124, 234]))]
        for table in variants:
            with self.subTest(columns=table.column_names, rows=table.num_rows):
                self.save_decoded(self.decoded)
                validate(self.root)
                self.save_decoded(table)
                with self.assertRaises(ValueError):
                    validate(self.root)
                self.assertFalse((self.root / 'historical-video-decode.json').exists())

    def test_changed_selected_policy_binding_rejects(self):
        policy = self.root / 'historical-video-policy.json'
        write_json(policy, {'selected_sha256': '0' * 64})
        with self.assertRaisesRegex(ValueError, 'does not identify'):
            validate(self.root)


if __name__ == '__main__':
    unittest.main(verbosity=2)
