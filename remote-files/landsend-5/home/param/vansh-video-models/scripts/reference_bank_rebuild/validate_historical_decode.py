"""Bind decoded encoder inputs to the exact historical video selection."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pyarrow.parquet as pq

from hf_sources import write_json
from video_pipeline import digest


def validate(output_dir):
    output_dir = Path(output_dir)
    receipt = output_dir / 'historical-video-decode.json'
    receipt.unlink(missing_ok=True)
    policy_path = output_dir / 'historical-video-policy.json'
    selected_path = output_dir / 'video-selected.parquet'
    decoded_path = output_dir / 'video-encoding-input.parquet'
    pinned = {path: digest(path) for path in (policy_path, selected_path, decoded_path)}
    policy = json.loads(policy_path.read_text())
    if policy.get('selected_sha256') != pinned[selected_path]:
        raise ValueError('Historical video policy does not identify the selected Parquet')
    selected = pq.read_table(selected_path)
    decoded = pq.read_table(decoded_path)
    if not set(selected.column_names) <= set(decoded.column_names):
        raise ValueError('Decoded table lost original selected columns')
    if not decoded.select(selected.column_names).equals(selected, check_metadata=False):
        raise ValueError('Decoded rows differ from the historical selection or its sorted row order')
    if any(digest(path) != value for path, value in pinned.items()):
        raise ValueError('Historical selection/decode inputs changed during verification')
    report = {'passed': True, 'checked_at': datetime.now(timezone.utc).isoformat(),
              'policy_id': policy.get('policy_id'), 'policy_sha256': pinned[policy_path],
              'selected_sha256': pinned[selected_path], 'decoded_sha256': pinned[decoded_path],
              'rows': selected.num_rows, 'selected_columns': selected.column_names,
              'comparison': 'Exact Arrow table equality in selected column and row order; schema metadata ignored'}
    write_json(receipt, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    report = validate(parser.parse_args().output_dir)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
