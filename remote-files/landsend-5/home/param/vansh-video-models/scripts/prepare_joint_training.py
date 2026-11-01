"""Create the exact studio/no-blackbands/no-crossbleed manifest and audit local conditioning assets."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import struct


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def header(path):
    with path.open('rb') as handle:
        length = struct.unpack('<Q', handle.read(8))[0]
        if not 2 <= length <= 1024 * 1024:
            raise ValueError('Invalid safetensors header length')
        return json.loads(handle.read(length))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-csv', type=Path, default=Path('/mnt/paramfs/vansh/aloe-conversational/csvs/aloe-studio-no-blackbands-high-quality.csv'))
    parser.add_argument('--data-root', type=Path, default=Path('/mnt/paramfs/vansh/aloe-conversational/conv-4.3k-5mins-muon-neutrino-256x480-v1'))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=32)
    args = parser.parse_args()
    with args.source_csv.open(newline='') as handle:
        reader = csv.DictReader(handle)
        fields, source = reader.fieldnames, list(reader)
    required = {'paths', 'latent_t', 'uuid', 'crossbleed', 'suspected_crossbleed', 'blackbands'}
    if not required <= set(fields):
        raise ValueError(f'Missing source columns: {required - set(fields)}')
    if any(row['blackbands'].strip() != 'no' for row in source):
        raise ValueError('Expected the no-blackbands studio source manifest')
    if len({row['paths'] for row in source}) != len(source):
        raise ValueError('Duplicate source clips')
    kept, rejected = [], []
    for row in source:
        (kept if row['crossbleed'].strip() == 'no' and row['suspected_crossbleed'].strip() == 'no' else rejected).append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (('aloe-studio-no-blackbands-high-quality-no-crossbleed.csv', kept), ('excluded-crossbleed.csv', rejected)):
        with (args.output_dir / name).open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    output = args.output_dir / 'aloe-studio-no-blackbands-high-quality-no-crossbleed.csv'
    report = {'source_csv': str(args.source_csv), 'source_sha256': sha256(args.source_csv),
              'training_csv': str(output), 'training_sha256': sha256(output),
              'input_chunks': len(source), 'retained_chunks': len(kept), 'excluded_chunks': len(rejected),
              'actors': len({row['uuid'] for row in kept}),
              'hours': sum(float(row['duration']) for row in kept) / 3600,
              'filter': "blackbands=='no' AND crossbleed=='no' AND suspected_crossbleed=='no'",
              'pose_file': 'pose_latents_electron_v3.safetensors', 'audio_file': 'mimi_local.safetensors'}
    (args.output_dir / 'manifest-receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)

    def inspect(row):
        problems = []
        for name in ('video_latents.safetensors', report['pose_file'], report['audio_file']):
            path = args.data_root / row['paths'] / name
            try:
                info = header(path)
                if name == report['audio_file']:
                    if 'codes' not in info or len(info['codes']['shape']) != 2 or info['codes']['shape'][0] < 16:
                        raise ValueError('Missing or invalid Mimi codebooks')
                else:
                    shape = info['latents']['shape']
                    if info['latents']['dtype'] != 'BF16' or shape != [int(row['latent_t']), 48, 30, 16]:
                        raise ValueError(f'Wrong latent shape: {shape}')
            except Exception as error:
                problems.append({'clip': row['paths'], 'file': name, 'type': type(error).__name__, 'error': str(error)})
        return problems

    with ThreadPoolExecutor(args.workers) as pool:
        issues = [issue for problems in pool.map(inspect, kept) for issue in problems]
    report.update(asset_issues=issues, asset_issue_counts=dict(Counter(f'{row["file"]}:{row["type"]}' for row in issues)),
                  assets_readable_and_valid=not issues)
    (args.output_dir / 'asset-audit.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'asset_issues'}), flush=True)
    if issues:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
