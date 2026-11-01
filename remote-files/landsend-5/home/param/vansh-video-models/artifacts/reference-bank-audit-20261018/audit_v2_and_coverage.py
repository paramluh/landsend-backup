"""Read-only audit of the prior v2 bank and actor coverage of both banks."""
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

BASE = Path('/mnt/paramfs/vansh/aloe-conversational')
V2 = BASE / 'reference-bank-256x480-muon-v2'
EXPANDED = Path('/mnt/paramfs/param/outputs/reference-compression/data/reference-bank-256x480-muon-encoded')
ARCHIVE = Path('/mnt/paramfs/vansh/richmond-7/data/csvs')
OUT = Path(__file__).parent


def rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def check(entry):
    expected, rel = entry.split(maxsplit=1)
    path = V2 / rel
    try:
        actual = digest(path)
        return {'path': rel, 'status': 'ok' if actual == expected else 'checksum_mismatch',
                'bytes': path.stat().st_size, 'expected_sha256': expected}
    except OSError as error:
        return {'path': rel, 'status': type(error).__name__, 'error': str(error),
                'exists': path.is_file(), 'expected_sha256': expected}


def main():
    refrows = rows(V2 / 'bank.csv')
    groups = defaultdict(list)
    for row in refrows:
        groups[(row['bank'], row['uuid'], int(row['property']))].append(int(row['index']))
    manifest = (V2 / 'SHA256SUMS').read_text().splitlines()
    with ThreadPoolExecutor(max_workers=8) as pool:
        checks = list(pool.map(check, manifest))
    expected = {str(Path(split) / actor / f'property_{prop}' / 'latents.safetensors')
                for split, actor, prop in groups}
    actual = {str(p.relative_to(V2)) for split in ('train', 'eval-unseen')
              for p in (V2 / split).glob('*/property_*/latents.safetensors')}
    properties = {'v2/train': defaultdict(set), 'v2/eval-unseen': defaultdict(set),
                  'expanded/bank': defaultdict(set)}
    for split, actor, prop in groups:
        properties['v2/' + split][actor].add(prop)
    expanded_rows = rows(EXPANDED / 'bank/bank.csv')
    for row in expanded_rows:
        properties['expanded/bank'][row['uuid']].add(int(row['property']))
    coverage = []
    names = ['aloe-studio-no-blackbands-high-quality.csv', 'aloe-app-selected-clips.csv',
             'aloe-studio-seen-eval.csv', 'aloe-studio-unseen-eval.csv', 'aloe-eval-beatles.csv']
    for folder in (ARCHIVE / 'current', BASE / 'csvs'):
        for name in names:
            path = folder / name
            data = rows(path)
            actors = {row['uuid'] for row in data}
            result = {'csv': str(path), 'sha256': digest(path), 'clips': len(data), 'actors': len(actors)}
            for label, mapping in properties.items():
                missing = actors - mapping.keys()
                result[label] = {
                    'actors_with_refs': len(actors) - len(missing),
                    'actors_without_refs': sorted(missing),
                    'clips_without_refs': sum(row['uuid'] in missing for row in data),
                    'actors_by_number_of_properties': dict(sorted(Counter(len(mapping.get(a, ())) for a in actors).items())),
                }
            coverage.append(result)
    report = {
        'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'uid': os.getuid(),
        'bank': str(V2), 'images': len(refrows), 'latent_files': len(groups),
        'images_by_split': dict(Counter(r['bank'] for r in refrows)),
        'origins': dict(Counter(r['origin'] for r in refrows)),
        'actors_by_split': {s: len({a for split, a, p in groups if split == s}) for s in ('train', 'eval-unseen')},
        'noncontiguous_indices': [list(key) for key, indices in groups.items() if sorted(indices) != list(range(len(indices)))],
        'missing_latent_paths': sorted(expected - actual), 'extra_latent_paths': sorted(actual - expected),
        'manifest_sha256': digest(V2 / 'SHA256SUMS'),
        'manifest_check_status_counts': dict(Counter(c['status'] for c in checks)),
        'manifest_checks': checks, 'coverage': coverage,
        'scope': 'Every v2 manifest entry was opened for checksum verification. Permission failures are not checksum successes. Metadata coverage is from bank.csv, not proof of readable tensors. No bank contents or permissions changed.',
    }
    dest = OUT / 'v2-and-coverage.json'
    dest.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('manifest_checks', 'coverage')}, indent=2))
    for c in coverage:
        print(c['csv'], c['actors'], 'actors', {k: v['actors_with_refs'] for k, v in c.items() if k in properties})
    print(dest)


if __name__ == '__main__':
    main()
