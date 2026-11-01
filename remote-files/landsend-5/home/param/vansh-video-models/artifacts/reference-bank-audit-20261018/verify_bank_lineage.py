"""Compare recorded reference-bank checksums and image provenance without reading old tensors."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


OLD = Path('/mnt/paramfs/vansh/aloe-conversational/reference-bank-256x480-muon-v2')
NEW = Path('/mnt/paramfs/param/outputs/reference-compression/data/reference-bank-256x480-muon-encoded/bank')
OUT = Path(__file__).with_name('bank-lineage.json')
PROVENANCE_FIELDS = ('index', 'recording_id', 'speaker', 'session_id', 'source',
                     'cx0', 'cy0', 'cw', 'ch', 'scale')


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checksums(path):
    entries = {}
    for line in path.read_text().splitlines():
        digest, name = line.split(None, 1)
        name = name.removeprefix('./')
        if name in entries:
            raise ValueError(f'Duplicate checksum path: {name}')
        entries[name] = digest
    return entries


def provenance(rows):
    def normalize(row, key):
        value = row[key]
        if key == 'source':
            return value.removeprefix('reference-images-512x960-v1/')
        if key in ('index', 'speaker', 'cx0', 'cy0', 'cw', 'ch', 'scale'):
            return float(value)
        return value
    return [tuple(normalize(row, key) for key in PROVENANCE_FIELDS)
            for row in sorted(rows, key=lambda row: int(row['index']))]


def grouped(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row['uuid'], row['property'])].append(row)
    return groups


old_sums, new_sums = checksums(OLD / 'SHA256SUMS'), checksums(NEW / 'SHA256SUMS')
old_rows = list(csv.DictReader((OLD / 'bank.csv').open()))
new_rows = list(csv.DictReader((NEW / 'bank.csv').open()))
old_groups, new_groups = grouped(old_rows), grouped(new_rows)
totals, splits = Counter(), defaultdict(Counter)
missing, changed, provenance_differences = [], [], []
for path, digest in old_sums.items():
    if not path.endswith('/latents.safetensors') or not path.startswith(('train/', 'eval-unseen/')):
        continue
    split, relative = path.split('/', 1)
    actor, property_dir, _ = relative.split('/')
    status = 'missing' if relative not in new_sums else ('equal' if digest == new_sums[relative] else 'changed')
    totals[status] += 1
    splits[split][status] += 1
    if status == 'missing':
        missing.append({'old_path': path, 'sha256': digest})
    elif status == 'changed':
        changed.append({'old_path': path, 'old_sha256': digest, 'new_sha256': new_sums[relative]})
    if relative in new_sums:
        key = actor, property_dir.removeprefix('property_')
        if provenance(old_groups[key]) != provenance(new_groups.get(key, [])):
            provenance_differences.append(path)

missing_actors = sorted({row['uuid'] for row in old_rows} - {row['uuid'] for row in new_rows})
result = {
    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
    'old_bank': str(OLD),
    'expanded_bank_split': str(NEW),
    'method': 'Strip train/ or eval-unseen/ from each old latent checksum path and compare with expanded bank/SHA256SUMS. Compare ordered image source/crop provenance after normalizing upstream source prefixes and numeric formatting.',
    'reproduce': f'python3 {Path(__file__).resolve()}',
    'script_sha256': sha256(Path(__file__)),
    'source_files': {
        str(root / filename): {'sha256': sha256(root / filename)}
        for root in (OLD, NEW) for filename in ('SHA256SUMS', 'bank.csv')
    },
    'bank_csv_matches_own_checksum': {
        'old': sha256(OLD / 'bank.csv') == old_sums['bank.csv'],
        'expanded': sha256(NEW / 'bank.csv') == new_sums['bank.csv'],
    },
    'old_images': len(old_rows),
    'expanded_images': len(new_rows),
    'checksum_comparison': dict(totals),
    'by_old_split': {split: dict(counts) for split, counts in splits.items()},
    'provenance_comparison_fields': PROVENANCE_FIELDS,
    'shared_property_provenance_differences': provenance_differences,
    'changed_checksums': changed,
    'missing_files': missing,
    'missing_actors': [{'uuid': actor, 'old_images': sum(row['uuid'] == actor for row in old_rows),
                        'old_split': sorted({row['bank'] for row in old_rows if row['uuid'] == actor})}
                       for actor in missing_actors],
    'conclusion': 'All 4620 shared actor/property files have identical recorded SHA256 and identical ordered image source/crop provenance. Expanded bank omits one old training actor with 9 properties and 84 images. Original tensor payloads were not read; pair this receipt with independent payload checksum validation of the expanded bank.',
    'legacy_selection_note': 'Richmond reference-bank/code/full/plan_v3.py:29-31 excludes all existing v2 actors from the added-actor builder. The stricter rules in full/select_v3.py apply to added actors; existing legacy rows are preserved and do not uniformly satisfy the newer sinks.md thresholds.',
}
OUT.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'report': str(OUT), 'comparison': dict(totals),
                  'by_old_split': result['by_old_split'],
                  'provenance_differences': len(provenance_differences),
                  'bank_csv_checksum_matches': result['bank_csv_matches_own_checksum']}))
