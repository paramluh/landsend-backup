"""Publish an audited studio-only CSV mirror without changing archived sources.

Studio membership comes from exact recording/speaker/chunk identities in
all-studio.csv. Actor identity alone is insufficient: studio actors can also
have app tracks, and a recording can have one studio and one app speaker.
"""

import argparse
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from upstream import BEATLES_HELD_ROWS, BLACKBANDS_CSV, CSV_ROOT, load_corpus


ROOT = Path('/mnt/paramfs/param/outputs/reference-bank-rebuild-20261018')
TRAINING_CSV = Path('/mnt/paramfs/param/outputs/joint-nine-ref-20261018/assets/data/'
                    'aloe-studio-no-blackbands-high-quality-no-crossbleed.csv')
FILES = ('all.csv', 'all-studio.csv', 'all-app.csv',
         'aloe-studio-no-blackbands-high-quality.csv', 'aloe-app-selected-clips.csv',
         'aloe-studio-seen-eval.csv', 'aloe-studio-unseen-eval.csv',
         'aloe-eval-beatles.csv', 'aloe-unseen-excess-chunks.csv')
APP_FILES = {'all-app.csv', 'aloe-app-selected-clips.csv'}
IDENTITY = ('uuid', 'session_id', 'recording_id', 'speaker', 'chunk')


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_table(path):
    data = Path(path).read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig'), newline=''))
    if set(IDENTITY) - set(reader.fieldnames or ()):
        raise ValueError(f'{path}: missing identity columns')
    rows = list(reader)
    if any(None in row or any(row.get(field) in (None, '') for field in IDENTITY) for row in rows):
        raise ValueError(f'{path}: malformed row or empty identity')
    return data, reader.fieldnames, rows


def chunk_key(row):
    return row['recording_id'], int(row['speaker']), int(row['chunk'])


def track_key(row):
    return chunk_key(row)[:2]


def identity(row):
    return row['uuid'], row['session_id'], *chunk_key(row)


def counts(rows):
    return {'rows': len(rows), 'chunks': len({chunk_key(row) for row in rows}),
            'tracks': len({track_key(row) for row in rows}),
            'actors': len({row['uuid'] for row in rows})}


def filtered_bytes(original, fields, rows, selected):
    if selected == rows:
        return original  # Preserve evaluation order and bytes when no rows change.
    handle = io.StringIO(newline='')
    writer = csv.DictWriter(handle, fieldnames=fields)
    writer.writeheader()
    writer.writerows(selected)
    return handle.getvalue().encode('utf-8')


def prepare(archive_root, output_dir, blackbands_csv, training_csv, *,
            expected_actors=None, expected_training_rows=None, expected_training_actors=None):
    archive_root, output_dir = Path(archive_root).resolve(), Path(output_dir).resolve()
    blackbands_csv, training_csv = Path(blackbands_csv).resolve(), Path(training_csv).resolve()
    if output_dir == archive_root or archive_root in output_dir.parents:
        raise ValueError('The derived scope must be outside the archived CSV directory')
    tables = {name: read_table(archive_root / name) for name in FILES}
    studio_rows, app_rows = tables['all-studio.csv'][2], tables['all-app.csv'][2]
    studio = {chunk_key(row): row for row in studio_rows}
    if not studio or len(studio) != len(studio_rows):
        raise ValueError('Authoritative all-studio.csv is empty or has duplicate chunk identities')
    if any(row.get('kind') != 'studio' for row in studio_rows):
        raise ValueError('Authoritative all-studio.csv contains a non-studio kind')
    studio_tracks = {track_key(row) for row in studio_rows}
    app_tracks = {track_key(row) for row in app_rows}
    if studio_tracks & app_tracks:
        raise ValueError('A source track occurs in both studio and app; cannot safely scan its video')
    track_identities = {}
    for row in studio_rows:
        key, value = track_key(row), (row['uuid'], row['session_id'])
        if key in track_identities and track_identities[key] != value:
            raise ValueError(f'Conflicting authoritative studio track identity: {key}')
        track_identities[key] = value

    def is_studio(row):
        authoritative = studio.get(chunk_key(row))
        if authoritative is None:
            return False
        if identity(row) != identity(authoritative) or row.get('kind', 'studio') != 'studio':
            raise ValueError(f'Identity/kind disagrees with authoritative studio chunk: {chunk_key(row)}')
        if row.get('paths') != authoritative.get('paths'):
            raise ValueError(f'Path disagrees with authoritative studio chunk: {chunk_key(row)}')
        return True

    derived, selected_rows, records = {}, {}, {}
    for name, (original, fields, rows) in tables.items():
        selected = [] if name in APP_FILES else [row for row in rows if is_studio(row)]
        if name == 'aloe-eval-beatles.csv' and selected != rows:
            raise ValueError('Every Beatles row must be studio; filtering would change historical held indices')
        selected_rows[name] = selected
        derived[name] = filtered_bytes(original, fields, rows, selected)
        records[name] = {'original_path': str(archive_root / name), 'original_sha256': sha256(original),
                         'derived_sha256': sha256(derived[name]), 'original': counts(rows),
                         'derived': counts(selected), 'excluded_rows': len(rows) - len(selected),
                         'excluded_actors': len({r['uuid'] for r in rows} - {r['uuid'] for r in selected}),
                         'ordering': 'original row order preserved'}
    if {chunk_key(row) for row in selected_rows['all.csv']} != set(studio):
        raise ValueError('all.csv does not contain every authoritative studio chunk')
    beatles = selected_rows['aloe-eval-beatles.csv']
    if len(beatles) <= max(BEATLES_HELD_ROWS):
        raise ValueError('Beatles CSV is too short for the historical 16 held rows')

    training_bytes, _, training_rows = read_table(training_csv)
    if not all(is_studio(row) for row in training_rows):
        raise ValueError('Current training CSV contains a non-studio chunk')
    if any(track_key(row) in app_tracks for row in training_rows):
        raise ValueError('Current training CSV overlaps app source tracks')
    for actual, expected, label in ((len(training_rows), expected_training_rows, 'training rows'),
                                   (len({r['uuid'] for r in training_rows}), expected_training_actors, 'training actors')):
        if expected is not None and actual != expected:
            raise ValueError(f'Expected {expected} {label}, found {actual}')

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.studio-csvs-', dir=output_dir.parent))
    try:
        for name, data in derived.items():
            with (staging / name).open('wb') as handle:
                handle.write(data)
                handle.flush()
                os.fsync(handle.fileno())
        corpus = load_corpus(staging, blackbands_csv)
        actors = {row['uuid'] for row in corpus.actors}
        if expected_actors is not None and len(actors) != expected_actors:
            raise ValueError(f'Expected {expected_actors} bank actors, found {len(actors)}')
        if {row['uuid'] for row in training_rows} - actors:
            raise ValueError('Current training actors are missing from the studio bank roster')
        source_tracks = {(r['recording_id'], r['speaker']) for r in corpus.tracks}
        if source_tracks - studio_tracks or source_tracks & app_tracks:
            raise ValueError('Derived load_corpus source tracks are not exclusively studio')
        manifest = {
            'schema_version': 1, 'scope': 'studio-only', 'complete': True,
            'archive_csv_root': str(archive_root), 'csv_root': str(output_dir),
            'policy': 'Exact all-studio.csv recording/speaker/chunk membership and actor/session/path identity; '
                      'app CSVs empty; source-track disjointness checked independently of shared actors',
            'preparation_code_sha256': sha256(Path(__file__).read_bytes()), 'files': records,
            'blackbands': {'path': str(blackbands_csv), 'sha256': sha256(blackbands_csv.read_bytes()),
                           'policy': 'Archived exclusions preserved unchanged'},
            'authoritative_studio': counts(studio_rows),
            'shared_studio_app_actors': len({r['uuid'] for r in studio_rows} & {r['uuid'] for r in app_rows}),
            'studio_app_track_overlap': 0,
            'corpus': {'actors': len(actors), 'actors_by_cohort': dict(Counter(r['cohort'] for r in corpus.actors)),
                       'source_tracks': len(corpus.tracks), 'all_clips': len(corpus.all_clips),
                       'actors_without_eligible_tracks': [r['uuid'] for r in corpus.actors if not r['eligible_track_count']],
                       'app_source_tracks': 0},
            'beatles': {'rows': len(beatles), 'bytes_and_order_preserved': True,
                        'held_row_indices': list(BEATLES_HELD_ROWS),
                        'held_identities': [dict(zip(IDENTITY, identity(beatles[i]))) for i in BEATLES_HELD_ROWS]},
            'training': {'path': str(training_csv), 'sha256': sha256(training_bytes), **counts(training_rows),
                         'non_studio_chunks': 0, 'app_source_tracks': 0, 'actors_outside_bank_roster': 0},
        }
        # Detect concurrent archive changes before publishing a completion manifest.
        for name, (original, _, _) in tables.items():
            if (archive_root / name).read_bytes() != original:
                raise ValueError(f'Archived input changed during preparation: {name}')
        if training_csv.read_bytes() != training_bytes:
            raise ValueError('Training CSV changed during preparation')
        manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + '\n').encode()
        with (staging / 'studio-scope.json').open('wb') as handle:
            handle.write(manifest_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        if output_dir.exists():
            if output_dir.is_symlink() or any((output_dir / name).is_symlink() for name in derived):
                raise ValueError('Existing scope contains symlinks')
            expected = dict(derived, **{'studio-scope.json': manifest_bytes})
            if any(not (output_dir / name).is_file() or (output_dir / name).read_bytes() != data
                   for name, data in expected.items()):
                raise ValueError('Existing scope differs; choose a new output directory, preserving prior provenance')
        else:
            staging.rename(output_dir)
            fd = os.open(output_dir.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive-csv-root', type=Path, default=CSV_ROOT)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'studio-only' / 'csvs')
    parser.add_argument('--blackbands-csv', type=Path, default=BLACKBANDS_CSV)
    parser.add_argument('--training-csv', type=Path, default=TRAINING_CSV)
    parser.add_argument('--expected-actors', type=int, default=543)
    parser.add_argument('--expected-training-rows', type=int, default=14414)
    parser.add_argument('--expected-training-actors', type=int, default=458)
    args = parser.parse_args()
    report = prepare(args.archive_csv_root, args.output_dir, args.blackbands_csv, args.training_csv,
                     expected_actors=args.expected_actors, expected_training_rows=args.expected_training_rows,
                     expected_training_actors=args.expected_training_actors)
    print(json.dumps({'csv_root': report['csv_root'], 'corpus': report['corpus'],
                      'training': report['training'], 'complete': True}), flush=True)


if __name__ == '__main__':
    main()
