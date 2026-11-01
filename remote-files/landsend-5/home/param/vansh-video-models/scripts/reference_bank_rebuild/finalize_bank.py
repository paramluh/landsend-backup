"""Merge fresh selections and verify the complete actor bank before publishing its manifest."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import shutil

import numpy as np
import pandas as pd

from encode_bank import (ENCODER_ROOT, VAE_SHA256, encoder_fingerprint, encoder_source_files,
                         load_encoder_module, selection_groups, sha256)
from upstream import (BEATLES_HELD_ROWS, BLACKBANDS_CSV, CSV_ROOT, REFERENCE_PREFIX, SELECTOR_FOLDER,
                      SLOTS, in_ranges, load_corpus, read_csv, timestamp_allowed)

HF_PREFIX = 'hf://buckets/InternalCan/aloe-conversationa-data/'
ARCHIVE = Path('/mnt/paramfs/vansh/richmond-7/reference-bank')
SCAN_SUMMARIES = ('targets-summary.json', 'scan-summary.json', 'selection-summary.json', 'decode-summary.json')
VALIDATION_MARKER = 'artifact-validation.json'


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def validate_selection(selection):
    required = {'recording_id', 'speaker', 'session_id', 'frame_index', 'time_s', 'cx0', 'cy0', 'cw', 'ch'}
    if not required <= set(selection):
        raise ValueError(f'Missing selection columns: {sorted(required - set(selection))}')
    groups = selection_groups(selection)
    if selection.iid.tolist() != list(range(len(selection))):
        raise ValueError('iid must follow globally contiguous selection row order')
    for field in ('iid', 'index', 'property', 'speaker', 'frame_index', 'cx0', 'cy0', 'cw', 'ch'):
        values = pd.to_numeric(selection[field], errors='raise').to_numpy(dtype=float)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise ValueError(f'Noninteger or missing {field}')
    if not selection.uuid.astype(str).str.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}').all():
        raise ValueError('Invalid actor UUID')
    if selection.frame_index.lt(0).any() or not np.isfinite(selection.time_s).all():
        raise ValueError('Invalid frame clock')
    if not np.allclose(selection.time_s, selection.frame_index / 25, rtol=0, atol=1e-9):
        raise ValueError('time_s must preserve the exact 25 fps frame clock')
    if selection.duplicated(['bank', 'uuid', 'property', 'recording_id', 'speaker', 'frame_index']).any():
        raise ValueError('Duplicate reference frame within an actor/property')
    scanned = selection[selection.origin == 'video-scan']
    if scanned.duplicated(['bank', 'uuid', 'recording_id', 'speaker', 'frame_index']).any():
        raise ValueError('A scanned frame was assigned to multiple properties')
    for row in selection.to_dict('records'):
        source = row['source']
        if not isinstance(source, str) or not source.startswith(HF_PREFIX):
            raise ValueError(f'Non-HF source for iid={row["iid"]}: {source}')
        key = source[len(HF_PREFIX):].split('#', 1)[0]
        if not key or Path(key).is_absolute() or '..' in Path(key).parts:
            raise ValueError(f'Invalid HF object key: {source}')
        if row['origin'] == 'ref-image':
            expected = f'{REFERENCE_PREFIX}/{row["recording_id"]}/speaker_{int(row["speaker"])}/{SELECTOR_FOLDER}/{row["slot"]}.png'
            if key != expected or '#' in source or not re.fullmatch('[0-9a-f]{64}', str(row.get('source_sha256'))):
                raise ValueError(f'Upstream source identity/hash mismatch: iid={row["iid"]}')
            if isinstance(row.get('image_path'), str) and row['image_path']:
                raise ValueError('Upstream image_path override would bypass the verified HF image')
        elif row['origin'] == 'video-scan':
            if not source.endswith(f'.mp4#frame={int(row["frame_index"])}') or not row.get('image_path'):
                raise ValueError(f'Scanned source/frame/path mismatch: iid={row["iid"]}')
            if not re.fullmatch('[0-9a-f]{64}', str(row.get('decoded_image_sha256'))):
                raise ValueError(f'Missing decoded PNG checksum: iid={row["iid"]}')
        else:
            raise ValueError(f'Unrecognized origin: {row["origin"]}')
        x, y, width, height = (int(row[field]) for field in ('cx0', 'cy0', 'cw', 'ch'))
        if int(row['property']) >= 5:
            if (x, y, width, height) != (0, 0, 512, 960):
                raise ValueError(f'Full-frame property has a face crop: iid={row["iid"]}')
        elif not (256 <= width <= 282 and height == round(width * 480 / 256)
                  and 0 <= x <= 512 - width and 0 <= y <= 960 - height):
            raise ValueError(f'Invalid face crop: iid={row["iid"]}')
    return groups


def merge_selections(upstream, video, beatles):
    frames = []
    for name, frame, bank, origin in (('upstream', upstream, 'bank', 'ref-image'),
                                     ('video', video, 'bank', 'video-scan'),
                                     ('beatles', beatles, 'beatles-eval', 'ref-image')):
        if frame.empty:
            continue
        frame = frame.copy()
        if not frame.bank.eq(bank).all() or not frame.origin.eq(origin).all():
            raise ValueError(f'Incorrect bank/origin in {name} selection')
        frame['candidate_iid'] = frame['iid'] if 'iid' in frame else pd.NA
        frame['selection_stage'] = name
        if frame.source.map(lambda source: not isinstance(source, str) or ('://' in source and not source.startswith(HF_PREFIX))).any():
            raise ValueError(f'Non-HF source in {name}')
        frame['source'] = frame.source.map(lambda source: source if source.startswith(HF_PREFIX) else HF_PREFIX + source)
        if name == 'video':
            frame['decoded_image_sha256'] = frame.image_path.map(sha256)
        frames.append(frame)
    if not frames:
        raise ValueError('All three selections are empty')
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.sort_values(['bank', 'uuid', 'property', 'index'], kind='stable').reset_index(drop=True)
    merged['iid'] = np.arange(len(merged), dtype=np.int64)
    for _, rows in merged.groupby(['bank', 'uuid', 'property'], sort=False):
        origins = rows.origin.tolist()
        if 'video-scan' in origins and 'ref-image' in origins[origins.index('video-scan'):]:
            raise ValueError('Scanned additions must follow all upstream references')
    validate_selection(merged)
    return merged


def validate_sources(selection, source_root, corpus, beatles_rows, holdout):
    tracks = {row['track']: row for row in corpus.tracks}
    beatles_tracks = {row['track']: row for row in beatles_rows}
    held = set(holdout['held_recordings'])
    rows, sources, errors = selection.to_dict('records'), [], []
    for row in rows:
        if row['origin'] == 'ref-image':
            path = Path(source_root) / row['source'][len(HF_PREFIX):]
            expected = row['source_sha256']
        else:
            path, expected = Path(row['image_path']), row['decoded_image_sha256']
        sources.append((path, expected))

    def digest(path):
        return sha256(path) if path.is_file() else None

    # Overlap paramfs reads, hashing each distinct path once in this call. Read
    # results in selection order so diagnostics and unreadable-file exceptions
    # retain their serial ordering regardless of worker completion order.
    with ThreadPoolExecutor(max_workers=64) as pool:
        hashes = {path: pool.submit(digest, path) for path in dict.fromkeys(path for path, _ in sources)}
        for row, (path, expected) in zip(rows, sources):
            track = f'{row["recording_id"]}/speaker_{int(row["speaker"])}'
            identity = (tracks if row['bank'] == 'bank' else beatles_tracks).get(track)
            if identity is None or identity['uuid'] != row['uuid'] or identity['session_id'] != row['session_id']:
                errors.append(f'iid={row["iid"]}: source identity differs from corpus CSV')
            elif row['bank'] == 'bank':
                if not timestamp_allowed(corpus, track, row['time_s'], identity['cohort']):
                    errors.append(f'iid={row["iid"]}: prohibited reference time')
            elif row['recording_id'] in held or any(in_ranges(row['time_s'], track, ranges)
                                                   for ranges in (corpus.ranges_blackbands, corpus.ranges_lighting)):
                errors.append(f'iid={row["iid"]}: prohibited Beatles donor')
            if hashes[path].result() != expected:
                errors.append(f'iid={row["iid"]}: source image absent or checksum mismatch: {path}')
    return {'checked_images': len(hashes), 'errors': errors}


def validate_scan_sources(selection, scan_dir, corpus, video_index):
    """Bind decoded references to the corpus track's resolved HF video."""
    from hf_sources import resolve_video

    scanned = selection[selection.origin.eq('video-scan')]
    if scanned.empty:
        return {'checked_rows': 0, 'errors': []}
    scan_dir, video_index = Path(scan_dir), Path(video_index)
    targets = json.loads((scan_dir / 'targets-summary.json').read_text())
    if targets.get('video_index_sha256') != sha256(video_index):
        raise ValueError('HF video index changed after target planning')
    if targets.get('corpus_source_hashes') != corpus.source_hashes:
        raise ValueError('Video targets used a different corpus')
    plan_hashes = {name: sha256(scan_dir / name) for name in ('tracks.json', 'targets-summary.json')}
    scan = json.loads((scan_dir / 'scan-summary.json').read_text())
    if scan.get('plan_sha256') != plan_hashes:
        raise ValueError('Video scan used a different source plan')
    index = json.loads(video_index.read_text())
    planned = json.loads((scan_dir / 'tracks.json').read_text())
    tracks = {row['track']: row for row in planned}
    if len(tracks) != len(planned):
        raise ValueError('Duplicate tracks in video source plan')
    identities = {row['track']: row for row in corpus.tracks}
    errors = []
    for track_name, rows in scanned.groupby(['recording_id', 'speaker'], sort=False):
        recording, speaker = track_name
        track = f'{recording}/speaker_{int(speaker)}'
        try:
            planned_track, identity = tracks.get(track), identities.get(track)
            if planned_track is None or identity is None:
                raise ValueError('Track absent from video plan or corpus')
            for field in ('uuid', 'session_id', 'recording_id', 'speaker'):
                expected = identity[field]
                if planned_track.get(field) != expected or not rows[field].eq(expected).all():
                    raise ValueError(f'Video plan/selection identity mismatch: {field}')
            if not rows.bank.eq(planned_track.get('bank', 'bank')).all():
                raise ValueError('Video plan/selection bank mismatch')
            video = resolve_video(recording, speaker, index, original_blob=identity.get('original_blob'))
            expected_source = HF_PREFIX + video['path']
            if (planned_track.get('source') != expected_source
                    or planned_track.get('video_source_metadata') != video):
                raise ValueError('Video plan differs from resolved HF object')
            for row in rows.to_dict('records'):
                if row['source'] != f'{expected_source}#frame={int(row["frame_index"])}':
                    raise ValueError(f'Selected HF video belongs to a different track: iid={row["iid"]}')
        except ValueError as error:
            errors.append(f'{track}: {error}')
    return {'checked_rows': len(scanned), 'plan_sha256': plan_hashes,
            'video_index_sha256': sha256(video_index), 'errors': errors}


def validate_decode_qa(output_dir, selection_hash, fingerprint, selection):
    path = Path(output_dir) / 'qa/decode-report.json'
    try:
        report = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError('Required decode QA report is missing or unreadable') from error
    if (report.get('passed') is not True or report.get('selection_sha256') != selection_hash
            or report.get('vae_sha256') != VAE_SHA256 or report.get('encoder_fingerprint') != fingerprint):
        raise ValueError('Decode QA must pass for this exact selection, VAE, and encoder implementation')
    if not report.get('images') or report['images'] != len(report.get('samples', [])):
        raise ValueError('Decode QA report has no valid sample evidence')
    selected = {row['iid']: row for row in selection.to_dict('records')}
    hashes, seen = {}, set()
    for sample in report['samples']:
        row = selected.get(sample.get('iid'))
        if (row is None or sample['iid'] in seen
                or any(sample.get(key) != row[key] for key in ('bank', 'uuid', 'property', 'index', 'origin'))):
            raise ValueError('Decode QA sample identity differs from the final selection')
        seen.add(sample['iid'])
        path = Path(output_dir) / row['bank'] / row['uuid'] / f'property_{int(row["property"])}/latents.safetensors'
        if path not in hashes:
            hashes[path] = sha256(path)
        if sample.get('latent_sha256') != hashes[path]:
            raise ValueError('Decode QA is stale for a sampled latent file')
    return report


def validate_latents(selection, output_dir, selection_hash, fingerprint=None):
    import torch
    from safetensors import SafetensorError, safe_open

    torch.set_num_threads(1)
    fingerprint = encoder_fingerprint() if fingerprint is None else fingerprint

    expected, hashes, errors, byte_counts = set(), {}, [], defaultdict(int)
    groups = selection_groups(selection)
    for bank, actor, prop, rows in groups:
        relative = f'{bank}/{actor}/property_{prop}/latents.safetensors'
        expected.add(relative)
        path = Path(output_dir) / relative
        try:
            digest = sha256(path)
            hashes[relative] = digest
            byte_counts[(bank, actor)] += path.stat().st_size
            with safe_open(path, framework='pt', device='cpu') as handle:
                if list(handle.keys()) != ['video']:
                    raise ValueError('Expected exactly the video tensor')
                tensor = handle.get_tensor('video')
                if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != (len(rows), 48, 30, 16):
                    raise ValueError(f'Wrong dtype/shape: {tensor.dtype}, {tuple(tensor.shape)}')
                if not torch.isfinite(tensor).all().item():
                    raise ValueError('Nonfinite latent values')
                metadata = {'uuid': actor, 'property': str(prop), 'slot': SLOTS[prop - 1], 'n': str(len(rows))}
                if handle.metadata() != metadata:
                    raise ValueError(f'Incorrect tensor metadata: {handle.metadata()}')
            receipt_relative = f'encoding-checkpoints/{bank}__{actor}__{prop}.json'
            receipt_path = Path(output_dir) / receipt_relative
            receipt = json.loads(receipt_path.read_text())
            required = {'sha256': digest, 'selection_sha256': selection_hash, 'vae_sha256': VAE_SHA256,
                        'encoder_fingerprint': fingerprint, 'n': len(rows)}
            if any(receipt.get(key) != value for key, value in required.items()):
                raise ValueError('Encoding receipt disagrees with final selection, VAE, implementation, or tensor')
            hashes[receipt_relative] = sha256(receipt_path)
        except (OSError, ValueError, RuntimeError, SafetensorError) as error:
            errors.append({'path': relative, 'error': str(error)})
    actual = {str(path.relative_to(output_dir)) for bank in ('bank', 'beatles-eval')
              for path in (Path(output_dir) / bank).rglob('*.safetensors')}
    for relative in sorted(actual - expected):
        errors.append({'path': relative, 'error': 'Unselected stale latent file'})
    return {'groups': len(groups), 'checked_files': len(expected & actual), 'errors': errors}, hashes, byte_counts


def completion_evidence(source_diagnostics, scan_dir):
    reasons = []
    if source_diagnostics.get('fatal_input_issues'):
        reasons.append('Upstream input diagnostics contain fatal issues')
    if source_diagnostics.get('incomplete_source_tracks') or source_diagnostics.get('issue_counts', {}).get('confirmed_hf_missing'):
        reasons.append('HF upstream source evidence is incomplete')
    summaries = {}
    for name in SCAN_SUMMARIES:
        path = Path(scan_dir) / name
        summaries[name] = json.loads(path.read_text()) if path.is_file() else None
        if summaries[name] is None or summaries[name].get('complete') is not True:
            reasons.append(f'Video stage incomplete: {name}')
    for name, fields in {'targets-summary.json': ('missing_tracks', 'missing_actors'),
                         'scan-summary.json': ('missing_sources', 'failures'),
                         'selection-summary.json': ('missing_chunks',)}.items():
        if summaries[name] and any(summaries[name].get(field, 0) for field in fields):
            reasons.append(f'Video stage reports missing inputs/failures: {name}')
    targets = summaries['targets-summary.json'] or {}
    scanned = summaries['scan-summary.json'] or {}
    selected = summaries['selection-summary.json'] or {}
    decoded = summaries['decode-summary.json'] or {}
    source_complete = (not source_diagnostics.get('fatal_input_issues')
                       and not source_diagnostics.get('incomplete_source_tracks')
                       and not source_diagnostics.get('issue_counts', {}).get('confirmed_hf_missing')
                       and targets.get('complete') is True
                       and not targets.get('missing_tracks', 0) and not targets.get('missing_actors', 0))
    available_complete = (not source_diagnostics.get('fatal_input_issues') and all(summaries.values())
                          and isinstance(targets.get('complete'), bool)
                          and scanned.get('available_targets_complete') is True
                          and scanned.get('missing_sources') == 0 and scanned.get('failures') == 0
                          and selected.get('missing_chunks') == 0 and 'rows' in selected
                          and decoded.get('rows') == selected['rows']
                          and all(summary.get('complete') is targets['complete']
                                  for summary in (scanned, selected, decoded)))
    if not available_complete:
        reasons.append('Available-source processing is not complete')
    return {'complete': not reasons, 'source_complete': bool(source_complete),
            'available_work_complete': bool(available_complete), 'reasons': reasons,
            'source_diagnostics': source_diagnostics, 'scan': summaries}


def publish_validation_marker(output_dir, report):
    """Publish last; the marker itself is outside the manifest to avoid a hash cycle."""
    output_dir = Path(output_dir)
    marker = {'schema_version': 1, 'artifact_valid': bool(report['artifact_valid']),
              'source_complete': bool(report['source_complete']),
              'available_work_complete': bool(report['available_work_complete']),
              'selection_sha256': report['selection_sha256'], 'vae_sha256': report['vae_sha256'],
              'encoder_fingerprint_sha256': report['encoder_fingerprint']['sha256'],
              'coverage_sha256': sha256(output_dir / 'coverage.json'),
              'manifest_sha256': sha256(output_dir / 'SHA256SUMS')}
    path = output_dir / VALIDATION_MARKER
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as handle:
        json.dump(marker, handle, indent=2, allow_nan=False)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(output_dir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return marker


def coverage(selection, actors, beatles_rows, byte_counts):
    roster = [('bank', row['uuid'], row['cohort'], int(row['eligible_track_count'])) for row in actors]
    roster += [('beatles-eval', actor, 'beatles-eval', None) for actor in sorted({row['uuid'] for row in beatles_rows})]
    counts = Counter(zip(selection.bank, selection.uuid, selection.property))
    origins = Counter(zip(selection.bank, selection.uuid, selection.origin))
    records = []
    for bank, actor, cohort, tracks in roster:
        values = [counts[(bank, actor, prop)] for prop in range(1, 10)]
        records.append({'bank': bank, 'uuid': actor, 'cohort': cohort, 'eligible_track_count': tracks,
                        **{f'property_{prop}': value for prop, value in enumerate(values, 1)},
                        'images': sum(values), 'nonempty_properties': sum(value > 0 for value in values),
                        'upstream_images': origins[(bank, actor, 'ref-image')],
                        'scanned_images': origins[(bank, actor, 'video-scan')], 'latent_bytes': byte_counts[(bank, actor)]})
    frame = pd.DataFrame(records)
    summaries = {}
    for bank, part in frame.groupby('bank', sort=True):
        summaries[bank] = {'actors': len(part), 'nonempty_actors': int(part.images.gt(0).sum()),
                           'actors_with_all_9': int(part.nonempty_properties.eq(9).sum()),
                           'latent_groups': int(part.nonempty_properties.sum()),
                           'zero_actors': part.loc[part.images.eq(0), 'uuid'].tolist(),
                           'images': int(part.images.sum()), 'latent_bytes': int(part.latent_bytes.sum()),
                           'images_per_actor_histogram': dict(sorted(Counter(map(int, part.images)).items())),
                           'nonempty_properties_histogram': dict(sorted(Counter(map(int, part.nonempty_properties)).items())),
                           'per_property': {str(prop): {'images': int(part[f'property_{prop}'].sum()),
                               'counts_histogram': dict(sorted(Counter(map(int, part[f'property_{prop}'])).items()))}
                               for prop in range(1, 10)}}
    return frame, summaries


def copy_provenance(output_dir, documents, evidence, encoder_sources=None, fingerprint=None):
    files, records = [], []
    inputs = [('documents', path) for path in documents]
    inputs += [('code', path) for path in sorted(Path(__file__).parent.glob('*.py'))]
    inputs += [('evidence', path) for path in evidence if path.is_file()]
    inputs += [(str(Path('archive') / relative.parent), ARCHIVE / relative) for relative in map(Path, (
        'code/full/plan_v3.py', 'code/full/select_v3.py', 'code/studio-actors/encode.py',
        'code/s2bundle/vscan_node.py', 'code/s2bundle/select_s2.py', 'code/s2bundle/encode_v2_node.py',
        'code/node/mvenv_req.txt', 'code/node/ref_audit/audit.py', 'code/node/ref_audit/audit59.py',
        'code/node/mouth_open_pilot/teeth.py', 'code/node/mouth_open_pilot/allteeth.py',
        'code/node/mouth_open_pilot/scan.py',
    ))]
    inputs.append(('archive/encoder', ARCHIVE.parent / 'reference-latents/encode/refs_lib.py'))
    encoder_inputs = {}
    for name, path in (encoder_sources or {}).items():
        kind = str(Path('encoder') / Path(name).parent)
        inputs.append((kind, path))
        encoder_inputs[Path('provenance') / kind / path.name] = fingerprint['inputs']['files'][name]
    names = set()
    for kind, source in inputs:
        relative = Path('provenance') / kind / source.name
        if relative in names:
            raise ValueError(f'Conflicting provenance filename: {relative}')
        names.add(relative)
        destination = Path(output_dir) / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        copied_hash = sha256(destination)
        if relative in encoder_inputs and encoder_inputs[relative] != copied_hash:
            raise ValueError(f'Encoder implementation changed during finalization: {source}')
        files.append(relative)
        records.append({'original_path': str(source), 'copied_path': str(relative), 'sha256': copied_hash})
    relative = Path('provenance/files.json')
    write_json(Path(output_dir) / relative, records)
    return files + [relative]


def finalize(args):
    (args.output_dir / VALIDATION_MARKER).unlink(missing_ok=True)
    selection = pd.read_parquet(args.selection)
    validate_selection(selection)
    selection_hash = sha256(args.selection)
    encoder_module = load_encoder_module(getattr(args, 'encoder_root', ENCODER_ROOT))
    fingerprint = encoder_fingerprint(module=encoder_module)
    decode_qa = validate_decode_qa(args.output_dir, selection_hash, fingerprint, selection)
    actors = read_csv(args.actors, ('uuid', 'cohort', 'eligible_track_count'))
    corpus = load_corpus(args.csv_root, args.blackbands_csv)
    roster = {row['uuid']: (row['cohort'], int(row['eligible_track_count'])) for row in actors}
    expected_roster = {row['uuid']: (row['cohort'], row['eligible_track_count']) for row in corpus.actors}
    if len(actors) != args.expected_actors or roster != expected_roster:
        raise ValueError('Actor roster differs from the entire requested corpus')
    beatles_rows = read_csv(args.csv_root / 'aloe-eval-beatles.csv', ('uuid', 'session_id', 'recording_id', 'speaker', 'chunk'))
    if set(selection.loc[selection.bank.eq('bank'), 'uuid']) - {row['uuid'] for row in actors}:
        raise ValueError('Selected actor outside the main bank roster')
    holdout = json.loads(args.beatles_holdout.read_text())
    if holdout['held_rows'] != list(BEATLES_HELD_ROWS) or set(holdout['held_recordings']) != {
        beatles_rows[index]['recording_id'] for index in BEATLES_HELD_ROWS
    }:
        raise ValueError('Beatles holdout differs from the documented fixed 16 rows')
    diagnostics = json.loads(args.source_diagnostics.read_text())
    if diagnostics['source_hashes'] != corpus.source_hashes:
        raise ValueError('Corpus CSVs changed after planning')
    source_report = validate_sources(selection, args.source_root, corpus, beatles_rows, holdout)
    video_index = getattr(args, 'video_index', None) or args.scan_dir.parent / 'hf-video-index.json'
    source_report['scan'] = validate_scan_sources(selection, args.scan_dir, corpus, video_index)
    source_report['errors'].extend(source_report['scan']['errors'])
    latent_report, hashes, byte_counts = validate_latents(selection, args.output_dir, selection_hash, fingerprint)
    evidence = completion_evidence(diagnostics, args.scan_dir)
    merge_path = args.selection.parent / 'selection-merge.json'
    merge = json.loads(merge_path.read_text())
    if merge['selection_sha256'] != sha256(args.selection):
        raise ValueError('Merge receipt does not identify this selection')
    targets = evidence['scan']['targets-summary.json']
    if targets and targets['upstream_sha256'] != merge['inputs']['upstream']['sha256']:
        raise ValueError('Video target planning used a different upstream selection')
    decoded = evidence['scan']['decode-summary.json']
    if decoded and decoded['rows'] != int(selection.origin.eq('video-scan').sum()):
        raise ValueError('Merged video row count differs from completed decode output')
    if holdout['selected_images'] != int(selection.bank.eq('beatles-eval').sum()):
        raise ValueError('Merged Beatles row count differs from selection receipt')
    if source_report['errors']:
        evidence['reasons'].append('Selected source verification failed')
    if latent_report['errors']:
        evidence['reasons'].append('Latent verification failed')
    evidence['complete'] = not evidence['reasons']
    evidence['artifact_valid'] = (evidence['available_work_complete']
                                  and not source_report['errors'] and not latent_report['errors'])
    actor_counts, summary = coverage(selection, actors, beatles_rows, byte_counts)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selection.to_csv(args.output_dir / 'bank.csv', index=False)
    actor_counts.to_csv(args.output_dir / 'actor-coverage.csv', index=False)
    # Preserve the archived consumer-facing layout independently of build metadata.
    for bank in ('bank', 'beatles-eval'):
        folder = args.output_dir / bank
        folder.mkdir(exist_ok=True)
        selection[selection.bank.eq(bank)].to_csv(folder / 'bank.csv', index=False)
        hashes[f'{bank}/bank.csv'] = sha256(folder / 'bank.csv')
        subset_hashes = {relative[len(bank) + 1:]: digest for relative, digest in hashes.items()
                         if relative.startswith(bank + '/')}
        (folder / 'SHA256SUMS').write_text(''.join(f'{digest}  ./{relative}\n'
                                                  for relative, digest in sorted(subset_hashes.items())))
        hashes[f'{bank}/SHA256SUMS'] = sha256(folder / 'SHA256SUMS')
        summary[bank]['folder_bytes'] = sum(path.stat().st_size for path in folder.rglob('*') if path.is_file())
    report = {**evidence, 'selection_sha256': selection_hash, 'vae_sha256': VAE_SHA256, 'decode_qa': decode_qa,
              'encoder_fingerprint': fingerprint,
              'tensor_contract': {'key': 'video', 'dtype': 'BF16', 'shape': ['n', 48, 30, 16], 'max_n': 20,
                                  'normalization': 'Wan2.2 latent mean/std folded into quant-conv mu weights/bias'},
              'sources': source_report, 'latents': latent_report, 'coverage': summary,
              'bank_csv_float_policy': 'Default lossless Python float representation; no decimal rounding'}
    write_json(args.output_dir / 'coverage.json', report)
    documents = args.document or [ARCHIVE / 'RECIPE.md', ARCHIVE / 'sinks.md']
    copied = copy_provenance(args.output_dir, documents, [args.actors, args.source_diagnostics, args.beatles_holdout, merge_path,
                             args.output_dir / 'qa/decode-report.json',
                             args.scan_dir / 'tracks.json', video_index,
                             *[args.scan_dir / name for name in SCAN_SUMMARIES]],
                             encoder_source_files(encoder_module), fingerprint)
    for path in ['bank.csv', 'actor-coverage.csv', 'coverage.json', *copied]:
        hashes[str(path)] = sha256(args.output_dir / path)
    selection_copy = args.output_dir / 'selection.parquet'
    if args.selection.resolve() != selection_copy.resolve():
        shutil.copyfile(args.selection, selection_copy)
    hashes['selection.parquet'] = sha256(selection_copy)
    manifest = ''.join(f'{digest}  {relative}\n' for relative, digest in sorted(hashes.items()))
    (args.output_dir / 'SHA256SUMS').write_text(manifest)
    publish_validation_marker(args.output_dir, report)
    print(json.dumps({'complete': report['complete'], 'reasons': report['reasons'], 'coverage': summary,
                      'manifest_entries': len(hashes)}), flush=True)
    return report['complete']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    merge = commands.add_parser('merge')
    for name in ('upstream', 'video', 'beatles', 'output-dir'):
        merge.add_argument(f'--{name}', type=Path, required=True)
    finish = commands.add_parser('finalize')
    for name in ('selection', 'actors', 'source-root', 'source-diagnostics', 'scan-dir', 'beatles-holdout', 'output-dir'):
        finish.add_argument(f'--{name}', type=Path, required=True)
    finish.add_argument('--csv-root', type=Path, default=CSV_ROOT)
    finish.add_argument('--blackbands-csv', type=Path, default=BLACKBANDS_CSV)
    finish.add_argument('--expected-actors', type=int, default=1759)
    finish.add_argument('--document', type=Path, action='append')
    finish.add_argument('--video-index', type=Path,
                        help='HF video index; defaults to SCAN_DIR/../hf-video-index.json')
    finish.add_argument('--encoder-root', type=Path, default=ENCODER_ROOT)
    args = parser.parse_args()
    if args.command == 'merge':
        paths = {name: getattr(args, name) for name in ('upstream', 'video', 'beatles')}
        selected = merge_selections(*(pd.read_parquet(path) for path in paths.values()))
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / VALIDATION_MARKER).unlink(missing_ok=True)
        selected.to_parquet(args.output_dir / 'selection.parquet', index=False)
        selected.to_csv(args.output_dir / 'bank.csv', index=False)
        write_json(args.output_dir / 'selection-merge.json', {
            'inputs': {name: {'path': str(path), 'sha256': sha256(path)} for name, path in paths.items()},
            'rows': len(selected), 'groups': len(selection_groups(selected)),
            'selection_sha256': sha256(args.output_dir / 'selection.parquet'),
            'row_order': 'bank, uuid, property, index; upstream images before scanned additions; iid=global row index',
        })
        print(json.dumps({'rows': len(selected), 'selection': str(args.output_dir / 'selection.parquet')}), flush=True)
    elif not finalize(args):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
