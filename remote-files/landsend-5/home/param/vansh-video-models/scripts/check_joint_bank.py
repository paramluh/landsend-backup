"""CPU smoke of the finalized studio bank through the actual joint dataset.

Run only after artifact-validation.json exists. This launches no workers/GPUs,
changes no bank/source data, and writes one review receipt at --output.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import tomllib


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def validate_historical_policy(bank, repo):
    """Require the archived studio policy and its manifest-bound merge lineage."""
    files = {line.split('  ', 1)[1] for line in (bank / 'SHA256SUMS').read_text().splitlines()}

    def bound(relative):
        if relative not in files:
            raise ValueError(f'Historical policy evidence is outside the validated manifest: {relative}')
        return bank / relative

    policy = json.loads(bound('provenance/documents/selection-policy.json').read_text())
    if policy.get('policy_id') != 'historical-studio-v1-filters-and-ranking-v1':
        raise ValueError('Bank does not use the intended historical studio selection policy')
    archive_hashes = {
        'archived-studio-select.py': 'bbcbd57c1b5edebe7bc760f8452e12f782a7a6d88b9b8f66cf72c41e0a323be6',
        'archived-beatles-plan.py': '58d68913aaae742663e8fd24213197aff2ac17416716eefb90ddcea2ddeb2b17',
    }
    for name, expected in archive_hashes.items():
        if sha256(bound('provenance/documents/' + name)) != expected:
            raise ValueError(f'Historical policy archive differs: {name}')
    if (policy.get('archived_selector_sha256') != archive_hashes['archived-studio-select.py']
            or policy.get('beatles_policy', {}).get('archived_beatles_selector_sha256') != archive_hashes['archived-beatles-plan.py']
            or policy.get('wrapper_sha256') != sha256(repo / 'scripts/reference_bank_rebuild/historical_studio_select.py')
            or policy.get('scope_manifest', {}).get('sha256') != sha256(bound('provenance/documents/studio-scope.json'))):
        raise ValueError('Historical image policy code or studio scope differs')
    merge = json.loads(bound('provenance/evidence/selection-merge.json').read_text())
    for stage, name in (('upstream', 'upstream-selection.parquet'), ('beatles', 'beatles-selection.parquet')):
        if policy.get('outputs', {}).get(name) != merge['inputs'][stage]['sha256']:
            raise ValueError(f'Final bank did not merge the historical {stage} selection')
    if merge['selection_sha256'] != sha256(bank / 'selection.parquet'):
        raise ValueError('Historical merge does not identify the final bank selection')
    video = json.loads(bound('provenance/documents/historical-video-policy.json').read_text())
    video_archive_sha256 = 'dff8511c0d748da675c25e076a5050def6c3ec5d5b00032ab37766dcae9b2d61'
    if (video.get('policy_id') != policy['policy_id']
            or video.get('archived_selector_sha256') != video_archive_sha256
            or sha256(bound('provenance/documents/archived-studio-select-v2.py')) != video_archive_sha256
            or video.get('wrapper_sha256') != sha256(repo / 'scripts/reference_bank_rebuild/historical_video_select.py')
            or video.get('upstream_sha256') != merge['inputs']['upstream']['sha256']):
        raise ValueError('Historical video selection policy or upstream input differs')
    decoded = json.loads(bound('provenance/documents/historical-video-decode.json').read_text())
    if (decoded.get('passed') is not True or decoded.get('policy_id') != policy['policy_id']
            or decoded.get('policy_sha256') != sha256(bound('provenance/documents/historical-video-policy.json'))
            or decoded.get('selected_sha256') != video.get('selected_sha256')
            or decoded.get('rows') != video.get('summary', {}).get('rows')
            or decoded.get('decoded_sha256') != merge['inputs']['video']['sha256']):
        raise ValueError('Historical video selection is not bound to the decoded bank inputs')
    return {'policy_id': policy['policy_id'], 'selection_policy_sha256': sha256(bound('provenance/documents/selection-policy.json')),
            'historical_video_policy_sha256': sha256(bound('provenance/documents/historical-video-policy.json')),
            'historical_video_decode_sha256': sha256(bound('provenance/documents/historical-video-decode.json')),
            'archived_source_sha256': {**archive_hashes, 'archived-studio-select-v2.py': video_archive_sha256}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-clips', type=int, default=14414)
    parser.add_argument('--expected-training-actors', type=int, default=458)
    parser.add_argument('--expected-bank-actors', type=int, default=543)
    parser.add_argument('--ranks', type=int, nargs='+', default=[0, 39])
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ.setdefault('OMP_NUM_THREADS', '4')
    sys.path.insert(0, str(args.repo.resolve()))
    import torch
    from safetensors import safe_open
    from distillation.joint_data import build_joint_data
    from distillation.joint_runtime import validate_reference_bank
    torch.set_num_threads(4)
    if any(rank < 0 or rank >= 40 for rank in args.ranks) or len(set(args.ranks)) != len(args.ranks):
        parser.error('--ranks must be distinct ranks from the production world size of 40')
    config = tomllib.loads(args.config.read_text())
    started = time.monotonic()
    bank = Path(config['refs']['bank']).resolve()
    if config['refs'].get('train_split') != 'bank':
        raise ValueError('Production config does not select the actor bank')
    validation = validate_reference_bank(bank)
    policy = validate_historical_policy(bank, args.repo)
    scope = json.loads((bank / 'provenance/documents/studio-scope.json').read_text())
    if scope.get('scope') != 'studio-only' or scope.get('complete') is not True:
        raise ValueError('Bank lacks its completed studio source audit')
    scope_root = Path(scope['csv_root'])
    for name, record in scope['files'].items():
        if sha256(scope_root / name) != record['derived_sha256']:
            raise ValueError(f'Studio source CSV differs from audited scope: {name}')
    with (scope_root / 'all-studio.csv').open(newline='') as handle:
        studio_rows = list(csv.DictReader(handle))
    studio_tracks = {(row['recording_id'], int(row['speaker'])): (row['uuid'], row['session_id'])
                     for row in studio_rows}
    import pandas as pd
    selected = pd.read_parquet(bank / 'selection.parquet')
    for row in selected.itertuples():
        if studio_tracks.get((row.recording_id, int(row.speaker))) != (row.uuid, row.session_id):
            raise ValueError(f'Selected reference is outside the audited studio tracks: iid={row.iid}')
    csv_path = Path(config['clipnames_root']) / config['train_csv']
    if sha256(csv_path) != scope['training']['sha256']:
        raise ValueError('Training CSV differs from the bank studio audit')
    with csv_path.open(newline='') as handle:
        clips = list(csv.DictReader(handle))
    actors = sorted({row['uuid'] for row in clips})
    if len(clips) != args.expected_clips or len(actors) != args.expected_training_actors:
        raise ValueError('Unexpected training manifest size or actor count')
    if any(row[key].strip() != 'no' for row in clips for key in ('blackbands', 'crossbleed', 'suspected_crossbleed')):
        raise ValueError('Training manifest includes excluded clips')
    with (bank / 'actor-coverage.csv').open(newline='') as handle:
        declared = {row['uuid']: row for row in csv.DictReader(handle) if row['bank'] == 'bank'}
    if len(declared) != args.expected_bank_actors or not set(actors) <= declared.keys():
        raise ValueError('Final bank roster does not cover the intended studio/training actors')
    actor_rows = []
    for actor in actors:
        counts = []
        for slot in range(1, 10):
            path = bank / 'bank' / actor / f'property_{slot}' / 'latents.safetensors'
            count = 0
            if path.exists():
                with safe_open(path, framework='pt', device='cpu') as tensors:
                    if list(tensors.keys()) != ['video']:
                        raise ValueError(f'Unexpected reference tensor keys: {path}')
                    value = tensors.get_slice('video')
                    shape = value.get_shape()
                    if len(shape) != 4 or shape[1:] != [48, 30, 16] or not 1 <= shape[0] <= 20 or value.get_dtype() != 'BF16':
                        raise ValueError(f'Wrong reference-bank tensor type/shape: {path}')
                    count = shape[0]
            if count != int(declared[actor][f'property_{slot}']):
                raise ValueError(f'Coverage receipt disagrees with actual slot: {actor}/{slot}')
            counts.append(count)
        actor_rows.append({'uuid': actor, 'slot_images': counts, 'nonempty_slots': sum(count > 0 for count in counts)})
    batches, signatures, epoch_steps = [], [], []
    for rank in args.ranks:
        data = build_joint_data(config, rank, 40)
        if {clip.uuid for clip in data.source.clips} != set(actors) or len(data.source.clips) != len(clips):
            raise ValueError('Actual dataset corpus differs from the audited manifest')
        group = data[0]
        if len(group) != 2 or [row['objective'] for row in group] != ['stabilizer', 'resolver']:
            raise ValueError('First group does not contain both paired objectives')
        for row in group:
            for name, shape in {'latents_low': (1, 48, 11, 30, 16), 'pose_latents_low': (1, 48, 11, 30, 16),
                                'refs': (1, 48, 9, 30, 16), 'ref_keep': (1, 9)}.items():
                if tuple(row[name].shape) != shape:
                    raise ValueError(f'Wrong first-group shape: rank {rank}/{name}')
            if row['ref_keep'].dtype != torch.bool or row['global_objective_rows'] != {'stabilizer': 40, 'resolver': 40}:
                raise ValueError('Wrong reference mask or global objective counts')
            for name, value in row.items():
                if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
                    raise ValueError(f'Nonfinite first-group tensor: rank {rank}/{name}')
            batches.append({'rank': rank, 'objective': row['objective'],
                            'shapes': {name: list(value.shape) for name, value in row.items() if isinstance(value, torch.Tensor)},
                            'reference_keep': row['ref_keep'].tolist(), 'window_ids': row['epoch_window_id'].tolist()})
        for name in ('refs', 'ref_keep', 'epoch_window_id', 'codes'):
            if not torch.equal(group[0][name], group[1][name]):
                raise ValueError(f'Objective pair differs in {name} on rank {rank}')
        signatures.append(data.signature)
        epoch_steps.append(len(data))
        del data, group
    if any(signature != signatures[0] for signature in signatures) or len(set(epoch_steps)) != 1:
        raise ValueError('CPU rank plans disagree')
    report = {'complete': True, 'cpu_only': True, 'config': str(args.config.resolve()), 'config_sha256': sha256(args.config),
              'training_csv_sha256': sha256(csv_path), 'clips': len(clips), 'training_actors': len(actors),
              'bank': str(bank), 'bank_validation': validation, 'bank_roster_actors': len(declared),
              'historical_policy': policy,
              'studio_selected_rows_checked': len(selected), 'non_studio_reference_tracks': 0,
              'training_actors_slot_histogram': dict(sorted(Counter(row['nonempty_slots'] for row in actor_rows).items())),
              'training_actors_with_all_9': sum(row['nonempty_slots'] == 9 for row in actor_rows),
              'training_actors_without_refs': [row['uuid'] for row in actor_rows if row['nonempty_slots'] == 0],
              'actor_slots': actor_rows, 'epoch_steps': epoch_steps[0], 'signature': signatures[0],
              'ranks_checked': args.ranks, 'first_group': batches,
              'source_sha256': {name: sha256(args.repo / name) for name in
                                ('distillation/joint_data.py', 'dataloading/ds_aloe.py', 'distillation/joint_runtime.py',
                                 'scripts/check_joint_bank.py')},
              'seconds': time.monotonic() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + '.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(args.output)
    print(json.dumps({key: report[key] for key in ('complete', 'clips', 'training_actors', 'training_actors_slot_histogram',
                       'training_actors_with_all_9', 'training_actors_without_refs', 'epoch_steps', 'seconds')}, indent=2))


if __name__ == '__main__':
    main()
