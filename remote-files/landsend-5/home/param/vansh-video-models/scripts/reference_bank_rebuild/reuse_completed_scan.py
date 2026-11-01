"""Reuse a completed scan only with its verified HF acquisition and exact inputs.

Prepare imports every required candidate checkpoint without downloading or
rehashing full videos. Verify runs after the ordinary distributed scanner has
independently checked the new plan on all nodes. Original acquisition receipts
are pinned by hash and current file stat; missing or changed evidence fails.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from download_and_scan import VerifiedInputs, require_selected_complete, track_inputs
from hf_sources import read_download_list, write_json
from import_scan_checkpoints import import_caches, track_map
from measurements import ARCHIVE, MODELS
from video_pipeline import code_config, digest, plan_identity


RECEIPT = 'scan-reuse-provenance.json'


def acquisition_evidence(cache, output, source_root, receipts, config):
    cache, output, receipts = Path(cache).resolve(), Path(output).resolve(), Path(receipts).resolve()
    if cache == output:
        raise ValueError('Completed scan cache must differ from the new output directory')
    old_identity, new_identity = plan_identity(cache), plan_identity(output)
    original = track_map(cache / 'tracks.json')
    final = track_map(output / 'tracks.json')
    if plan_identity(cache) != old_identity or plan_identity(output) != new_identity:
        raise ValueError('Scan plan changed while reading its track identities')
    paths = [cache / name for name in ('tracks.json', 'targets-summary.json', 'downloads.tsv',
             'scan-summary.json', 'scan-code-inputs.json', 'download-scan-status.json')]
    paths += [receipts, Path(str(receipts) + '.summary.json'), output / 'downloads.tsv']
    pinned = {str(path): digest(path) for path in paths}
    old_report = json.loads((cache / 'scan-summary.json').read_text())
    require_selected_complete(old_report, list(original.values()), old_identity)
    status = json.loads((cache / 'download-scan-status.json').read_text())
    if (status.get('stage') != 'finished' or status.get('available_targets_complete') is not True
            or status.get('plan_sha256') != old_identity or status.get('scan_summary') != old_report):
        raise ValueError('Original HF acquisition/scan did not finish for its exact plan')
    if json.loads((cache / 'scan-code-inputs.json').read_text()) != config:
        raise ValueError('Original scan code/model fingerprint differs')
    for name, track in final.items():
        if original.get(name) != track:
            raise ValueError(f'New track is absent or differs from completed scan: {name}')
    for directory, tracks in ((cache, original), (output, final)):
        expected = {(key, path) for track in tracks.values() for key, path, _ in track_inputs(track)}
        actual = read_download_list(directory / 'downloads.tsv', source_root)
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError(f'Download list differs from exact planned inputs: {directory}')
    verified = VerifiedInputs(receipts, list(original.values()))
    verified.update()
    verified.assert_unchanged()
    summary = json.loads(Path(str(receipts) + '.summary.json').read_text())
    if (summary.get('failures') != [] or summary.get('missing') != []
            or summary.get('requested') != len(verified.expected)
            or status.get('selected_inputs_verified') != len(verified.expected)
            or verified.missing() or verified.offset != receipts.stat().st_size):
        raise ValueError('Original HF receipt set is incomplete or mismatched')
    if any(digest(Path(path)) != expected for path, expected in pinned.items()):
        raise ValueError('Original acquisition evidence changed during verification')
    evidence = {'cache': str(cache), 'output_dir': str(output), 'source_root': str(Path(source_root).resolve()),
                'receipt_path': str(receipts), 'original_plan_sha256': old_identity,
                'plan_sha256': new_identity, 'code_inputs': config, 'evidence_sha256': pinned,
                'original_verified_inputs': len(verified.verified), 'selected_tracks': len(final),
                'expected_chunks': sum(len(track['usable_chunks']) for track in final.values()),
                'selected_inputs': len({path for track in final.values() for _, path, _ in track_inputs(track)}),
                'acquisition_policy': 'Inherited HF SHA256/Xet verification with exact unchanged local stat; no video rehash or download'}
    return evidence, final, verified


def prepare(cache, output, source_root, receipts, config):
    output = Path(output)
    (output / RECEIPT).unlink(missing_ok=True)
    evidence, final, verified = acquisition_evidence(cache, output, source_root, receipts, config)
    report = import_caches(output, [cache], config)
    expected = {(name, chunk) for name, track in final.items() for chunk in track['usable_chunks']}
    imported = [row for item in report['caches'] for row in item['chunks']]
    if (report['absent_chunks'] or report['rejected_chunks']
            or report['imported_chunks'] + report['already_valid_chunks'] != len(expected)
            or len(imported) != len(expected)
            or {(row['track'], row['chunk']) for row in imported} != expected
            or any(row['status'] not in ('imported', 'already-valid') for row in imported)):
        raise ValueError('Completed scan cannot supply every selected chunk; inspect scan-cache-import.json')
    verified.assert_unchanged()
    if plan_identity(output) != evidence['plan_sha256']:
        raise ValueError('New scan plan changed during import')
    result = {**evidence, 'prepared': True, 'available_targets_complete': False,
              'complete': False, 'prepared_at': datetime.now(timezone.utc).isoformat(),
              'import_report_sha256': digest(output / 'scan-cache-import.json')}
    write_json(output / RECEIPT, result)
    return result


def verify(cache, output, source_root, receipts, config):
    output = Path(output)
    prepared = json.loads((output / RECEIPT).read_text())
    if prepared.get('prepared') is not True:
        raise ValueError('Required prepared scan reuse receipt is absent')
    prepared.update(complete=False, available_targets_complete=False)
    write_json(output / RECEIPT, prepared)
    evidence, final, verified = acquisition_evidence(cache, output, source_root, receipts, config)
    if any(prepared.get(key) != value for key, value in evidence.items()):
        raise ValueError('Scan reuse evidence changed after import')
    if digest(output / 'scan-cache-import.json') != prepared['import_report_sha256']:
        raise ValueError('Scan import receipt changed')
    report = json.loads((output / 'scan-summary.json').read_text())
    require_selected_complete(report, list(final.values()), evidence['plan_sha256'])
    if json.loads((output / 'scan-code-inputs.json').read_text()) != config:
        raise ValueError('Final reused scan code/model fingerprint differs')
    distributed = json.loads((output / 'distributed-scan.json').read_text())
    if (distributed.get('aggregation_verified') is not True
            or distributed.get('available_targets_complete') is not True
            or len(distributed.get('nodes', [])) != 5
            or len(distributed.get('returncodes', {})) != 5
            or any(value != 0 for value in distributed['returncodes'].values())):
        raise ValueError('Required successful five-node scan verification is absent')
    verified.assert_unchanged()
    prepared.update(complete=True, available_targets_complete=True,
                    source_complete=report['complete'], finished_at=datetime.now(timezone.utc).isoformat(),
                    scan_summary_sha256=digest(output / 'scan-summary.json'),
                    distributed_scan_sha256=digest(output / 'distributed-scan.json'))
    write_json(output / RECEIPT, prepared)
    return prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify'))
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--receipts', type=Path, help='Defaults to CACHE/../hf-video-receipts.jsonl')
    parser.add_argument('--archive-root', type=Path, default=ARCHIVE)
    parser.add_argument('--model-root', type=Path, default=MODELS)
    args = parser.parse_args()
    receipts = args.receipts or args.cache.parent / 'hf-video-receipts.jsonl'
    function = prepare if args.command == 'prepare' else verify
    result = function(args.cache, args.output_dir, args.source_root, receipts,
                      code_config(args.archive_root, args.model_root))
    print(json.dumps({key: result[key] for key in ('complete', 'expected_chunks', 'selected_tracks', 'selected_inputs')}))


if __name__ == '__main__':
    main()
