"""Resume the complete HF-only build after the concurrent source acquisition starts.

Missing HF inputs remain explicit source-coverage evidence. A bank is usable
only after all available work and artifact checks pass; validated artifacts can
still report source_complete=false when upstream candidates are unavailable.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

from process_output import start_logged


def work_root(root, subdir):
    root = Path(root).resolve()
    if not subdir:
        return root
    relative = Path(subdir)
    if relative.is_absolute() or any(part in ('.', '..') for part in relative.parts):
        raise ValueError('--work-subdir must be a relative directory inside --root')
    result = (root / relative).resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError('--work-subdir must identify a separate directory inside --root')
    return result


def process_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # An unreaped downloader has exited even though kill(pid, 0) still succeeds.
    stat = Path(f'/proc/{pid}/stat')
    try:
        return stat.read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def join_prefetch_downloads(registry, status, poll_seconds=30):
    """Join registered source writers before starting the full-plan downloader.

    Only current matching successful summaries prove completion: hf_sources
    publishes them after its download pool closes. PIDs are liveness evidence,
    never targets for a stop signal. Register the whole job set before entry;
    queued jobs may acquire PIDs while this function waits.
    """
    registry = Path(registry)
    stage = 'waiting-for-early-video-downloads'
    expected = None
    try:
        while True:
            if not registry.is_file():
                raise RuntimeError(f'Missing {registry}; register all early download jobs, or an empty jobs list')
            jobs = json.loads(registry.read_text())['jobs']
            identities = []
            for job in jobs:
                receipts, requested, pid = Path(job['receipts']), job['requested'], job['pid']
                if (not receipts.is_absolute() or type(requested) is not int or requested < 0
                        or (pid is not None and (type(pid) is not int or pid <= 0))):
                    raise ValueError(f'Invalid early download registration: {job}')
                identities.append((str(receipts), requested))
            if len({row[0] for row in identities}) != len(identities):
                raise ValueError('Duplicate early download receipt paths in registry')
            identities = sorted(identities)
            if expected is None:
                expected = identities
            elif identities != expected:
                raise ValueError('Early download job set changed after join began; register all jobs before this stage')
            waiting = []
            for job in jobs:
                summary_path = Path(job['receipts'] + '.summary.json')
                if summary_path.is_file():
                    summary = json.loads(summary_path.read_text())
                    if summary.get('requested') != job['requested']:
                        raise RuntimeError(f'Stale early download summary {summary_path}: expected requested={job["requested"]}')
                    if summary.get('failures') != [] or summary.get('missing') != []:
                        raise RuntimeError(f'Early download has failed or missing sources; repair/resume it: {summary_path}')
                elif job['pid'] is not None and not process_alive(job['pid']):
                    raise RuntimeError(f'Early downloader PID {job["pid"]} exited without {summary_path}; resume that download')
                else:
                    waiting.append(job)
            status(stage, running=bool(waiting), registered_jobs=len(jobs),
                   completed_jobs=len(jobs) - len(waiting), waiting_jobs=waiting)
            if not waiting:
                return
            time.sleep(poll_seconds)
    except Exception as error:
        status(stage, running=False, failed=True, error=f'{type(error).__name__}: {error}')
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--token-name', default='oauth-paramluhadiya')
    p.add_argument('--gpus', type=int, default=8)
    p.add_argument('--csv-root', type=Path, help='Derived corpus CSVs; requires a separate --work-subdir')
    p.add_argument('--expected-actors', type=int, default=1759)
    p.add_argument('--work-subdir', default='', help='Separate plans, selections, aggregates and bank under ROOT')
    p.add_argument('--scan-workers', type=int, default=64, help='CPU scan workers per node')
    p.add_argument('--decode-workers', type=int, default=64, help='Concurrent selected-frame decodes')
    p.add_argument('--selection-policy', choices=('strict', 'historical-studio'), default='strict')
    p.add_argument('--training-csv', type=Path, help='Required for strict actor-coverage validation')
    p.add_argument('--rules-root', type=Path, help='Pinned strict sinks.md and RECIPE.md directory')
    p.add_argument('--reuse-video-scan', type=Path,
                   help='Completed scan with unchanged verified HF inputs; require every new target chunk from this cache')
    p.add_argument('--encoder-nodes', nargs='+', default=['192.168.0.4', '192.168.0.9', '192.168.0.7',
                                                        '192.168.0.8', '192.168.0.6'])
    args = p.parse_args()
    if args.expected_actors < 1 or args.scan_workers < 1 or args.decode_workers < 1 or args.gpus < 1:
        p.error('Actor count, scan workers and GPU count must be positive')
    if args.csv_root is not None and not args.work_subdir:
        p.error('--csv-root requires --work-subdir to preserve existing outputs')
    if args.selection_policy == 'historical-studio' and (not args.csv_root or not args.work_subdir):
        p.error('Historical studio selection requires explicit --csv-root and a separate --work-subdir')
    if args.selection_policy == 'strict' and (args.training_csv is None or args.rules_root is None):
        p.error('--selection-policy strict requires --training-csv and --rules-root')
    if args.reuse_video_scan and len(set(args.encoder_nodes)) != 5:
        p.error('Completed scan reuse requires five distinct nodes for final verification')
    # Bound Xet requests per process; a large number of small PNG requests
    # otherwise triggers CAS throttling before saturating the network.
    os.environ.setdefault('HF_XET_DATA_MAX_CONCURRENT_FILE_DOWNLOADS', '4')
    os.environ.setdefault('HF_XET_CLIENT_AC_MAX_DOWNLOAD_CONCURRENCY', '8')
    os.environ.setdefault('HF_XET_CLIENT_AC_INITIAL_DOWNLOAD_CONCURRENCY', '2')
    os.environ.setdefault('RAYON_NUM_THREADS', '2')
    root = args.root.resolve()
    work = work_root(root, args.work_subdir)
    if args.reuse_video_scan and args.reuse_video_scan.resolve() == (work / 'video-scan').resolve():
        p.error('--reuse-video-scan must preserve a separate completed scan directory')
    work.mkdir(parents=True, exist_ok=True)
    scoped = work != root
    from upstream import CSV_ROOT
    csv_root = args.csv_root.resolve() if args.csv_root else CSV_ROOT
    scripts = Path(__file__).resolve().parent
    py = sys.executable
    logs = work / 'stage-logs'
    logs.mkdir(exist_ok=True)
    state_path = work / 'build-state.json'
    state = {'started_at': datetime.now(timezone.utc).isoformat(), 'completed_stages': [], 'complete': False,
             'work_root': str(work), 'csv_root': str(csv_root), 'expected_actors': args.expected_actors,
             'scan_workers_per_node': args.scan_workers, 'decode_workers': args.decode_workers,
             'selection_policy': args.selection_policy,
             'training_csv': str(args.training_csv.resolve()) if args.training_csv else None,
             'rules_root': str(args.rules_root.resolve()) if args.rules_root else None,
             'reuse_video_scan': str(args.reuse_video_scan.resolve()) if args.reuse_video_scan else None}

    def status(stage, **kwargs):
        state.update(stage=stage, updated_at=datetime.now(timezone.utc).isoformat(), **kwargs)
        tmp = state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state, indent=2) + '\n')
        tmp.replace(state_path)
        print(json.dumps({'stage': stage, **kwargs}), flush=True)

    def command(stage, script, *arguments, allowed_codes=(0,)):
        status(stage, running=True)
        argv = [py, '-u', str(scripts / script), *map(str, arguments)]
        process, output = start_logged(argv, logs / f'{stage}.log')
        try:
            returncode = process.wait()
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait()
            output.join()
        if returncode not in allowed_codes:
            status(stage, running=False, failed=True, returncode=returncode)
            raise RuntimeError(f'{stage} failed ({returncode}); see {logs / (stage + ".log")}')
        state['completed_stages'].append(stage)
        status(stage, running=False, last_returncode=returncode)
        return returncode

    metadata = root / 'hf-upstream-metadata-receipts.jsonl'
    images = root / 'hf-upstream-image-receipts.jsonl'
    reports = [Path(str(path) + '.missing.json') for path in (metadata, images)]
    sources, plan, shared_measures = root / 'sources', work / 'plan', root / 'measurements'
    measures = work / 'measurements'
    selection, video, bank = work / 'selection', work / 'video-scan', work / 'reference-bank-256x480-muon-encoded'
    for path in (plan, measures, selection, video, bank):
        path.mkdir(exist_ok=True)
    command('inventory', 'upstream.py', 'inventory', '--csv-root', csv_root, '--output-dir', plan)
    inventory = json.loads((plan / 'inventory.json').read_text())
    if args.selection_policy == 'historical-studio':
        from hf_sources import write_json
        inventory['selection_policy'] = 'Fresh historical studio/Beatles image thresholds; no old selections or latents reused'
        inventory['selection_policy_document'] = str(selection / 'RECIPE-historical-studio.md')
        write_json(plan / 'inventory.json', inventory)
    if inventory['actors'] != args.expected_actors:
        status('inventory-count-mismatch', running=False, failed=True, actual_actors=inventory['actors'])
        raise ValueError('Scoped corpus actor count differs from --expected-actors')
    status('waiting-for-upstream-downloads-and-measurements', running=True)
    while True:
        runtime = shared_measures / 'watch-runtime.json'
        failures = shared_measures / 'watch-failures.json'
        if runtime.is_file() and failures.is_file() and json.loads(failures.read_text()):
            status('upstream-measurements-failed', running=False, failed=True)
            raise RuntimeError('Upstream measurement failures; inspect watch-failures.json and resume')
        summaries = [Path(str(path) + '.summary.json') for path in (metadata, images)]
        if all(path.is_file() for path in summaries):
            results = [json.loads(path.read_text()) for path in summaries]
            if any(row['failures'] for row in results):
                status('upstream-downloads-failed', running=False, failed=True)
                raise RuntimeError('HF upstream downloads have failures; resume downloads before rebuilding')
            # A registered producer must finish before a fresh scope aggregates.
            done = (scoped and not runtime.is_file()) or (shared_measures / 'watch-complete.json').is_file()
            log = root / 'measure-upstream.log'
            if not done and log.is_file():
                last = log.read_text().splitlines()[-1:]
                try:
                    done = bool(last and json.loads(last[0]).get('complete'))
                except json.JSONDecodeError:
                    pass
            if done:
                break
        if runtime.is_file():
            pid = json.loads(runtime.read_text())['pid']
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                status('upstream-measurement-worker-exited', running=False, failed=True)
                raise RuntimeError('Measurement watcher exited before successful completion')
        time.sleep(30)
    missing_args = [item for path in reports for item in ('--missing-report', path)]
    runtime = shared_measures / 'watch-runtime.json'
    measurement_scripts = Path(json.loads(runtime.read_text())['source_directory']) if runtime.is_file() else scripts
    if scoped:
        command('aggregate-measurements', measurement_scripts / 'aggregate_scope.py', '--tracks', plan / 'tracks.txt',
                '--data-root', sources, '--checkpoint-root', shared_measures, '--output-dir', measures, '--workers', 32)
    else:
        command('aggregate-measurements', measurement_scripts / 'measure_upstream.py', '--tracks', plan / 'tracks.txt',
                '--data-root', sources, '--output-dir', measures, '--workers', 32, *missing_args)
    command('plan', 'parallel_plan.py', '--csv-root', csv_root,
            '--data-root', sources, '--output-dir', plan, '--workers', 64, *missing_args)
    table_args = ['--plan', plan / 'plan.parquet', '--p14', measures / 'p14.parquet',
                  '--p59', measures / 'p59.parquet', '--teeth', measures / 'teeth.parquet']
    if args.selection_policy == 'historical-studio':
        command('select-historical-studio', 'historical_studio_select.py', *table_args,
                '--actors', plan / 'actors.csv', '--csv-root', csv_root, '--output-dir', selection)
    else:
        command('select-upstream', 'upstream.py', 'select', *table_args, '--actors', plan / 'actors.csv', '--output-dir', selection)
        command('select-beatles', 'upstream.py', 'select-beatles', *table_args,
                '--beatles-csv', csv_root / 'aloe-eval-beatles.csv', '--output-dir', selection)
    upstream = selection / 'upstream-selection.parquet'
    command('plan-video-scan', 'video_pipeline.py', 'targets', '--upstream', upstream,
            '--video-index', root / 'hf-video-index.json', '--source-root', sources,
            '--csv-root', csv_root, '--output-dir', video, '--allow-incomplete')
    join_prefetch_downloads(root / 'prefetch-downloads.json', status)
    if args.reuse_video_scan:
        reuse_args = ['--cache', args.reuse_video_scan.resolve(), '--output-dir', video, '--source-root', sources]
        command('prepare-completed-video-scan', 'reuse_completed_scan.py', 'prepare', *reuse_args)
        command('verify-reused-video-scan-on-five-nodes', 'distributed_scan.py', '--output-dir', video,
                '--workers', args.scan_workers, '--nodes', *args.encoder_nodes, '--allow-incomplete')
        command('verify-completed-video-scan', 'reuse_completed_scan.py', 'verify', *reuse_args)
    else:
        scan_caches = [path for path in sorted((root / 'prefetch-previews').glob('*/video-targets'))
                       if (path / 'tracks.json').is_file() and (path / 'candidates').is_dir()]
        command('download-and-scan-video-sources', 'download_and_scan.py', '--token-name', args.token_name,
                '--output-dir', video, '--source-root', sources, '--receipts', work / 'hf-video-receipts.jsonl',
                '--workers', args.scan_workers, '--nodes', *args.encoder_nodes,
                *[argument for path in scan_caches for argument in ('--cache', path)])
    if args.selection_policy == 'historical-studio':
        command('select-video-frames', 'historical_video_select.py', '--output-dir', video,
                '--upstream', upstream, '--allow-incomplete')
    else:
        command('select-video-frames', 'video_pipeline.py', 'select', '--output-dir', video,
                '--upstream', upstream, '--allow-incomplete')
    command('decode-video-frames', 'video_pipeline.py', 'decode', '--output-dir', video,
            '--workers', args.decode_workers, '--allow-incomplete')
    if args.selection_policy == 'historical-studio':
        command('validate-historical-video-decode', 'validate_historical_decode.py', '--output-dir', video)
    command('merge-selection', 'finalize_bank.py', 'merge', '--upstream', upstream,
            '--video', video / 'video-encoding-input.parquet', '--beatles', selection / 'beatles-selection.parquet',
            '--output-dir', bank)
    quality_reports, quality_args = [], []
    if args.selection_policy == 'strict':
        from measurements import ARCHIVE
        quality_reports = [work / 'proofs' / f'strict-quality-{stage}.json' for stage in ('preencode', 'postencode')]
        quality_args = ['--selection', bank / 'selection.parquet', '--measurements-dir', measures,
                        '--source-root', sources, '--scan-dir', video, '--fit-score', ARCHIVE / 'fit_score.parquet',
                        '--training-csv', args.training_csv.resolve(), '--rules-root', args.rules_root.resolve(),
                        '--workers', args.decode_workers]
        command('strict-quality-preencode', 'strict_quality_check.py', *quality_args, '--report', quality_reports[0])
    command('encode', 'distributed_encode.py', 'run', '--selection', bank / 'selection.parquet',
            '--source-root', sources, '--output-dir', bank, '--gpus-per-node', args.gpus,
            '--nodes', *args.encoder_nodes)
    command('decode-check', 'decode_check.py', '--selection', bank / 'selection.parquet',
            '--source-root', sources, '--output-dir', bank)
    if args.selection_policy == 'strict':
        command('strict-quality-postencode', 'strict_quality_check.py', *quality_args,
                '--bank-root', bank, '--report', quality_reports[1])
    documents = []
    scope_manifest = csv_root / 'studio-scope.json'
    if args.selection_policy == 'historical-studio':
        documents = [selection / name for name in ('RECIPE-historical-studio.md', 'selection-policy.json',
                                                  'archived-studio-select.py', 'archived-beatles-plan.py')]
        documents += [video / name for name in ('historical-video-policy.json', 'archived-studio-select-v2.py',
                                              'historical-video-decode.json')]
    else:
        documents = [args.rules_root / 'RECIPE.md', args.rules_root / 'sinks.md']
    if scope_manifest.is_file():
        documents.append(scope_manifest)
    if args.reuse_video_scan:
        documents += [video / 'scan-reuse-provenance.json', video / 'scan-cache-import.json']
    documents += quality_reports
    document_args = [argument for document in documents for argument in ('--document', document)]
    result = command('finalize', 'finalize_bank.py', 'finalize', '--selection', bank / 'selection.parquet',
                     '--actors', plan / 'actors.csv', '--source-root', sources,
                     '--csv-root', csv_root, '--expected-actors', args.expected_actors,
                     '--video-index', root / 'hf-video-index.json',
                     '--source-diagnostics', plan / 'input-diagnostics.json', '--scan-dir', video,
                     '--beatles-holdout', selection / 'beatles-holdout.json', '--output-dir', bank,
                     *document_args, allowed_codes=(0, 2))
    report = json.loads((bank / 'coverage.json').read_text())
    with (bank / 'selection.parquet').open('rb') as handle:
        selection_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    if (report['selection_sha256'] != selection_hash or report['sources']['errors'] or report['latents']['errors']
            or bool(report['complete']) != (result == 0)):
        status('final-validation-failed', running=False, failed=True)
        raise RuntimeError('Final report failed integrity checks; partial output must not be used')
    status('finished' if result == 0 else 'incomplete-source-coverage', running=False, complete=result == 0,
           output_dir=str(bank), finished_at=datetime.now(timezone.utc).isoformat())


if __name__ == '__main__':
    main()
