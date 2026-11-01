import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
import uuid


MARKER = 'checkpoint.complete.json'
POLL_SECONDS = 60


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 ** 2), b''):
            value.update(chunk)
    return value.hexdigest()


def file_generation(path):
    stat = path.stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def inspect_checkpoint(directory, checkpoint_id):
    path = Path(directory)
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ValueError('Checkpoint must be an absolute, real directory')
    marker = json.loads((path / MARKER).read_text())
    meta = json.loads((path / 'metadata.json').read_text())
    if marker['checkpoint_id'] != checkpoint_id or meta['checkpoint_id'] != checkpoint_id:
        raise ValueError('Checkpoint generation differs from --checkpoint-id')
    files = {item['path']: item['size'] for item in marker['files']}
    required = {'metadata.json', 'student.safetensors', 'critic.safetensors'}
    if len(files) != len(marker['files']) or not required.issubset(files) or MARKER in files:
        raise ValueError('Invalid checkpoint completion manifest')
    if meta.get('optimizer_state_retained') is True and any(
            f'rank-{rank:05d}.pt' not in files for rank in range(meta['world_size'])):
        raise ValueError('Checkpoint claims retained optimizer state but is missing rank shards')
    files[MARKER] = (path / MARKER).stat().st_size
    if set(files) != {entry.name for entry in path.iterdir()}:
        raise ValueError('Checkpoint contains unlisted or temporary files; it may be actively changing')
    for name, size in files.items():
        file = path / name
        if Path(name).name != name or name in ('.', '..') or file.is_symlink() or not file.is_file():
            raise ValueError(f'Unsafe checkpoint entry: {name}')
        if file.stat().st_size != size:
            raise ValueError(f'Incomplete checkpoint file: {name}')
    warmup = (meta.get('critic_warmup_complete') is True and meta['counters']['student_updates'] == 0
              and path.name == 'critic-warmup')
    pruned = meta.get('optimizer_state_retained') is False and not any(name.startswith('rank-') for name in files)
    if meta.get('optimizer_state_retained') is False and not pruned:
        raise ValueError('Pruned checkpoint still contains optimizer shards')
    return {'path': str(path), 'checkpoint_id': checkpoint_id, 'move_eligible': warmup or pruned,
            'optimizer_state_retained': meta.get('optimizer_state_retained'),
            'files': [{'path': name, 'size': files[name], 'generation': file_generation(path / name)}
                      for name in sorted(files)]}


def scan_pruned(directory):
    run = Path(directory)
    if not run.is_absolute() or not run.is_dir() or run.is_symlink():
        raise ValueError('Run must be an absolute, real directory')
    entries, pending = [], []
    for path in sorted(run.glob('step-*')):
        match = re.fullmatch(r'step-(\d+)', path.name)
        if not match or not path.is_dir():
            continue
        try:
            marker = json.loads((path / MARKER).read_text())
            snapshot = inspect_checkpoint(path, marker['checkpoint_id'])
        except (FileNotFoundError, ValueError):
            pending.append(path.name)
            continue
        entries.append({'path': str(path), 'step': int(match[1]), 'checkpoint_id': snapshot['checkpoint_id'],
                        'optimizer_state_retained': snapshot['optimizer_state_retained'],
                        'bytes': sum(item['size'] for item in snapshot['files'])})
    full = [entry for entry in entries if entry['optimizer_state_retained'] is True]
    latest = max(full, key=lambda entry: entry['step']) if full else None
    older = [entry for entry in entries if latest and entry['step'] < latest['step']]
    return {'latest_full': latest, 'candidates': [entry for entry in older if entry['optimizer_state_retained'] is False],
            'unpruned_older': [entry['path'] for entry in older if entry['optimizer_state_retained'] is True],
            'pending': pending, 'free_bytes': shutil.disk_usage(run).free}


def verify_source(snapshot, directory=None):
    current = inspect_checkpoint(directory or snapshot['path'], snapshot['checkpoint_id'])
    expected = [{key: item[key] for key in ('path', 'size', 'generation')} for item in snapshot['files']]
    if current['files'] != expected:
        raise ValueError('Source checkpoint changed during archival; source must remain untouched')


def snapshot_checkpoint(directory, checkpoint_id):
    result = inspect_checkpoint(directory, checkpoint_id)
    for item in result['files']:
        item['sha256'] = digest(Path(directory) / item['path'])
    verify_source(result)
    return result


def verify_copy(directory, snapshot):
    actual = snapshot_checkpoint(directory, snapshot['checkpoint_id'])
    keys = ('path', 'size', 'sha256')
    if [{key: item[key] for key in keys} for item in actual['files']] != [
            {key: item[key] for key in keys} for item in snapshot['files']]:
        raise ValueError('Destination checkpoint content differs from source SHA256 manifest')
    for item in actual['files']:
        with (Path(directory) / item['path']).open('rb') as handle:
            os.fsync(handle.fileno())


def write_receipt(path, receipt):
    temporary = path.with_name(path.name + f'.tmp-{uuid.uuid4().hex}')
    with temporary.open('x') as handle:
        json.dump(receipt, handle, indent=2)
        handle.write('\n')
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def retire_source(snapshot, receipt):
    if not snapshot['move_eligible']:
        raise ValueError('Move is restricted to completed critic-warmup or optimizer-pruned checkpoints')
    verify_source(snapshot)
    source = Path(snapshot['path'])
    retired = source.with_name(f'.{source.name}.archiving-{uuid.uuid4().hex}')
    source.rename(retired)
    try:
        verify_source(snapshot, retired)
    except Exception:
        if not source.exists():
            retired.rename(source)
        raise
    write_receipt(source.with_name(source.name + '.archive.json'), receipt)
    shutil.rmtree(retired)


def worker(operation, payload):
    if operation == 'scan-pruned':
        return scan_pruned(payload['path'])
    if operation == 'exists':
        return {'exists': Path(payload['path']).exists()}
    if operation == 'snapshot':
        return snapshot_checkpoint(payload['path'], payload['checkpoint_id'])
    if operation == 'prepare':
        destination = Path(payload['path'])
        if not destination.is_absolute() or destination.exists():
            raise ValueError('Destination must be a new absolute path')
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f'.{destination.name}.incomplete-{uuid.uuid4().hex}')
        temporary.mkdir()
        return {'temporary': str(temporary)}
    if operation == 'verify-copy':
        verify_copy(payload['path'], payload['snapshot'])
    elif operation == 'verify-source':
        verify_source(payload['snapshot'])
    elif operation == 'publish':
        destination = Path(payload['path'])
        if destination.exists():
            raise FileExistsError(destination)
        Path(payload['temporary']).rename(destination)
        write_receipt(destination.with_name(destination.name + '.archive.json'), payload['receipt'])
        descriptor = os.open(destination.parent, os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    elif operation == 'retire':
        retire_source(payload['snapshot'], payload['receipt'])
    else:
        raise ValueError(operation)
    return {'ok': True}


def remote(host, operation, payload):
    code = Path(__file__).read_text()
    command = shlex.join(['python3', '-c', code, '--internal', operation, '--payload', json.dumps(payload)])
    output = subprocess.check_output(['ssh', '-o', 'BatchMode=yes', host, command], text=True)
    return json.loads(output)


def validate_hosts(args):
    for host in (args.source_host, args.destination_host):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.@:-]*', host):
            raise ValueError(f'Invalid SSH host: {host}')
    if args.source_host == args.destination_host:
        raise ValueError('Source and destination hosts must differ')


def archive(args):
    validate_hosts(args)
    print('Hashing completed source checkpoint...', flush=True)
    snapshot = remote(args.source_host, 'snapshot', {'path': args.source, 'checkpoint_id': args.checkpoint_id})
    if args.move and not snapshot['move_eligible']:
        raise ValueError('Move requires critic-warmup or a checkpoint with optimizer shards already pruned')
    temporary = args.destination
    if not args.existing_destination:
        temporary = remote(args.destination_host, 'prepare', {'path': args.destination})['temporary']
        print(f'Copying to {args.destination_host}:{temporary}', flush=True)
        command = shlex.join(['rsync', '-a', '--protect-args', '--info=stats2', '-e', 'ssh -o BatchMode=yes', '--', args.source.rstrip('/') + '/',
                              f'{args.destination_host}:{temporary}/'])
        subprocess.run(['ssh', '-o', 'BatchMode=yes', args.source_host, command], check=True)
    print('Verifying destination SHA256 and unchanged source generation...', flush=True)
    remote(args.destination_host, 'verify-copy', {'path': temporary, 'snapshot': snapshot})
    remote(args.source_host, 'verify-source', {'snapshot': snapshot})
    receipt = {'source_host': args.source_host, 'source': args.source, 'destination_host': args.destination_host,
               'destination': args.destination, 'checkpoint_id': args.checkpoint_id,
               'files': [{key: item[key] for key in ('path', 'size', 'sha256')} for item in snapshot['files']]}
    if not args.existing_destination:
        remote(args.destination_host, 'publish', {'path': args.destination, 'temporary': temporary, 'receipt': receipt})
    if args.move:
        print('Verified destination published; retiring unchanged source...', flush=True)
        remote(args.source_host, 'retire', {'snapshot': snapshot, 'receipt': receipt})
    print(json.dumps({'archived': True, 'moved': args.move, 'checkpoint_id': args.checkpoint_id,
                      'destination_host': args.destination_host, 'destination': args.destination,
                      'bytes': sum(item['size'] for item in snapshot['files'])}), flush=True)


def watch_pruned(args):
    validate_hosts(args)
    if not args.move or args.final_step is None or args.poll_seconds <= 0:
        raise ValueError('--watch-pruned requires --move, --final-step, and positive --poll-seconds')
    previous = None
    while True:
        status = remote(args.source_host, 'scan-pruned', {'path': args.source})
        changed = {key: value for key, value in status.items() if key != 'free_bytes'}
        if changed != previous:
            print(json.dumps({'archive_monitor': status}), flush=True)
            previous = changed
        if status['candidates']:
            for candidate in status['candidates']:
                destination = str(Path(args.destination) / Path(candidate['path']).name)
                existing = remote(args.destination_host, 'exists', {'path': destination})['exists']
                archive(argparse.Namespace(source_host=args.source_host, source=candidate['path'],
                                           destination_host=args.destination_host, destination=destination,
                                           checkpoint_id=candidate['checkpoint_id'], move=True,
                                           existing_destination=existing))
            continue
        latest = status['latest_full']
        if latest and latest['step'] >= args.final_step and not status['pending'] and not status['unpruned_older']:
            print(json.dumps({'archive_monitor_complete': True, 'latest_full_kept': latest['path']}), flush=True)
            return
        time.sleep(args.poll_seconds)


def main():
    parser = argparse.ArgumentParser(description='Copy verified checkpoint generations between SSH hosts; optionally retire immutable sources.')
    parser.add_argument('--source-host')
    parser.add_argument('--source')
    parser.add_argument('--destination-host')
    parser.add_argument('--destination')
    parser.add_argument('--checkpoint-id')
    parser.add_argument('--move', action='store_true')
    parser.add_argument('--existing-destination', action='store_true',
                        help='Verify an existing copy instead of transferring; combine with --move to retire the source')
    parser.add_argument('--watch-pruned', action='store_true',
                        help='Monitor a source run directory and archive pruned steps only after a newer full checkpoint completes')
    parser.add_argument('--final-step', type=int)
    parser.add_argument('--poll-seconds', type=float, default=POLL_SECONDS)
    parser.add_argument('--internal', choices=['snapshot', 'prepare', 'verify-copy', 'verify-source', 'publish', 'retire',
                                             'scan-pruned', 'exists'],
                        help=argparse.SUPPRESS)
    parser.add_argument('--payload', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.internal:
        print(json.dumps(worker(args.internal, json.loads(args.payload))))
    else:
        if not all((args.source_host, args.source, args.destination_host, args.destination)):
            parser.error('--source-host, --source, --destination-host, and --destination are required')
        if args.watch_pruned:
            watch_pruned(args)
        else:
            if not args.checkpoint_id:
                parser.error('--checkpoint-id is required for a single checkpoint')
            archive(args)


if __name__ == '__main__':
    main()
