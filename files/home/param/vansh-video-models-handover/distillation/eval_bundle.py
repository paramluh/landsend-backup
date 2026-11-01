"""Verified student-only evaluation bundles, explicitly not resumable checkpoints."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import uuid

from distillation.archive_checkpoint import digest, inspect_checkpoint, verify_source


MARKER = 'evaluation.complete.json'
SOURCE_MARKER = 'source.checkpoint.complete.json'
FORMAT = 'student-evaluation-v1'
PAYLOAD = ('student.safetensors', 'metadata.json', SOURCE_MARKER)


def read_bundle(directory):
    """Verify every bundled byte before returning original student metadata."""
    directory = Path(directory)
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('Evaluation bundle must be a real directory')
    if {entry.name for entry in directory.iterdir()} != {*PAYLOAD, MARKER}:
        raise ValueError('Evaluation bundle has missing or unexpected files')
    if any((directory / name).is_symlink() for name in (*PAYLOAD, MARKER)):
        raise ValueError('Evaluation bundle must not contain symlinks')
    receipt = json.loads((directory / MARKER).read_text())
    if receipt.get('format') != FORMAT or receipt.get('training_resume_supported') is not False:
        raise ValueError('Invalid evaluation bundle format')
    files = receipt['files']
    if len(files) != len(PAYLOAD) or {item['path'] for item in files} != set(PAYLOAD):
        raise ValueError('Invalid evaluation bundle manifest')
    for item in files:
        path = directory / item['path']
        if not path.is_file() or path.stat().st_size != item['size'] or digest(path) != item['sha256']:
            raise ValueError(f'Evaluation bundle content differs: {item["path"]}')
    meta = json.loads((directory / 'metadata.json').read_text())
    source = json.loads((directory / SOURCE_MARKER).read_text())
    if meta['checkpoint_id'] != receipt['checkpoint_id'] or source['checkpoint_id'] != receipt['checkpoint_id']:
        raise ValueError('Evaluation bundle source checkpoint identities differ')
    original = {item['path']: item['size'] for item in source['files']}
    for name in ('student.safetensors', 'metadata.json'):
        if original.get(name) != (directory / name).stat().st_size:
            raise ValueError(f'Evaluation payload disagrees with source completion manifest: {name}')
    return directory / 'student.safetensors', meta


def create_bundle(source, destination, checkpoint_id):
    """Copy only student/metadata; stat-check full source without reading optimizer data.

    The retained source marker has a different filename, so training checkpoint
    readers cannot mistake this subset for a complete training checkpoint.
    """
    source, destination = Path(source).absolute(), Path(destination).absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    if source == destination or source in destination.parents:
        raise ValueError('Evaluation bundle must be outside the source checkpoint')
    before = inspect_checkpoint(source, checkpoint_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.partial-{uuid.uuid4().hex}')
    temporary.mkdir()
    try:
        files = []
        for name in PAYLOAD:
            origin = source / ('checkpoint.complete.json' if name == SOURCE_MARKER else name)
            value = hashlib.sha256()
            size = 0
            with origin.open('rb') as reader, (temporary / name).open('xb') as writer:
                for chunk in iter(lambda: reader.read(8 * 1024 ** 2), b''):
                    writer.write(chunk)
                    value.update(chunk)
                    size += len(chunk)
                writer.flush()
                os.fsync(writer.fileno())
            files.append({'path': name, 'size': size, 'sha256': value.hexdigest()})
        verify_source(before)
        receipt = {'format': FORMAT, 'checkpoint_id': checkpoint_id,
                   'training_resume_supported': False, 'files': files,
                   'source': {'host': socket.gethostname(), 'directory': str(source),
                              'checkpoint_id': checkpoint_id,
                              'completion_verified': 'all manifest entries and source generations checked; only bundled files hashed',
                              'optimizer_state_retained': before['optimizer_state_retained'],
                              'files': before['files']}}
        (temporary / MARKER).write_text(json.dumps(receipt, indent=2) + '\n')
        with (temporary / MARKER).open('rb') as handle:
            os.fsync(handle.fileno())
        read_bundle(temporary)
        verify_source(before)
        if destination.exists():
            raise FileExistsError(destination)
        temporary.rename(destination)
        return receipt
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    command = parser.add_subparsers(dest='command', required=True)
    create = command.add_parser('create')
    create.add_argument('--source', required=True)
    create.add_argument('--destination', required=True)
    create.add_argument('--checkpoint-id', required=True)
    verify = command.add_parser('verify')
    verify.add_argument('directory')
    args = parser.parse_args()
    if args.command == 'create':
        result = create_bundle(args.source, args.destination, args.checkpoint_id)
    else:
        path, meta = read_bundle(args.directory)
        result = {'verified': True, 'student': str(path), 'checkpoint_id': meta['checkpoint_id'],
                  'training_resume_supported': False}
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
