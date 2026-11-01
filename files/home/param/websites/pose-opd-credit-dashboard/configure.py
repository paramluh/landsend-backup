#!/usr/bin/env python3
"""Select a run read-only; the refresher picks up the configuration automatically."""
import argparse
import json
from pathlib import Path

from build_stage3 import ROOT, PAGE, atomic_write, read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--name', help='Optional display name')
    args = parser.parse_args()
    run = args.run_dir.expanduser().absolute()
    if not run.is_dir():
        parser.error(f'Run directory does not exist: {run}')
    if not any((run / name).is_file() for name in
               ('identity.json', 'preparation.json', 'metrics.jsonl', 'events-rank0.jsonl')):
        parser.error('Run has no pose training metadata or telemetry')
    path = ROOT / f's3branches_{PAGE}.json'
    config = read_config(path)
    config['run_dir'] = str(run)
    if args.name:
        config.setdefault('overrides', {})['run'] = args.name
    atomic_write(path, json.dumps(config, indent=2) + '\n')
    print(f'Selected {run}; the service will read it on its next refresh (within 60 seconds).')


if __name__ == '__main__':
    main()
