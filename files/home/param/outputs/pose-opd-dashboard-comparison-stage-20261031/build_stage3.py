#!/usr/bin/env python3
"""Build the isolated pose-credit monitor; never imports training code."""
import argparse
import hashlib
import json
import os
from pathlib import Path

from pose_doc import build_doc

ROOT = Path(__file__).resolve().parent
PAGE = 'pose_opd_credit'


def atomic_write(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(content, encoding='utf-8')
    temporary.replace(path)


def read_config(path):
    config = json.loads(Path(path).read_text())
    if not isinstance(config, dict):
        raise ValueError('Dashboard configuration must be an object')
    return config


def build(config_path=None, output=None):
    config = read_config(config_path or ROOT / f's3branches_{PAGE}.json')
    out = Path(output or ROOT / 'dist')
    run_dir = config.get('run_dir')
    if run_dir:
        run_dir = Path(os.path.expandvars(run_dir)).expanduser()
        if not run_dir.is_absolute():
            raise ValueError('run_dir must be absolute')
    doc = build_doc(run_dir, config.get('overrides'))
    doc['title'] = config.get('title', 'Pose OPD · credit assignment')
    doc['source_branch'] = config.get('source_branch')
    doc['run_configured'] = bool(run_dir)
    template = (ROOT / 'pose.tpl.html').read_text()
    assets = {name: (ROOT / name).read_text() for name in ('dashboard.js', 'dashboard.css')}
    version = hashlib.sha256((template + ''.join(assets.values())).encode()).hexdigest()[:16]
    payload = json.dumps(doc, separators=(',', ':'), allow_nan=False)
    embedded = payload.replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    html = template.replace('__DATA__', embedded).replace('__VERSION__', version)
    for name, content in assets.items():
        atomic_write(out / name, content)
    atomic_write(out / f's3live_{PAGE}.json', payload + '\n')
    atomic_write(out / f'stage3_{PAGE}.html', html)
    atomic_write(out / f'stage3_{PAGE}_artifact.html', html)
    atomic_write(out / 'index.html', html)
    atomic_write(out / f'ver_s3_{PAGE}.txt', version + '\n')
    return doc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    doc = build(args.config, args.output)
    print(json.dumps({'state': doc.get('state'), 'run': doc.get('run'), 'update': doc.get('step')}))


if __name__ == '__main__':
    main()
