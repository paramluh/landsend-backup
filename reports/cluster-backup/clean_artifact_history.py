"""Remove temporary unsafe revisions from only this task's new artifact dataset.

Run after all initial artifact/dataset writers have settled. Existing home Git
repositories and checkpoint repositories are deliberately outside this scope.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import RevisionNotFoundError

ROOT = Path('/home/param/outputs/landsend-backup-20261031')
REPO = 'InternalCan/landsend-artifacts-backup-param'
OUT = ROOT / 'artifact-history-cleanup.json'
WORK = ROOT / 'artifact-history-cleanup'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(temp, path)


def inventory(api, revision):
    files = {}
    for entry in api.list_repo_tree(REPO, repo_type='dataset', revision=revision, recursive=True):
        if not hasattr(entry, 'blob_id'):
            continue
        lfs = entry.lfs
        sha = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
        files[entry.path] = {'bytes': entry.size, 'blob_id': entry.blob_id, 'sha256': sha}
    return files


def fingerprint(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--squash', action='store_true', help='Perform cleanup after guards pass')
    args = parser.parse_args()
    inputs = read(ROOT / 'archive-coverage/HISTORY_SCRUB_INPUT.json')
    assert inputs['repo_id'] == REPO and inputs['current_replacements_verified']
    bundles = read(ROOT / 'archive-coverage/BUNDLE_AUDIT_VERIFIED.json')
    assert bundles['state'] == 'verified'
    assert bundles['all_selected_and_directly_nested_bundles_absent_known_credential']
    opaque = read(ROOT / 'archive-coverage/FIVE_SOURCE_CONTAINERS_VERIFIED.json')
    assert opaque['state'] == 'verified_current_head_history_scrub_required'
    assert opaque['all_five_nested_credential_scans_clean']
    subprocess.run([sys.executable, str(ROOT / 'orphans/verify_initial_archives.py')], check=True, timeout=300)
    initial = read(ROOT / 'INITIAL_ARTIFACT_VERIFICATION.json')
    assert initial['state'] == 'verified' and not initial['failures']
    assert len(initial['nodes']) == 7 and all(not node['workers'] for node in initial['nodes'])
    api = HfApi()
    info = api.repo_info(REPO, repo_type='dataset')
    assert info.private, 'Backup dataset is not private'
    assert info.created_at >= dt.datetime(2026, 11, 1, 0, 0, tzinfo=dt.timezone.utc), 'Not this task\'s newly-created repository'
    refs = api.list_repo_refs(REPO, repo_type='dataset', include_pull_requests=True)
    assert [branch.name for branch in refs.branches] == ['main'] and not refs.tags
    assert not refs.converts and not refs.pull_requests, 'Unexpected additional refs'
    old_revision = info.sha
    assert initial['revision'] == old_revision, 'Writer committed after initial verification'
    before = inventory(api, old_revision)
    assert len(opaque['node0']) == 3
    for source in opaque['node0'].values():
        assert source['state'] == 'verified' and source['nested_credential_matches_after'] == 0
        assert before[source['part']]['sha256'] == source['archive_sha256'], source['part']
    replacements = []
    for repair in read(ROOT / 'archive-coverage/OPAQUE_REPAIRS_VERIFIED.json'):
        for part in repair['receipt']['parts']:
            path = repair['host'] + '/artifacts/' + part['part']
            assert part['state'] == 'verified' and part['nested_credential_matches_after'] == 0
            assert before[path]['sha256'] == part['sha256'], path
            replacements.append({'path': path, 'sha256': part['sha256']})
    assert {part['path'] for part in replacements} == set(inputs['bad_parts'])
    assert api.repo_info(REPO, repo_type='dataset').sha == old_revision, 'Writer still active during preflight'
    write(WORK / 'before-tree.json', {'revision': old_revision, 'files': before})
    preflight = {'state': 'preflight_verified', 'repo_id': REPO, 'repo_type': 'dataset',
                 'old_revision': old_revision, 'created_at': info.created_at.isoformat(),
                 'files': len(before), 'tree_sha256': fingerprint(before),
                 'affected_parts': replacements, 'time': time.time()}
    write(WORK / 'preflight.json', preflight)
    if not args.squash:
        print(json.dumps(preflight))
        return
    # This mutation is restricted to the new task-owned dataset and leaves its
    # complete current file tree unchanged. Original home repositories are untouched.
    api.super_squash_history(REPO, repo_type='dataset', branch='main',
                             commit_message='Verified sanitized cluster backup')
    clean = api.repo_info(REPO, repo_type='dataset')
    assert clean.private
    after = inventory(api, clean.sha)
    assert after == before, 'Current file tree changed during history cleanup'
    history = api.list_repo_commits(REPO, repo_type='dataset')
    assert len(history) == 1 and history[0].commit_id == clean.sha, 'Unexpected history after squash'
    unreachable = []
    for revision in inputs['known_bad_revisions']:
        try:
            api.repo_info(REPO, repo_type='dataset', revision=revision)
        except RevisionNotFoundError:
            unreachable.append(revision)
        else:
            raise RuntimeError('An unsafe prior revision remains reachable: ' + revision)
    receipt = {**preflight, 'state': 'verified', 'clean_revision': clean.sha,
               'known_bad_revisions_unreachable': len(unreachable) == len(inputs['known_bad_revisions']),
               'unreachable_revisions': unreachable, 'current_file_tree_unchanged': True,
               'verified_at_unix': time.time(),
               'scope': 'Only new artifact dataset; original home Git and checkpoint histories preserved'}
    write(WORK / 'after-tree.json', {'revision': clean.sha, 'files': after})
    write(OUT, receipt)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
