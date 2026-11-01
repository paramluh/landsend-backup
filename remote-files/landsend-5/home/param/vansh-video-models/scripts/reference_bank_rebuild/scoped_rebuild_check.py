"""Bounded checks for isolated controller outputs and scoped checkpoint reuse."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from aggregate_scope import aggregate
from hf_sources import write_json
from measure_upstream import source_fingerprint
import run_rebuild
from upstream import REFERENCE_PREFIX, SELECTOR_FOLDER


def logged_runner(run):
    def start(argv, log_path):
        result = run(argv)
        return (SimpleNamespace(wait=lambda: result.returncode, poll=lambda: result.returncode),
                SimpleNamespace(join=lambda: None))
    return start


class ScopedAggregateTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.sources = self.root / 'sources'
        self.shared = self.root / 'measurements'
        self.output = self.root / 'studio-only/measurements'
        self.tracks = self.root / 'tracks.txt'
        self.track = 'studio-recording/speaker_1'
        self.tracks.write_text(self.track + '\n')
        self.config = {'model': 'unchanged-fixture'}
        folder = self.sources / REFERENCE_PREFIX / self.track / SELECTOR_FOLDER
        folder.mkdir(parents=True)
        self.image = folder / 'front.png'
        self.image.write_bytes(b'fixture-image')
        (folder / 'manifest.json').write_text('{}')
        (folder / 'selected_poses.parquet').write_bytes(b'fixture-keypoints')
        self.checkpoint = self.shared / 'tracks' / (self.track.replace('/', '__') + '.json')
        write_json(self.checkpoint, {'complete': True, 'track': self.track,
            'fingerprint': source_fingerprint(self.track, self.sources, self.config),
            'p14': [{'track': self.track, 'slot': 'front'}], 'p59': [], 'teeth': {'track': self.track, 'ok': False}})
        # An invalid unrelated app checkpoint must never be parsed or awaited.
        (self.shared / 'tracks/app-recording__speaker_1.json').write_text('invalid app checkpoint')
        (self.shared / 'failures.json').write_text('preserve global failure history')

    def aggregate(self):
        with patch('aggregate_scope.measurement_config', return_value=self.config):
            return aggregate(self.tracks, self.sources, self.shared, self.output, workers=2)

    def test_only_scoped_tracks_are_validated_and_shared_outputs_are_unchanged(self):
        before = {path: path.read_bytes() for path in self.shared.rglob('*') if path.is_file()}
        report = self.aggregate()
        self.assertTrue(report['complete'])
        self.assertEqual(report['tracks'], 1)
        self.assertEqual(set(report['checkpoint_sha256']), {self.track})
        self.assertEqual(pd.read_parquet(self.output / 'p14.parquet').track.tolist(), [self.track])
        self.assertEqual(before, {path: path.read_bytes() for path in self.shared.rglob('*') if path.is_file()})

    def test_changed_source_cannot_reuse_measurement_checkpoint(self):
        self.image.write_bytes(b'changed-image-content')
        with self.assertRaisesRegex(RuntimeError, '1 scoped measurement checkpoints'):
            self.aggregate()
        self.assertFalse((self.output / 'complete.json').exists())
        self.assertEqual(json.loads((self.output / 'failures.json').read_text())[0]['track'], self.track)

    def test_wrong_track_rows_are_rejected(self):
        value = json.loads(self.checkpoint.read_text())
        value['p14'][0]['track'] = 'app-recording/speaker_1'
        write_json(self.checkpoint, value)
        with self.assertRaises(RuntimeError):
            self.aggregate()


class ScopedControllerTests(unittest.TestCase):
    def test_historical_controller_reuses_only_verified_scans_and_copies_policy_provenance(self):
        for fail_import in (False, True):
            with self.subTest(fail_import=fail_import), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                csvs = root / 'studio-only/csvs'
                work = root / 'studio-training-policy'
                cache = root / 'studio-only/video-scan'
                csvs.mkdir(parents=True)
                write_json(csvs / 'studio-scope.json', {'scope': 'studio-only'})
                write_json(root / 'prefetch-downloads.json', {'jobs': []})
                for name in ('metadata', 'image'):
                    write_json(root / f'hf-upstream-{name}-receipts.jsonl.summary.json', {'failures': []})
                commands = []

                def fake_command(argv, **kwargs):
                    commands.append(argv)
                    script = Path(argv[2]).name
                    if script == 'upstream.py' and argv[3] == 'inventory':
                        write_json(work / 'plan/inventory.json', {'actors': 543, 'selection_policy': 'strict'})
                    if script == 'reuse_completed_scan.py' and argv[3] == 'prepare' and fail_import:
                        return SimpleNamespace(returncode=1)
                    if script == 'finalize_bank.py' and argv[3] == 'finalize':
                        bank = Path(argv[argv.index('--output-dir') + 1])
                        selection = bank / 'selection.parquet'
                        selection.write_bytes(b'historical-fixture-selection')
                        write_json(bank / 'coverage.json', {'selection_sha256': hashlib.sha256(selection.read_bytes()).hexdigest(),
                                                          'sources': {'errors': []}, 'latents': {'errors': []}, 'complete': True})
                    return SimpleNamespace(returncode=0)

                argv = ['run_rebuild.py', '--root', str(root), '--work-subdir', 'studio-training-policy',
                        '--csv-root', str(csvs), '--expected-actors', '543', '--selection-policy', 'historical-studio',
                        '--reuse-video-scan', str(cache), '--scan-workers', '96']
                with patch.object(sys, 'argv', argv), patch('run_rebuild.start_logged', side_effect=logged_runner(fake_command)):
                    if fail_import:
                        with self.assertRaisesRegex(RuntimeError, 'prepare-completed-video-scan failed'):
                            run_rebuild.main()
                    else:
                        run_rebuild.main()
                scripts = [Path(command[2]).name for command in commands]
                self.assertEqual(scripts.count('historical_studio_select.py'), 1)
                self.assertNotIn('download_and_scan.py', scripts)
                self.assertNotIn('hf_sources.py', scripts)
                self.assertFalse(any(Path(c[2]).name == 'upstream.py' and c[3] in ('select', 'select-beatles') for c in commands))
                inventory = json.loads((work / 'plan/inventory.json').read_text())
                self.assertIn('historical', inventory['selection_policy'])
                state = json.loads((work / 'build-state.json').read_text())
                self.assertEqual(state['selection_policy'], 'historical-studio')
                if fail_import:
                    self.assertNotIn('distributed_scan.py', scripts)
                    self.assertNotIn('finalize_bank.py', scripts)
                    self.assertFalse(state['complete'])
                    continue
                prepare_index = next(i for i, c in enumerate(commands) if Path(c[2]).name == 'reuse_completed_scan.py' and c[3] == 'prepare')
                verify_index = next(i for i, c in enumerate(commands) if Path(c[2]).name == 'reuse_completed_scan.py' and c[3] == 'verify')
                scan_index = scripts.index('distributed_scan.py')
                self.assertLess(prepare_index, scan_index)
                self.assertLess(scan_index, verify_index)
                self.assertEqual(commands[prepare_index][commands[prepare_index].index('--source-root') + 1], str(root / 'sources'))
                scan = commands[scan_index]
                self.assertEqual(len(scan[scan.index('--nodes') + 1:scan.index('--allow-incomplete')]), 5)
                final = next(c for c in commands if Path(c[2]).name == 'finalize_bank.py' and c[3] == 'finalize')
                for name in ('RECIPE-historical-studio.md', 'selection-policy.json', 'archived-studio-select.py', 'archived-beatles-plan.py'):
                    self.assertIn(str(work / 'selection' / name), final)
                self.assertIn(str(work / 'video-scan/scan-reuse-provenance.json'), final)
                self.assertIn(str(work / 'video-scan/historical-video-policy.json'), final)
                self.assertIn(str(work / 'video-scan/archived-studio-select-v2.py'), final)
                self.assertIn(str(work / 'video-scan/historical-video-decode.json'), final)
                self.assertNotIn(str(Path('/mnt/paramfs/vansh/richmond-7/reference-bank/RECIPE.md')), final)
                self.assertEqual(scripts.count('historical_video_select.py'), 1)
                self.assertFalse(any(Path(c[2]).name == 'video_pipeline.py' and c[3] == 'select' for c in commands))
                decode_index = next(i for i, c in enumerate(commands) if Path(c[2]).name == 'video_pipeline.py' and c[3] == 'decode')
                self.assertLess(decode_index, scripts.index('validate_historical_decode.py'))
                self.assertLess(scripts.index('validate_historical_decode.py'), scripts.index('finalize_bank.py'))

    def test_scope_path_must_be_separate_and_cannot_escape_shared_root(self):
        root = Path('/tmp/scoped-test')
        self.assertEqual(run_rebuild.work_root(root, 'studio-only'), root / 'studio-only')
        for invalid in ('../elsewhere', '/elsewhere', '.'):
            with self.assertRaises(ValueError):
                run_rebuild.work_root(root, invalid)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            old = root / 'old/video-scan'
            old.mkdir(parents=True)
            write_json(old / 'tracks.json', {'preserve': True})
            with patch.object(sys, 'argv', ['run_rebuild.py', '--root', str(root), '--work-subdir', 'old',
                                           '--training-csv', str(root / 'training.csv'),
                                           '--rules-root', str(root / 'spec'),
                                           '--reuse-video-scan', str(old)]), self.assertRaises(SystemExit):
                run_rebuild.main()
            self.assertEqual(json.loads((old / 'tracks.json').read_text()), {'preserve': True})

    def test_controller_propagates_scope_and_never_waits_on_global_app_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            work = root / 'studio-only'
            csvs = work / 'csvs'
            csvs.mkdir(parents=True)
            write_json(csvs / 'studio-scope.json', {'scope': 'studio-only'})
            write_json(root / 'build-state.json', {'preserve': 'earlier build'})
            write_json(root / 'measurements/watch-failures.json', ['unrelated app validation failure'])
            write_json(root / 'prefetch-downloads.json', {'jobs': []})
            for name in ('metadata', 'image'):
                write_json(root / f'hf-upstream-{name}-receipts.jsonl.summary.json', {'failures': []})
            commands = []
            def fake_command(argv, **kwargs):
                commands.append(argv)
                script = Path(argv[2]).name
                if script == 'upstream.py' and argv[3] == 'inventory':
                    plan = Path(argv[argv.index('--output-dir') + 1])
                    write_json(plan / 'inventory.json', {'actors': 543})
                if script == 'finalize_bank.py' and argv[3] == 'finalize':
                    bank = Path(argv[argv.index('--output-dir') + 1])
                    selection = bank / 'selection.parquet'
                    selection.write_bytes(b'fixture-selection')
                    write_json(bank / 'coverage.json', {'selection_sha256': hashlib.sha256(selection.read_bytes()).hexdigest(),
                                                      'sources': {'errors': []}, 'latents': {'errors': []}, 'complete': True})
                return SimpleNamespace(returncode=0)
            argv = ['run_rebuild.py', '--root', str(root), '--work-subdir', 'studio-only',
                    '--csv-root', str(csvs), '--expected-actors', '543', '--scan-workers', '96',
                    '--training-csv', str(root / 'training.csv'), '--rules-root', str(root / 'spec')]
            with patch.object(sys, 'argv', argv), patch('run_rebuild.start_logged', side_effect=logged_runner(fake_command)), \
                    patch('run_rebuild.time.sleep') as sleep:
                run_rebuild.main()
            sleep.assert_not_called()
            self.assertEqual(json.loads((root / 'build-state.json').read_text()), {'preserve': 'earlier build'})
            self.assertTrue(json.loads((work / 'build-state.json').read_text())['complete'])
            by_stage = {Path(command[2]).name + ':' + command[3]: command for command in commands}
            aggregate_command = next(command for command in commands if Path(command[2]).name == 'aggregate_scope.py')
            self.assertEqual(aggregate_command[aggregate_command.index('--tracks') + 1], str(work / 'plan/tracks.txt'))
            self.assertEqual(aggregate_command[aggregate_command.index('--checkpoint-root') + 1], str(root / 'measurements'))
            self.assertFalse(any(Path(command[2]).name == 'measure_upstream.py' for command in commands))
            for key in ('upstream.py:inventory', 'parallel_plan.py:--csv-root', 'video_pipeline.py:targets', 'finalize_bank.py:finalize'):
                command = by_stage[key]
                self.assertEqual(command[command.index('--csv-root') + 1], str(csvs))
            planner = by_stage['parallel_plan.py:--csv-root']
            self.assertEqual(planner[planner.index('--workers') + 1], '64')
            decoder = by_stage['video_pipeline.py:decode']
            self.assertEqual(decoder[decoder.index('--workers') + 1], '64')
            final = by_stage['finalize_bank.py:finalize']
            self.assertEqual(final[final.index('--expected-actors') + 1], '543')
            self.assertEqual(final[final.index('--video-index') + 1], str(root / 'hf-video-index.json'))
            self.assertIn(str(csvs / 'studio-scope.json'), final)
            overlap = next(command for command in commands if Path(command[2]).name == 'download_and_scan.py')
            self.assertEqual(overlap[overlap.index('--workers') + 1], '96')
            self.assertEqual(overlap[overlap.index('--source-root') + 1], str(root / 'sources'))
            encode = by_stage['distributed_encode.py:run']
            self.assertEqual(encode[encode.index('--gpus-per-node') + 1], '8')
            self.assertEqual(len(encode[encode.index('--nodes') + 1:]), 5)
            precheck, postcheck = [c for c in commands if Path(c[2]).name == 'strict_quality_check.py']
            self.assertLess(commands.index(precheck), commands.index(encode))
            self.assertLess(commands.index(encode), commands.index(postcheck))
            self.assertLess(commands.index(postcheck), commands.index(final))
            self.assertIn('--bank-root', postcheck)
            for check in (precheck, postcheck):
                self.assertEqual(check[check.index('--training-csv') + 1], str(root / 'training.csv'))
                self.assertEqual(check[check.index('--rules-root') + 1], str(root / 'spec'))
                self.assertEqual(check[check.index('--workers') + 1], '64')
                self.assertIn(check[check.index('--report') + 1], final)
            self.assertIn(str(root / 'spec/sinks.md'), final)
            self.assertIn(str(root / 'spec/RECIPE.md'), final)

    def test_failed_quality_check_prevents_encoding_or_publication(self):
        for failed_check in (1, 2):
            with self.subTest(failed_check=failed_check), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                work, csvs = root / 'studio', root / 'csvs'
                csvs.mkdir()
                write_json(root / 'prefetch-downloads.json', {'jobs': []})
                for name in ('metadata', 'image'):
                    write_json(root / f'hf-upstream-{name}-receipts.jsonl.summary.json', {'failures': []})
                commands, checks = [], []

                def fake_command(argv):
                    commands.append(argv)
                    script = Path(argv[2]).name
                    if script == 'upstream.py' and argv[3] == 'inventory':
                        write_json(work / 'plan/inventory.json', {'actors': 543})
                    if script == 'strict_quality_check.py':
                        checks.append(argv)
                        if len(checks) == failed_check:
                            return SimpleNamespace(returncode=1)
                    return SimpleNamespace(returncode=0)

                argv = ['run_rebuild.py', '--root', str(root), '--work-subdir', 'studio',
                        '--csv-root', str(csvs), '--expected-actors', '543', '--training-csv', str(root / 'training.csv'),
                        '--rules-root', str(root / 'spec')]
                with patch.object(sys, 'argv', argv), patch('run_rebuild.start_logged', side_effect=logged_runner(fake_command)):
                    with self.assertRaisesRegex(RuntimeError, 'strict-quality-.* failed'):
                        run_rebuild.main()
                self.assertFalse(any(Path(c[2]).name == 'finalize_bank.py' and c[3] == 'finalize' for c in commands))
                if failed_check == 1:
                    self.assertFalse(any(Path(c[2]).name == 'distributed_encode.py' for c in commands))
                self.assertFalse(json.loads((work / 'build-state.json').read_text())['complete'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
