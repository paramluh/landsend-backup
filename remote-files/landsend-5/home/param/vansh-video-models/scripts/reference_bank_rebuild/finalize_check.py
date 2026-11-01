"""Small CPU checks for archived pixel parity and final bank integrity failures."""
import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import tempfile
from threading import Event, Lock
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from uuid import UUID

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

import finalize_bank
from encode_bank import VAE_SHA256, encoder_fingerprint, encoding_checkpoint_matches, prepare, sha256
from finalize_bank import (HF_PREFIX, completion_evidence, coverage, merge_selections, validate_decode_qa,
                           validate_latents, validate_scan_sources, validate_sources)
from upstream import BEATLES_HELD_ROWS, REFERENCE_PREFIX, SELECTOR_FOLDER, SLOTS


def archived_function(path, name, namespace):
    tree = ast.parse(Path(path).read_text())
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def reference(prop=1, index=0, frame=1, bank='bank'):
    slot = SLOTS[prop - 1]
    return {'uuid': str(UUID(int=1)), 'bank': bank, 'property': prop, 'slot': slot, 'index': index,
            'iid': 100, 'origin': 'ref-image', 'recording_id': 'a' * 40, 'speaker': 0, 'session_id': 'session',
            'frame_index': frame, 'time_s': frame / 25, 'cx0': 20, 'cy0': 30, 'cw': 256, 'ch': 480,
            'source': f'{REFERENCE_PREFIX}/{"a" * 40}/speaker_0/{SELECTOR_FOLDER}/{slot}.png',
            'source_sha256': '0' * 64}


class SourceHashChecks(unittest.TestCase):
    def fixture(self, root):
        rows = [reference(), reference(), reference(prop=2, frame=25), reference(),
                {**reference(bank='beatles-eval'), 'recording_id': 'b' * 40},
                {**reference(bank='beatles-eval'), 'recording_id': 'c' * 40}]
        for index, row in enumerate(rows):
            row['iid'] = index
            row['source'] = HF_PREFIX + f'{REFERENCE_PREFIX}/{row["recording_id"]}/speaker_0/{SELECTOR_FOLDER}/{row["slot"]}.png'
        rows[1]['session_id'] = 'wrong-session'
        rows[3].update(origin='video-scan', image_path=str(root / 'decoded.png'),
                       decoded_image_sha256='0' * 64)
        paths = [Path(row['image_path']) if row['origin'] == 'video-scan'
                 else root / row['source'][len(HF_PREFIX):] for row in rows]
        for index in (0, 3, 4, 5):
            paths[index].parent.mkdir(parents=True, exist_ok=True)
            paths[index].write_bytes(f'original-image-{index}'.encode())
            rows[index]['decoded_image_sha256' if index == 3 else 'source_sha256'] = sha256(paths[index])
        rows[5]['source_sha256'] = '0' * 64
        identities = [{key: row[key] for key in ('uuid', 'recording_id', 'speaker', 'session_id')}
                      for row in (rows[0], rows[4], rows[5])]
        for row in identities:
            row.update(track=f'{row["recording_id"]}/speaker_0', cohort='train')
        corpus = SimpleNamespace(tracks=identities[:1], ranges_eval={identities[0]['track']: [(1, 1)]},
                                 ranges_blackbands={identities[2]['track']: [(0, 1)]},
                                 ranges_lighting={}, ranges_excess={})
        return pd.DataFrame(rows), corpus, identities[1:], {'held_recordings': ['b' * 40]}, paths

    def test_parallel_matches_serial_with_ordered_errors_and_unique_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected, corpus, beatles, holdout, paths = self.fixture(root)
            with patch.object(finalize_bank, 'ThreadPoolExecutor',
                              side_effect=lambda **kwargs: ThreadPoolExecutor(max_workers=1)), \
                 patch.object(finalize_bank, 'sha256', wraps=sha256) as serial_hash:
                serial = validate_sources(selected, root, corpus, beatles, holdout)
            self.assertEqual(serial_hash.call_count, 4)  # One missing path; repeated path hashes once.
            expected = [f'iid=1: source identity differs from corpus CSV',
                        f'iid=1: source image absent or checksum mismatch: {paths[1]}',
                        'iid=2: prohibited reference time',
                        f'iid=2: source image absent or checksum mismatch: {paths[2]}',
                        'iid=4: prohibited Beatles donor',
                        'iid=5: prohibited Beatles donor',
                        f'iid=5: source image absent or checksum mismatch: {paths[5]}']
            self.assertEqual(serial, {'checked_images': 5, 'errors': expected})
            finished, lock, last_finished = [], Lock(), Event()

            def out_of_order_hash(path):
                if path == paths[0]:
                    self.assertTrue(last_finished.wait(3), 'Distinct paths were not hashed concurrently')
                digest = sha256(path)
                with lock:
                    finished.append(path)
                if path == paths[-1]:
                    last_finished.set()
                return digest

            with patch.object(finalize_bank, 'sha256', side_effect=out_of_order_hash):
                parallel = validate_sources(selected, root, corpus, beatles, holdout)
            self.assertEqual(parallel, serial)
            self.assertEqual(Counter(finished), Counter({path: 1 for path in paths if path != paths[2]}))
            self.assertLess(finished.index(paths[-1]), finished.index(paths[0]))

    def test_changed_image_is_rehashed_on_next_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected, corpus, beatles, holdout, paths = self.fixture(root)
            selected = selected.iloc[:1]
            self.assertEqual(validate_sources(selected, root, corpus, beatles, holdout),
                             {'checked_images': 1, 'errors': []})
            paths[0].write_bytes(b'changed-after-first-validation')
            report = validate_sources(selected, root, corpus, beatles, holdout)
            self.assertEqual(report, {'checked_images': 1, 'errors': [
                f'iid=0: source image absent or checksum mismatch: {paths[0]}']})

    def test_unreadable_image_still_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected, corpus, beatles, holdout, paths = self.fixture(root)

            def denied(path):
                if path == paths[0]:
                    raise PermissionError('source cannot be read')
                return sha256(path)

            with patch.object(finalize_bank, 'sha256', side_effect=denied):
                with self.assertRaisesRegex(PermissionError, 'source cannot be read'):
                    validate_sources(selected, root, corpus, beatles, holdout)


class FinalizeChecks(unittest.TestCase):
    def test_changed_encoder_fingerprint_rejects_resume_and_finalization(self):
        selected = merge_selections(pd.DataFrame([reference()]), pd.DataFrame(), pd.DataFrame())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            implementation = root / 'implementation.py'
            implementation.write_text('normalization = "original"\n')
            actor = selected.uuid.iloc[0]
            destination = root / 'bank' / actor / 'property_1/latents.safetensors'
            destination.parent.mkdir(parents=True)
            save_file({'video': torch.zeros((1, 48, 30, 16), dtype=torch.bfloat16)}, destination,
                      metadata={'uuid': actor, 'property': '1', 'slot': 'front', 'n': '1'})
            receipt = root / 'encoding-checkpoints' / f'bank__{actor}__1.json'
            receipt.parent.mkdir()
            with patch('encode_bank.encoder_source_files', return_value={'models/muon/constants.py': implementation}):
                original = encoder_fingerprint(module=object())
                info = {'sha256': sha256(destination), 'selection_sha256': '1' * 64,
                        'vae_sha256': VAE_SHA256, 'n': 1, 'encoder_fingerprint': original}
                receipt.write_text(json.dumps(info))
                self.assertTrue(encoding_checkpoint_matches(receipt, destination, '1' * 64, original, 1))
                self.assertFalse(validate_latents(selected, root, '1' * 64, original)[0]['errors'])
                implementation.write_text('normalization = "changed"\n')
                changed = encoder_fingerprint(module=object())
                self.assertNotEqual(original['sha256'], changed['sha256'])
                self.assertFalse(encoding_checkpoint_matches(receipt, destination, '1' * 64, changed, 1))
                self.assertIn('implementation', validate_latents(selected, root, '1' * 64, changed)[0]['errors'][0]['error'])
                implementation.write_text('normalization = "original"\n')
                with patch('encode_bank.PIL.__version__', 'changed-version'):
                    changed_version = encoder_fingerprint(module=object())
                self.assertNotEqual(original['sha256'], changed_version['sha256'])
                self.assertFalse(encoding_checkpoint_matches(receipt, destination, '1' * 64, changed_version, 1))
                # Legacy smoke/build receipts cannot silently inherit the new implementation identity.
                del info['encoder_fingerprint']
                receipt.write_text(json.dumps(info))
                self.assertFalse(encoding_checkpoint_matches(receipt, destination, '1' * 64, original, 1))
                self.assertIn('implementation', validate_latents(selected, root, '1' * 64, original)[0]['errors'][0]['error'])

    def test_decode_qa_must_pass_for_exact_selection_and_vae(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = merge_selections(pd.DataFrame([reference()]), pd.DataFrame(), pd.DataFrame())
            fingerprint = {'sha256': '3' * 64}
            with self.assertRaisesRegex(ValueError, 'missing or unreadable'):
                validate_decode_qa(root, '1' * 64, fingerprint, selected)
            path = root / 'qa/decode-report.json'
            path.parent.mkdir()
            latent = root / 'bank' / selected.uuid.iloc[0] / 'property_1/latents.safetensors'
            latent.parent.mkdir(parents=True)
            latent.write_bytes(b'original-tensor')
            sample = {key: selected.iloc[0][key] for key in ('bank', 'uuid', 'origin')}
            sample.update(iid=0, index=0, property=1, PSNR_dB=39.0, latent_sha256=sha256(latent))
            valid = {'passed': True, 'selection_sha256': '1' * 64, 'vae_sha256': VAE_SHA256,
                     'encoder_fingerprint': fingerprint, 'images': 1, 'samples': [sample]}
            for change in ({'passed': False}, {'passed': 1}, {'selection_sha256': '2' * 64},
                           {'vae_sha256': '0' * 64}, {'encoder_fingerprint': {'sha256': '4' * 64}},
                           {'images': 0}, {'samples': []}, {'samples': [{**sample, 'index': 1}]},
                           {'samples': [{**sample, 'latent_sha256': '0' * 64}]}):
                with self.subTest(change=change):
                    path.write_text(json.dumps({**valid, **change}))
                    with self.assertRaises(ValueError):
                        validate_decode_qa(root, '1' * 64, fingerprint, selected)
            path.write_text(json.dumps(valid))
            self.assertEqual(validate_decode_qa(root, '1' * 64, fingerprint, selected), valid)
            latent.write_bytes(b're-encoded-tensor')
            with self.assertRaisesRegex(ValueError, 'stale'):
                validate_decode_qa(root, '1' * 64, fingerprint, selected)

    def test_marker_only_published_after_complete_finalization_with_honest_source_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output, sources, scan, csvs = (root / name for name in ('output', 'sources', 'scan', 'csvs'))
            for path in (output, sources, scan, csvs):
                path.mkdir()
            original = reference()
            source = sources / original['source']
            source.parent.mkdir(parents=True)
            source.write_bytes(b'verified-source-image')
            original['source_sha256'] = sha256(source)
            selected = merge_selections(pd.DataFrame([original]), pd.DataFrame(), pd.DataFrame())
            selection = output / 'selection.parquet'
            selected.to_parquet(selection, index=False)
            actor = original['uuid']
            actors = root / 'actors.csv'
            pd.DataFrame([{'uuid': actor, 'cohort': 'train', 'eligible_track_count': 1}]).to_csv(actors, index=False)
            pd.DataFrame([{'uuid': str(UUID(int=2)), 'session_id': 'held-session', 'recording_id': 'held',
                           'speaker': 0, 'chunk': number} for number in range(40)]).to_csv(csvs / 'aloe-eval-beatles.csv', index=False)
            holdout, diagnostics = root / 'holdout.json', root / 'diagnostics.json'
            holdout.write_text(json.dumps({'held_rows': list(BEATLES_HELD_ROWS), 'held_recordings': ['held'], 'selected_images': 0}))
            diagnostics.write_text(json.dumps({'source_hashes': {}, 'fatal_input_issues': 0,
                                               'incomplete_source_tracks': ['known-HF-gap']}))
            summaries = {'targets-summary.json': {'complete': True, 'upstream_sha256': '2' * 64},
                         'scan-summary.json': {'complete': True, 'available_targets_complete': True, 'missing_sources': 0, 'failures': 0},
                         'selection-summary.json': {'complete': True, 'missing_chunks': 0, 'rows': 0},
                         'decode-summary.json': {'complete': True, 'rows': 0}}
            for name, value in summaries.items():
                (scan / name).write_text(json.dumps(value))
            (output / 'selection-merge.json').write_text(json.dumps({'selection_sha256': sha256(selection),
                                                                    'inputs': {'upstream': {'sha256': '2' * 64}}}))
            fingerprint = {'sha256': '3' * 64, 'inputs': {'files': {}}}
            latent = output / 'bank' / actor / 'property_1/latents.safetensors'
            latent.parent.mkdir(parents=True)
            save_file({'video': torch.zeros((1, 48, 30, 16), dtype=torch.bfloat16)}, latent,
                      metadata={'uuid': actor, 'property': '1', 'slot': 'front', 'n': '1'})
            receipt = output / 'encoding-checkpoints' / f'bank__{actor}__1.json'
            receipt.parent.mkdir()
            receipt.write_text(json.dumps({'sha256': sha256(latent), 'selection_sha256': sha256(selection),
                                           'vae_sha256': VAE_SHA256, 'encoder_fingerprint': fingerprint, 'n': 1}))
            (output / 'qa').mkdir()
            (output / 'qa/decode-report.json').write_text(json.dumps({
                'passed': True, 'selection_sha256': sha256(selection), 'vae_sha256': VAE_SHA256,
                'encoder_fingerprint': fingerprint, 'images': 1, 'samples': [
                    {'bank': 'bank', 'uuid': actor, 'property': 1, 'index': 0, 'iid': 0,
                     'origin': 'ref-image', 'latent_sha256': sha256(latent), 'PSNR_dB': 39}]}))
            identity = {key: original[key] for key in ('uuid', 'recording_id', 'speaker', 'session_id')}
            identity.update(track=f'{original["recording_id"]}/speaker_0', cohort='train')
            corpus = SimpleNamespace(actors=[{'uuid': actor, 'cohort': 'train', 'eligible_track_count': 1}],
                                     tracks=[identity], source_hashes={}, ranges_eval={}, ranges_blackbands={},
                                     ranges_lighting={}, ranges_excess={})
            args = SimpleNamespace(output_dir=output, selection=selection, actors=actors, csv_root=csvs,
                                   blackbands_csv=root / 'unused', expected_actors=1, beatles_holdout=holdout,
                                   source_diagnostics=diagnostics, source_root=sources, scan_dir=scan, document=[])
            marker = output / finalize_bank.VALIDATION_MARKER
            marker.write_text('{"artifact_valid": true}')
            with patch.object(finalize_bank, 'load_corpus', return_value=corpus), \
                 patch.object(finalize_bank, 'load_encoder_module', return_value=object()), \
                 patch.object(finalize_bank, 'encoder_fingerprint', return_value=fingerprint), \
                 patch.object(finalize_bank, 'encoder_source_files', return_value={}):
                with patch.object(finalize_bank, 'copy_provenance', side_effect=OSError('interrupted publication')):
                    with self.assertRaisesRegex(OSError, 'interrupted publication'):
                        finalize_bank.finalize(args)
                self.assertTrue((output / 'coverage.json').is_file())
                self.assertFalse(marker.exists())
                self.assertFalse(finalize_bank.finalize(args))  # source coverage remains incomplete
            saved = json.loads(marker.read_text())
            self.assertTrue(saved['artifact_valid'])
            self.assertTrue(saved['available_work_complete'])
            self.assertFalse(saved['source_complete'])
            self.assertEqual(saved['coverage_sha256'], sha256(output / 'coverage.json'))
            self.assertEqual(saved['manifest_sha256'], sha256(output / 'SHA256SUMS'))
            self.assertNotIn(finalize_bank.VALIDATION_MARKER, (output / 'SHA256SUMS').read_text())

    def test_scanned_source_bound_to_corpus_track_and_hf_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            track = 'a' * 40 + '/speaker_0'
            identity = {key: reference()[key] for key in ('uuid', 'session_id', 'recording_id', 'speaker')}
            identity.update(track=track)
            corpus = SimpleNamespace(tracks=[identity], source_hashes={'all.csv': '2' * 64})
            video = {'path': f'reference-videos-selected-actors/{"a" * 40}/session/speaker_0/video/v1/full.mp4',
                     'size': 1234, 'xet_hash': '3' * 64}
            source = HF_PREFIX + video['path']
            planned = {**identity, 'bank': 'bank', 'source': source, 'video_source_metadata': video}
            index = root / 'hf-video-index.json'
            index.write_text(json.dumps({track: [video]}))
            targets = root / 'targets-summary.json'
            targets.write_text(json.dumps({'video_index_sha256': sha256(index),
                                            'corpus_source_hashes': corpus.source_hashes}))

            def save_plan(value):
                (root / 'tracks.json').write_text(json.dumps([value]))
                (root / 'scan-summary.json').write_text(json.dumps({'plan_sha256': {
                    name: sha256(root / name) for name in ('tracks.json', 'targets-summary.json')}}))

            save_plan(planned)
            selected = pd.DataFrame([{**reference(), 'origin': 'video-scan', 'source': source + '#frame=1'}])
            self.assertFalse(validate_scan_sources(selected, root, corpus, index)['errors'])
            # A valid-looking URI/hash and even a matching actor cannot allow another track's video.
            wrong_source = source.replace('a' * 40, 'b' * 40)
            for changes in ({'source': wrong_source + '#frame=1'}, {'source': source + '#frame=2'},
                            {'session_id': 'other'}, {'uuid': str(UUID(int=2))}):
                with self.subTest(changes=changes):
                    wrong = selected.assign(**changes)
                    self.assertTrue(validate_scan_sources(wrong, root, corpus, index)['errors'])
            save_plan({**planned, 'source': wrong_source})
            self.assertTrue(validate_scan_sources(selected, root, corpus, index)['errors'])
            save_plan({**planned, 'video_source_metadata': {**video, 'xet_hash': '4' * 64}})
            self.assertTrue(validate_scan_sources(selected, root, corpus, index)['errors'])
            save_plan(planned)
            (root / 'tracks.json').write_text(json.dumps([{**planned, 'source': wrong_source}]))
            with self.assertRaisesRegex(ValueError, 'different source plan'):
                validate_scan_sources(selected, root, corpus, index)
            save_plan(planned)
            index.write_text(json.dumps({track: [{**video, 'size': 5678}]}))
            with self.assertRaisesRegex(ValueError, 'index changed'):
                validate_scan_sources(selected, root, corpus, index)

    def test_archived_pixel_preparation_exact(self):
        torch.set_num_threads(1)
        common = {'torch': torch, 'F': F, 'np': np, 'Image': Image}
        downsample = archived_function('/mnt/paramfs/vansh/richmond-7/reference-latents/encode/refs_lib.py', 'downsample', common)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.random.default_rng(381).integers(0, 256, (960, 512, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(root / 'image.png')
            namespace = {**common, 'DATA': str(root), 'refs_lib': SimpleNamespace(downsample=downsample)}
            archived = archived_function('/mnt/paramfs/vansh/richmond-7/reference-bank/code/studio-actors/encode.py', 'prep', namespace)
            cases = [(1, 256, 480), (2, 260, 488), (3, 281, 527), (4, 256, 480),
                     *[(prop, 512, 960) for prop in range(5, 10)]]
            for prop, width, height in cases:
                with self.subTest(property=prop, width=width):
                    row = {**reference(prop), 'source': 'image.png', 'cw': width, 'ch': height}
                    if prop >= 5:
                        row.update(cx0=0, cy0=0)
                    current, old = prepare(row, root), archived(row)
                    self.assertTrue(torch.equal(current, old))
                    self.assertEqual(tuple(current.shape), (3, 1, 480, 256))
                    self.assertEqual(current.dtype, torch.float32)
                    self.assertTrue(current.ge(-1).all() and current.le(1).all())
                    prepared = ((old[:, 0] + 1) * 127.5).round().byte().permute(1, 2, 0).numpy()
                    Image.fromarray(prepared).save(root / 'prepared.png')
                    row.update(origin='video-scan', image_path=str(root / 'prepared.png'), prepared_image=True)
                    self.assertTrue(torch.equal(prepare(row, root), old))

    def test_merge_global_ids_exact_times_and_bad_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            decoded = Path(directory) / 'decoded.png'
            Image.new('RGB', (512, 960)).save(decoded)
            upstream = pd.DataFrame([reference()])
            scanned = pd.DataFrame([{**reference(index=1, frame=250), 'origin': 'video-scan',
                                     'source': HF_PREFIX + 'video/full.mp4#frame=250', 'image_path': str(decoded)}])
            beatles = pd.DataFrame([reference(bank='beatles-eval')])
            merged = merge_selections(upstream, scanned, beatles)
            self.assertEqual(merged.iid.tolist(), [0, 1, 2])
            self.assertEqual(merged['index'].tolist(), [0, 1, 0])
            self.assertEqual(merged.time_s.tolist(), [0.04, 10.0, 0.04])
            csv_path = Path(directory) / 'bank.csv'
            merged.to_csv(csv_path, index=False)
            loaded = pd.read_csv(csv_path, float_precision='round_trip')
            self.assertEqual(loaded.time_s.tolist(), merged.time_s.tolist())
            wrong = scanned.copy()
            wrong['index'] = 0
            with self.assertRaises(ValueError):
                merge_selections(upstream, wrong, beatles)
            wrong = upstream.copy()
            wrong['source'] = 'gs://old/reference.png'
            with self.assertRaises(ValueError):
                merge_selections(wrong, scanned, beatles)

    def test_latent_metadata_receipts_and_nonfinite_detection(self):
        selected = merge_selections(pd.DataFrame([reference()]), pd.DataFrame(), pd.DataFrame())
        fingerprint = encoder_fingerprint()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor = selected.uuid.iloc[0]
            path = root / 'bank' / actor / 'property_1/latents.safetensors'
            path.parent.mkdir(parents=True)
            receipt = root / 'encoding-checkpoints' / f'bank__{actor}__1.json'
            receipt.parent.mkdir()
            tensor = torch.zeros((1, 48, 30, 16), dtype=torch.bfloat16)
            metadata = {'uuid': actor, 'property': '1', 'slot': 'front', 'n': '1'}

            def save():
                save_file({'video': tensor}, path, metadata=metadata)
                receipt.write_text(json.dumps({'sha256': sha256(path), 'selection_sha256': '1' * 64,
                                               'vae_sha256': VAE_SHA256, 'encoder_fingerprint': fingerprint, 'n': 1}))

            save()
            report, _, sizes = validate_latents(selected, root, '1' * 64)
            self.assertFalse(report['errors'])
            self.assertGreater(sizes[('bank', actor)], 0)
            tensor[0, 0, 0, 0] = float('nan')
            save()
            self.assertIn('Nonfinite', validate_latents(selected, root, '1' * 64)[0]['errors'][0]['error'])
            tensor.zero_()
            metadata['slot'] = 'mouth_open'
            save()
            self.assertIn('metadata', validate_latents(selected, root, '1' * 64)[0]['errors'][0]['error'])
            metadata['slot'] = 'front'
            save()
            self.assertIn('receipt', validate_latents(selected, root, '2' * 64)[0]['errors'][0]['error'])

    def test_source_incompleteness_and_all_zero_actors_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, value in {'targets-summary.json': {'complete': True},
                                'scan-summary.json': {'complete': True, 'available_targets_complete': True, 'missing_sources': 0, 'failures': 0},
                                'selection-summary.json': {'complete': True, 'missing_chunks': 0, 'rows': 0},
                                'decode-summary.json': {'complete': True, 'rows': 0}}.items():
                (root / name).write_text(json.dumps(value))
            diagnostics = {'fatal_input_issues': 0, 'incomplete_source_tracks': ['track'],
                           'issue_counts': {'confirmed_hf_missing': 5}}
            report = completion_evidence(diagnostics, root)
            self.assertFalse(report['complete'])
            self.assertTrue(report['available_work_complete'])
            self.assertFalse(report['source_complete'])
            self.assertIn('HF upstream source evidence is incomplete', report['reasons'])
            diagnostics.update(incomplete_source_tracks=[], issue_counts={})
            self.assertTrue(completion_evidence(diagnostics, root)['complete'])
            # Missing decoded rows or contradictory completion flags cannot be
            # promoted as valid available work merely because sources exist.
            (root / 'decode-summary.json').write_text(json.dumps({'complete': True, 'rows': 1}))
            self.assertFalse(completion_evidence(diagnostics, root)['available_work_complete'])
            (root / 'decode-summary.json').write_text(json.dumps({'complete': False, 'rows': 0}))
            self.assertFalse(completion_evidence(diagnostics, root)['available_work_complete'])
            (root / 'decode-summary.json').write_text(json.dumps({'complete': True, 'rows': 0}))
            (root / 'targets-summary.json').write_text(json.dumps({'complete': True, 'missing_tracks': 1}))
            self.assertFalse(completion_evidence(diagnostics, root)['complete'])
        selected = merge_selections(pd.DataFrame([reference()]), pd.DataFrame(), pd.DataFrame())
        actors = [{'uuid': str(UUID(int=index)), 'cohort': 'train', 'eligible_track_count': 0} for index in range(1, 1760)]
        actor_counts, summary = coverage(selected, actors, [], defaultdict(int))
        self.assertEqual(len(actor_counts), 1759)
        self.assertEqual(len(summary['bank']['zero_actors']), 1758)
        self.assertEqual(summary['bank']['images'], 1)


if __name__ == '__main__':
    unittest.main()
