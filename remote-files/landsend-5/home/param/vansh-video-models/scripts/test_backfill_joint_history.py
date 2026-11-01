"""Local safety tests; these never initialize or mutate W&B."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('backfill_joint_history', Path(__file__).with_name('backfill_joint_history.py'))
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


def metric(number, loss=1.):
    return {'rounds': number, 'warmup_step': min(number, 150),
            'epoch_step': max(number - 150, 0), 'student_updates': max(number - 150, 0),
            'critic_updates': number * 4, 'student_rows_seen': max(number - 150, 0) * 40,
            'phase': 'critic_warmup' if number <= 150 else 'joint', 'benchmark': False,
            'critic': {'loss': loss}}


class ArchiveTests(unittest.TestCase):
    def history(self, rows, attempts, trailing=b''):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)
        path.joinpath('metrics.jsonl').write_bytes(b''.join((json.dumps(row) + '\n').encode() for row, _ in rows) + trailing)
        mapping = {}
        for line, (row, name) in enumerate(rows, 1):
            digest = archive.bytes_sha(archive.canonical(row).encode())
            mapping.setdefault(digest, []).append({'attempt': name, 'attempt_log_line': line})
        return path, (attempts, mapping)

    def test_latest_resumed_rows_replace_original_and_bind_provenance(self):
        attempts = {'production-01': {'created_at': '2026-10-18T10:00:00Z', 'resume_round': 0},
                    'production-02': {'created_at': '2026-10-18T16:00:00Z', 'resume_round': 600}}
        rows = [(metric(n), 'production-01') for n in range(1, 603)]
        rows += [(metric(n, 2.), 'production-02') for n in (601, 602)]
        path, evidence = self.history(rows, attempts, trailing=b'{"rounds":603')
        with patch.object(archive, 'attempt_inventory', return_value=evidence):
            chosen, rejected, proof = archive.select_history(path, 602)
        self.assertEqual(chosen[600][0]['critic']['loss'], 2.)
        self.assertEqual(chosen[600][2]['source_line'], 603)
        self.assertEqual([r['round'] for r in rejected], [601, 602])
        self.assertEqual(proof['selected_rows_by_attempt'], {'production-01': 600, 'production-02': 2})
        self.assertEqual(proof['source_snapshot_complete_lines'], 604)

    def test_unreplayed_original601_is_rejected(self):
        rows = [(metric(n), 'production-01') for n in range(1, 602)]
        path, evidence = self.history(rows, {'production-01': {'created_at': 'a', 'resume_round': 0}})
        with patch.object(archive, 'attempt_inventory', return_value=evidence), self.assertRaisesRegex(ValueError, 'abandoned'):
            archive.select_history(path, 601)

    def test_replay_from_earlier_attempt_and_ambiguous_identity_rejected(self):
        rows = [(metric(1), 'new'), (metric(1, 2.), 'old')]
        attempts = {'new': {'created_at': 'b', 'resume_round': 0}, 'old': {'created_at': 'a', 'resume_round': 0}}
        path, evidence = self.history(rows, attempts)
        with patch.object(archive, 'attempt_inventory', return_value=evidence), self.assertRaisesRegex(ValueError, 'reversed'):
            archive.select_history(path, 1)
        path, evidence = self.history([(metric(1), 'new'), (metric(1), 'new')], attempts)
        with patch.object(archive, 'attempt_inventory', return_value=evidence), self.assertRaisesRegex(ValueError, 'exact attempt matches'):
            archive.select_history(path, 1)

    def test_missing_round_and_wrong_counters_fail(self):
        path, evidence = self.history([(metric(1), 'a'), (metric(3), 'a')], {'a': {'created_at': 'a', 'resume_round': 0}})
        with patch.object(archive, 'attempt_inventory', return_value=evidence), self.assertRaisesRegex(ValueError, 'contiguous'):
            archive.select_history(path, 3)
        row = metric(600); row['student_updates'] += 1
        with self.assertRaisesRegex(ValueError, 'counters'):
            archive.validate_counters(row)
        row = metric(600); row['critic']['loss'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            archive.validate_counters(row)

    def test_projection_preserves_historical_absence(self):
        import sys
        sys.path.insert(0, str(archive.REPO))
        from distillation.joint_metrics import project_training_metrics
        projected = project_training_metrics(metric(100))
        self.assertEqual(projected['train/critic_loss'], 1.)
        self.assertEqual(projected['phase/critic_warmup'], 1)
        self.assertNotIn('train/student_loss', projected)
        self.assertFalse(any(key.startswith(('learning_rate/', 'stabilizer_diagnostics/', 'resolver_diagnostics/')) for key in projected))

    def test_readback_checks_each_scalar_and_rejects_surplus(self):
        payload = [{'train/round': 1, 'train/critic_loss': 2.}]
        actual = [{**payload[0], '_step': 1, '_runtime': 1., '_timestamp': 10.}]
        class Remote:
            url = 'https://example.test/archive'
            def scan_history(self, **kwargs): return iter(actual)
        class API:
            def flush(self): pass
            def run(self, path): return Remote()
        manifest = {'entity': 'a', 'project': 'b', 'archive_run_id': 'archive'}
        self.assertEqual(archive.verify_history(API(), manifest, payload, 0)['verified_scalar_values'], 2)
        actual[0]['sensitivity_history/raw_history_0_max_amplification'] = None
        self.assertEqual(archive.verify_history(API(), manifest, payload, 0)['verified_scalar_values'], 2)
        for key, value in [('train/critic_loss', 3.), ('unrecorded', 7.), ('_step', 2)]:
            actual[:] = [{**payload[0], '_step': 1, key: value}]
            with self.assertRaisesRegex(RuntimeError, 'every prepared scalar'):
                archive.verify_history(API(), manifest, payload, 0)

    def sensitivity_fixture(self):
        directory = tempfile.TemporaryDirectory(); self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        chosen = [(metric(n), b'', {'attempt': 'production-01', 'source_line': n}) for n in range(1, 5)]
        payload = [{'train/round': n} for n in range(1, 5)]
        log = root / 'node-0.log'
        case = {'status': 'ok', 'maximum_amplification': .8, 'last_frame_amplification': .3,
                'frames': [{'target_index': 1, 'amplification': .8, 'nonfinite_rows': 0},
                           {'target_index': 2, 'amplification': .3, 'nonfinite_rows': 0}]}
        report = {'format': 'joint-latent-sensitivity-v1', 'status': 'ok', 'measured_target_frames': 2,
                  'requested_input_rms': .01, 'seeds': [1729], 'history_indices_oldest_first': [0, 1, 2],
                  'reference_indices': [0, 4, 8], 'rollouts': [{'seed': 1729, 'cases': {
                      name: copy.deepcopy(case) for name in ('history_0', 'history_1', 'history_2',
                                                             'sink', 'reference_0', 'reference_4', 'reference_8')}}]}
        events = []
        for number in (2, 4):
            cp = root / f'step-{number:06d}-actual'; cp.mkdir()
            counters = {key: chosen[number-1][0][key] for key in ('rounds', 'student_updates', 'critic_updates')}
            for filename, data in {
                'metadata.json': {'checkpoint_id': f'id{number}', 'counters': counters},
                'checkpoint.complete.json': {'checkpoint_id': f'id{number}', 'files': []},
                'evaluation.complete.json': {'checkpoint_id': f'id{number}',
                    'reports': [{'weights': 'raw', 'file': 'sensitivity-raw.json'}]},
                'sensitivity-raw.json': report,
            }.items(): (cp/filename).write_text(json.dumps(data))
            events.append({'event': 'checkpoint_complete', 'path': str(cp), 'counters': counters})
        log.write_text(''.join(json.dumps(event)+'\n' for event in events))
        evidence = {'attempts': {'production-01': {'node_log': str(log)}}}
        return root, chosen, payload, evidence

    def test_only_actual_checkpoints_gain_exact_compact_sensitivity(self):
        root, chosen, payload, evidence = self.sensitivity_fixture()
        original = {str(p): p.read_bytes() for p in root.rglob('*.json')}
        proof = archive.merge_checkpoint_sensitivity(root, chosen, payload, evidence)
        self.assertEqual(proof['checkpoint_rounds'], [2, 4])
        self.assertEqual([len(row) for row in payload], [1, 23, 1, 23])
        self.assertEqual(payload[1]['sensitivity_history/raw_history_0_max_amplification'], .8)
        self.assertEqual(proof['checkpoints'][0]['completion_event']['node_log_line'], 1)
        self.assertEqual(proof['checkpoints'][0]['reports'][0]['sha256'],
                         archive.sha(root/'step-000002-actual/sensitivity-raw.json'))
        self.assertEqual(original, {str(p): p.read_bytes() for p in root.rglob('*.json')})

    def test_sensitivity_rejects_wrong_attempt_or_invalid_report(self):
        root, chosen, payload, evidence = self.sensitivity_fixture()
        chosen[1][2]['attempt'] = 'production-02'
        with self.assertRaisesRegex(ValueError, 'retained attempt'):
            archive.merge_checkpoint_sensitivity(root, chosen, payload, evidence)
        chosen[1][2]['attempt'] = 'production-01'
        path = root/'step-000002-actual/sensitivity-raw.json'
        report = json.loads(path.read_text()); report['rollouts'][0]['cases']['history_0']['maximum_amplification'] = .9
        path.write_text(json.dumps(report))
        with self.assertRaisesRegex(ValueError, 'aggregate differs'):
            archive.merge_checkpoint_sensitivity(root, chosen, payload, evidence)

    def test_valid_pruned_identity_lineage_required(self):
        root, chosen, payload, evidence = self.sensitivity_fixture()
        cp = root/'step-000002-actual'
        metadata = json.loads((cp/'metadata.json').read_text())
        weights = {'student.safetensors': {'size': 3}}
        metadata.update(checkpoint_id='pruned', optimizer_state_retained=False,
                        optimizer_pruned_from_checkpoint_id='id2', optimizer_pruned_weights=weights)
        (cp/'metadata.json').write_text(json.dumps(metadata))
        (cp/'student.safetensors').write_bytes(b'123')
        (cp/'checkpoint.complete.json').write_text(json.dumps({'checkpoint_id': 'pruned',
              'files': [{'path': 'student.safetensors', 'size': 3}]}))
        receipt = cp.with_name(cp.name+'.optimizer-prune.json')
        receipt.write_text(json.dumps({'path': str(cp), 'previous_checkpoint_id': 'id2',
             'checkpoint_id': 'pruned', 'optimizer_state_retained': False, 'weights_unchanged': True, 'weights': weights}))
        proof = archive.merge_checkpoint_sensitivity(root, chosen, payload, evidence)
        self.assertEqual(proof['checkpoints'][0]['checkpoint_id'], 'pruned')
        self.assertEqual(proof['checkpoints'][0]['original_evaluation_checkpoint_id'], 'id2')
        self.assertIn(str(receipt), proof['checkpoints'][0]['source_sha256'])
        bad = json.loads(receipt.read_text()); bad['previous_checkpoint_id'] = 'wrong'
        receipt.write_text(json.dumps(bad))
        with self.assertRaisesRegex(ValueError, 'prune lineage'):
            archive.merge_checkpoint_sensitivity(root, chosen, [{'train/round': n} for n in range(1, 5)], evidence)

    def recovery_fixture(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        base = Path(temporary.name); run = base/'run'; run.mkdir()
        cp = run/'step-001314-reviewed'; cp.mkdir()
        job3 = base/'jobs/production-03'; job4 = base/'jobs/production-04'
        job3.mkdir(parents=True); job4.mkdir()
        row = metric(1314)
        counters = {key: row[key] for key in ('rounds', 'warmup_step', 'epoch_step', 'student_updates', 'critic_updates', 'student_rows_seen')}
        def write(path, value):
            path.write_text(json.dumps(value)+'\n')
            return {'path': str(path), 'sha256': archive.sha(path)}
        def line_ref(path, number):
            return {'log': str(path), 'line': number,
                    'sha256': archive.bytes_sha(path.read_bytes().splitlines(keepends=True)[number-1])}
        metadata = {'checkpoint_id': archive.RECOVERY_CHECKPOINT_ID, 'counters': counters}
        meta_ref = write(cp/'metadata.json', metadata)
        (cp/'rank-00000.pt').write_bytes(b'123')
        complete = {'checkpoint_id': archive.RECOVERY_CHECKPOINT_ID, 'files': [{'path': 'rank-00000.pt', 'size': 3}]}
        complete_ref = write(cp/'checkpoint.complete.json', complete)
        evaluation_ref = write(cp/'evaluation.complete.json', {'checkpoint_id': archive.RECOVERY_CHECKPOINT_ID,
                    'reports': [{'weights': 'raw', 'file': 'sensitivity-raw.json'}]})
        raw_ref = write(cp/'sensitivity-raw.json', {'source': 'existing evaluation fixture'})
        producer_job = write(job3/'job.json', {'id': 'job3', 'source': '/immutable/source3', 'created_at': '2026-10-18T19:00:00Z'})
        (job3/'node-0.log').write_text(json.dumps(row)+'\nRuntimeError: Checkpoint identity does not belong to this run\n')
        request_ref = write(job3/'request.json', {'request': 'save_q'})
        marker = run/'archived-save_q.pending'; marker.write_text('reviewed request')
        durable = {'schema': 'joint-durable-checkpoint-recovery-v1', 'status': 'pending_startup_completion',
                   'run_directory': str(run), 'checkpoint': str(cp), 'checkpoint_id': archive.RECOVERY_CHECKPOINT_ID,
                   'counters': counters, 'checkpoint_metadata_sha256': meta_ref['sha256'],
                   'checkpoint_complete_sha256': complete_ref['sha256'],
                   'latest_pointer': {'checkpoint': str(cp), 'counters': counters},
                   'manifest_file_count': 1, 'manifest_total_bytes': 3, 'rank_shards': 1,
                   'all_manifest_file_sizes_verified': True, 'producer_job': producer_job['path'],
                   'producer_job_sha256': producer_job['sha256'], 'producer_source': '/immutable/source3',
                   'producer_last_metric': line_ref(job3/'node-0.log', 1),
                   'producer_failure_lines': [{k:v for k,v in line_ref(job3/'node-0.log', 2).items() if k != 'log'}],
                   'worker_exits': [{'job_id': 'job3', 'returncode': 1} for _ in range(5)],
                   'save_request': {'request_receipt': request_ref['path'], 'request_receipt_sha256': request_ref['sha256'],
                                    'archived_marker': str(marker), 'marker_sha256': archive.sha(marker)}}
        durable_ref = write(run/'durable.json', durable)
        recovery_job = write(job4/'job.json', {'mode': 'train', 'created_at': '2026-10-18T20:00:00Z',
                            'overrides': [f'resume_from={cp}']})
        config_ref, runtime_ref = write(run/'config.json', {}), write(run/'runtime.json', {})
        resume_ref = write(run/'resume-record.json', {'resume_from': str(cp), 'config': config_ref['path'],
                           'config_sha256': config_ref['sha256'], 'runtime': runtime_ref['path'], 'runtime_sha256': runtime_ref['sha256']})
        tracker = {'id': archive.PROTECTED_RUN_ID, 'online_metrics_verified': True, 'verified_round': 1314,
                   'verified_student_updates': 1164, 'verified_sensitivity_metrics': 22,
                   'sensitivity_report_uploads_verified': True, 'sensitivity_report_files': [{
                       'name': f'sensitivity_reports/{archive.RECOVERY_CHECKPOINT_ID}/sensitivity-raw.json',
                       'sha256': raw_ref['sha256'], 'remote_sha256_verified': True}]}
        event = {'event': 'checkpoint_evaluation_recovered', 'path': str(cp), 'counters': counters,
                 'checkpoint_id': archive.RECOVERY_CHECKPOINT_ID, 'tracking': tracker}
        (job4/'node-0.log').write_text(json.dumps({'event': 'startup', 'counters': counters})+'\n'+json.dumps(event)+'\n')
        gate_ref = write(run/'gate.json', {'passed': True})
        final = {'schema': 'joint-durable-checkpoint-recovery-completion-v1', 'status': 'completed',
                 'durable_save_receipt': durable_ref, 'recovery_job': recovery_job, 'resume_record': resume_ref,
                 'recovery_event': line_ref(job4/'node-0.log', 2), 'evaluation': evaluation_ref,
                 'reports': [raw_ref], 'gate_proofs': [gate_ref]}
        final_path = run/'completion.json'; write(final_path, final)
        evidence = {'attempts': {'production-03': {'job_manifest': producer_job['path'], 'job_manifest_sha256': producer_job['sha256']}}}
        args = (final_path, run, cp, metadata, complete, row, {'attempt': 'production-03'}, evidence)
        return args, final, durable_ref['sha256']

    def test_completed_recovery_is_explicit_and_pending_is_rejected(self):
        args, final, digest = self.recovery_fixture()
        with patch.object(archive, 'DURABLE_RECOVERY_SHA256', digest):
            proof = archive.recovered_completion(*args)
            self.assertEqual(proof['kind'], 'recovered_evaluation_after_durable_save')
            self.assertEqual(proof['producer_attempt'], 'production-03')
            final['status'] = 'pending_startup_completion'; args[0].write_text(json.dumps(final))
            with self.assertRaisesRegex(ValueError, 'completed recovery receipt'):
                archive.recovered_completion(*args)

    def test_recovery_rejects_unreviewed_or_modified_checkpoint(self):
        args, _, digest = self.recovery_fixture()
        with self.assertRaisesRegex(ValueError, 'Unreviewed durable-save'):
            archive.recovered_completion(*args)
        with patch.object(archive, 'DURABLE_RECOVERY_SHA256', digest):
            (args[2]/'rank-00000.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'absent or changed'):
                archive.recovered_completion(*args)

    def test_recovery_rejects_training_advance_before_evaluation(self):
        args, final, digest = self.recovery_fixture()
        log = Path(final['recovery_event']['log'])
        lines = log.read_bytes().splitlines(keepends=True)
        lines[0] = (json.dumps(metric(1315))+'\n').encode(); log.write_bytes(b''.join(lines))
        with patch.object(archive, 'DURABLE_RECOVERY_SHA256', digest), self.assertRaisesRegex(ValueError, 'before new training'):
            archive.recovered_completion(*args)

    def test_recovery_rejects_unverified_raw_publication_and_changed_report(self):
        args, final, digest = self.recovery_fixture()
        log = Path(final['recovery_event']['log']); lines = log.read_text().splitlines(keepends=True)
        event = json.loads(lines[1]); event['tracking']['sensitivity_report_files'][0]['remote_sha256_verified'] = False
        lines[1] = json.dumps(event)+'\n'; log.write_text(''.join(lines))
        final['recovery_event']['sha256'] = archive.bytes_sha(lines[1].encode()); args[0].write_text(json.dumps(final))
        with patch.object(archive, 'DURABLE_RECOVERY_SHA256', digest):
            with self.assertRaisesRegex(ValueError, 'remote byte verification'):
                archive.recovered_completion(*args)
            Path(final['reports'][0]['path']).write_text('{}')
            with self.assertRaisesRegex(ValueError, 'pinned hash'):
                archive.recovered_completion(*args)


if __name__ == '__main__':
    unittest.main()
