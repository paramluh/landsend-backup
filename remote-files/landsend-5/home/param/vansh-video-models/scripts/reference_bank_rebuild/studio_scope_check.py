"""Studio membership must constrain source tracks, including shared actors."""

import csv
import tempfile
import unittest
from pathlib import Path

from prepare_studio_scope import FILES, prepare, read_table
from upstream import BEATLES_HELD_ROWS, load_corpus


FIELDS = ('paths', 'uuid', 'session_id', 'recording_id', 'speaker', 'chunk',
          'kind', 'blackbands', 'lighting_rejection')


def row(recording, actor, chunk=0, speaker=1, kind='studio'):
    return dict(paths=f'recordings/{recording}/speaker_{speaker}/chunks/{chunk:06d}',
                uuid=actor, session_id=f'session-{recording}', recording_id=recording,
                speaker=speaker, chunk=chunk, kind=kind, blackbands='no', lighting_rejection='0')


def write(path, rows, fields=FIELDS):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class StudioScopeChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / 'archive'
        self.archive.mkdir()
        self.output = self.root / 'studio-only' / 'csvs'
        self.shared = row('shared-recording', 'shared-actor')
        self.app = row('shared-recording', 'shared-actor', speaker=2, kind='app')
        self.beatles = [row('beatles', 'beatles-actor', chunk=i) for i in range(40)]
        unseen, excess = row('unseen', 'unseen-actor'), row('unseen', 'unseen-actor', chunk=1)
        studio = [self.shared, *self.beatles, unseen, excess]
        unknown_chunk = row('shared-recording', 'shared-actor', chunk=99)
        self.rows = {'all.csv': [self.app, unknown_chunk, *studio], 'all-studio.csv': studio,
                     'all-app.csv': [self.app], 'aloe-app-selected-clips.csv': [self.app],
                     'aloe-studio-no-blackbands-high-quality.csv': [self.shared, *self.beatles],
                     'aloe-studio-seen-eval.csv': [], 'aloe-studio-unseen-eval.csv': [unseen],
                     'aloe-eval-beatles.csv': self.beatles, 'aloe-unseen-excess-chunks.csv': [excess]}
        for name in FILES:
            write(self.archive / name, self.rows[name])
        self.blackbands = self.root / 'blackbands.csv'
        write(self.blackbands, [], ('recording_id', 'speaker', 'chunk'))
        self.training = self.root / 'training.csv'
        write(self.training, [self.shared])

    def prepare(self):
        return prepare(self.archive, self.output, self.blackbands, self.training,
                       expected_actors=3, expected_training_rows=1, expected_training_actors=1)

    def test_shared_actor_and_recording_do_not_admit_app_or_unknown_chunks(self):
        originals = {name: (self.archive / name).read_bytes() for name in FILES}
        report = self.prepare()
        corpus = load_corpus(self.output, self.blackbands)
        self.assertEqual({r['track'] for r in corpus.tracks},
                         {'shared-recording/speaker_1', 'beatles/speaker_1', 'unseen/speaker_1'})
        self.assertEqual(len(corpus.all_clips), 43)
        self.assertEqual(report['shared_studio_app_actors'], 1)
        self.assertEqual(report['files']['all.csv']['excluded_rows'], 2)
        self.assertEqual(report['corpus']['actors_by_cohort'], {'train': 2, 'eval-unseen': 1})
        for name in ('all-app.csv', 'aloe-app-selected-clips.csv'):
            self.assertEqual(read_table(self.output / name)[2], [])
        self.assertEqual((self.output / 'aloe-eval-beatles.csv').read_bytes(), originals['aloe-eval-beatles.csv'])
        self.assertEqual(report['beatles']['held_row_indices'], list(BEATLES_HELD_ROWS))
        self.assertEqual(report, self.prepare())
        self.assertEqual(originals, {name: (self.archive / name).read_bytes() for name in FILES})

    def test_nonstudio_beatles_fails_before_publication(self):
        write(self.archive / 'aloe-eval-beatles.csv', [self.app, *self.beatles[1:]])
        with self.assertRaisesRegex(ValueError, 'Every Beatles row must be studio'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_same_track_in_both_domains_is_rejected(self):
        write(self.archive / 'all-app.csv', [dict(self.shared, kind='app', chunk=99)])
        with self.assertRaisesRegex(ValueError, 'both studio and app'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_authoritative_actor_mismatch_is_rejected(self):
        changed = [dict(r, uuid='wrong-actor') if r == self.shared else r for r in self.rows['all.csv']]
        write(self.archive / 'all.csv', changed)
        with self.assertRaisesRegex(ValueError, 'Identity/kind disagrees'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_app_training_clip_is_rejected_despite_shared_actor(self):
        write(self.training, [self.app])
        with self.assertRaisesRegex(ValueError, 'training CSV contains a non-studio'):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_modified_existing_scope_is_preserved_and_rejected(self):
        self.prepare()
        path = self.output / 'all-app.csv'
        write(path, [self.app])
        changed = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'Existing scope differs'):
            self.prepare()
        self.assertEqual(path.read_bytes(), changed)


if __name__ == '__main__':
    unittest.main()
