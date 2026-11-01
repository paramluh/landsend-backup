"""Causal row orchestration and cached-engine adapter tests without GPU loads."""
import types
import unittest
from pathlib import Path

import numpy as np
import torch

from finetune.pose_opd_credit.rollout import (
    CODEBOOKS, CODEBOOK_SIZE, CODE_DELAY, NATURAL_SAMPLING, PAD,
    make_fastppm_class, rollout_masks, rollout_row, rollout_rows,
)


def seed_row(frames=30, prefix=7, prefill=25):
    total = prefix + frames + CODE_DELAY
    offsets = np.arange(CODEBOOKS, dtype=np.int64) * CODEBOOK_SIZE
    ids = np.full((2, total), PAD, dtype=np.int64)
    ids[1, :prefix + prefill] = np.arange(prefix + prefill)
    depth = np.full((2, total, CODEBOOKS), -1, dtype=np.int64)
    pose = np.full((total, CODEBOOKS), -1, dtype=np.int64)
    depth[1, :prefix] = offsets + 2  # conditioning clip is preserved
    pose[:prefix] = offsets + 3
    depth[1, prefix + CODE_DELAY:prefix + CODE_DELAY + prefill] = offsets + 4
    pose[prefix + CODE_DELAY:prefix + CODE_DELAY + prefill] = offsets + 5
    return dict(P=np.array([prefix]), vin=np.array([prefix, frames]), pad_id=PAD,
                conversation_frames=frames, prefill_frames=prefill,
                speech_delay=2, pose_delay=2, ids=ids, depth=depth,
                pdepth=pose, present=np.ones((2, total)), pid=np.arange(total), segs=[total],
                sampled_masks=rollout_masks(prefix, frames, prefill))


class FakeCachedEngine:
    """Model the engine's timeline contract, including its delayed output."""

    def __init__(self):
        self.calls = []

    def check_audio_features(self, prefix, frames):
        self.checked = (prefix, frames)

    def start(self, instances, seed, params, f0):
        self.asserted_params = params
        self.f0 = f0
        self.instance = instances[0]
        self.prefix = int(self.instance["P"])
        self.text = self.instance["ids"][1].copy()
        self.audio = self.instance["depth"][1].copy()
        self.pose = self.instance["pdepth"].copy()
        n = len(self.text)
        self.lp = {"text": np.full(n, np.nan, np.float32),
                   "audio": np.full((n, CODEBOOKS), np.nan, np.float32),
                   "pose": np.full((n, CODEBOOKS), np.nan, np.float32)}

    def step(self, frame, *, graphs, force, cold_pad):
        position = self.prefix + frame
        self.calls.append((frame, force, cold_pad))
        force = force or {}
        # Emit EOB repeatedly: rollout must still run to the dataset row end.
        self.text[position] = force.get("lane", np.array([151645]))[0]
        offsets = np.arange(CODEBOOKS) * CODEBOOK_SIZE
        self.audio[position] = force.get("codes", (offsets + 11)[None])[0]
        self.pose[position] = force.get("pose", (offsets + 12)[None])[0]
        if "lane" not in force:
            self.lp["text"][position] = -0.4
        if "codes" not in force:
            self.lp["audio"][position] = -0.5
        if "pose" not in force:
            self.lp["pose"][position] = -0.6

    def readback(self, row, total):
        return dict(lane_b=self.text, dep_b=self.audio, pdep=self.pose,
                    lane_a=self.instance["ids"][0].copy(),
                    dep_a=self.instance["depth"][0].copy())

    def behavior_logprobs(self, total):
        return self.lp


class RolloutTests(unittest.TestCase):
    def test_batched_different_prefixes_lengths_and_terminal_forcing(self):
        class Batch:
            NI = 2
            def __init__(self):
                self.engines = [FakeCachedEngine(), FakeCachedEngine()]
                self.calls = []
            def check_audio_features(self, prefix, frames, instance=0):
                self.engines[instance].check_audio_features(prefix, frames)
            def start(self, instances, seeds, params, f0):
                self.seeds = seeds
                for engine, row, seed in zip(self.engines, instances, seeds):
                    engine.start([row], seed, params, f0)
            def step(self, frame, graphs, force, cold_pad):
                self.calls.append((frame, force))
                for i, engine in enumerate(self.engines):
                    if engine.prefix + frame >= len(engine.text):
                        continue
                    selected = {key: value[i:i+1] for key, value in (force or {}).items()}
                    if "lane" in selected and selected["lane"][0] < 0:
                        del selected["lane"]
                    engine.step(frame, graphs=graphs, force=selected, cold_pad=cold_pad)
            def readback(self, instance, total):
                return self.engines[instance].readback(0, total)
            def behavior_logprobs(self, total, instance=0):
                return self.engines[instance].behavior_logprobs(total)
        rows = [seed_row(frames=28, prefix=5), seed_row(frames=33, prefix=11)]
        engine = Batch()
        values = rollout_rows(engine, rows, seeds=[17, 91])
        self.assertEqual(engine.seeds, [17, 91])
        self.assertEqual(len(engine.calls), 10)
        np.testing.assert_array_equal(dict(engine.calls)[28]["lane"], [PAD, -1])
        for i, (row, result) in enumerate(zip(rows, values)):
            reference = rollout_row(FakeCachedEngine(), row, seed=[17, 91][i])
            for key in ("text", "audio", "pose"):
                np.testing.assert_array_equal(getattr(result, key), getattr(reference, key))
                np.testing.assert_array_equal(result.behavior_logprobs[key], reference.behavior_logprobs[key])

        # Dashboard counters stay fresh inside the row and never alter draws,
        # causal state, terminal forcing, or behavior probabilities.
        rows = [seed_row(frames=90, prefix=5), seed_row(frames=90, prefix=11)]
        events = []
        observed = rollout_rows(Batch(), rows, seeds=[17, 91], progress=lambda **event: events.append(event))
        silent = rollout_rows(Batch(), rows, seeds=[17, 91])
        self.assertEqual([event['current_frame'] for event in events], [50, 75, 92])
        self.assertTrue(all(event['total_frames'] == 92 for event in events))
        for actual, reference in zip(observed, silent):
            for stream in ('text', 'audio', 'pose'):
                np.testing.assert_array_equal(getattr(actual, stream), getattr(reference, stream))
                np.testing.assert_array_equal(actual.behavior_logprobs[stream], reference.behavior_logprobs[stream])

    def test_full_row_delayed_prefill_and_forced_flush(self):
        row = seed_row()
        original = {key: value.copy() for key, value in row.items() if isinstance(value, np.ndarray)}
        engine = FakeCachedEngine()
        result = rollout_row(engine, row, seed=81)
        self.assertEqual(engine.f0, 25)
        self.assertEqual(engine.checked, (7, 30))
        self.assertEqual([entry[0] for entry in engine.calls], list(range(25, 32)))
        self.assertEqual(engine.asserted_params, NATURAL_SAMPLING)
        for frame, force, cold_pad in engine.calls:
            self.assertFalse(cold_pad)
            force = force or {}
            self.assertEqual("codes" in force, frame in (25, 26))
            self.assertEqual("pose" in force, frame in (25, 26))
            self.assertEqual("lane" in force, frame in (30, 31))
        self.assertEqual(int(result.sampled_masks["text"].sum()), 5)
        self.assertEqual(int(result.sampled_masks["audio"].sum()), 5)
        np.testing.assert_array_equal(np.flatnonzero(result.sampled_masks["text"]), np.arange(32, 37))
        np.testing.assert_array_equal(np.flatnonzero(result.sampled_masks["audio"]), np.arange(34, 39))
        for key, before in original.items():
            np.testing.assert_array_equal(row[key], before)
        for stream, mask in result.sampled_masks.items():
            self.assertTrue(np.all(result.behavior_logprobs[stream][~mask] == 0))
            self.assertTrue(np.isfinite(result.behavior_logprobs[stream]).all())

    def test_ground_truth_future_rejected(self):
        for stream in ("text", "audio", "pose"):
            row = seed_row()
            position = np.flatnonzero(row["sampled_masks"][stream])[0]
            if stream == "text":
                row["ids"][1, position] = 42
            elif stream == "audio":
                row["depth"][1, position, 0] = 42
            else:
                row["pdepth"][position, 0] = 42
            with self.subTest(stream=stream), self.assertRaisesRegex(ValueError, "Future B"):
                rollout_row(FakeCachedEngine(), row, seed=1)

    def test_wrong_mask_and_incomplete_prefill_rejected(self):
        row = seed_row()
        row["sampled_masks"]["audio"][32] = True
        with self.assertRaisesRegex(ValueError, "sampling mask"):
            rollout_row(FakeCachedEngine(), row, seed=1)
        row = seed_row()
        row["pdepth"][7 + 2 + 24] = -1
        with self.assertRaisesRegex(ValueError, "physical prefill"):
            rollout_row(FakeCachedEngine(), row, seed=1)

    def test_missing_behavior_likelihood_rejected(self):
        class Missing(FakeCachedEngine):
            def behavior_logprobs(self, total):
                self.lp["pose"][-1, 14] = np.nan
                return self.lp
        with self.assertRaisesRegex(RuntimeError, "likelihoods"):
            rollout_row(Missing(), seed_row(), seed=1)

    def test_forced_prefix_corruption_rejected(self):
        class Corrupt(FakeCachedEngine):
            def readback(self, row, total):
                self.pose[0, 0] += 1
                return super().readback(row, total)
        with self.assertRaisesRegex(RuntimeError, "forced pose"):
            rollout_row(Corrupt(), seed_row(), seed=1)

    def test_native_adapter_injects_exact_positions_in_both_paths(self):
        # Exercise the actual mixin methods with a CPU stand-in for the native
        # stack: its input module is the exact location used by FastPPM.
        class Base:
            def _prefill_row(self, instance, count):
                return self.m.inp(torch.zeros(count, 2))
            def _seg(self, _index):
                return self.m.inp(torch.zeros(2, 2))
        cls = make_fastppm_class(types.SimpleNamespace(FastPPM=Base))
        actor = object.__new__(cls)
        actor.m = types.SimpleNamespace(inp=torch.nn.Identity())
        actor._opd_audio = torch.arange(40, dtype=torch.float32).reshape(2, 10, 2)
        actor._opd_input_mode = None
        actor.d0 = torch.device("cpu")
        actor.NI = 2
        actor.arI = {actor.d0: torch.arange(2)}
        actor.Tv = {actor.d0: torch.tensor([7, 4])}
        hook = actor.m.inp.register_forward_hook(actor._add_audio)
        try:
            torch.testing.assert_close(actor._prefill_row(1, 4), actor._opd_audio[1, :4])
            torch.testing.assert_close(actor._seg(0), actor._opd_audio[torch.arange(2), torch.tensor([7, 4])])
            actor.Tv[actor.d0].fill_(8)
            torch.testing.assert_close(actor._seg(0), actor._opd_audio[:, 8])
            torch.testing.assert_close(actor.m.inp(torch.ones(1, 2)), torch.ones(1, 2))
            self.assertIsNone(actor._opd_input_mode)
        finally:
            hook.remove()

    def test_generated_arrays_feed_real_replay_builder(self):
        from finetune.pose_opd_credit.data import replay_from_rollout
        row = seed_row()
        result = rollout_row(FakeCachedEngine(), row, seed=81)
        bundle = replay_from_rollout(row, result.text, result.audio, result.pose)
        student = bundle["student"]
        np.testing.assert_array_equal(student["y"], np.full(5, 151645))
        np.testing.assert_array_equal(student["light"], np.arange(34, 39))
        np.testing.assert_array_equal(student["plight"], np.arange(34, 39))
        self.assertTrue(np.all(student["depth"][1, 34:, 0] == 11))
        self.assertTrue(np.all(student["pdepth"][34:, 0] == 12))

    @unittest.skipUnless(Path("/home/param/datasets/aloe-duplex/pose/POSE_INDEX.json").is_file(),
                         "Installed Aloe dataset required")
    def test_real_empty_prefix_rollout_preserves_prefill_and_teacher_histories(self):
        from finetune.pose_opd_credit.config import TrainingConfig
        from finetune.pose_opd_credit.data import ProductionRows, replay_from_rollout

        # This source row has valid pose but no supported later-sibling clip.
        # Use the real prefix builder and sanitization, then exercise the full
        # cached orchestration/replay contract without allocating GPU weights.
        data = ProductionRows(TrainingConfig(data_variant="raw", missing_clip_policy="empty"))
        row = data.load((0, 1837, 0), load_audio=False)
        self.assertEqual(row["voice_clip"], 0)
        self.assertEqual(row["voice_drawn"], 1)
        self.assertEqual(int(row["P"][0]), 5)
        n = row["conversation_frames"]
        self.assertGreater(n, 27)
        np.testing.assert_array_equal(row["ids"][1, :5], [151644, 8948, 198, 151645, 198])
        self.assertTrue(np.all(row["depth"][:, :5] == -1))
        self.assertTrue(np.all(row["pdepth"][:5] == -1))
        self.assertTrue(np.all(row["ids"][1, 30:] == PAD))
        self.assertTrue(np.all(row["depth"][1, 32:] == -1))
        self.assertTrue(np.all(row["pdepth"][32:] == -1))
        before = {key: row[key].copy() for key in ("ids", "depth", "pdepth")}

        engine = FakeCachedEngine()
        result = rollout_row(engine, row, seed=81)
        self.assertEqual(engine.checked, (5, n))
        self.assertEqual(engine.f0, 25)
        self.assertEqual([call[0] for call in engine.calls], list(range(25, n + 2)))
        for frame, force, _ in engine.calls:
            force = force or {}
            self.assertEqual("codes" in force, frame < 27)
            self.assertEqual("pose" in force, frame < 27)
            self.assertEqual("lane" in force, frame >= n)
        np.testing.assert_array_equal(np.flatnonzero(result.sampled_masks["text"]), np.arange(30, 5 + n))
        for stream in ("audio", "pose"):
            np.testing.assert_array_equal(np.flatnonzero(result.sampled_masks[stream]), np.arange(32, 7 + n))
        for stream, values, reference in (("text", result.text, before["ids"][1]),
                                           ("audio", result.audio, before["depth"][1]),
                                           ("pose", result.pose, before["pdepth"])):
            forced = ~result.sampled_masks[stream]
            np.testing.assert_array_equal(values[forced], reference[forced])
            self.assertTrue(np.all(result.behavior_logprobs[stream][forced] == 0))
        for key, reference in before.items():
            np.testing.assert_array_equal(row[key], reference)

        bundle = replay_from_rollout(row, result.text, result.audio, result.pose)
        for view in bundle.values():
            np.testing.assert_array_equal(view["ids"][1], result.text)
            np.testing.assert_array_equal(view["depth"][1], result.audio)
            np.testing.assert_array_equal(view["y"], np.full(n - 25, 151645))
            np.testing.assert_array_equal(view["light"], np.arange(32, 7 + n))
            self.assertEqual(view["voice_clip"], 0)
        for role in ("student", "pose_teacher"):
            np.testing.assert_array_equal(bundle[role]["pdepth"], result.pose)
            np.testing.assert_array_equal(bundle[role]["plight"], np.arange(32, 7 + n))
        self.assertNotIn("pdepth", bundle["audio_teacher"])
        self.assertNotIn("aref", bundle["pose_teacher"])

    def test_real_depth_override_records_every_autoregressive_draw(self):
        class Decoder:
            def __init__(self):
                self.dec = types.SimpleNamespace(cb0_emb=torch.nn.Embedding(8, 2))
                self.seen = []
            def slot0(self, hv, zb):
                return hv
            def step(self, x, position):
                self.seen.append((position, x.clone()))
                return x[:, -1]
            def head(self, hidden, codebook):
                return torch.arange(8, dtype=torch.float32)[None] * (1 + codebook / 32)
            def emb(self, code, codebook):
                return code[:, None, None].expand(-1, 1, 2).float()
            def gen(self, *args, **kwargs):
                raise AssertionError("Natural override was not installed")
        class Base:
            def __init__(self, **kwargs):
                self.pose = True; self.V = 151682; self.lmax = 10; self.H = 2; self.NI = kwargs["n_inst"]
                self.d0 = self.dl = self.dp = torch.device("cpu")
                self.m = types.SimpleNamespace(inp=torch.nn.Identity())
                self.dec_s, self.dec_p = Decoder(), Decoder()
        native = types.SimpleNamespace(FastPPM=Base,
            row_multinomial=lambda p, g: torch.multinomial(p, 1, generator=g)[:, 0])
        cls = make_fastppm_class(native)
        engine = cls()
        try:
            generator = torch.Generator().manual_seed(81)
            codes = engine.dec_p.gen(torch.zeros(1, 2), torch.tensor([3]), generator, 1., 0)
            self.assertEqual(tuple(codes.shape), (1, 32))
            self.assertEqual(len(engine.dec_p.seen), 31)
            for book in range(1, 32):
                expected = torch.log_softmax(torch.arange(8, dtype=torch.float32) * (1 + book / 32), 0)
                torch.testing.assert_close(engine._opd_now["pose"][0, book], expected[codes[0, book]])
                if book > 1:
                    _, actual_input = engine.dec_p.seen[book - 1]
                    self.assertEqual(int(actual_input[0, 0, 0]), int(codes[0, book - 1]))
        finally:
            engine.close()
        self.assertNotIn("gen", engine.dec_p.__dict__)

    def test_native_text_sampler_excludes_padding_and_records_probability(self):
        captured = []
        def draw(probability, generator):
            captured.append(probability.clone())
            return torch.multinomial(probability, 1, generator=generator)[:, 0]
        native = types.SimpleNamespace(FastPPM=object, row_multinomial=draw, PAD=1, NW=2)
        cls = make_fastppm_class(native)
        actor = object.__new__(cls)
        actor.dl = torch.device("cpu"); actor.NI = 1
        actor.Tv = {actor.dl: torch.tensor([3])}
        actor.arI = {actor.dl: torch.arange(1)}
        actor.adapt = torch.zeros(1, 5)
        actor._opd_force = None; actor._opd_executable = 4
        actor._opd_uniforms = None
        actor._opd_lane_force = torch.full((1,), -1, dtype=torch.long)
        actor.hn_prev = torch.zeros(1, 2)
        actor.text_logits = lambda _hidden: torch.tensor([[0., 1., 2., 3., 1000., 1000.]])
        actor.zt = torch.zeros(1, 6); actor.probs = torch.zeros(1, 2); actor.ptok = torch.zeros(1, 2)
        actor.gen_t = torch.Generator().manual_seed(81)
        actor.laneB = torch.zeros(1, 8, dtype=torch.long); actor.tok_now = torch.zeros(1, dtype=torch.long)
        actor.ring = types.SimpleNamespace(push=lambda _token: None)
        actor._opd_history = {"text": torch.zeros(1, 8)}
        actor._text()
        self.assertEqual(float(captured[0][:, 4:].sum()), 0.)
        token = int(actor.tok_now[0])
        self.assertLess(token, 4)
        expected = torch.log_softmax(torch.arange(4, dtype=torch.float32), 0)[token]
        torch.testing.assert_close(actor._opd_history["text"][0, 3], expected)
        actor._opd_lane_force.fill_(1)
        actor._text()
        self.assertEqual(len(captured), 2)  # graph-compatible draw is ignored for the forced row
        self.assertEqual(int(actor.tok_now[0]), 1)
        self.assertEqual(float(actor._opd_history["text"][0, 3]), 0.)


if __name__ == "__main__":
    unittest.main()
