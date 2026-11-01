"""CPU checks for isolated conversation packing used only by benchmarks."""
import copy
import unittest

import torch

from finetune.duplex_distill.profile_packing import apply_global_weights, pack_prepared
from finetune.duplex_distill.losses import distribution_kl


def row(t, marker, *, vox=True):
    sel = torch.arange(1, t - 1)
    x = {"kind": "pose", "dummy": False, "segs": [t],
         "ids": torch.full((2, t), marker, dtype=torch.long),
         "depth": torch.full((2, t, 32), marker, dtype=torch.long),
         "present": torch.ones(2, t), "pid": torch.arange(t)[None],
         "sel": sel, "light": torch.tensor([t - 1]), "plight": torch.tensor([t - 2]),
         "pdepth": torch.full((t, 32), marker, dtype=torch.long),
         "cb0": torch.full((t,), marker, dtype=torch.long),
         "pcb0": torch.full((t,), marker, dtype=torch.long),
         "own": torch.tensor([[len(sel), t, 1, t, 1]], dtype=torch.float64),
         "P": torch.tensor([1]), "y": torch.full_like(sel, marker)}
    for k in ("word", "pad", "eob", "nw", "text_weight"):
        x[k] = torch.ones(len(sel))
    for k in ("speech_weight", "pose_weight"):
        x[k] = torch.ones(t)
    x["text_mask"] = torch.ones(len(sel), dtype=torch.bool)
    x["speech_mask"] = x["pose_mask"] = torch.ones(t, dtype=torch.bool)
    if vox:
        x.update(vin=torch.tensor([1, t - 2]), vpos=torch.arange(1, t - 1),
                 vlat=torch.full((1, 4 * (t - 2), 3), float(marker)))
    else:
        x["ids"][0] = 151670
        x["depth"][0] = -1
        x["present"][0] = 0
    return x


class ProfilePackingTests(unittest.TestCase):
    def test_indices_latents_and_source_preservation(self):
        a, b = row(5, 7), row(7, 13)
        before = copy.deepcopy([a, b])
        packed, info = pack_prepared([a, b])
        self.assertEqual(info["row_timeline_frames"], [5, 7])
        self.assertEqual(info["timeline_frames"], 12)
        self.assertEqual(packed["pid"].tolist(), [[0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6]])
        self.assertEqual(packed["sel"].tolist(), [1, 2, 3, 6, 7, 8, 9, 10])
        self.assertEqual(packed["light"].tolist(), [4, 11])
        self.assertEqual(packed["plight"].tolist(), [3, 10])
        self.assertEqual(packed["vpos"].tolist(), [1, 2, 3, 6, 7, 8, 9, 10])
        self.assertEqual(packed["vin"].tolist(), [[1, 3], [6, 5]])
        self.assertEqual(packed["vlat"].shape, (1, 32, 3))
        injected = torch.zeros(12)
        stacked = packed["vlat"].reshape(1, 8, 12).mean(-1)[0]
        injected.index_copy_(0, packed["vpos"], stacked)
        self.assertEqual(injected.tolist(), [0, 7, 7, 7, 0, 0, 13, 13, 13, 13, 13, 0])
        self.assertTrue(torch.equal(packed["pdepth"][packed["plight"], 0], torch.tensor([7, 13])))
        self.assertTrue(torch.equal(packed["ids"][1, packed["sel"]], packed["y"]))
        for original, saved in zip((a, b), before):
            for key, value in original.items():
                if isinstance(value, torch.Tensor):
                    self.assertTrue(torch.equal(value, saved[key]), key)

    def test_segments_isolate_attention_and_one_frame_history(self):
        packed, _ = pack_prepared([row(5, 7), row(7, 13)])
        seg = torch.repeat_interleave(torch.arange(2), torch.tensor(packed["segs"]))
        positions = torch.arange(12)
        causal = (positions[:, None] >= positions[None, :]) & (seg[:, None] == seg[None, :])
        weights = torch.zeros(12, 12).masked_fill(~causal, -torch.inf).softmax(-1)
        values = packed["ids"][1].float()
        expected = weights @ values
        changed = values.clone(); changed[:5] = 900
        self.assertTrue(torch.equal((weights @ changed)[5:], expected[5:]))
        shifted = values.roll(1)
        starts = torch.tensor([0, packed["segs"][0]])
        shifted[starts] = -1
        self.assertEqual(shifted[5].item(), -1)
        self.assertEqual(shifted[6].item(), 13)

    def test_native_monoplex_a_stays_empty(self):
        packed, info = pack_prepared([row(5, 7, vox=False), row(7, 13, vox=False)])
        self.assertTrue((packed["ids"][0] == 151670).all())
        self.assertTrue((packed["depth"][0] == -1).all())
        self.assertFalse(packed["present"][0].any())
        self.assertNotIn("vlat", packed)
        self.assertEqual(info["audio_frames"], 0)

    def test_rejects_unknown_and_mixed_schema(self):
        a, b = row(5, 7), row(7, 13)
        with self.assertRaisesRegex(ValueError, "Unknown"):
            pack_prepared([dict(a, extra=torch.ones(1)), dict(b, extra=torch.ones(1))])
        with self.assertRaisesRegex(ValueError, "same prepared schema"):
            pack_prepared([a, row(7, 13, vox=False)])

    def test_packed_balanced_kl_gradients_equal_independent_global_token_means(self):
        # Two simulated ranks with two unequal-length conversations each. One
        # conversation has no pose but retains real text/speech supervision.
        generator = torch.Generator().manual_seed(712)
        groups = [[row(5, 7), row(7, 13)], [row(6, 3), row(4, 11)]]
        missing = groups[0][1]
        missing["pdepth"].fill_(-1); missing["pcb0"].fill_(-100)
        missing["pose_mask"] = torch.zeros(7, dtype=torch.bool)
        missing["plight"] = torch.empty(0, dtype=torch.long)
        missing["own"][0, 3:] = 0
        groups[0][0]["text_mask"][0] = False
        counts = torch.zeros(5, dtype=torch.float64)
        payloads = []
        for pair in groups:
            payload_pair = []
            for item in pair:
                t = item["ids"].shape[1]
                counts += torch.tensor([item["text_mask"].sum(), item["speech_mask"].sum(),
                                        item["pose_mask"].sum(), len(item["light"]), len(item["plight"])])
                payload_pair.append({
                    "features": torch.randn(t, 4, dtype=torch.float64, generator=generator),
                    "teachers": [torch.randn(t, 17, dtype=torch.float64, generator=generator) for _ in range(3)],
                    "depth_teachers": [torch.randn(t, 31, 17, dtype=torch.float64, generator=generator) for _ in range(2)],
                })
            payloads.append(payload_pair)
        initial = torch.randn(4, 17, dtype=torch.float64, generator=generator)
        packed_parameter = initial.clone().requires_grad_()
        reference_parameter = initial.clone().requires_grad_()
        packed_objective = packed_parameter.sum() * 0
        for pair, payload_pair in zip(groups, payloads):
            normalized = apply_global_weights(pair, counts, world=2)
            packed, _ = pack_prepared(normalized)
            features = torch.cat([p["features"] for p in payload_pair])
            teacher = [torch.cat([p["teachers"][j] for p in payload_pair]) for j in range(3)]
            depth_teacher = [torch.cat([p["depth_teachers"][j] for p in payload_pair]) for j in range(2)]
            logits = features @ packed_parameter
            for name, index, selected, target, coefficient in [
                ("text", 0, packed["sel"], packed["y"], 1 / 3),
                ("speech", 1, torch.arange(len(features)), packed["cb0"], 0.25 / 3),
                ("pose", 2, torch.arange(len(features)), packed["pcb0"], 0.25 / 3),
            ]:
                value, _ = distribution_kl(logits[selected], teacher[index][selected], target,
                                          packed[name + "_weight"], mask=packed[name + "_mask"])
                packed_objective = packed_objective + coefficient * value / 2  # FSDP mean.
            for j, key in enumerate(("light", "plight")):
                selected = packed[key]
                value, _ = distribution_kl(logits[selected, None, :].expand(-1, 31, -1),
                                          depth_teacher[j][selected],
                                          position_weights=torch.full((len(selected),), 2 / (31 * counts[3 + j]), dtype=torch.float64))
                packed_objective = packed_objective + (0.75 / 3) * value / 2
        # Independent reference: form each row's KL sums and divide by global
        # valid counts directly. No packing helper or normalized weights used.
        reference_objective = reference_parameter.sum() * 0
        def kl(student, teacher):
            teacher_logp = teacher.log_softmax(-1)
            return (teacher_logp.exp() * (teacher_logp - student.log_softmax(-1))).sum(-1)
        for pair, payload_pair in zip(groups, payloads):
            for item, payload in zip(pair, payload_pair):
                logits = payload["features"] @ reference_parameter
                for name, j, selected, coefficient in [
                    ("text", 0, item["sel"], 1 / 3),
                    ("speech", 1, torch.arange(len(logits)), 0.25 / 3),
                    ("pose", 2, torch.arange(len(logits)), 0.25 / 3),
                ]:
                    values = kl(logits[selected], payload["teachers"][j][selected])
                    reference_objective = reference_objective + coefficient * values[item[name + "_mask"]].sum() / counts[j]
                for j, key in enumerate(("light", "plight")):
                    selected = item[key]
                    values = kl(logits[selected, None, :].expand(-1, 31, -1), payload["depth_teachers"][j][selected])
                    reference_objective = reference_objective + (0.75 / 3) * values.sum() / (31 * counts[3 + j])
        packed_objective.backward(); reference_objective.backward()
        torch.testing.assert_close(packed_objective, reference_objective, atol=2e-8, rtol=2e-8)
        torch.testing.assert_close(packed_parameter.grad, reference_parameter.grad, atol=2e-8, rtol=2e-8)
        self.assertFalse(normalized[0]["text_weight"] is groups[-1][0]["text_weight"])
        self.assertTrue(torch.equal(groups[0][0]["text_weight"], torch.ones(3)))
        self.assertFalse(apply_global_weights(groups[0], counts, 2)[1]["pose_weight"].any())

    def test_dummy_rows_contribute_no_cb0_or_depth_gradient(self):
        real, dummy = row(5, 7), dict(row(7, 13), dummy=True)
        counts = torch.tensor([3., 5., 5., 1., 1.], dtype=torch.float64)
        packed, info = pack_prepared(apply_global_weights([real, dummy], counts, 1))
        self.assertFalse(packed["dummy"])
        self.assertEqual((info["real_rows"], info["dummy_rows"]), (1, 1))
        self.assertEqual(packed["light"].tolist(), [4])
        self.assertEqual(packed["plight"].tolist(), [3])
        self.assertFalse(packed["own"][1].any())
        self.assertTrue((packed["cb0"][5:] == -100).all())
        self.assertTrue((packed["pcb0"][5:] == -100).all())
        logits = torch.randn(12, 17, dtype=torch.float64, requires_grad=True)
        teacher = torch.randn(12, 17, dtype=torch.float64)
        loss = logits.sum() * 0
        for name, selected, target, scale in [
            ("text", packed["sel"], packed["y"], 1 / 3),
            ("speech", torch.arange(12), packed["cb0"], 0.25 / 3),
            ("pose", torch.arange(12), packed["pcb0"], 0.25 / 3),
        ]:
            value, _ = distribution_kl(logits[selected], teacher[selected], target,
                                      packed[name + "_weight"], mask=packed[name + "_mask"])
            loss = loss + scale * value
        for j, key in enumerate(("light", "plight")):
            selected = packed[key]
            value, _ = distribution_kl(logits[selected, None, :].expand(-1, 31, -1),
                                      teacher[selected, None, :].expand(-1, 31, -1),
                                      position_weights=torch.full((len(selected),), 1 / (31 * counts[3 + j])))
            loss = loss + (0.75 / 3) * value
        loss.backward()
        self.assertGreater(float(logits.grad[:5].abs().sum()), 0)
        self.assertEqual(float(logits.grad[5:].abs().sum()), 0)
        # Original dummy fallback inputs/indices are not modified in place.
        self.assertEqual(dummy["light"].tolist(), [6])
        self.assertEqual(dummy["plight"].tolist(), [5])
        self.assertTrue((dummy["cb0"] == 13).all())

    def test_all_dummy_pack_is_empty_supervision_with_valid_segments(self):
        originals = [dict(row(5, 7), dummy=True), dict(row(7, 13), dummy=True)]
        packed, info = pack_prepared(apply_global_weights(originals, torch.zeros(5), 56))
        self.assertTrue(packed["dummy"])
        self.assertEqual(packed["segs"], [5, 7])
        self.assertEqual((info["real_rows"], info["dummy_rows"]), (0, 2))
        for key in ("light", "plight"):
            self.assertEqual(packed[key].numel(), 0)
        for key in ("text_mask", "speech_mask", "pose_mask", "text_weight", "speech_weight", "pose_weight", "own"):
            self.assertFalse(packed[key].any(), key)


if __name__ == "__main__":
    unittest.main()
