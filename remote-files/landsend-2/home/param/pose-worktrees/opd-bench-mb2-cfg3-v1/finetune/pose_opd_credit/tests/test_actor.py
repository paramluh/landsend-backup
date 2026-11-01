"""CPU publication, dtype preservation and distributed error agreement."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn
import torch.distributed as dist
import torch.multiprocessing as mp

from finetune.pose_opd_credit.actor import build_actor, publish_actor


class Rotary(nn.Module):
    def __init__(self, config=None, device=None):
        super().__init__()
        self.register_buffer("inv_freq", torch.arange(2, dtype=torch.float32, device=device))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.rotary_emb = Rotary()
        self.model.weight = nn.Parameter(torch.ones(2, 2))


class Tower(nn.Module):
    def __init__(self, vocab):
        super().__init__()
        self.text_emb = nn.Embedding(vocab, 2)


class PPM(nn.Module):
    """Small constructor-compatible model; no production kernels are imported."""
    fail = False

    def __init__(self, c32, c4=None, c06=None, proj_r=3, tower_vocab=8,
                 pose_head=False, decp_bb=False, vadapter=None, nopose=False):
        super().__init__()
        if self.fail:
            raise RuntimeError("constructor failed")
        self.inp = nn.Linear(2, 2)
        self.layers = nn.ModuleList([nn.Linear(2, 2) for _ in range(c32.num_hidden_layers)])
        self.head = nn.Module()
        self.head.proj = nn.Module()
        self.head.proj.G = nn.Parameter(torch.ones(proj_r, 2))
        self.tower_s, self.tower_p = Tower(tower_vocab), Tower(tower_vocab)
        self.dec_s, self.dec_p = Decoder(), Decoder()
        if decp_bb:
            self.decp_bb = nn.Linear(2, 2)
        if pose_head:
            self.pose_head = nn.Linear(2, 2)
        if vadapter:
            self.vadapter = nn.Linear(2, 2)
            self.vadapter.register_buffer("target_rms", torch.tensor(0.1234567, dtype=torch.float32))
            self.vgain = nn.Module()
            self.vgain.theta = nn.Parameter(torch.tensor([0.1234567]))


def _distributed_worker(rank, rendezvous):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import distribute_tensor, Shard, Replicate
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", world_size=2, rank=rank)
    try:
        mesh = init_device_mesh("cpu", (2,))
        expected = torch.arange(12, dtype=torch.float32).reshape(4, 3) / 13
        student = nn.Module()
        student.weight = nn.Parameter(distribute_tensor(expected, mesh, [Shard(0)]))
        student.vgain = nn.Module()
        student.vgain.theta = nn.Parameter(distribute_tensor(torch.tensor([0.1234567]), mesh, [Replicate()]))
        student.register_buffer("target_rms", torch.tensor(0.7654321))
        actor = None
        if rank == 0:
            actor = nn.Module()
            actor.weight = nn.Parameter(torch.zeros(4, 3, dtype=torch.bfloat16), requires_grad=False)
            actor.vgain = nn.Module()
            actor.vgain.theta = nn.Parameter(torch.zeros(1), requires_grad=False)
            actor.register_buffer("target_rms", torch.tensor(0.))
        result = publish_actor(student, actor)
        assert result["tensors"] == 3
        if actor is not None:
            assert torch.equal(actor.weight, expected.bfloat16())
            assert torch.equal(actor.vgain.theta, torch.tensor([0.1234567]))
            assert torch.equal(actor.target_rms, torch.tensor(0.7654321))
            actor.weight = nn.Parameter(torch.zeros(3, 3, dtype=torch.bfloat16), requires_grad=False)
        try:
            publish_actor(student, actor)
        except RuntimeError as exc:
            assert "shape/kind differs" in str(exc)
        else:
            raise AssertionError("All ranks must reject an invalid actor destination")
    finally:
        dist.destroy_process_group()


class ActorTests(unittest.TestCase):
    def test_build_and_publish_preserves_gain_buffers_and_updates_weights(self):
        cfgs = (SimpleNamespace(num_hidden_layers=4), None, None)
        student = PPM(*cfgs, vadapter={"enabled": True}, decp_bb=True)
        actor, devices = build_actor(student, cfgs, ("cpu", "cpu"), {"enabled": True}, 0.5)
        self.assertEqual(len(devices), 4)
        self.assertEqual(actor.decp_bb_alpha, 0.5)
        self.assertEqual(actor.vgain.theta.dtype, torch.float32)
        self.assertEqual(actor.vadapter.target_rms.dtype, torch.float32)
        self.assertTrue(all(not p.requires_grad for p in actor.parameters()))
        result = publish_actor(student, actor)
        self.assertGreater(result["copied_bytes"], 0)
        self.assertTrue(actor._opd_actor_published)
        with torch.no_grad():
            student.inp.weight.add_(1)
        publish_actor(student, actor)
        self.assertTrue(torch.equal(actor.inp.weight, student.inp.weight.bfloat16()))
        self.assertTrue(torch.equal(actor.vgain.theta, student.vgain.theta))

    def test_builder_restores_default_dtype_on_failure(self):
        cfgs = (SimpleNamespace(num_hidden_layers=4), None, None)
        student = PPM(*cfgs)
        original = torch.get_default_dtype()
        PPM.fail = True
        try:
            with self.assertRaisesRegex(RuntimeError, "constructor failed"):
                build_actor(student, cfgs, ("cpu", "cpu"))
        finally:
            PPM.fail = False
        self.assertEqual(torch.get_default_dtype(), original)

    def test_invalid_destination_fails_before_gather(self):
        student = nn.Linear(2, 3)
        with self.assertRaisesRegex(RuntimeError, "dtype"):
            publish_actor(student, copy.deepcopy(student))
        self.assertFalse(hasattr(student, "_opd_actor_published"))

    def test_two_rank_dtensor_cast_gather_and_collective_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(_distributed_worker, args=(str(Path(tmp) / "rendezvous"),), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
