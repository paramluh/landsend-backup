"""D32 test 5 (infer side): infer_ppm loads a consolidated DCP export (w-N) and a ckpt_local_export copy (local-N) of a checkpoint WITH the
training-only pose_head: both load (pose_head ignored), every inference tensor equal between the two, and the pose_head is not in the model.
usage: python tests/test_d32_infer.py <w-N dir> <local-N dir>"""
import sys
import torch
sys.path.insert(0, "/data/home/amartyavarma/ppm/code")
import infer_ppm as I  # noqa: E402

ms = []
for p in sys.argv[1:3]:
    ci = I.ckpt_info(p)
    if ci["kind"] == "dcp":
        assert "pose_head.weight" in ci["shapes"], "the export should carry the head (training state)"
    m, _, _ = I.build_model(ci, ["cuda:0"])
    assert not any(n.startswith("pose_head") for n, _ in m.named_parameters())
    ms.append({n: p_.detach().cpu() for n, p_ in m.named_parameters()})
    del m
    torch.cuda.empty_cache()
a, b = ms
assert set(a) == set(b)
bad = [n for n in a if not torch.equal(a[n], b[n])]
print(f"[infer] DCP export and local export: {len(a)} inference tensors, equal: {len(a) - len(bad)}; differing: {bad[:4]}")
assert not bad
print("[infer] PASS: infer_ppm loads both formats with the extra pose_head keys ignored")
