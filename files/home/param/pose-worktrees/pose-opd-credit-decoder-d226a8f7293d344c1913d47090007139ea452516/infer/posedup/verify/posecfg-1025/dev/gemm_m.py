# micro test: is F.linear row r bitwise the same for M = 1, 2, 3, 4 rows (bf16, cuBLAS) at the trunk / tower shapes?
import torch, torch.nn.functional as F
torch.manual_seed(0)
dev = "cuda:0"
shapes = {"bb_q": (8192, 5120), "bb_k": (1024, 5120), "bb_o": (5120, 8192), "bb_gate": (25600, 5120), "bb_down": (5120, 25600),
          "tw_q": (4096, 2560), "tw_k": (1024, 2560), "tw_o": (2560, 4096), "tw_gate": (9728, 2560), "tw_down": (2560, 9728),
          "head_proj": (2048, 5120), "cond": (2560, 5120), "decbb": (1024, 5120)}
with torch.no_grad():
    for n, (o, i) in shapes.items():
        W = (torch.randn(o, i, device=dev) * 0.02).to(torch.bfloat16)
        x = torch.randn(1, 4, i, device=dev).to(torch.bfloat16)
        ref = {M: F.linear(x[:, :M], W) for M in (1, 2, 3, 4)}
        res = []
        for a, b in ((1, 2), (2, 3), (1, 3), (3, 4), (2, 4)):
            res.append(f"{a}v{b}:{'=' if torch.equal(ref[a][:, :a], ref[b][:, :a]) else 'DIFF'}")
        print(n, " ".join(res), flush=True)
