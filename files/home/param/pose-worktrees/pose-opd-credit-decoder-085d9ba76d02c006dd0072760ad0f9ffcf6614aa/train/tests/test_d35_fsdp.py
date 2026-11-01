"""D35: FSDP2 check (2 GPUs): a fully_shard'ed head (own unit, as model_ppm.build shards pose_head) whose input is DETACHED (s = 0) still gets
its gradient reduce-scattered, equal to the non-detached run's head grad; the 'backbone' unit gets none from the head."""
import os, sys
sys.path.insert(0, sys.argv[1])
import torch, torch.distributed as dist, torch.nn as nn
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
import model_ppm as M
dist.init_process_group("nccl"); r = dist.get_rank(); torch.cuda.set_device(r); dev = torch.device("cuda", r)
class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.bb = nn.Linear(64, 512); self.head = nn.Linear(512, 2048)
    def forward(self, x):
        return self.bb(x)
def run(s):
    torch.manual_seed(0)
    m = Net().to(dev)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(m.head, mp_policy=mp); fully_shard(m.bb, mp_policy=mp); fully_shard(m, mp_policy=mp)
    g = torch.Generator(device=dev).manual_seed(1 + r); x = torch.randn(300, 64, device=dev, generator=g); lab = torch.randint(0, 2048, (300,), device=dev, generator=g)
    hn = m(x)
    tower = hn.float().pow(2).mean()
    ce = nn.functional.cross_entropy(m.head(M.posehead_in(hn, s)).float(), lab)
    (tower + ce).backward()
    return {n: p.grad.full_tensor().clone() for n, p in m.named_parameters()}
a, b, t = run(None), run(0.0), None
ok = all(torch.equal(a[n], b[n]) for n in a if n.startswith("head"))
bbdiff = max(float((a[n] - b[n]).abs().max()) for n in a if n.startswith("bb"))
if r == 0: print(f"FSDP2 detached-input head grads == attached: {ok}; head grads nonzero: {all(float(b[n].abs().sum()) > 0 for n in b if n.startswith('head'))}; bb grad differs (head part removed): {bbdiff:.3e}")
dist.destroy_process_group()
