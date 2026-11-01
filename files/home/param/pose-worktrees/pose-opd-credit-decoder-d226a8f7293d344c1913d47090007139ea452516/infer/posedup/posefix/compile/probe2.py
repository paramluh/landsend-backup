import torch, torch._dynamo as dyn
from torch import Tensor
import torch._inductor.config as ic
ic.fallback_random = True
dev = torch.device("cuda:0"); torch.cuda.set_device(dev)
_GENS = []

def reg(g):
    _GENS.append(g); return len(_GENS) - 1

@torch.library.custom_op("probe::mn1", mutates_args=())
def mn1(p: Tensor, gid: int) -> Tensor:
    return torch.multinomial(p, 1, generator=_GENS[gid])

@mn1.register_fake
def _(p, gid):
    return p.new_empty(p.shape[0], 1, dtype=torch.long)

g1 = torch.Generator(device=dev); g1.manual_seed(5)
G1 = reg(g1)

@torch._dynamo.assume_constant_result
def gid_of(g):
    for i, x in enumerate(_GENS):
        if x is g:
            return i
    raise KeyError

def samp_a(z):            # gid via assume_constant_result on the generator
    return mn1(torch.softmax(z, -1), gid_of(g1))

class Holder:
    pass
H = Holder(); H.gen = g1

def samp_b(z):            # dict by id
    return mn1(torch.softmax(z, -1), _GENS.index(H.gen))

z = torch.randn(2, 64, device=dev)
for nm, fn in (("assume_const", samp_a), ("list.index", samp_b)):
    try:
        ex = dyn.explain(fn)(z)
        print(nm, "graphs", ex.graph_count, "breaks", ex.graph_break_count, [str(b.reason)[:300] for b in ex.break_reasons][:2], flush=True)
    except Exception as e:
        print(nm, "FAIL", repr(e)[:300], flush=True)
# numerics: compiled vs eager with same seed
def samp_c(z, gid: int):
    return mn1(torch.softmax(z / 0.7, -1), gid)
g1.manual_seed(5); e = [samp_c(z, G1) for _ in range(5)]
cf = torch.compile(samp_c, mode="max-autotune-no-cudagraphs", dynamic=False)
g1.manual_seed(5); c = [cf(z, G1) for _ in range(5)]
print("eager==compiled samples:", all(torch.equal(a, b) for a, b in zip(e, c)), flush=True)
ex = dyn.explain(samp_c)(z, G1); print("int gid arg: graphs", ex.graph_count, "breaks", ex.graph_break_count, flush=True)
# graph capture of the compiled fn with registered generator
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for _ in range(2): cf(z, G1)
torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
gr = torch.cuda.CUDAGraph(); gr.register_generator_state(g1)
with torch.cuda.graph(gr, stream=s):
    out = cf(z, G1)
g1.manual_seed(9); ref = [samp_c(z, G1).clone() for _ in range(3)]
g1.manual_seed(9); rep = []
for _ in range(3):
    gr.replay(); rep.append(out.clone())
print("graph replay == eager:", all(torch.equal(a, b) for a, b in zip(ref, rep)), flush=True)
