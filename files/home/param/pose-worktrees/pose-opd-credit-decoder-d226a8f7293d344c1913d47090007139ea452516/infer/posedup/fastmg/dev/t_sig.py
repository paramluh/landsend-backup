"""cost of the flag-raise primitives inside a 3-program kernel that stores 3 x 5120 fp32 rows to 4 (local) destinations."""
import torch, triton, triton.language as tl, json
d = torch.device("cuda:0")


@triton.jit
def k(PART, DST, FLG, N, V: tl.constexpr, G: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    m = offs < N
    x = tl.load(PART + row * N + offs, mask=m, other=0.0)
    for g in tl.static_range(G):
        dd = tl.load(DST + g).to(tl.pointer_type(tl.float32)) + row * N
        tl.store(dd + offs, x, mask=m)
    tl.debug_barrier()
    if V == 0:
        for g in tl.static_range(G):
            f = tl.load(FLG + g).to(tl.pointer_type(tl.int64)) + row
            tl.atomic_xchg(f, 7, sem="release", scope="sys")
    elif V == 1:
        tl.inline_asm_elementwise("fence.acq_rel.sys; mov.u32 $0, $1;", "=r,r", [row], dtype=tl.int32, is_pure=False, pack=1)
        for g in tl.static_range(G):
            f = tl.load(FLG + g).to(tl.pointer_type(tl.int64)) + row
            tl.store(f, 7)
    elif V == 2:
        for g in tl.static_range(G):
            f = tl.load(FLG + g).to(tl.pointer_type(tl.int64)) + row
            tl.atomic_xchg(f, 7, sem="release", scope="gpu")
    elif V == 4:
        for g in tl.static_range(G):
            f = tl.load(FLG + g).to(tl.pointer_type(tl.int64)) + row
            tl.inline_asm_elementwise("st.release.sys.global.u64 [$1], $2; mov.u32 $0, 0;", "=r,l,l", [f, tl.full([], 7, tl.int64)],
                                      dtype=tl.int32, is_pure=False, pack=1)


N, G = 5120, 4
part = torch.randn(3, N, device=d)
bufs = [torch.zeros(3, N, device=d) for _ in range(G)]
flags = torch.zeros(G, 8, dtype=torch.int64, device=d)
DST = torch.tensor([b.data_ptr() for b in bufs], dtype=torch.int64, device=d)
FLG = torch.tensor([flags[g].data_ptr() for g in range(G)], dtype=torch.int64, device=d)
res = {}
for V in (3, 0, 1, 2, 4):
    for nw in (4, 8):
        fn = lambda: k[(3,)](part, DST, FLG, N, V=V, G=G, BLOCK=8192, num_warps=nw)
        try:
            s = torch.cuda.Stream(d); s.wait_stream(torch.cuda.current_stream(d))
            with torch.cuda.stream(s):
                fn(); fn()
            torch.cuda.synchronize(d)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                for _ in range(200):
                    fn()
            g.replay(); torch.cuda.synchronize(d)
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); g.replay(); e1.record(); torch.cuda.synchronize(d)
            res[f"V{V}_w{nw}"] = round(e0.elapsed_time(e1) / 200 * 1000, 2)
        except Exception as ex:
            res[f"V{V}_w{nw}"] = str(ex)[:200]
print(json.dumps(res))
