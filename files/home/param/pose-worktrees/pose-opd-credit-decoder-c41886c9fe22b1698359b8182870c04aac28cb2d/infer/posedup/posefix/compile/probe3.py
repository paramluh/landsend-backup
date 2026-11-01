import sys
sys.path.insert(0, "/data/home/amartyavarma/ppm/aloe-m1vox-ft/code/vendor")
import torch
import pinned_fa3
m = pinned_fa3._load_module()
fi = m.flash_attn_interface
op = fi.ops.fwd
print(type(op), op)
pk = op.default if hasattr(op, "default") else op
print("SCHEMA", pk._schema)
for k in ("Meta", "CUDA", "CompositeImplicitAutograd"):
    try:
        print(k, torch._C._dispatch_has_kernel_for_dispatch_key(pk.name(), k))
    except Exception as e:
        print(k, "?", e)
import torch._library.fake_impl as fi_
print("fake registered:", torch._library.simple_registry.singleton.find(pk.name()).fake_impl.kernel is not None)
