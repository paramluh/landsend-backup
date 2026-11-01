"""text streams past the fresh-pose end: exact per-rank Gemma cap / chat 1st-pass end, by the trainer's own rules (train_ppm.text_src rand,
8,1,1: 1 text micro per step before and after the switch; Gemma packs world-strided, never repeat; chat = cpacks[RANK::WORLD], wraps)."""
import hashlib, json, sys
N_GEMMA, N_CHAT, WORLD, SEED, T_MAX = 2585267, 1582612, 128, 20261011, int(sys.argv[1]) if len(sys.argv) > 1 else 60000
h64 = lambda s: int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")
res = []
for r in range(WORLD):
    cap_g = (N_GEMMA - 0 - r - 1) // WORLD + 1
    n_c = len(range(r, N_CHAT, WORLD))
    g = c = 0; g_end = c_end = None; at = {}
    for t in range(T_MAX):
        s = "gemma" if h64(f"ppm-text-src|{SEED}|{t}|{r}") & 1 == 0 else "chat"
        if s == "gemma":
            if g + 1 > cap_g and g_end is None: g_end = t
            g += 1
        else:
            if c + 1 > n_c and c_end is None: c_end = t
            c += 1
        if t + 1 in (16861, 17352): at[t + 1] = (g, c)
        if g_end is not None and c_end is not None: break
    res.append({"rank": r, "cap_gemma": cap_g, "chat_per_rank": n_c, "gemma_end": g_end, "chat_2ndpass_from": c_end,
                "used_by_17352": at.get(17352), "used_by_16861": at.get(16861)})
ge = [x["gemma_end"] for x in res if x["gemma_end"] is not None]; ce = [x["chat_2ndpass_from"] for x in res if x["chat_2ndpass_from"] is not None]
print(json.dumps({"gemma_end_min": min(ge) if len(ge) == WORLD else None, "gemma_end_max": max(ge) if ge else None,
                  "chat_2ndpass_min": min(ce) if len(ce) == WORLD else None, "chat_2ndpass_max": max(ce) if ce else None,
                  "rank0": res[0], "gemma_left_after_17352_min": min(x["cap_gemma"] - x["used_by_17352"][0] for x in res),
                  "chat_left_after_17352_min": min(x["chat_per_rank"] - x["used_by_17352"][1] for x in res)}))
