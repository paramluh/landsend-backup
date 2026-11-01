"""STAGE-3 PERSONA SIDECAR reader (S3_PERSONA=1 in data_s3q; off by default, not used by training yet).

Each row shard <root>/rows-XXXXX-of-NNNNN.words.arrow has a sidecar <root>/rows-XXXXX-of-NNNNN.persona.arrow (built by
/home/param/websites/pose-duplex-engine/deps/qwen-1ch-data/persona/scripts/join_node.py; Weka copy under .../persona/sidecars/<node>/): Arrow IPC file,
zstd, record batches of exactly BATCH (schema metadata 'batch_rows', 64) rows, SAME row order as words.arrow. Columns: row_key, and per
data role side in (user, assistant): <side>_status (ok | missing | skipped_<reason> | mismatch_<reason>), <side>_note (all reasons),
<side>_system_prompt, <side>_sections_json, <side>_tokens_json, <side>_flags_json, <side>_version (content only when status == ok),
<side>_prompt_tokens (Qwen3 tokens of the system prompt, -1 if not ok), <side>_start_sec, <side>_end_sec.

get(i, row_key) -> {side: {...}} for node row i (asserts the sidecar row_key == the row's row_key).
by_channel(p, swap) -> {"A": p[role of model channel A], "B": ...}: channel A holds data role SIDES[1 if swap else 0] (data_s3q's rule)."""
import pyarrow as pa
import pyarrow.ipc as ipc

SIDES = ("user", "assistant")
FIELDS = ("status", "note", "system_prompt", "sections_json", "tokens_json", "flags_json", "version", "prompt_tokens", "start_sec", "end_sec")


def sidecar_path(root, shard):
    return f"{root}/{shard}.persona.arrow"


class PersonaSidecar:
    def __init__(self, node, root=None):
        self.node, self.root, self.rd, self.B = node, root or node["flat_root"], {}, {}

    def _open(self, k):
        if k not in self.rd:
            r = ipc.open_file(pa.memory_map(sidecar_path(self.root, self.node["shards"][k]), "r"))
            md = r.schema.metadata or {}
            assert md.get(b"format") == b"s3-persona-sidecar-v1", md.get(b"format")
            self.rd[k], self.B[k] = r, int(md[b"batch_rows"])
        return self.rd[k], self.B[k]

    def get(self, i, row_key):
        k, r = int(self.node["shard_of"][i]), int(self.node["row_of"][i])
        rd, B = self._open(k)
        b = rd.get_batch(r // B)
        rec = b.slice(r % B, 1).to_pylist()[0]
        assert rec["row_key"] == row_key, ("persona sidecar misaligned", self.node["shards"][k], r, rec["row_key"], row_key)
        return {s: {f: rec[f"{s}_{f}"] for f in FIELDS} for s in SIDES}


def by_channel(p, swap):
    a, b = SIDES[1 if swap else 0], SIDES[0 if swap else 1]
    return {"A": dict(p[a], role=a), "B": dict(p[b], role=b)}
