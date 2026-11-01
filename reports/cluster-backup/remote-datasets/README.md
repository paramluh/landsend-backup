# Remote dataset backup and restoration

All regular dataset files on landsend-1, 2, 3, 4, 5, and 12 were individually hashed. Identical contents were linked to node0's pinned original Hugging Face sources or its verified derivative archives. Remote-only or changed contents were uploaded to the private dataset `InternalCan/landsend-artifacts-backup-param` under `<host>/datasets/`. Two duplicate file contents are recorded in `cross-node-aliases.json`.

`VERIFIED_REMOTE_DATASETS.json` contains aggregate counts, each node's summary, and SHA256/size verification of uploaded archives and metadata. `node0-catalog.json` records every baseline file hash and source. `node0-derived-archive-proof.json` proves all 28,007 derived baseline files are present in completed node0 archives with matching metadata and no content redactions.

For a remote node:

1. Download its `<host>/datasets/verification/verification.json`, `cross-node-aliases.json`, and `<host>/datasets/COMPLETE.json` from the private backup dataset.
2. Download and extract the archives listed in that COMPLETE file. They use paths beginning `home/param/datasets/`. Preserve the manifests alongside the archives.
3. Extract node0's dataset archives into a separate node0 restoration directory. For aliases to other nodes, extract those nodes' unique archives as well. Cross-node aliases currently refer to landsend-3 only.
4. Run the helper, replacing the paths below. It obtains unchanged original files from their exact pinned Hugging Face revisions, copies verified derived files from the extracted node0 backup, restores duplicate aliases, verifies every regular file, and recreates the recorded dataset symlinks.

```bash
python3 restore_remote_dataset.py \
  --report verification.json \
  --destination restored/home/param/datasets \
  --node0-datasets node0/home/param/datasets \
  --aliases cross-node-aliases.json \
  --other-node-datasets landsend-3=node3/home/param/datasets
```

The helper needs `huggingface_hub` and authentication for the original private dataset sources. Existing files are overwritten only with `--overwrite`. Symlinks must resolve within the selected dataset directory.

The backup dataset may later have its new backup-only history squashed. Its current main branch retains the same artifact paths and verified content hashes. Pinned revisions referring to pre-existing original source datasets remain unchanged.
