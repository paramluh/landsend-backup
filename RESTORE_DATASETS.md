# Restore dataset contents

First extract the bulk backup with `restore_archives.py`. This recovers unique derivatives and each host's newly uploaded dataset files. Most dataset bytes already existed in other private HF repositories and are referenced by exact revision and content hash.

Restore the node-zero references:

```bash
python3 restore_existing_datasets.py \
  hf-backup/landsend-0/datasets/metadata/verified-existing-datasets.json \
  --destination restored/landsend-0
```

Then fill and verify each remote host. All paths below assume the same `hf-backup` download and `restored` extraction directories. The metadata is in the bulk HF dataset, so it remains available independently of the original cluster.

```bash
for host in landsend-1 landsend-2 landsend-3 landsend-4 landsend-5 landsend-12; do
  python3 restore_remote_dataset.py \
    --report "hf-backup/$host/datasets/verification/verification.json" \
    --destination "restored/$host/home/param/datasets" \
    --node0-datasets restored/landsend-0/home/param/datasets \
    --aliases "hf-backup/$host/datasets/verification/cross-node-aliases.json" \
    --other-node-datasets landsend-3=restored/landsend-3/home/param/datasets
done
```

The remote restore helper verifies every regular file against its recorded SHA256 or Git-blob SHA1. It refuses to overwrite mismatching files unless `--overwrite` is explicit. It downloads referenced files from pinned HF revisions and fills verified cross-host aliases. The helper also recreates and validates dataset symlinks, including aliases within the same node; artifact symlinks are restored by `restore_archives.py`.

Dataset audit summaries, upload receipts and the baseline derivative proof are under `reports/cluster-backup/remote-datasets/`.
