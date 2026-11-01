"""
Precomputes each chunk's Orpheus hiddens (precompute/orpheus.py), one teacher-forced forward over the whole chunk
(words.json + mimi.safetensors, all 32 codebooks): <out>/<paths>/llm_hiddens.safetensors `hiddens` bf16 [T + 1, 4096].
One process per GPU: chunks are the union of the csvs, sorted, every num_shards-th from shard; existing outputs are
skipped. Logs each chunk's cb0 CE.

    python -m precompute.llm_hiddens --root R --out O --csvs a.csv b.csv --shard i --num-shards n [--weights W]

W: a dir from Orpheus.export (default: the HF checkpoint). precompute/llm_hiddens.sh runs a node's 8 GPUs.
"""
import argparse
import csv
import json
import os
import time

import torch
from safetensors.torch import load_file, save_file

from precompute.orpheus import Orpheus, assemble, chunk_words

NAME = 'llm_hiddens.safetensors'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--csvs', nargs='+', required=True)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--weights', default=None)
    args = parser.parse_args()

    rels = sorted({row['paths'].strip() for path in args.csvs for row in csv.DictReader(open(path))})
    rels = rels[args.shard::args.num_shards]
    todo = [rel for rel in rels if not os.path.exists(os.path.join(args.out, rel, NAME))]
    tag = f'[llm_hiddens:{args.shard}/{args.num_shards}]'
    print(f'{tag} {len(rels)} chunks, {len(todo)} to do', flush=True)
    model = Orpheus('cuda', args.weights)
    start = time.time()
    for i, rel in enumerate(todo):
        chunk = os.path.join(args.root, rel)
        codes = load_file(os.path.join(chunk, 'mimi.safetensors'))['codes']
        text, depth = assemble(model.tokenizer, chunk_words(json.load(open(os.path.join(chunk, 'words.json')))), codes)
        text, depth = text[None].cuda(), depth[None].cuda()
        hiddens = model(text, depth)
        ce = model.cb0_ce(hiddens, depth)
        out = os.path.join(args.out, rel)
        os.makedirs(out, exist_ok=True)
        save_file({'hiddens': hiddens[0].cpu().contiguous()}, os.path.join(out, NAME + '.tmp'))
        os.replace(os.path.join(out, NAME + '.tmp'), os.path.join(out, NAME))
        print(f'{tag} {i + 1}/{len(todo)} {rel} P={text.shape[1]} cb0_ce={ce:.3f} '
              f'{(time.time() - start) / (i + 1):.2f} s/chunk', flush=True)


if __name__ == '__main__':
    main()
