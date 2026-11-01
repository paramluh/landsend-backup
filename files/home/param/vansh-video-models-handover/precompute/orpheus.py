"""
Frozen Orpheus TTS trunk (vidcon-llm's MoshiTTSQwen, HF akv13/8b-cb0head-cd1-lightrand checkpoint-42021, Qwen3-8B)
run teacher-forced over a whole chunk, and vidcon-llm's input grid (orpheus/collators.py MoshiSpeechCollator).

For T Mimi frames the sequence has P = T + 1 positions. Position p sums the text token of instant p (subwords at
round(start / 0.08), shift_left, [W] markers, pad) and the 32 code embeddings of Mimi frame p - 1 (code_delay = 1,
codes offset by d * 2048). The final-layer hidden at p is the producer of Mimi frame p (lm_head(hidden[p]) predicts
its codebook 0); the trailing position produces nothing. The depth model is not loaded.
"""

import os

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from transformers import AutoTokenizer, Qwen3Config, Qwen3Model

REPO = 'akv13/8b-cb0head-cd1-lightrand'
CHECKPOINT = 'checkpoint-42021'
PAD = 151944
WORD = 151943
FRAME_DURATION = 0.08
CODE_DELAY = 1
DEPTH = 32
CODEBOOK_SIZE = 2048
HIDDEN_DIM = 4096


def word_groups(tokenizer, words):
    text, spans = '', []
    for i, w in enumerate(words):
        start = len(text)
        text += w['word'] if i == 0 else ' ' + w['word']
        spans.append((start, len(text)))
    enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
    return [[t for t, (s, _) in zip(enc['input_ids'], enc['offset_mapping']) if start <= s < end] for start, end in spans]


def text_frames(tokenizer, words, num_frames):
    frames = [PAD] * num_frames
    write = -1
    for w, tokens in zip(words, word_groups(tokenizer, words)):
        pos = max(round(w['start'] / FRAME_DURATION), write + 1)
        for token in tokens:
            if pos >= num_frames:
                break
            frames[pos] = token
            write = pos
            pos += 1
    for pos in range(1, num_frames):  # shift_left
        if frames[pos] != PAD and frames[pos - 1] == PAD:
            frames[pos - 1], frames[pos] = frames[pos], PAD
    for pos in range(1, num_frames):  # mark_word_starts
        if frames[pos] not in (PAD, WORD) and frames[pos - 1] == PAD:
            frames[pos - 1] = WORD
    return frames


def assemble(tokenizer, words, codes):
    """words [{word, start chunk-local s}], codes [32, T] -> text ids [T + 1], depth ids [T + 1, 32] (-1 = no codes)."""
    T = codes.shape[1]
    text = torch.tensor(text_frames(tokenizer, words, T) + [PAD] * CODE_DELAY, dtype=torch.long)
    offsets = torch.arange(DEPTH, dtype=torch.long)[:, None] * CODEBOOK_SIZE
    depth = torch.cat([torch.full((CODE_DELAY, DEPTH), -1, dtype=torch.long), (codes[:DEPTH].long() + offsets).T])
    return text, depth


def chunk_words(words_json):
    """Our chunk words.json (speaker-filtered, `start` recording-local) -> chunk-local [{word, start}]."""
    return [{'word': w['word'], 'start': w['local_start']} for w in words_json['words']]


class Orpheus(nn.Module):
    """weights: a dir written by export() (config.json, model.safetensors, tokenizer/), else the HF checkpoint."""

    def __init__(self, device='cuda', weights=None):
        super().__init__()
        if weights is None:
            snap = snapshot_download(REPO, allow_patterns=[f'{CHECKPOINT}/*', 'tokenizer/*'])
            weights, tokenizer = os.path.join(snap, CHECKPOINT), os.path.join(snap, 'tokenizer')
        else:
            tokenizer = os.path.join(weights, 'tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        config = Qwen3Config.from_pretrained(weights)
        config._attn_implementation = 'sdpa'
        with torch.device('meta'):
            self.model = Qwen3Model(config)
            self.code_embedding = nn.Embedding(DEPTH * CODEBOOK_SIZE, config.hidden_size)
            self.lm_head = nn.Linear(config.hidden_size, CODEBOOK_SIZE, bias=False)
        with safe_open(os.path.join(weights, 'model.safetensors'), 'pt') as f:
            state = {key: f.get_tensor(key).to(device, torch.bfloat16) for key in f.keys()
                     if key.startswith(('model.', 'code_embedding.', 'lm_head.'))}
        self.load_state_dict(state, assign=True)
        # inv_freq is a non-persistent buffer, so it is still on meta after the load.
        self.model.rotary_emb = type(self.model.rotary_emb)(config, device=device)
        self.requires_grad_(False)
        self.eval()
        self.config = config

    def export(self, out):
        """The bf16 trunk, code_embedding and lm_head (what the hiddens and cb0 CE need) plus the tokenizer."""
        os.makedirs(out, exist_ok=True)
        save_file({k: v.contiguous() for k, v in self.state_dict().items()}, os.path.join(out, 'model.safetensors'))
        self.config.save_pretrained(out)
        self.tokenizer.save_pretrained(os.path.join(out, 'tokenizer'))

    @torch.no_grad()
    def forward(self, text, depth):
        """text [B, P], depth [B, P, 32] -> final-layer hiddens [B, P, 4096] bf16."""
        mask = depth[:, :, 0] >= 0
        feedback = self.code_embedding(depth.clamp_min(0)).sum(dim=2) * mask[..., None].to(torch.bfloat16)
        embeds = self.model.get_input_embeddings()(text) + feedback
        return self.model(inputs_embeds=embeds, use_cache=False).last_hidden_state

    @torch.no_grad()
    def cb0_ce(self, hiddens, depth):
        """Mean codebook-0 CE of lm_head(hidden[p]) against depth row p + 1 (Mimi frame p), over valid rows."""
        targets = depth[:, 1:, 0]
        mask = targets >= 0
        logits = self.lm_head(hiddens[:, :-1][mask]).float()
        return nn.functional.cross_entropy(logits, targets[mask]).item()
