from pathlib import Path
p=Path('/home/param/pose-worktrees/pose-basic-opd/finetune/pose_opd_credit/runtime.py')
s=p.read_text().replace("if role == 'audio_teacher':\n        return {'hn': hn", "if role in ('audio_teacher', 'self_teacher'):\n        return {'hn': hn")
s=s.replace("    loss, kl, lp = chunked_text_objective(\n", "    if b.get('basic_opd', False):\n        from .basic_opd import chunked_sampled_text_objective\n        loss, kl, lp = chunked_sampled_text_objective(\n            x, self.lm_head.weight, replacement, teacher, b['text_weight'],\n            special_ids=M.SPEC, executable_vocab=M.TOWER_VOCAB,\n            chunk_size=b['text_chunk_frames'], lexical_weight=b['text_lexical_weight'])\n        return {'hn': hn, 'loss': loss, 'kl': kl, 'logp': lp}\n    loss, kl, lp = chunked_text_objective(\n")
p.write_text(s)
