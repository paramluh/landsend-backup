import argparse
import gc
import math
import os
import shutil
import time
from contextlib import nullcontext
from datetime import timedelta
from functools import partial
from pathlib import Path

import toml
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from dataloading.ds_aloe import AloeChunkSource, AloeEvalClips, PretrainChunkSource, ShuffledDenseWindows
from dataloading.ds_words import WordsChunkSource, WordsDenseWindows, WordsEvalClips
from dataloading.sources import MIN_LATENT_FRAMES
from learn_pose import (
    DDP_BUCKET_MB, DIAGNOSTICS_EVERY, PIPELINES, LayeredModel, SAVE_DTYPE, build_pretrain_source, make_loader,
)
from models.pose.settings import FPS, VAE_TEMPORAL_STRIDE
from utils.harness import (
    build_ema, build_lr_scheduler, build_optimizer, check_manual_signals, get_most_recent_run_dir,
    get_next_run_dir, load_optim_state, save_model_dir, save_optim_state, wandb_resume_kwargs,
)
from utils.common import DTYPE_MAP, empty_cuda_cache, is_main_process
from utils.evaluation import evaluate_suites
from utils import saver


# torch.compile builds its graphs in the first steps and the first eval, after the setup gc.freeze(); left unfrozen,
# every gen-2 collection rescans them on one rank and stalls the whole DP group at the all-reduce.
GC_REFREEZE_STEPS = 4
REQUIRED_KEYS = (
    'save_every_n_steps', 'epochs', 'lr', 'min_lr', 'warmup_steps', 'gradient_clipping',
    'micro_batch_size_per_gpu', 'gradient_accumulation_steps', 'num_dataloader_workers',
    'num_history', 'num_present', 'history_dropout', 'output_dir', 'root_dir', 'clipnames_root',
    'eval_csvs', 'eval_every_n_steps', 'eval_num_dataloader_workers',
    'eval_micro_batch_size_per_gpu', 'activation_checkpointing', 'compile_blocks',
    'activation_checkpoint_skip',
)


def load_config(path):
    config = toml.load(path)
    assert config['model']['type'] in PIPELINES, config['model']['type']
    for key in REQUIRED_KEYS:
        assert key in config, key
    config['save_dtype'] = DTYPE_MAP[config.get('save_dtype', SAVE_DTYPE)]
    config['model']['dtype'] = DTYPE_MAP[config['model']['dtype']]
    config['model']['transformer_dtype'] = DTYPE_MAP[config['model']['transformer_dtype']]
    if 'resume_from' in config and not config.get('node_sharded'):
        config['model']['checkpoint_path'] = os.path.join(config['resume_from'], 'model.safetensors')
    return config


def build_data(config, rank, world_size):
    B, GA = config['micro_batch_size_per_gpu'], config['gradient_accumulation_steps']
    H, P, dropout = config['num_history'], config['num_present'], config['history_dropout']
    refs = config.get('refs')
    # [refs] bank: the actor bank root; eval sets read its train/ split unless refs.eval_banks names another.
    bank = lambda split: refs and os.path.join(refs['bank'], split)
    eval_banks = refs.get('eval_banks', {}) if refs else {}
    Source, Windows, Clips = AloeChunkSource, ShuffledDenseWindows, AloeEvalClips
    if 'llm_hiddens_root' in config:
        Source = partial(WordsChunkSource, hiddens_root=config['llm_hiddens_root'])
        Windows, Clips = WordsDenseWindows, WordsEvalClips
    if 'buckets' in config:
        train_source = PretrainChunkSource(build_pretrain_source(config, config['clipnames_root'], config['buckets']), config['root_dir'])
    else:
        train_source = Source(config['root_dir'], config['clipnames_root'], config['train_csv'],
                              bank('train'), pose_file=config['pose_latents'], mimi_file=config['mimi_codes'],
                              bad_audio=config.get('bad_audio', 0))
    # node_sharded: train_csv lists only the chunks on this node's disk, so the node's ranks split it among themselves.
    sharded = config.get('node_sharded', False)
    data_rank, data_world = (int(os.environ['LOCAL_RANK']), int(os.environ['LOCAL_WORLD_SIZE'])) if sharded else (rank, world_size)
    kwargs = {'windows_per_epoch': config['windows_per_epoch']} if 'windows_per_epoch' in config else {}
    train_data = Windows(
        train_source, H, P, dropout, data_rank, data_world, B, GA,
        refs and (refs['drop_counts'], refs['drop_p']),
        saver.gloo if sharded else None, **kwargs,
    )
    eval_batch = config['eval_micro_batch_size_per_gpu']
    eval_key = lambda key: config[f'eval_{key}' if f'eval_{key}' in config else key]
    eval_data = {
        name: Clips(
            Source(eval_key('root_dir'), eval_key('clipnames_root'), csv_name,
                   bank(eval_banks.get(name, 'train')), pose_file=eval_key('pose_latents'),
                   mimi_file=eval_key('mimi_codes')),
            H, P, rank, world_size, eval_batch,
        )
        for name, csv_name in config['eval_csvs'].items()
    }
    return train_data, eval_data, B * world_size * GA


def _step_uniques(chunks, people):
    chunks = torch.cat(chunks)
    people = torch.cat(people)
    gathered_c = [torch.empty_like(chunks) for _ in range(dist.get_world_size())]
    gathered_p = [torch.empty_like(people) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_c, chunks)
    dist.all_gather(gathered_p, people)
    return int(torch.unique(torch.cat(gathered_c)).numel()), int(torch.unique(torch.cat(gathered_p)).numel())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to TOML configuration file.')
    args = parser.parse_args()
    config = load_config(args.config)

    torch.multiprocessing.set_sharing_strategy('file_system')
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // int(os.environ.get('LOCAL_WORLD_SIZE', world_size))))
    # All blocks share one compiled code object. Dynamo guards grad mode and num_threads, and the
    # checkpointed first step recomputes on the 1-thread autograd thread, so step 1, the steady state
    # and eval each add variants; past the default limit of 8 every block silently runs eager.
    torch._dynamo.config.recompile_limit = 64

    GA = config['gradient_accumulation_steps']
    assert config['num_present'] < MIN_LATENT_FRAMES, \
        f'num_present={config["num_present"]} must be < MIN_LATENT_FRAMES={MIN_LATENT_FRAMES}'

    # No shared filesystem: only rank 0 has the run dir, and checkpoints go through utils.saver.
    rank0_disk = config.get('node_sharded', False)
    if rank0_disk:
        saver.init_gloo()
    meta = None
    if rank0_disk and 'resume_from' in config:
        meta = saver.load_meta(Path(config['resume_from']))

    model = PIPELINES[config['model']['type']](config)
    train_data, eval_data, global_batch_size = build_data(config, rank, world_size)
    steps_per_epoch = len(train_data) // GA
    total_steps = config['epochs'] * steps_per_epoch
    eval_loaders = {}
    for name, dataset in eval_data.items():
        if not len(dataset):
            raise RuntimeError(f'Eval set {name} is smaller than one global microbatch')
        eval_loaders[name] = make_loader(config, dataset, config['eval_num_dataloader_workers'], range(len(dataset)))

    model.load_diffusion_model()
    layered = LayeredModel(model.to_layers(), config['activation_checkpointing']).to('cuda')
    parameters_to_train = [p for p in layered.parameters() if p.requires_grad]
    # Null/drop tokens are trained but left out of the clip norm: a near-zero token before a no-affine LayerNorm gets
    # 1/sqrt(eps) times larger gradients, which would dominate it.
    parameters_to_clip = [p for p in parameters_to_train
                          if not getattr(p, 'original_name', '').endswith(('null_token', 'no_audio_tokens', 'drop_token'))]
    optimizer = build_optimizer(config, model, parameters_to_train)
    ddp_model = DDP(layered, device_ids=[local_rank], broadcast_buffers=False, gradient_as_bucket_view=True, bucket_cap_mb=DDP_BUCKET_MB)
    ema = build_ema(config, optimizer, layered)
    lr_scheduler = build_lr_scheduler(config, optimizer, steps_per_epoch)
    loss_fn = model.get_loss_fn()
    initial_layer = layered.layers[0]
    audio_layers = [layer for layer in layered.layers if getattr(layer, 'audio_block', None) is not None]

    step = 1
    examples = global_batch_size
    wandb_resume = {}
    if 'resume_from' in config:
        if rank0_disk:
            saver.load(Path(config['resume_from']), meta, optimizer, ema, steps_per_epoch)
        else:
            meta = load_optim_state(Path(config['resume_from']), optimizer, ema, steps_per_epoch)
        for _ in range(meta['step']):
            lr_scheduler.step()
        step = meta['step'] + 1
        examples = meta['examples'] + global_batch_size
        wandb_resume = wandb_resume_kwargs(config, meta)
        if is_main_process():
            print(f'Resumed from {config["resume_from"]} at step {step}')

    train_loader = make_loader(config, train_data, config['num_dataloader_workers'],
                               range((step - 1) * GA, total_steps * GA))
    data_iter = iter(train_loader)

    if is_main_process():
        run_dir = get_next_run_dir(config['output_dir'])
        os.makedirs(run_dir)
        shutil.copy(args.config, run_dir)
        print(f'Global batch size = {global_batch_size} windows, {steps_per_epoch} steps/epoch')
    dist.barrier()
    if rank0_disk:
        run_dir = Path(run_dir) if is_main_process() else None
    else:
        run_dir = Path(get_most_recent_run_dir(config['output_dir']))

    wandb_enable = 'wandb' in config and is_main_process()
    if wandb_enable:
        import wandb
        wandb.init(project=config['wandb']['project'], name=config['wandb'].get('run_name'), config=config, dir=run_dir, **wandb_resume)
        wandb.define_metric('step')
        wandb.define_metric('*', step_metric='step')

    def run_eval(eval_step, eval_examples):
        if not eval_loaders:
            return
        empty_cuda_cache()
        log_dict = {}
        by_name = {}
        for name, loader in eval_loaders.items():
            results = by_name[name] = evaluate_suites(model, layered, loss_fn, loader)
            if is_main_process():
                for (suite, slot), value in results.items():
                    print(f'[eval] step={eval_step} {name}/{suite}/slot-{slot} loss={value:.4f}')
            if wandb_enable:
                for (suite, slot), value in results.items():
                    if suite.endswith('/swapped-refs'):
                        real = suite.removesuffix('/swapped-refs')
                        log_dict[f'eval/swapped/{name}/{real}/slot-{slot}'] = value
                        log_dict[f'diagnostics/{name}-swapped-minus-real/{real}/slot-{slot}'] = value - results[(real, slot)]
                    else:
                        log_dict[f'eval/{name}/{suite}/slot-{slot}'] = value
        if wandb_enable and {'seen', 'unseen'} <= by_name.keys():
            seen, unseen = by_name['seen'], by_name['unseen']
            for (suite, slot), value in seen.items():
                if (suite, slot) in unseen and not suite.endswith('/swapped-refs'):
                    log_dict[f'diagnostics/unseen-minus-seen-percentage/{suite}/slot-{slot}'] = 100 * (unseen[(suite, slot)] - value) / value
        if ema is not None:
            with ema.swapped_in(optimizer, config['save_dtype']):
                ema_by_name = {name: evaluate_suites(model, layered, loss_fn, loader, swap_refs=False)
                               for name, loader in eval_loaders.items()}
            for name, results in ema_by_name.items():
                if is_main_process():
                    for (suite, slot), value in results.items():
                        print(f'[eval-ema] step={eval_step} {name}/{suite}/slot-{slot} loss={value:.4f}')
                if wandb_enable:
                    for (suite, slot), value in results.items():
                        log_dict[f'ema/{name}/{suite}/slot-{slot}'] = value
                        log_dict[f'diagnostics/{name}-ema-minus-raw/{suite}/slot-{slot}'] = value - by_name[name][(suite, slot)]
        if wandb_enable:
            log_dict['step'] = eval_examples if 'x_axis_examples' in config else eval_step
            wandb.log(log_dict)
        empty_cuda_cache()

    if config.get('eval_step_zero') and 'resume_from' not in config:
        run_eval(0, 0)

    epoch_loss = 0.0
    epoch_steps = 0
    saved_step = None
    empty_cuda_cache()
    gc.freeze()

    first_step = step
    patch_volume = math.prod(model.model_arch['patch_size'])
    while True:
        step_start = time.perf_counter()
        layered.checkpoint_all = step == first_step
        step_loss = torch.zeros((), device='cuda')
        step_chunks, step_people = [], []
        step_tokens = 0
        last = step >= total_steps or ('max_steps' in config and step >= config['max_steps'])
        for micro in range(GA):
            batch = next(data_iter)
            step_tokens += batch['latents_low'][:, 0].numel() // patch_volume
            if 'ref_keep' in batch:
                step_tokens += batch['ref_latents'][:, 0].numel() // patch_volume
            step_chunks.append(batch['chunk_index'].to('cuda', non_blocking=True))
            step_people.append(batch['person_index'].to('cuda', non_blocking=True))
            features, labels = model.prepare_inputs(batch)
            with (nullcontext() if micro == GA - 1 or step == first_step else ddp_model.no_sync()):
                loss = loss_fn(ddp_model(features), labels)
                (loss / GA).backward()
            step_loss += loss.detach()
        step_loss /= GA
        dist.all_reduce(step_loss, op=dist.ReduceOp.AVG)
        n_chunks, n_people = _step_uniques(step_chunks, step_people)

        clip = config['gradient_clipping']
        grad_norm = nn.utils.clip_grad_norm_(parameters_to_clip, clip if clip > 0 else float('inf'))
        optimizer.step()
        if ema is not None:
            ema.update()
        lr_scheduler.step()
        optimizer.zero_grad()

        loss_value = step_loss.item()
        grad_norm_value = grad_norm.item()
        step_time = time.perf_counter() - step_start
        epoch_loss += loss_value
        epoch_steps += 1
        x_axis = examples if 'x_axis_examples' in config else step
        if is_main_process():
            print(
                f'step={step} loss={loss_value:.4f} lr={lr_scheduler.get_last_lr()[0]:.3e} '
                f'grad_norm={grad_norm_value:.3f} chunks={n_chunks} people={n_people}'
            )

        bucket_losses = model.get_train_loss_buckets()
        model.reset_train_loss_buckets()
        if wandb_enable:
            last_lrs = lr_scheduler.get_last_lr()
            log_dict = {
                'train/loss': loss_value, 'train/lr': last_lrs[0], 'train/grad_norm': grad_norm_value,
                'data/window_hours': examples * (model.num_history + model.num_present) * VAE_TEMPORAL_STRIDE / FPS / 3600,
                'data/step_chunks': n_chunks, 'data/step_people': n_people, 'step': x_axis,
                'data/step_time': step_time, 'data/tokens_per_s': step_tokens / step_time,
            }
            groups = config['optimizer'].get('groups', [])
            for gid, lr_value in zip(optimizer.group_ids[1:], last_lrs[1:]):
                log_dict[f'train/lr_{groups[gid - 1]["prefix"].rstrip(".")}'] = lr_value
            for bucket_name, avg_loss in bucket_losses.items():
                log_dict[f'loss_buckets/{bucket_name}'] = avg_loss
            if step % DIAGNOSTICS_EVERY == 0:
                for i, block in enumerate(model.audio_adapter.blocks):
                    log_dict[f'audio/out_proj_{i}'] = block.out_proj.weight.detach().norm().item()
                for i, block in enumerate(model.hidden_adapter.blocks if model.hidden_adapter else ()):
                    log_dict[f'hidden/out_proj_{i}'] = block.out_proj.weight.detach().norm().item()
                for i, layer in enumerate(audio_layers):
                    ratios = (layer.audio_injection_ratio / layer.audio_injection_count).tolist()
                    for slot, ratio in enumerate(ratios, start=1):
                        log_dict[f'audio/inject_ratio_{i}/slot-{slot}'] = ratio
                if initial_layer.audio_drop_seen:
                    log_dict['train/audio_dropped'] = initial_layer.audio_dropped.item() / initial_layer.audio_drop_seen
            wandb.log(log_dict)
        if step % DIAGNOSTICS_EVERY == 0:
            for layer in audio_layers:
                layer.audio_injection_ratio = 0
                layer.audio_injection_count = 0
            initial_layer.audio_dropped.zero_()
            initial_layer.audio_drop_seen = 0

        if step % steps_per_epoch == 0:
            if wandb_enable:
                wandb.log({'train/epoch_loss': epoch_loss / epoch_steps, 'epoch': step // steps_per_epoch, 'step': x_axis})
            epoch_loss = 0.0
            epoch_steps = 0

        manual_save, manual_save_optim, manual_quit = check_manual_signals(run_dir)
        save_optim = manual_save_optim or manual_quit or (
            'save_optimizer_every_n_steps' in config and step % config['save_optimizer_every_n_steps'] == 0)
        if save_optim or manual_save or step % config['save_every_n_steps'] == 0:
            if rank0_disk:
                saver.save(run_dir and run_dir / f'step{step}', model, layered, ema, optimizer, config,
                           args.config, save_optim, step, examples, steps_per_epoch,
                           wandb.run if wandb_enable else None)
            else:
                save_dir = run_dir / f'step{step}'
                save_model_dir(save_dir, model, layered, ema, config, args.config)
                if save_optim:
                    save_optim_state(save_dir, optimizer, ema, step, examples, steps_per_epoch, wandb.run if wandb_enable else None)
            saved_step = step
        if manual_quit:
            break

        if eval_loaders and step % config['eval_every_n_steps'] == 0:
            run_eval(step, examples)
            gc.freeze()
        if step < first_step + GC_REFREEZE_STEPS:
            gc.freeze()
        if last:
            break
        step += 1
        examples += global_batch_size

    if saved_step != step and rank0_disk:
        saver.save(run_dir and run_dir / f'step{step}', model, layered, ema, optimizer, config,
                   args.config, False, step, examples, steps_per_epoch)
    elif saved_step != step:
        save_model_dir(run_dir / f'step{step}', model, layered, ema, config, args.config)
    if is_main_process():
        print('TRAINING COMPLETE!')
    dist.destroy_process_group()
