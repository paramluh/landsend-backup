"""
universal_wide_bimodal training: every optimizer step runs both modes on each of its GA microbatches, stabilising then
resolving, forward and backward, with DDP syncing only on the step's last backward. Each mode's loss is weighted by
its share of the present frames (num_present / total) and / GA, so the objective is per present frame.
"""
import argparse
import gc
import math
import os
import shutil
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import toml
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

from dataloading.ds_bimodal import BimodalWindows
from dataloading.sources import MIN_LATENT_FRAMES
from learn_aloe import GC_REFREEZE_STEPS, REQUIRED_KEYS, _step_uniques, build_data
from learn_pose import DDP_BUCKET_MB, DIAGNOSTICS_EVERY, SAVE_DTYPE, LayeredModel, make_loader
from models.bimodal.wan_bimodal import TAG, BimodalWidePipeline
from models.pose.settings import FPS, VAE_TEMPORAL_STRIDE
from utils.harness import (
    build_ema, build_lr_scheduler, build_optimizer, check_manual_signals, get_most_recent_run_dir,
    get_next_run_dir, load_optim_state, save_model_dir, save_optim_state, wandb_resume_kwargs,
)
from utils.common import DTYPE_MAP, empty_cuda_cache, is_main_process
from utils.evaluation import evaluate_suites
from utils import saver


MODES = ('stabilising', 'resolving')
RECOMPILE_LIMIT = 64


class BimodalLayers(nn.Module):
    """
    Each mode's LayeredModel over one set of modules, so DDP sees every parameter once and every backward reaches all.
    """

    def __init__(self, stacks, checkpoint_blocks):
        super().__init__()
        self.stacks = nn.ModuleDict({mode: LayeredModel(layers, checkpoint_blocks) for mode, layers in stacks.items()})

    def forward(self, inputs, mode):
        return self.stacks[mode](inputs)


@torch.no_grad()
def present_loss_sums(output, labels, num_clean):
    """
    Per window, the summed MSE of its valid present frames and their count.

    -> sums [B], counts [B]
    """
    target, loss_mask = labels
    frame_loss = (output.float() - target.float()).square().mean(dim=(1, 3, 4))[:, num_clean:]
    valid = loss_mask[:, 0, num_clean:, 0, 0] > 0
    return (frame_loss * valid).sum(dim=1), valid.sum(dim=1)


def load_config(path):
    config = toml.load(path)
    assert config['model']['type'] == TAG, config['model']['type']
    assert not config.keys() & config['stabilising'].keys(), 'shared keys stay at the top level'
    config.update(config.pop('stabilising'))
    for key in REQUIRED_KEYS:
        assert key in config, key
    assert not {'refs', 'llm_hiddens_root', 'history_corruption'} & config.keys(), 'not carried into resolving windows'
    config['save_dtype'] = DTYPE_MAP[config.get('save_dtype', SAVE_DTYPE)]
    config['model']['dtype'] = DTYPE_MAP[config['model']['dtype']]
    config['model']['transformer_dtype'] = DTYPE_MAP[config['model']['transformer_dtype']]
    if 'resume_from' in config and not config['node_sharded']:
        config['model']['checkpoint_path'] = os.path.join(config['resume_from'], 'model.safetensors')
    return config


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
    torch._dynamo.config.recompile_limit = RECOMPILE_LIMIT
    # The modes share compiled code objects: one static graph per mode geometry and batch size, not one dynamic graph.
    torch._dynamo.config.automatic_dynamic_shapes = False

    GA = config['gradient_accumulation_steps']
    assert config['num_present'] < MIN_LATENT_FRAMES, \
        f'num_present={config["num_present"]} must be < MIN_LATENT_FRAMES={MIN_LATENT_FRAMES}'

    # No shared filesystem: only rank 0 has the run dir, and checkpoints go through utils.saver.
    rank0_disk = config['node_sharded']
    if rank0_disk:
        saver.init_gloo()
    meta = None
    if rank0_disk and 'resume_from' in config:
        meta = saver.load_meta(Path(config['resume_from']))

    model = BimodalWidePipeline(config)
    stabilising_data, eval_data, global_batch_size = build_data(config, rank, world_size)
    train_data = BimodalWindows(stabilising_data, config['resolving']['num_history'])
    steps_per_epoch = len(train_data) // GA
    total_steps = config['epochs'] * steps_per_epoch
    eval_loaders = {}
    for name, dataset in eval_data.items():
        if not len(dataset):
            raise RuntimeError(f'Eval set {name} is smaller than one global microbatch')
        eval_loaders[name] = make_loader(config, dataset, config['eval_num_dataloader_workers'], range(len(dataset)))

    model.load_diffusion_model()
    views = {'stabilising': model, 'resolving': model.resolving}
    layered = BimodalLayers({mode: view.to_layers() for mode, view in views.items()}, config['activation_checkpointing']).to('cuda')
    if 'trainable' in config:  # original_name prefixes; every other parameter is frozen
        for p in layered.parameters():
            p.requires_grad_(getattr(p, 'original_name', '').startswith(tuple(config['trainable'])))
    parameters_to_train = [p for p in layered.parameters() if p.requires_grad]
    if is_main_process():
        print(f'Training {sum(p.numel() for p in parameters_to_train):,} of {sum(p.numel() for p in layered.parameters()):,} parameters')
    parameters_to_clip = [p for p in parameters_to_train
                          if not getattr(p, 'original_name', '').endswith(('null_token', 'no_audio_tokens', 'drop_token'))]
    optimizer = build_optimizer(config, model, parameters_to_train)
    ddp_model = DDP(layered, device_ids=[local_rank], broadcast_buffers=False, gradient_as_bucket_view=True, bucket_cap_mb=DDP_BUCKET_MB)
    ema = build_ema(config, optimizer, layered)
    lr_scheduler = build_lr_scheduler(config, optimizer, steps_per_epoch)
    loss_fns = {mode: view.get_loss_fn() for mode, view in views.items()}
    frame_weights = {mode: view.num_present / sum(v.num_present for v in views.values()) for mode, view in views.items()}
    initial_layers = {mode: stack.layers[0] for mode, stack in layered.stacks.items()}
    audio_layers = {mode: [layer for layer in stack.layers if getattr(layer, 'audio_block', None) is not None]
                    for mode, stack in layered.stacks.items()}

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
        print(f'Global batch size = {global_batch_size} windows per mode, {steps_per_epoch} steps/epoch')
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

    def evaluate(prefix, eval_step, log_dict):
        """
        Evaluates the current weights, logged as {prefix}/{set}/{mode}/{suite}/slot-k.

        -> {(set, mode): {(suite, slot): loss}}
        """
        by_name = {}
        for name, loader in eval_loaders.items():
            for mode, view in views.items():
                results = by_name[name, mode] = evaluate_suites(view, layered.stacks[mode], loss_fns[mode], loader, swap_refs=False)
                for (suite, slot), value in results.items():
                    if is_main_process():
                        print(f'[{prefix}] step={eval_step} {name}/{mode}/{suite}/slot-{slot} loss={value:.4f}')
                    log_dict[f'{prefix}/{name}/{mode}/{suite}/slot-{slot}'] = value
        return by_name

    def run_eval(eval_step, eval_examples):
        if not eval_loaders:
            return
        empty_cuda_cache()
        log_dict = {}
        raw = evaluate('eval', eval_step, log_dict)
        if {'seen', 'unseen'} <= eval_loaders.keys():
            for mode in MODES:
                seen, unseen = raw['seen', mode], raw['unseen', mode]
                for (suite, slot), value in seen.items():
                    log_dict[f'diagnostics/unseen-minus-seen-percentage/{mode}/{suite}/slot-{slot}'] = 100 * (unseen[suite, slot] - value) / value
        if ema is not None:
            with ema.swapped_in(optimizer, config['save_dtype']):
                averaged = evaluate('ema', eval_step, log_dict)
            for (name, mode), results in averaged.items():
                for (suite, slot), value in results.items():
                    log_dict[f'diagnostics/{name}-ema-minus-raw/{mode}/{suite}/slot-{slot}'] = value - raw[name, mode][suite, slot]
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
        for stack in layered.stacks.values():
            stack.checkpoint_all = step == first_step
        # Per mode, then the universal loss: each window's mean over its 1 + P valid present frames across both modes.
        step_losses = torch.zeros(len(MODES) + 1, device='cuda')
        step_chunks, step_people = [], []
        step_tokens = 0
        last = step >= total_steps or ('max_steps' in config and step >= config['max_steps'])
        for micro in range(GA):
            batches = next(data_iter)
            step_chunks.append(batches[0]['chunk_index'].to('cuda', non_blocking=True))
            step_people.append(batches[0]['person_index'].to('cuda', non_blocking=True))
            sums = counts = 0
            for index, (mode, batch) in enumerate(zip(MODES, batches)):
                step_tokens += batch['latents_low'][:, 0].numel() // patch_volume
                features, labels = views[mode].prepare_inputs(batch)
                sync = step == first_step or (micro == GA - 1 and index == len(MODES) - 1)
                with (nullcontext() if sync else ddp_model.no_sync()):
                    output = ddp_model(features, mode)
                    loss = loss_fns[mode](output, labels)
                    (frame_weights[mode] * loss / GA).backward()
                step_losses[index] += loss.detach()
                window_sums, window_counts = present_loss_sums(output, labels, views[mode].num_clean)
                sums, counts = sums + window_sums, counts + window_counts
            step_losses[-1] += (sums / counts).mean()
        step_losses /= GA
        dist.all_reduce(step_losses, op=dist.ReduceOp.AVG)
        n_chunks, n_people = _step_uniques(step_chunks, step_people)

        clip = config['gradient_clipping']
        grad_norm = nn.utils.clip_grad_norm_(parameters_to_clip, clip if clip > 0 else float('inf'))
        optimizer.step()
        if ema is not None:
            ema.update()
        lr_scheduler.step()
        optimizer.zero_grad()

        *values, loss_value = step_losses.tolist()
        mode_losses = dict(zip(MODES, values))
        grad_norm_value = grad_norm.item()
        step_time = time.perf_counter() - step_start
        epoch_loss += loss_value
        epoch_steps += 1
        x_axis = examples if 'x_axis_examples' in config else step
        if is_main_process():
            print(
                f'step={step} loss={loss_value:.4f} ' + ' '.join(f'{mode}={value:.4f}' for mode, value in mode_losses.items())
                + f' lr={lr_scheduler.get_last_lr()[0]:.3e} grad_norm={grad_norm_value:.3f} chunks={n_chunks} people={n_people}'
            )

        bucket_losses = {mode: view.get_train_loss_buckets() for mode, view in views.items()}
        for view in views.values():
            view.reset_train_loss_buckets()
        if wandb_enable:
            last_lrs = lr_scheduler.get_last_lr()
            log_dict = {
                'train/loss': loss_value, 'train/lr': last_lrs[0], 'train/grad_norm': grad_norm_value,
                **{f'train/loss_{mode}': value for mode, value in mode_losses.items()},
                'data/window_hours': examples * (model.num_history + model.num_present) * VAE_TEMPORAL_STRIDE / FPS / 3600,
                'data/step_chunks': n_chunks, 'data/step_people': n_people, 'step': x_axis,
                'data/step_time': step_time, 'data/tokens_per_s': step_tokens / step_time,
            }
            groups = config['optimizer'].get('groups', [])
            for gid, lr_value in zip(optimizer.group_ids[1:], last_lrs[1:]):
                log_dict[f'train/lr_{groups[gid - 1]["prefix"].rstrip(".")}'] = lr_value
            for bucket_name, avg_loss in bucket_losses['stabilising'].items():
                log_dict[f'loss_buckets/{bucket_name}'] = avg_loss
            for bucket_name, avg_loss in bucket_losses['resolving'].items():
                log_dict[f'loss_buckets/resolving/{bucket_name}'] = avg_loss
            if step % DIAGNOSTICS_EVERY == 0:
                for i, block in enumerate(model.audio_adapter.blocks):
                    log_dict[f'audio/out_proj_{i}'] = block.out_proj.weight.detach().norm().item()
                for mode in MODES:
                    for i, layer in enumerate(audio_layers[mode]):
                        ratios = (layer.audio_injection_ratio / layer.audio_injection_count).tolist()
                        for slot, ratio in enumerate(ratios, start=1):
                            log_dict[f'audio/{mode}/inject_ratio_{i}/slot-{slot}'] = ratio
                seen = sum(layer.audio_drop_seen for layer in initial_layers.values())
                if seen:
                    log_dict['train/audio_dropped'] = sum(layer.audio_dropped.item() for layer in initial_layers.values()) / seen
            wandb.log(log_dict)
        if step % DIAGNOSTICS_EVERY == 0:
            for mode in MODES:
                for layer in audio_layers[mode]:
                    layer.audio_injection_ratio = 0
                    layer.audio_injection_count = 0
                initial_layers[mode].audio_dropped.zero_()
                initial_layers[mode].audio_drop_seen = 0

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
