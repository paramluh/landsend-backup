import argparse
import gc
import os
import shutil
from contextlib import nullcontext
from pathlib import Path

import toml
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader

from dataloading.ds_dense import DenseWindows, EvalClips, SHUFFLE_SEED
from dataloading.sources import MIN_LATENT_FRAMES, PretrainClipSource
from models.pose.settings import FPS, VAE_TEMPORAL_STRIDE
from models.pose.wan_universal import UniversalWidePipeline
from utils.harness import (
    build_ema, build_lr_scheduler, build_optimizer, check_manual_signals, get_most_recent_run_dir,
    get_next_run_dir, load_optim_state, save_model_dir, save_optim_state, wandb_resume_kwargs,
)
from utils.common import DTYPE_MAP, empty_cuda_cache, is_main_process
from utils.evaluation import evaluate_suites


PIPELINES = {
    'universal_wide': UniversalWidePipeline,
}
REQUIRED_KEYS = (
    'save_every_n_steps', 'epochs', 'lr', 'min_lr', 'warmup_steps', 'gradient_clipping',
    'micro_batch_size_per_gpu', 'gradient_accumulation_steps', 'num_dataloader_workers',
    'num_history', 'num_present', 'history_dropout', 'output_dir', 'root_dir', 'clipnames_root',
    'buckets', 'activation_checkpointing', 'compile_blocks', 'activation_checkpoint_skip',
)
SAVE_DTYPE = 'bfloat16'
DDP_BUCKET_MB = 500  # 25 MB buckets leave ~230 ms of all-reduce exposed after the last backward at 8 ranks; 500 MB, ~160 ms
DIAGNOSTICS_EVERY = 100  # steps between audio diagnostics (injection ratios and dropout rate are averaged over the window)


def load_config(path):
    config = toml.load(path)
    assert config['model']['type'] in PIPELINES, config['model']['type']
    for key in REQUIRED_KEYS:
        assert key in config, key
    if 'eval_buckets' in config:
        for key in ('eval_every_n_steps', 'eval_num_dataloader_workers', 'eval_micro_batch_size_per_gpu'):
            assert key in config, key
    config['save_dtype'] = DTYPE_MAP[config.get('save_dtype', SAVE_DTYPE)]
    config['model']['dtype'] = DTYPE_MAP[config['model']['dtype']]
    config['model']['transformer_dtype'] = DTYPE_MAP[config['model']['transformer_dtype']]
    if 'resume_from' in config:
        config['model']['checkpoint_path'] = os.path.join(config['resume_from'], 'model.safetensors')
    return config


def build_pretrain_source(config, clipnames_root, buckets):
    # pose = 'v2' gates rows on the manifest's pose_v2 column and loads *_pose_latents_v2.safetensors.
    pose_version = config.get('pose')
    return PretrainClipSource(
        config['root_dir'], buckets, clipnames_root,
        pose_column='pose_v2' if pose_version == 'v2' else 'has_pose',
        pose_version=pose_version, shuffle_seed=SHUFFLE_SEED,
    )


def build_data(config, rank, world_size):
    """(train dataset, eval dataset or None, windows per global step)."""
    B, GA = config['micro_batch_size_per_gpu'], config['gradient_accumulation_steps']
    H, P, dropout = config['num_history'], config['num_present'], config['history_dropout']
    train_data = DenseWindows(
        build_pretrain_source(config, config['clipnames_root'], config['buckets']), H, P, dropout, rank, world_size, B, GA)
    eval_data = None
    if 'eval_buckets' in config:
        eval_data = EvalClips(
            build_pretrain_source(config, config.get('eval_clipnames_root', config['clipnames_root']), config['eval_buckets']),
            rank, world_size, config['eval_micro_batch_size_per_gpu'])
    return train_data, eval_data, B * world_size * GA


def make_loader(config, dataset, num_workers, items):
    kwargs = {'prefetch_factor': config['prefetch_factor']} if 'prefetch_factor' in config and num_workers else {}
    return DataLoader(
        dataset, batch_size=None, sampler=items, num_workers=num_workers, pin_memory=True,
        persistent_workers=num_workers > 0, **kwargs,
    )


def _call_layer(layer, *tensors):
    return layer(tensors)


class LayeredModel(nn.Module):
    """
    Sequential to_layers() run. With checkpoint_blocks, TransformerLayer blocks past
    activation_checkpoint_skip recompute in backward. Reentrant: the blocks are
    torch.compile'd and the non-reentrant variant miscounts their saved tensors.
    checkpoint_all recomputes every block: during the first backward of a run DDP still
    holds autograd's gradient tensors next to its buckets (+1x the gradients), so that
    step keeps its stored activations minimal.
    """

    def __init__(self, layers, checkpoint_blocks):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.checkpoint_blocks = checkpoint_blocks
        self.checkpoint_all = False

    def forward(self, inputs):
        for layer in self.layers:
            if self.checkpoint_blocks and torch.is_grad_enabled() and hasattr(layer, 'checkpoint') and (self.checkpoint_all or layer.checkpoint):
                inputs = checkpoint(_call_layer, layer, *inputs, use_reentrant=True)
            else:
                inputs = layer(inputs)
        return inputs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to TOML configuration file.')
    args = parser.parse_args()
    config = load_config(args.config)

    torch.multiprocessing.set_sharing_strategy('file_system')
    dist.init_process_group('nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    # Torch defaults CPU intra-op threads to the physical core count per process; with
    # LOCAL_WORLD_SIZE ranks on one node that oversubscribes the cores.
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // int(os.environ.get('LOCAL_WORLD_SIZE', world_size))))
    # All blocks share one compiled code object. Dynamo guards grad mode and num_threads, and the
    # checkpointed first step recomputes on the 1-thread autograd thread, so step 1, the steady state
    # and eval each add variants; past the default limit of 8 every block silently runs eager.
    torch._dynamo.config.recompile_limit = 64

    GA = config['gradient_accumulation_steps']
    assert config['num_present'] < MIN_LATENT_FRAMES, \
        f'num_present={config["num_present"]} must be < MIN_LATENT_FRAMES={MIN_LATENT_FRAMES}'

    model = PIPELINES[config['model']['type']](config)
    train_data, eval_data, global_batch_size = build_data(config, rank, world_size)
    steps_per_epoch = len(train_data) // GA
    total_steps = config['epochs'] * steps_per_epoch
    eval_loader = None
    if eval_data is not None:
        if not len(eval_data):
            raise RuntimeError('Eval set is smaller than one global microbatch')
        eval_loader = make_loader(config, eval_data, config['eval_num_dataloader_workers'], range(len(eval_data)))

    model.load_diffusion_model()
    layered = LayeredModel(model.to_layers(), config['activation_checkpointing']).to('cuda')
    parameters_to_train = [p for p in layered.parameters() if p.requires_grad]
    optimizer = build_optimizer(config, model, parameters_to_train)  # re-homes parameter storage; before DDP
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
        meta = load_optim_state(Path(config['resume_from']), optimizer, ema, steps_per_epoch)
        for _ in range(meta['step']):
            lr_scheduler.step()
        step = meta['step'] + 1
        examples = meta['examples'] + global_batch_size
        wandb_resume = wandb_resume_kwargs(config, meta)
        if is_main_process():
            print(f'Resumed from {config["resume_from"]} at step {step}')
    
    # Item order is a fixed seeded permutation and each step consumes GA items per rank,
    # so the stream position is a function of the step counter alone.
    train_loader = make_loader(config, train_data, config['num_dataloader_workers'],
                               range((step - 1) * GA, total_steps * GA))
    data_iter = iter(train_loader)

    if is_main_process():
        run_dir = get_next_run_dir(config['output_dir'])
        os.makedirs(run_dir)
        shutil.copy(args.config, run_dir)
        print(f'Global batch size = {global_batch_size} windows, {steps_per_epoch} steps/epoch')
    dist.barrier()
    run_dir = Path(get_most_recent_run_dir(config['output_dir']))

    wandb_enable = 'wandb' in config and is_main_process()
    if wandb_enable:
        import wandb
        wandb.init(project=config['wandb']['project'], name=config['wandb'].get('run_name'), config=config, dir=run_dir, **wandb_resume)
        wandb.define_metric('step')
        wandb.define_metric('*', step_metric='step')

    def run_eval(eval_step, eval_examples):
        if eval_loader is None:
            return
        empty_cuda_cache()
        results = evaluate_suites(model, layered, loss_fn, eval_loader)
        if is_main_process():
            for (suite, slot), value in results.items():
                print(f'[eval] step={eval_step} {suite}/slot-{slot} loss={value:.4f}')
        if wandb_enable:
            log_dict = {f'eval/{suite}/slot-{slot}': value for (suite, slot), value in results.items()}
            log_dict['step'] = eval_examples if 'x_axis_examples' in config else eval_step
            wandb.log(log_dict)
        empty_cuda_cache()

    if config.get('eval_step_zero') and 'resume_from' not in config:
        run_eval(0, 0)

    epoch_loss = 0.0
    epoch_steps = 0
    saved_step = None
    empty_cuda_cache()
    # Move the static model graph out of gen2 GC scans; collector pauses mid-step straggle the DP group.
    gc.freeze()

    # DDP points .grad at its bucket views during the first synchronised backward; syncing every microbatch of the
    # first step installs them before any no_sync() backward could allocate standalone grads next to the buckets.
    first_step = step
    while True:
        layered.checkpoint_all = step == first_step
        step_loss = torch.zeros((), device='cuda')
        for micro in range(GA):
            features, labels = model.prepare_inputs(next(data_iter))
            with (nullcontext() if micro == GA - 1 or step == first_step else ddp_model.no_sync()):
                loss = loss_fn(ddp_model(features), labels)
                (loss / GA).backward()
            step_loss += loss.detach()
        step_loss /= GA
        dist.all_reduce(step_loss, op=dist.ReduceOp.AVG)

        clip = config['gradient_clipping']
        grad_norm = nn.utils.clip_grad_norm_(parameters_to_train, clip if clip > 0 else float('inf'))
        optimizer.step()
        if ema is not None:
            ema.update()
        lr_scheduler.step()
        optimizer.zero_grad()

        loss_value = step_loss.item()
        grad_norm_value = grad_norm.item()
        epoch_loss += loss_value
        epoch_steps += 1
        x_axis = examples if 'x_axis_examples' in config else step
        if is_main_process():
            print(f'step={step} loss={loss_value:.4f} lr={lr_scheduler.get_last_lr()[0]:.3e} grad_norm={grad_norm_value:.3f}')

        bucket_losses = model.get_train_loss_buckets()
        model.reset_train_loss_buckets()
        if wandb_enable:
            last_lrs = lr_scheduler.get_last_lr()
            log_dict = {
                'train/loss': loss_value, 'train/lr': last_lrs[0], 'train/grad_norm': grad_norm_value,
                'data/window_hours': examples * (model.num_history + model.num_present) * VAE_TEMPORAL_STRIDE / FPS / 3600, 'step': x_axis,
            }
            groups = config['optimizer'].get('groups', [])
            for gid, lr_value in zip(optimizer.group_ids[1:], last_lrs[1:]):  # rank 0 may not hold every group
                log_dict[f'train/lr_{groups[gid - 1]["prefix"].rstrip(".")}'] = lr_value
            for bucket_name, avg_loss in bucket_losses.items():
                log_dict[f'loss_buckets/{bucket_name}'] = avg_loss
            if step % DIAGNOSTICS_EVERY == 0:
                for i, block in enumerate(model.audio_adapter.blocks):
                    log_dict[f'audio/out_proj_{i}'] = block.out_proj.weight.detach().norm().item()
                for i, layer in enumerate(audio_layers):
                    ratios = (layer.audio_injection_ratio / layer.audio_injection_count).tolist()
                    for slot, ratio in enumerate(ratios, start=1):
                        log_dict[f'audio/inject_ratio_{i}/slot-{slot}'] = ratio
                if initial_layer.audio_dropout_p:
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

        # touch <run_dir>/save: model + ema; save_o: also optimizer shards (resumable); save_q: save_o then exit.
        manual_save, manual_save_optim, manual_quit = check_manual_signals(run_dir)
        save_optim = manual_save_optim or manual_quit or (
            'save_optimizer_every_n_steps' in config and step % config['save_optimizer_every_n_steps'] == 0)
        if save_optim or manual_save or step % config['save_every_n_steps'] == 0:
            save_dir = run_dir / f'step{step}'
            save_model_dir(save_dir, model, layered, ema, config, args.config)
            if save_optim:
                save_optim_state(save_dir, optimizer, ema, step, examples, steps_per_epoch, wandb.run if wandb_enable else None)
            saved_step = step
        if manual_quit:
            break

        if eval_loader is not None and step % config['eval_every_n_steps'] == 0:
            run_eval(step, examples)
        if step >= total_steps or ('max_steps' in config and step >= config['max_steps']):
            break
        step += 1
        examples += global_batch_size

    if saved_step != step:
        save_model_dir(run_dir / f'step{step}', model, layered, ema, config, args.config)
    if is_main_process():
        print('TRAINING COMPLETE!')
    dist.destroy_process_group()
