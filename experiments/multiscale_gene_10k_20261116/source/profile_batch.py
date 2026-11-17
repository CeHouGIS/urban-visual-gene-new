"""Measure true two-rank RGB+LPIPS updates at increasing CUDA microbatches."""
import argparse
from copy import deepcopy
from datetime import timedelta
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from common import load_config, now, write_json
from data import rank_ids
from model import create_model
from train import distributed_update, gradient_equivalence, perceptual_model, synchronized_parameters


def profile(config, variant):
    rank, world = dist.get_rank(), dist.get_world_size()
    error = gradient_equivalence(config, variant, local_count=7, accumulated_micro=3)
    model = create_model(config, variant)
    root = Path(config['root'])
    saved = torch.load(root / f'runs/{variant}/checkpoint_resume.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(saved['model'])
    ddp = DDP(model, device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
    perceptual = perceptual_model()
    data = np.load(Path(config['cache']) / 'train.rgb.npy', mmap_mode='r')
    ids = rank_ids(0, len(data), config['training']['effective_batch_size'], config['seed'], rank, world)
    prepared = np.array(data[ids], copy=True)
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    results = []
    for micro in [64, 128, 192, 256]:
        # Extrapolate conservatively from the measured previous point. Avoid
        # attempting a batch predicted to exceed 92% of physical VRAM.
        if results and results[-1].get('status') == 'passed':
            prior = results[-1]
            projected = prior['peak_allocated_bytes_max_rank'] * micro / prior['micro_batch_size']
            if projected > total * 0.92:
                results.append({'micro_batch_size': micro, 'status': 'skipped_memory_prediction',
                                'predicted_allocated_bytes': projected})
                break
        current = deepcopy(config)
        current['training']['micro_batch_size'] = micro
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        distributed_update(ddp, optimizer, perceptual, prepared, current)
        times = []
        for _ in range(3):
            dist.barrier()
            torch.cuda.synchronize()
            started = time.perf_counter()
            values, usage, grad = distributed_update(ddp, optimizer, perceptual, prepared, current)
            torch.cuda.synchronize()
            elapsed = torch.tensor(time.perf_counter() - started, device='cuda')
            dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
            times.append(float(elapsed))
        peaks = torch.tensor([torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()],
                             device='cuda', dtype=torch.int64)
        dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
        delta = synchronized_parameters(model)
        result = {'status': 'passed', 'micro_batch_size': micro,
                  'global_batch_size': config['training']['effective_batch_size'],
                  'accumulation_steps': math.ceil(len(ids) / micro),
                  'last_microbatch_size_per_rank': (len(ids) - 1) % micro + 1,
                  'seconds_per_global_update': times,
                  'median_seconds_per_global_update': float(np.median(times)),
                  'images_per_second': config['training']['effective_batch_size'] / float(np.median(times)),
                  'peak_allocated_bytes_max_rank': int(peaks[0]),
                  'peak_reserved_bytes_max_rank': int(peaks[1]),
                  'physical_vram_bytes_per_gpu': total,
                  'rank_parameter_max_difference': delta, 'last_loss': float(values[0])}
        results.append(result)
        if rank == 0:
            print(json.dumps(result), flush=True)
            write_json(root / f'reports/batch_tuning/{variant}.profile.json',
                {'status': 'profiling', 'variant': variant, 'results': results, 'updated_utc': now()})
    if rank == 0:
        write_json(root / f'reports/batch_tuning/{variant}.profile.json',
            {'status': 'complete', 'variant': variant, 'results': results,
             'uneven_accumulation_gradient_equivalence_max_abs_error': error,
             'weights_discarded': True, 'production_checkpoint_step': saved['step'],
             'updated_utc': now()})
    dist.barrier()


if __name__ == '__main__':
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--variant', required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError('Batch profiling requires a leased pair of CUDA GPUs')
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(4)
    dist.init_process_group('nccl', timeout=timedelta(minutes=10))
    profile(config, args.variant)
    dist.destroy_process_group()
