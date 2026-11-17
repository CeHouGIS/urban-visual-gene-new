#!/usr/bin/env python3
"""Bounded continuous-slot edit diagnostic using one of our two existing GPUs."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def now():
    return datetime.now(timezone.utc).isoformat()


def status(output, phase, **values):
    from urbangene_v2.million import atomic_json
    atomic_json(output / 'status.json', {'phase': phase, 'updated_utc': now(), **values})


def identity(pid):
    stat = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return stat[19], Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ')


def select_slot(masks):
    """Fixed spatial ROI, no class or instance labels and no manual result selection."""
    import torch
    h, w = masks.shape[-2:]
    roi = torch.zeros((h, w), device=masks.device)
    roi[int(.35*h):int(.9*h), int(.2*w):int(.8*w)] = 1
    score = (masks * roi).sum((1, 2))
    score[0] = -1
    return int(score.argmax())


def worker(args):
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    from urbangene_v2.composition import SlotProgram, decode_program, edit_diagnostics
    from urbangene_v2.backbone import coordinates
    from urbangene_v2.discovery import DiscoveryModel
    from urbangene_v2.million import ShardedFeatures, atomic_json

    output = Path(args.output)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('Diagnostic inference requires exactly one explicitly selected GPU')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.20, 0)
    device = torch.device('cuda:0')
    checkpoint = torch.load(output / 'checkpoint_source.pt', map_location='cpu', weights_only=False)
    cfg, step = checkpoint['config'], checkpoint['step']
    model = DiscoveryModel(**cfg['model']).to(device).eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    data = Path(cfg['data_root'])
    cache = Path(args.cache)
    validation = ShardedFeatures(data / 'features.validation.index.json')
    # Prefer only fully published files, with no feature hashing or full-file readback.
    for row in validation.shards:
        local = cache / Path(row['path']).relative_to(data)
        if local.exists():
            row['path'] = str(local)
    records = []
    for manifest in sorted((data / 'rgb_shards/validation').glob('images_*.jsonl')):
        records.extend(json.loads(line) for line in manifest.read_text().splitlines())
    first_train = json.loads((data / 'features.train.index.json').read_text())['shards'][0]
    local_train = cache / Path(first_train['path']).relative_to(data)
    training_features = np.load(local_train if local_train.exists() else first_train['path'], mmap_mode='r')
    # One shared display projection, fitted on train-only patches. This is not RGB decoding.
    samples = np.concatenate([np.array(training_features[i, :, ::4, ::4], dtype=np.float32)
                              .reshape(1024, -1).T for i in range(16)])
    x = torch.from_numpy(samples).to(device)
    mean = x.mean(0)
    centered = x - mean
    _, basis = torch.linalg.eigh(centered.T @ centered / (len(x) - 1))
    basis = basis[:, -3:]
    projected = centered @ basis
    lo, hi = torch.quantile(projected, .01, dim=0), torch.quantile(projected, .99, dim=0)
    np.savez(output / 'display_pca_train_only.npz', mean=mean.cpu().numpy(),
             basis=basis.cpu().numpy(), lo=lo.cpu().numpy(), hi=hi.cpu().numpy())

    def pca_image(feature, size=240):
        f = feature[0].flatten(1).T
        rgb = (((f - mean) @ basis - lo) / (hi-lo).clamp_min(1e-8)).clamp(0, 1)
        return Image.fromarray((rgb.reshape(28, 28, 3).cpu().numpy()*255).astype('uint8')).resize(
            (size, size), Image.Resampling.NEAREST)

    def heat_image(values, maximum):
        norm = (values / max(maximum, 1e-12)).clamp(0, 1).cpu().numpy()
        rgb = np.stack([norm, norm*.2, norm*0], -1)
        return Image.fromarray((rgb*255).astype('uint8')).resize((240, 240), Image.Resampling.NEAREST)

    def source_image(index, soft):
        with Image.open(records[index]['image']) as opened:
            original = opened.convert('RGB').resize((240, 240), Image.Resampling.BICUBIC)
        selected = Image.fromarray((soft.cpu().numpy()*255).astype('uint8')).resize(
            (240, 240), Image.Resampling.BILINEAR)
        tint = Image.new('RGB', (240, 240), '#ff4d00')
        return original, Image.composite(Image.blend(original, tint, .65), original, selected)

    def weight_statistics(weights):
        grid = coordinates(28, 28, weights.device, weights.dtype)
        total = weights.sum()
        center = (weights[..., None]*grid).sum((0, 1))/total.clamp_min(1e-12)
        return {'mean_weight': float(weights.mean()),
                'center_xy': center.tolist() if total > 0 else None}

    # Fixed pre-existing validation previews; A/B pairs chosen before rendering.
    pairs = [(0, 48), (32, 16)]
    summary = {'checkpoint_step': step, 'image_generation_performed': False,
               'discrete_gene_program': False, 'target_C_used': False,
               'model_parameters_updated': False, 'inference_device': 'cuda:0',
               'gpu_uuid': os.environ['CUDA_VISIBLE_DEVICES'], 'pairs': [],
               'selection': 'foreground slot with largest soft mass in fixed ROI x=.2:.8, y=.35:.9',
               'display': 'train-only PCA of DINO features; colors are not generated RGB',
               'independent_ground_truth': False}
    with torch.inference_mode():
        for number, (a_idx, b_idx) in enumerate(pairs):
            folder = output / f'pair_{number+1:02d}'
            folder.mkdir()
            a_feature, b_feature = validation[a_idx][None].to(device), validation[b_idx][None].to(device)
            torch.manual_seed(10000+a_idx)
            a_out = model(a_feature)
            torch.manual_seed(10000+b_idx)
            b_out = model(b_feature)
            a = SlotProgram.from_encoding(a_out, 28, 28, records[a_idx]['id'])
            b = SlotProgram.from_encoding(b_out, 28, 28, records[b_idx]['id'])
            ai, bi = select_slot(a_out['soft_masks'][0]), select_slot(b_out['soft_masks'][0])
            baseline = decode_program(model.decoder, a)
            torch.testing.assert_close(baseline['reconstructed_features'], a_out['reconstructed_features'])
            a.save(folder / 'A.program.pt')
            b.save(folder / 'B.program.pt')
            mass = a_out['soft_masks'][0].mean((1, 2)).clone()
            mass[0], mass[ai] = float('inf'), float('inf')
            spare = int(mass.argmin())
            distance = float(a.geometry[ai, 0])
            translation = (-.5 if distance > 0 else .5, 0.)
            donor_translation = tuple((a.geometry[ai, :2] - b.geometry[bi, :2]).tolist())
            donor_scale = tuple((a.geometry[ai, 2:] / b.geometry[bi, 2:]).tolist())
            variants = {
                'identity': (a, [ai], [ai]),
                'delete': (a.delete([ai]), [ai], []),
                'move': (a.transform([ai], translation), [ai], [ai]),
                'scale': (a.transform([ai], scale=(1.5, 1.5)), [ai], [ai]),
                'duplicate': (a.delete([spare]).duplicate([ai], translation), [ai, spare], [ai, spare]),
                'replace_B': (a.replace_from([ai], b, [bi], donor_translation, donor_scale), [ai], [ai]),
            }
            # Isolate the effect of the duplicate from the explicitly removed spare slot.
            freed = decode_program(model.decoder, a.delete([spare]))
            outcomes, deltas, displays = {}, {}, []
            for name, (program, old_ids, new_ids) in variants.items():
                program.save(folder / f'{name}.program.pt')
                reloaded = SlotProgram.load(folder / f'{name}.program.pt', device)
                out = decode_program(model.decoder, reloaded)
                direct = decode_program(model.decoder, program)
                torch.testing.assert_close(out['reconstructed_features'], direct['reconstructed_features'],
                                           rtol=0, atol=0)
                metrics, delta, support = edit_diagnostics(baseline, out, old_ids, new_ids)
                if name == 'duplicate':
                    metrics['new_copy_effect_mse_vs_freed_slot_control'] = float(
                        (out['reconstructed_features']-freed['reconstructed_features']).square().mean())
                    metrics['explicitly_removed_spare_slot'] = spare
                weights = out['soft_masks'][0, new_ids].sum(0) if new_ids else out['soft_masks'][0, ai]
                metrics['original_selected_weight'] = weight_statistics(a_out['soft_masks'][0, ai])
                metrics['edited_selected_weight'] = weight_statistics(weights)
                if name == 'move':
                    old_center = metrics['original_selected_weight']['center_xy']
                    new_center = metrics['edited_selected_weight']['center_xy']
                    metrics['requested_translation_xy'] = translation
                    metrics['observed_weight_center_translation_xy'] = [
                        right-left for left,right in zip(old_center, new_center)]
                outcomes[name] = metrics
                deltas[name] = delta
                np.savez_compressed(folder / f'{name}.npz',
                    reconstructed_features=out['reconstructed_features'][0].cpu().numpy(),
                    soft_masks=out['soft_masks'][0].cpu().numpy(),
                    feature_change_mse_map=delta.cpu().numpy(), support=support.cpu().numpy())
                displays.append((name, pca_image(out['reconstructed_features']), weights))
            original_a, overlay_a = source_image(a_idx, a_out['soft_masks'][0, ai])
            original_b, overlay_b = source_image(b_idx, b_out['soft_masks'][0, bi])
            sources = Image.new('RGB', (4*248, 290), 'white')
            draw = ImageDraw.Draw(sources)
            for col, (title, tile) in enumerate([
                    ('A | original photograph', original_a), (f'A | selected slot {ai}', overlay_a),
                    ('B | original photograph', original_b), (f'B | selected slot {bi}', overlay_b)]):
                draw.text((col*248+4, 12), title, fill='black')
                sources.paste(tile, (col*248+4, 40))
            sources.save(folder / 'sources.png')
            canvas = Image.new('RGB', (6*248, 865), 'white')
            draw = ImageDraw.Draw(canvas)
            draw.text((5, 5), f'CONTINUOUS SLOT EDITS | checkpoint {step} | PCA FEATURES, NOT RGB GENERATION', fill='black')
            maximum = max(float(v.max()) for v in deltas.values())
            for col, (name, tile, weights) in enumerate(displays):
                x = col*248+4
                draw.text((x, 31), name, fill='black')
                canvas.paste(tile, (x, 54))
                draw.text((x, 310), 'Selected-slot soft weight', fill='black')
                weight = Image.fromarray((weights.clamp(0, 1).cpu().numpy()*255).astype('uint8')).convert('RGB')
                canvas.paste(weight.resize((240, 240), Image.Resampling.NEAREST), (x, 332))
                draw.text((x, 588), 'Feature change | shared scale', fill='black')
                canvas.paste(heat_image(deltas[name], maximum), (x, 610))
            canvas.save(folder / 'composition.png')
            pair = {'A': records[a_idx]['id'], 'B': records[b_idx]['id'],
                    'A_validation_index': a_idx, 'B_validation_index': b_idx,
                    'A_selected_slot': ai, 'B_selected_slot': bi,
                    'translation_xy_normalized': translation,
                    'duplicate_spare_slot': spare,
                    'A_reconstruction_mse': float((baseline['reconstructed_features']-a_feature).square().mean()),
                    'outcomes': outcomes}
            atomic_json(folder / 'metrics.json', pair)
            summary['pairs'].append(pair)
            print(json.dumps({'pair': number+1, 'selected_A': ai, 'selected_B': bi,
                              'mse': {k: v['feature_change_mse'] for k,v in outcomes.items()}}), flush=True)
    atomic_json(output / 'metrics.json', summary)
    status(output, 'complete', checkpoint_step=step, gpu_uuid=os.environ['CUDA_VISIBLE_DEVICES'],
           image_generation_performed=False, pairs=len(pairs))


def supervise(args):
    from urbangene_v2.common import writable
    output = writable(args.output)
    run = Path(args.run)
    execution = json.loads((run / 'execution.json').read_text())
    gpus = execution['physical_gpus']
    if len(gpus) != 2:
        raise ValueError('Expected the original two-GPU training allocation')
    pid = args.training_worker
    before = identity(pid)
    if b'run_real1m.py' not in before[1] or b'--worker train' not in before[1]:
        raise ValueError('Will only coordinate with the registered project training worker')
    if b'--config' not in before[1] or str(run.name).split('m1_')[1].encode() not in before[1]:
        raise ValueError('Training process belongs to another experiment')
    cache_state = json.loads((run / 'local_cache_migration_20261112/status.json').read_text())
    if cache_state['status'] != 'copying' or cache_state['copied_bytes']/cache_state['total_bytes'] > .9:
        raise RuntimeError('Local-cache handoff may be imminent; queue diagnostic instead')
    uuid = gpus[0]['uuid']
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(run / 'checkpoint_best_recon.pt', output / 'checkpoint_source.pt')
    protocol = {'created_utc': now(), 'experiment': str(run), 'training_worker': pid,
                'training_identity': before[0], 'gpu_uuid': uuid, 'max_total_gpus': 2,
                'pairs': [[0,48], [32,16]], 'selection_fixed_before_results': True,
                'no_new_training': True, 'target_C_used': False,
                'continuous_slots_not_discrete_genes': True,
                'rgb_generation': False, 'max_diagnostic_seconds': 240}
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    lock = (run / 'slot_edit_diagnostic.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    paused, watchdog = False, None
    try:
        if identity(pid) != before:
            raise RuntimeError('Training process identity changed before pause')
        os.kill(pid, signal.SIGSTOP)
        paused = True
        # Independent backstop: resume only this exact process if the supervisor dies.
        watchdog_code = '''import pathlib,os,signal,sys,time
time.sleep(250)
try:
 p=int(sys.argv[1]); current=pathlib.Path(f'/proc/{p}/stat').read_text().rsplit(')',1)[1].split()[19]
 if current==sys.argv[2]: os.kill(p,signal.SIGCONT)
except (FileNotFoundError,ProcessLookupError): pass
'''
        watchdog = subprocess.Popen([sys.executable, '-c', watchdog_code, str(pid), before[0]],
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL, start_new_session=True)
        status(output, 'diagnostic_running_training_temporarily_paused', training_worker=pid, gpu_uuid=uuid)
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': uuid, 'PYTHONDONTWRITEBYTECODE': '1',
               'OPENBLAS_NUM_THREADS': '4', 'OMP_NUM_THREADS': '4', 'TMPDIR': args.cache}
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--worker',
                   '--output', str(output), '--cache', args.cache]
        with (output / 'diagnostic.log').open('w') as log:
            subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                           timeout=240, check=True)
    except BaseException as error:
        status(output, 'failed', error=str(error), error_type=type(error).__name__)
        raise
    finally:
        if paused and identity(pid) == before:
            os.kill(pid, signal.SIGCONT)
            (output / 'TRAINING_RESUMED.json').write_text(json.dumps(
                {'training_worker': pid, 'resumed_utc': now(), 'checkpoint_rollback': False})+'\n')
        if watchdog is not None:
            watchdog.terminate()
            watchdog.wait(timeout=5)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cache', required=True)
    parser.add_argument('--run')
    parser.add_argument('--training-worker', type=int)
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    worker(args) if args.worker else supervise(args)
