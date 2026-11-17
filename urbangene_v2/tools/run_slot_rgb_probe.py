#!/usr/bin/env python3
"""Bounded GPU-only, continuous-slot RGB diagnostic, alongside our two-GPU run."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def now():
    return datetime.now(timezone.utc).isoformat()


def main(args):
    import numpy as np
    from PIL import Image, ImageDraw
    import torch
    from torch.nn import functional as F
    from diffusers import AutoencoderKL
    from urbangene_v2.common import writable
    from urbangene_v2.composition import SlotProgram, decode_program
    from urbangene_v2.discovery import DiscoveryModel
    from urbangene_v2.million import ShardedFeatures, atomic_json
    from urbangene_v2.rgb_probe import SlotRGBBridge, freeze_module, decode_vae_rgb

    start = time.monotonic()
    output = writable(args.output)
    run, source = Path(args.run), Path(args.composition)
    execution = json.loads((run / 'execution.json').read_text())
    allocated = {g['uuid'] for g in execution['physical_gpus']}
    selected = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if len(allocated) != 2 or selected not in allocated:
        raise ValueError('Probe must share exactly one GPU from the existing two-GPU allocation')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('One explicitly selected GPU required; no CPU model fallback')
    output.mkdir(parents=True, exist_ok=args.render_only)
    (output / 'source').mkdir(exist_ok=args.render_only)
    for path in [Path(__file__), ROOT/'src/urbangene_v2/rgb_probe.py', ROOT/'tests/test_rgb_probe.py']:
        if not args.render_only:
            shutil.copyfile(path, output/'source'/path.name)
    cache = Path(args.cache) / output.name
    cache.mkdir(parents=True, exist_ok=args.render_only)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.cuda.set_per_process_memory_fraction(.28)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = 'cuda:0'

    def status(phase, **values):
        record = {'phase': phase, 'updated_utc': now(), 'elapsed_seconds': round(time.monotonic()-start, 1),
                  'gpu_uuid': selected, 'cpu_fallback': False, **values}
        atomic_json(output/'status.json', record)
        print(json.dumps(record), flush=True)
        if time.monotonic()-start > args.max_seconds:
            raise TimeoutError('Bounded diagnostic time budget exceeded')

    checkpoint = torch.load(source/'checkpoint_source.pt', map_location='cpu', weights_only=False)
    discovery = freeze_module(DiscoveryModel(**checkpoint['config']['model'])).to(device)
    discovery.load_state_dict(checkpoint['model'], strict=True)
    projection = discovery.decoder.network[-1]
    bridge = SlotRGBBridge(projection.weight[:-1], projection.bias[:-1]).to(device)
    vae = freeze_module(AutoencoderKL.from_pretrained(
        args.vae, local_files_only=True, use_safetensors=True, low_cpu_mem_usage=False)).to(device)
    data_root = Path(checkpoint['config']['data_root'])
    protocol = {**vars(args), 'created_utc': now(), 'checkpoint_step': checkpoint['step'],
                'shared_gpu_uuid': selected, 'existing_allocation': sorted(allocated),
                'max_physical_gpus_total': 2, 'main_training_paused': False,
                'diagnostic_only': True, 'continuous_slots_not_discrete_genes': True,
                'trainable': 'shared CNN from reconstructed slot features to raw VAE latent',
                'frozen': ['DINOv3 feature cache', 'step9000 slot encoder and decoder', 'pretrained VAE'],
                'inference_inputs': 'serialized slots and geometry plus shared model weights only',
                'rgb_and_vae_latent_use': 'training targets and separate evaluation controls only',
                'target_C_used': False, 'selection': 'first train shard; first 64 validation rows',
                'latent_convention': 'unscaled posterior mode; direct VAE decode, no scheduler',
                'resize': 'torch bicubic antialias to448, clamp, round to uint8',
                'not_a_validated_gene_composition_model': True}
    if args.render_only:
        protocol = json.loads((output/'protocol.json').read_text())
    else:
        atomic_json(output/'protocol.json', protocol)
    status('preparing_training_targets', train_count=args.train_count, validation_count=64)

    def rgb(row):
        with Image.open(row['image']) as image:
            x = torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2,0,1).float()/255
        x = F.interpolate(x[None], (448,448), mode='bicubic', align_corners=False, antialias=True)[0]
        return (x.clamp(0,1)*255).round().byte()

    prepared = {}
    datasets = {}
    if args.render_only:
        for split in ['train','validation']:
            prepared[split] = {key: np.load(cache/f'{split}.{filename}.npy',mmap_mode='r')
                               for key,filename in [('projected','projected'),('latent','latent'),('rgb','rgb')]}
            datasets[split] = json.loads((output/f'{split}_selection.json').read_text())
    else:
        for split, count in [('train', args.train_count), ('validation', 64)]:
            ds = ShardedFeatures(data_root/f'features.{split}.index.json')
            records = []
            for row in ds.shards:
                records.extend(json.loads(line) for line in Path(row['manifest']).read_text().splitlines())
                local = Path(args.cache) / Path(row['path']).relative_to(data_root)
                if local.exists():
                    row['path'] = str(local)
                if len(records) >= count:
                    break
            records = records[:count]
            if len(records) != count or any(r['split'] != split for r in records):
                raise ValueError('Unpaired feature/image metadata')
            # Tiny metadata check only: do not hash/read back the feature payloads.
            for row in ds.shards[:1]:
                meta = Path(row['manifest'])
                original = Path(json.loads((data_root/f'features.{split}.index.json').read_text())['shards'][0]['path'])
                sidecar = original.with_suffix('.npy.json')
                if sidecar.exists():
                    ids = json.loads(sidecar.read_text()).get('ids')
                    if ids and ids[:min(count,len(ids))] != [r['id'] for r in records[:len(ids)]]:
                        raise ValueError('RGB/feature row identity mismatch')
            atomic_json(output/f'{split}_selection.json', records)
            n_dim = bridge.basis.shape[1]
            projected = np.lib.format.open_memmap(cache/f'{split}.projected.npy', 'w+', np.float16,
                                                  shape=(count,n_dim,28,28))
            latents = np.lib.format.open_memmap(cache/f'{split}.latent.npy', 'w+', np.float16,
                                               shape=(count,4,56,56))
            pixels = np.lib.format.open_memmap(cache/f'{split}.rgb.npy', 'w+', np.uint8,
                                              shape=(count,3,448,448))
            vae_mse = []
            with ThreadPoolExecutor(max_workers=8) as pool, torch.no_grad():
                for begin in range(0, count, 4):
                    end = min(begin+4,count)
                    images = torch.stack(list(pool.map(rgb, records[begin:end])))
                    pixels[begin:end] = images.numpy()
                    images = images.to(device).float()/255
                    latent = vae.encode(images*2-1).latent_dist.mode()
                    latents[begin:end] = latent.cpu().numpy().astype(np.float16)
                    features = torch.stack([ds[i] for i in range(begin,end)]).to(device)
                    if split == 'validation':
                        # Exactly the same deterministic per-image seeds as saved edit programs.
                        frozen = []
                        for offset in range(len(features)):
                            torch.manual_seed(10000+begin+offset)
                            frozen.append(discovery(features[offset:offset+1])['reconstructed_features'])
                        reconstructed = torch.cat(frozen)
                        recon = decode_vae_rgb(vae, latent)
                        vae_mse.extend((recon-images).square().mean((1,2,3)).tolist())
                    else:
                        torch.manual_seed(args.seed+begin)
                        reconstructed = discovery(features)['reconstructed_features']
                    projected[begin:end] = bridge.project(reconstructed).cpu().numpy().astype(np.float16)
                    if end % 128 == 0 or end == count:
                        projected.flush(); latents.flush(); pixels.flush()
                        status('preparing_'+split, prepared=end, count=count)
            prepared[split] = {'projected': projected, 'latent': latents, 'rgb': pixels,
                               'vae_mse': vae_mse}
            datasets[split] = records
            del ds
    del checkpoint
    torch.cuda.empty_cache()

    # Cache is small enough to preload the projected training inputs and targets on this GPU.
    train_x = torch.from_numpy(np.array(prepared['train']['projected'])).to(device)
    train_z = torch.from_numpy(np.array(prepared['train']['latent'])).to(device)
    val_x = torch.from_numpy(np.array(prepared['validation']['projected'])).to(device)
    val_z = torch.from_numpy(np.array(prepared['validation']['latent'])).to(device)
    if args.render_only:
        saved = torch.load(output/'bridge_final.pt',map_location='cpu',weights_only=False)
        bridge.load_state_dict(saved['model'],strict=True)
        bridge.eval()
    else:
        with torch.no_grad():
            bridge.input_mean.copy_(train_x.float().mean((0,2,3),keepdim=True))
            bridge.input_std.copy_(train_x.float().std((0,2,3),keepdim=True).clamp_min(.01))
            bridge.latent_mean.copy_(train_z.float().mean((0,2,3),keepdim=True))
        optimizer = torch.optim.AdamW(bridge.parameters(), lr=3e-4, weight_decay=.01)
        torch.manual_seed(args.seed+555)
        history = output/'history.jsonl'
        best_loss, best_state = float('inf'), None

        def latent_loss(prediction, target):
            target = target.float()
            return F.mse_loss(prediction,target) + .25*F.mse_loss(
                F.avg_pool2d(prediction,4), F.avg_pool2d(target,4))

        @torch.no_grad()
        def val_loss():
            return sum(float(latent_loss(bridge.latent_from_projected(x.float()),z))*len(x)
                       for x,z in zip(val_x.split(8),val_z.split(8)))/len(val_x)

        status('training_latent_bridge', updates=args.updates)
        for step in range(1,args.updates+1):
            bridge.train()
            ids = torch.randint(len(train_x),(args.batch_size,),device=device)
            prediction = bridge.latent_from_projected(train_x[ids].float())
            loss = latent_loss(prediction, train_z[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bridge.parameters(),1.)
            optimizer.step()
            if step % 100 == 0 or step == 1 or step == args.updates:
                bridge.eval()
                validation = val_loss()
                record = {'step':step,'stage':'latent','train_loss':float(loss.detach()),
                          'validation_latent_loss':validation,'utc':now()}
                with history.open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
                if validation < best_loss:
                    best_loss = validation
                    best_state = {k:v.detach().cpu().clone() for k,v in bridge.state_dict().items()}
                    torch.save({'model':best_state, 'step':step, 'width':96, 'protocol':protocol},
                               output/'bridge_best_latent.pt')
                status('training_latent_bridge', **record)
        bridge.load_state_dict(best_state)

        # A short RGB-target refinement lets gradients pass through the frozen VAE.
        # No original RGB is an input to the adapter, including during refinement.
        optimizer = torch.optim.AdamW(bridge.parameters(), lr=5e-5, weight_decay=.01)
        status('refining_rgb_bridge', updates=args.rgb_updates)
        for step in range(1,args.rgb_updates+1):
            ids = torch.randint(len(train_x),(2,),device=device)
            x, z = train_x[ids].float(), train_z[ids].float()
            target = torch.from_numpy(np.array(prepared['train']['rgb'][ids.cpu().numpy()])).to(device).float()/255
            predicted_latent = bridge.latent_from_projected(x)
            decoded = (vae.decode(predicted_latent).sample.float()+1)/2
            rgb_loss = F.l1_loss(decoded,target) + F.l1_loss(F.avg_pool2d(decoded,8),F.avg_pool2d(target,8))
            loss = rgb_loss + .1*latent_loss(predicted_latent,z)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bridge.parameters(),1.)
            optimizer.step()
            if step % 25 == 0 or step == args.rgb_updates:
                record = {'step':step,'stage':'rgb','train_loss':float(loss.detach()),
                          'rgb_loss':float(rgb_loss.detach()),'utc':now()}
                with history.open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
                status('refining_rgb_bridge',**record)
        assert all(p.grad is None for p in vae.parameters())
        assert all(p.grad is None for p in discovery.parameters())
        bridge.eval()
        torch.save({'model':bridge.state_dict(),'width':96,'protocol':protocol}, output/'bridge_final.pt')

    def pil(tensor):
        return Image.fromarray((tensor.detach().clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype('uint8'))

    def contact_sheet(rows, titles, path, tile=224, label_rows=None):
        canvas = Image.new('RGB',(len(titles)*(tile+8)+8,len(rows)*(tile+30)+40),'white')
        draw = ImageDraw.Draw(canvas)
        for col,title in enumerate(titles):
            draw.text((8+col*(tile+8),10),title,fill='black')
        for row,tiles in enumerate(rows):
            top = 40+row*(tile+30)
            for col,img in enumerate(tiles):
                canvas.paste(img.resize((tile,tile),Image.Resampling.LANCZOS),(8+col*(tile+8),top))
            if label_rows:
                draw.text((8,top+tile+3),label_rows[row],fill='black')
        canvas.save(path)

    # Evaluation is canonical program replay for the four already published codes.
    # Feature preparation used TF32 and half caches; old programs were encoded with
    # different arithmetic. Do not silently compare them as identical encodings.
    identity_checks = []
    with torch.no_grad():
        for index,pair,source_name in [(0,1,'A'),(48,1,'B'),(32,2,'A'),(16,2,'B')]:
            program = SlotProgram.load(source/f'pair_{pair:02d}'/f'{source_name}.program.pt',device)
            feature = decode_program(discovery.decoder,program)['reconstructed_features']
            canonical = bridge.project(feature).half()
            identity_checks.append({'validation_index':index,
                                    'max_projected_difference_from_reencoding':float(
                                        (canonical.float()-val_x[index:index+1].float()).abs().max()),
                                    'canonical_input':'previously serialized continuous slot program'})
            val_x[index:index+1].copy_(canonical)
    atomic_json(output/'program_replay_identity.json',identity_checks)

    @torch.no_grad()
    def evaluate():
        rows, metrics, recon_images = [], [], {}
        rendered = output/'validation'
        rendered.mkdir(exist_ok=args.render_only)
        for i in range(len(val_x)):
            target = torch.from_numpy(np.array(prepared['validation']['rgb'][i])).to(device).float()[None]/255
            x = val_x[i:i+1].float()
            pred = decode_vae_rgb(vae,bridge.latent_from_projected(x))
            # Fixed mismatched image code, rather than selecting an easy negative.
            wrong = decode_vae_rgb(vae,bridge.latent_from_projected(val_x[(i+16)%64:(i+16)%64+1].float()))
            mean = decode_vae_rgb(vae,bridge.latent_from_projected(bridge.input_mean.expand_as(x)))
            direct = decode_vae_rgb(vae,val_z[i:i+1].float())
            scores = {name:float((value-target).square().mean()) for name,value in
                      [('slot_rgb_mse',pred),('mismatched_code_mse',wrong),('mean_code_mse',mean),('vae_mse',direct)]}
            metrics.append({'index':i,'id':datasets['validation'][i]['id'],**scores})
            recon_images[i] = pil(pred[0])
            if i in [0,16,32,48]:
                for name,img in [('original',target),('vae_control',direct),('slot_rgb',pred),('mismatched_code',wrong)]:
                    pil(img[0]).save(rendered/f'{i:03d}_{name}.png')
                rows.append([pil(target[0]),pil(direct[0]),pil(pred[0]),pil(wrong[0])])
        contact_sheet(rows,['Original photo','VAE control','Slot code -> RGB','Mismatched code'],
                      output/'reconstruction_comparison.png',label_rows=[datasets['validation'][i]['id'] for i in [0,16,32,48]])
        averages = {key:sum(row[key] for row in metrics)/len(metrics) for key in metrics[0] if key.endswith('_mse')}
        averages.update({key.replace('_mse','_psnr_db'): -10*math.log10(max(value,1e-12))
                         for key,value in list(averages.items())})
        edit_rows, edit_metrics = [], []
        for number,a_index,b_index in [(1,0,48),(2,32,16)]:
            folder = output/f'pair_{number:02d}'
            folder.mkdir(exist_ok=args.render_only)
            images = []
            baseline = None
            for name in ['identity','delete','move','duplicate','replace_B']:
                program = SlotProgram.load(source/f'pair_{number:02d}'/f'{name}.program.pt',device)
                features = decode_program(discovery.decoder,program)['reconstructed_features']
                if name == 'identity':
                    torch.testing.assert_close(bridge.project(features).half(),val_x[a_index:a_index+1],
                                               rtol=0,atol=0)
                result = decode_vae_rgb(vae,bridge(features))
                if baseline is None:
                    baseline = result
                image = pil(result[0])
                image.save(folder/f'{name}.png')
                images.append(image)
                edit_metrics.append({'pair':number,'edit':name,'rgb_mse_vs_identity':float((result-baseline).square().mean()),
                                     'no_target_C':True})
            b_original = pil(torch.from_numpy(np.array(prepared['validation']['rgb'][b_index])).float()/255)
            edit_rows.append([b_original]+images)
        contact_sheet(edit_rows,['Donor B original','A code -> RGB','Delete slot','Move slot','Duplicate slot','Replace from B'],
                      output/'composition_rgb.png',label_rows=['A: Austin; B: London (Canada)','A: Lagos; B: Bogota'])
        # Training reconstruction is a capacity/underfit diagnostic, separate from held-out performance.
        train_rows=[]
        for i in [0,128,512,1024]:
            if i >= len(train_x):
                continue
            target = pil(torch.from_numpy(np.array(prepared['train']['rgb'][i])).float()/255)
            prediction = pil(decode_vae_rgb(vae,bridge.latent_from_projected(train_x[i:i+1].float()))[0])
            train_rows.append([target,prediction])
        contact_sheet(train_rows,['Training original','Training slot RGB'],output/'training_comparison.png')
        atomic_json(output/'metrics.json',{'validation_count':64,'averages':averages,'per_image':metrics,
                                         'edits':edit_metrics,'discrete_gene_model':False,
                                         'instance_or_composition_accuracy_measured':False})
        return averages

    status('rendering_validation_and_saved_edits')
    averages = evaluate()
    report = f'''# Continuous-slot RGB diagnostic\n\nCreated UTC: {now()}\n\n## What this is\n\nA frozen step9000 M1 slot model followed by a newly trained shared CNN into a frozen pretrained VAE. This is a diagnostic of continuous-slot information, not the discrete Gene program requested for the final V2 architecture. It does not establish object discovery or composition generalization.\n\nTraining: {args.train_count} images from the training split, {args.updates} latent updates, then {args.rgb_updates} RGB refinement updates. Validation: 64 separate images; fixed displayed indices 0,16,32,48. Reference RGB and VAE latents are training targets or evaluation controls. At inference the image branch receives only slot-decoded features. Edits replay previously saved programs; no target C is read.\n\nOnly one of the two original physical GPUs is shared. Main training is not paused. Original checkpoints and the frozen main source are not modified.\n\n## Metrics\n\n```json\n{json.dumps(averages,indent=2)}\n```\n\nPSNR is computed from mean MSE over 64 validation images. These are reconstruction metrics, not edit accuracy or proof of good generated scenes. Mismatched code uses a fixed offset of16 in validation, without result selection. Mean-code control uses the training-set mean projected code.\n\n## Outputs\n\n- reconstruction_comparison.png: original / VAE control / slot RGB / mismatched code.\n- composition_rgb.png: donor original and identity/delete/move/duplicate/replace RGB.\n- training_comparison.png: training reconstruction diagnostic.\n- bridge_final.pt and bridge_best_latent.pt: shared adapter weights and protocol.\n- Source step9000 weights and programs: {source}\n- Local regenerable preparation cache: {cache}\n\nKnown limits: DINO feature training discards some appearance; current slots can cover fragments or multiple objects. The regression bridge can blur fine detail, and geometry edits can leave the training distribution. No diffusion prior, discrete Gene dictionary, or A+B-to-C correctness test is included.\n'''
    (output/'REPORT.md').write_text(report)
    status('complete',averages=averages,checkpoint_step=9000,image_generation_performed=True,
           peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated())


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--composition',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--vae',required=True)
    parser.add_argument('--train-count',type=int,default=2048)
    parser.add_argument('--updates',type=int,default=1200)
    parser.add_argument('--rgb-updates',type=int,default=100)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--max-seconds',type=int,default=1500)
    parser.add_argument('--seed',type=int,default=20261112)
    parser.add_argument('--render-only',action='store_true',help='Finish evaluation from existing weights and local prepared arrays')
    args=parser.parse_args()
    try:
        main(args)
    except Exception as exc:
        from urbangene_v2.million import atomic_json
        path=Path(args.output)
        if path.exists():
            atomic_json(path/'failure.json',{'error':repr(exc),'updated_utc':now()})
        raise
