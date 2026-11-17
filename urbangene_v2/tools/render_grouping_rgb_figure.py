#!/usr/bin/env python3
"""Compare matching original photographs, encoder attention groups, and saved RGB."""
import argparse
import colorsys
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main(args):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import torch
    from urbangene_v2.common import writable
    from urbangene_v2.composition import SlotProgram
    from urbangene_v2.discovery import DiscoveryModel
    from urbangene_v2.million import ShardedFeatures, atomic_json

    probe, cache = Path(args.probe), Path(args.cache)
    protocol = json.loads((probe/'protocol.json').read_text())
    gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if gpu not in protocol['existing_allocation']:
        raise ValueError('Select one GPU from the existing two-card project allocation')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Encoder inference requires exactly one GPU, no CPU fallback')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.12)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = writable(args.output)
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(Path(__file__), output/Path(__file__).name)
    source = Path(protocol['composition'])
    checkpoint = torch.load(source/'checkpoint_source.pt', map_location='cpu', weights_only=False)
    if checkpoint['step'] != protocol['checkpoint_step']:
        raise ValueError('Grouping and RGB reconstruction must use the same checkpoint')
    model = DiscoveryModel(**checkpoint['config']['model']).cuda().eval().requires_grad_(False)
    model.load_state_dict(checkpoint['model'], strict=True)
    data = Path(checkpoint['config']['data_root'])
    validation = ShardedFeatures(data/'features.validation.index.json')
    for row in validation.shards:
        local = cache/Path(row['path']).relative_to(data)
        if local.exists():
            row['path'] = str(local)
    records = json.loads((probe/'validation_selection.json').read_text())
    palette = np.array([[int(round(channel*255)) for channel in colorsys.hls_to_rgb(
        (i*.61803398875+.13)%1, [.47,.59,.51][i%3], .68)] for i in range(32)], dtype=np.uint8)
    palette[0] = [100,116,139]
    rows, metadata = {}, []
    identities = [(0,1,'A','Austin'),(16,2,'B','Bogota'),
                  (32,2,'A','Lagos'),(48,1,'B','London (Canada)')]
    with torch.inference_mode():
        for index,pair,which,city in identities:
            torch.manual_seed(10000+index)
            encoded = model.encoder(validation[index][None].cuda())
            program = SlotProgram.load(source/f'pair_{pair:02d}'/f'{which}.program.pt','cuda:0')
            # Replay the exact encoding used for the already saved RGB image.
            torch.testing.assert_close(encoded['slot_features'][0], program.content,rtol=1e-4,atol=1e-5)
            torch.testing.assert_close(encoded['estimated_geometry'][0],program.geometry,rtol=1e-4,atol=1e-5)
            weights = encoded['attention_masks'][0].cpu().numpy()
            labels = weights.argmax(0).astype(np.uint8)
            assert weights.shape == (32,28,28) and np.allclose(weights.sum(0),1,atol=1e-5)
            with Image.open(probe/'validation'/f'{index:03d}_original.png') as opened:
                original = opened.convert('RGB').copy()
            with Image.open(probe/'validation'/f'{index:03d}_slot_rgb.png') as opened:
                reconstruction = opened.convert('RGB').copy()
            assert original.size == reconstruction.size == (448,448)
            big_labels = np.asarray(Image.fromarray(labels).resize(original.size,Image.Resampling.NEAREST))
            colors = palette[big_labels]
            overlay = np.rint(.32*np.asarray(original)+.68*colors).astype(np.uint8)
            # One-pixel boundaries; do not smooth the 28x28 inferred patch assignments.
            boundary = np.zeros(big_labels.shape,dtype=bool)
            boundary[1:,:] |= big_labels[1:,:] != big_labels[:-1,:]
            boundary[:,1:] |= big_labels[:,1:] != big_labels[:,:-1]
            overlay[boundary] = [245,245,245]
            grouping = Image.fromarray(overlay)
            grouping.save(output/f'{index:03d}_encoder_grouping.png')
            Image.fromarray(palette[labels]).resize(original.size,Image.Resampling.NEAREST).save(
                output/f'{index:03d}_encoder_groups_flat.png')
            np.savez_compressed(output/f'{index:03d}_encoder_attention.npz',
                                attention_masks=weights,hard_group_ids=labels,palette=palette)
            rows[index] = (original, grouping, reconstruction, city)
            metadata.append({'validation_index':index,'id':records[index]['id'],'city':city,
                             'program':str(source/f'pair_{pair:02d}'/f'{which}.program.pt'),
                             'max_abs_slot_replay_difference':float((encoded['slot_features'][0]-program.content).abs().max()),
                             'visible_argmax_groups':int(len(np.unique(labels)))})
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,20)
    small=ImageFont.truetype(font_path,17)

    def figure(indices,path):
        margin,gap,tile=18,16,448
        width=2*margin+3*tile+2*gap
        top,row_height,footer=64,486,40
        canvas=Image.new('RGB',(width,top+row_height*len(indices)+footer),'white')
        draw=ImageDraw.Draw(canvas)
        for col,title in enumerate(['Original photo','Encoder grouping','Slot code to RGB reconstruction']):
            x=margin+col*(tile+gap)
            box=draw.textbbox((0,0),title,font=font)
            draw.text((x+(tile-(box[2]-box[0]))/2,20),title,font=font,fill='#111827')
        for row,index in enumerate(indices):
            original,grouping,reconstruction,city=rows[index]
            y=top+row*row_height
            for col,image in enumerate([original,grouping,reconstruction]):
                canvas.paste(image,(margin+col*(tile+gap),y))
            draw.text((margin,y+tile+7),city,font=small,fill='#334155')
        draw.text((margin,canvas.height-30),
                  'Checkpoint 9000 | Encoder attention argmax on 28 x 28 patches | Colors indicate slot IDs within each image',
                  font=small,fill='#475569')
        canvas.save(path)

    figure([0,16,32,48],output/'original_encoder_grouping_slot_rgb.png')
    figure([0,32],output/'original_encoder_grouping_slot_rgb_two_examples.png')
    atomic_json(output/'figure_metadata.json',{
        'created_utc':datetime.now(timezone.utc).isoformat(), 'checkpoint_step':checkpoint['step'],
        'gpu_uuid':gpu,'new_training':False,'cpu_model_fallback':False,
        'grouping_source':'encoder attention_masks, NOT decoder soft_masks',
        'grouping_display':'argmax over32 slots per28x28 patch, nearest resize, colored overlay and1px boundaries',
        'group_colors':'slot indices within each image, not shared semantic classes',
        'original_and_rgb_source':str(probe/'validation'),'examples':metadata})
    print(output/'original_encoder_grouping_slot_rgb.png',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',required=True)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--output',required=True)
    main(parser.parse_args())
