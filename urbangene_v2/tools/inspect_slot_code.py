#!/usr/bin/env python3
"""Inspect stored slot tensors as data; performs no model inference or training."""
import argparse
import json
from pathlib import Path
import shutil


def main(args):
    import numpy as np
    import torch
    from PIL import Image
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    probe, source = Path(args.probe), Path(args.program)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(Path(__file__),output/Path(__file__).name)
    raw=torch.load(source,map_location='cpu',weights_only=True)
    content, geometry=raw['content'].numpy(),raw['geometry'].numpy()
    gate=torch.sigmoid(raw['existence']).numpy()
    selected=30
    arrays=np.load(probe/'grouping_rgb_figure_20261112/000_encoder_attention.npz')
    weights=arrays['attention_masks'][selected]
    original=Image.open(probe/'validation/000_original.png').convert('RGB')
    reconstructed=Image.open(probe/'validation/000_slot_rgb.png').convert('RGB')
    overlay=np.asarray(original).astype(float)
    weight_image=np.asarray(Image.fromarray(weights).resize(original.size,Image.Resampling.NEAREST))
    # Display all unthresholded attention values, without a hand-drawn object mask.
    intensity=weight_image/max(float(weights.max()),1e-12)
    tinted=overlay*(.25+.75*intensity[...,None])
    fig=plt.figure(figsize=(15,9),layout='constrained',facecolor='white')
    grid=fig.add_gridspec(2,3,height_ratios=[1.1,1.15],width_ratios=[1,1,1])
    for col,(title,img) in enumerate([
            ('Original photo',np.asarray(original)),
            ('Encoder attention for Slot 30',tinted.astype('uint8')),
            ('Slot code to RGB reconstruction',np.asarray(reconstructed))]):
        ax=fig.add_subplot(grid[0,col]);ax.imshow(img);ax.set_title(title,fontsize=13,pad=10);ax.axis('off')
    ax=fig.add_subplot(grid[1,:2])
    limit=float(np.max(np.abs(content)))
    heat=ax.imshow(content,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto',interpolation='nearest')
    ax.add_patch(Rectangle((-.5,selected-.5),127.99,.99,fill=False,edgecolor='#ffcc00',linewidth=2))
    ax.set_title('Stored content matrix: 32 slots x 128 continuous values',fontsize=13,pad=10)
    ax.set_xlabel('Embedding dimension');ax.set_ylabel('Slot index')
    ax.set_xticks(list(range(0,128,16))+[127]);ax.set_yticks(list(range(0,32,4))+[30,31])
    fig.colorbar(heat,ax=ax,shrink=.9,pad=.02,label='Raw vector value')
    detail=fig.add_subplot(grid[1,2]);detail.axis('off')
    first=content[selected,:12]
    lines=['Slot 30: actual saved values','',
           'content[0:12] of 128:']
    for start in range(0,12,3):
        lines.append('  '+', '.join(f'{v:+.4f}' for v in first[start:start+3]))
    cx,cy,sx,sy=geometry[selected]
    lines += ['',f'center_xy = ({cx:+.4f}, {cy:+.4f})',
              f'scale_xy  = ({sx:.4f}, {sy:.4f})',
              f'gate logit = {raw["existence"][selected].item():.4f}',
              f'sigmoid gate = {gate[selected]:.4f}',
              f'active = {bool(raw["active"][selected])}',
              '', 'Position uses normalized coordinates.',
              'Scales are spatial standard deviations.',
              'Gate is not calibrated object confidence.']
    detail.text(0,.98,'\n'.join(lines),va='top',family='monospace',fontsize=10.5,linespacing=1.5)
    fig.suptitle('Actual reconstruction code | Austin | checkpoint 9000',fontsize=17)
    fig.savefig(output/'actual_slot_code.png',dpi=150)
    plt.close(fig)
    np.savetxt(output/'slot_content_32x128.csv',content,delimiter=',',fmt='%.8g')
    entries=[]
    for i in range(len(content)):
        entries.append({'slot_index':i,'content':content[i].tolist(),'geometry':geometry[i].tolist(),
                        'existence_logit':float(raw['existence'][i]),'sigmoid_gate':float(gate[i]),
                        'active':bool(raw['active'][i]),'origin':raw['origins'][i]})
    (output/'actual_slot_code.json').write_text(json.dumps({
        'source_program':str(source),'format':raw['format'],'discrete_gene_program':False,
        'content_shape':list(content.shape),'geometry_shape':list(geometry.shape),
        'existence_shape':list(raw['existence'].shape),'active_shape':list(raw['active'].shape),
        'geometry_order':['center_x','center_y','spatial_std_x','spatial_std_y'],
        'encoder_masks_saved_in_program':False,'soft_masks_for_figure':'evaluation visualization only',
        'background_slot_index':0,'gate_is_calibrated_object_confidence':False,'slots':entries,
        },indent=2)+'\n')
    print(output/'actual_slot_code.png')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program',required=True)
    parser.add_argument('--probe',required=True)
    parser.add_argument('--output',required=True)
    main(parser.parse_args())
