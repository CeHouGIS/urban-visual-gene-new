#!/usr/bin/env python3
"""Display every saved encoder group on its source photograph, without inference."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main(args):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from urbangene_v2.common import writable

    probe = Path(args.probe)
    grouping = probe/'grouping_rgb_figure_20261112'
    output = writable(args.output)
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(Path(__file__),output/Path(__file__).name)
    fontfile='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    title_font=ImageFont.truetype(fontfile,24)
    label_font=ImageFont.truetype(fontfile,19)
    small_font=ImageFont.truetype(fontfile,16)
    metadata={'created_utc':datetime.now(timezone.utc).isoformat(),
              'checkpoint_step':9000,'new_model_inference':False,'new_training':False,
              'input':'previously saved encoder attention and exact original RGB',
              'assignment':'argmax over32 encoder attention weights at each28x28 patch',
              'grid_resize':'nearest neighbor; no morphology or hand-selected threshold',
              'gallery':'assigned pixels keep original color, all other pixels dimmed to15%',
              'heatmap':'raw attention on fixed0..1 scale shared by all slots',
              'warning':'Slot IDs are per-image group indices, not semantic classes or verified object instances',
              'examples':[]}

    for index,city in [(0,'Austin'),(16,'Bogota'),(32,'Lagos'),(48,'London (Canada)')]:
        folder=output/f'{index:03d}_{city.split()[0].lower()}'
        folder.mkdir()
        original=Image.open(probe/'validation'/f'{index:03d}_original.png').convert('RGB')
        rgb=np.asarray(original)
        saved=np.load(grouping/f'{index:03d}_encoder_attention.npz')
        weights=saved['attention_masks']
        labels=weights.argmax(0)
        assert weights.shape==(32,28,28) and np.allclose(weights.sum(0),1,atol=1e-5)
        width,height=original.size
        enlarged=np.asarray(Image.fromarray(labels.astype('uint8')).resize((width,height),Image.Resampling.NEAREST))
        assert np.array_equal(enlarged[::16,::16],labels)
        tiles,records=[],[]
        for slot in range(32):
            selected=enlarged==slot
            assigned=labels==slot
            patches=int(assigned.sum())
            # Keep the source image as a dim spatial context, with exact winning patches bright.
            highlighted=np.rint(rgb*(.15+.85*selected[...,None])).astype('uint8')
            region=np.full_like(rgb,238)
            region[selected]=rgb[selected]
            soft=np.asarray(Image.fromarray(weights[slot]).resize((width,height),Image.Resampling.NEAREST))
            heat=np.stack([soft,soft**.5,.15*(1-soft)],-1)
            heat=np.rint(np.clip(heat,0,1)*255).astype('uint8')
            mask=Image.fromarray((selected*255).astype('uint8'))
            mask.save(folder/f'slot_{slot:02d}_assignment.png')
            Image.fromarray(highlighted).save(folder/f'slot_{slot:02d}_context.png')
            Image.fromarray(region).save(folder/f'slot_{slot:02d}_region.png')
            Image.fromarray(heat).save(folder/f'slot_{slot:02d}_attention_raw.png')
            crop_bbox=None
            if selected.any():
                ys,xs=np.nonzero(selected)
                crop_bbox=[int(xs.min()),int(ys.min()),int(xs.max()+1),int(ys.max()+1)]
                Image.fromarray(region).crop(crop_bbox).save(folder/f'slot_{slot:02d}_region_crop.png')
            triptych=Image.new('RGB',(3*448+4*12,504),'white')
            draw=ImageDraw.Draw(triptych)
            for col,(name,img) in enumerate([
                ('Location in original',Image.fromarray(highlighted)),
                ('Assigned region (argmax)',Image.fromarray(region)),
                ('Soft attention (fixed 0-1)',Image.fromarray(heat))]):
                x=12+col*460
                draw.text((x,8),f'Slot {slot:02d} | {name}',font=small_font,fill='#111827')
                triptych.paste(img,(x,36))
            draw.text((12,486),f'{city} | assigned patches: {patches}/784 ({100*patches/784:.1f}%)',
                      font=small_font,fill='#475569')
            triptych.save(folder/f'slot_{slot:02d}_detail.png')
            tiles.append(Image.fromarray(highlighted))
            records.append({'slot':slot,'background_designation':slot==0,'assigned_patches':patches,
                            'assigned_area_fraction':patches/784,'mean_attention':float(weights[slot].mean()),
                            'max_attention':float(weights[slot].max()),'assigned_bbox_pixels':crop_bbox,
                            'detail_image':str(folder/f'slot_{slot:02d}_detail.png')})

        def gallery(slots,path,columns=4,tile=288):
            rows=(len(slots)+columns-1)//columns
            gap,margin,header,cell_h,footer=16,18,85,tile+54,50
            canvas=Image.new('RGB',(2*margin+columns*tile+(columns-1)*gap,header+rows*cell_h+footer),'white')
            draw=ImageDraw.Draw(canvas)
            draw.text((margin,12),f'{city} | All encoder slot regions | Checkpoint 9000',font=title_font,fill='#0f172a')
            draw.text((margin,47),'Bright = patches assigned to this slot; dim = original photo for location.',
                      font=small_font,fill='#475569')
            for k,slot in enumerate(slots):
                x=margin+(k%columns)*(tile+gap)
                y=header+(k//columns)*cell_h
                title=f'Slot {slot:02d}'+(' (BG)' if slot==0 else '')
                draw.text((x,y),title,font=label_font,fill='#111827')
                canvas.paste(tiles[slot].resize((tile,tile),Image.Resampling.LANCZOS),(x,y+28))
                count=records[slot]['assigned_patches']
                draw.text((x,y+tile+31),f'{count} patches | {100*count/784:.1f}% area',font=small_font,fill='#475569')
            draw.text((margin,canvas.height-37),'Slots can cover object fragments, sky or road; they are not confirmed object instances.',
                      font=small_font,fill='#475569')
            canvas.save(path)

        gallery(list(range(16)),folder/'slots_00_15.png')
        gallery(list(range(16,32)),folder/'slots_16_31.png')
        gallery(list(range(32)),folder/'slots_00_31_overview.png',columns=8,tile=224)
        metadata['examples'].append({'validation_index':index,'city':city,'slots':records})
        (folder/'README.md').write_text(
            f'# {city}：逐个查看32个Slot\n\n'
            '本目录展示既有encoder attention在原图上的区域，没有额外模型推理或训练。'
            '每个28×28图块按32个Slot中最大的attention分配；保留全部Slot，包括没有赢得图块的Slot。\n\n'
            '- slots_00_15.png、slots_16_31.png：分成两页的完整32组。\n'
            '- slots_00_31_overview.png：一页总览。\n'
            '- slot_XX_detail.png：原图位置、分配区域、原始soft attention三列。\n'
            '- slot_XX_region_crop.png：分配区域的紧凑裁剪，空分组不生成该文件。\n\n'
            '高亮区域来自原图，只用于观察分组；它不是单Slot独立生成的RGB。'
            '颜色/编号没有跨图片的固定语义。Slot0按架构被指定为背景槽，但其实际区域仍由训练决定。\n')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(output/'000_austin/slots_00_15.png',flush=True)
    print(output/'000_austin/slots_16_31.png',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe',required=True)
    parser.add_argument('--output',required=True)
    main(parser.parse_args())
