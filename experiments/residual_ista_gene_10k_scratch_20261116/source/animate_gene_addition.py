"""Progressive real dictionary synthesis and frozen decoding; no RGB interpolation."""
import argparse
import gc
import json
import os
from pathlib import Path
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from common import load_config, write_json, now
from reference.supervisor import acquire


def export_frames(config, destination, test_index, chunk):
    leased=acquire(config,1)
    while not leased:
        write_json(destination/'progress.json',{'status':'waiting_for_gpu','updated_utc':now()})
        time.sleep(10)
        leased=acquire(config,1)
    gpu,handle=leased[0]
    try:
        os.environ['CUDA_VISIBLE_DEVICES']=gpu['uuid']
        import torch
        from model import create_model
        from data import image_tensor
        torch.set_num_threads(4)
        root=Path(config['root'])
        checkpoint=root/'runs/multiscale_residual_ista/checkpoint_best.pt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        model=create_model(config,'multiscale_residual_ista')
        model.load_state_dict(saved['model'])
        model.set_step(saved['step'])
        model.eval()
        pixels=np.load(Path(config['cache'])/'test.rgb.npy',mmap_mode='r',allow_pickle=False)
        target=image_tensor(pixels,np.array([test_index]))
        enabled=torch.cuda.is_bf16_supported()
        def amp():
            return torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled)
        with torch.inference_mode(),amp():
            full=model(target)
        activations=full['activations']
        scales=list(model.scales)
        ordering={}
        for scale in scales:
            masses=activations[scale].float().sum((0,2,3)).cpu().numpy()
            used=np.flatnonzero(masses>0)
            ordering[scale]=used[np.lexsort((used,-masses[used]))].tolist()
        current={s:torch.zeros_like(full['features'][s]) for s in scales}
        states=[{s:current[s].clone() for s in scales}]
        counts={s:0 for s in scales}
        specifications=[{'phase':'zero','counts':dict(counts),'added_scale':None,'added_gene_ids':[]}]
        synthesis_errors={}
        with torch.inference_mode():
            for scale in scales:
                masked=torch.zeros_like(activations[scale])
                order=ordering[scale]
                for start in range(0,len(order),chunk):
                    chosen=order[start:start+chunk]
                    masked[:,chosen]=activations[scale][:,chosen]
                    with amp():
                        current[scale]=model.dictionaries[scale](masked)*model.gates[scale]
                    counts[scale]+=len(chosen)
                    states.append({s:current[s].clone() for s in scales})
                    specifications.append({'phase':scale,'counts':dict(counts),
                                           'added_scale':scale,'added_gene_ids':chosen})
                synthesis_errors[scale]=float((current[scale]-full['decoder_features'][scale]).abs().max())
            outputs=[]
            batch_size=8
            for start in range(0,len(states),batch_size):
                batch={s:torch.cat([frame[s] for frame in states[start:start+batch_size]]) for s in scales}
                with amp():
                    decoded=model.decoder(batch)
                outputs.extend(decoded.float().cpu().numpy())
                write_json(destination/'progress.json',{'status':'decoding','frames_done':min(start+batch_size,len(states)),
                    'frames_total':len(states),'GPU':gpu,'updated_utc':now()})
            # Same single-image batch shape gives an independent endpoint check.
            with amp():
                endpoint=model.decoder(states[-1])
            endpoint_error=float((endpoint.float()-full['reconstruction'].float()).abs().max())
            if max(synthesis_errors.values())>1e-6 or endpoint_error>1e-6:
                raise AssertionError(f'Final feature replay differs: {synthesis_errors}, RGB {endpoint_error}')
            # Save the exactly checked endpoint rather than a batch-kernel variant.
            outputs[-1]=endpoint[0].float().cpu().numpy()
        arrays={'original':target[0].cpu().numpy(),'decoded_frames':np.stack(outputs),
                'full_reconstruction':full['reconstruction'][0].float().cpu().numpy()}
        np.savez_compressed(destination/'animation_frames_float.npz',**arrays)
        image_record=json.loads((root/'data/images.test.jsonl').read_text().splitlines()[test_index])
        manifest={'checkpoint':str(checkpoint),'checkpoint_step':saved['step'],'test_index':test_index,
            'image_id':image_record['id'],'image_city':image_record['city'],
            'source_resolution':[256,256],'GPU':gpu,'BF16':enabled,'frames':specifications,
            'active_genes_per_scale':{s:len(ordering[s]) for s in scales},'gene_order':ordering,
            'genes_per_step':chunk,'scale_order':scales,
            'within_scale_order':'descending sum of nonnegative activation coefficients, then gene ID',
            'frame_construction':'Freeze original A. Unmask complete spatial activation maps of selected genes. Recompute Z_s=D_s*A_s. Decode all scales with the same frozen nonlinear decoder.',
            'rgb_interpolation':False,'feature_renormalization':False,'coefficients_reinferred_per_frame':False,
            'inactive_genes_skipped':True,'final_feature_max_abs_errors':synthesis_errors,
            'final_rgb_endpoint_max_abs_error':endpoint_error,'full_final_mae':float(np.abs(outputs[-1]-arrays['original']).mean()),
            'zero_code_is_learned_decoder_output_not_a_blank_canvas':True,
            'note':'Partial codes are interventions not explicitly trained as valid scenes. No semantic ordering or strict monotonic reconstruction improvement is assumed.',
            'training_performed':False,'updated_utc':now()}
        write_json(destination/'manifest.json',manifest)
        print(json.dumps({'status':'frames_exported','count':len(states),'active_genes':manifest['active_genes_per_scale'],
                          'final_feature_errors':synthesis_errors,'final_rgb_error':endpoint_error}),flush=True)
    finally:
        handle.close()


def render_gif(destination):
    arrays=np.load(destination/'animation_frames_float.npz',allow_pickle=False)
    metadata=json.loads((destination/'manifest.json').read_text())
    specs=metadata['frames']
    side=384;pad=16;gap=16;width=2*side+2*pad+gap;height=614;image_y=83
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    fonts={size:ImageFont.truetype(font_path,size) for size in [13,16,18,22]}
    colors={'coarse':'#f3b562','medium':'#59c8d4','fine':'#b59bff'}
    scales=['coarse','medium','fine']
    grids={'coarse':'16x16','medium':'32x32','fine':'64x64'}
    def tile(value):
        rgb=np.rint(np.clip(value.transpose(1,2,0),0,1)*255).astype(np.uint8)
        return Image.fromarray(rgb).resize((side,side),Image.Resampling.BICUBIC)
    original=tile(arrays['original'])
    totals=metadata['active_genes_per_scale'];total=sum(totals.values())
    frames=[];durations=[]
    for index,(spec,prediction) in enumerate(zip(specs,arrays['decoded_frames'])):
        canvas=Image.new('RGB',(width,height),'#101820');draw=ImageDraw.Draw(canvas)
        draw.text((pad,12),'Progressively adding visual gene features',font=fonts[22],fill='white')
        right=pad+side+gap
        draw.text((pad,52),'Original photo (fixed)',font=fonts[18],fill='#e1e7eb')
        draw.text((right,52),'Decoded accumulated features',font=fonts[18],fill='#e1e7eb')
        canvas.paste(original,(pad,image_y));canvas.paste(tile(prediction),(right,image_y))
        phase=spec['phase']
        added=sum(spec['counts'].values())
        if index==len(specs)-1:
            stage='Complete reconstruction: all active genes included'
        elif phase=='zero':
            stage='Start: all gene coefficients set to zero'
        else:
            stage=f'Adding {phase} features ({grids[phase]} grid)'
        draw.text((pad,481),stage,font=fonts[18],fill=colors.get(phase,'#e1e7eb'))
        draw.text((pad,510),f'Genes added: {added}/{total}   |   One step adds up to {metadata["genes_per_step"]} genes',font=fonts[16],fill='#c1cbd2')
        segment_width=(width-2*pad-2*gap)//3
        for s_index,s in enumerate(scales):
            x=pad+s_index*(segment_width+gap);y=542
            n=spec['counts'][s];den=totals[s]
            draw.text((x,y),f'{s} {n}/{den}',font=fonts[16],fill=colors[s])
            draw.rounded_rectangle((x,y+25,x+segment_width,y+33),radius=3,fill='#34424d')
            if n:
                draw.rounded_rectangle((x,y+25,x+round(segment_width*n/den),y+33),radius=3,fill=colors[s])
        draw.text((pad,590),'Frozen spatial codes and decoder; every frame is decoded from the accumulated feature sum.',font=fonts[13],fill='#a8b7c2')
        frames.append(canvas)
        at_boundary=index+1==len(specs) or specs[index+1]['phase']!=phase
        durations.append(1800 if index==len(specs)-1 else 1000 if index==0 else 650 if at_boundary else 100)
    # One shared palette prevents a changing color map from moving the fixed original.
    sample_ids=np.linspace(0,len(frames)-1,min(20,len(frames)),dtype=int)
    small=[frames[i].resize((204,154),Image.Resampling.BILINEAR) for i in sample_ids]
    palette_sheet=Image.new('RGB',(204*5,154*((len(small)+4)//5)))
    for n,im in enumerate(small):
        palette_sheet.paste(im,((n%5)*204,(n//5)*154))
    palette=palette_sheet.quantize(colors=256,method=Image.Quantize.MEDIANCUT)
    indexed=[frame.quantize(palette=palette,dither=Image.Dither.NONE) for frame in frames]
    output=destination/'progressive_gene_addition.gif'
    indexed[0].save(output,save_all=True,append_images=indexed[1:],duration=durations,loop=0,
                    disposal=2,optimize=False)
    stage_ids=[0]+[i for i,s in enumerate(specs) if i>0 and (i==len(specs)-1 or specs[i+1]['phase']!=s['phase'])]
    contact=Image.new('RGB',(width*2,height*2),'#101820')
    for i,idx in enumerate(stage_ids):
        contact.paste(frames[idx],((i%2)*width,(i//2)*height))
    contact.save(destination/'stage_contact_sheet.png')
    frames[-1].save(destination/'final_frame.png')
    # Check the serialized artifact and that the displayed original stays fixed.
    with Image.open(output) as gif:
        first=None
        for index in range(gif.n_frames):
            gif.seek(index)
            image=gif.convert('RGB')
            crop=np.array(image.crop((pad,image_y,pad+side,image_y+side)))
            if first is None:
                first=crop
            elif not np.array_equal(first,crop):
                raise AssertionError('Original-photo panel changed during GIF playback')
        gif_frames=gif.n_frames
        if gif_frames!=len(specs):
            raise AssertionError(f'Unexpected GIF frame count: {gif_frames} vs {len(specs)}')
    write_json(destination/'render_report.json',{'status':'passed','GIF':str(output),'size_bytes':output.stat().st_size,
        'frames':gif_frames,'width':width,'height':height,'loop':True,'duration_seconds':sum(durations)/1000,
        'source_stays_fixed':True,'palette':'shared 256-color palette; GIF quantization changes display colors slightly',
        'keyframes':stage_ids,'updated_utc':now()})
    (destination/'README.md').write_text('# 特征逐步相加动画\n\n左侧原图固定，右侧是真实冻结解码器的输出。示例为测试集第 199 张（从 0 开始编号），即此前展示的第二张街景。\n\n从所有系数为 0 开始，依次添加粗、中、细尺度的基因；每个基因加入的是整张空间激活图。每尺度按系数总量从高到低排序，每步最多添加 4 个。跳过这张图片上完全未激活的基因。每帧重新计算字典线性合成 Z_s=D_s*A_s，使用同一解码器生成 RGB，没有在 RGB 图像之间做淡入。\n\n初始画面是零输入特征经过已训练解码器的输出，不保证空白。部分编码的中间画面未经专门训练，不保证合理街景或误差单调下降。最终特征与完整编码严格核对，最终 RGB 端点也与完整重建核对。GIF 的 256 色量化及显示放大只影响显示。\n\nmanifest.json 记录全部加入顺序、检查点、端点误差；animation_frames_float.npz 保存原始浮点解码帧。stage_contact_sheet.png 为初始、粗尺度完成、中尺度完成、全部完成的四阶段对照。\n')
    write_json(destination/'progress.json',{'status':'complete','GIF':str(output),'updated_utc':now()})
    print(json.dumps({'status':'complete','GIF':str(output),'frames':gif_frames,'duration_seconds':sum(durations)/1000,
                      'MB':round(output.stat().st_size/1e6,2)}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--test-index',type=int,default=199)
    parser.add_argument('--genes-per-step',type=int,default=4)
    args=parser.parse_args()
    if args.genes_per_step<1:
        raise ValueError('Positive gene chunk size required')
    c=load_config(args.config)
    output=Path(c['root'])/'reports'/f'gene_addition_gif_test_{args.test_index:04d}'
    output.mkdir(exist_ok=False)
    export_frames(c,output,args.test_index,args.genes_per_step)
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    render_gif(output)
