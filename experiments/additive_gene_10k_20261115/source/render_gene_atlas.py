"""All 256 dictionary entries: natural activations and explicit decoder probes.

Use saved held-out codes, the fixed checkpoint, and a leased GPU. No training,
label assignment, activation maximization, or invented examples for unused genes.
"""
import argparse
import csv
import html
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import load_config, now, write_json


def render(config, output):
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    import torch
    from torch.nn import functional as F
    from data import image_tensor, open_pixels
    from model import sum_contributions
    from train import amp_context, create_model, to_pil

    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError('This rendering requires exactly one leased CUDA GPU')
    torch.set_num_threads(4)
    root=Path(config['root'])
    source=root/'reports/unseen_prediction_20261115/test_gene_codes.npz'
    codes=np.load(source,allow_pickle=False)
    indices=codes['gene_indices']
    values=codes['activation_values']
    dictionary=codes['dictionary']
    rows=[json.loads(x) for x in (root/'data/images.test.jsonl').read_text().splitlines()]
    if not np.array_equal(codes['test_ids'],np.array([r['id'] for r in rows])):
        raise ValueError('Saved code IDs and test manifest differ')
    n,c=dictionary.shape
    count,k,h,w=indices.shape
    if indices.shape!=values.shape or n!=256 or count!=len(rows):
        raise ValueError('Unexpected dictionary or saved code shapes')
    mass=np.zeros((count,n),np.float64)
    position_counts=np.zeros(n,np.int64)
    for i in range(count):
        positive=values[i]>0
        mass[i]=np.bincount(indices[i][positive],weights=values[i][positive],minlength=n)
        position_counts+=np.bincount(indices[i][positive],minlength=n)
    old=json.loads((root/'runs/sparse_n256_s4/test_metrics.json').read_text())
    if not np.array_equal(position_counts,np.array(old['gene_activation_counts'])):
        raise AssertionError('Activation accounting does not reproduce final test evaluation')
    shared_scale=float(np.quantile(values[values>0],0.99))
    probe_amplitude=float(np.median(values[values>0]))
    usage=(mass>0).sum(0)
    top_indices=[np.argsort(-mass[:,gene],kind='stable')[:min(3,int(usage[gene]))].tolist() for gene in range(n)]
    model=create_model(config,'sparse_n256_s4')
    checkpoint_path=root/'runs/sparse_n256_s4/checkpoint_best.pt'
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    if not np.allclose(model.dictionary.detach().cpu().numpy(),dictionary,atol=1e-6):
        raise AssertionError('Dictionary checkpoint and saved code dictionary differ')
    pixels=open_pixels(config,'test')
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    try:
        font=ImageFont.truetype(font_path,14)
        small=ImageFont.truetype(font_path,12)
        titlefont=ImageFont.truetype(font_path,20)
    except OSError:
        font=small=titlefont=ImageFont.load_default()
    directories=['cards','probes','natural_overlays','only_gene_rgb','thumbnails']
    for directory in directories:
        (output/directory).mkdir()
    entries=[]
    probe_images=[]
    natural_images=[]
    started=time.monotonic()

    def activation_tensor(sample_indices):
        ids=torch.from_numpy(indices[sample_indices].astype(np.int64)).cuda()
        coeff=torch.from_numpy(values[sample_indices]).cuda()
        return torch.zeros(len(sample_indices),n,h,w,device='cuda').scatter_(1,ids,coeff)

    with torch.inference_mode():
        # Check that exported sparse codes reproduce the live encoder path.
        # Match export batch size: bf16 convolution rounding can change TopK ties
        # when encoder kernel selection changes with the batch shape.
        replay_batch=min(config['training']['micro_batch_size'],count)
        replay_indices=np.arange(replay_batch)
        target=image_tensor(pixels,replay_indices,'cuda')
        with amp_context(config):
            live=model(target)
            recovered=sum_contributions(activation_tensor(replay_indices),model.dictionary)
            recovered_rgb=model.decoder(recovered)
        replay_error=float((recovered_rgb.float()-live['reconstruction'].float()).abs().max())
        if replay_error>0.02:
            raise AssertionError(f'Saved code replay diverges unexpectedly: {replay_error}')
        del live,recovered,recovered_rgb,target
        zero_latent=torch.zeros(1,c,h,w,device='cuda')
        with amp_context(config):
            zero_rgb=model.decoder(zero_latent).float()
        to_pil(zero_rgb[0]).save(output/'zero_code_decoder_baseline.png')
        yy,xx=torch.meshgrid(torch.arange(h,device='cuda'),torch.arange(w,device='cuda'),indexing='ij')
        gaussian=torch.exp(-((xx-(w-1)/2).square()+(yy-(h-1)/2).square())/(2*2.0**2))
        gaussian=gaussian/gaussian.max()
        Image.fromarray((gaussian.cpu().numpy()*255).round().astype('uint8')).resize((256,256)).save(output/'probe_activation_pattern.png')
        for begin in range(0,n,16):
            with amp_context(config):
                probe_latent=model.dictionary[begin:begin+16,:,None,None]*gaussian[None,None]*probe_amplitude
                probe_rgb=model.decoder(probe_latent).float()
            for offset in range(len(probe_rgb)):
                gene=begin+offset
                probe=to_pil(probe_rgb[offset])
                probe.save(output/f'probes/gene_{gene:03d}.png')
                probe_images.append(probe)

        for gene in range(n):
            selected=top_indices[gene]
            entry={'gene':gene,'test_images_active':int(usage[gene]),'test_activation_positions':int(position_counts[gene]),
                'total_activation_mass':float(mass[:,gene].sum()),'active_in_test':bool(usage[gene]),
                'probe_amplitude':probe_amplitude,'selected_test_indices':selected,
                'selected_test_ids':[rows[i]['id'] for i in selected],
                'natural_example_ranking':'sum of positive spatial coefficients, descending',
                'card':f'cards/gene_{gene:03d}.png','probe':f'probes/gene_{gene:03d}.png',
                'thumbnail':f'thumbnails/gene_{gene:03d}.jpg'}
            height=190+len(selected)*300 if selected else 450
            card=Image.new('RGB',(1536,height),'white')
            draw=ImageDraw.Draw(card)
            draw.text((10,7),f'Gene {gene:03d} | {int(usage[gene])}/1024 test images | {int(position_counts[gene])} active positions',fill='#172b3a',font=titlefont)
            draw.text((10,38),'64-D dictionary vector (signed coefficients; not RGB)',fill='#273645',font=font)
            strip=np.repeat(np.clip((dictionary[gene]+1)/2,0,1)[None],20,axis=0)
            signed=np.zeros((20,c,3),np.uint8)
            signed[:,:,0]=np.clip(255*np.maximum(dictionary[gene],0)[None]*4,0,255).astype('uint8')
            signed[:,:,2]=np.clip(255*np.maximum(-dictionary[gene],0)[None]*4,0,255).astype('uint8')
            signed[:,:,1]=30
            card.paste(Image.fromarray(signed).resize((512,30),Image.Resampling.NEAREST),(10,62))
            draw.text((10,99),'Red: positive; blue: negative. Same display scale across all genes.',fill='#526071',font=small)
            card.paste(probe_images[gene].resize((128,128)),(560,38))
            draw.text((702,42),'Standard decoder probe',fill='#273645',font=font)
            draw.text((702,66),'Only this gene, same central activation pattern and amplitude.',fill='#526071',font=small)
            draw.text((702,86),'This isolated code can be outside the training distribution.',fill='#526071',font=small)
            draw.text((702,106),'It is a decoder response, not an object or semantic label.',fill='#526071',font=small)
            draw.text((10,171),'Natural examples below: strongest 3 test photos by total activation. Shared activation scale; RGB difference x3.',fill='#273645',font=font)
            if selected:
                acts=activation_tensor(selected)
                with amp_context(config):
                    latent=sum_contributions(acts,model.dictionary)
                    reconstruction=model.decoder(latent).float()
                    contribution=model.contribution(acts,gene)
                    only_rgb=model.decoder(contribution).float()
                    removed_rgb=model.decoder(latent-contribution).float()
                ablation=(reconstruction-removed_rgb).abs()
                entry['natural_examples']=[]
                for row_number,sample_index in enumerate(selected):
                    base=190+row_number*300
                    source_image=Image.fromarray(np.array(pixels[sample_index]))
                    activation=acts[row_number,gene].cpu().numpy()
                    heat=Image.fromarray(np.clip(activation/shared_scale*255,0,255).astype('uint8')).resize((256,256),Image.Resampling.NEAREST)
                    heat_array=np.array(heat).astype(np.float32)/255
                    color=np.stack([np.ones_like(heat_array),np.full_like(heat_array,0.2),np.zeros_like(heat_array)],axis=-1)
                    alpha=0.7*heat_array[...,None]
                    overlay=Image.fromarray(np.rint(np.array(source_image)*(1-alpha)+color*255*alpha).clip(0,255).astype('uint8'))
                    tiles=[source_image,to_pil(reconstruction[row_number]),overlay,to_pil(only_rgb[row_number]),
                           to_pil(removed_rgb[row_number]),to_pil(ablation[row_number]*3)]
                    labels=['Original test photo','Full reconstruction','Gene activation overlay','Only this gene -> RGB',
                            'Reconstruction minus gene','Absolute removal effect x3']
                    for column,(label,tile) in enumerate(zip(labels,tiles)):
                        draw.text((column*256+5,base+2),label,fill='#273645',font=font)
                        card.paste(tile,(column*256,base+24))
                    info=f'Test #{sample_index} | {rows[sample_index]["city"]} | {rows[sample_index]["id"]} | activation mass {mass[sample_index,gene]:.3f}'
                    draw.text((5,base+281),info,fill='#526071',font=small)
                    entry['natural_examples'].append({'test_index':sample_index,'id':rows[sample_index]['id'],
                        'city':rows[sample_index]['city'],'activation_mass':float(mass[sample_index,gene]),
                        'removal_rgb_mae':float(ablation[row_number].mean())})
                    if row_number==0:
                        overlay.save(output/f'natural_overlays/gene_{gene:03d}.png')
                        to_pil(only_rgb[0]).save(output/f'only_gene_rgb/gene_{gene:03d}.png')
                        natural_images.append(overlay)
                thumb=Image.new('RGB',(256,128),'white')
                thumb.paste(natural_images[-1].resize((128,128)),(0,0))
                thumb.paste(probe_images[gene].resize((128,128)),(128,0))
            else:
                draw.text((10,210),'NOT ACTIVATED IN THESE 1024 TEST PHOTOS',fill='#7b4e19',font=titlefont)
                draw.text((10,250),'No natural activation, isolated natural code, or removal effect is available for this gene in this test set.',fill='#526071',font=font)
                draw.text((10,280),'The probe above is synthetic. Test inactivity does not prove permanent inactivity during training.',fill='#526071',font=font)
                blank=Image.new('RGB',(256,256),'#eceff2')
                blank_draw=ImageDraw.Draw(blank)
                blank_draw.text((24,112),'No test activation',fill='#64748b',font=font)
                natural_images.append(blank)
                thumb=Image.new('RGB',(256,128),'white')
                thumb.paste(blank.resize((128,128)),(0,0))
                thumb.paste(probe_images[gene].resize((128,128)),(128,0))
            card.save(output/entry['card'])
            thumb.save(output/entry['thumbnail'],quality=90)
            entries.append(entry)
            write_json(output/'progress.json',{'status':'rendering','genes_completed':gene+1,'total_genes':n,'updated_utc':now()})

    def atlas(images,first,last,columns,cell_side,name,kind):
        row_count=(last-first+columns-1)//columns
        cell_height=cell_side+30
        canvas=Image.new('RGB',(columns*cell_side,row_count*cell_height+50),'white')
        painter=ImageDraw.Draw(canvas)
        painter.text((8,8),f'Genes {first:03d}-{last-1:03d} | {kind}',fill='#172b3a',font=titlefont)
        painter.text((8,31),'Number = gene ID; images = test photos with activation. No test activation is shown in grey.',fill='#526071',font=small)
        for gene in range(first,last):
            x=((gene-first)%columns)*cell_side
            y=50+((gene-first)//columns)*cell_height
            canvas.paste(images[gene].resize((cell_side,cell_side)),(x,y+28))
            painter.text((x+4,y+5),f'{gene:03d} | {int(usage[gene])}',fill='#172b3a',font=small)
        canvas.save(output/name)

    atlas(natural_images,0,256,16,112,'all_256_natural_activation_atlas.png','natural activation overlays')
    atlas(probe_images,0,256,16,112,'all_256_probe_atlas.png','isolated standard decoder probes')
    for page in range(4):
        atlas(natural_images,page*64,(page+1)*64,8,144,f'natural_genes_{page*64:03d}_{(page+1)*64-1:03d}.png','natural activation overlays')
        atlas(probe_images,page*64,(page+1)*64,8,144,f'probe_genes_{page*64:03d}_{(page+1)*64-1:03d}.png','isolated decoder probes')
    build_natural_only_atlas(output,entries)
    most_used=np.argsort(-position_counts,kind='stable')[:4].tolist()
    summary=Image.new('RGB',(256*4,42+4*(256+30)),'white')
    painter=ImageDraw.Draw(summary)
    for col,label in enumerate(['Natural activation overlay','Only natural gene -> RGB','Standard decoder probe','Gene ID / test usage']):
        painter.text((col*256+5,12),label,fill='#172b3a',font=font)
    for row_number,gene in enumerate(most_used):
        y=42+row_number*286
        summary.paste(natural_images[gene],(0,y))
        summary.paste(Image.open(output/f'only_gene_rgb/gene_{gene:03d}.png'),(256,y))
        summary.paste(probe_images[gene],(512,y))
        painter.text((780,y+60),f'Gene {gene:03d}',fill='#172b3a',font=titlefont)
        painter.text((780,y+100),f'{int(usage[gene])}/1024 test photos',fill='#526071',font=font)
        painter.text((780,y+130),f'{int(position_counts[gene])} positions',fill='#526071',font=font)
    summary.save(output/'most_used_4_overview.png')

    write_json(output/'genes.json',entries)
    with (output/'gene_usage.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['gene','test_images_active','test_activation_positions','total_activation_mass','active_in_test'])
        writer.writeheader()
        for entry in entries:
            writer.writerow({key:entry[key] for key in writer.fieldnames})
    np.savetxt(output/'dictionary_vectors.csv',dictionary,delimiter=',',header=','.join(f'dim_{i}' for i in range(c)),comments='')
    result={'status':'complete','genes':n,'active_in_test':int((usage>0).sum()),'inactive_in_test':int((usage==0).sum()),
        'test_images':count,'single_gene_cards':len(list((output/'cards').glob('*.png'))),
        'checkpoint':str(checkpoint_path),'checkpoint_step':checkpoint['step'],
        'weights_updated':False,'code_source':str(source),'code_replay_max_abs_rgb_error':replay_error,
        'code_replay_batch_size':replay_batch,
        'encoding_precision_note':'Replay uses original export batch size. Changing batch size can change bf16 encoder rounding and TopK membership; all atlas interventions use the same saved coefficients.',
        'activation_heatmap_scale_q99':shared_scale,'probe_amplitude_global_nonzero_median':probe_amplitude,
        'probe_pattern':'16x16 central Gaussian, sigma=2 latent grid cells, peak=1',
        'heatmap':'red opacity proportional to coefficient, same global q99 scale with clipping',
        'dictionary_strip':'red positive, blue negative, shared fixed scale; dimensions are not RGB',
        'selection':'top 3 test photos per gene by summed activation; not a random example set',
        'most_used_overview_gene_ids':most_used,'render_elapsed_seconds':round(time.monotonic()-started,2),'updated_utc':now()}
    write_json(output/'summary.json',result)
    build_gallery(output,entries)
    report='''# 全部 256 个基因的独立图册

本图册使用固定的第 6000 步共享字典模型和 1024 张测试图的实际编码，仅作推理，没有重新训练。每个基因均有独立 PNG 卡片，含 64 维字典向量、标准单基因解码探测；在测试集激活的基因另外展示最强的 3 张自然照片、激活位置、单基因独立解码和移除该基因后的影响。

## 如何读图

- **自然激活图**：原照片叠加该基因的空间激活，所有基因使用统一强度色标。它是激活位置，不是实例边界，也不能据此直接给基因命名。
- **Only this gene -> RGB**：在一张真实照片的编码中只保留这一基因的空间贡献，直接输入解码器。图像可能模糊、变色或呈现全局结构，因为该输入偏离正常多基因协同编码。
- **Reconstruction minus gene**：保持其它贡献不变，仅移除这一基因。差异图取完整重建和移除后的 RGB 绝对差乘 3，不等于可线性相加的 RGB 图层。
- **标准探测图**：256 个基因都使用相同的中央高斯激活图和幅值，分别解码。此图描述解码器对单个方向的响应，不能解释为基因自身代表的物体。
- **未激活项**：116 个基因在当前测试集没有非零激活，以灰底标注。保留其向量和标准探测，不虚构自然激活例子；这不等于它在整个训练过程中从未使用。

解码器是非线性网络：D(sum g_k) 一般不等于 sum D(g_k)。独立基因图不能当作像素图层直接叠回原图。全零编码的解码结果单独保存在 zero_code_decoder_baseline.png，便于检查解码器本身的默认输出。

## 文件

- index.html：可按编号检索、按测试活跃状态筛选的全部图册；点击缩略图打开对应完整卡片。
- cards/gene_000.png 至 gene_255.png：全部 256 个基因的独立卡片。
- all_256_natural_activation_atlas.png：全部基因的自然激活概览，灰底为未激活项。
- all_256_only_gene_rgb_atlas.png：每个基因取其激活最强的测试照片，只保留该基因进行 RGB 解码；灰底表示没有自然激活样例。
- all_256_probe_atlas.png：全部基因的标准解码探测概览。
- natural_genes_*.png、probe_genes_*.png：每页 64 个基因的清晰分页。
- most_used_4_overview.png：按非零位置数排序的最常用 4 个基因示例，不按视觉效果挑选。
- genes.json、gene_usage.csv：每个基因的使用频率、样例来源和移除影响。
- dictionary_vectors.csv：原始 256×64 单位范数字典，每行对应一个基因。
- summary.json：探测幅值、色标、选择规则和代码回放误差。

本轮图册支持检查每个基因在当前模型中的作用；不由此自动推出语义独立、对象发现或稳定因果因素。
'''
    (output/'README.md').write_text(report)
    write_json(output/'progress.json',result)
    print(json.dumps(result),flush=True)


def build_gallery(output,entries):
    items=[]
    for entry in entries:
        gene=entry['gene']
        status='active' if entry['active_in_test'] else 'inactive'
        items.append(f'''<a class="gene" data-id="{gene}" data-status="{status}" href="{entry['card']}" target="_blank">
<img src="{entry['thumbnail']}" loading="lazy" alt="基因 {gene:03d}">
<div><strong>基因 {gene:03d}</strong><span>{entry['test_images_active']}/1024 张激活</span></div>
<small>{'左：自然激活；右：标准探测' if entry['active_in_test'] else '左：测试未激活；右：标准探测'}</small></a>''')
    text='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>256 个视觉基因图册</title><style>
body{font:16px system-ui,sans-serif;color:#193143;background:#f3f6f8;margin:0}main{max-width:1400px;margin:auto;padding:28px}h1{margin-bottom:10px}p{line-height:1.7;max-width:1000px}.tools{display:flex;gap:12px;align-items:center;flex-wrap:wrap;background:white;padding:16px;border-radius:12px;position:sticky;top:0;z-index:2}input,select{font:inherit;padding:9px;border:1px solid #bbc9d3;border-radius:6px}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(256px,1fr));gap:14px;margin-top:20px}.gene{display:block;background:white;border:1px solid #dce4e9;border-radius:10px;overflow:hidden;text-decoration:none;color:inherit}.gene img{display:block;width:100%;height:auto}.gene div{display:flex;justify-content:space-between;padding:10px 10px 4px;gap:8px}.gene span,.gene small{color:#526574;font-size:12px}.gene small{display:block;padding:3px 10px 12px}.gene:hover{border-color:#287fa6}.gene[hidden]{display:none}a{color:#136386}.note{padding:12px 16px;border-left:4px solid #75a9be;background:#e9f2f7}
</style><main><h1>256 个视觉基因，逐个查看</h1>
<p>固定模型在 1,024 张测试照片上的结果：140 个有自然激活，116 个在该测试集中未激活。点击任一卡片，查看完整的激活、单独解码和删除效果。</p>
<p class="note">每个缩略图左侧展示自然激活位置，右侧展示统一条件下的单基因解码探测。单个基因是 64 维特征方向，可能影响颜色、纹理或多个区域；独立解码图未必是一个物体，不能把这些图像直接相加还原照片。</p>
<p><a href="all_256_only_gene_rgb_atlas.png">全部基因独立解码总览（实际激活）</a> · <a href="all_256_natural_activation_atlas.png">全部自然激活总览</a> · <a href="all_256_probe_atlas.png">全部单基因探测总览</a> · <a href="zero_code_decoder_baseline.png">零编码解码对照</a> · <a href="README.md">读图说明</a></p>
<div class="tools"><label>基因编号 <input id="search" placeholder="例如 136" inputmode="numeric"></label><label>范围 <select id="status"><option value="all">全部 256 个</option><option value="active">测试中有激活</option><option value="inactive">测试中未激活</option></select></label><span id="count"></span></div><div class="grid">'''+''.join(items)+'''</div></main><script>
const search=document.getElementById('search'),status=document.getElementById('status'),count=document.getElementById('count');
function filter(){let visible=0;document.querySelectorAll('.gene').forEach(el=>{const q=search.value.trim(),id=el.dataset.id;const match=!q||id===String(Number(q))||id.padStart(3,'0').includes(q);el.hidden=!(match&&(status.value==='all'||el.dataset.status===status.value));if(!el.hidden)visible++});count.textContent=`显示 ${visible} 个基因`};search.addEventListener('input',filter);status.addEventListener('change',filter);filter();
</script></html>'''
    (output/'index.html').write_text(text)


def build_natural_only_atlas(output,entries):
    """Assemble already-rendered isolated RGBs; no model inference needed."""
    from PIL import Image, ImageDraw, ImageFont
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,12)
    title=ImageFont.truetype(font_path,20)
    tiles=[]
    for entry in entries:
        if entry['active_in_test']:
            with Image.open(output/f"only_gene_rgb/gene_{entry['gene']:03d}.png") as source:
                tiles.append(source.convert('RGB').copy())
        else:
            blank=Image.new('RGB',(256,256),'#eceff2')
            ImageDraw.Draw(blank).text((24,112),'No test activation',fill='#64748b',font=font)
            tiles.append(blank)
    for first,last,columns,side,name in [(0,256,16,112,'all_256_only_gene_rgb_atlas.png')]+[
        (p*64,(p+1)*64,8,144,f'only_rgb_genes_{p*64:03d}_{(p+1)*64-1:03d}.png') for p in range(4)]:
        height=side+30
        canvas=Image.new('RGB',(columns*side,((last-first+columns-1)//columns)*height+50),'white')
        draw=ImageDraw.Draw(canvas)
        draw.text((8,8),f'Genes {first:03d}-{last-1:03d} | only this gene -> RGB (natural activation)',fill='#172b3a',font=title)
        draw.text((8,31),'Labels: gene ID | active test photos. Each gene uses its strongest photo. Grey = no test activation. RGBs are not additive image layers.',fill='#526071',font=font)
        for gene in range(first,last):
            x=((gene-first)%columns)*side
            y=50+((gene-first)//columns)*height
            canvas.paste(tiles[gene].resize((side,side)),(x,y+28))
            draw.text((x+4,y+5),f"{gene:03d} | {entries[gene]['test_images_active']}",fill='#172b3a',font=font)
        canvas.save(output/name)


def supervise(config_path,output):
    from supervisor import acquire
    config=load_config(config_path)
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'execution.json',{'status':'waiting_for_gpu','max_gpus':1,'pid':os.getpid(),'updated_utc':now()})
    while True:
        leases=acquire(config,1)
        if leases:
            break
        time.sleep(config['compute']['poll_seconds'])
    gpu,handle=leases[0]
    try:
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],PYTHONDONTWRITEBYTECODE='1',TMPDIR=config['temporary_directory'],
                   OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
        with (output/'worker.log').open('w') as log:
            proc=subprocess.Popen([config['python'],str(Path(__file__).resolve()),'--config',str(config_path),
                                   '--output',str(output),'--worker'],env=env,stdout=log,stderr=subprocess.STDOUT)
            write_json(output/'execution.json',{'status':'rendering','worker_pid':proc.pid,'physical_gpu':gpu,'updated_utc':now()})
            code=proc.wait()
        write_json(output/'execution.json',{'status':'complete' if code==0 else 'failed','returncode':code,'physical_gpu':gpu,'updated_utc':now()})
        if code:
            raise SystemExit(code)
    finally:
        handle.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    if args.worker:
        render(load_config(args.config),args.output)
    else:
        supervise(args.config,args.output)
