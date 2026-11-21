"""Render saved downstream results on CPU. No neural models are executed."""
import csv
import fcntl
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import config,now,write_json,rows,rank_scores


def hide(axes):
    for ax in np.asarray(axes).flat:ax.set_xticks([]);ax.set_yticks([])


def retrieval(c,pixels,metadata,out):
    exports=Path(c['root'])/'exports'
    if not all((exports/name/'cosine_scores.npy').exists() for name in c['sources']):return
    hits={}
    for source in c['sources']:
        scores=np.load(exports/source/'cosine_scores.npy')
        hits[source]={q:rank_scores(scores[q],metadata,q) for q in c['queries']}
    for q in c['queries']:
        fig,axes=plt.subplots(3,4,figsize=(13,10),constrained_layout=True)
        for row,source in enumerate(c['sources']):
            axes[row,0].imshow(pixels[q]);axes[row,0].set_title(f'Query {q} | {metadata[q]["city"]}\n{source}',fontsize=10)
            for col,hit in enumerate(hits[source][q],1):
                axes[row,col].imshow(pixels[hit['index']]);axes[row,col].set_title(
                    f'Top {col} | {hit["city"]}\ncosine={hit["cosine"]:.3f}',fontsize=10)
        hide(axes)
        fig.suptitle('Held-out street retrieval | 1000-image gallery\nSame panorama/grid excluded; distinct returned grids; cosine is not accuracy')
        fig.savefig(out/f'retrieval_query_{q:04d}.png',dpi=130);plt.close(fig)
    write_json(out/'retrieval.json',dict(gallery_images=len(metadata),queries=c['queries'],results=hits,
        embedding='1/2/4 spatial average-pooling readout from one feature grid; concatenate and L2 normalize',
        dimensions=dict(dino_reference=1024*21,atoms_2048=2048*21,atoms_4096=4096*21),
        limitation='No human relevance labels. Absolute cosine values are not comparable accuracy scores across representations.',updated_utc=now()))


def atoms(c,m,pixels,metadata,out):
    directory=Path(c['root'])/'exports'/f'atoms_{m}'
    if not (directory/'READY.json').exists():return
    q=np.load(Path(c['local'])/'exports'/f'atoms_{m}'/'validation.codes.npy',mmap_mode='r')
    stats=np.load(directory/'atom_statistics.npz');means=np.load(directory/'per_image_mean_weights.npy')
    step=c['base']['compression']['quantization_step'];weights=stats['mean_weight'];coverage=stats['coverage'];presence=stats['image_presence']
    with (out/f'all_atom_statistics_{m}.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['atom_id','mean_weight','nonzero_fraction','image_presence_fraction'])
        writer.writerows(zip(range(m),weights,coverage,presence))
    top=np.argsort(-weights)[:4]
    fig,axes=plt.subplots(4,3,figsize=(12,15),constrained_layout=True)
    instances={}
    for row,g in enumerate(top):
        selected=[];grids=set()
        for i in np.argsort(-means[:,g],kind='stable'):
            i=int(i)
            if metadata[i]['grid_id'] in grids:continue
            grids.add(metadata[i]['grid_id']);selected.append(i)
            if len(selected)==3:break
        instances[int(g)]=selected
        maximum=max(float(q[selected,g].max())*step,.125)
        for col,i in enumerate(selected):
            ax=axes[row,col];ax.imshow(pixels[i])
            heat=np.asarray(q[i,g],dtype=np.float32)*step
            ax.imshow(heat,cmap='inferno',vmin=0,vmax=maximum,alpha=.65,extent=(-.5,255.5,255.5,-.5),interpolation='nearest')
            ax.set_title(f'Atom {g} | {metadata[i]["city"]} | image {i}\nmean={heat.mean():.3f}, active={100*(heat>0).mean():.1f}%',fontsize=10)
    hide(axes);fig.suptitle(f'Final {m}-atom SAE | same atom across three distinct grid cells\nFour largest mean-weight atoms; fixed color range per atom; no semantic names inferred')
    fig.savefig(out/f'reusable_atoms_{m}.png',dpi=130);plt.close(fig)
    write_json(out/f'reusable_atoms_{m}.json',dict(atom_to_validation_indices=instances,selection='largest global mean weights; top response images from distinct grids'))
    audit=np.load(directory/'batch_audit.npz');maps=audit['mean_maps'].astype(np.float32)
    top8=np.argsort(-weights)[:8]
    fig,axes=plt.subplots(len(audit['steps']),8,figsize=(17,8),constrained_layout=True)
    for col,g in enumerate(top8):
        maximum=max(float(maps[:,g].max()),.125)
        for row,s in enumerate(audit['steps']):
            axes[row,col].imshow(maps[row,g],cmap='magma',vmin=0,vmax=maximum,interpolation='nearest')
            axes[row,col].set_title(f'Atom {g} | batch {s}',fontsize=9)
    hide(axes);fig.suptitle(f'{m} atoms | same FINAL SAE on four actual 128-photo training batches\nMean activation maps; this is not a history of different training checkpoints')
    fig.savefig(out/f'batch_activation_maps_{m}.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(14,6),sharex=True,constrained_layout=True)
    axes[0].bar(np.arange(m),weights,width=1);axes[0].set_ylabel('Mean weight')
    axes[1].bar(np.arange(m),coverage,width=1);axes[1].set(xlabel='Atom ID (local to this model)',ylabel='Active position fraction')
    fig.suptitle(f'All {m} atoms | 1000 held-out photos')
    fig.savefig(out/f'all_atom_weights_{m}.png',dpi=130);plt.close(fig)


def reconstruction(c,out):
    directory=Path(c['root'])/'runs/decoder'
    if not (directory/'query_rgb.npz').exists():return None
    data=np.load(directory/'query_rgb.npz');original=data['original'];ids=data['indices'];step=int(data['step'])
    fig,axes=plt.subplots(len(ids),4,figsize=(13,3.4*len(ids)),constrained_layout=True)
    for row,index in enumerate(ids):
        axes[row,0].imshow(original[row]);axes[row,0].set_title(f'Original | validation {index}')
        for col,source in enumerate(c['sources'],1):
            rgb=data[source][row].clip(0,1)
            mae=np.abs(rgb-original[row].astype(np.float32)/255).mean()
            axes[row,col].imshow(rgb);axes[row,col].set_title(f'{source}\nMAE={mae:.4f}',fontsize=10)
    hide(axes);fig.suptitle(f'New shared 1024-channel RGB decoder | step {step}/{c["decoder"]["steps"]}\nFrozen DINO/SAEs; no RGB skip; output 256 x 256')
    fig.savefig(out/'decode_overview.png',dpi=130);plt.close(fig)
    for row,index in enumerate(ids):
        fig,axes=plt.subplots(2,4,figsize=(13,6.8),constrained_layout=True)
        axes[0,0].imshow(original[row]);axes[0,0].set_title(f'Original #{index}');axes[1,0].axis('off')
        for col,source in enumerate(c['sources'],1):
            rgb=data[source][row].clip(0,1);error=np.abs(rgb-original[row].astype(np.float32)/255).mean(-1)
            axes[0,col].imshow(rgb);axes[0,col].set_title(f'{source}\nMAE={error.mean():.4f}',fontsize=10)
            im=axes[1,col].imshow(error,cmap='magma',vmin=0,vmax=.5);axes[1,col].set_title('Absolute RGB error')
        hide(axes);fig.colorbar(im,ax=list(axes[1,1:]),location='bottom',shrink=.8,label='Mean absolute error across RGB channels')
        fig.suptitle(f'Shared decoder step {step}/{c["decoder"]["steps"]}')
        fig.savefig(out/f'error_query_{index:04d}.png',dpi=130);plt.close(fig)
    return step


def curves(c,out):
    file=Path(c['local'])/'runs/decoder/train.jsonl'
    if not file.exists():return
    values=[]
    for line in file.read_text().splitlines():
        try:values.append(json.loads(line))
        except json.JSONDecodeError:pass
    fig,ax=plt.subplots(figsize=(9,4),constrained_layout=True)
    for source in c['sources']:
        rows=[r for r in values if r['source']==source]
        if rows:ax.plot([r['step'] for r in rows],[r['mae'] for r in rows],label=source,alpha=.6)
    ax.set(xlabel='RGB decoder update',ylabel='Training RGB MAE');ax.legend();ax.grid(alpha=.2)
    fig.savefig(out/'decoder_training_curves.png',dpi=130);plt.close(fig)


def main(c):
    root=Path(c['root']);out=root/'reports';lock=(out/'render.lock').open('a+')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:return
    pixels=np.load(Path(c['rgb_cache'])/'validation.rgb.npy',mmap_mode='r');metadata=rows(c,'validation')
    retrieval(c,pixels,metadata,out)
    for m in c['dictionary_sizes']:atoms(c,m,pixels,metadata,out)
    step=reconstruction(c,out);curves(c,out)
    result=root/'runs/decoder/result.json'
    data=json.loads(result.read_text()) if result.exists() else None
    lines=['# DINOv3 + SAE：原框架后续实验','',
        '冻结 DINOv3 ViT-L/16 和最终 2048/4096 SAE，接续字典合成、共享 RGB 解码、检索、批次激活分析及真实算术编码。',
        '',f'RGB 解码器当前展示步数：{step if step is not None else "等待训练"} / {c["decoder"]["steps"]}。',
        '输入为完整的 1024×32×32 特征，输出与旧框架相同为 256×256。DINO 输入为 512×512。',
        '三种源按同一组图片轮换训练一个全新 RGB 解码器，仅用 RGB MAE；不反向更新 SAE/DINO，没有原图跳跃连接。','']
    if data:
        lines+=['| 解码源 | RGB MAE | 平均逐图 PSNR (dB) |','|---|---:|---:|']
        for name,value in data['results'].items():lines.append(f'| {name} | {value["mae"]:.5f} | {value["mean_image_psnr_db"]:.2f} |')
    lines+=['','检索直接读取原始 DINO 特征或量化 SAE 激活，单个特征网格做 1/2/4 空间池化拼接，排除同一全景/格网并使返回格网互异。池化是检索读出，不新增学习分支。',
        '没有人工语义相关性标签，余弦分数不能当作准确率，也不能跨表示直接判断优劣。RGB 质量同时受解码器训练预算影响。','',
        '所有 atom 均输出激活权重/覆盖率/图片出现率；批次分析用同一个最终 SAE 编码原训练批次，不是历史模型的训练轨迹。',
        '实际码流检验 3 张图整数精确回读；每图文件大小不含共享模型和熵表。bpp 同时注明 512 输入像素或 256 输出像素的分母。','',
        '[原图及重建](decode_overview.png) · [误差图](error_query_0666.png) · [检索](retrieval_query_0666.png)',
        '[2048 复用实例](reusable_atoms_2048.png) · [4096 复用实例](reusable_atoms_4096.png)']
    (out/'RESULT_zh.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'render_status.json',dict(status='complete',decoder_step=step,updated_utc=now()))


if __name__=='__main__':main(config())
