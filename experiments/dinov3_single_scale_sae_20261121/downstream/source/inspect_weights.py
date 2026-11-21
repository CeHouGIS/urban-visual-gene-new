"""Print and plot exact stored SAE weights. CPU data readout only."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from common import config,rows,write_json,now


def main(c):
    index=666;row=16;col=16
    root=Path(c['root']);out=root/'reports/exact_weights';out.mkdir(exist_ok=True)
    pixels=np.load(Path(c['rgb_cache'])/'validation.rgb.npy',mmap_mode='r')
    metadata=rows(c,'validation');delta=c['base']['compression']['quantization_step']
    summaries={};arrays={}
    for m in c['dictionary_sizes']:
        q=np.load(Path(c['local'])/'exports'/f'atoms_{m}'/'validation.codes.npy',mmap_mode='r')
        a=np.array(q[index],dtype=np.float32)*delta;arrays[f'A_{m}']=a
        point=a[:,row,col];order=np.lexsort((np.arange(m),-point));top=order[:12]
        gene=int(top[0]);heat=a[gene]
        with (out/f'point_weights_{m}.csv').open('w',newline='') as stream:
            writer=csv.writer(stream);writer.writerow(['atom_id','weight','integer_symbol','nonzero'])
            writer.writerows((g,float(point[g]),int(q[index,g,row,col]),int(point[g]>0)) for g in range(m))
        np.savetxt(out/f'atom_{m}_{gene}_matrix.csv',heat,delimiter=',',fmt='%.3f')
        summaries[f'atoms_{m}']=dict(shape=list(a.shape),grid_row=row,grid_col=col,
            nonzero_atoms_at_point=int((point>0).sum()),zeros_at_point=int((point==0).sum()),
            top_atoms_at_point=[dict(atom_id=int(g),weight=float(point[g])) for g in top],
            inspected_atom=gene,matrix_center_4x4=heat[14:18,14:18].tolist(),
            grid_indices_are_zero_based=True)
        fig,axes=plt.subplots(2,2,figsize=(13,9),constrained_layout=True)
        axes[0,0].imshow(pixels[index]);axes[0,0].add_patch(Rectangle((col*8-.5,row*8-.5),8,8,
            fill=False,color='red',linewidth=2))
        axes[0,0].set_title(f'Original #{index} | {metadata[index]["city"]}\nRed = grid (row {row}, col {col}); not an object mask',fontsize=11)
        axes[0,0].set_xticks([]);axes[0,0].set_yticks([])
        active=np.flatnonzero(point)
        axes[0,1].vlines(active,0,point[active],linewidth=.6)
        axes[0,1].set(xlim=(-.5,m-.5),ylim=(0,max(2.8,float(point.max())*1.15)),
            xlabel='Atom ID',ylabel='Weight A_g(16,16)',
            title=f'Complete {m}-dimensional vector at this position\n{int((point>0).sum())} nonzero; {int((point==0).sum())} zero')
        axes[0,1].grid(axis='y',alpha=.2)
        y=np.arange(len(top));axes[1,0].barh(y,point[top]);axes[1,0].invert_yaxis()
        axes[1,0].set_yticks(y,labels=[f'Atom {g}' for g in top])
        axes[1,0].set(xlim=(0,max(2.8,float(point.max())*1.25)),xlabel='Actual activation weight',
            title='12 largest weights at the marked position')
        for j,g in enumerate(top):axes[1,0].text(float(point[g])+.03,j,f'{point[g]:.3f}',va='center',fontsize=10)
        im=axes[1,1].imshow(heat,cmap='inferno',vmin=0,vmax=max(3.,float(heat.max())),interpolation='nearest')
        axes[1,1].add_patch(Rectangle((col-.5,row-.5),1,1,fill=False,color='cyan',linewidth=1.5))
        axes[1,1].set(xlabel='Feature grid column',ylabel='Feature grid row',
            title=f'Atom {gene}: complete 32 x 32 weight matrix\nA({row},{col}) = {point[gene]:.3f}')
        fig.colorbar(im,ax=axes[1,1],label='Activation weight')
        fig.suptitle(f'{m}-atom SAE | actual saved coefficients A = 0.125 Q\nNo re-training or inference; all indices zero-based',fontsize=13)
        fig.savefig(out/f'weights_{m}_image_{index:04d}.png',dpi=140);plt.close(fig)
    np.savez_compressed(out/f'weights_image_{index:04d}.npz',image_index=index,
        quantization_step=delta,**arrays)
    write_json(out/'weights_summary.json',dict(validation_index=index,models=summaries,updated_utc=now()))
    print(json.dumps(summaries,ensure_ascii=False),flush=True)


if __name__=='__main__':main(config())
