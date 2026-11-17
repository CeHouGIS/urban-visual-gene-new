"""GPU deletion audit for all active genes on the fixed 128-city diagnostic."""
import json
from pathlib import Path
import numpy as np
import torch
from common import write_json, now
from data import image_tensor
from reuse_metrics import summarize_reuse


@torch.inference_mode()
def audit_reuse(model, pixels, config, destination, stopped):
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError('Functional reuse must freeze every model parameter')
    settings=config['protocol']['reuse_evaluation']
    model.eval();model.rate_index=settings['primary_rate_index']
    samples=json.loads((Path(config['root'])/'data/confirmation_selection.json').read_text())['samples']
    if len(samples)!=len(pixels) or len({r['ucdb_id'] for r in samples})!=len(samples):
        raise AssertionError('Functional reuse requires one distinct city per diagnostic image')
    records=[];presence=np.zeros(model.settings['dictionary_size'],dtype=np.int64)
    genes=[f'{s}:{g}' for s in model.scales for g in range(model.gene_counts[s])]
    for index,sample in enumerate(samples):
        if stopped():return False
        x=image_tensor(pixels,np.array([index]));result=model(x)
        base=result['reconstruction'];baseline=(base-x).abs().mean()
        codes=result['activations']
        targets=[(s,int(g)) for s,a in codes.items() for g in (a[0]!=0).any((1,2)).nonzero().flatten().tolist()]
        offset=0
        for s in model.scales:
            presence[offset:offset+model.gene_counts[s]]+=(codes[s][0]!=0).any((1,2)).cpu().numpy()
            offset+=model.gene_counts[s]
        for begin in range(0,len(targets),settings['deletion_batch_size']):
            if stopped():return False
            chunk=targets[begin:begin+settings['deletion_batch_size']];n=len(chunk)
            edited={s:a.expand(n,-1,-1,-1).clone() for s,a in codes.items()}
            for i,(s,g) in enumerate(chunk):edited[s][i,g]=0
            y=model.decoder(model.synthesize_codes(edited))
            change=(y-base).abs().mean((1,2,3));increase=(y-x).abs().mean((1,2,3))-baseline
            for i,(s,g) in enumerate(chunk):
                records.append({'image_index':index,'city_id':str(sample['ucdb_id']),
                    'city':sample['city'],'country':sample['country'],'continent':sample['continent'],
                    'gene_key':f'{s}:{g}','baseline_mae':float(baseline),
                    'mae_increase':float(increase[i]),'rgb_change_mae':float(change[i])})
        if index%16==0 or index+1==len(samples):
            write_json(destination/'functional_reuse_progress.json',{'images_complete':index+1,
                'images_total':len(samples),'deletions':len(records),'updated_utc':now()})
    summary=summarize_reuse(records,genes,len(samples),settings)
    if summary['active_gene_calls']!=int(presence.sum()):raise AssertionError('Missing active-gene deletion')
    # The same stochastic-free inference can have tiny FP32/bf16 support
    # differences; keep this audit's exact presence counts with its deletions.
    summary.update(primary_gene_cap=config['protocol']['primary_gene_cap'],
                   evaluation_precision='FP32, TF32 disabled',checkpoint_step=model.step,
                   per_gene_image_count=presence.tolist(),all_weights_frozen=True)
    write_json(destination/'functional_reuse_deletions.json',records)
    write_json(destination/'functional_reuse_summary.json',summary)
    return True
