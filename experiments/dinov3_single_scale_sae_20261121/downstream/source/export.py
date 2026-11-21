"""GPU-only final SAE export for retrieval, batch audit and exact coding."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch

from common import config,now,write_json,atomic_npy,check_upstream_config
from downstream_model import load_codec,embedding
from features import FeatureStore
from utils import GridSampler


def amp():
    return torch.autocast('cuda',dtype=torch.bfloat16)


def tensor(store,ids):
    return torch.from_numpy(store[np.asarray(ids)]).cuda()


@torch.no_grad()
def main(c,atoms):
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    torch.set_num_threads(4)
    check_upstream_config(c)
    root=Path(c['root']); local=Path(c['local'])/'exports'/f'atoms_{atoms}'
    local.mkdir(parents=True,exist_ok=True)
    durable=root/'exports'/f'atoms_{atoms}';durable.mkdir(exist_ok=True)
    model,provenance=load_codec(c,atoms)
    store=FeatureStore(c['base'],'validation')
    train=FeatureStore(c['base'],'train')
    n=len(store);batch=c['export_batch']
    qout=np.lib.format.open_memmap(local/'validation.codes.npy',mode='w+',dtype=np.uint8,shape=(n,atoms,32,32))
    means=np.zeros((n,atoms),dtype=np.float32)
    sums=np.zeros(atoms,dtype=np.float64);nonzero=np.zeros(atoms,dtype=np.int64);presence=np.zeros(atoms,dtype=np.int64)
    vectors=[];raw_vectors=[];ratios=[]
    ids=np.linspace(0,n-1,c['validation_images'],dtype=int);ids_set=set(ids.tolist())
    selected_reports=[]
    start=time.monotonic()
    for first in range(0,n,batch):
        if (root/'STOP_REQUESTED').exists():raise InterruptedError('Stop requested')
        chosen=np.arange(first,min(first+batch,n))
        f=tensor(store,chosen)
        with amp():restored,q,units=model(f)
        values=q.cpu().numpy().astype(np.uint8);qout[first:first+len(values)]=values
        active=values>0
        means[first:first+len(values)]=values.mean((2,3))*model.step
        sums+=values.sum((0,2,3),dtype=np.float64)*model.step
        nonzero+=active.sum((0,2,3));presence+=active.reshape(len(values),atoms,-1).any(2).sum(0)
        vectors.append(embedding(q*model.step).cpu())
        if atoms==min(c['dictionary_sizes']):raw_vectors.append(embedding(f).cpu())
        mask=np.array([i in ids_set for i in chosen])
        if mask.any():
            select=torch.from_numpy(np.flatnonzero(mask)).cuda()
            _,_,report=model.objective(f[select],(restored[select],q[select],units[select]),c['base']['lambda_bpp'])
            selected_reports.append((int(mask.sum()),report))
            ratios.extend(torch.nn.functional.cosine_similarity(restored[select].float(),f[select],dim=1).mean((1,2)).cpu().tolist())
        if (first//batch)%16==0 or first+batch>=n:
            qout.flush()
            write_json(durable/'progress.json',dict(status='exporting_validation',images=first+len(values),
                total=n,sae=provenance,elapsed_seconds=time.monotonic()-start,updated_utc=now()))
    qout.flush();del qout
    vectors=torch.cat(vectors)
    atomic_npy(local/'embeddings.npy',vectors.numpy().astype(np.float16))
    scores=(vectors.cuda()@vectors.cuda().T).float().cpu().numpy()
    atomic_npy(durable/'cosine_scores.npy',scores)
    if raw_vectors:
        reference=root/'exports/dino_reference';reference.mkdir(exist_ok=True)
        raw=torch.cat(raw_vectors)
        atomic_npy(reference/'embeddings.npy',raw.numpy().astype(np.float16))
        atomic_npy(reference/'cosine_scores.npy',(raw.cuda()@raw.cuda().T).float().cpu().numpy())
        write_json(reference/'READY.json',dict(images=n,dimension=raw.shape[1],encoder_signature=provenance['encoder_signature'],
            normalization='Same train-centered/scalar-normalized DINO as the SAE input',updated_utc=now()))
    atomic_npy(durable/'per_image_mean_weights.npy',means)
    np.savez(durable/'atom_statistics.npz',mean_weight=sums/(n*32*32),coverage=nonzero/(n*32*32),image_presence=presence/n)
    metric={k:sum(count*r[k] for count,r in selected_reports)/len(ids) for k in selected_reports[0][1]}
    metric.update(images=len(ids),validation_centered_feature_cosine=float(np.mean(ratios)),
        atoms_unseen_on_1000=int((presence==0).sum()),sae=provenance)
    write_json(durable/'feature_evaluation.json',metric)
    probabilities=model.entropy.probabilities().cpu().numpy()
    atomic_npy(durable/'probabilities.npy',probabilities)
    # Re-encode the exact training batches with the SAME FINAL SAE, not historical models.
    sampler=GridSampler(c['base']['manifests']['train'],c['base']['seed'])
    audits=[];mean_maps=[]
    for step in c['batch_audit_steps']:
        chosen=sampler.indices((step-1)*c['batch_audit_size'],c['batch_audit_size'])
        maps=torch.zeros(atoms,32,32,device='cuda')
        totals=np.zeros(atoms,dtype=np.int64)
        appearances=np.zeros(atoms,dtype=np.int64)
        for first in range(0,len(chosen),batch):
            f=tensor(train,chosen[first:first+batch])
            with amp():q=model(f)[1]
            maps+=q.float().sum(0)*model.step
            q=q.cpu().numpy().astype(np.uint8)
            totals+=(q>0).sum((0,2,3))
            appearances+=(q>0).reshape(len(q),atoms,-1).any(2).sum(0)
        maps/=len(chosen)
        mean_maps.append(maps.cpu().numpy().astype(np.float16))
        audits.append(dict(training_step=step,indices=chosen.tolist(),
            active_atoms_per_position=float(totals.sum()/(len(chosen)*32*32)),
            distinct_atoms_in_batch=int((appearances>0).sum()),image_presence=appearances.tolist()))
    np.savez(durable/'batch_audit.npz',mean_maps=np.stack(mean_maps),steps=np.array(c['batch_audit_steps']))
    write_json(durable/'batch_audit.json',dict(batches=audits,
        interpretation='Same frozen final SAE on actual image batches of training steps 1/400/800/1200; not training history.'))
    # Dictionary repetition in the full 1024 x 3 x 3 atom space.
    d=model.dictionary.atoms().flatten(1)
    cosine=d@d.T;cosine.fill_diagonal_(-2)
    nearest=cosine.max(1).values.cpu().numpy()
    atomic_npy(durable/'nearest_atom_cosine.npy',nearest)
    write_json(durable/'READY.json',dict(status='complete',atoms=atoms,sae=provenance,images=n,
        embeddings_dimension=atoms*21,code_cache=str(local/'validation.codes.npy'),
        local_embeddings=str(local/'embeddings.npy'),quantization_step=model.step,
        native_activation_shape=[atoms,32,32],nearest_positive_cosine_mean=float(nearest.mean()),
        nearest_positive_cosine_over_095=int((nearest>.95).sum()),updated_utc=now()))
    write_json(durable/'progress.json',dict(status='complete',images=n,updated_utc=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--atoms',type=int,required=True)
    main(config(),p.parse_args().atoms)
