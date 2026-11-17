"""Leased-GPU predictions, real RGB errors and explicit residual diagnostics."""
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F
from common import write_json
from data import image_tensor
from model import pyramid_targets


def to_pil(value):
    array=value.detach().float().clamp(0,1).cpu().permute(1,2,0).numpy()
    return Image.fromarray(np.rint(array*255).astype(np.uint8))


def sheet(rows,destination,side):
    canvas=Image.new('RGB',(side*len(rows[0]),(side+24)*len(rows)),'white')
    draw=ImageDraw.Draw(canvas)
    for i,tiles in enumerate(rows):
        for j,(label,value) in enumerate(tiles):
            draw.text((j*side+3,i*(side+24)+3),label,fill='black')
            tile=to_pil(value).resize((side,side),Image.Resampling.BILINEAR)
            canvas.paste(tile,(j*side,i*(side+24)+24))
    canvas.save(destination)


@torch.inference_mode()
def preview(model,pixels,config,output,label):
    prior=model.training
    model.eval()
    number=min(config['training']['preview_count'],len(pixels))
    # Spread across the manifest, avoiding a single panorama's four views.
    ids=np.linspace(0,len(pixels)-1,number,dtype=int)
    target=image_tensor(pixels,ids)
    enabled=config['training']['precision']=='bfloat16_if_supported' and torch.cuda.is_bf16_supported()
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled):
        result=model(target)
    directory=output/'previews'
    directory.mkdir(exist_ok=True)
    side=target.shape[-1]
    sheet([[('Original real street photo',target[i]),('Reconstruction',result['reconstruction'][i]),
            ('Absolute RGB error x3',(target[i]-result['reconstruction'][i]).abs()*3)]
           for i in range(number)],directory/f'{label}.png',side)
    targets=pyramid_targets(target)
    rows=[]
    for i in range(min(number,2)):
        row=[('Original',target[i]),('Full reconstruction',result['reconstruction'][i])]
        for scale in model.scales:
            predicted=result['residual_predictions'][scale][i]
            truth=targets[scale][i]
            if scale!='coarse':
                predicted=.5+predicted*2
                truth=.5+truth*2
            row.extend([(f'{scale} target',truth),(f'{scale} aux prediction',predicted)])
        rows.append(row)
    sheet(rows,directory/f'{label}.pyramid.png',side)
    serialized={'image_indices':ids[:2], 'note':np.array('Additive spatial feature codes. RGB decoding is nonlinear.')}
    stats={}
    edits=[]
    for scale in model.scales:
        activation=result['activations'][scale]
        serialized[f'{scale}_activations']=activation[:2].float().cpu().numpy()
        if model.sparse: serialized[f'{scale}_atoms']=model.dictionaries[scale].atoms.float().cpu().numpy()
        stats[scale]={'nonzero_fraction':float((activation>0).float().mean()),
                      'genes_used':int((activation>0).any((0,2,3)).sum()),'gate':model.gates[scale]}
    for i in range(min(number,2)):
        row=[('Original',target[i]),('Full reconstruction',result['reconstruction'][i])]
        for scale in model.scales:
            features={s:v[i:i+1].clone() for s,v in result['decoder_features'].items()}
            features[scale].zero_()
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled):
                deleted=model.decoder(features)[0]
            row.append((f'Without {scale}',deleted))
        edits.append(row)
    sheet(edits,directory/f'{label}.scale_deletion.png',side)
    np.savez_compressed(directory/f'{label}.codes.npz',**serialized)
    write_json(directory/f'{label}.json',{'step':model.step,'rate_index':model.rate_index,'indices':ids.tolist(),'scales':stats,
        'error_definition':'abs(prediction_float - RGB target) x3; display clipped only',
        'residual_display':'coarse RGB; medium/fine display 0.5 + 2*residual, clipped',
        'auxiliary_heads_are_not_final_rgb_decoder':True})
    model.train(prior)
