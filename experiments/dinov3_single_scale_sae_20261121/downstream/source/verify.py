"""GPU end-to-end check: arithmetic file -> exact coefficients -> dictionary -> RGB."""
import json
from pathlib import Path
import numpy as np
import torch
from common import config,write_json,now,fingerprint
from downstream_model import RGBDecoder,load_codec
from model import tensor_hash


@torch.no_grad()
def main(c):
    if not torch.cuda.is_available():raise RuntimeError('GPU required')
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic=True
    root=Path(c['root'])
    saved=torch.load(root/'runs/decoder/checkpoint_final.pt',map_location='cpu',weights_only=False)
    if saved['fingerprint']!=fingerprint(c) or saved['step']!=c['decoder']['steps']:
        raise ValueError('Require matching final RGB decoder')
    decoder=RGBDecoder().cuda().eval().requires_grad_(False)
    decoder.load_state_dict(saved['model'],strict=True)
    results=[];rgb={}
    for m in c['dictionary_sizes']:
        model,provenance=load_codec(c,m)
        if saved['codec_hashes'][f'atoms_{m}']!=provenance['model_hash']:
            raise ValueError('Decoder and arithmetic stream use different SAE dictionaries')
        local=Path(c['local'])/'exports'/f'atoms_{m}'
        original=np.load(local/'validation.codes.npy',mmap_mode='r')
        images=[]
        for i in c['arithmetic_validation_indices']:
            recovered=np.load(local/f'decoded_codes_{i:04d}.npy')
            if not np.array_equal(original[i],recovered):raise AssertionError('Stored coefficient mismatch')
            a=torch.from_numpy(np.array(original[i:i+1],copy=True)).cuda().float()*model.step
            b=torch.from_numpy(recovered[None].copy()).cuda().float()*model.step
            with torch.autocast('cuda',dtype=torch.bfloat16):
                direct=model.dictionary(a);restored=model.dictionary(b)
                x=decoder(direct).float();y=decoder(restored).float()
            if not torch.equal(x,y):raise AssertionError('Byte stream and direct-code RGB decoding differ')
            results.append(dict(atoms=m,validation_index=i,coefficient_roundtrip_exact=True,
                feature_max_absolute_difference=float((direct-restored).abs().max()),
                rgb_max_absolute_difference=float((x-y).abs().max())))
            images.append(y.cpu().numpy().transpose(0,2,3,1)[0])
        rgb[f'atoms_{m}']=np.stack(images)
        del model
    np.savez(root/'reports/arithmetic_rgb.npz',indices=np.array(c['arithmetic_validation_indices']),**rgb)
    write_json(root/'reports/stream_decode_verification.json',dict(status='complete',results=results,
        decoder_hash=tensor_hash(decoder),note='Exact reconstruction of the exported integer codes, not lossless reconstruction of the original photo.',updated_utc=now()))


if __name__=='__main__':main(config())
