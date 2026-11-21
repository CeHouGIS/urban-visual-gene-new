"""CPU-only symbolic coding and durable archives; no neural inference."""
import argparse
from pathlib import Path
import time
import numpy as np
from common import config,write_json,now,mirror_file
from codec import make_cdf,save_tables,write_image,read_image


def main(c,atoms):
    root=Path(c['root']);directory=root/'exports'/f'atoms_{atoms}'
    local=Path(c['local'])/'exports'/f'atoms_{atoms}'
    out=directory/'codec';out.mkdir(exist_ok=True)
    q=np.load(local/'validation.codes.npy',mmap_mode='r')
    cdf=make_cdf(np.load(directory/'probabilities.npy'))
    save_tables(out/'probability_tables.npz',{'single':cdf})
    results=[];start=time.monotonic()
    for index in c['arithmetic_validation_indices']:
        path=out/f'validation_{index:04d}.uvg'
        metric=write_image(path,{'single':q[index]},{'single':cdf})
        recovered=read_image(path,{'single':cdf})['single']
        if not np.array_equal(recovered,q[index]):raise AssertionError('Real arithmetic-file roundtrip mismatch')
        np.save(local/f'decoded_codes_{index:04d}.npy',recovered)
        results.append(dict(validation_index=index,file_kib=path.stat().st_size/1024,
            bpp_per_512_input_pixel=path.stat().st_size*8/512**2,
            bpp_per_256_rgb_output_pixel=path.stat().st_size*8/256**2,**metric))
        write_json(out/'progress.json',dict(status='coding',completed=len(results),total=3,
            elapsed_seconds=time.monotonic()-start,updated_utc=now()))
    write_json(out/'result.json',dict(images=results,mean_file_kib=float(np.mean([r['file_kib'] for r in results])),
        exact_quantized_roundtrip=True,shared_entropy_tables_bytes=(out/'probability_tables.npz').stat().st_size,
        limitation='Files exclude shared DINO/SAE/dictionary/RGB decoder and entropy-table costs; output is a feature code, not a standalone photograph.',updated_utc=now()))
    # Raw validation activations are a local mmap cache. The durable compressed
    # archive preserves every coefficient for later atom inspection.
    archive=local/'validation_codes.npz'
    if not archive.exists():
        partial=archive.with_name(archive.name+'.partial')
        with partial.open('wb') as stream:np.savez_compressed(stream,codes=q,quantization_step=c['base']['compression']['quantization_step'])
        partial.replace(archive)
    mirror_file(archive,directory/'validation_codes.npz')
    mirror_file(local/'embeddings.npy',directory/'embeddings.npy')
    write_json(out/'READY.json',dict(status='complete',atoms=atoms,full_validation_codes_preserved=True,
        code_archive_bytes=(directory/'validation_codes.npz').stat().st_size,updated_utc=now()))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--atoms',type=int,required=True)
    main(config(),p.parse_args().atoms)
