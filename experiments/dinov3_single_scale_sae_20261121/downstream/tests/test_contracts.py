import numpy as np
import pytest
from common import phase_indices,rank_scores
from codec import make_cdf,write_image,read_image


def test_retrieval_excludes_panorama_and_grid_and_deduplicates():
    rows=[dict(pano=p,grid_id=g,city=str(i)) for i,(p,g) in enumerate([
        ('q','g0'),('q','g1'),('p2','g0'),('p3','g2'),('p4','g2'),('p5','g3'),('p6','g4')])]
    hits=rank_scores(np.arange(7,0,-1),rows,0)
    assert [h['index'] for h in hits]==[3,5,6]


def test_shared_decoder_rotates_sources_on_identical_images():
    class Sampler:
        def indices(self,offset,count):return np.arange(offset,offset+count)
    sources=['dino_reference','atoms_2048','atoms_4096']
    a=[phase_indices(Sampler(),i,32,sources) for i in range(6)]
    assert [v[0] for v in a]==sources*2
    assert all(np.array_equal(a[i][1],a[0][1]) for i in [1,2])
    assert all(np.array_equal(a[i][1],np.arange(32,64)) for i in [3,4,5])


@pytest.mark.parametrize('atoms',[2048,4096])
def test_real_file_wide_dictionary_exact_roundtrip(tmp_path,atoms):
    rng=np.random.default_rng(20261121)
    probabilities=np.full((atoms,64),1e-5)
    probabilities[:,0]=.8;probabilities[:,1]=.19
    cdf=make_cdf(probabilities)
    assert (np.diff(cdf)>0).all() and (cdf[:,-1]==65536).all()
    q=rng.integers(0,64,(atoms,2,2),dtype=np.uint8)
    path=tmp_path/'image.uvg'
    metric=write_image(path,{'single':q},{'single':cdf})
    assert np.array_equal(read_image(path,{'single':cdf})['single'],q)
    assert metric['file_bits']==path.stat().st_size*8
