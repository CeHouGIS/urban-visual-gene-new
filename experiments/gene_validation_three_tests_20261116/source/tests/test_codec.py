import numpy as np
import pytest
from codec import pack,unpack

def test_equal_wire_sizes_and_roundtrips():
 for side in [16,32,64]:
  for width,per in [(3,1),(12,4),(48,16)]:
   d=np.random.default_rng(1).normal(size=(width,side,side)).astype('f4')
   sparse=np.zeros((128,side,side),dtype='f4');sparse.ravel()[:side*side*per]=.12345
   a=pack(d,'dense');b=pack(sparse,'sparse',side*side*per)
   assert len(a)==len(b)==32+2*width*side*side
   np.testing.assert_array_equal(unpack(a),d.astype('f2').astype('f4'))
   np.testing.assert_array_equal(unpack(b),sparse.astype('f2').astype('f4'))

def test_zero_padding_does_not_change_index_zero():
 x=np.zeros((8,2,2),dtype='f4');x[0,0,0]=1
 np.testing.assert_array_equal(x,unpack(pack(x,'sparse',10)))

def test_capacity_and_truncation_errors():
 with pytest.raises(ValueError):pack(np.ones((8,2,2)),'sparse',2)
 with pytest.raises(ValueError):unpack(pack(np.ones((2,2,2)),'dense')[:-1])
