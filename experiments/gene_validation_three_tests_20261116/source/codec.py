"""Fixed-length little endian codec; no entropy coding or hidden index payload."""
import struct
import numpy as np
HEADER=struct.Struct('<4s5I8x')

def pack(array,kind,capacity=None):
 a=np.asarray(array,dtype=np.float32)
 if a.ndim!=3:raise ValueError('Expected CHW')
 c,h,w=a.shape
 if kind=='dense':
  payload=a.astype('<f2').tobytes();k=a.size;flag=0
 elif kind=='sparse':
  k=int(capacity);flat=a.ravel();indices=np.flatnonzero(flat)
  if len(indices)>k:raise ValueError('Sparse capacity exceeded')
  # Unused capacity is padding, not a misleading claim of a smaller code.
  ii=np.zeros(k,dtype='<u4');vv=np.zeros(k,dtype='<f2');ii[:len(indices)]=indices;vv[:len(indices)]=flat[indices]
  payload=ii.tobytes()+vv.tobytes();flag=1
 else:raise ValueError(kind)
 return HEADER.pack(b'UGV1',c,h,w,k,flag)+payload

def unpack(blob):
 magic,c,h,w,k,flag=HEADER.unpack_from(blob)
 if magic!=b'UGV1' or flag not in [0,1]:raise ValueError('Invalid header')
 if flag==0:
  if k!=c*h*w or len(blob)!=HEADER.size+2*k:raise ValueError('Invalid dense payload')
  return np.frombuffer(blob,dtype='<f2',offset=HEADER.size,count=k).astype(np.float32).reshape(c,h,w)
 if len(blob)!=HEADER.size+6*k:raise ValueError('Invalid sparse payload')
 idx=np.frombuffer(blob,dtype='<u4',offset=HEADER.size,count=k)
 val=np.frombuffer(blob,dtype='<f2',offset=HEADER.size+4*k,count=k)
 if np.any(idx>=c*h*w):raise ValueError('Index out of range')
 arr=np.zeros(c*h*w,dtype=np.float32)
 np.add.at(arr,idx,val.astype(np.float32))
 return arr.reshape(c,h,w)
