"""Self-contained per-scale gene codes, fixed quantizer shared by the model."""
import struct
import numpy as np
HEADER=struct.Struct('<4s7I')
MAGIC=b'UGG1'

def maximum_bytes(quotas,shapes):
 return sum(HEADER.size+int(k)*(2+int(h)*int(w)) for k,(h,w) in zip(quotas,shapes))

def pack(values,expression_max=4.):
 """Encode one CHW low-resolution activation array after uint8 quantization."""
 a=np.asarray(values,dtype=np.float32)
 if a.ndim!=3 or min(a.shape)<1 or a.shape[0]>65535:raise ValueError('Expected valid gene/H/W array')
 if expression_max<=0 or not np.isfinite(expression_max):raise ValueError('Invalid quantization bound')
 if not np.isfinite(a).all() or np.any(a<0) or np.any(a>expression_max+1e-5):raise ValueError('Out-of-range coefficient')
 q=np.rint(np.clip(a/expression_max,0,1)*255).astype(np.uint8)
 ids=np.flatnonzero(q.any(axis=(1,2))).astype('<u2')
 g,h,w=a.shape
 return HEADER.pack(MAGIC,g,h,w,len(ids),8,0,0)+ids.tobytes()+q[ids].tobytes()

def unpack(blob,expression_max=4.):
 if len(blob)<HEADER.size:raise ValueError('Truncated header')
 magic,g,h,w,n,bits,r0,r1=HEADER.unpack_from(blob)
 if magic!=MAGIC or not 0<g<=65535 or not 0<h<=65536 or not 0<w<=65536 or n>g or bits!=8 or r0 or r1:raise ValueError('Invalid header')
 if not np.isfinite(expression_max) or expression_max<=0:raise ValueError('Invalid quantization bound')
 if len(blob)!=HEADER.size+n*(2+h*w):raise ValueError('Invalid payload length')
 ids=np.frombuffer(blob,dtype='<u2',offset=HEADER.size,count=n)
 if np.any(ids>=g) or (n>1 and np.any(ids[1:]<=ids[:-1])):raise ValueError('Gene IDs must be sorted, unique, in range')
 q=np.frombuffer(blob,dtype=np.uint8,offset=HEADER.size+2*n,count=n*h*w).reshape(n,h,w)
 if n and np.any(~q.any(axis=(1,2))):raise ValueError('Empty gene should not be transmitted')
 result=np.zeros((g,h,w),dtype=np.float32)
 result[ids]=q.astype(np.float32)*(np.float32(expression_max)/np.float32(255))
 return result
