"""Normal-calibrated directional comparisons: exact frozen numerical operations.

No file discovery, models, fitting, data access or implicit resource paths.
Input/library validity and explicit resource binding belong to the caller.
"""
import time
import numpy as np

BASES={'within_M05','meanCov_M05','within_I01','within_N01_fast','canonical_S07','predicted_teacher04'}

ALIASES={'within_M05_F0':'within_M05','within_M05_FC4':'within_M05','meanCov_M05_FC4':'meanCov_M05',
 'within_I01_FC4':'within_I01','within_N01_fast_F0':'within_N01_fast','canonical_S07_F0':'canonical_S07','predicted_teacher04_F0':'predicted_teacher04'}

def readings(method):return 4 if ALIASES.get(method,method) in ['within_M05','meanCov_M05','within_I01'] else 1

def raw_field(x,method,engine,model,masks=None):
 base=ALIASES.get(method,method);assert base in BASES
 x=np.asarray(x);assert x.dtype==np.float32 and x.shape in [(32,32,4,384),(32,32,384)] and np.isfinite(x).all()
 t=time.perf_counter()
 if base in ['within_M05','meanCov_M05']:
  assert x.ndim==4;z=x.mean(-2,dtype=np.float64).astype(np.float32).reshape(-1,384)
 elif base=='within_I01':assert x.ndim==4;z=x.reshape(-1,384)
 else:z=(x[:,:,0] if x.ndim==4 else x).reshape(-1,384)
 if base=='predicted_teacher04':
  z=z.astype(np.float64);z=(z+model['bias']+(z-model['mu'])@model['correction']).astype(np.float32)
 else:z=((z.astype(np.float64)-model['center'])@model['matrix']).astype(np.float32)
 head=time.perf_counter()-t;t=time.perf_counter();d=engine(z)[0];search=time.perf_counter()-t;t=time.perf_counter()
 if base=='within_I01':
  d=d.reshape(32,32,4)
  if method=='within_I01_FC4':
   assert masks is not None and masks.shape==(32,32,4) and masks.dtype==bool
   d=(d.astype(np.float64)*masks).mean(-1,dtype=np.float64).astype(np.float32)
  else:d=d.mean(-1,dtype=np.float64).astype(np.float32)
 d=d.reshape(32,32)*np.float32(192)
 if method in ALIASES and method!='within_I01_FC4':
  assert masks is not None and masks.shape==(32,32,4) and masks.dtype==bool
  if method.endswith('_FC4'):d=(d.astype(np.float64)*masks.mean(-1,dtype=np.float64)).astype(np.float32)
  else:d=d*masks[...,0]
 assert d.shape==(32,32) and d.dtype==np.float32 and np.isfinite(d).all()
 return d,dict(normal_head_and_feature_organization_s=head,reference_search_s=search,field_pooling_foreground_s=time.perf_counter()-t)
