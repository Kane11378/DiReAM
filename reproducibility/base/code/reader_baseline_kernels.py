"""Fixed native baseline numerical operations. Resource and cohort IO is external."""
from pathlib import Path
import hashlib
import numpy as np
import torch
import cv2
from reader_common import sha,under as local_path,canonical_hash
from reader_baseline_adapters import TorchExactNN,PatchCoreOfficialWholeView,AnomalyDINOOfficialS

def check_views(views):
 if not isinstance(views,(list,tuple)) or not views:raise ValueError('Nonempty list of actual RGB views required')
 out=[]
 for v in views:
  a=np.asarray(v)
  if a.ndim!=3 or a.shape[-1]!=3 or a.dtype!=np.uint8 or min(a.shape[:2])<1:
   raise ValueError('Each actual view must be uint8 HWC3; variable H/W allowed')
  out.append(np.ascontiguousarray(a))
 return out

def predict_turn(model,views,turn):
 """One exact query turn, one call per actual image; never stack unequal grids."""
 views=check_views(views);maps=[];scores=[];raw=[];geometry=[];timings=[]
 for v in views:
  rotated=np.ascontiguousarray(np.rot90(v,int(turn),axes=(0,1)))
  r=model.predict(rotated[None]);m=np.asarray(r['maps'][0],np.float32)
  if m.shape!=rotated.shape[:2]:raise ValueError('Adapter must return native map in original rotated-image H/W')
  key='native_maps' if 'native_maps' in r else 'patch_maps'
  native=np.asarray(r[key][0],np.float32)
  score=float(r['view_scores'][0])
  if not np.isfinite(m).all() or not np.isfinite(native).all() or not np.isfinite(score):
   raise ValueError('Nonfinite native baseline output')
  maps.append(m);raw.append(native);scores.append(score)
  geometry.append({'input_hw':list(v.shape[:2]),'rotated_input_hw':list(rotated.shape[:2]),
   'original_frame_map_hw':list(m.shape),'native_grid_hw':list(native.shape),'adapter':r['geometry']})
  timings.append(r['timing'])
 return {'maps_original':maps,'raw_maps':raw,'view_scores':np.asarray(scores,np.float64),
         'geometry':geometry,'adapter_timings':timings,'turn':int(turn)}

def aligned_mean(turns):
 """Mean in each real view's original H/W BEFORE final standard256 resize."""
 if [int(r['turn']) for r in turns]!=[0,1,2,3]:raise ValueError('Exactly fixed C4 turns0,1,2,3 required')
 n=len(turns[0]['maps_original'])
 if any(len(r['maps_original'])!=n for r in turns):raise ValueError('Actual view membership changed between turns')
 aligned=[]
 for v in range(n):
  mm=[np.ascontiguousarray(np.rot90(r['maps_original'][v],-int(r['turn']),axes=(0,1))) for r in turns]
  if any(m.shape!=mm[0].shape for m in mm):raise ValueError('Inverse-rotation H/W mismatch')
  aligned.append(np.mean(np.stack(mm).astype(np.float64),axis=0).astype(np.float32))
 vs=np.mean(np.stack([r['view_scores'] for r in turns]).astype(np.float64),axis=0)
 return aligned,vs

def maps256(original):
 return np.stack([cv2.resize(m,(256,256),interpolation=cv2.INTER_LINEAR) for m in original]).astype(np.float32)

class StreamedExactNN(TorchExactNN):
 """Same inherited exact search; only lossless bank loading/norm workspace differs."""
 def load_shards(self,records,cache_root):
  if not records:raise ValueError('Empty full bank')
  rows=sum(int(r['shape'][0]) for r in records);dim=int(records[0]['shape'][1])
  self.bank=torch.empty((rows,dim),device=self.device,dtype=torch.float32)
  self.norm=torch.empty(rows,device=self.device,dtype=torch.float32)
  offset=0
  for rec in records:
   path=local_path(cache_root,rec['file'])
   if sha(path)!=rec['sha256']:raise ValueError('Bank shard integrity mismatch')
   a=np.load(path,mmap_mode='r',allow_pickle=False)
   if list(a.shape)!=rec['shape'] or a.dtype!=np.float32 or a.shape[1]!=dim:raise ValueError('Lossless bank shard geometry changed')
   for s in range(0,len(a),16384):
    block=np.array(a[s:s+16384],copy=True,order='C')
    b=torch.from_numpy(block).to(self.device);n=len(b)
    self.bank[offset:offset+n].copy_(b)
    self.norm[offset:offset+n].copy_((b*b).sum(1))
    offset+=n
   del a
  if offset!=rows:raise AssertionError('Every frozen bank row must be loaded')
  return {'shape':[rows,dim],'exact_search_implementation':'inherited frozen TorchExactNN.run unchanged',
          'norm_chunk_rows':16384,'approximation':False,'all_ordered_vectors':True}

def encoder(model,kind):return model.model.backbone if kind=='patchcore' else model.wrapper.model

def source_features(model,kind,views):
 """Image-wise native encoding preserves ordering but avoids whole-library copies."""
 features=[];geometry=[]
 for vi,image in enumerate(check_views(views)):
  augmented=[image] if kind=='patchcore' else model.upstream_utils.augment_image(image)
  if kind=='anomalydino' and len(augmented)!=8:raise ValueError('Native eight reference augmentations required')
  for ai,im in enumerate(augmented):
   if kind=='patchcore':
    f=np.asarray(model.model._embed(model._prepare(im[None]).to(model.device)),np.float32)
    grid=[model.image_size//8,model.image_size//8]
   else:
    f,grid=model._features(im);f=model._normalize(f)
   f=np.ascontiguousarray(f,np.float32)
   if f.ndim!=2 or f.shape[1]!=(1024 if kind=='patchcore' else 384) or not np.isfinite(f).all():raise ValueError('Native source feature invalid')
   features.append(f)
   geometry.append({'real_view':vi,'native_augmentation_index':ai,'input_hw':list(im.shape[:2]),
                    'patch_grid_hw':list(grid),'feature_shape':list(f.shape)})
 return np.concatenate(features),geometry

def create_model(kind,device='cuda'):
 if kind=='patchcore':return PatchCoreOfficialWholeView(device=device,image_size=320,coreset_fraction=.01,seed=0,search_backend='torch-exact')
 if kind=='anomalydino':return AnomalyDINOOfficialS(device=device,resolution=448,search_backend='torch-exact')
 raise ValueError(kind)

def as_list(x):return [int(a) for a in x]

def model_buffers(model,kind):
 m=model.model.backbone if kind=='patchcore' else model.wrapper.model
 buffers={n:{'shape':as_list(v.shape),'dtype':str(v.dtype),
  'sha256':hashlib.sha256(v.detach().cpu().contiguous().numpy().tobytes()).hexdigest()}
  for n,v in m.named_buffers()}
 return {'buffers':buffers,'fingerprint':canonical_hash(buffers)}

def prepare_prediction_mode(model,kind):
 if kind=='patchcore':model.model.forward_modules.eval()
 else:model.wrapper.model.eval()
