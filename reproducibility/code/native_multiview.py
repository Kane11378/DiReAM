"""Complete native S14 unmasked task adapter with slot/compression choices.

Original native aspect-preserving 448-pixel short-edge/eight normal rotations, one query reading per view.
Known-slot and fixed 1% compression are explicitly named adaptations.
Normal source reuse is optional, explicit, and checked against ordered members.
AnomalyDINO_U_pool_norot instead fits full pooled memory from direct unwarped
normal views. Its queries remain unmasked, with one reading per actual view.
"""
from pathlib import Path
import sys,argparse,time,gc
sys.path.insert(0,str(Path(__file__).resolve().parent))
from complete_methods import Runtime,read,dump,sha,image,select
import numpy as np
PROFILES=['AnomalyDINO_U_pool','AnomalyDINO_U_pool_coreset01','AnomalyDINO_U_slot','AnomalyDINO_U_slot_coreset01','AnomalyDINO_U_pool_norot']
def require_norot_source(spec):
 assert spec.get('normal_rotation') is False and spec.get('normal_augmentations_per_view')==1,'No-rotation fitting cannot reuse an eight-rotation bank'
 assert spec.get('normal_input_path')=='direct_unwarped_rgb' and spec.get('memory')=='full' and spec.get('reference_foreground') is False,'Require explicit direct, unmasked, full normal source'
def normal_records(folder):
 spec=read(folder/'normal_shards.json');root=Path(spec.get('explicit_source_directory',folder));records=spec['records']
 for r in records:assert sha(root/r['file'])==r['sha256']
 return spec,root,records
def materialize(root,records,slot):
 n=sum(r['shape'][0] if slot is None else r['slot_slices'][slot][1]-r['slot_slices'][slot][0] for r in records);out=np.empty((n,384),np.float32);off=0
 for r in records:
  x=np.load(root/r['file'],mmap_mode='r');a,b=(0,len(x)) if slot is None else r['slot_slices'][slot];out[off:off+b-a]=x[a:b];off+=b-a
 assert off==n;return out
def load_engines(folder,profile):
 import torch
 from reader_baseline_adapters import TorchExactNN
 spec,root,records=normal_records(folder);engines=[];slots=range(spec['views']) if '_slot' in profile else [None]
 if profile=='AnomalyDINO_U_pool_norot':
  require_norot_source(spec);fit=read(folder/'FIT.json')
  assert fit['profile']==profile and fit['normal_ids']==spec['normal_ids'] and fit['normal_shards_sha256']==sha(folder/'normal_shards.json')
 for slot in slots:
  e=TorchExactNN('cuda',q_chunk=1024,r_chunk=65536);tag='pool' if slot is None else f'slot_{slot}'
  if profile.endswith('coreset01'):e.fit(np.load(folder/(tag+'_bank.npy')))
  else:
   n=sum(r['shape'][0] if slot is None else r['slot_slices'][slot][1]-r['slot_slices'][slot][0] for r in records);e.bank=torch.empty((n,384),dtype=torch.float32,device='cuda');e.norm=torch.empty(n,dtype=torch.float32,device='cuda');off=0
   for r in records:
    x=np.load(root/r['file'],mmap_mode='r');a,b=(0,len(x)) if slot is None else r['slot_slices'][slot];z=torch.from_numpy(np.array(x[a:b],np.float32,order='C')).cuda();e.bank[off:off+len(z)]=z;e.norm[off:off+len(z)]=(z*z).sum(1);off+=len(z)
  engines.append(e)
 return engines
def main():
 import torch,cv2
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['fit','predict']);p.add_argument('--profile',required=True,choices=PROFILES);p.add_argument('--base-reader',type=Path,required=True);p.add_argument('--resources',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--category',required=True);p.add_argument('--image-source',type=Path,required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--normal-source',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
 manifest=read(a.manifest);g=manifest['groups'][a.category];resources=read(a.resources);resources['_source_json']=str(a.resources.resolve());runtime=Runtime('AnomalyDINO_F',a.base_reader,resources);model=runtime.model
 if a.image_source.is_file():assert sha(a.image_source)==manifest['archive_sha256']
 else:assert a.image_source.is_dir()
 if a.mode=='fit':
  a.model.mkdir(parents=True,exist_ok=False);ids=[r['global_identity'] for r in g['reference']];start=time.perf_counter()
  if a.profile=='AnomalyDINO_U_pool_norot':assert g['reference'] and all(len(r['real_view_boxes'])==5 for r in g['reference']),'The unmasked no-rotation profile uses five-view units'
  if a.normal_source:
   s,root,records=normal_records(a.normal_source);assert s['normal_ids']==ids
   normal_spec=dict(records=records,normal_ids=ids,views=s['views'],explicit_source_directory=str(root.resolve()),normal_source_sha256=sha(a.normal_source/'normal_shards.json'))
   if a.profile=='AnomalyDINO_U_pool_norot':
    require_norot_source(s);normal_spec.update(normal_rotation=False,normal_augmentations_per_view=1,normal_input_path='direct_unwarped_rgb',memory='full',reference_foreground=False)
   dump(a.model/'normal_shards.json',normal_spec)
  else:
   records=[];views=len(g['reference'][0]['real_view_boxes'])
   for i,row in enumerate(g['reference']):
    chunks=[];slices=[];off=0
    for im in image(row,a.image_source):
     if a.profile=='AnomalyDINO_U_pool_norot':aug=[im]
     else:aug=model.upstream_utils.augment_image(im);assert len(aug)==8
     begin=off
     for z in aug:f,_=model._features(z);q=model._normalize(f);chunks.append(q);off+=len(q)
     slices.append((begin,off))
    assert len(slices)==views;f=a.model/f'normal_{i:05d}.npy';x=np.concatenate(chunks);np.save(f,x);records.append(dict(global_identity=row['global_identity'],file=f.name,shape=list(x.shape),slot_slices=slices,sha256=sha(f)))
   normal_spec=dict(records=records,normal_ids=ids,views=views)
   if a.profile=='AnomalyDINO_U_pool_norot':normal_spec.update(normal_rotation=False,normal_augmentations_per_view=1,normal_input_path='direct_unwarped_rgb',memory='full',reference_foreground=False)
   dump(a.model/'normal_shards.json',normal_spec)
  s,root,records=normal_records(a.model)
  if a.profile.endswith('coreset01'):
   torch.use_deterministic_algorithms(True)
   for slot in (range(s['views']) if '_slot' in a.profile else [None]):
    tag='pool' if slot is None else f'slot_{slot}';x=materialize(root,records,slot);bank,ix=select(x,.01,resources);np.save(a.model/(tag+'_bank.npy'),bank);np.save(a.model/(tag+'_indices.npy'),ix);del x,bank;gc.collect();torch.cuda.empty_cache()
  fit=dict(status='ACTUAL_RAW_NORMAL_FIT_COMPLETE',profile=a.profile,normal_ids=ids,normal_sources_reused_explicitly=bool(a.normal_source),elapsed_s=time.perf_counter()-start,GT_reads=0)
  if a.profile=='AnomalyDINO_U_pool_norot':fit.update(normal_rotation=False,normal_augmentations_per_view=1,normal_input_path='direct_unwarped_rgb',memory='full',query_foreground=False,reference_foreground=False,normal_vectors=sum(r['shape'][0] for r in records),normal_shards_sha256=sha(a.model/'normal_shards.json'))
  dump(a.model/'FIT.json',fit)
 else:
  assert a.output is not None;a.output.mkdir(parents=True,exist_ok=False);engines=load_engines(a.model,a.profile);records=[]
  for i,row in enumerate(g['calibration']+g['query']):
   common=[];native=[];patch=[];scores=[]
   for v,im in enumerate(image(row,a.image_source)):
    f,grid=model._features(im);q=model._normalize(f);e=engines[v] if len(engines)>1 else engines[0];d=e.run(1,q)[0][:,0]/2;k=int(len(d)*.01);scores.append(float(d.max() if k==0 else np.mean(np.sort(d)[::-1][:k])));field=model.upstream_utils.dists2map(d.reshape(grid),im.shape).astype(np.float32);native.append(field);common.append(cv2.resize(field,(256,256),interpolation=cv2.INTER_LINEAR));patch.append(d.reshape(grid))
   f=a.output/f'{i:05d}.npz';np.savez(f,maps256=np.stack(common),native_maps=np.stack(native),native_view_scores=np.asarray(scores,np.float64),object_score=np.array(max(scores),np.float64),actual_readings_per_view=np.array(1,np.int32),global_identity=np.array(row['global_identity']),**{f'native_patch_{v}':z for v,z in enumerate(patch)});records.append(dict(global_identity=row['global_identity'],file=f.name,sha256=sha(f)))
  dump(a.output/'COMPLETE.json',dict(status='ACTUAL_RAW_RGB_PREDICTION_COMPLETE',method=a.profile,records=records,manifest_sha256=sha(a.manifest),GT_reads=0))
if __name__=='__main__':main()
