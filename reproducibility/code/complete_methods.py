"""Raw-image normal fitting and prediction for complete competing systems.

All data and external pretrained resources are supplied explicitly. No download,
implicit workspace discovery, GT access, parameter grid or efficacy selection.
The fixed-study reader supplies the unchanged S14 instrument/search kernels.
AnomalyDINO_F_norot fits full memory from direct, unwarped normal RGB views;
it retains native aspect geometry and the query-only foreground readout.
"""
from pathlib import Path
import os,sys,json,hashlib,io,argparse,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
os.environ.setdefault('XFORMERS_DISABLED','1')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
LAYERS=[-12,-13,-14,-15,-16,-17,-18]
PROFILES=['within_M05','within_M05_FC4','meanCov_M07','meanCov_M07_FC4','raw_mean_FC4','within_N01_fast_FC4','within_N01_fast_F0']
ALLOWED_PROFILES=PROFILES+['within_N01_fast']
METHODS=['MH_PatchCore','SubspaceAD','SubspaceAD_slot','AnomalyDINO_F','AnomalyDINO_F_coreset01','AnomalyDINO_F_norot','DiReAM']
def profile_base(profile):return 'within_N01_fast' if profile.startswith('within_N') else 'within_M05' if profile.startswith('within_M') else 'meanCov_M07' if profile.startswith('meanCov') else 'raw_mean'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def dump(p,z):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(z,indent=2,ensure_ascii=False),encoding='utf8')
def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def image(row,tar_path):
 from PIL import Image
 source=Path(tar_path)
 if source.is_dir():
  p=(source/row['rgb_member']).resolve();assert p.is_relative_to(source.resolve()),'RGB path escapes explicit image directory';b=p.read_bytes()
 else:
  with source.open('rb') as f:f.seek(row['rgb_offset_data_uncompressed']);b=f.read(row['rgb_member_bytes'])
 assert hashlib.sha256(b).hexdigest()==row['source_payload_sha256'],'Original RGB payload changed'
 with Image.open(io.BytesIO(b)) as im:rgb=np.array(im.convert('RGB'))
 return [rgb[y0:y1,x0:x1].copy() for x0,y0,x1,y1 in row['real_view_boxes']]
def load_params(p):
 with np.load(p,allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
 for k in ['k']:
  if k in d:d[k]=int(d[k])
 if 'eps' in d:d['eps']=float(d['eps'])
 return d
def select(source,ratio,resources):
 import torch
 pc=Path(resources['upstream']['patchcore']);p=pc/'src/patchcore/sampler.py'
 assert sha(p)=='39612cd2b486865ece304348f740f4bc700a1c2e8684fbab3c2ed016c7801f2d'
 sys.path.insert(0,str(pc/'src'));from coreset_runtime import ExactUsefulSampler
 np.random.seed(0);torch.manual_seed(0)
 sampler=ExactUsefulSampler(ratio,torch.device('cuda'),number_of_starting_points=10,dimension_to_project_features_to=128)
 with torch.no_grad():bank=sampler.run(source)
 assert np.array_equal(bank,source[sampler.indices]);return np.ascontiguousarray(bank),sampler.indices
class Runtime:
 def __init__(self,method,base,resources,profiles=None):
  import torch
  self.method=method;self.resources=resources;self.base=Path(base);self.profiles=profiles or PROFILES
  assert all(p in ALLOWED_PROFILES for p in self.profiles)
  sys.path.insert(0,str(self.base/'code'));sys.path.insert(0,str(ROOT/'code'))
  torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
  # The source native systems use non-deterministic-algorithm mode; the fixed
  # S14 controlled/compression recipes explicitly enable deterministic mode.
  torch.use_deterministic_algorithms(method in ['DiReAM','AnomalyDINO_F_coreset01'])
  if method=='MH_PatchCore':self.mh()
  elif method.startswith('SubspaceAD'):self.subspace()
  else:self.small()
 def mh(self):
  import torch,yaml
  from torchvision.models import get_model_weights
  source=ROOT/'third_party/MH_PatchCore';sys.path.insert(0,str(source))
  from mhpc.eval.config import load_run_config
  from mhpc.core.plugins.plugin_chain import build_runtime_plugin_chain
  from mhpc.util.param_binding import build_plugin_bind_context
  from mhpc.core.mh_patch_core import MHPatchCore
  from mhpc.core.plugins.dataloader.mvtec_dataloader_augment.dataset_runtime import build_transforms
  # Weight lookup is explicit: the exact V2 file must already be installed in
  # this supplied torch cache. No implicit weight download is permitted here.
  cache=Path(self.resources['mhpc_torch_hub']);wp=cache/'checkpoints/wide_resnet50_2-9ba9bcbe.pth'
  assert sha(wp)==read(ROOT/'protocol/model_pins.json')['wide_resnet50_2_v2_sha256']
  assert get_model_weights('wide_resnet50_2').DEFAULT.url.rsplit('/',1)[-1]==wp.name,'This runtime DEFAULT must be the pinned WR50-V2 checkpoint'
  torch.hub.set_dir(str(cache))
  self.config=yaml.safe_load((source/'configs/mvtec/baselines/mvtec_streaming_mh_patchcore.yaml').read_text('utf8'))
  self.config['runtime']['device']='auto';self.config['pipeline']['slots']['dataloader']['params']['num_workers']=0;self.config['render']['progress']['enabled']=False
  self.mh_config_functions=(load_run_config,build_runtime_plugin_chain,build_plugin_bind_context,MHPatchCore)
  self.transform=build_transforms(resize=(256,256),img_size=(224,224),augment=False,cfg=None,dtype=torch.float32)
 def mh_initialize(self,folder):
  import torch,yaml
  load,build,bind,Model=self.mh_config_functions
  self.config['paths']['output_root']=str(folder);self.config['pipeline']['slots']['dataloader']['params']['dataset_root']=str(folder)
  p=folder/'runtime_config.yaml';p.write_text(yaml.safe_dump(self.config),encoding='utf8');c=load(p)
  bundle=build(selection_map=c.plugins.as_selection_map(),bind_context=bind(training_contract=c.training.contract,seed=c.experiment.seed),slot_params_map=c.slot_params)
  self.model=Model(device=torch.device('cuda'),training_contract=c.training.contract,fit_epochs=dict(c.training.fit_epochs),plugin_bundle=bundle)
 def subspace(self):
  source=ROOT/'third_party/SubspaceAD';sys.path.insert(0,str(source/'src'))
  from subspacead.core.extractor import FeatureExtractor
  from subspacead.core.pca import PCAModel
  from subspacead.post_process.scoring import calculate_anomaly_scores,post_process_map,aggregate_image_score
  giant=Path(self.resources['subspace_giant']);assert giant.is_dir()
  pins=read(ROOT/'protocol/model_pins.json')['giant_files']
  for pin in pins:assert sha(giant/pin['file'])==pin['sha256']
  self.model=FeatureExtractor(str(giant));self.PCA=PCAModel;self.score=calculate_anomaly_scores;self.post=post_process_map;self.aggregate=aggregate_image_score
  # Literal state identities are translated against the ORIGINAL 41 states.
  # The author full-shot branch never uses attention/saliency. Preserve eager
  # FP32 active operations and remove only unconsumed outputs/descendants.
  assert len(self.model.model.encoder.layer)==40
  self.model.model.encoder.layer=self.model.model.encoder.layer[:29]
 def giant_features(self,im):
  import torch
  from PIL import Image
  m=self.model
  with torch.no_grad():
   inputs=m.processor(images=[Image.fromarray(im)],return_tensors='pt',do_resize=True,size={'height':672,'width':672},do_center_crop=False,crop_size={'height':672,'width':672}).to(next(m.model.parameters()).device)
   out=m.model(**inputs,output_hidden_states=True,output_attentions=False)
   return m._aggregate_layers(out.hidden_states,[29,28,27,26,25,24,23],[],'mean',5,2304,48,48).reshape(2304,1536)
 def small(self):
  import torch
  from reader_common import StudyProtocol,ResourcePaths
  from reader_dino_encoder import DinoC4Encoder
  from reader_baseline_adapters import configure_resources,AnomalyDINOOfficialS
  study=StudyProtocol.load(self.base/'protocol/study.json');resources=ResourcePaths.load(self.resources['_source_json'])
  # Use the supplied reader's explicit-resource loader.
  if self.method=='DiReAM':self.encoder=DinoC4Encoder('cuda',study=study,resources=resources)
  else:
   configure_resources(study,resources,'anomalydino');self.model=AnomalyDINOOfficialS(device='cuda',resolution=448,search_backend='torch-exact')
   p=ROOT/'protocol/s14_native_encoder_buffers.npz';assert sha(p)==read(ROOT/'protocol/model_pins.json')['native_s14_buffer_sha256']
   buffers=dict(self.model.wrapper.model.named_buffers())
   with np.load(p,allow_pickle=False) as z:
    assert set(z.files)==set(buffers)
    with torch.no_grad():
     for k,v in buffers.items():v.copy_(torch.from_numpy(z[k]).to(device=v.device,dtype=v.dtype))
 def fit(self,rows,tar_path,folder):
  import torch
  folder=Path(folder);folder.mkdir(parents=True,exist_ok=False)
  ids=[r['global_identity'] for r in rows];start=time.perf_counter()
  if self.method=='MH_PatchCore':
   from torch.utils.data import Dataset,DataLoader
   self.mh_initialize(folder);transform=self.transform
   class ExplicitNormals(Dataset):
    def __init__(self):self.pairs=[(i,v) for i,r in enumerate(rows) for v in range(len(r['real_view_boxes']))]
    def __len__(self):return len(self.pairs)
    def __getitem__(self,j):
     i,v=self.pairs[j];im=image(rows[i],tar_path)[v];return transform(torch.from_numpy(im).permute(2,0,1))
   self.model.fit(DataLoader(ExplicitNormals(),batch_size=16,shuffle=False,num_workers=0));self.model.save_to_path(folder/'checkpoint')
  elif self.method.startswith('SubspaceAD'):
   cache=folder/'normal_features.npy';bindings=[(i,v) for i,r in enumerate(rows) for v in range(len(r['real_view_boxes']))]
   f=np.lib.format.open_memmap(cache,mode='w+',dtype=np.float32,shape=(len(bindings),2304,1536))
   for j,(i,v) in enumerate(bindings):f[j]=self.giant_features(image(rows[i],tar_path)[v])
   f.flush()
   if self.method=='SubspaceAD_slot':
    slots=len(rows[0]['real_view_boxes']);assert all(len(r['real_view_boxes'])==slots for r in rows)
    for slot in range(slots):
     def gen(slot=slot):
      for j,a in enumerate(f):
       if bindings[j][1]==slot:yield np.asarray(a)
     p=self.PCA(ev=.99,whiten=False).fit(gen,1536,len(rows)*2304,len(rows));np.savez(folder/f'pca_slot_{slot}.npz',**p)
   else:
    def gen():
     for a in f:yield np.asarray(a)
    p=self.PCA(ev=.99,whiten=False).fit(gen,1536,len(bindings)*2304,len(bindings));np.savez(folder/'pca_params.npz',**p)
  elif self.method.startswith('AnomalyDINO_F'):
   if self.method=='AnomalyDINO_F_norot':assert rows and all(len(r['real_view_boxes'])==1 for r in rows),'The masked no-rotation profile uses single-image units'
   source=[]
   for i,row in enumerate(rows):
    chunks=[]
    for im in image(row,tar_path):
     if self.method=='AnomalyDINO_F_norot':aug=[im]
     else:aug=self.model.upstream_utils.augment_image(im);assert len(aug)==8
     for a in aug:features,_=self.model._features(a);chunks.append(self.model._normalize(features))
    p=folder/f'normal_{i:05d}.npy';z=np.concatenate(chunks);np.save(p,z);source.append(dict(file=p.name,shape=list(z.shape),sha256=sha(p)))
   normal_spec=dict(records=source)
   if self.method=='AnomalyDINO_F_norot':normal_spec.update(normal_rotation=False,normal_augmentations_per_view=1,normal_input_path='direct_unwarped_rgb',memory='full',normal_ids=ids,reference_foreground=False)
   dump(folder/'normal_shards.json',normal_spec)
   if self.method.endswith('coreset01'):
    x=np.concatenate([np.load(folder/r['file']) for r in source]);bank,ix=select(x,.01,self.resources);np.save(folder/'bank.npy',bank);np.save(folder/'indices.npy',ix)
  else:
   from normal_statistics import covariance,canonical_covariance,transform
   slots=len(rows[0]['real_view_boxes']);assert all(len(r['real_view_boxes'])==slots for r in rows)
   source=np.lib.format.open_memmap(folder/'normal_components.npy',mode='w+',dtype=np.float32,shape=(len(rows),slots,32,32,4,384))
   for i,row in enumerate(rows):source[i]=self.encoder.encode_views(np.stack(image(row,tar_path)))
   source.flush()
   for slot in range(slots):
    raw=np.array(source[:,slot],np.float32,order='C');mean=raw.mean(-2,dtype=np.float64).astype(np.float32).reshape(-1,384);center=mean.mean(0,dtype=np.float64);W,_,err=covariance(raw);C=canonical_covariance(mean)
    for name,cov,alpha in [('within_M05',W,.5),('meanCov_M07',C,.07),('raw_mean',None,None),('within_N01_fast',W,.5)]:
     d=folder/f'slot_{slot}';d.mkdir(exist_ok=True)
     if cov is None:rep=mean
     else:
      matrix,_=transform(cov,alpha);np.savez(d/(name+'_model.npz'),matrix=matrix,center=center);x=raw.reshape(-1,384) if name=='within_N01_fast' else mean;rep=((x.astype(np.float64)-center)@matrix).astype(np.float32)
     bank,ix=select(rep,.01 if name=='within_N01_fast' else .04,self.resources);np.save(d/(name+'_bank.npy'),bank);np.save(d/(name+'_indices.npy'),ix)
  torch.cuda.synchronize();fit=dict(status='ACTUAL_RAW_NORMAL_FIT_COMPLETE',method=self.method,normal_ids=ids,GT_reads=0,elapsed_s=time.perf_counter()-start)
  if self.method=='AnomalyDINO_F_norot':fit.update(normal_rotation=False,normal_augmentations_per_view=1,normal_input_path='direct_unwarped_rgb',memory='full',query_foreground=True,reference_foreground=False,normal_vectors=sum(r['shape'][0] for r in source),normal_shards_sha256=sha(folder/'normal_shards.json'))
  dump(folder/'FIT.json',fit)
 def load(self,folder):
  import torch
  from reader_baseline_adapters import TorchExactNN
  folder=Path(folder);self.folder=folder
  if self.method=='MH_PatchCore':self.mh_initialize(folder);self.model.load_from_path(folder/'checkpoint')
  elif self.method=='SubspaceAD':self.pca=[load_params(folder/'pca_params.npz')]
  elif self.method=='SubspaceAD_slot':self.pca=[load_params(p) for p in sorted(folder.glob('pca_slot_*.npz'))]
  elif self.method.startswith('AnomalyDINO_F'):
   if self.method=='AnomalyDINO_F_norot':
    fit=read(folder/'FIT.json');spec=read(folder/'normal_shards.json')
    assert fit['method']==self.method and fit['normal_rotation'] is False and fit['memory']=='full'
    assert fit['normal_shards_sha256']==sha(folder/'normal_shards.json')
    assert spec['normal_rotation'] is False and spec['normal_augmentations_per_view']==1 and spec['normal_input_path']=='direct_unwarped_rgb' and spec['normal_ids']==fit['normal_ids']
   self.nn=TorchExactNN('cuda',q_chunk=1024,r_chunk=65536)
   if self.method.endswith('coreset01'):self.nn.fit(np.load(folder/'bank.npy'))
   else:
    records=read(folder/'normal_shards.json')['records'];n=sum(r['shape'][0] for r in records);self.nn.bank=torch.empty((n,384),device='cuda');self.nn.norm=torch.empty(n,device='cuda');off=0
    for r in records:
     p=folder/r['file'];assert sha(p)==r['sha256'];x=torch.from_numpy(np.load(p)).cuda();self.nn.bank[off:off+len(x)]=x;self.nn.norm[off:off+len(x)]=(x*x).sum(1);off+=len(x)
  elif self.method=='DiReAM':
   from reader_search import HostOriginalSearch
   self.resident={}
   for slot,d in enumerate(sorted(folder.glob('slot_*'))):
    if not d.is_dir():continue
    for base in dict.fromkeys(profile_base(p) for p in self.profiles):
     bank=np.load(d/(base+'_bank.npy'));self.resident[(slot,base)]=(HostOriginalSearch(bank,'cuda',copy_host=False),load_params(d/(base+'_model.npz')) if base!='raw_mean' else None)
 def predict(self,vv):
  import torch,cv2
  from PIL import Image
  if self.method=='MH_PatchCore':
   x=torch.stack([self.transform(torch.from_numpy(im).permute(2,0,1)) for im in vv]);p=self.model.infer_batch(x);native=np.asarray(p.pred_maps,np.float32);common=np.zeros((len(vv),256,256),np.float32);common[:,16:240,16:240]=native
   return {'native_maps':native,'maps256':common,'object_score':np.array(max(p.image_scores),np.float64)}
  if self.method.startswith('SubspaceAD'):
   native=[];scores=[]
   for v,im in enumerate(vv):
    f=self.giant_features(im);d=self.score(f,self.pca[v] if self.method=='SubspaceAD_slot' else self.pca[0],'reconstruction',0).reshape(48,48);m=self.post(d,672);native.append(m);scores.append(self.aggregate(m,'mtop1p'))
   return {'native_maps':np.asarray(native,np.float32),'maps256':np.stack([cv2.resize(m,(256,256),interpolation=cv2.INTER_LINEAR) for m in native]),'object_score':np.array(max(scores),np.float64)}
  if self.method.startswith('AnomalyDINO_F'):
   native=[];maps=[];scores=[];patches=[]
   for im in vv:
    f,grid=self.model._features(im);np.random.seed(0);mask=self.model.wrapper.compute_background_mask(f,grid,threshold=10,masking_type=True);d=np.zeros(len(f),np.float64)
    if mask.any():d[mask]=self.nn.run(1,self.model._normalize(f[mask]))[0][:,0]/2
    k=int(len(d)*.01);scores.append(float(d.max() if k==0 else np.sort(d)[-k:].mean()));patches.append(d.reshape(grid));m=self.model.upstream_utils.dists2map(d.reshape(grid),im.shape).astype(np.float32);native.append(m);maps.append(cv2.resize(m,(256,256),interpolation=cv2.INTER_LINEAR))
   result={'maps256':np.stack(maps),'object_score':np.array(max(scores),np.float64)}
   # Actual views of one unit have the same geometry in the supplied protocols.
   result.update(native_maps=np.stack(native),native_patch_maps=np.stack(patches))
   if self.method=='AnomalyDINO_F_norot':result['actual_readings_per_view']=np.array(1,np.int32)
   return result
  from reader_dino_encoder import square_views,normalize_components
  from reader_search import HostOriginalSearch
  from reader_dino_matching import final_fields
  from foreground_policy import foreground_mask
  enc=[];masks=[];mu=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];sd=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
  directions=[0] if all(p.startswith('within_N01_fast') and not p.endswith('_FC4') for p in self.profiles) else range(4)
  need_foreground=any(p.endswith(('_FC4','_F0')) for p in self.profiles)
  for im in vv:
   square=square_views(im[None]);components=[];mm=[]
   for k in directions:
    turn=np.rot90(square,k,axes=(1,2)).copy();x=torch.from_numpy(turn.transpose(0,3,1,2).copy()).cuda().float()/255;x=(x-mu)/sd
    with torch.inference_mode():token=self.encoder.model.get_intermediate_layers(x,n=1,reshape=False,return_class_token=False,norm=True)[0].cpu().numpy()
    z,_=normalize_components(np.rot90(token.reshape(1,32,32,384),-k,axes=(1,2)).copy(),'error');components.append(z[0]);mm.append(np.ascontiguousarray(np.rot90(foreground_mask(token[0],(32,32)).reshape(32,32),-k)) if need_foreground else np.zeros((32,32),bool))
   enc.append(np.stack(components,-2));masks.append(np.stack(mm,-1))
  result={}
  for profile in self.profiles:
   base=profile_base(profile);raw=[]
   for slot,(x,m) in enumerate(zip(enc,masks)):
    engine,p=self.resident[(slot,base)];q=x[:,:,0].reshape(-1,384) if base=='within_N01_fast' else x.mean(-2,dtype=np.float64).astype(np.float32).reshape(-1,384)
    if base!='raw_mean':q=((q.astype(np.float64)-p['center'])@p['matrix']).astype(np.float32)
    z=engine(q)[0].reshape(32,32)*np.float32(192)
    if profile.endswith('_FC4'):z=(z.astype(np.float64)*m.mean(-1,dtype=np.float64)).astype(np.float32)
    elif profile.endswith('_F0'):z=z*m[...,0]
    raw.append(z)
   fields=final_fields(np.asarray(raw,np.float32));result.update({profile+'_'+k:v for k,v in fields.items()})
  result['actual_directions_per_view']=np.array(len(directions),np.int32)
  return result
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['fit','predict']);p.add_argument('--method',required=True,choices=METHODS);p.add_argument('--profiles',nargs='+',choices=ALLOWED_PROFILES);p.add_argument('--base-reader',required=True,type=Path);p.add_argument('--resources',required=True,type=Path);p.add_argument('--manifest',required=True,type=Path);p.add_argument('--category',required=True);p.add_argument('--image-source','--visa-tar',dest='visa_tar',required=True,type=Path);p.add_argument('--model',required=True,type=Path,help='State folder; MH-PatchCore writes and loads runtime_config.yaml here');p.add_argument('--output',type=Path);a=p.parse_args()
 manifest=read(a.manifest);group=manifest['groups'][a.category]
 if a.visa_tar.is_file():assert a.visa_tar.stat().st_size==manifest['archive_bytes'];assert sha(a.visa_tar)==manifest['archive_sha256']
 else:assert a.visa_tar.is_dir()
 resources=read(a.resources);resources['_source_json']=str(a.resources.resolve());runtime=Runtime(a.method,a.base_reader,resources,a.profiles)
 if a.mode=='fit':runtime.fit(group['reference'],a.visa_tar,a.model)
 else:
  assert a.output is not None;runtime.load(a.model);a.output.mkdir(parents=True,exist_ok=False);rows=group['calibration']+group['query'];records=[]
  for i,row in enumerate(rows):
   z=runtime.predict(image(row,a.visa_tar));f=a.output/f'{i:05d}.npz';np.savez(f,**z,global_identity=np.array(row['global_identity']));records.append(dict(global_identity=row['global_identity'],file=f.name,sha256=sha(f)))
  dump(a.output/'COMPLETE.json',dict(status='ACTUAL_RAW_RGB_PREDICTION_COMPLETE',method=a.method,manifest_sha256=sha(a.manifest),records=records,GT_reads=0))
if __name__=='__main__':main()
