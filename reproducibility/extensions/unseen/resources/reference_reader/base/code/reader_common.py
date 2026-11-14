"""Portable fixed-cohort IO and identities. No model, GPU or implicit downloads.

Data archives, weights and upstream source trees are supplied explicitly in a
resource JSON. RGB prediction and label preparation are separate stages.
"""
from __future__ import annotations
from pathlib import Path,PurePosixPath
import csv,hashlib,io,json,os,tarfile
import numpy as np
from PIL import Image

ROLES=('reference176','healthy_calibration64','new_confirmation')

def require(condition,message):
 if not condition:raise ValueError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def byte_sha(data):return hashlib.sha256(data).hexdigest()
def array_sha(a):return byte_sha(np.ascontiguousarray(a).tobytes())
def canonical_hash(value):return byte_sha(json.dumps(value,sort_keys=True,separators=(',',':')).encode())
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def atomic_bytes(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_name(path.name+'.partial')
 with temp.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
 os.replace(temp,path)
def atomic_json(path,value):atomic_bytes(path,(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode())
def atomic_npz(path,**arrays):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.stem+'.partial.npz')
 np.savez_compressed(tmp,**arrays);os.replace(tmp,path)
def atomic_npy(path,array):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.stem+'.partial.npy')
 np.save(tmp,array,allow_pickle=False);os.replace(tmp,path)
def load_sealed_npz(path,identity_hash,object_id,validate=None,recover=True):
 path=Path(path);seal=path.with_suffix('.SEALED.json')
 if not path.exists():
  require(not seal.exists(),'Seal without its payload');return None
 with np.load(path,allow_pickle=False) as z:
  require(str(z['identity_hash'].item())==identity_hash and str(z['global_identity'].item())==object_id,'Prediction NPZ identity changed')
  arrays={k:z[k].copy() for k in z.files}
 if validate is not None:validate(arrays)
 rec={'identity_hash':identity_hash,'global_identity':object_id,'sha256':sha(path)}
 if seal.exists():require(read_json(seal)==rec,'Prediction payload/seal changed')
 else:
  require(recover,'Unsealed prediction payload');atomic_json(seal,rec)
 return arrays
def write_sealed_npz(path,identity_hash,object_id,arrays,validate=None):
 path=Path(path);require(not path.exists() and not path.with_suffix('.SEALED.json').exists(),'Preserve existing prediction payload')
 atomic_npz(path,**arrays,identity_hash=np.array(identity_hash),global_identity=np.array(object_id))
 return load_sealed_npz(path,identity_hash,object_id,validate)
def safe_member(name):
 require(isinstance(name,str) and bool(name) and not any(x in name for x in ['\\',':','\0']),'Unsafe member path')
 p=PurePosixPath(name);require(not p.is_absolute() and '..' not in p.parts and bool(p.parts),'Unsafe member path')
 return str(p)
def under(root,relative):
 root=Path(root).resolve();p=root.joinpath(*PurePosixPath(safe_member(relative)).parts)
 require(p.resolve().is_relative_to(root),'Path escapes controlled directory')
 cursor=p
 while cursor!=root and cursor!=cursor.parent:
  require(not cursor.is_symlink(),'Symlink in controlled path');cursor=cursor.parent
 return p.resolve()
def pin(root,path):return {'path':Path(path).resolve().relative_to(Path(root).resolve()).as_posix(),'sha256':sha(path)}
def verify_pin(root,record):
 p=under(root,record['path']);require(p.is_file() and sha(p)==record['sha256'],'Pinned file changed: '+record['path']);return p
def group_slug(group):return hashlib.sha256(group.encode()).hexdigest()[:16]
def configure_compute():
 import torch
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)

class StudyProtocol:
 @classmethod
 def load(cls,path):return cls(path)
 def __init__(self,path):
  self.path=Path(path).resolve();self.root=self.path.parent.parent;self.data=read_json(self.path);self.sha256=sha(self.path)
  require(self.data.get('schema')=='directional-normal-reference-study-v1','Unsupported study schema')
  self.identity=canonical_hash({'study_sha256':self.sha256,'schema':self.data['schema']})
  self.models=read_json(verify_pin(self.root,self.data['models']))
  self.archives=read_json(verify_pin(self.root,self.data['data_sources']))['archives']
  cohort=verify_pin(self.root,self.data['cohort'])
  with cohort.open(encoding='utf-8',newline='') as f:self.rows=list(csv.DictReader(f))
  for row in self.rows:
   for k in ['global_order','role_order','rgb_member_bytes','rgb_offset_data_uncompressed','gt_member_bytes','gt_offset_data_uncompressed']:
    row[k]=None if row[k]=='' else int(row[k])
   row['keep']=row['keep']=='true';row['normal_zero_mask_rule']=row['normal_zero_mask_rule']=='true'
   row['expected_geometry']=json.loads(row.pop('expected_geometry_json'))
   row['expected_view_decoded_sha256']=json.loads(row.pop('expected_view_decoded_sha256_json'))
  self.by_id={r['global_identity']:r for r in self.rows}
  require(len(self.by_id)==len(self.rows),'Duplicate cohort identity')
  require([r['global_order'] for r in self.rows]==list(range(len(self.rows))),'Cohort order changed')
  self.groups=[g['group'] for g in self.data['groups']]
  self.expected_families=list(self.data['expected_families'])
  require(len(set(self.groups))==len(self.groups),'Duplicate source/category group')
  for g in self.data['groups']:
   rr=[r for r in self.rows if r['source']+'::'+r['category']==g['group']]
   for role in ROLES:
    role_rows=[r for r in rr if r['role']==role]
    require(len(role_rows)==g['nominal_counts'][role],'Nominal cohort count changed')
    require(sum(r['keep'] for r in role_rows)==g['actual_counts'][role],'Reviewed cohort count changed')
    require([r['role_order'] for r in role_rows]==list(range(len(role_rows))),'Role order changed')
    if role!='new_confirmation':require(all(r['original_label']=='normal' for r in role_rows),'Normal source/calibration label changed')
  require(all(r['role'] in ROLES and r['archive_key'] in self.archives for r in self.rows),'Unknown role/archive')
 def members(self,group,role):
  require(group in self.groups and role in ROLES,'Unknown fixed group/role')
  return [r for r in self.rows if r['keep'] and r['source']+'::'+r['category']==group and r['role']==role]

class ResourcePaths:
 @classmethod
 def load(cls,path):return cls(path)
 def __init__(self,path):
  self.path=Path(path).resolve();self.data=read_json(self.path)
  require(self.data.get('schema')=='explicit-resource-paths-v1','Unsupported resource path schema')
 def resolve(self,kind,key):
  raw=self.data.get(kind,{}).get(key);require(isinstance(raw,str) and bool(raw),'Explicit resource missing: '+kind+'/'+key)
  require(not raw.startswith('REPLACE_'),'Fill explicit resource path: '+kind+'/'+key)
  p=Path(raw);p=(self.path.parent/p).resolve() if not p.is_absolute() else p.resolve()
  require(p.exists(),'Resource path absent: '+kind+'/'+key);return p
 def archive(self,key):return self.resolve('archives',key)
 def weight(self,key):return self.resolve('weights',key)
 def upstream(self,key):return self.resolve('upstream',key)
 def verify_archive(self,study,key):
  p=self.archive(key);rec=study.archives[key]
  require(p.is_file() and p.stat().st_size==rec['bytes'] and sha(p)==rec['sha256'],'Archive identity mismatch: '+key);return p
 def verify_model(self,study,key):
  rec=study.models['models'][key];p=self.weight(rec['weight_resource'])
  require(sha(p)==rec['weight_sha256'],'Weight SHA differs: '+key)
  for source in rec.get('upstream',[]):
   root=self.upstream(source['resource'])
   for f in study.models['upstream'][source['resource']]['files']:
    z=under(root,f['path']);require(z.is_file() and z.stat().st_size==f['bytes'] and sha(z)==f['sha256'],'Upstream source differs: '+f['path'])
  return p

def decode_rgb(data,max_pixels):
 with Image.open(io.BytesIO(data)) as im:
  require(im.format in {'PNG','JPEG'},'Only PNG/JPEG RGB supported')
  require(0<im.width*im.height<=max_pixels,'RGB dimensions exceed bound')
  require(getattr(im,'n_frames',1)==1,'Multiple-frame image not supported')
  header={'format':im.format,'original_mode':im.mode,'exif_transpose_applied':False}
  rgb=np.asarray(im.convert('RGB'),dtype=np.uint8).copy()
 return rgb,header
def real_view_boxes(source,width,height):
 require(width>0 and height>0,'Empty image')
 if source=='MANTA':
  require(width%5==0,'MANTA width must divide into five real views');step=width//5
  return [[k*step,0,(k+1)*step,height] for k in range(5)]
 require(source=='VisA','Unknown real-view source');return [[0,0,width,height]]
def crop_views(a,boxes):return [a[y0:y1,x0:x1].copy() for x0,y0,x1,y1 in boxes]
def decode_mask(data,max_pixels):
 with Image.open(io.BytesIO(data)) as im:
  require(im.format=='PNG','GT must be PNG')
  require(0<im.width*im.height<=max_pixels,'GT decoded size exceeds bound')
  require(getattr(im,'n_frames',1)==1,'GT multiple frames not allowed')
  require(im.mode in {'1','L','P','I','I;16','I;16B','I;16L','RGB','RGBA'},'Unsupported GT mode')
  a=np.asarray(im).copy()
  if a.ndim==2:return a!=0
  return np.any(a[...,:3]!=0,axis=2)

def selected_payloads(archive_path,entry,wanted,consume,max_member):
 """Explicit bounded members only, one gzip pass; no extractall or link target."""
 found=set();archive_path=Path(archive_path)
 if entry.get('compression')=='gzip' or archive_path.name.endswith(('.tar.gz','.tgz')):
  seen=set()
  with tarfile.open(archive_path,mode='r|gz') as tf:
   for m in tf:
    name=safe_member(m.name);require(not m.issym() and not m.islnk() and (m.isdir() or m.isfile()),'Special/link archive entry')
    require(name not in seen,'Repeated archive header');seen.add(name);require(0<=m.size<=max_member,'Member outside bound')
    if name not in wanted:continue
    r=wanted[name];require(m.isfile() and m.size==r['bytes'] and m.offset_data==r['offset'],'Archive header identity mismatch')
    f=tf.extractfile(m);require(f is not None,'Missing selected member');data=f.read(max_member+1)
    require(len(data)==m.size,'Truncated archive payload');consume(r['row'],data);found.add(name)
 else:
  with archive_path.open('rb') as f:
   for name,r in sorted(wanted.items(),key=lambda kv:kv[1]['offset']):
    offset=int(r['offset']);require(offset>=512 and offset%512==0,'Invalid tar offset');f.seek(offset-512)
    m=tarfile.TarInfo.frombuf(f.read(512),'utf-8','strict')
    require(safe_member(m.name)==name and m.isfile() and not m.issym() and not m.islnk(),'Tar header/member mismatch')
    require(m.size==r['bytes'] and 0<m.size<=max_member and offset+m.size<=entry['bytes'],'Tar size/bound mismatch')
    data=f.read(m.size);require(len(data)==m.size,'Truncated tar payload');consume(r['row'],data);found.add(name)
 require(found==set(wanted),'Selected members absent')

class NativeInputs:
 def __init__(self,study,resources,run):
  self.study=study;self.resources=resources;self.run=Path(run).resolve();self.run.mkdir(parents=True,exist_ok=True)
  self.max_member=study.data['io']['max_member_bytes'];self.max_pixels=study.data['io']['max_decoded_pixels']
  ident={'study_identity':study.identity,'study_sha256':study.sha256}
  p=self.run/'RUN_IDENTITY.json'
  if p.exists():require(read_json(p)==ident,'Run directory belongs to another study')
  else:atomic_json(p,ident)
 def _paths(self,oid,stage):
  stem=byte_sha(oid.encode());return self.run/'inputs'/stage/(stem+'.payload'),self.run/'inputs'/stage/(stem+'.json')
 def _receipt(self,oid,stage):
  payload,receipt=self._paths(oid,stage);r=read_json(receipt)
  require(r['study_identity']==self.study.identity and r['global_identity']==oid,'Input receipt identity changed')
  if r.get('payload') is not None:require(sha(payload)==r['payload_sha256'],'Prepared input bytes changed')
  return r
 def prepare_rgb(self):
  completed=self.run/'RGB_COMPLETE.json'
  if completed.exists():
   old=read_json(completed);require(old['study_identity']==self.study.identity,'RGB stage identity changed')
   require([r['global_identity'] for r in old['rows']]==[r['global_identity'] for r in self.study.rows if r['keep']],'RGB stage coverage changed')
   for r in old['rows']:verify_pin(self.run,r['receipt']);self._receipt(r['global_identity'],'rgb')
   return old['rows']
  wanted={}
  for row in self.study.rows:
   if not row['keep']:continue
   p,s=self._paths(row['global_identity'],'rgb')
   if s.exists():self._receipt(row['global_identity'],'rgb');continue
   wanted.setdefault(row['archive_key'],{})[row['rgb_member']]={'row':row,'bytes':row['rgb_member_bytes'],'offset':row['rgb_offset_data_uncompressed']}
  def consume(row,data):
   require(byte_sha(data)==row['expected_rgb_sha256'],'Selected RGB byte identity changed')
   rgb,header=decode_rgb(data,self.max_pixels);boxes=real_view_boxes(row['source'],rgb.shape[1],rgb.shape[0]);views=crop_views(rgb,boxes)
   require(array_sha(rgb)==row['expected_composite_decoded_sha256'],'RGB decoded identity changed')
   require([array_sha(v) for v in views]==row['expected_view_decoded_sha256'],'Real view identity changed')
   require([rgb.shape[1],rgb.shape[0]]==row['expected_geometry']['composite_wh'] and boxes==row['expected_geometry']['real_view_boxes'],'Native geometry changed')
   p,s=self._paths(row['global_identity'],'rgb');atomic_bytes(p,data)
   atomic_json(s,{'study_identity':self.study.identity,'global_identity':row['global_identity'],'payload':p.relative_to(self.run).as_posix(),
    'payload_sha256':byte_sha(data),'decoded_sha256':array_sha(rgb),'view_decoded_sha256':[array_sha(v) for v in views],
    'geometry':row['expected_geometry'],'header':header})
  for key,requests in wanted.items():selected_payloads(self.resources.verify_archive(self.study,key),self.study.archives[key],requests,consume,self.max_member)
  rows=[{'global_identity':r['global_identity'],'receipt':pin(self.run,self._paths(r['global_identity'],'rgb')[1])} for r in self.study.rows if r['keep']]
  atomic_json(self.run/'RGB_COMPLETE.json',{'study_identity':self.study.identity,'rows':rows,'GT_read':False})
  return rows
 def read_rgb(self,object_id):
  row=self.study.by_id[object_id];require(row['keep'],'Frozen excluded object cannot be substituted')
  r=self._receipt(object_id,'rgb');p,_=self._paths(object_id,'rgb');rgb,_=decode_rgb(p.read_bytes(),self.max_pixels)
  require(r['payload_sha256']==row['expected_rgb_sha256'] and r['decoded_sha256']==row['expected_composite_decoded_sha256'],'RGB receipt differs from cohort identity')
  require(array_sha(rgb)==r['decoded_sha256'],'RGB decoded identity changed')
  views=crop_views(rgb,r['geometry']['real_view_boxes']);require([array_sha(v) for v in views]==r['view_decoded_sha256'],'Real view identity changed');return views
 def prepare_labels(self):
  validate_all_predictions(self.study,self.run);wanted={};records={}
  completed=self.run/'LABELS_COMPLETE.json'
  if completed.exists():
   old=read_json(completed);require(old['study_identity']==self.study.identity,'Label stage identity changed')
   require([r['global_identity'] for r in old['rows']]==[r['global_identity'] for r in self.study.rows if r['keep'] and r['role']=='new_confirmation'],'Label stage coverage changed')
   for r in old['rows']:verify_pin(self.run,r['receipt']);self._receipt(r['global_identity'],'gt')
   return old
  rows=[r for r in self.study.rows if r['keep'] and r['role']=='new_confirmation']
  for row in rows:
   oid=row['global_identity'];p,s=self._paths(oid,'gt')
   if s.exists():records[oid]=self._receipt(oid,'gt');continue
   if row['original_label']=='normal':
    require(row['normal_zero_mask_rule'] and not row['gt_member'],'Normal zero declaration inconsistent')
    rec={'study_identity':self.study.identity,'global_identity':oid,'payload':None,'valid_pixel_GT':True,'normal_zero_mask':True,'native_positive_count':0}
    atomic_json(s,rec);records[oid]=rec
   elif not row['gt_member']:
    rec={'study_identity':self.study.identity,'global_identity':oid,'payload':None,'valid_pixel_GT':False,'reason':'missing_gt_member','normal_zero_mask':False}
    atomic_json(s,rec);records[oid]=rec
   else:wanted.setdefault(row['archive_key'],{})[row['gt_member']]={'row':row,'bytes':row['gt_member_bytes'],'offset':row['gt_offset_data_uncompressed']}
  def consume(row,data):
   oid=row['global_identity'];p,s=self._paths(oid,'gt');atomic_bytes(p,data)
   rec={'study_identity':self.study.identity,'global_identity':oid,'payload':p.relative_to(self.run).as_posix(),'payload_sha256':byte_sha(data),'normal_zero_mask':False}
   try:
    mask=decode_mask(data,self.max_pixels);require(list(mask.shape[::-1])==row['expected_geometry']['composite_wh'],'Native mask/RGB shape mismatch')
    rec.update(valid_pixel_GT=True,binary_decoded_sha256=array_sha(mask.astype(np.uint8)),native_positive_count=int(mask.sum()))
   except (ValueError,OSError) as exc:rec.update(valid_pixel_GT=False,reason=str(exc))
   atomic_json(s,rec);records[oid]=rec
  for key,requests in wanted.items():selected_payloads(self.resources.verify_archive(self.study,key),self.study.archives[key],requests,consume,self.max_member)
  validate_all_predictions(self.study,self.run)
  manifest={'study_identity':self.study.identity,'stage':'labels','binary_rule':'native nonzero ignore alpha; PIL NEAREST256; object labels unchanged',
   'rows':[{'global_identity':r['global_identity'],'object_anomalous':r['original_label']=='anomalous','receipt':pin(self.run,self._paths(r['global_identity'],'gt')[1]),**records[r['global_identity']]} for r in rows]}
  atomic_json(self.run/'LABELS_COMPLETE.json',manifest);return manifest
 def label_record(self,object_id):return self._receipt(object_id,'gt')
 def read_mask(self,object_id):
  row=self.study.by_id[object_id];r=self._receipt(object_id,'gt');require(r['valid_pixel_GT'],'Invalid pixel GT; retain original object label')
  if r['normal_zero_mask']:return [np.zeros(hw,bool) for hw in row['expected_geometry']['real_view_hw']]
  p,_=self._paths(object_id,'gt');mask=decode_mask(p.read_bytes(),self.max_pixels)
  require(array_sha(mask.astype(np.uint8))==r['binary_decoded_sha256'],'Binary mask identity changed')
  return crop_views(mask,row['expected_geometry']['real_view_boxes'])
 def read_gt256(self,object_id):
  row=self.study.by_id[object_id];r=self._receipt(object_id,'gt');v=row['expected_geometry']['nviews']
  gt=np.zeros((v,256,256),bool)
  if r['valid_pixel_GT']:gt=np.stack([np.asarray(Image.fromarray(m.astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST))>0 for m in self.read_mask(object_id)])
  return {'maps':gt,'valid_pixel_GT':r['valid_pixel_GT'],'object_anomalous':row['original_label']=='anomalous',
   'native_positive_count':r.get('native_positive_count'),'processed_positive_count':int(gt.sum()) if r['valid_pixel_GT'] else None}

def family_dir(run,family,group):return Path(run)/'predictions'/family/group_slug(group)
def finalize_predictions(study,run,family,group,systems,rows,calibration_artifacts):
 """Shared sealed output contract. Every query and calibration payload is pinned."""
 run=Path(run).resolve();target=family_dir(run,family,group);expected=[r['global_identity'] for r in study.members(group,'new_confirmation')]
 require(family in study.expected_families,'Unknown study family')
 require(list(systems)==study.data['family_systems'][family],'Fixed family system set/order changed')
 require([r['global_identity'] for r in rows]==expected,'Complete fixed query order/coverage required')
 for r in rows:
  require(set(r['systems'])==set(systems),'Declared system identity missing')
  require(all(z['file'] in {p['path'] for p in r['outputs']} for z in r['systems'].values()),'System points to an unpinned payload')
  for p in r['outputs']:verify_pin(run,p)
 for p in calibration_artifacts:verify_pin(run,p)
 manifest={'schema':'fixed-prediction-manifest-v1','study_identity':study.identity,'family':family,'group':group,'systems':list(systems),
  'rows':rows,'calibration_artifacts':calibration_artifacts,'GT_read':False}
 atomic_json(target/'PREDICTION_MANIFEST.json',manifest)
 complete={'study_identity':study.identity,'family':family,'group':group,'systems':list(systems),'query_ids':expected,
  'prediction_manifest':pin(run,target/'PREDICTION_MANIFEST.json')}
 atomic_json(target/'COMPLETE.json',complete);return complete

def validate_family_predictions(study,run,family,group):
 run=Path(run).resolve();p=family_dir(run,family,group)/'COMPLETE.json';r=read_json(p)
 require(r['study_identity']==study.identity and r['family']==family and r['group']==group,'Completed prediction identity changed')
 require(r['query_ids']==[z['global_identity'] for z in study.members(group,'new_confirmation')],'Completed query set changed')
 m=read_json(verify_pin(run,r['prediction_manifest']));require(m['study_identity']==study.identity and m['GT_read'] is False,'Prediction manifest binding changed')
 require(m['family']==family and m['group']==group and m['systems']==study.data['family_systems'][family] and r['systems']==m['systems'],'Family systems changed')
 require([z['global_identity'] for z in m['rows']]==r['query_ids'],'Manifest coverage changed')
 for z in m['rows']:
  require(set(z['systems'])==set(m['systems']) and all(s['file'] in {p['path'] for p in z['outputs']} for s in z['systems'].values()),'Unpinned/missing system')
  for f in z['outputs']:verify_pin(run,f)
 for f in m['calibration_artifacts']:verify_pin(run,f)
 return r
def validate_all_predictions(study,run):
 return [validate_family_predictions(study,run,f,g) for f in study.expected_families for g in study.groups]
