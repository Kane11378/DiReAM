"""Evaluate actual raw predictions and original GT; expected CSVs are not inputs.

One category/method per call, then equal-category scalar aggregation. Native
fields retain their own geometry; common256 fields serve within-unit/budget
comparisons. Test GT is used only here, never by fitting or prediction.
The same evaluation accepts AnomalyDINO_F_norot and AnomalyDINO_U_pool_norot;
their thresholds are recomputed from their own sealed calibration scores.
"""
from pathlib import Path
import sys,argparse,csv,io,math
from contextlib import nullcontext
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parent))
from complete_methods import read,dump,sha,ROOT
def table(p,rr):
 keys=list(dict.fromkeys(k for r in rr for k in r))
 with Path(p).open('w',encoding='utf8',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rr)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-reader',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--category',required=True);p.add_argument('--method',required=True);p.add_argument('--predictions',type=Path,required=True);m=p.add_mutually_exclusive_group(required=True);m.add_argument('--visa-tar',type=Path);m.add_argument('--mask-source',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 sys.path.insert(0,str(a.base_reader/'code'));from reader_statistics_core import binary_metrics
 spec=read(a.manifest)
 if a.visa_tar:assert sha(a.visa_tar)==spec['archive_sha256']
 g=spec['groups'][a.category];sealed=read(a.predictions/'COMPLETE.json');rr=g['calibration']+g['query'];assert len(rr)==len(sealed['records']);arrays=[]
 if a.method in ['AnomalyDINO_F_norot','AnomalyDINO_U_pool_norot']:assert sealed['method']==a.method and sealed['manifest_sha256']==sha(a.manifest),'Predictions must match the no-rotation method and manifest'
 for row,rec in zip(rr,sealed['records']):
  path=a.predictions/rec['file'];assert row['global_identity']==rec['global_identity'] and sha(path)==rec['sha256']
  with np.load(path,allow_pickle=False) as z:arrays.append({k:z[k].copy() for k in z.files})
 # Metadata labels and pixels enter only after all predictions are sealed.
 gt=[];native=[];bad=[]
 with (a.visa_tar.open('rb') if a.visa_tar else nullcontext(None)) as archive:
  for row in g['query']:
   w,h=row['composite_wh'];positive=row['label_stratum']=='anomaly';bad.append(positive)
   if positive:
    if archive is not None:
     offsets=('gt_offset_data_uncompressed' in row,'gt_member_bytes' in row)
     assert offsets[0]==offsets[1],'Partial GT offset metadata is invalid'
     if offsets[0]:
      archive.seek(row['gt_offset_data_uncompressed']);b=archive.read(row['gt_member_bytes']);assert len(b)==row['gt_member_bytes']
     else:
      # The common-task fixture supplies the exact original member name.
      # Resolve it only now, after full prediction sealing and archive SHA.
      import tarfile
      archive.seek(0)
      with tarfile.open(fileobj=archive,mode='r:') as tf:
       member=tf.getmember(row['gt_member']);assert member.isfile()
       b=tf.extractfile(member).read();assert len(b)==member.size
    else:
     path=(a.mask_source/row['gt_member']).resolve();assert path.is_relative_to(a.mask_source.resolve()) and sha(path)==row['GT_payload_sha256'];b=path.read_bytes()
    with Image.open(io.BytesIO(b)) as im:x=np.asarray(im)
    x=np.any(x>0,axis=2) if x.ndim==3 else x>0
   else:x=np.zeros((h,w),bool)
   assert x.shape==(h,w);views=[x[y0:y1,x0:x1] for x0,y0,x1,y1 in row['real_view_boxes']];native.append(views)
   gt.append(np.stack([np.asarray(Image.fromarray(v.astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST))>0 for v in views]))
 gt=np.stack(gt);bad=np.asarray(bad,bool);ncal=len(g['calibration']);out=[];units=[]
 profiles=[k.removesuffix('_maps256') for k in arrays[0] if k.endswith('_maps256')] if a.method=='DiReAM' else [a.method]
 for method in profiles:
  fieldkey=method+'_maps256' if a.method=='DiReAM' else 'maps256';scorekey=method+'_object_score' if a.method=='DiReAM' else 'object_score'
  fields=np.stack([z[fieldkey] for z in arrays]);scores=np.array([float(z[scorekey]) for z in arrays]);threshold=float(max(scores[:ncal]));alarm=scores[ncal:]>threshold;query=fields[ncal:];ur=[]
  for i,row in enumerate(g['query']):
   m=binary_metrics(gt[i],query[i]) if bad[i] and gt[i].any() else {};u=dict(category=a.category,method=method,global_identity=row['global_identity'],object_anomalous=bool(bad[i]),pixel_AP_eligible=bool(bad[i] and gt[i].any()),object_pixel_AP=m.get('AP'),native_alarm=int(alarm[i]),native_object_score=float(scores[ncal+i]),native_object_threshold=threshold)
   if m:
    order=np.argsort(-query[i].reshape(-1),kind='stable');yy=gt[i].ravel();den=int(yy.sum())
    for b in [.01,.05]:
     c=float(yy[order[:math.floor(len(yy)*b)]].sum()/den);u[f'coverage_{b:g}']=c;u[f'gated_coverage_{b:g}']=c*int(alarm[i])
   ur.append(u)
  pool=binary_metrics(gt,query);im=binary_metrics(bad,scores[ncal:]);r=dict(category=a.category,method=method,queries=len(bad),mean_object_pixel_AP=float(np.mean([u['object_pixel_AP'] for u in ur if u['object_pixel_AP'] is not None])),pooled_pixel_AP=pool['AP'],pooled_pixel_AUROC=pool['AUROC'],object_AP=im['AP'],object_AUROC=im['AUROC'],sensitivity=float(alarm[bad].mean()),false_alarm_rate=float(alarm[~bad].mean()),true_alarms=int(alarm[bad].sum()),false_alarms=int(alarm[~bad].sum()))
  for key in ['coverage_0.01','coverage_0.05','gated_coverage_0.01','gated_coverage_0.05']:r[key]=float(np.mean([u[key] for u in ur if key in u]))
  if a.method!='DiReAM':
   labels=[];maps=[]
   for i,z in enumerate(arrays[ncal:]):
    for v,yy in enumerate(native[i]):
     if a.method=='MH_PatchCore':
      import torch
      from torchvision import tv_tensors
      sys.path.insert(0,str(ROOT/'third_party/MH_PatchCore'));from mhpc.core.plugins.dataloader.mvtec_dataloader_augment.dataset_runtime import build_transforms
      transform=build_transforms(resize=(256,256),img_size=(224,224),augment=False,cfg=None,dtype=torch.float32)
      _,yy=transform(torch.zeros((3,*yy.shape),dtype=torch.uint8),tv_tensors.Mask(torch.from_numpy(yy.astype(np.uint8))));yy=np.asarray(yy)>0
     elif a.method.startswith('SubspaceAD'):yy=np.asarray(Image.fromarray(yy).resize((672,672),Image.Resampling.NEAREST))
     labels.append(yy.ravel());maps.append(z['native_maps'][v].ravel())
   mm=binary_metrics(np.concatenate(labels),np.concatenate(maps));r.update(native_pixel_AP=mm['AP'],native_pixel_AUROC=mm['AUROC'])
  out.append(r);units+=ur
 a.output.mkdir(parents=True,exist_ok=False);table(a.output/'SUMMARY.csv',out);table(a.output/'PER_UNIT.csv',units);dump(a.output/'COMPLETE.json',dict(status='ACTUAL_FIELD_AND_GT_EVALUATION_COMPLETE',predictions_sha256=sha(a.predictions/'COMPLETE.json'),manifest_sha256=sha(a.manifest),expected_tables_used_as_inputs=False))
if __name__=='__main__':main()
