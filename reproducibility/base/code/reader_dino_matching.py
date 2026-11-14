"""Portable frozen six-operation DINO matching and field construction.

Adapted from frozen prospective_matching.py (8ce73fd710e133112130d6416cb86cfd6d796aa4e7ff84fc7762f10cd89ab77f)
and prospective_dino_worker.py (3a22538724009898972445f6342521a259fb5eb49523ad2a2f1dbc497064105e).
No source/query selection, fitting, labels, filesystem or implicit paths.
"""
from __future__ import annotations
import gc
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from reader_common import require
from reader_search import HostOriginalSearch,HostOriginalResponseBank

ARMS=('single','normalaug','independent','cyclic','meanfeature','meanfeature_unit')
SYSTEMS=tuple('dino_'+x for x in ARMS)

def normalized(a):
    z=np.asarray(a).astype(np.float64);norm=np.sqrt(np.square(z).sum(-1,keepdims=True))
    require(not (norm==0).any(),'Zero pooled DINO vector; no epsilon')
    return (z/norm).astype(np.float32)

def match_response_slot(bank,queries,device='cuda',progress=None):
    b=np.asarray(bank);q=np.asarray(queries)
    require(b.ndim==5 and q.ndim==5 and b.shape[1:]==q.shape[1:] and b.shape[-2:]==(4,384),'Explicit[R/Q,H,W,4,384] required')
    require(len(b)>0 and len(q)>0 and b.dtype==np.float32 and q.dtype==np.float32,'Nonempty FP32 source/query required')
    r,h,w,_,_=b.shape;n=len(q);flatb=np.ascontiguousarray(b).reshape(r*h*w,4,384);flatq=np.ascontiguousarray(q).reshape(n,h*w,4,384)
    fields={k:np.empty((n,h,w),np.float32) for k in ARMS};witness={}
    for group in ['single','independent','cyclic','meanfeature','meanfeature_unit']:
        if group=='cyclic':engine=HostOriginalResponseBank(flatb,device)
        else:
            refs=flatb[:,0] if group=='single' else flatb.reshape(-1,384) if group=='independent' else flatb.mean(1,dtype=np.float64).astype(np.float32)
            if group=='meanfeature_unit':refs=normalized(refs)
            engine=HostOriginalSearch(refs,device)
        for i,z in enumerate(flatq):
            if group=='single':
                value,ids=engine(z[:,0]);fields['single'][i]=value.reshape(h,w)
                if i==0:witness['single_reference']=np.empty((n,h,w),np.int64)
                witness['single_reference'][i]=ids.reshape(h,w)
            elif group=='independent':
                value,ids=engine(z.reshape(-1,384));value=value.reshape(-1,4)
                fields['normalaug'][i]=value[:,0].reshape(h,w);fields['independent'][i]=value.mean(1,dtype=np.float64).astype(np.float32).reshape(h,w)
                if i==0:witness['independent_donors']=np.empty((n,h,w,4),np.int32)
                witness['independent_donors'][i]=ids.reshape(h,w,4)
            elif group=='cyclic':
                ans=engine(z);fields['cyclic'][i]=ans['cyclic_cost'].reshape(h,w)
                if i==0:
                    witness['cyclic_reference']=np.empty((n,h,w),np.int32);witness['cyclic_shift']=np.empty((n,h,w),np.uint8)
                witness['cyclic_reference'][i]=ans['cyclic_reference'].reshape(h,w);witness['cyclic_shift'][i]=ans['cyclic_shift'].reshape(h,w)
            else:
                query=z.mean(1,dtype=np.float64).astype(np.float32)
                if group=='meanfeature_unit':query=normalized(query)
                fields[group][i]=engine(query)[0].reshape(h,w)
            if progress is not None and (i%16==0 or i==n-1):progress({'operation':group,'objects_complete':i+1,'objects_total':n})
        del engine;gc.collect()
        if torch.device(device).type=='cuda':torch.cuda.empty_cache()
    for k in fields:fields[k]*=np.float32(192)
    return fields,witness

def final_fields(raw):
    raw=np.asarray(raw);require(raw.ndim==3 and raw.shape[1:]==(32,32) and raw.dtype==np.float32 and np.isfinite(raw).all(),'Finite[V,32,32] FP32 field required')
    full=torch.nn.functional.interpolate(torch.from_numpy(raw[:,None]),(256,256),mode='bilinear',align_corners=False).numpy()[:,0]
    for v in range(len(full)):full[v]=gaussian_filter(full[v],sigma=4)
    scores=np.sort(raw.reshape(len(raw),-1),axis=-1)[:,-10:].mean(-1,dtype=np.float64)
    return {'raw32':raw,'maps256':full,'view_scores':scores,'object_score':np.array(scores.max(),np.float64)}

def healthy_thresholds(fields):
    require(bool(fields),'Nonempty fixed normal-calibration fields required')
    return {arm:{'map_quantile99':float(np.quantile(np.stack([z[arm+'_maps256'] for z in fields]),.99,method='higher')),
        'native_object_quantile99':float(np.quantile([float(z[arm+'_object_score']) for z in fields],.99,method='higher')),
        'decision':'score > threshold; nominal .99 calibration quantile, not a population false-alarm guarantee'} for arm in ARMS}
