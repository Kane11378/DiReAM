"""Complete observed PDN geometry, healthy scale and physical map projection.

Portability adaptation: pure functions selected without any local-root imports.
"""
import numpy as np
import torch
from torch.nn import functional as F
from PIL import Image

@torch.inference_mode()
def encode(model, views, retain_directions=True):
    """Fine[V,56,56,D], response[V,48,48,K,D], K=4 or1, no extrapolation."""
    a=np.asarray(views)
    if a.ndim!=4 or a.shape[1:]!=(256,256,3) or a.dtype!=np.uint8 or not len(a):
        raise ValueError('Use actual prepared 256-square views')
    v=len(a);device=next(model.parameters()).device
    def read(b):
        t=torch.from_numpy(b.transpose(0,3,1,2).copy()).to(device).float()/255
        return model(t).cpu().numpy().transpose(0,2,3,1)
    fine=read(a)
    if fine.shape!=(v,56,56,384):raise ValueError('Descriptor geometry changed')
    response=np.empty((v,48,48,4 if retain_directions else 1,384),np.float32)
    # Direction zero follows the established full256 half-resolution interlace.
    half=a[:,::2,::2]
    for py in range(2):
        for px in range(2):response[:,py::2,px::2,0]=read(half[:,2*py:,2*px:])
    for k in range(1,response.shape[3]):
        rotated=np.rot90(a[:,:253,:253],k,axes=(1,2)).copy()
        half=rotated[:,::2,::2];z=np.empty((v,48,48,384),np.float32)
        for py in range(2):
            for px in range(2):z[:,py::2,px::2]=read(half[:,2*py:,2*px:])
        response[:,:,:,k]=np.rot90(z,-k,axes=(1,2))
    if not np.isfinite(fine).all() or not np.isfinite(response).all():raise ValueError('Nonfinite descriptor')
    return fine,response


def physical_map256(field):
    """Project anchors centred at16+4j; bilinear interior, nearest-border extension.

    Plain56→256 interpolation would move the first centre to~1.79, a different
    geometry. Here grid_sample uses explicit image coordinates and border values.
    This projection is new standard full-map evaluation, not the old56 endpoint.
    """
    a=np.asarray(field,dtype=np.float32)
    if a.shape[-2:]!=(56,56):raise ValueError('56-square physical anchor grid required')
    axis=(torch.arange(256,dtype=torch.float32)-16)/4
    normalized=2*axis/55-1
    gy,gx=torch.meshgrid(normalized,normalized,indexing='ij')
    grid=torch.stack([gx,gy],-1)[None].expand(int(np.prod(a.shape[:-2])),256,256,2)
    with torch.inference_mode():
        result=F.grid_sample(torch.from_numpy(a.reshape(-1,1,56,56).copy()),grid,
                             mode='bilinear',padding_mode='border',align_corners=True)
    return result.numpy().reshape(*a.shape[:-2],256,256)


def native_object_score(field):
    """Established per-real-view top31 anchor mean, maximum actual view."""
    a=np.asarray(field,dtype=np.float64)
    if a.ndim!=3 or a.shape[-2:]!=(56,56):raise ValueError('One actual view group required')
    return float(np.partition(a.reshape(len(a),-1),3136-31,axis=-1)[:,-31:].mean(-1).max())


def fit_region_scale(healthy_fields):
    """Robust healthy-only affine A→fallback scale; identical rule for all readers.

    All supplied objects are caller-declared calibration normals. No anomaly
    labels or outcome search enter this function. Degenerate spread is explicit.
    """
    a=np.asarray(healthy_fields,dtype=np.float32)
    if a.ndim!=4 or a.shape[-2:]!=(56,56) or not np.isfinite(a).all():raise ValueError('Healthy full anchor fields required')
    mask=np.zeros((56,56),bool);mask[4:52,4:52]=True
    qa=np.quantile(a[...,mask],[.5,.95]);qf=np.quantile(a[...,~mask],[.5,.95])
    if qa[1]<=qa[0] or qf[1]<=qf[0]:raise ValueError('Degenerate healthy region spread; no silent fallback')
    scale=float((qf[1]-qf[0])/(qa[1]-qa[0]));offset=float(qf[0]-scale*qa[0])
    return {'rule':'healthy pooled median/p95 affine A-to-common-fallback','A_q50_q95':qa.tolist(),
            'fallback_q50_q95':qf.tolist(),'scale':scale,'offset':offset,'normal_objects':len(a)}


def apply_region_scale(field,parameters):
    a=np.array(field,dtype=np.float32,copy=True)
    a[...,4:52,4:52]=(a[...,4:52,4:52].astype(np.float64)*parameters['scale']+parameters['offset']).astype(np.float32)
    return a


