"""Portable frozen DINO-S14 square448 C4 instrument.

Project-authored numerical body adapted from dino_c4_response.py, source SHA256
25018a18695d3e8b2d518bdb22f88353df4ccd8c560a305ee1ca4d894ea679ec.
Only explicit upstream/weight resource resolution replaces machine-specific locations.
DINOv2 upstream copyright/license remains with its separately supplied source.
No model, downloads, image discovery, labels or GPU work occurs on import.
"""
from __future__ import annotations
from pathlib import Path
import importlib,sys,os
import numpy as np
from PIL import Image
import torch
from reader_common import require

SIDE,PATCH,GRID,DIM,ROTATIONS=448,14,32,384,4
WEIGHT_SHA256='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
DINO_COMMIT='7764ea0f912e53c92e82eb78a2a1631e92725fc8'

def square_views(views):
    a=np.asarray(views)
    require(a.ndim==4 and a.shape[-1]==3 and a.dtype==np.uint8 and min(a.shape[:3])>0,'Explicit uint8[V,H,W,3] views required')
    return np.stack([np.asarray(Image.fromarray(im).resize((SIDE,SIDE),Image.Resampling.BICUBIC)) for im in a]).astype(np.uint8,copy=False)

def normalize_components(features,zero_policy='error'):
    a=np.asarray(features)
    require(a.shape[-1]==DIM and np.isfinite(a).all(),'Finite384D components required')
    require(zero_policy in ('error','keep'),'Explicit zero policy required')
    x=np.asarray(a,dtype=np.float32);norm=np.sqrt(np.square(x.astype(np.float64)).sum(-1,keepdims=True));zero=norm[...,0]==0
    if zero.any() and zero_policy=='error':raise ValueError(f'{int(zero.sum())} exactzero components; cosine undefined')
    out=np.divide(x.astype(np.float64),norm,out=np.zeros_like(norm*x,dtype=np.float64),where=norm>0).astype(np.float32)
    return np.ascontiguousarray(out),int(zero.sum())

def load_fixed_model(study,resources,device='cuda'):
    weights=resources.verify_model(study,'dino-response');root=resources.upstream('dinov2').resolve()
    rec=study.models['models']['dino-response']
    require(rec['weight_sha256']==WEIGHT_SHA256,'Frozen no-register S14 weights required')
    require((root/'dinov2/hub/backbones.py').is_file(),'Explicit DINOv2 checkout must contain dinov2/hub/backbones.py')
    if 'dinov2' in sys.modules:
        require(Path(sys.modules['dinov2'].__file__).resolve().is_relative_to(root),'Another DINOv2 tree already imported; use an isolated process')
    os.environ.setdefault('XFORMERS_DISABLED','1')
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    hub=importlib.import_module('dinov2.hub.backbones')
    require(Path(hub.__file__).resolve().is_relative_to(root),'DINOv2 import escaped explicit source tree')
    model=hub.dinov2_vits14(pretrained=False)
    state=torch.load(weights,map_location='cpu',weights_only=True);status=model.load_state_dict(state,strict=True);del state
    require(model.num_register_tokens==0 and model.patch_size==PATCH and model.embed_dim==DIM,'Frozen DINOv2 ViT-S/14 geometry required')
    model=model.to(device=device,dtype=torch.float32).eval().requires_grad_(False)
    return model,dict(code_commit=DINO_COMMIT,weight_sha256=WEIGHT_SHA256,strict_load=str(status),model='DINOv2 ViT-S/14 no-register FP32')

class DinoC4Encoder:
    def __init__(self,device='cuda',*,model=None,study=None,resources=None,batch_size=1,zero_policy='error'):
        self.device=torch.device(device);self.batch_size=int(batch_size);self.zero_policy=zero_policy
        require(self.batch_size==1 and self.zero_policy=='error','Frozen batch1/zero-error instrument required')
        if model is None:
            require(study is not None and resources is not None,'Explicit study and resource paths required')
            model,self.identity=load_fixed_model(study,resources,device)
        else:
            self.identity={'model':'caller-supplied synthetic test, not official efficacy evidence'}
            model=model.to(device=device,dtype=torch.float32).eval().requires_grad_(False)
        self.model=model;self.last_encoding=None

    @torch.inference_mode()
    def encode_views(self,views):
        # The following numerical body is identical to the frozen encoder.
        square=square_views(views);output=np.empty((len(square),GRID,GRID,ROTATIONS,DIM),np.float32)
        mean=torch.tensor([.485,.456,.406],dtype=torch.float32,device=self.device)[None,:,None,None]
        std=torch.tensor([.229,.224,.225],dtype=torch.float32,device=self.device)[None,:,None,None]
        zero_counts=[]
        for k in range(ROTATIONS):
            rotated=np.rot90(square,k,axes=(1,2)).copy();blocks=[]
            for start in range(0,len(square),self.batch_size):
                x=torch.from_numpy(rotated[start:start+self.batch_size].transpose(0,3,1,2).copy()).to(self.device).float()/255
                x=(x-mean)/std
                with torch.autocast(device_type=self.device.type,enabled=False):
                    token=self.model.get_intermediate_layers(x,n=1,reshape=False,return_class_token=False,norm=True)[0]
                require(tuple(token.shape)==(len(x),GRID*GRID,DIM) and token.dtype==torch.float32,'Frozen token geometry/dtype changed')
                blocks.append(token.cpu().numpy().reshape(len(x),GRID,GRID,DIM))
            block=np.concatenate(blocks,axis=0);aligned=np.rot90(block,-k,axes=(1,2)).copy()
            component,zeros=normalize_components(aligned,self.zero_policy);output[...,k,:]=component;zero_counts.append(zeros)
        self.last_encoding={'zero_token_counts_by_rotation':zero_counts,'input_views':len(square),'shape':list(output.shape)}
        return output
