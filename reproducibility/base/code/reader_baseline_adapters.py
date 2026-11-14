"""Portable native baseline adapters with explicit external resources.

The frozen numerical classes are retained; IO configuration is separate.
"""
from __future__ import annotations
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time
import hashlib
from typing import Iterable

import numpy as np
from PIL import Image

PC_COMMIT = 'fcaa92f124fb1ad74a7acf56726decd4b27cbcad'
AD_COMMIT = 'b9d1c2648e3a5247437d4d953d907a8f3d994457'
DINO_COMMIT = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
PC_ROOT = AD_ROOT = DINO_ROOT = None
WEIGHTS = {}

def configure_resources(study, resources, kind):
    global PC_ROOT, AD_ROOT, DINO_ROOT, WEIGHTS
    # Every resource is explicit; no discovery, downloads or home-cache fallback.
    if kind not in ('patchcore','anomalydino'): raise ValueError('Explicit baseline required')
    wp = resources.verify_model(study, kind)
    if kind=='patchcore': PC_ROOT=resources.upstream('patchcore')
    else: AD_ROOT=resources.upstream('anomalydino');DINO_ROOT=resources.upstream('dinov2')
    WEIGHTS[kind]=(wp,study.models['models'][kind]['weight_sha256'])


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def _module(name,path):
    s=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
    return m


def _images(x):
    a=np.asarray(x)
    if a.ndim!=4 or a.shape[-1]!=3 or a.dtype!=np.uint8:
        raise ValueError('Explicit uint8 [V,H,W,3] RGB required; no copied/fabricated views.')
    if min(a.shape[:3])<1: raise ValueError('Empty images')
    return a


def _sync(device):
    import torch
    if torch.device(device).type=='cuda': torch.cuda.synchronize(device)


class TorchExactNN:
    """Exact squared-L2 backend replacing only FAISS IndexFlatL2, no approximate index.

    A computational adaptation for native Windows GPU search. Native FAISS CPU is
    also supported for direct numeric parity. TF32 is disabled to avoid large
    cancellation errors. Reference and query chunks bound workspace size.
    """
    def __init__(self,device='cuda',q_chunk=256,r_chunk=16384):
        self.device=device;self.q_chunk=q_chunk;self.r_chunk=r_chunk;self.bank=None
    def fit(self,features):
        import torch
        self.bank=torch.as_tensor(np.ascontiguousarray(features,dtype=np.float32),device=self.device)
        self.norm=(self.bank*self.bank).sum(1)
    def run(self,n_nearest_neighbours,query_features,index_features=None):
        import torch
        if index_features is not None:
            tmp=TorchExactNN(self.device,self.q_chunk,self.r_chunk);tmp.fit(index_features)
            return tmp.run(n_nearest_neighbours,query_features)
        if self.bank is None:raise RuntimeError('Fit memory first')
        k=int(n_nearest_neighbours)
        if not 0<k<=len(self.bank): raise ValueError('Invalid k')
        q=np.ascontiguousarray(query_features,dtype=np.float32);ds=[];ii=[]
        with torch.inference_mode():
            for b in range(0,len(q),self.q_chunk):
                x=torch.as_tensor(q[b:b+self.q_chunk],device=self.device);xn=(x*x).sum(1,keepdim=True)
                best=torch.full((len(x),k),float('inf'),device=self.device); idx=torch.full((len(x),k),-1,dtype=torch.int64,device=self.device)
                for r in range(0,len(self.bank),self.r_chunk):
                    z=self.bank[r:r+self.r_chunk]
                    d=(xn+self.norm[r:r+len(z)][None,:]-2*x@z.T).clamp_min_(0)
                    v,j=torch.topk(d,min(k,len(z)),largest=False,sorted=True)
                    candidates=torch.cat([best,v],1);ids=torch.cat([idx,j+r],1)
                    best,order=torch.topk(candidates,k,largest=False,sorted=True);idx=torch.gather(ids,1,order)
                ds.append(best.cpu().numpy());ii.append(idx.cpu().numpy())
        return np.concatenate(ds),np.concatenate(ii)
    def reset_index(self):self.bank=None;self.norm=None


class PatchCoreOfficialWholeView:
    """Official WR50 PatchCore; 320 whole-view geometry is explicitly adapted.

    Native official 320 runs resize the short edge to366 then crop320. Here the
    complete rectangular view is explicitly resized to320x320 with aspect stretch
    so every original location is represented, then maps return to originalH/W.
    All feature, approximate greedy coreset, scoring and native sigma4 rules are
    inherited unchanged from the pinned author source. No rotation augmentation.
    """
    def __init__(self,device='cuda',image_size=320,coreset_fraction=.01,seed=0,
                 search_backend='faiss-cpu',faiss_threads=2):
        import torch
        import torchvision.models as tvm
        from torchvision import transforms
        if str(PC_ROOT/'src') not in sys.path:sys.path.insert(0,str(PC_ROOT/'src'))
        import patchcore.patchcore as pc
        import patchcore.sampler as sampler
        import patchcore.common as common
        wp,ws=WEIGHTS['patchcore'];assert _sha(wp)==ws
        torch.manual_seed(seed);np.random.seed(seed)
        self.device=device;self.image_size=int(image_size);self.seed=int(seed)
        self.search_backend=search_backend
        if torch.device(device).type=='cuda':
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        backbone=tvm.wide_resnet50_2(weights=None)
        state=torch.load(wp,map_location='cpu',weights_only=True)
        loaded=backbone.load_state_dict(state,strict=True);del state
        backbone.name='wideresnet50'
        self.transform=transforms.Compose([transforms.Resize((self.image_size,self.image_size)),transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        self.nn=(common.FaissNN(False,faiss_threads) if search_backend=='faiss-cpu'
                 else TorchExactNN(device))
        if search_backend not in ('faiss-cpu','torch-exact'):raise ValueError(search_backend)
        self.model=pc.PatchCore(torch.device(device))
        self.model.load(backbone=backbone,layers_to_extract_from=['layer2','layer3'],
            device=torch.device(device),input_shape=(3,self.image_size,self.image_size),
            pretrain_embed_dimension=1024,target_embed_dimension=1024,patchsize=3,
            patchstride=1,anomaly_score_num_nn=1,
            featuresampler=sampler.ApproximateGreedyCoresetSampler(coreset_fraction,torch.device(device),number_of_starting_points=10,dimension_to_project_features_to=128),
            nn_method=self.nn)
        self.protocol={'name':f'PatchCore-WR50-{self.image_size} whole-view adaptation','author_commit':PC_COMMIT,
            'weights_sha256':ws,'strict_load':str(loaded),'input_size':self.image_size,
            'geometry':f'whole original HxW resized to{self.image_size}x{self.image_size} with aspect stretch, no crop; map returned to original HxW',
            'layers':['layer2','layer3'],'embedding':1024,'patchsize':3,'knn':1,
            'coreset_fraction':coreset_fraction,'coreset':'official ApproximateGreedyCoresetSampler,128 projection,10 starting points',
            'reference_augmentation':'none','patch_score':'native FAISS squared L2, mean k=1',
            'image_score':'native maximum patch score; repository does not implement paper Eq7 reweight',
            'native_smoothing_sigma':4,'search_backend':search_backend,'seed':seed}
    def _prepare(self,views):
        import torch
        return torch.stack([self.transform(Image.fromarray(x)) for x in _images(views)])
    def fit(self,reference_objects:Iterable[np.ndarray],reference_ids:Iterable[str]):
        objects=list(reference_objects);ids=list(map(str,reference_ids))
        if len(objects)!=len(ids) or len(set(ids))!=len(ids):raise ValueError('Unique reference object identity required')
        # Native fit accepts an iterable of tensors; no DataLoader/worker side effects.
        _sync(self.device);t=time.perf_counter()
        def batches():
            for obj in objects:
                vv=_images(obj)
                for v in range(len(vv)):yield self._prepare(vv[v:v+1])
        self.model.fit(batches());_sync(self.device)
        self.reference_ids=ids
        nviews=sum(len(_images(o)) for o in objects)
        self.protocol.update(reference_ids=ids,reference_objects=len(ids),reference_images=nviews,
            candidate_patches=nviews*(self.image_size//8)**2,
            retained_patches=len(self.model.anomaly_scorer.detection_features),fit_seconds=time.perf_counter()-t)
        return dict(self.protocol)
    def predict(self,views):
        import cv2
        vv=_images(views);out=[];scores=[];raw=[]
        _sync(self.device);t=time.perf_counter();prep=0.;inference=0.
        for im in vv:
            s=time.perf_counter();x=self._prepare(im[None]);prep+=time.perf_counter()-s
            _sync(self.device);s=time.perf_counter();ss,mm=self.model.predict(x)
            _sync(self.device);inference+=time.perf_counter()-s
            native=np.asarray(mm[0],dtype=np.float32);raw.append(native)
            out.append(cv2.resize(native,(im.shape[1],im.shape[0]),interpolation=cv2.INTER_LINEAR));scores.append(float(ss[0]))
        return {'maps':np.stack(out),'native_maps':np.stack(raw),
            'view_scores':np.asarray(scores,dtype=np.float64),'object_score':max(scores),
            'timing':{'prepare_seconds':prep,'native_inference_seconds':inference,'request_seconds':time.perf_counter()-t},
            'geometry':{'input_hw':list(vv.shape[1:3]),'native_map_hw':[self.image_size,self.image_size],
                        'back_projection':'native square sigma4 map linearly resized to original HxW; full original extent, aspect-stretch adaptation'}}


class AnomalyDINOOfficialS:
    """Author S14/448 agnostic-no-mask pipeline, explicit multi-image container.

    Source objects are real view groups. Native matching pools all provided normal
    images and their8 image-space rotations. There is no invented cross-view joint
    detector. The returned object score is the declared max of native image scores.
    """
    def __init__(self,device='cuda',resolution=448,search_backend='faiss-cpu',faiss_threads=2):
        import torch
        from torchvision import transforms
        os.environ.setdefault('XFORMERS_DISABLED','1')
        if str(DINO_ROOT) not in sys.path:sys.path.insert(0,str(DINO_ROOT))
        from dinov2.hub.backbones import dinov2_vits14
        self.upstream_backbones=_module('upstream_anomalydino_backbones',AD_ROOT/'src/backbones.py')
        self.upstream_utils=_module('upstream_anomalydino_utils',AD_ROOT/'src/utils.py')
        wp,ws=WEIGHTS['anomalydino'];assert _sha(wp)==ws
        self.device=device;self.resolution=int(resolution);self.search_backend=search_backend
        if search_backend not in ('faiss-cpu','torch-exact'):raise ValueError(search_backend)
        if torch.device(device).type=='cuda':
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=dinov2_vits14(pretrained=False).eval()
        state=torch.load(wp,map_location='cpu',weights_only=True)
        loaded=model.load_state_dict(state,strict=True);del state
        self.wrapper=self.upstream_backbones.DINOv2Wrapper.__new__(self.upstream_backbones.DINOv2Wrapper)
        self.wrapper.model=model.to(device);self.wrapper.device=device
        self.wrapper.smaller_edge_size=self.resolution;self.wrapper.half_precision=False
        self.wrapper.model_name='dinov2_vits14'
        # Identical transform to author DINOv2Wrapper.load_model; local loader removes torch.hub's unpinned fetch.
        self.wrapper.transform=transforms.Compose([
            transforms.Resize(size=self.resolution,interpolation=transforms.InterpolationMode.BICUBIC,antialias=True),
            transforms.ToTensor(),transforms.Normalize((.485,.456,.406),(.229,.224,.225))])
        if search_backend=='faiss-cpu':
            import faiss
            faiss.omp_set_num_threads(faiss_threads);self.nn=None
        else:self.nn=TorchExactNN(device)
        self.protocol={'name':f'AnomalyDINO-S{self.resolution} agnostic-no-mask view-group adaptation',
            'author_commit':AD_COMMIT,'dinov2_code_commit':DINO_COMMIT,'weights_sha256':ws,
            'strict_load':str(loaded),'parameters':sum(p.numel() for p in model.parameters()),
            'register_tokens':model.num_register_tokens,'short_edge_resolution':self.resolution,
            'features':'author get_intermediate_layers(x)[0], last normalized patch tokens,384D',
            'geometry':'native short-edge448, preserve aspect, crop trailing pixels to patch14 multiple; author map helper resizes patch grid to original HxW',
            'reference_rotations_degrees':[0,45,90,135,180,225,270,315],
            'rotation_backend':'author cv2 warpAffine INTER_LINEAR BORDER_DEFAULT',
            'masking':False,'masking_reason':'official fallback for unknown/custom categories is agnostic_no_mask',
            'coreset':None,'metric':'unit-L2 vectors; squared L2/2 = cosine distance;1NN',
            'image_score':'mean floor(0.01*num_patches) largest raw patch distances; maximum if floor=0',
            'object_score':'maximum native image score across supplied real views (evaluation container adaptation)',
            'map':'author dists2map: bilinear to original image shape then scipy gaussian_filter sigma4',
            'search_backend':search_backend}
    def _features(self,image):
        t,g=self.wrapper.prepare_image(image)
        f=self.wrapper.extract_features(t).astype(np.float32)
        return f,g
    @staticmethod
    def _normalize(f):
        # Equivalent to author faiss.normalize_L2, retaining zero vectors as zero.
        return np.ascontiguousarray(f/np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-30),dtype=np.float32)
    def fit(self,reference_objects:Iterable[np.ndarray],reference_ids:Iterable[str]):
        objects=list(reference_objects);ids=list(map(str,reference_ids))
        if len(objects)!=len(ids) or len(set(ids))!=len(ids):raise ValueError('Unique reference object identity required')
        _sync(self.device);t=time.perf_counter();features=[]
        for obj in objects:
            for im in _images(obj):
                for augmented in self.upstream_utils.augment_image(im):
                    f,g=self._features(augmented);features.append(f)
        f=np.concatenate(features).astype(np.float32)
        if self.search_backend=='faiss-cpu':
            import faiss
            faiss.normalize_L2(f);self.nn=faiss.IndexFlatL2(f.shape[1]);self.nn.add(f)
        else:self.nn.fit(self._normalize(f))
        _sync(self.device)
        self.reference_ids=ids;self.protocol.update(reference_ids=ids,reference_objects=len(ids),
            reference_images=sum(len(_images(o)) for o in objects),
            augmented_reference_images=len(features),retained_patches=len(f),feature_dim=f.shape[1],
            fit_seconds=time.perf_counter()-t)
        return dict(self.protocol)
    def predict(self,views):
        vv=_images(views);maps=[];patches=[];scores=[]
        _sync(self.device);t=time.perf_counter();feature_s=0.;search_s=0.;map_s=0.
        for im in vv:
            _sync(self.device);s=time.perf_counter();f,g=self._features(im)
            _sync(self.device);feature_s+=time.perf_counter()-s
            s=time.perf_counter()
            if self.search_backend=='faiss-cpu':
                import faiss
                faiss.normalize_L2(f);d,ids=self.nn.search(f,1)
            else:d,ids=self.nn.run(1,self._normalize(f))
            distances=d[:,0]/2
            # Exactly the author mean_top1p formula; no change to use smoothed maxima.
            k=int(len(distances)*.01)
            score=float(np.max(distances) if k==0 else np.mean(np.sort(distances)[::-1][:k]))
            _sync(self.device);search_s+=time.perf_counter()-s
            s=time.perf_counter();p=distances.reshape(g)
            maps.append(self.upstream_utils.dists2map(p,im.shape).astype(np.float32));patches.append(p);scores.append(score)
            map_s+=time.perf_counter()-s
        return {'maps':np.stack(maps),'patch_maps':np.stack(patches),
            'view_scores':np.asarray(scores,dtype=np.float64),'object_score':max(scores),
            'timing':{'feature_seconds':feature_s,'search_seconds':search_s,'map_seconds':map_s,'request_seconds':time.perf_counter()-t},
            'geometry':{'input_hw':list(vv.shape[1:3]),'patch_hw':list(patches[0].shape),
                        'preprocessed_resized_hw':list(self.wrapper.transform(Image.fromarray(vv[0])).shape[-2:]),
                        'patch_crop_hw':[int(v)*14 for v in patches[0].shape],
                        'back_projection':'native author bilinear directly from cropped token grid to original HxW; sigma4 original pixels; trailing resize-pixel crop retained as author convention'}}


def source_identities():
    """No model load, data discovery or GPU work."""
    return {'patchcore_commit':PC_COMMIT,'anomalydino_commit':AD_COMMIT,
            'dinov2_commit':DINO_COMMIT,'adapter_sha256':_sha(__file__),
            'weights':{k:{'path':str(p),'sha256':_sha(p),'expected_sha256':s} for k,(p,s) in WEIGHTS.items()}}
