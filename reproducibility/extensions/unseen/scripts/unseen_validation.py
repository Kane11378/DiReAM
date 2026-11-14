"""Prospectively fixed MANTA same-source, new-category confirmation.

Run through run_bound_raw.py with explicit resource bindings, saved protocol
and a separate output directory. Inputs stay read-only. No GPU is used by
freeze, stage-rgb or contracts. Annotation pixels are opened only after every
prediction is sealed.
"""
from __future__ import annotations
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('XFORMERS_DISABLED', '1')
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '4')
import argparse, csv, gc, hashlib, io, json, math, re, subprocess, sys, tarfile, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
sys.dont_write_bytecode = True
SCRIPT = Path(__file__).resolve()
EXTENSION = SCRIPT.parents[1]
ROOT = EXTENSION/'resources'
OUT = EXTENSION/'records'
INV = ROOT/'provenance/category_inventory'
ARCHIVES = ROOT/'provenance/archives'
STUDY = ROOT/'study'
READER = ROOT/'reference_reader'
CODE = READER/'code'
BASECODE = READER/'base/code'
PC = STUDY/'baseline_sources/patchcore/patchcore-inspection-fcaa92f124fb1ad74a7acf56726decd4b27cbcad'
CATEGORIES = ('soybean', 'terminal')
PROFILES = ('N_W_U', 'I_W_U', 'M_raw_U', 'M_W_U', 'M_ordinary05_U')
PAIRS = (('I_W_U', 'N_W_U'), ('M_W_U', 'I_W_U'),
         ('M_W_U', 'M_raw_U'), ('M_W_U', 'M_ordinary05_U'))
NREF, NCAL, DIM, GRID, VIEWS = 176, 64, 384, 32, 5
BUDGET = int(NREF*GRID*GRID*.04)

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path): return json.loads(Path(path).read_text('utf-8-sig'))
def now(): return datetime.now(timezone.utc).isoformat()
def write(path, value, *, replace=False):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        if read(path) != value: raise ValueError('Existing record differs: '+str(path))
        return
    tmp=path.with_suffix(path.suffix+'.pending')
    with tmp.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)

def csvwrite(path, rows):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys=list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream, fieldnames=keys); writer.writeheader(); writer.writerows(rows)


def frozen():
    f=read(OUT/'FROZEN.json')
    for record in f['source_and_code_pins']:
        if sha(ROOT/record['path']) != record['sha256']:
            raise ValueError('Changed frozen input/code: '+record['path'])
    if sha(OUT/'MEMBERSHIP.json') != f['membership_sha256']:
        raise ValueError('Changed frozen membership')
    return f


def stage_rgb():
    f=frozen(); rows=read(OUT/'MEMBERSHIP.json')['rows']; dest=OUT/'rgb'
    receipt=OUT/'RGB_COMPLETE.json'
    if receipt.exists():
        r=read(receipt)
        for row in r['rows']:
            if sha(OUT/row['local_rgb'])!=row['payload_sha256']: raise ValueError('Changed staged RGB')
        return
    from PIL import Image
    import numpy as np
    wanted={(row['archive_file'],row['rgb_member']):row for row in rows}; staged={}
    for archive in sorted({row['archive_file'] for row in rows}):
        expected=next(row['archive_sha256'] for row in rows if row['archive_file']==archive)
        if sha(ARCHIVES/archive)!=expected: raise ValueError('Archive SHA mismatch')
        with tarfile.open(ARCHIVES/archive,'r|gz') as stream:
            for member in stream:
                key=(archive,member.name)
                if key not in wanted: continue
                row=wanted[key]; pp=PurePosixPath(member.name)
                if not member.isfile() or member.size!=row['rgb_member_bytes'] or pp.is_absolute() or '..' in pp.parts or ':' in member.name or '\\' in member.name:
                    raise ValueError('Unsafe/mismatched selected member')
                path=dest.joinpath(*pp.parts)
                if not path.resolve().is_relative_to(dest.resolve()): raise ValueError('Path escape')
                payload=stream.extractfile(member).read()
                if len(payload)!=member.size: raise ValueError('Short selected RGB')
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists() and path.read_bytes()!=payload: raise ValueError('Existing RGB differs')
                if not path.exists(): path.write_bytes(payload)
                with Image.open(io.BytesIO(payload)) as image:
                    rgb=np.asarray(image.convert('RGB'))
                if rgb.shape!=(256,1280,3): raise ValueError('Unexpected Tiny five-slot layout')
                digest=hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest()
                staged[key]={**row,'local_rgb':str(path.relative_to(OUT)),
                             'payload_sha256':hashlib.sha256(payload).hexdigest(),'decoded_sha256':digest,
                             'native_composite_HW':[256,1280]}
        print(json.dumps({'stage':'stage_rgb','archive':archive,'staged':len(staged),'GT_payload_reads':0}),flush=True)
    if len(staged)!=len(rows): raise ValueError('Missing selected RGB')
    ordered=[staged[(row['archive_file'],row['rgb_member'])] for row in rows]
    duplicates=[digest for digest,n in Counter(row['decoded_sha256'] for row in ordered).items() if n>1]
    if duplicates:
        write(OUT/'DUPLICATE_STOP.json',{'decoded_duplicates':duplicates,'replacement':False})
        raise ValueError('Selected units contain exact decoded duplicates; frozen no replacement')
    write(receipt,{'status':'COMPLETE_RGB_ONLY','utc':now(),'freeze_sha256':sha(OUT/'FROZEN.json'),
                   'RGB_units':len(ordered),'GT_payload_reads':0,'rows':ordered})

def configure_gpu():
    # Configure deterministic CUDA execution for the explicitly bound device.
    # Resource checks and the caller serialize access before this function runs.
    import torch
    if not torch.cuda.is_available(): raise ValueError('CUDA unavailable; no silent CPU scientific substitution')
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False; torch.use_deterministic_algorithms(True)
    return torch

def encode(category):
    frozen(); torch=configure_gpu(); import numpy as np
    from PIL import Image
    sys.path.insert(0,str(STUDY/'operators'))
    from dino_c4_response import DinoC4Encoder
    rows=[row for row in read(OUT/'RGB_COMPLETE.json')['rows'] if row['category']==category]
    if not rows: raise ValueError('Stage all selected RGB first')
    enc=DinoC4Encoder('cuda',batch_size=1); dest=OUT/'encoded'/category; dest.mkdir(parents=True,exist_ok=True)
    write(dest/'ENCODER_PROTOCOL.json',enc.protocol)
    for role in ('reference','calibration','query'):
        rr=[row for row in rows if row['role']==role]; path=dest/(role+'.npy'); seal=path.with_suffix('.json')
        if seal.exists():
            if sha(path)!=read(seal)['sha256']: raise ValueError('Changed encoded cache')
            continue
        progress=dest/(role+'_PROGRESS.json'); start=read(progress)['next'] if progress.exists() else 0
        a=np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',dtype=np.float32,
                                 shape=(len(rr),5,32,32,4,384))
        if a.shape!=(len(rr),5,32,32,4,384): raise ValueError('Wrong cache shape')
        for i in range(start,len(rr)):
            row=rr[i]; rgbpath=OUT/row['local_rgb']
            if sha(rgbpath)!=row['payload_sha256']: raise ValueError('RGB payload changed')
            with Image.open(rgbpath) as image: rgb=np.asarray(image.convert('RGB'))
            views=[np.ascontiguousarray(rgb[y0:y1,x0:x1]) for x0,y0,x1,y1 in row['view_boxes_xyxy']]
            a[i]=enc.encode_views(np.stack(views)); a.flush()
            write(progress,{'next':i+1,'total':len(rr),'freeze_sha256':sha(OUT/'FROZEN.json')},replace=True)
            if i%16==0 or i==len(rr)-1:
                print(json.dumps({'stage':'encode','category':category,'role':role,'complete':i+1,'total':len(rr),'GT_reads':0}),flush=True)
        del a; gc.collect()
        write(seal,{'status':'COMPLETE','sha256':sha(path),'shape':[len(rr),5,32,32,4,384],
                    'global_identities':[row['global_identity'] for row in rr],
                    'freeze_sha256':sha(OUT/'FROZEN.json'),'GT_payload_reads':0})
    del enc; gc.collect(); torch.cuda.empty_cache()

def select_bank(bank,ratio,dest,torch):
    import numpy as np
    sys.path.insert(0,str(PC/'src')); sys.path.insert(0,str(CODE))
    from coreset_runtime import ExactUsefulSampler
    np.random.seed(0); torch.manual_seed(0)
    sampler=ExactUsefulSampler(ratio,torch.device('cuda'),number_of_starting_points=10,dimension_to_project_features_to=128)
    selected=sampler.run(bank); indices=sampler.indices
    if len(indices)!=BUDGET or not np.array_equal(selected,bank[indices]): raise ValueError('Coreset count/index mismatch')
    np.save(dest/'indices.npy',indices); np.save(dest/'bank.npy',selected)
    write(dest/'SELECTION.json',{'ratio':ratio,'source_rows':len(bank),'retained_rows':len(indices),
                               'indices_sha256':sha(dest/'indices.npy'),'bank_sha256':sha(dest/'bank.npy'),
                               'seed':0,'project_dimensions':128,'starting_points':10})
    del sampler; gc.collect(); torch.cuda.empty_cache()

def fit(category):
    frozen(); torch=configure_gpu(); import numpy as np
    sys.path.insert(0,str(CODE)); from normal_statistics import covariance,transform,canonical_covariance
    path=OUT/'encoded'/category/'reference.npy'; reference_hash=sha(path)
    if reference_hash!=read(path.with_suffix('.json'))['sha256']: raise ValueError('Unsealed reference cache')
    bank=np.load(path,mmap_mode='r'); dest=OUT/'fit'/category
    for slot in range(VIEWS):
        slotdir=dest/f'slot_{slot}'; slotdir.mkdir(parents=True,exist_ok=True)
        if (slotdir/'COMPLETE.json').exists(): continue
        raw=np.array(bank[:,slot]); mean=raw.mean(-2,dtype=np.float64).astype(np.float32).reshape(-1,DIM)
        center=mean.mean(0,dtype=np.float64); W,total,error=covariance(raw); ordinary=canonical_covariance(mean)
        Lw,mw=transform(W,.5); Lo,mo=transform(ordinary,.5)
        np.savez(slotdir/'covariances.npz',within=W,ordinary=ordinary,total=total)
        specs={'component_W':(raw.reshape(-1,DIM),Lw,center,.01),
               'M_raw_U':(mean,np.eye(DIM,dtype=np.float32),np.zeros(DIM,np.float64),.04),
               'M_W_U':(mean,Lw,center,.04),'M_ordinary05_U':(mean,Lo,center,.04)}
        for name,(features,matrix,mu,ratio) in specs.items():
            d=slotdir/name; d.mkdir(exist_ok=True)
            if (d/'COMPLETE.json').exists(): continue
            transformed=features if name=='M_raw_U' else ((features.astype(np.float64)-mu)@matrix).astype(np.float32)
            select_bank(transformed,ratio,d,torch); np.savez(d/'model.npz',matrix=matrix,center=mu)
            write(d/'COMPLETE.json',{'bank_sha256':sha(d/'bank.npy'),'model_sha256':sha(d/'model.npz'),
                                      'selection_sha256':sha(d/'SELECTION.json'),'B':BUDGET,'reference_cache_sha256':reference_hash})
            del transformed; gc.collect(); torch.cuda.empty_cache()
        write(slotdir/'COMPLETE.json',{'profiles':['component_W','M_raw_U','M_W_U','M_ordinary05_U'],
                                    'within':mw,'ordinary':mo,'covariance_decomposition_error':error,
                                    'freeze_sha256':sha(OUT/'FROZEN.json'),'GT_payload_reads':0})
        del raw,mean; gc.collect(); torch.cuda.empty_cache()
        print(json.dumps({'stage':'fit','category':category,'slot':slot,'B':BUDGET}),flush=True)

def predict(category):
    frozen(); torch=configure_gpu(); import numpy as np
    sys.path.insert(0,str(STUDY/'operators')); from host_original_search import HostOriginalSearch
    dest=OUT/'predictions'/category; dest.mkdir(parents=True,exist_ok=True)
    inputs={}
    for role in ('calibration','query'):
        path=OUT/'encoded'/category/(role+'.npy')
        if sha(path)!=read(path.with_suffix('.json'))['sha256']: raise ValueError('Unsealed query/calibration cache')
        inputs[role]=np.load(path,mmap_mode='r')
    n=sum(len(a) for a in inputs.values()); fieldshape=(n,VIEWS,GRID,GRID)
    for slot in range(VIEWS):
        receipt=dest/f'slot_{slot}_COMPLETE.json'
        if receipt.exists(): continue
        field={name:np.lib.format.open_memmap(dest/(name+'_raw32.npy'),mode='r+' if (dest/(name+'_raw32.npy')).exists() else 'w+',dtype=np.float32,shape=fieldshape) for name in PROFILES}
        for kind,names in (('component_W',('N_W_U','I_W_U')),('M_raw_U',('M_raw_U',)),('M_W_U',('M_W_U',)),('M_ordinary05_U',('M_ordinary05_U',))):
            bankdir=OUT/'fit'/category/f'slot_{slot}'/kind; seal=read(bankdir/'COMPLETE.json')
            if sha(bankdir/'bank.npy')!=seal['bank_sha256'] or sha(bankdir/'model.npz')!=seal['model_sha256']: raise ValueError('Changed fitted state')
            bank=np.load(bankdir/'bank.npy'); z=np.load(bankdir/'model.npz'); matrix=z['matrix']; center=z['center']; z.close()
            engine=HostOriginalSearch(bank,'cuda'); index=0
            for role,a in inputs.items():
                for i in range(len(a)):
                    response=np.array(a[i,slot]).reshape(GRID*GRID,4,DIM)
                    if kind=='component_W':
                        query=((response.astype(np.float64)-center)@matrix).astype(np.float32)
                        one=engine(np.ascontiguousarray(query[:,0]))[0]
                        all4=engine(query.reshape(-1,DIM))[0].reshape(GRID*GRID,4)
                        field['N_W_U'][index,slot]=one.reshape(GRID,GRID)*np.float32(192)
                        field['I_W_U'][index,slot]=all4.mean(1,dtype=np.float64).astype(np.float32).reshape(GRID,GRID)*np.float32(192)
                    else:
                        query=response.mean(1,dtype=np.float64).astype(np.float32)
                        if kind!='M_raw_U': query=((query.astype(np.float64)-center)@matrix).astype(np.float32)
                        field[kind][index,slot]=engine(query)[0].reshape(GRID,GRID)*np.float32(192)
                    index+=1
            for name in names: field[name].flush()
            del engine,bank; gc.collect(); torch.cuda.empty_cache()
            print(json.dumps({'stage':'predict','category':category,'slot':slot,'profile':kind,'units':n,'GT_reads':0}),flush=True)
        del field; gc.collect()
        write(receipt,{'status':'COMPLETE_SLOT','category':category,'slot':slot,'units':n,'GT_reads':0})
    write(dest/'COMPLETE.json',{'status':'ALL_FIVE_PREDICTION_PROFILES_SEALED_BEFORE_GT',
                               'freeze_sha256':sha(OUT/'FROZEN.json'),'calibration':NCAL,'query':n-NCAL,
                               'files':[{'profile':name,'path':name+'_raw32.npy','sha256':sha(dest/(name+'_raw32.npy'))} for name in PROFILES],
                               'GT_payload_reads':0})

def evaluate():
    frozen(); import numpy as np
    from PIL import Image
    sys.path.insert(0,str(BASECODE)); from reader_statistics_core import binary_metrics
    from reader_dino_matching import final_fields
    # A single barrier spans both fixed categories and all five profiles.
    for cat in CATEGORIES:
        seal=read(OUT/'predictions'/cat/'COMPLETE.json')
        for record in seal['files']:
            if sha(OUT/'predictions'/cat/record['path'])!=record['sha256']: raise ValueError('Changed/unsealed predictions')
    write(OUT/'GT_ACCESS_BARRIER.json',{'utc':now(),'both_categories_all_profiles_sealed':True,
                                    'prediction_seals':[{'category':cat,'sha256':sha(OUT/'predictions'/cat/'COMPLETE.json')} for cat in CATEGORIES]})
    members=read(OUT/'MEMBERSHIP.json')['rows']; wanted={(row['archive_file'],row['gt_member']):row for row in members if row['role']=='query' and row['gt_member']}; masks={}
    for archive in sorted({key[0] for key in wanted}):
        with tarfile.open(ARCHIVES/archive,'r|gz') as stream:
            for member in stream:
                key=(archive,member.name)
                if key not in wanted: continue
                row=wanted[key]
                if not member.isfile() or member.size!=row['gt_member_bytes']: raise ValueError('GT member mismatch')
                payload=stream.extractfile(member).read()
                with Image.open(io.BytesIO(payload)) as image: array=np.asarray(image)
                array=np.any(array>0,axis=2) if array.ndim==3 else array>0
                if array.shape!=(256,1280): raise ValueError('GT/RGB layout mismatch')
                masks[row['global_identity']]=np.stack([array[y0:y1,x0:x1] for x0,y0,x1,y1 in row['view_boxes_xyxy']])
    summary=[]; units=[]; values={}
    for cat in CATEGORIES:
        rr=[row for row in members if row['category']==cat and row['role']=='query']; bad=np.array([row['anomalous'] for row in rr],bool)
        gt=np.stack([masks[row['global_identity']] if row['anomalous'] else np.zeros((VIEWS,256,256),bool) for row in rr])
        eligible=bad & gt.reshape(len(gt),-1).any(1)
        values[cat]={}
        for name in PROFILES:
            raw=np.load(OUT/'predictions'/cat/(name+'_raw32.npy'),mmap_mode='r')
            scores=np.array([float(final_fields(np.array(x))['object_score']) for x in raw])
            threshold=float(max(scores[:NCAL])); alarms=scores[NCAL:]>threshold
            maps=[]; aps=[]
            for i,row in enumerate(rr):
                full=final_fields(np.array(raw[NCAL+i]))['maps256']; maps.append(full)
                ap=binary_metrics(gt[i],full)['AP'] if eligible[i] else None
                if ap is not None: aps.append(ap)
                u={'category':cat,'profile':name,'global_identity':row['global_identity'],'anomalous':bool(bad[i]),
                   'Unit_AP_eligible':bool(eligible[i]),'Unit_AP':ap,'native_score':float(scores[NCAL+i]),
                   'native_threshold':threshold,'alarm':int(alarms[i])}
                if eligible[i]:
                    order=np.argsort(-full.ravel(),kind='stable'); y=gt[i].ravel(); den=int(y.sum())
                    for budget in (.01,.05):
                        coverage=float(y[order[:math.floor(len(y)*budget)]].sum()/den)
                        u['coverage_'+str(budget)]=coverage; u['gated_coverage_'+str(budget)]=coverage*int(alarms[i])
                units.append(u)
            values[cat][name]=np.asarray(aps,np.float64)
            # Pool conventional pixel metrics once per profile; never treat pixels as bootstrap replicates.
            pooled=binary_metrics(gt,np.stack(maps)); image=binary_metrics(bad,scores[NCAL:])
            us=[u for u in units if u['category']==cat and u['profile']==name]
            row={'category':cat,'profile':name,'query_units':len(rr),'anomalous_units':int(bad.sum()),'Unit_AP_eligible':int(eligible.sum()),
                 'Unit_AP':float(np.mean(aps)),'Pixel_AP':pooled['AP'],'Pixel_AUROC':pooled['AUROC'],
                 'Image_AP':image['AP'],'Image_AUROC':image['AUROC'],'sensitivity':float(alarms[bad].mean()),
                 'FAR':float(alarms[~bad].mean()),'false_alarms':int(alarms[~bad].sum()),'threshold':threshold}
            for key in ('coverage_0.01','coverage_0.05','gated_coverage_0.01','gated_coverage_0.05'):
                row[key]=float(np.mean([u[key] for u in us if key in u]))
            summary.append(row); del maps; gc.collect()
    csvwrite(OUT/'PER_UNIT.csv',units); csvwrite(OUT/'SUMMARY.csv',summary)
    rng=np.random.default_rng(20261031); boot={}; contrasts=[]
    for cat in CATEGORIES:
        n=len(values[cat]['M_W_U']); draws=rng.integers(0,n,size=(10000,n)); boot[cat]={}
        for left,right in PAIRS:
            difference=values[cat][left]-values[cat][right]; resamples=difference[draws].mean(1); boot[cat][(left,right)]=resamples
            low,high=np.quantile(resamples,[.025,.975])
            contrasts.append({'category':cat,'left':left,'right':right,'difference_pp':float(difference.mean()*100),'lower95_pp':float(low*100),'upper95_pp':float(high*100),'n_anomalous_units':n})
    for left,right in PAIRS:
        resamples=np.mean([boot[cat][(left,right)] for cat in CATEGORIES],axis=0); low,high=np.quantile(resamples,[.025,.975])
        point=np.mean([(values[cat][left]-values[cat][right]).mean() for cat in CATEGORIES])
        contrasts.append({'category':'equal_two_category_macro','left':left,'right':right,'difference_pp':float(point*100),'lower95_pp':float(low*100),'upper95_pp':float(high*100),'n_anomalous_units':sum(len(values[cat][left]) for cat in CATEGORIES)})
    csvwrite(OUT/'CONTRASTS.csv',contrasts)
    write(OUT/'COMPLETE.json',{'status':'ACTUALLY_EXECUTED_SAME_SOURCE_NEW_CATEGORY_CONFIRMATION', 'utc':now(),
                             'freeze_sha256':sha(OUT/'FROZEN.json'),'categories':list(CATEGORIES),'profiles':list(PROFILES),
                             'results':[{'path':name,'sha256':sha(OUT/name)} for name in ('PER_UNIT.csv','SUMMARY.csv','CONTRASTS.csv')],
                             'inference_scope':'Fixed new categories in the previously inventoried MANTA Tiny source; no new manufacturing batch or new dataset claim. Pointwise intervals conditional on fixed references and calibration.'})
    print(json.dumps({'stage':'evaluate','summary':summary,'contrasts':contrasts}),flush=True)

def contracts():
    import numpy as np
    sys.path.insert(0,str(BASECODE)); from reader_statistics_core import binary_metrics
    from reader_dino_matching import final_fields
    assert BUDGET==7208 and int(NREF*GRID*GRID*4*.01)==BUDGET
    q=np.random.default_rng(7).normal(size=(5,32,32)).astype(np.float32)
    field=final_fields(q); assert field['maps256'].shape==(5,256,256)
    expected=np.sort(q.reshape(5,-1),axis=1)[:,-10:].mean(1,dtype=np.float64).max()
    assert float(field['object_score'])==expected
    assert binary_metrics([True,False],[1.,0.])['AP']==1.
    # Cross-camera order must not change joint AP, and a view is never a replicate.
    labels=np.zeros((5,256,256),bool); labels[0,1,1]=True
    scores=np.zeros_like(labels,dtype=np.float32); scores[0,1,1]=1
    assert binary_metrics(labels,scores)['AP']==binary_metrics(labels[::-1],scores[::-1])['AP']==1.
    write(OUT/'CPU_CONTRACTS.json',{'status':'PASS','utc':now(),'B_per_slot':BUDGET,'joint_five_view_AP':True,
                                   'native_top10_then_max_readout':True,'GPU_calls':0,'RGB_or_GT_payload_reads':0})
    print('CPU_CONTRACTS_PASS',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('contracts','stage-rgb','encode','fit','predict','evaluate'))
    parser.add_argument('--category',choices=CATEGORIES)
    args=parser.parse_args()
    if args.stage in ('encode','fit','predict') and args.category is None: parser.error('--category required')
    if args.stage=='contracts': contracts()
    elif args.stage=='stage-rgb': stage_rgb()
    elif args.stage=='encode': encode(args.category)
    elif args.stage=='fit': fit(args.category)
    elif args.stage=='predict': predict(args.category)
    else: evaluate()
