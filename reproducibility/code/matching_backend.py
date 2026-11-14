"""Build, load and query normal-reference states with fixed profile identities.

Importing this module performs no GPU work. Production requires CUDA and
verified device evidence. Existing states and reusable components are checked
against their full payload identities before use.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import uuid
import numpy as np

HERE=Path(__file__).resolve().parent
RUN=HERE.parent
PARENT_ID='parent_panel_20261025T134519Z'
WEIGHT_SHA='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
VISA=['candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2','pcb1','pcb2','pcb3','pcb4','pipe_fryum']
NATIVE={
 'NativeNoRot_full':'NativeNoRot_full_F',
 'NativeNoRot_exact':'NativeNoRot_exactK_F',
 'NativeNoRot_known_slot_U':'NativeNoRot_exactK_slot_U',
 'NativeNoRot_pooled_U':'NativeNoRot_exactK_pooled_U',
 'NativeNoRot_known_slot_U_full':'NativeNoRot_full_slot_U',
 'NativeNoRot_pooled_U_full':'NativeNoRot_full_pooled_U'}
DIRECTIONAL_PROFILES={'N_W_U':('component',1,False),'I_W_U':('component',4,False),'M_W_U':('mean',4,False),
    'N_W_F0':('component',1,True),'I_W_FC4':('component',4,True),'M_W_FC4':('mean',4,True)}
_FILE_VERIFY={}
_CONTEXT_CACHE={}
_ROLE_CACHE={}

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.parent/('.j_'+uuid.uuid4().hex[:8])
    with tmp.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def checked(path,expected):
    path=Path(path).resolve();stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns,expected)
    if key not in _FILE_VERIFY:
        actual=sha(path)
        if actual!=expected:raise ValueError(f'Changed pinned file: {path}; expected {expected}; got {actual}')
        _FILE_VERIFY[key]=True
    return path
def _under(root,path):
    path=Path(path)
    q=(path if path.is_absolute() else Path(root)/path).resolve()
    if not q.is_relative_to(Path(root).resolve()):raise ValueError('Path outside explicit root')
    return q
def profile_spec(name):
    if name in NATIVE:
        legacy=NATIVE[name];return dict(family='native',legacy=legacy,
            bank_kind='native',readings=1,mask=name in ['NativeNoRot_full','NativeNoRot_exact'],
            pooled='pooled' in name,full='full' in name)
    if name in DIRECTIONAL_PROFILES:
        kind,n,mask=DIRECTIONAL_PROFILES[name];return dict(family='diream',bank_kind=kind,readings=n,mask=mask,pooled=False,full=False)
    raise ValueError(f'Unknown production profile: {name!r}')







def _gpu(context):
    if context.get('gpu_safety_checked') is not True:raise RuntimeError('Current CUDA resource checks were not supplied by the driver')
    # Recheck the current runtime report; a caller-written Boolean is insufficient.
    from inference_runner import validate_runtime
    validate_runtime(context,formal=False)
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; no scientific CPU fallback')
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    return torch

def _production_modules(context):
    key=(context['reader_root'],context['base_reader'],context['resources_json'])
    if key in _CONTEXT_CACHE:return _CONTEXT_CACHE[key]
    base=Path(context['base_reader']);reader=Path(context['reader_root']);pins=_parent_pins(context)
    selected=['reader_common.py','reader_dino_encoder.py','reader_search.py','reader_dino_matching.py','reader_baseline_adapters.py']
    basepins=__import__('source_identity').verify_sources(reader)['resolved_pins']
    for name in selected:
        rel='code/'+name
        if 'base/'+rel not in basepins:raise ValueError('Source absent from current package pins: '+rel)
        checked(base/rel,basepins['base/'+rel])
    for p in [reader/'code/complete_methods.py',reader/'code/coreset_runtime.py',reader/'code/foreground_policy.py',Path(context['resources_json'])]:
        if str(p.resolve()) not in pins:raise ValueError('Required source absent from parent pins: '+str(p))
        checked(p,pins[str(p.resolve())])
    # Never resolve models by hub download; resources file points to current pinned local trees.
    resources=read(context['resources_json']);resources['_source_json']=context['resources_json']
    checked(resources['weights']['anomalydino'],WEIGHT_SHA)
    for folder in [str(base/'code'),str(reader/'code'),str(HERE/'feature_adapter'),str(HERE/'directional_adapter')]:
        if folder not in sys.path:sys.path.insert(0,folder)
    cm=importlib.import_module('complete_methods')
    if Path(cm.__file__).resolve()!=(reader/'code/complete_methods.py').resolve():raise ValueError('Wrong imported reader')
    normal=importlib.import_module('reader_dino_encoder')
    search=importlib.import_module('reader_search')
    matching=importlib.import_module('reader_dino_matching')
    fg=importlib.import_module('foreground_policy')
    native_adapter=importlib.import_module('exact_native');directional_adapter=importlib.import_module('directional_profiles')
    value=dict(cm=cm,normal=normal,search=search,matching=matching,foreground=fg,native_adapter=native_adapter,directional_adapter=directional_adapter,resources=resources)
    _CONTEXT_CACHE[key]=value;return value

def _state_key(config,members,spec,targets):
    return dict(category=config['category'],cohort_id='paddy_A176' if config['category']=='paddy' else 'visa12',
        budget_id=config.get('budget_id','B'),family=spec['family'],bank_kind=spec['bank_kind'],
        pooled=spec['pooled'],masked_native=spec['mask'] if spec['family']=='native' else False,
        full=spec['full'],targets_per_slot=targets,membership_digest=members['membership_digest'],
        fit_membership_digest=digest([r['global_identity'] for r in members['reference']]),
        role_source_sha256=members['source_sha256'],weight_sha256=WEIGHT_SHA,
        preprocessing='Native_aspect448_patch14_rawPCA' if spec['family']=='native' else 'square448_C4_inverse_align_normalized_FP32',
        mean_dtype='FP64_mean_then_FP32',transform_dtype='FP64_with_FP32_W_then_FP32',second_normalization=False)

def _file_pin(path):
    path=Path(path).resolve();return dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
def _check_pin(pin):
    p=checked(pin['path'],pin['sha256'])
    if p.stat().st_size!=pin['bytes']:raise ValueError('Pinned state file size changed')
    return p

def _existing_transformed(config,context,physical,kind):
    rows=[];category=config['category']
    for item in physical:
        root=Path(item['bank_path']).parent
        stem='within_M05' if kind=='mean' else ('within_I01' if category=='paddy' else 'within_N01_fast')
        if kind=='mean':
            bp=checked(item['bank_path'],item['bank_sha256']);ip=checked(item['indices_path'],item['indices_sha256']);mp=checked(item['model_path'],item['model_sha256'])
        else:
            meta=read(root/('COMPLETE.json' if category=='paddy' else stem+'_FIT.json'))
            if category=='paddy':
                pins={Path(r['file']).name:r['sha256'] for r in meta['files']}
                expected={s:pins[stem+'_'+s] for s in ['bank.npy','indices.npy','model.npz']}
            else:
                expected={'bank.npy':meta['bank_sha256'],'indices.npy':meta['selection']['indices_sha256'],
                          'model.npz':item['model_sha256']}
            bp=checked(root/(stem+'_bank.npy'),expected['bank.npy']);ip=checked(root/(stem+'_indices.npy'),expected['indices.npy']);mp=checked(root/(stem+'_model.npz'),expected['model.npz'])
            with np.load(item['model_path'],allow_pickle=False) as a,np.load(mp,allow_pickle=False) as b:
                if set(a.files)!=set(b.files) or any(not np.array_equal(a[k],b[k]) for k in a.files):raise ValueError('Component W/center differs from mean state')
        bank=np.load(bp,mmap_mode='r',allow_pickle=False);ix=np.load(ip,allow_pickle=False)
        expected_rows=item['retained_rows'];candidates=item['candidate_mean_rows']*(4 if kind=='component' else 1)
        if bank.dtype!=np.float32 or bank.shape!=(expected_rows,384) or ix.shape!=(expected_rows,) or not np.isfinite(bank).all():raise ValueError('Inherited state invalid')
        if ix.min()<0 or ix.max()>=candidates:raise ValueError('Inherited selection outside full fit-only candidate pool')
        rows.append(dict(slot_id=item['slot_id'],bank=_file_pin(bp),indices=_file_pin(ip),model=_file_pin(mp),
            candidate_rows=candidates,retained_rows=expected_rows,representation=kind,source_kind='HISTORICAL_STATE_IDENTITY_VERIFIED_NOT_PREDICTION_REUSE'))
    return rows

def verify_component_cache(config,context=None):
    """Full exact-file hash, metadata + membership/source pins; no sampling shortcut."""
    context=context_defaults(context);category=config['category'];members=roles(config,context)
    if category=='paddy':raise ValueError('No explicitly bound all-fit Paddy cache; rebuild original RGB for this branch')
    row=next(r for r in read(Path(context['parent_run_root'])/'protocol/source_audit/CACHE_AVAILABILITY.json')['cache_states'] if r['category']==category)
    root=Path(row['root']);p=root/'responses.npy';encoded=root/'ENCODED.json'
    pins=_parent_pins(context)
    if str(encoded.resolve()) not in pins:raise ValueError('Encoded cache metadata is not parent-pinned')
    checked(encoded,pins[str(encoded.resolve())]);meta=read(encoded)
    ids=[r['global_identity'] for role in ['reference','calibration','query'] for r in members[role]]
    if meta['identities']!=ids:raise ValueError('Cache identity order mismatch')
    checked(p,meta['responses_sha256']);a=np.load(p,mmap_mode='r',allow_pickle=False)
    if a.dtype!=np.float32 or a.shape!=(len(ids),32,32,4,384):raise ValueError('Cache shape/dtype mismatch')
    return a,dict(path=str(p),sha256=meta['responses_sha256'],verified_full_payload=True,
        metadata_sha256=sha(encoded),ordered_member_digest=digest(ids),fit_n=len(members['reference']),
        preprocessing='Pinned source encoder identity; quarter-turn0/90/180/270 inverse-aligned unit components')

def _make_transformed(config,context,destination,physical,kind,targets,modules):
    members=roles(config,context);directory=Path(destination);directory.mkdir(parents=True,exist_ok=False)
    use_cache=context.get('component_cache_mode','verify')=='verify' and config['category']!='paddy'
    array,cache_pin=verify_component_cache(config,context) if use_cache else (None,None)
    encoder=None;rows=[];n=len(members['reference'])
    for slot,item in enumerate(physical):
        mp=checked(item['model_path'],item['model_sha256'])
        with np.load(mp,allow_pickle=False) as z:model={k:z[k].copy() for k in z.files}
        candidates=n*1024*(4 if kind=='component' else 1)
        if targets[slot]>candidates:raise ValueError('Requested K exceeds complete candidate pool')
        candidate_file=directory/f'candidate_slot_{slot}.npy'
        candidate=np.lib.format.open_memmap(candidate_file,mode='w+',dtype=np.float32,shape=(candidates,384))
        cursor=0
        for i,row in enumerate(members['reference']):
            if array is not None:components=np.asarray(array[i])
            else:
                if encoder is None:encoder=modules['cm'].Runtime('DiReAM',context['base_reader'],modules['resources'],profiles=['within_M05']).encoder
                components=encoder.encode_views(np.stack(decode_unit(row,context)))[slot]
            representation=components.reshape(-1,384) if kind=='component' else components.mean(-2,dtype=np.float64).astype(np.float32).reshape(-1,384)
            transformed=((representation.astype(np.float64)-model['center'])@model['matrix']).astype(np.float32)
            if not np.isfinite(transformed).all():raise ValueError('Nonfinite transformed fit candidate')
            candidate[cursor:cursor+len(transformed)]=transformed;cursor+=len(transformed)
        candidate.flush()
        if cursor!=candidates:raise ValueError('Incomplete normal candidate matrix')
        bank,ix,metadata=modules['native_adapter'].select_exact(candidate,targets[slot],modules['resources'],Path(context['reader_root'])/'code','cuda')
        bp=directory/f'bank_slot_{slot}.npy';ip=directory/f'indices_slot_{slot}.npy'
        np.save(bp,bank,allow_pickle=False);np.save(ip,ix,allow_pickle=False)
        rows.append(dict(slot_id=slot,bank=_file_pin(bp),indices=_file_pin(ip),model=_file_pin(mp),
            candidate_rows=candidates,retained_rows=targets[slot],representation=kind,selection=metadata,
            candidate_pin=_file_pin(candidate_file),cache=cache_pin,source_kind='RAW_RGB_REENCODED_FIT' if array is None else 'VERIFIED_CACHE_FIT'))
        del candidate
    return rows

def _native_candidates(config,context,runtime,members,provenance):
    """One complete candidate pool per category, shared only after pin checks.

    Default rebuilds every fit-normal from direct original RGB. Optional reuse of previously derived native candidate slices
    with direct-input proof requires all source shards and three raw-feature parity checks;
    zero-angle naming alone never qualifies a source for use.
    """
    identity=digest(dict(category=config['category'],reference=[(r['global_identity'],r['source_payload_sha256']) for r in members['reference']],provenance=provenance))
    out=Path(context['run_root'])/'artifacts/nc'/identity[:16]
    if out.exists():
        spec=read(out/'NATIVE_CANDIDATES.json')
        if spec.get('candidate_pool_identity_sha256')!=identity:raise ValueError('Native candidate short-path collision')
        expected=[(r['global_identity'],slot) for r in members['reference'] for slot in range(len(r['real_view_boxes']))]
        if spec['ordered_normal_ids']!=[r['global_identity'] for r in members['reference']] or spec['provenance']!=provenance:
            raise ValueError('Shared native candidate membership/provenance mismatch')
        if [(r['global_identity'],r['slot_id']) for r in spec['records']]!=expected:raise ValueError('Shared native candidates incomplete')
        for row in spec['records']:checked(out/row['file'],row['sha256'])
        return out
    out.parent.mkdir(parents=True,exist_ok=True)
    temporary=out.parent/('.c_'+uuid.uuid4().hex[:8])
    if context.get('native_candidate_mode','raw_rebuild')!='verified_native_slices':
        runtime.build_candidates(members['reference'],members['calibration'],members['query'],lambda r:decode_unit(r,context),temporary)
    else:
        if config['category']=='paddy':raise ValueError('No single-view native candidate-slice proof for Paddy; use raw_rebuild')
        root=Path(context['native_candidate_roots'][config['category']]).resolve()
        complete=read(root/'COMPLETE.json');checked(root/'BINDING.json',complete['binding_sha256']);checked(root/'FIT.json',complete['fit_sha256'])
        fitting=read(root/'FIT.json');proof_path=checked(root/'NORMAL_SLICES.json',fitting['source_sha256']);proof=read(proof_path)
        if proof['status']!='PROVED_DIRECT_INPUT_EQUIVALENT_ZERO_ANGLE_FEATURE_SLICES':raise ValueError('Native candidate direct-input proof absent')
        records=proof['records'];ids=[r['global_identity'] for r in members['reference']]
        if [r['global_identity'] for r in records]!=ids or read(root/'BINDING.json')['normal_ids']!=ids:raise ValueError('Native candidate roles differ')
        arrays=[];raw_checks=[]
        for i,(record,member) in enumerate(zip(records,members['reference'])):
            if record['raw_member']['source_payload_sha256']!=member['source_payload_sha256']:raise ValueError('Native raw-source pin differs')
            if len(record['slices'])!=1 or len(record['geometry'])!=1:raise ValueError('Native candidate source has unexpected actual slots')
            geometry=record['geometry'][0]
            if geometry['native_augmentation_index']!=0 or geometry['real_view']!=0:raise ValueError('Native candidate slice is not the original actual view')
            for f,h in [('source_manifest','source_manifest_sha256'),('source_metadata','source_metadata_sha256')]:checked(record[f],record[h])
            p=checked(_under(context['source_root'],record['path']),record['sha256']);a=np.load(p,mmap_mode='r',allow_pickle=False)
            if list(a.shape)!=record['shape'] or a.dtype!=np.float32:raise ValueError('Native candidate source shard shape/dtype differs')
            lo,hi=record['slices'][0];z=np.ascontiguousarray(a[lo:hi])
            if len(z)!=record['normalized_vectors'] or not np.isfinite(z).all():raise ValueError('Native candidate slice count or values invalid')
            if i<3:
                features,grid=runtime.model._features(decode_unit(member,context)[0]);fresh=np.ascontiguousarray(runtime.model._normalize(features),np.float32)
                if fresh.shape!=z.shape or tuple(grid)!=tuple(geometry['patch_grid_hw']) or not np.allclose(fresh,z,atol=1e-6,rtol=1e-5):
                    raise ValueError('Original-RGB/native-candidate-slice parity failed; no silent fallback')
                raw_checks.append(dict(unit_id=member['global_identity'],bitwise=bool(np.array_equal(fresh,z)),max_abs_error=float(np.max(np.abs(fresh.astype(np.float64)-z.astype(np.float64))))))
            arrays.append((z,geometry['patch_grid_hw'],member))
        temporary.mkdir(parents=True,exist_ok=False);new_records=[]
        for i,(z,grid,member) in enumerate(arrays):
            name=f'normal_{i:05d}_slot_0.npy';np.save(temporary/name,z,allow_pickle=False)
            new_records.append(dict(file=name,slot_id=0,global_identity=member['global_identity'],source_payload_sha256=member['source_payload_sha256'],shape=list(z.shape),grid=list(grid),sha256=sha(temporary/name)))
        payload=dict(schema='diream-native-candidates-v1',status='COMPLETE',ordered_normal_ids=ids,slots=1,
            normal_rotation=False,normal_augmentations_per_view=1,normal_input_path='direct_unwarped_rgb',reference_foreground=False,
            provenance=provenance,records=new_records,source_reuse='direct_input_equivalent_slice_proof',
            proof_sha256=sha(proof_path),raw_normal_parity_checks=raw_checks)
        atomic_json(temporary/'NATIVE_CANDIDATES.json',payload)
    saved=read(temporary/'NATIVE_CANDIDATES.json');saved['candidate_pool_identity_sha256']=identity
    atomic_json(temporary/'NATIVE_CANDIDATES.json',saved)
    os.replace(temporary,out);return out

def build_state(config,context,output_dir):
    context=context_defaults(context);spec=profile_spec(config['profile_id']);members=roles(config,context)
    targets,physical=_budget(config,context);binding=_state_key(config,members,spec,targets)
    out=Path(output_dir).resolve()
    if not out.is_relative_to(Path(context['run_root']).resolve()):raise ValueError('New state must be inside current run')
    if out.exists():
        state=validate_state_manifest(config,context,out);return state
    temp=out.parent/('.s_'+uuid.uuid4().hex[:8]);temp.mkdir(parents=True,exist_ok=False)
    state=dict(schema='diream-bound-state-v1',status='COMPLETE',binding=binding,
               state_id=config.get('state_id',digest(binding)),profile_family=spec['family'],bank_kind=spec['bank_kind'],
               source_kind='NEW_STATE_BINDING',GT_reads=0)
    try:
        if spec['family']=='diream' and config.get('budget_id','B')=='B' and context.get('state_reuse_mode')=='historical_verified':
            state['slots']=_existing_transformed(config,context,physical,spec['bank_kind'])
        else:
            _gpu(context);modules=_production_modules(context)
            if spec['family']=='diream':
                state['slots']=_make_transformed(config,context,temp/'transformed',physical,spec['bank_kind'],targets,modules)
            else:
                provenance=dict(weight_sha256=WEIGHT_SHA,native_encoder_buffer_sha256=read(Path(context['reader_root'])/'protocol/model_pins.json')['native_s14_buffer_sha256'],
                    normalization='Original_Native_L2_FP32',native_geometry='aspect448_shortedge_patch14_trailingcrop')
                runtime=modules['native_adapter'].NativeNoRotExactRuntime(spec['legacy'],context['reader_root'],context['base_reader'],modules['resources'],provenance)
                candidate_dir=_native_candidates(config,context,runtime,members,provenance)
                rawstate=runtime.fit_from_candidates(candidate_dir,temp/'native',members['reference'],members['calibration'],members['query'],targets)
                state.update(native_relative='native',native_state=rawstate,provenance=provenance)
        # Paths into the temporary tree must become final absolute paths before atomic rename.
        def relocate(value):
            if isinstance(value,str):return value.replace(str(temp),str(out))
            if isinstance(value,list):return [relocate(x) for x in value]
            if isinstance(value,dict):return {k:relocate(v) for k,v in value.items()}
            return value
        state=relocate(state)
        if spec['family']=='native':
            atomic_json(temp/'native/NATIVE_STATE.json',state['native_state'])
        state['state_identity_sha256']=digest({k:v for k,v in state.items() if k!='state_identity_sha256'})
        atomic_json(temp/'STATE.json',state);os.replace(temp,out)
        return validate_state_manifest(config,context,out)
    except BaseException as e:
        atomic_json(temp/'FAILED.json',dict(status='FAILED',error=repr(e),GT_reads=0));raise

def validate_state_manifest(config,context,state_dir):
    context=context_defaults(context);members=roles(config,context);spec=profile_spec(config['profile_id']);targets,_=_budget(config,context)
    state=read(Path(state_dir)/'STATE.json');identity=state.pop('state_identity_sha256',None)
    if digest(state)!=identity:raise ValueError('State manifest identity mismatch')
    state['state_identity_sha256']=identity
    if config.get('state_id') is not None and state.get('state_id')!=config['state_id']:raise ValueError('State short-path identity collision')
    if state['status']!='COMPLETE' or state['binding']!=_state_key(config,members,spec,targets):raise ValueError('Wrong profile/bank/model/member/geometry state')
    if spec['family']=='diream':
        if len(state['slots'])!=len(targets):raise ValueError('State actual slot count mismatch')
        for slot,row in enumerate(state['slots']):
            if row['slot_id']!=slot or row['representation']!=spec['bank_kind'] or row['retained_rows']!=targets[slot]:raise ValueError('State row count/kind mismatch')
            bp=_check_pin(row['bank']);ip=_check_pin(row['indices']);mp=_check_pin(row['model'])
            b=np.load(bp,mmap_mode='r',allow_pickle=False);ix=np.load(ip,allow_pickle=False)
            if b.dtype!=np.float32 or b.shape!=(targets[slot],384) or ix.shape!=(targets[slot],) or not np.isfinite(b).all():raise ValueError('State arrays invalid')
            if ix.min()<0 or ix.max()>=row['candidate_rows']:raise ValueError('State index outside normal pool')
            with np.load(mp,allow_pickle=False) as z:
                if set(z.files)!={'matrix','center'} or z['matrix'].shape!=(384,384) or z['center'].shape!=(384,) or not all(np.isfinite(z[k]).all() for k in z.files):raise ValueError('Invalid W/center')
    else:
        ns=read(Path(state_dir)/state['native_relative']/'NATIVE_STATE.json')
        if ns!=state['native_state']:raise ValueError('Native inner/outer binding mismatch')
        if ns['ordered_normal_ids']!=[r['global_identity'] for r in members['reference']]:raise ValueError('Native fit membership mismatch')
        nslots=len(members['reference'][0]['real_view_boxes'])
        groups=[list(range(nslots))] if spec['pooled'] else [[slot] for slot in range(nslots)]
        if ns['slots']!=nslots or [row['slots'] for row in ns['states']]!=groups:raise ValueError('Native pooled/known-slot topology differs')
        if ns['profile_id']!=spec['legacy'] or ns['profile_spec']!={'foreground':spec['mask'],'slot_mode':'pooled' if spec['pooled'] else 'known_slot'}:
            raise ValueError('Native inner runtime profile differs')
        if ns['normal_rotation'] is not False or ns['reference_foreground'] is not False or ns['normal_augmentations_per_view']!=1 or ns['normal_input_path']!='direct_unwarped_rgb':
            raise ValueError('Native normal-side method identity differs')
        manifest_path=checked(ns['candidate_manifest'],ns['candidate_manifest_sha256']);candidate_meta=read(manifest_path)
        expected_pairs=[(r['global_identity'],slot) for r in members['reference'] for slot in range(nslots)]
        if [(r['global_identity'],r['slot_id']) for r in candidate_meta['records']]!=expected_pairs:raise ValueError('Native candidate members differ')
        if candidate_meta['ordered_normal_ids']!=ns['ordered_normal_ids'] or candidate_meta['provenance']!=state['provenance']:raise ValueError('Native candidate provenance differs')
        for row,selected_slots in zip(ns['states'],groups):
            bp=checked(Path(state_dir)/'native'/row['bank'],row['bank_sha256']);ip=checked(Path(state_dir)/'native'/row['indices'],row['indices_sha256'])
            candidate_rows=sum(r['shape'][0] for r in candidate_meta['records'] if r['slot_id'] in selected_slots)
            expected_k=candidate_rows if spec['full'] else sum(targets[s] for s in selected_slots)
            if row['candidate_rows']!=candidate_rows or row['retained_rows']!=expected_k:raise ValueError('Native bank differs from externally fixed absolute budget')
            bank=np.load(bp,mmap_mode='r',allow_pickle=False);indices=np.load(ip,allow_pickle=False)
            if bank.dtype!=np.float32 or bank.shape!=(expected_k,384) or not np.isfinite(bank).all():raise ValueError('Native bank shape/dtype/values invalid')
            if indices.dtype.kind not in 'iu' or indices.shape!=(expected_k,) or indices.min()<0 or indices.max()>=candidate_rows:raise ValueError('Native selected indices invalid')
            if spec['full'] and not np.array_equal(indices,np.arange(candidate_rows)):raise ValueError('Full Native index order differs')
    return state

class BoundRuntime:
    def __init__(self,config,context,state_dir):
        self.config=dict(config);self.context=context_defaults(context);self.state_dir=Path(state_dir)
        self.spec=profile_spec(config['profile_id']);self.state=validate_state_manifest(config,self.context,state_dir)
        _gpu(self.context);self.modules=_production_modules(self.context);m=self.modules
        if self.spec['family']=='native':
            self.runtime=m['native_adapter'].NativeNoRotExactRuntime(self.spec['legacy'],self.context['reader_root'],self.context['base_reader'],m['resources'],self.state['provenance'])
            self.runtime.load(self.state_dir/'native');self.resident=[]
        else:
            self.encoder=m['cm'].Runtime('DiReAM',self.context['base_reader'],m['resources'],profiles=['within_M05']).encoder
            self.resident=[];self.bound_operators=[]
            for row in self.state['slots']:
                bank=np.load(row['bank']['path'],mmap_mode='r',allow_pickle=False)
                with np.load(row['model']['path'],allow_pickle=False) as z:model={k:z[k].copy() for k in z.files}
                engine=m['search'].HostOriginalSearch(bank,'cuda',copy_host=False)
                m['directional_adapter'].validate_model(model)
                legacy=m['directional_adapter'].profile(config['profile_id']).legacy_profile
                self.resident.append((engine,model))
                # Literal validated state/engine/profile captured once at load.
                # Per-request source raw_field guards remain; adapter-added W
                # finite/shape scans are outside the complete-request timer.
                self.bound_operators.append(lambda x,mask,e=engine,w=model,p=legacy:
                    m['directional_adapter']._legacy_raw_field(x,p,e,w,mask))
    def predict_views(self,views):
        if len(views)!=(5 if self.config['category']=='paddy' else 1):raise ValueError('Wrong actual view count')
        if self.spec['family']=='native':
            result=self.runtime.predict(views);calls=len(views);pca=calls if self.spec['mask'] else 0
        else:
            raws=[];masks=[];calls=pca=0;m=self.modules
            for slot,image in enumerate(views):
                components,mask,counts=m['directional_adapter'].encode_view(self.encoder,image,self.config['profile_id'],square_views=m['normal'].square_views,
                    normalize_components=m['normal'].normalize_components,foreground_mask=m['foreground'].foreground_mask,device='cuda')
                raw,_=self.bound_operators[slot](components,mask)
                raws.append(raw);calls+=counts['actual_encoder_calls'];pca+=counts['actual_pca_calls']
                if mask is not None:masks.append(mask)
            result=m['matching'].final_fields(np.asarray(raws,np.float32))
            if masks:result['actual_masks']=np.asarray(masks,bool)
        result.update(fields=result['maps256'],native_score=result['object_score'],actual_encoder_calls=np.array(calls,np.int32),actual_PCA_calls=np.array(pca,np.int32),
            actual_views=np.array(len(views),np.int32),encoder_readings_per_view=np.array(self.spec['readings'],np.int32),
            distance_queries_per_anchor=np.array(4 if self.config['profile_id'].startswith('I_') else 1,np.int32))
        if 'actual_masks' in result:result['masks']=result['actual_masks']
        if self.spec['family']=='diream':
            result['matching_query_vectors']=np.array(len(views)*1024*(4 if self.config['profile_id'].startswith('I_') else 1),np.int64)
            result['actual_search_invocations']=np.array(len(views),np.int32)
        # Full output validation belongs to driver.normalize_output outside the
        # request timer. Original primitive input checks are retained unchanged.
        return result

def load_state(config,context,state_dir):return BoundRuntime(config,context,state_dir)



def normal_regression(config,context,output_dir):
    """Actual fixed first-three calibration RGB regression; never reads GT."""
    context=context_defaults(context);_gpu(context);members=roles(config,context)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    sid=config.get('state_id',digest(_state_key(config,members,profile_spec(config['profile_id']),_budget(config,context)[0])))
    state_dir=Path(context['run_root'])/'artifacts/s'/sid[:16]
    build_state(config,context,state_dir);runtime=load_state(config,context,state_dir)
    baseline_path=_baseline(config,context);baseline=np.load(baseline_path,allow_pickle=False) if baseline_path else None
    native_baselines=_native_baseline(config,context,members)
    if baseline is not None:
        key='normal_ids' if 'normal_ids' in baseline.files else 'identities'
        if [str(x) for x in baseline[key]]!=[r['global_identity'] for r in members['calibration']]:raise ValueError('Historical calibration members differ')
    records=[]
    for i,row in enumerate(members['calibration'][:3]):
        result=runtime.predict_views(decode_unit(row,context));file=out/f'normal_{i:03d}.npz';np.savez(file,**result)
        if not np.isfinite(result['maps256']).all() or not np.isfinite(result['object_score']).all():raise ValueError('Nonfinite normal regression prediction')
        item=dict(category=config['category'],profile_id=config['profile_id'],budget_id=config.get('budget_id','B'),unit_id=row['global_identity'],actual_views=len(row['real_view_boxes']),
            prediction_sha256=sha(file),state_sha256=runtime.state['state_identity_sha256'],atol=1e-6,rtol=1e-5,
            actual_encoder_calls=int(result['actual_encoder_calls']),actual_PCA_calls=int(result['actual_PCA_calls']),
            source_kind='RAW_RGB_RECOMPUTED_NORMAL_TECHNICAL_REGRESSION',GT_reads=0)
        old=None
        if baseline is not None:old={k:baseline[k][i] for k in ['maps256','object_score','raw32'] if k in baseline.files}
        elif native_baselines is not None:
            with np.load(native_baselines[i],allow_pickle=False) as src:
                if str(src['global_identity'])!=row['global_identity']:raise ValueError('Historical Native input identity differs')
                old={k:src[k].copy() for k in src.files}
        if old is None:item.update(status='NOT_APPLICABLE_NO_HISTORICAL_PROFILE',map_max_abs_error='',score_abs_error='',raw_max_abs_error='',mask_equal='')
        else:
            mapdiff=np.abs(result['maps256'].astype(np.float64)-old['maps256'].astype(np.float64))
            scoreerr=abs(float(result['object_score'])-float(old['object_score']));rawerr=''
            passed=np.allclose(result['maps256'],old['maps256'],atol=1e-6,rtol=1e-5) and np.isclose(float(result['object_score']),float(old['object_score']),atol=1e-6,rtol=1e-5)
            if 'raw32' in result and 'raw32' in old:
                rawerr=float(np.max(np.abs(result['raw32'].astype(np.float64)-old['raw32'].astype(np.float64))))
                passed=passed and np.allclose(result['raw32'],old['raw32'],atol=1e-6,rtol=1e-5)
            oldpath=native_baselines[i] if native_baselines is not None else baseline_path
            item.update(status='PASSED' if passed else 'FAILED',map_max_abs_error=float(mapdiff.max()),score_abs_error=scoreerr,raw_max_abs_error=rawerr,
                map_bitwise=bool(np.array_equal(result['maps256'],old['maps256'])),score_bitwise=bool(np.array_equal(result['object_score'],old['object_score'])),
                baseline_path=str(oldpath),baseline_sha256=sha(oldpath),mask_equal='NOT_APPLICABLE_UNMASKED')
            if 'actual_masks' in result and config['category']!='paddy':
                cache=next(r for r in read(Path(context['parent_run_root'])/'protocol/source_audit/CACHE_AVAILABILITY.json')['cache_states'] if r['category']==config['category'])
                maskpath=Path(cache['root'])/'masks.npy';checked(maskpath,cache['encoded_metadata']['masks_sha256'])
                prior=np.load(maskpath,mmap_mode='r',allow_pickle=False)[i][None]
                if config['profile_id']=='N_W_F0':prior=np.repeat(prior[...,:1],4,axis=-1)
                equal=np.array_equal(result['actual_masks'],prior);item['mask_equal']=bool(equal)
                if not equal:item['status']='FAILED'
            if config['profile_id']=='NativeNoRot_full':
                equal='native_masks' in result and 'native_mask' in old and np.array_equal(result['native_masks'][0],old['native_mask'])
                item['mask_equal']=bool(equal)
                if not equal:item['status']='FAILED'
                if 'native_patch_map' in old:
                    item['raw_max_abs_error']=float(np.max(np.abs(result['native_patch_maps'][0]-old['native_patch_map'])))
                    if not np.allclose(result['native_patch_maps'][0],old['native_patch_map'],atol=1e-6,rtol=1e-5):item['status']='FAILED'
        records.append(item)
    if baseline is not None:baseline.close()
    atomic_json(out/'REGRESSION.json',dict(status='PASSED' if all(r['status']=='PASSED' for r in records) else
        'NOT_APPLICABLE' if all(r['status'].startswith('NOT_APPLICABLE') for r in records) else 'FAILED',rows=records,GT_reads=0))
    headers=list(dict.fromkeys(k for r in records for k in r))
    with (out/'NORMAL_REGRESSION.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,headers);w.writeheader();w.writerows(records)
    return records

def integration_fixture(config,context,output_dir):
    """Real CUDA integration fixture from four existing fit units, not science.

    Native exact-K fit/load/predict and new I component fit/load/predict exercise
    low K=64 solely for integration. This state is never accepted by build_state
    or the production STATE validator and cannot replace any formal full bank.
    """
    context=context_defaults(context);_gpu(context);modules=_production_modules(context)
    category=config['category']
    if category!='chewinggum':raise ValueError('Prespecified integration representative is Chewinggum only')
    members=roles(dict(config,profile_id='M_W_U'),context);fit=members['reference'][:4];cal=members['calibration'][:3]
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    provenance=dict(weight_sha256=WEIGHT_SHA,native_encoder_buffer_sha256=read(Path(context['reader_root'])/'protocol/model_pins.json')['native_s14_buffer_sha256'],
        normalization='Original_Native_L2_FP32',native_geometry='aspect448_shortedge_patch14_trailingcrop')
    native=modules['native_adapter'].NativeNoRotExactRuntime('NativeNoRot_exactK_F',context['reader_root'],context['base_reader'],modules['resources'],provenance)
    native.build_candidates(fit,members['calibration'],members['query'],lambda r:decode_unit(r,context),out/'native_candidates')
    native_state=native.fit_from_candidates(out/'native_candidates',out/'native_state',fit,members['calibration'],members['query'],[64])
    native.load(out/'native_state');records=[]
    for i,row in enumerate(cal):
        pred=native.predict(decode_unit(row,context));np.savez(out/f'native_normal_{i}.npz',**pred)
        if pred['maps256'].shape!=(1,256,256) or not np.isfinite(pred['maps256']).all():raise ValueError('Native real fixture output invalid')
        records.append(dict(profile_id='NativeNoRot_exact',unit_id=row['global_identity'],status='PASSED',fit_units=4,retained_rows=64,
            prediction_sha256=sha(out/f'native_normal_{i}.npz'),historical_regression='NOT_APPLICABLE_NEW_EXACT_K',actual_views=1))
    del native
    # Use actual raw fit RGB for this small fixture, with full-normal W held fixed.
    encoder=modules['cm'].Runtime('DiReAM',context['base_reader'],modules['resources'],profiles=['within_M05']).encoder
    physical=_budget(dict(config,profile_id='M_W_U',budget_id='B'),context)[1][0]
    mp=checked(physical['model_path'],physical['model_sha256'])
    with np.load(mp,allow_pickle=False) as z:model={k:z[k].copy() for k in z.files}
    components=[encoder.encode_views(np.stack(decode_unit(r,context)))[0] for r in fit]
    source=np.stack(components).reshape(-1,384)
    transformed=((source.astype(np.float64)-model['center'])@model['matrix']).astype(np.float32)
    bank,indices,selection=modules['native_adapter'].select_exact(transformed,64,modules['resources'],Path(context['reader_root'])/'code','cuda')
    np.save(out/'component_bank.npy',bank);np.save(out/'component_indices.npy',indices)
    if not np.array_equal(bank,transformed[indices]) or len(bank)!=64:raise ValueError('Real component exact-K failed')
    engine=modules['search'].HostOriginalSearch(np.load(out/'component_bank.npy',allow_pickle=False),'cuda',copy_host=False)
    for i,row in enumerate(cal):
        x,mask,counts=modules['directional_adapter'].encode_view(encoder,decode_unit(row,context)[0],'I_W_U',square_views=modules['normal'].square_views,
            normalize_components=modules['normal'].normalize_components,foreground_mask=modules['foreground'].foreground_mask,device='cuda')
        actual,_=modules['directional_adapter'].raw_field(x,'I_W_U',engine,model)
        directional=np.stack([modules['directional_adapter'].raw_field(x[:,:,k,:],'N_W_U',engine,model)[0] for k in range(4)])
        manual=directional.mean(0,dtype=np.float64).astype(np.float32)
        error=float(np.max(np.abs(actual.astype(np.float64)-manual.astype(np.float64))))
        if not np.allclose(actual,manual,atol=1e-6,rtol=1e-5):raise ValueError('New I actual four-min operator cross-check failed')
        result=modules['matching'].final_fields(actual[None]);np.savez(out/f'I_normal_{i}.npz',**result)
        records.append(dict(profile_id='I_W_U',unit_id=row['global_identity'],status='PASSED',fit_units=4,retained_rows=64,
            candidate_rows=len(transformed),prediction_sha256=sha(out/f'I_normal_{i}.npz'),operator_max_abs_error=error,
            historical_regression='NOT_APPLICABLE_NEW_I_PROFILE',actual_views=1,actual_encoder_calls=counts['actual_encoder_calls']))
    report=dict(status='PASSED',scope='CUDA_NORMAL_INTEGRATION_FIXTURE',
        fit_role=[r['global_identity'] for r in fit],calibration_role=[r['global_identity'] for r in cal],
        formal_states_created=False,GT_reads=0,raw_normal_units_encoded=4+3+4+3,rows=records,
        native_state=native_state,component_selection=selection,
        fixed_full_normal_model_sha256=physical['model_sha256'],atol=1e-6,rtol=1e-5)
    atomic_json(out/'FIXTURE_RESULT.json',report);return report


from dataset_bindings import patch_backend as _bind_portable_backend
_bind_portable_backend(sys.modules[__name__])

