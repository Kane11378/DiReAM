"""Coordinate fixed-profile inference, interrupted runs and request timing.

Production uses the Windows/CUDA backend with explicit device checks. Planning
and help do not load Torch or open annotations. Synthetic fixtures are confined
to the test interfaces.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from state_store import (atomic_json, atomic_npz, digest, file_sha, job_lock,
                          member_digest, now, read_json, validate_records, within)

RUN_ROOT = Path(__file__).resolve().parents[1]
PARENT_ROOT = RUN_ROOT/'protocol/evaluation/parent'
SOURCE_ROOT = RUN_ROOT
PROCESS_ENTRY_CLOCK = time.perf_counter()
_ACTIVE_OPERATION = {}
PROFILES = {
    'NativeNoRot_full', 'NativeNoRot_exact', 'NativeNoRot_known_slot_U',
    'NativeNoRot_pooled_U', 'NativeNoRot_known_slot_U_full', 'NativeNoRot_pooled_U_full',
    'N_W_U', 'I_W_U', 'M_W_U', 'N_W_F0', 'I_W_FC4', 'M_W_FC4',
}


def default_context(run_root=RUN_ROOT, parent_root=PARENT_ROOT):
    return dict(run_root=str(Path(run_root).resolve()), parent_run_root=str(Path(parent_root).resolve()),
                source_root=str(SOURCE_ROOT), device='cuda', gpu_safety_checked=False)


def strict_profile(name):
    if name not in PROFILES:
        raise ValueError(f'Unregistered fixed-panel profile: {name}')
    return name


def _state_kind(profile):
    if profile.startswith(('N_W_', 'I_W_')):
        return 'within_component'
    if profile.startswith('M_W_'):
        return 'within_mean'
    return profile


def plan(parent_root=PARENT_ROOT, run_root=RUN_ROOT):
    """Use only exact inherited checkpoint files; no project discovery scan."""
    parent_root, run_root = Path(parent_root).resolve(), Path(run_root).resolve()
    status_path = parent_root/'coverage.json'
    budget_path = parent_root/'protocol/RESOLVED_BUDGETS.json'
    member_path = parent_root/'protocol/source_audit/MEMBERSHIP_AUDIT.json'
    status, budget_doc, members = map(read_json, (status_path, budget_path, member_path))
    if status['run_id'] != 'parent_panel_20261025T134519Z' or status['analysis_source_id'] != 'fixed_panel_evaluation':
        raise ValueError('Parent checkpoint identity differs')
    budgets = {b['category']:b for b in budget_doc['rows']}
    groups = {g['category']:g for g in members['groups']}
    requirements, configs = [], {}
    for index, requirement in enumerate(status['configs']):
        c, p, b = requirement['category'], strict_profile(requirement['profile_id']), requirement['budget_id']
        if c not in groups or b not in {'half_B', 'B', 'double_B', 'full'}:
            raise ValueError('Unexpected category/budget outside inherited protocol')
        identity = {k:requirement[k] for k in ('cohort_id', 'category', 'profile_id', 'budget_id')}
        config_id = digest(identity)
        if b == 'full':
            k = None
        else:
            field = {'B':'actual_B_per_slot', 'half_B':'half_B_per_slot', 'double_B':'double_B_per_slot'}[b]
            k = budgets[c][field]
        state_identity = dict(cohort_id=identity['cohort_id'], category=c, budget_id=b,
                              representation=_state_kind(p), k_per_slot=k)
        state_id = digest(state_identity)
        if config_id not in configs:
            configs[config_id] = dict(identity, config_id=config_id, state_id=state_id, k_per_slot=k,
                state_identity=state_identity, expected_counts=groups[c]['counts'],
                expected_role_digests=groups[c]['ordered_identity_digests'],
                source_root=str(SOURCE_ROOT), run_root=str(run_root),parent_run_root=str(parent_root),
                scientific_identity='fixed_panel_evaluation_previously_seen_supplementary',requirements=[])
        rid=f'requirement_{index+1:03d}'
        configs[config_id]['requirements'].append(rid)
        requirements.append(dict(requirement_id=rid, config_id=config_id, state_id=state_id,
            analysis_source_id=requirement['analysis_source_id'], phase=requirement['phase'],
            shared_execution=len(configs[config_id]['requirements'])>1))
    if len(requirements) != 148 or sum(r['phase']=='quality' for r in requirements)!=123:
        raise ValueError('Inherited coverage differs from the fixed full panel')
    return dict(schema='diream-inference-plan-v1',analysis_source_id='fixed_panel_evaluation',analysis_id='fixed_panel_rebuild',
        parent_run_id='parent_panel_20261025T134519Z',run_root=str(run_root),
        source_files=[dict(path=str(p),sha256=file_sha(p)) for p in (status_path,budget_path,member_path)],
        requirements=requirements,configs=list(configs.values()),
        counts=dict(requirements=len(requirements),quality_requirements=123,timing_requirements=25,
                    unique_quality_configs=len({r['config_id'] for r in requirements if r['phase']=='quality'}),
                    unique_execution_configs=len(configs),unique_state_ids=len({v['state_id'] for v in configs.values()})))


def select_config(doc, config_id):
    matches=[c for c in doc['configs'] if c['config_id']==config_id or c['config_id'].startswith(config_id)]
    if len(matches)!=1:
        raise ValueError('Config ID must identify exactly one fixed configuration')
    return dict(matches[0])


def validate_runtime(context, *, formal=True, timing=False):
    """Use explicit raw-rebuild resource and environment reports."""
    from dataset_bindings import validate_runtime as validate_rebuild
    return validate_rebuild(context,formal=formal,timing=timing)


def _backend():
    return importlib.import_module('matching_backend')


def _state_root(config,context):
    return within(context['run_root'],Path(context['run_root'])/'artifacts/s'/config['state_id'][:16])


def _prediction_root(config,context):
    return within(context['run_root'],Path(context['run_root'])/'artifacts/p'/config['config_id'][:16])


def state_hash(state_dir):
    path=Path(state_dir)/'STATE.json'
    state=read_json(path)
    identity=state.get('state_identity_sha256')
    if not identity or digest({k:v for k,v in state.items() if k!='state_identity_sha256'})!=identity:
        raise ValueError('Semantic state identity has changed')
    return identity


def normalize_output(result,global_identity,actual_views):
    import numpy as np
    fields=np.asarray(result.get('maps256',result.get('fields')))
    score=np.asarray(result.get('object_score',result.get('native_score')))
    if fields.shape!=(actual_views,256,256) or fields.dtype!=np.float32 or not np.isfinite(fields).all():
        raise ValueError('Prediction must contain finite FP32[actual_views,256,256]')
    if score.size!=1 or not np.isfinite(score).all():
        raise ValueError('Native object score must be one finite scalar')
    arrays=dict(maps256=fields,object_score=score.reshape(()),global_identity=np.array(global_identity))
    for k in ('raw32','masks'):
        if k in result and result[k] is not None:
            arrays[k]=np.asarray(result[k])
    return arrays


def predict_with_runtime(config,context,runtime,group,outdir,backend,*,source_kind='RAW_RECOMPUTED'):
    """Core atomic loop. Tests pass explicit runtime/fixtures; CLI never does."""
    outdir=within(context['run_root'],outdir)
    outdir.mkdir(parents=True,exist_ok=True)
    config=dict(config,membership_digest=member_digest(group))
    expected=[(role,row) for role in ('calibration','query') for row in group[role]]
    if set(r['global_identity'] for r in group['reference']) & {r['global_identity'] for _,r in expected}:
        raise ValueError('Fit role leaked into prediction roles')
    with job_lock(outdir):
        complete_path=outdir/'COMPLETE.json'
        if complete_path.exists():
            return validate_records(outdir,read_json(complete_path),group,config)
        progress_path=outdir/'PROGRESS.json'
        progress=read_json(progress_path) if progress_path.exists() else dict(
            config_id=config['config_id'],state_sha256=config['state_sha256'],
            membership_digest=config['membership_digest'],records=[],status='PREDICTING')
        for key in ('config_id','state_sha256','membership_digest'):
            if progress.get(key)!=config[key]:
                raise ValueError('Resume identity does not match current config/state/members')
        records=progress['records']
        if len(records)>len(expected):
            raise ValueError('Resume has extra units')
        for index,record in enumerate(records):
            role,row=expected[index]
            if record['role']!=role or record['global_identity']!=row['global_identity']:
                raise ValueError('Resume member order differs')
            if file_sha(within(outdir,outdir/record['file']))!=record['sha256']:
                raise ValueError('Resume payload changed')
        for index in range(len(records),len(expected)):
            role,row=expected[index]
            # Scientific prediction can validate each source. Timing uses the
            # separate preverification path with verify=False during requests.
            views=backend.decode_unit(row,context,verify=True)
            result=runtime.predict_views(views)
            arrays=normalize_output(result,row['global_identity'],len(views))
            path=outdir/f'{index:05d}.npz'
            atomic_npz(path,arrays)
            records.append(dict(global_identity=row['global_identity'],unit_id=row['global_identity'],
                role=role,file=path.name,sha256=file_sha(path),actual_views=len(views)))
            progress.update(records=records,updated_utc=now())
            atomic_json(progress_path,progress,replace=progress_path.exists())
        seal={k:config[k] for k in ('config_id','category','profile_id','budget_id','cohort_id','state_sha256')}
        seal.update(status='COMPLETE_PREDICTIONS',membership_digest=config['membership_digest'],
                    source_kind=source_kind,records=records,completed_utc=now())
        validate_records(outdir,seal,group,config)
        atomic_json(complete_path,seal)
        return seal


def _load_for_science(config,context):
    validate_runtime(context,formal=True)
    backend=_backend()
    state_dir=_state_root(config,context)
    runtime=backend.load_state(config,context,state_dir)
    config=dict(config,state_sha256=state_hash(state_dir))
    return config,runtime,backend


def fit(config,context):
    validate_runtime(context,formal=True)
    outdir=_state_root(config,context)
    outdir.parent.mkdir(parents=True,exist_ok=True)
    with job_lock(outdir.parent/(outdir.name+'_fitlock')):
        if (outdir/'STATE.json').exists():
            _backend().validate_state_manifest(config,context,outdir)
            return dict(status='VERIFIED_EXISTING_STATE',state_dir=str(outdir))
        return _backend().build_state(config,context,outdir)


def predict(config,context):
    config,runtime,backend=_load_for_science(config,context)
    return predict_with_runtime(config,context,runtime,backend.roles(config,context),
                                _prediction_root(config,context),backend)


def _eval_context(config,context):
    config=dict(config,state_sha256=state_hash(_state_root(config,context)))
    path=Path(context['run_root'])/'protocol/EVALUATOR_SOURCES.json'
    sources=read_json(path)
    return config,sources


def evaluate(config,context):
    receipt=validate_runtime(context,formal=True)
    config,sources=_eval_context(config,context)
    group=_backend().roles(config,context)
    config=dict(config,membership_digest=member_digest(group))
    seal_path=_prediction_root(config,context)/'COMPLETE.json'
    if seal_path.exists() and read_json(seal_path).get('source_kind')=='HISTORICAL_REUSED_VERIFIED':
        reuse=importlib.import_module('state_reuse')
        descriptor=reuse.frozen_descriptor(config,context)
        config['normal_regression_receipt_sha256']=descriptor['normal_regression_receipt']['sha256']
    outdir=within(context['run_root'],Path(context['run_root'])/'results/e'/config['config_id'][:16])
    return importlib.import_module('evaluation').evaluate_predictions(
        _prediction_root(config,context),group,config,sources['evaluator_sources'],
        sources['gt_sources'][config['category']],outdir,protocol_record=receipt)


def _rss(peak=False):
    """Current process working set; no optional psutil dependency."""
    if os.name=='nt':
        import ctypes
        from ctypes import wintypes
        class Counters(ctypes.Structure):
            _fields_=[('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD),
                ('PeakWorkingSetSize',ctypes.c_size_t),('WorkingSetSize',ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage',ctypes.c_size_t),('QuotaPagedPoolUsage',ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage',ctypes.c_size_t),('QuotaNonPagedPoolUsage',ctypes.c_size_t),
                ('PagefileUsage',ctypes.c_size_t),('PeakPagefileUsage',ctypes.c_size_t)]
        c=Counters(); c.cb=ctypes.sizeof(c)
        kernel=ctypes.WinDLL('kernel32',use_last_error=True)
        psapi=ctypes.WinDLL('psapi',use_last_error=True)
        kernel.GetCurrentProcess.restype=wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes=[wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
        psapi.GetProcessMemoryInfo.restype=wintypes.BOOL
        if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb):
            raise OSError('Cannot read actual RSS')
        return int(c.PeakWorkingSetSize if peak else c.WorkingSetSize)
    import resource
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value*(1 if sys.platform=='darwin' else 1024))


def request_schedule(first_three):
    if len(first_three)!=3:
        raise ValueError('Exactly first three ordered calibration units required')
    return [('first',first_three[0])]+[('warmup',first_three[i]) for i in (1,2)]+[
        ('resident',first_three[i%3]) for i in range(27)]


def measure_requests(runtime,first_three,decode,verify,synchronize,memory,clock=time.perf_counter):
    """Testable boundary: verification before timer, validation after timer."""
    for row in first_three:
        verify(row)
    samples=[]
    for index,(phase,row) in enumerate(request_schedule(first_three)):
        synchronize()
        start=clock()
        views=decode(row)
        result=runtime.predict_views(views)
        synchronize()
        elapsed=clock()-start
        normalize_output(result,row['global_identity'],len(views))
        def count(name):
            value=result.get(name)
            return '' if value is None else int(value)
        samples.append(dict(request_index=index,phase=phase,unit_id=row['global_identity'],
            actual_views=len(views),total_request_s=elapsed,rss_bytes=_rss(),**memory(),
            encoder_readings_per_view=count('encoder_readings_per_view'),
            actual_encoder_calls=count('actual_encoder_calls'),actual_PCA_calls=count('actual_PCA_calls'),
            matching_query_vectors=count('matching_query_vectors'),actual_search_invocations=count('actual_search_invocations'),
            distance_queries_per_anchor=count('distance_queries_per_anchor'),
            status='COMPLETED'))
    return samples


def state_accounting(runtime,state_dir):
    """Count loaded feature, metric, index and search payloads separately."""
    import numpy as np
    state=runtime.state
    files={str(Path(state_dir)/'STATE.json')};feature=metric=index=norm=origin=device_feature=0
    if state['profile_family']=='diream':
        engines=[x[0] for x in runtime.resident]
        for slot in state['slots']:
            bank=np.load(slot['bank']['path'],mmap_mode='r',allow_pickle=False)
            ix=np.load(slot['indices']['path'],mmap_mode='r',allow_pickle=False)
            feature+=bank.nbytes;index+=ix.nbytes
            with np.load(slot['model']['path'],allow_pickle=False) as z:
                metric+=sum(z[k].nbytes for k in z.files)
            files.update(slot[k]['path'] for k in ('bank','indices','model'))
        for engine in engines:
            p=engine.payload
            norm+=p['device_norm_bytes'];origin+=p['device_origin_bytes'];device_feature+=p['device_centered_bytes']
    else:
        inner=state['native_state'];folder=Path(state_dir)/state['native_relative']
        files.add(str(folder/'NATIVE_STATE.json'))
        for row in inner['states']:
            bank=np.load(folder/row['bank'],mmap_mode='r',allow_pickle=False)
            ix=np.load(folder/row['indices'],mmap_mode='r',allow_pickle=False)
            feature+=bank.nbytes;index+=ix.nbytes
            files.update([str(folder/row['bank']),str(folder/row['indices'])])
        unique={id(x):x for x in runtime.runtime.engines.values()}
        for engine in unique.values():
            device_feature+=engine.bank.numel()*engine.bank.element_size()
            norm+=engine.norm.numel()*engine.norm.element_size()
    return dict(feature_bytes=int(feature),metric_center_bytes=int(metric),selected_index_payload_bytes=int(index),
        search_norm_bytes=int(norm),search_origin_bytes=int(origin),device_feature_bytes=int(device_feature),
        auxiliary_state_bytes=int(metric+index+norm+origin),
        serialized_bytes=sum(Path(p).stat().st_size for p in files),
        accounting_scope='Inference bank/index/W/center/search arrays; allocator/RSS/model weights reported separately')


def write_csv_new(path,records):
    headers=list(dict.fromkeys(k for row in records for k in row))
    with Path(path).open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,headers);writer.writeheader();writer.writerows(records)


def benchmark_one(config,context):
    validate_runtime(context,formal=True,timing=True)
    outdir=within(context['run_root'],Path(context['run_root'])/'results/t'/config['config_id'][:16])
    if (outdir/'COMPLETE.json').exists():
        raise FileExistsError('Timing series already exists; retain it and use a new run for new timings')
    backend=_backend()
    group=backend.roles(config,context)
    first=group['calibration'][:3]
    import torch
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    init_start=time.perf_counter()
    runtime=backend.load_state(config,context,_state_root(config,context))
    torch.cuda.synchronize()
    initialization_s=time.perf_counter()-init_start
    process_initialization_s=time.perf_counter()-PROCESS_ENTRY_CLOCK
    state_sha=state_hash(_state_root(config,context))
    accounting=state_accounting(runtime,_state_root(config,context))
    environment=dict(python=sys.executable,python_version=sys.version,platform=sys.platform,
        torch_version=torch.__version__,cuda_version=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
        torch_threads=torch.get_num_threads(),tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
        tf32_cudnn=torch.backends.cudnn.allow_tf32)
    result=subprocess.run(['nvidia-smi','--query-gpu=driver_version,name','--format=csv,noheader'],
                          capture_output=True,text=True)
    environment['nvidia_smi']=dict(exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr)
    samples=measure_requests(runtime,first,
        lambda row:backend.decode_unit(row,context,verify=False),
        lambda row:backend.verify_source(row,context),torch.cuda.synchronize,
        lambda:dict(cuda_allocated_bytes=torch.cuda.memory_allocated(),cuda_reserved_bytes=torch.cuda.memory_reserved()))
    expected_readings=backend.profile_spec(config['profile_id'])['readings']
    expected_searches=4 if config['profile_id'].startswith('I_W_') else 1
    for sample in samples:
        if sample['encoder_readings_per_view']!=expected_readings or sample['distance_queries_per_anchor']!=expected_searches:
            raise ValueError('Actual request reading/search count is missing or differs from the registered profile')
        if sample['actual_encoder_calls']!=sample['actual_views']*expected_readings or sample['actual_PCA_calls']=='':
            raise ValueError('Actual encoder/PCA request counts were not recorded')
    import numpy as np
    resident=np.array([r['total_request_s'] for r in samples if r['phase']=='resident'])
    summary=dict(status='COMPLETED',run_id=Path(context['run_root']).name,
        profile_id=config['profile_id'],category=config['category'],quality_cohort_id=config['cohort_id'],
        budget_id=config['budget_id'],state_sha256=state_sha,process_id=os.getpid(),
        initialization_s=initialization_s,process_initialization_s=process_initialization_s,
        first_request_s=samples[0]['total_request_s'],resident_n=27,
        resident_median_s=float(np.median(resident)),resident_p95_s=float(np.quantile(resident,.95)),
        cuda_allocated_peak_bytes=torch.cuda.max_memory_allocated(),
        cuda_reserved_peak_bytes=torch.cuda.max_memory_reserved(),rss_peak_bytes=_rss(peak=True),
        rss_sampled_max_bytes=max(s['rss_bytes'] for s in samples),rss_peak_scope='OS_process_peak_including_initialization_and_requests',
        series_id=Path(context['run_root']).name+'_raw_request_v1',
        query_feature_cache=False,query_mask_cache=False,input_SHA_inside_timer=False,
        RGB_decode_inside_timer=True,output_validation_inside_timer=False,persistence_inside_timer=False,
        single_profile_process=True,source_kind='RAW_RECOMPUTED',**accounting)
    outdir.mkdir(parents=True,exist_ok=True)
    atomic_json(outdir/'ENVIRONMENT.json',environment)
    summary['environment_sha256']=file_sha(outdir/'ENVIRONMENT.json')
    summary['timing_scope']='RGB_decode+required_encoder_masks+transform+search+postprocess+native_score'
    quality_path=Path(context['run_root'])/'results/e'/config['config_id'][:16]/'SUMMARY.csv'
    if quality_path.exists():
        with quality_path.open(encoding='utf-8-sig',newline='') as f:
            quality=list(csv.DictReader(f))
        if len(quality)!=1 or quality[0]['state_sha256']!=state_sha:
            raise ValueError('Quality/timing state identity differs')
        summary['category_unit_ap']=quality[0]['mean_object_pixel_AP']
        summary['quality_source']='same_profile_state_verified_evaluation'
    else:
        summary['category_unit_ap']=''
        summary['quality_source']='NOT_AVAILABLE_NO_QUALITY_POINT_INFERRED'
    for sample in samples:
        sample.update(run_id=Path(context['run_root']).name,series_id=summary['series_id'],
            category=config['category'],profile_id=config['profile_id'],budget_id=config['budget_id'],
            state_sha256=state_sha,process_id=os.getpid())
    atomic_json(outdir/'REQUESTS.json',dict(samples=samples,config_id=config['config_id'],state_sha256=state_sha))
    atomic_json(outdir/'TIMING_SUMMARY.json',summary)
    write_csv_new(outdir/'REQUESTS.csv',samples)
    write_csv_new(outdir/'TIMING_SUMMARY.csv',[summary])
    atomic_json(outdir/'COMPLETE.json',dict(status='COMPLETE_TIMING',
        requests_sha256=file_sha(outdir/'REQUESTS.json'),summary_sha256=file_sha(outdir/'TIMING_SUMMARY.json')))
    return summary


def bootstrap(doc,context):
    receipt=validate_runtime(context,formal=True)
    requirements=[]
    with (Path(context['parent_run_root'])/'results/PAIRED_CONTRASTS.csv').open(encoding='utf-8-sig',newline='') as f:
        seen=set()
        for row in csv.DictReader(f):
            if row['logical_contrast_id'] not in seen:
                requirements.append(row); seen.add(row['logical_contrast_id'])
    sources=read_json(Path(context['run_root'])/'protocol/EVALUATOR_SOURCES.json')['evaluator_sources']
    dirs=[Path(context['run_root'])/'results/e'/c['config_id'][:16] for c in doc['configs']]
    dirs=[p for p in dirs if (p/'COMPLETE.json').exists()]
    return importlib.import_module('evaluation').bootstrap_contrasts(dirs,requirements,sources,
        within(context['run_root'],Path(context['run_root'])/'results/bootstrap'),protocol_record=receipt)






