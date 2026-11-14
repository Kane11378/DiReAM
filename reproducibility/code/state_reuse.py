"""Validate and materialize explicitly supplied historical predictions.

Reuse preserves the original arrays and requires exact source, state, member
and normal-regression identities. No model, annotation or new prediction is
created by the copy operation. Raw rebuilding uses image_inference.py.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import numpy as np

from state_store import (atomic_json,atomic_npz,digest,file_sha,job_lock,
                          member_digest,now,read_json,validate_records,within)

SUPPORTED = {'M_W_U':'within_M05','M_W_FC4':'within_M05_FC4','N_W_F0':'within_N01_fast_F0'}
PADDY = {'M_W_U':'within_M05','N_W_U':'within_N01_fast','I_W_U':'within_I01'}


def pin(path):
    path=Path(path).resolve()
    return dict(path=str(path),bytes=path.stat().st_size,sha256=file_sha(path))


def checked(row):
    p=Path(row['path'])
    if p.stat().st_size!=row['bytes'] or file_sha(p)!=row['sha256']:
        raise ValueError(f'Historical source changed: {p}')
    return p


def legacy_profile(config):
    if config['budget_id']!='B':
        raise ValueError('Historical reuse is limited to original exact B states')
    mapping=PADDY if config['category']=='paddy' else SUPPORTED
    if config['profile_id'] not in mapping:
        raise ValueError('No historical source mapping for this exact profile; Native full requires its own regression and state-equivalence check')
    return mapping[config['profile_id']]


def _validate_arrays(data,role,group):
    expected=[r['global_identity'] for r in group[role]]
    idkey='normal_ids' if role=='calibration' else 'query_ids'
    if idkey not in data:
        idkey='identities'
    if idkey not in data or [str(x) for x in data[idkey]]!=expected:
        raise ValueError('Historical aggregate member/order mismatch')
    count=len(expected)
    views=len(group[role][0]['real_view_boxes'])
    if len(group['calibration'])!=64:
        raise ValueError('Historical reuse requires the original 64 calibration members')
    if 'maps256' not in data or data['maps256'].shape!=(count,views,256,256) or data['maps256'].dtype!=np.float32:
        raise ValueError('Historical maps have wrong count/grid/dtype')
    if 'object_score' not in data or data['object_score'].shape!=(count,) or not np.isfinite(data['object_score']).all():
        raise ValueError('Historical native scores invalid')
    if not np.isfinite(data['maps256']).all():
        raise ValueError('Historical maps nonfinite')
    if 'raw32' in data and (data['raw32'].shape!=(count,views,32,32) or data['raw32'].dtype!=np.float32 or not np.isfinite(data['raw32']).all()):
        raise ValueError('Historical raw32 invalid')
    return data


def _verify_regression(config,group,state_sha,receipt_path,calibration,baseline_path=None):
    """Verify the specific fixed-three raw regression and its actual payloads."""
    path=Path(receipt_path)
    doc=read_json(path)
    if doc.get('status')!='PASSED' or doc.get('GT_reads')!=0 or len(doc.get('rows',[]))!=3:
        raise ValueError('Specific real normal regression has not passed')
    for index,(row,unit) in enumerate(zip(doc['rows'],group['calibration'][:3])):
        for key in ('category','profile_id','budget_id'):
            if row.get(key)!=config[key]:raise ValueError('Regression belongs to another profile/category/budget')
        if row.get('state_sha256')!=state_sha or row.get('unit_id')!=unit['global_identity'] or row.get('status')!='PASSED':
            raise ValueError('Regression state/member/status differs')
        if row.get('atol')!=1e-6 or row.get('rtol')!=1e-5 or row.get('GT_reads')!=0:
            raise ValueError('Regression tolerances or data role changed')
        if row.get('source_kind')!='RAW_RGB_RECOMPUTED_NORMAL_TECHNICAL_REGRESSION':
            raise ValueError('Only actual raw-RGB normal regression permits reuse')
        masked=config['profile_id'] in {'M_W_FC4','N_W_F0'}
        if (masked and row.get('mask_equal') is not True) or (not masked and row.get('mask_equal')!='NOT_APPLICABLE_UNMASKED'):
            raise ValueError('Required exact normal mask regression missing')
        expected_reads=(1 if config['profile_id'].startswith('N_') else 4)*len(unit['real_view_boxes'])
        if row.get('actual_encoder_calls')!=expected_reads:
            raise ValueError('Regression did not execute the registered real readings')
        if row.get('actual_PCA_calls')!=(expected_reads if masked else 0):
            raise ValueError('Regression did not execute the registered PCA calls')
        if baseline_path is not None and Path(row['baseline_path']).resolve()!=Path(baseline_path).resolve():
            raise ValueError('Specific normal regression used a different historical calibration source')
        payload=path.parent/f'normal_{index:03d}.npz'
        if file_sha(payload)!=row['prediction_sha256'] or file_sha(row['baseline_path'])!=row['baseline_sha256']:
            raise ValueError('Regression input/output hashes changed')
        with np.load(payload,allow_pickle=False) as actual:
            if not np.allclose(actual['maps256'],calibration['maps256'][index],atol=1e-6,rtol=1e-5):
                raise ValueError('Actual normal map no longer matches selected historical aggregate')
            if not np.isclose(float(actual['object_score']),float(calibration['object_score'][index]),atol=1e-6,rtol=1e-5):
                raise ValueError('Actual normal score no longer matches selected historical aggregate')
    return pin(path)


def prepare_source(config,context,regression_receipt):
    """CPU-only pre-freeze provenance check. Does not generate predictions/AP."""
    import matching_backend as backend
    import inference_runner as runner
    legacy=legacy_profile(config)
    group=backend.roles(config,context)
    state_dir=runner._state_root(config,context)
    state=backend.validate_state_manifest(config,context,state_dir)
    state_sha=state['state_identity_sha256']
    source=Path(context['source_root'])
    if config['category']=='paddy':
        root=Path(context['historical_prediction_roots'][config['category']]).resolve()
        completion=root/(legacy+'_COMPLETE.json')
        complete=read_json(completion)
        metadata=[completion,root/'BINDING.json',root/'PREDICTIONS_SEALED.json',root/'CONTROLLED_COMPLETE.json']
        bind=read_json(root/'BINDING.json');sealed=read_json(root/'PREDICTIONS_SEALED.json')
        if bind.get('category')!='paddy' or bind.get('library')!='A176' or legacy not in bind.get('methods',[]):
            raise ValueError('Historical Paddy cohort/profile differs')
        if bind.get('query_ids')!=[r['global_identity'] for r in group['query']] or sealed.get('binding_sha256')!=file_sha(root/'BINDING.json'):
            raise ValueError('Historical Paddy query or binding seal differs')
        output_pins={r['file']:r['sha256'] for r in complete['outputs']}
        controlled={r['file']:r['sha256'] for r in read_json(root/'CONTROLLED_COMPLETE.json')['outputs']}
        if any(controlled.get(k)!=v for k,v in output_pins.items()):
            raise ValueError('Historical Paddy per-profile/control seals differ')
    else:
        root=Path(context['historical_prediction_roots'][config['category']]).resolve()
        completion=root/'COMPLETE.json';complete=read_json(completion)
        metadata=[completion,root/'BINDING.json',root/'ENCODED.json',root/'FIT.json']
        parent_pins=backend._parent_pins(context)
        for p in metadata:
            expected=parent_pins.get(str(p.resolve()))
            if expected is None or file_sha(p)!=expected:raise ValueError('Historical metadata differs from parent pin')
        bind=read_json(root/'BINDING.json')
        if bind.get('category')!=config['category'] or legacy not in bind.get('profiles',[]):
            raise ValueError('Historical full-panel method binding differs')
        for key,p in [('binding_sha256',root/'BINDING.json'),('encoding_sha256',root/'ENCODED.json'),('fit_sha256',root/'FIT.json')]:
            if complete.get(key)!=file_sha(p):raise ValueError('Historical complete metadata chain differs')
        fit=read_json(root/'FIT.json')['metadata']['within_N01_fast' if legacy.startswith('within_N') else 'within_M05']
        if fit['bank_sha256']!=state['slots'][0]['bank']['sha256'] or fit['selection']['indices_sha256']!=state['slots'][0]['indices']['sha256']:
            raise ValueError('Historical bank/indices differ from active bound state')
        output_pins={r['file']:r['sha256'] for r in complete['files']}
    role_pins={};loaded={}
    for role,filename in [('calibration',legacy+'_normal_calibration.npz'),('query',legacy+'.npz')]:
        path=root/filename
        if filename not in output_pins or file_sha(path)!=output_pins[filename]:
            raise ValueError('Historical complete aggregate SHA mismatch')
        with np.load(path,allow_pickle=False) as z:
            loaded[role]={k:z[k].copy() for k in z.files}
        _validate_arrays(loaded[role],role,group)
        role_pins[role]=pin(path)
    regression=_verify_regression(config,group,state_sha,regression_receipt,loaded['calibration'],role_pins['calibration']['path'])
    return dict(config_id=config['config_id'],category=config['category'],profile_id=config['profile_id'],
        budget_id=config['budget_id'],cohort_id=config['cohort_id'],legacy_profile=legacy,
        state_sha256=state_sha,state_manifest=pin(state_dir/'STATE.json'),membership_digest=member_digest(group),
        normal_regression_receipt=regression,metadata=[pin(p) for p in metadata],roles=role_pins,
        source_kind='HISTORICAL_REUSED_VERIFIED',status='PREPARED_CPU_IDENTITY_NOT_YET_MATERIALIZED')


def frozen_descriptor(config,context):
    import inference_runner as runner
    runner.validate_runtime(context,formal=True)
    root=Path(context['run_root'])
    path=root/'protocol/HISTORICAL_REUSE_PINS.json'
    protocol=read_json(root/'protocol/PROTOCOL_RESOLVED.json')
    if protocol.get('historical_reuse_pins_sha256')!=file_sha(path):
        raise ValueError('Historical reuse sources are absent from/changed after the actual scientific freeze')
    doc=read_json(path)
    if doc.get('status')!='PREPARED_VERIFIED_HISTORICAL_SOURCES':raise ValueError('Reuse source plan is not prepared')
    found=[r for r in doc['configs'] if r['config_id']==config['config_id']]
    if len(found)!=1:raise ValueError('This exact profile has no frozen reuse source')
    return found[0]


def _materialize(config,group,descriptor,outdir):
    """No-GPU copy loop; only called after production preconditions above."""
    outdir=Path(outdir);outdir.mkdir(parents=True,exist_ok=True)
    with job_lock(outdir):
        complete_path=outdir/'COMPLETE.json'
        if complete_path.exists():
            return validate_records(outdir,read_json(complete_path),group,config)
        progress_path=outdir/'PROGRESS.json'
        progress=read_json(progress_path) if progress_path.exists() else dict(
            config_id=config['config_id'],state_sha256=config['state_sha256'],membership_digest=config['membership_digest'],
            source_descriptor_digest=digest(descriptor),records=[],source_kind='HISTORICAL_REUSED_VERIFIED')
        for key in ('config_id','state_sha256','membership_digest'):
            if progress.get(key)!=config[key]:raise ValueError('Historical reuse resume identity differs')
        if progress.get('source_descriptor_digest')!=digest(descriptor):raise ValueError('Historical reuse source plan changed')
        expected=[(role,index,row) for role in ('calibration','query') for index,row in enumerate(group[role])]
        records=progress['records']
        if len(records)>len(expected):raise ValueError('Reuse has extra progress units')
        for rec,(role,index,row) in zip(records,expected):
            if rec['global_identity']!=row['global_identity'] or rec['role']!=role or file_sha(outdir/rec['file'])!=rec['sha256']:
                raise ValueError('Historical reuse previous unit changed')
        arrays={}
        for role in ('calibration','query'):
            with np.load(checked(descriptor['roles'][role]),allow_pickle=False) as z:
                arrays[role]={k:z[k].copy() for k in z.files}
            _validate_arrays(arrays[role],role,group)
        for ordinal in range(len(records),len(expected)):
            role,index,row=expected[ordinal]
            original=arrays[role]
            payload=dict(maps256=original['maps256'][index],object_score=original['object_score'][index],
                         global_identity=np.array(row['global_identity']))
            if 'raw32' in original:payload['raw32']=original['raw32'][index]
            path=outdir/f'{ordinal:05d}.npz';atomic_npz(path,payload)
            records.append(dict(role=role,unit_id=row['global_identity'],global_identity=row['global_identity'],
                actual_views=len(row['real_view_boxes']),file=path.name,sha256=file_sha(path),
                original_aggregate_sha256=descriptor['roles'][role]['sha256'],original_aggregate_index=index))
            progress.update(records=records,updated_utc=now())
            atomic_json(progress_path,progress,replace=progress_path.exists())
        seal={k:config[k] for k in ('config_id','category','profile_id','budget_id','cohort_id','state_sha256','membership_digest')}
        seal.update(status='COMPLETE_PREDICTIONS',source_kind='HISTORICAL_REUSED_VERIFIED',records=records,
            normal_regression_receipt_sha256=descriptor['normal_regression_receipt']['sha256'],
            historical_source_descriptor=descriptor,historical_source_descriptor_sha256=digest(descriptor),
            completed_utc=now(),new_model_encodings=0,rounded_metrics_used=False)
        if config.get('synthetic_fixture'):seal['synthetic_fixture']=True
        validate_records(outdir,seal,group,config)
        atomic_json(complete_path,seal)
        return seal


def reuse_existing(config,context,regression_receipt=None,outdir=None):
    import matching_backend as backend
    import inference_runner as runner
    legacy_profile(config)  # No implicit fallback, including Native full.
    descriptor=frozen_descriptor(config,context)
    if regression_receipt and file_sha(regression_receipt)!=descriptor['normal_regression_receipt']['sha256']:
        raise ValueError('Specific regression receipt differs from frozen source plan')
    receipt=checked(descriptor['normal_regression_receipt'])
    actual=prepare_source(config,context,receipt)
    if actual!=descriptor:raise ValueError('Historical source/state/member/normal receipt differs after freeze')
    group=backend.roles(config,context)
    config=dict(config,state_sha256=descriptor['state_sha256'],membership_digest=descriptor['membership_digest'],
                normal_regression_receipt_sha256=descriptor['normal_regression_receipt']['sha256'])
    target=runner._prediction_root(config,context) if outdir is None else within(context['run_root'],outdir)
    return _materialize(config,group,descriptor,target)




