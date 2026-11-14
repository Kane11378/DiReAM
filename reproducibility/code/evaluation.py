"""Validate prediction records and compute whole-unit evaluation statistics.

Annotation pixels are opened after calibration and query predictions pass their
state, member and byte checks. Generated-array fixtures use a separate entry
point. Expected result tables are never numerical inputs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import sys
import tarfile
import uuid

import numpy as np
from PIL import Image

VISA12 = ('candle','capsules','cashew','chewinggum','fryum','macaroni1','macaroni2',
          'pcb1','pcb2','pcb3','pcb4','pipe_fryum')
PRIOR6 = ('candle','capsules','cashew','pipe_fryum','pcb1','pcb2')
REMAINING6 = tuple(c for c in VISA12 if c not in PRIOR6)
FIVE_ENDPOINTS = ('unit_ap','pixel_ap','pixel_auroc','image_ap','image_auroc')
PROFILE_IDS = {'N_W_U','I_W_U','M_W_U','N_W_F0','I_W_FC4','M_W_FC4',
               'NativeNoRot_full','NativeNoRot_exact','NativeNoRot_known_slot_U',
               'NativeNoRot_pooled_U','NativeNoRot_known_slot_U_full','NativeNoRot_pooled_U_full'}
from dataset_bindings import sources as _portable_sources
SOURCE_DEFAULTS = _portable_sources()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as src:
        return hashlib.file_digest(src, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')


def write_new(path, value):
    with Path(path).open('xb') as out:
        out.write(json_bytes(value))


def csv_new(path, records, headers=None):
    keys = list(headers or dict.fromkeys(k for row in records for k in row))
    with Path(path).open('x', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=keys)
        writer.writeheader()
        writer.writerows(records)


def _atomic_directory(output_dir, populate):
    target = Path(output_dir).resolve()
    require(not target.exists(), f'Output already exists: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = target.with_name(target.name+'.partial.'+uuid.uuid4().hex)
    stage.mkdir()
    # Failed work is deliberately retained, never promoted on partial success.
    result = populate(stage)
    os.rename(stage, target)
    return result


def membership_digest(group):
    ordered = {role: [row['global_identity'] for row in group[role]]
               for role in ('reference','calibration','query')}
    return hashlib.sha256(json.dumps(ordered,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def prediction_membership_digest(group):
    ordered = [(role, row['global_identity']) for role in ('calibration','query') for row in group[role]]
    return hashlib.sha256(json.dumps(ordered,separators=(',',':')).encode()).hexdigest()


def _pinned_module(sources, name):
    require(name in sources, f'Missing pinned evaluator source: {name}')
    pin = sources[name]
    require(sha(pin['path']) == pin['sha256'], f'Evaluator source SHA changed: {name}')
    module_name = '_evaluation_pinned_'+name+'_'+pin['sha256'][:16]
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, pin['path'])
        require(spec is not None and spec.loader is not None, 'Cannot load pinned evaluator')
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]


def _verify_protocol(receipt):
    require(receipt is not None, 'Verified fixed-protocol record required before annotation evaluation')
    receipt = read(receipt) if isinstance(receipt, (str,Path)) else receipt
    require(receipt.get('status') == 'PROTOCOL_VERIFIED', 'Protocol record is not verified')
    freeze_path = Path(receipt['freeze_path'])
    freeze = read(freeze_path)
    require(freeze.get('status') == 'FROZEN', 'Valid new scientific freeze required')
    protocol_path = Path(receipt['protocol_path'])
    digest = sha(protocol_path)
    if 'reader_metadata' in freeze or 'public_projection' in freeze:
        from public_projection import ROOT, protocol_pair
        current=protocol_pair(ROOT,freeze_path,protocol_path)
        require(digest == receipt['protocol_sha256'] == current, 'Public protocol/freeze hash mismatch')
    else:
        require(digest == receipt['protocol_sha256'] == freeze.get('protocol_sha256',freeze.get('resolved_sha256')), 'Scientific protocol/freeze hash mismatch')


def _safe_prediction_path(root, relative):
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe prediction path')
    p = (root/rel).resolve()
    require(p.is_relative_to(root) and p.is_file() and not p.is_symlink(), 'Prediction path missing/escapes root')
    return p


def _array_alias(data, left, right):
    require(left in data or right in data, f'Prediction missing {left}/{right}')
    if left in data and right in data:
        require(np.array_equal(data[left],data[right]), f'Conflicting prediction aliases {left}/{right}')
    return data[left] if left in data else data[right]


def validate_prediction_complete(prediction_dir, group, config):
    """Validate all sealed cal+query predictions without opening any GT.

    ``membership_digest`` is the driver's full role manifest digest; it is
    compared exactly to config. Ordered member identities are independently
    checked against the supplied original role rows, not trusted from a digest.
    """
    root = Path(prediction_dir).resolve()
    seal_path = root/'COMPLETE.json'
    require(seal_path.is_file(), 'Full prediction COMPLETE.json is missing')
    seal = read(seal_path)
    require(seal.get('status') == 'COMPLETE_PREDICTIONS', 'Prediction inventory is not complete')
    require(config.get('profile_id') in PROFILE_IDS, 'Unregistered evaluation profile')
    for key in ('config_id','category','profile_id','budget_id','cohort_id','state_sha256','membership_digest'):
        require(key in config and seal.get(key) == config[key], f'Prediction binding mismatch: {key}')
    require(seal.get('source_kind') in {'RAW_RECOMPUTED','HISTORICAL_REUSED_VERIFIED'}, 'Unverified prediction source kind')
    if seal['source_kind'] == 'HISTORICAL_REUSED_VERIFIED':
        require(seal.get('normal_regression_receipt_sha256') == config.get('normal_regression_receipt_sha256')
                and bool(config.get('normal_regression_receipt_sha256')), 'Historical reuse needs successful normal regression receipt')
    for role in ('reference','calibration','query'):
        require(role in group and len(group[role]) > 0, f'Missing exact role: {role}')
    require(len(group['calibration']) == 64, 'Fixed production calibration role must contain exactly 64 units')
    ids = {role:[r['global_identity'] for r in group[role]] for role in ('reference','calibration','query')}
    require(all(len(v)==len(set(v)) for v in ids.values()), 'Duplicate role identities')
    require(all(not(set(ids[a]) & set(ids[b])) for a,b in [('reference','calibration'),('reference','query'),('calibration','query')]), 'Role leakage')
    expected = [(role,r) for role in ('calibration','query') for r in group[role]]
    require(len(seal.get('records',[])) == len(expected), 'Incomplete calibration/query prediction inventory')
    loaded = []
    for (role,row), record in zip(expected,seal['records']):
        identity = row['global_identity']
        require(record.get('global_identity') == identity and record.get('role') == role, 'Prediction member order/role changed')
        require(record.get('actual_views') == len(row['real_view_boxes']), 'Prediction acquisition-slot count changed')
        path = _safe_prediction_path(root,record['file'])
        require(sha(path) == record['sha256'], 'Prediction file hash changed')
        with np.load(path, allow_pickle=False) as src:
            data = {k:src[k].copy() for k in src.files}
        require('global_identity' in data and str(data['global_identity'].item()) == identity,
                'Prediction NPZ unit identity missing/differs')
        maps = _array_alias(data,'maps256','fields')
        score = _array_alias(data,'object_score','native_score')
        require(maps.dtype == np.float32 and maps.shape == (len(row['real_view_boxes']),256,256), 'Prediction field shape/dtype differs')
        require(np.isfinite(maps).all() and np.asarray(score).size == 1 and np.isfinite(score).all(), 'Nonfinite/invalid prediction')
        if role == 'calibration':
            require(row.get('label_stratum') == 'normal', 'Calibration role contains nonnormal unit')
        loaded.append(dict(maps256=maps,object_score=float(np.asarray(score).reshape(()))))
    return seal, loaded


def _read_gt(group, source):
    'Called only after complete prediction seals and fixed-protocol validation.'
    require(source.get('kind') in {'directory','tar','tar_gz'}, 'Explicit GT directory/tar/tar_gz required')
    path = Path(source['path']).resolve()
    if source['kind'] in {'tar','tar_gz'}:
        require(source.get('sha256') and sha(path) == source['sha256'], 'GT archive SHA mismatch')
    compressed_members={}
    if source['kind']=='tar_gz':
        wanted={r['gt_member'] for r in group['query'] if r['label_stratum']=='anomaly'}
        # One bounded streaming pass through the pinned original archive, not
        # one gzip rewind/decompression per anomalous unit.
        with tarfile.open(path,mode='r|gz') as tar:
            for member in tar:
                if member.name not in wanted:continue
                require(member.isfile() and not member.issym() and not member.islnk(),'GT tar member not regular')
                require(member.size<128*1024**2,'GT tar member unexpectedly large')
                compressed_members[member.name]=tar.extractfile(member).read()
                if len(compressed_members)==len(wanted):break
        require(set(compressed_members)==wanted,'Missing explicit GT member in compressed tar')
    all_gt=[]; anomalous=[]
    archive = path.open('rb') if source['kind']=='tar' else None
    try:
        for row in group['query']:
            w,h=row['composite_wh']
            positive=row['label_stratum']=='anomaly'
            anomalous.append(positive)
            if positive:
                if source['kind']=='tar_gz':
                    b=compressed_members[row['gt_member']]
                elif archive is not None:
                    if row.get('gt_offset_data_uncompressed') is not None:
                        require(row.get('gt_member_bytes',0)>0, 'Incomplete GT offsets')
                        archive.seek(row['gt_offset_data_uncompressed'])
                        b=archive.read(row['gt_member_bytes'])
                        require(len(b)==row['gt_member_bytes'], 'GT payload truncated')
                    else:
                        archive.seek(0)
                        with tarfile.open(fileobj=archive,mode='r:') as tf:
                            member=tf.getmember(row['gt_member'])
                            require(member.isfile(), 'GT member is not regular')
                            b=tf.extractfile(member).read()
                else:
                    relative=source.get('member_paths',{}).get(row['global_identity'],row.get(source.get('member_key','gt_member')))
                    require(relative is not None,'Missing explicit GT member path')
                    p=(path/relative).resolve()
                    require(p.is_relative_to(path) and not p.is_symlink(), 'GT member escapes source')
                    b=p.read_bytes()
                    require(row.get('GT_payload_sha256') is not None, 'GT payload SHA is required for directory source')
                expected_hash=row.get('GT_payload_sha256')
                if expected_hash:
                    require(hashlib.sha256(b).hexdigest()==expected_hash,'GT member SHA mismatch')
                with Image.open(io.BytesIO(b)) as image:
                    x=np.asarray(image)
                x=np.any(x>0,axis=2) if x.ndim==3 else x>0
            else:
                x=np.zeros((h,w),dtype=bool)
            require(x.shape==(h,w), 'Original GT geometry differs')
            views=[x[y0:y1,x0:x1] for x0,y0,x1,y1 in row['real_view_boxes']]
            require(all(v.size>0 for v in views), 'Empty acquisition view')
            maps=np.stack([np.asarray(Image.fromarray(v.astype(np.uint8)).resize((256,256),Image.Resampling.NEAREST))>0 for v in views])
            all_gt.append(maps)
    finally:
        if archive is not None:archive.close()
    return np.stack(all_gt),np.asarray(anomalous,dtype=bool)


def _compute_evaluation(group, config, seal, loaded, gt, bad, binary_metrics):
    ncal=len(group['calibration'])
    fields=np.stack([z['maps256'] for z in loaded])
    scores=np.asarray([z['object_score'] for z in loaded],dtype=np.float64)
    require(len(scores)==ncal+len(group['query']), 'Prediction counts changed')
    # With 64 calibration normals, the q99-higher threshold is their maximum.
    threshold=float(max(scores[:ncal]))
    alarm=scores[ncal:]>threshold
    query=fields[ncal:]
    require(gt.dtype==bool and gt.shape==query.shape and bad.shape==(len(query),), 'GT/query whole-unit geometry differs')
    units=[]
    for i,row in enumerate(group['query']):
        valid=bool(bad[i] and gt[i].any())
        m=binary_metrics(gt[i],query[i]) if valid else {}
        reason='eligible_abnormal_positive_GT' if valid else 'normal_unit' if not bad[i] else 'abnormal_zero_positive_GT'
        u=dict(category=config['category'],method=config['profile_id'],profile_id=config['profile_id'],
            budget_id=config['budget_id'],cohort_id=config['cohort_id'],config_id=config['config_id'],
            state_sha256=config['state_sha256'],global_identity=row['global_identity'],
            object_anomalous=bool(bad[i]),actual_views=len(row['real_view_boxes']),
            pixel_AP_eligible=valid,AP_eligibility_reason=reason,object_pixel_AP=m.get('AP'),
            native_alarm=int(alarm[i]),native_object_score=float(scores[ncal+i]),
            native_object_threshold=threshold,source_kind=seal['source_kind'])
        for b in [.01,.05]:
            coverage=None
            if valid:
                order=np.argsort(-query[i].reshape(-1),kind='stable')
                yy=gt[i].ravel()
                coverage=float(yy[order[:math.floor(len(yy)*b)]].sum()/int(yy.sum()))
            u[f'coverage_{b:g}']=coverage
            u[f'gated_coverage_{b:g}']=None if coverage is None else coverage*int(alarm[i])
        units.append(u)
    pool=binary_metrics(gt,query)
    image=binary_metrics(bad,scores[ncal:])
    ap=[u['object_pixel_AP'] for u in units if u['object_pixel_AP'] is not None]
    summary=dict(category=config['category'],method=config['profile_id'],profile_id=config['profile_id'],
        budget_id=config['budget_id'],cohort_id=config['cohort_id'],config_id=config['config_id'],
        state_sha256=config['state_sha256'],reference_count=len(group['reference']),calibration_count=ncal,
        queries=len(bad),query_normal_n=int((~bad).sum()),query_anomalous_n=int(bad.sum()),ap_eligible_n=len(ap),
        mean_object_pixel_AP=float(np.mean(ap)) if ap else None,pooled_pixel_AP=pool['AP'],
        pooled_pixel_AUROC=pool['AUROC'],object_AP=image['AP'],object_AUROC=image['AUROC'],
        sensitivity=float(alarm[bad].mean()) if bad.any() else None,
        false_alarm_rate=float(alarm[~bad].mean()) if (~bad).any() else None,
        true_alarms=int(alarm[bad].sum()),false_alarms=int(alarm[~bad].sum()),
        source_kind=seal['source_kind'],threshold_rule='max_of_64_normal_scores_strict_greater')
    for key in ['coverage_0.01','coverage_0.05','gated_coverage_0.01','gated_coverage_0.05']:
        values=[u[key] for u in units if u[key] is not None]
        summary[key]=float(np.mean(values)) if values else None
    return summary,units


def evaluate_predictions(prediction_dir, group, config, evaluator_sources, gt_source,
                         output_dir, protocol_record=None, synthetic_fixture=False):
    """Production evaluation; complete seals and successful P0 freeze required."""
    require(not synthetic_fixture, 'Use fixture subcommand/API; production has no synthetic fallback')
    _verify_protocol(protocol_record)
    seal,loaded=validate_prediction_complete(prediction_dir,group,config)
    require(not seal.get('synthetic_fixture',False),'Synthetic prediction cannot enter production GT evaluation')
    stats=_pinned_module(evaluator_sources,'statistics_core')
    gt,bad=_read_gt(group,gt_source)
    summary,units=_compute_evaluation(group,config,seal,loaded,gt,bad,stats.binary_metrics)
    def populate(stage):
        csv_new(stage/'SUMMARY.csv',[summary])
        csv_new(stage/'PER_UNIT.csv',units)
        complete=dict(status='COMPLETE_EVALUATION',scientific_source_id='fixed_panel_evaluation',
            config=config,prediction_complete_sha256=sha(Path(prediction_dir)/'COMPLETE.json'),
            membership_digest=config['membership_digest'],source_kind=seal['source_kind'],
            evaluator_sources=evaluator_sources,outputs=[dict(file=p.name,sha256=sha(p)) for p in sorted(stage.glob('*.csv'))],
            expected_tables_used_as_inputs=False,GT_opened_only_after_complete_predictions=True)
        write_new(stage/'COMPLETE.json',complete)
        return complete
    return _atomic_directory(output_dir,populate)


def _csv_rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as src:
        return list(csv.DictReader(src))


def validate_evaluation_complete(directory):
    root=Path(directory).resolve()
    seal=read(root/'COMPLETE.json')
    require(seal.get('status') in {'COMPLETE_EVALUATION','SYNTHETIC_EVALUATION_COMPLETE'}, 'Evaluation inventory incomplete')
    for record in seal['outputs']:
        p=_safe_prediction_path(root,record['file'])
        require(sha(p)==record['sha256'],'Evaluation CSV hash mismatch')
    summary=_csv_rows(root/'SUMMARY.csv')
    units=_csv_rows(root/'PER_UNIT.csv')
    require(len(summary)==1 and len(units)>0,'Evaluation missing category/whole-unit rows')
    ids=[u['global_identity'] for u in units]
    require(len(ids)==len(set(ids)),'Duplicate evaluated whole unit')
    return seal,summary[0],units


def _number(x):
    return None if x in ('',None,'None') else float(x)


def _categories_for_requirement(row):
    category=row['category_or_macro']
    if category=='all12':return list(VISA12)
    if category=='prior_used6':return list(PRIOR6)
    if category=='remaining6':return list(REMAINING6)
    require(category in VISA12 or category=='paddy','Unknown fixed contrast category/stratum')
    return [category]


def bootstrap_contrasts(evaluation_dirs, requirements, evaluator_sources, outdir,
                        protocol_record=None, synthetic_fixture=False):
    """Compute paired whole-unit intervals for each fixed contrast.

    Fail closed on a missing fixed category or mismatched unit identity. Logical
    duplicate requirements reuse one estimate; they are not independent results.
    """
    require(not synthetic_fixture,'Use fixture entry point; production statistics has no synthetic fallback')
    _verify_protocol(protocol_record)
    return _bootstrap_impl(evaluation_dirs,requirements,evaluator_sources,outdir,False)


def _bootstrap_impl(evaluation_dirs, requirements, evaluator_sources, outdir, fixture):
    panel=_pinned_module(evaluator_sources,'panel_statistics')
    evaluations={}
    for directory in evaluation_dirs:
        seal,summary,units=validate_evaluation_complete(directory)
        require(fixture or seal['status']=='COMPLETE_EVALUATION','Synthetic evaluation cannot enter scientific statistics')
        key=(summary['cohort_id'],summary['category'],summary['profile_id'],summary['budget_id'])
        require(key not in evaluations,'Duplicate evaluation state/config')
        evaluations[key]=(seal,summary,units)
    contrasts=[];macros=[];requirements_out=[];estimates={}
    for requirement in requirements:
        cats=_categories_for_requirement(requirement)
        l,r=requirement['left_profile'],requirement['right_profile']
        lb,rb=requirement['left_budget'],requirement['right_budget']
        cohort=requirement['cohort_id']
        selected=[];summary_pairs=[]
        # Original panel_statistics uses method key; add separate comparison
        # labels locally for budgets, leaving per-evaluation identity untouched.
        for category in cats:
            ka=(cohort,category,l,lb);kb=(cohort,category,r,rb)
            require(ka in evaluations and kb in evaluations, f'Incomplete fixed category contrast: {category} {l}/{r}')
            a,b=evaluations[ka],evaluations[kb]
            require(a[0]['membership_digest']==b[0]['membership_digest'],'Paired evaluation member-source mismatch')
            require({u['global_identity'] for u in a[2]}=={u['global_identity'] for u in b[2]},'Paired whole-unit mismatch')
            for tag,entry in [('LEFT',a),('RIGHT',b)]:
                for u in entry[2]:selected.append(dict(u,method=tag))
            summary_pairs.append((a[1],b[1]))
        logical=(cohort,tuple(cats),l,r,lb,rb)
        if logical not in estimates:
            groupname=requirement['category_or_macro']
            original=panel.contrasts(selected,{groupname:cats},[('LEFT','RIGHT')],cohort)
            require(len(original)==7,'Original paired endpoints incomplete')
            ap=next(x for x in original if x['endpoint']=='object_pixel_AP')
            row=dict(cohort_id=cohort,stratum=requirement.get('stratum','category'),
                category_or_macro=groupname,left_profile=l,right_profile=r,left_budget=lb,right_budget=rb,
                endpoint='unit_ap',difference_pp=100*ap['difference'],ci_low_pp=100*ap['pointwise95_lo'],
                ci_high_pp=100*ap['pointwise95_hi'],bootstrap_n=10000,bootstrap_seed=20261022,
                paired_units_n=ap['eligible_units'],categories_n=len(cats),
                inference_role=ap['inference'],status='SYNTHETIC_TEST_ONLY' if fixture else 'COMPLETED')
            contrasts.append(row)
            estimates[logical]=len(contrasts)-1
            for tag,index in [('LEFT',0),('RIGHT',1)]:
                vals=[s[index] for s in summary_pairs]
                mapping=dict(unit_ap='mean_object_pixel_AP',pixel_ap='pooled_pixel_AP',pixel_auroc='pooled_pixel_AUROC',
                             image_ap='object_AP',image_auroc='object_AUROC')
                macro=dict(cohort_id=cohort,stratum=groupname,profile_id=l if tag=='LEFT' else r,
                           budget_id=lb if tag=='LEFT' else rb,categories_n=len(cats),status=row['status'])
                for key,old in mapping.items():
                    numbers=[_number(x.get(old)) for x in vals]
                    macro[key]=float(np.mean(numbers)) if all(n is not None for n in numbers) else None
                macros.append(macro)
        requirements_out.append(dict(requirement,shared_estimate_index=estimates[logical]))
    def populate(stage):
        csv_new(stage/'PAIRED_CONTRASTS.csv',contrasts)
        # Equal fixed-category average, never a repooled all-category pixel AP.
        seen=set();unique_macros=[]
        for row in macros:
            key=(row['cohort_id'],row['stratum'],row['profile_id'],row['budget_id'])
            if key not in seen:seen.add(key);unique_macros.append(row)
        csv_new(stage/'MACRO.csv',unique_macros)
        write_new(stage/'REQUIREMENT_SHARING.json',requirements_out)
        complete=dict(status='SYNTHETIC_STATISTICS_COMPLETE' if fixture else 'COMPLETE_STATISTICS',
            seed=20261022,replications=10000,evaluator_sources=evaluator_sources,
            source_evaluations=[dict(path=str(Path(d).resolve()),sha256=sha(Path(d)/'COMPLETE.json')) for d in evaluation_dirs],
            outputs=[dict(file=p.name,sha256=sha(p)) for p in sorted(stage.iterdir())],
            estimates_n=len(contrasts),logical_requirements_n=len(requirements_out),
            inference='Whole-unit paired within fixed categories; no category resampling; pointwise conditional 95%')
        write_new(stage/'COMPLETE.json',complete)
        return complete
    return _atomic_directory(outdir,populate)


def run_synthetic_fixture(output_dir, evaluator_sources=None):
    """Explicit generated-data E2E fixture, never scientific result output."""
    sources=evaluator_sources or SOURCE_DEFAULTS
    target=Path(output_dir)
    require('fixture' in target.name.lower(),'Synthetic output directory name must contain fixture')
    require(not target.exists(),'Fixture output already exists')
    target.mkdir(parents=True)
    stats=_pinned_module(sources,'statistics_core')
    # 64 normal calibration, 2 query whole units with 2 real views each.
    def unit(identity,label):
        return dict(global_identity=identity,label_stratum=label,real_view_boxes=[[0,0,256,256],[256,0,512,256]],composite_wh=[512,256])
    group=dict(reference=[unit('fixture:fit:0','normal')],
        calibration=[unit(f'fixture:cal:{i}','normal') for i in range(64)],
        query=[unit('fixture:q:normal','normal'),unit('fixture:q:bad','anomaly')])
    gt=np.zeros((2,2,256,256),dtype=bool);gt[1,0,0,0]=True;gt[1,1,0,0]=True
    bad=np.array([False,True])
    evaluations=[]
    for method,score in [('M_W_U',.9),('I_W_U',.8)]:
        config=dict(config_id='fixture_'+method,category='candle',profile_id=method,budget_id='B',cohort_id='visa12',
                    state_sha256=hashlib.sha256(method.encode()).hexdigest(),membership_digest=membership_digest(group))
        pred=target/(method+'_predictions');pred.mkdir()
        records=[]
        for role in ('calibration','query'):
            for i,row in enumerate(group[role]):
                maps=np.zeros((2,256,256),dtype=np.float32)
                object_score=np.float64(.1)
                if role=='query' and i==1:
                    maps[0,0,0]=np.float32(score);maps[1,0,0]=np.float32(score)
                    object_score=np.float64(score)
                p=pred/f'{role}_{i:04d}.npz'
                np.savez(p,maps256=maps,object_score=object_score,global_identity=np.array(row['global_identity']))
                records.append(dict(role=role,unit_id=row['global_identity'],global_identity=row['global_identity'],
                                    file=p.name,sha256=sha(p),actual_views=2))
        seal=dict(config,status='COMPLETE_PREDICTIONS',source_kind='RAW_RECOMPUTED',
                  synthetic_fixture=True,records=records)
        write_new(pred/'COMPLETE.json',seal)
        checked,loaded=validate_prediction_complete(pred,group,config)
        summary,units=_compute_evaluation(group,config,checked,loaded,gt,bad,stats.binary_metrics)
        # Every output is labeled synthetic, including source_kind.
        summary['source_kind']='SYNTHETIC_FIXTURE_ONLY'
        for row in units:row['source_kind']='SYNTHETIC_FIXTURE_ONLY'
        ev=target/(method+'_evaluation');ev.mkdir()
        csv_new(ev/'SUMMARY.csv',[summary]);csv_new(ev/'PER_UNIT.csv',units)
        complete=dict(status='SYNTHETIC_EVALUATION_COMPLETE',config=config,membership_digest=config['membership_digest'],
            outputs=[dict(file=p.name,sha256=sha(p)) for p in sorted(ev.glob('*.csv'))])
        write_new(ev/'COMPLETE.json',complete);evaluations.append(ev)
    requirement=dict(cohort_id='visa12',stratum='category',category_or_macro='candle',left_profile='M_W_U',
                     right_profile='I_W_U',left_budget='B',right_budget='B',requirement_id='fixture_1')
    result=_bootstrap_impl(evaluations,[requirement],sources,target/'fixture_statistics',True)
    write_new(target/'FIXTURE_COMPLETE.json',dict(status='SYNTHETIC_E2E_COMPLETE_NOT_SCIENCE',
        original_GT_opened=False,GPU_used=False,pretrained_model_used=False,calibration_units=64,query_units=2,
        real_views_per_unit=2,bootstrap_n=10000,bootstrap_seed=20261022,statistics_complete=result))
    return read(target/'FIXTURE_COMPLETE.json')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest='command',required=True)
    check=subs.add_parser('validate-predictions',help='CPU-only integrity check; never reads GT')
    check.add_argument('--predictions',type=Path,required=True);check.add_argument('--group-json',type=Path,required=True)
    check.add_argument('--config-json',type=Path,required=True)
    evaluate=subs.add_parser('evaluate',help='Annotation evaluation requires a verified fixed-protocol record')
    for name in ['predictions','group-json','config-json','sources-json','gt-source-json','protocol-record','output']:
        evaluate.add_argument('--'+name,type=Path,required=True)
    bootstrap=subs.add_parser('bootstrap',help='Production fixed-category paired whole-unit statistics')
    bootstrap.add_argument('--inputs-json',type=Path,required=True);bootstrap.add_argument('--output',type=Path,required=True)
    fixture=subs.add_parser('fixture',help='Explicit synthetic E2E only; no model or original GT')
    fixture.add_argument('--output',type=Path,required=True);fixture.add_argument('--sources-json',type=Path)
    args=parser.parse_args(argv)
    if args.command=='validate-predictions':
        seal,loaded=validate_prediction_complete(args.predictions,read(args.group_json),read(args.config_json))
        result=dict(status='PREDICTION_BYTES_AND_MEMBERS_VERIFIED_NO_GT',units=len(loaded),config_id=seal['config_id'])
    elif args.command=='evaluate':
        result=evaluate_predictions(args.predictions,read(args.group_json),read(args.config_json),read(args.sources_json),
                                    read(args.gt_source_json),args.output,args.protocol_record)
    elif args.command=='bootstrap':
        spec=read(args.inputs_json)
        result=bootstrap_contrasts(spec['evaluation_dirs'],spec['requirements'],spec['evaluator_sources'],args.output,spec['protocol_record'])
    else:
        result=run_synthetic_fixture(args.output,read(args.sources_json) if args.sources_json else None)
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
