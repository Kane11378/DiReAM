"""Validate saved request observations and recompute reported summaries."""
from pathlib import Path
import hashlib, json, math

def require(ok,message):
    if not ok:raise ValueError(message)


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def number(value):
    if value in ('',None,'None'):return ''
    result=float(value);require(math.isfinite(result),'Nonfinite scalar in sealed output');return result


def read_timing(root,config,state_sha,group):
    root=Path(root)
    if not (root/'COMPLETE.json').exists():return None
    complete=read(root/'COMPLETE.json');require(complete.get('status')=='COMPLETE_TIMING','Timing is not sealed complete')
    require(sha(root/'REQUESTS.json')==complete['requests_sha256'] and sha(root/'TIMING_SUMMARY.json')==complete['summary_sha256'],'Timing raw/summary bytes changed')
    request_doc=read(root/'REQUESTS.json');summary=read(root/'TIMING_SUMMARY.json')
    require(summary.get('source_kind')=='RAW_RECOMPUTED' and summary.get('single_profile_process') is True,'Timing source/process identity differs')
    for key in ('category','profile_id','budget_id'):
        require(summary.get(key)==config[key],'Timing profile/category/budget mismatch')
    require(summary['state_sha256']==state_sha and request_doc['state_sha256']==state_sha and request_doc['config_id']==config['config_id'],'Timing state/config mismatch')
    require(sha(root/'ENVIRONMENT.json')==summary['environment_sha256'],'Timing environment changed')
    environment=read(root/'ENVIRONMENT.json')
    require(environment.get('platform')=='win32','This fixed Windows timing series cannot include another OS measurement')
    rr=request_doc['samples'];require(len(rr)==30,'Expected1first+2warmup+27resident requests')
    expected=[('first',0),('warmup',1),('warmup',2)]+[('resident',i%3) for i in range(27)]
    for i,(record,(phase,index)) in enumerate(zip(rr,expected)):
        require(record['request_index']==i and record['phase']==phase,'Timing phase/order mismatch')
        require(record['unit_id']==group['calibration'][index]['global_identity'],'Timing calibration selection changed')
        require(record['state_sha256']==state_sha and record['process_id']==summary['process_id'],'Request process/state mismatch')
        require(record['actual_views']==len(group['calibration'][index]['real_view_boxes']),'Request actual-view count mismatch')
        require(number(record['total_request_s'])!='' and float(record['total_request_s'])>=0,'Invalid raw request time')
        readings=4 if config['profile_id'].startswith(('I_W_','M_W_')) else 1
        searches=4 if config['profile_id'].startswith('I_W_') else 1
        pca=4 if config['profile_id'] in {'I_W_FC4','M_W_FC4'} else 1 if config['profile_id'] in {'N_W_F0','NativeNoRot_full','NativeNoRot_exact'} else 0
        require(record['encoder_readings_per_view']==readings and record['distance_queries_per_anchor']==searches and
                record['actual_encoder_calls']==readings*record['actual_views'] and record['actual_PCA_calls']==pca*record['actual_views'],
                'Actual encoder/PCA/search request schedule mismatch')
    resident=sorted(float(r['total_request_s']) for r in rr if r['phase']=='resident')
    import numpy as np
    require(summary['resident_n']==27 and math.isclose(summary['resident_median_s'],float(np.median(resident)),rel_tol=1e-12,abs_tol=1e-12),'Resident median differs from raw requests')
    require(math.isclose(summary['resident_p95_s'],float(np.quantile(resident,.95)),rel_tol=1e-12,abs_tol=1e-12),'Resident p95 differs from raw requests')
    require(summary.get('query_feature_cache') is False and summary.get('query_mask_cache') is False and
            summary.get('input_SHA_inside_timer') is False and summary.get('RGB_decode_inside_timer') is True and
            summary.get('output_validation_inside_timer') is False and summary.get('persistence_inside_timer') is False,
            'Timing boundary differs from fixed request protocol')
    return dict(summary=summary,requests=rr,complete_sha256=sha(root/'COMPLETE.json'))


