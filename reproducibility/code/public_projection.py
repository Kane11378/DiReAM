"""Validate scientific metadata, saved observations and their current byte pins.

Original state/config/evaluator hashes identify the original scientific run.
Current metadata digests verify the public projection; absent banks/maps are not
certified by metadata alone. No fitting, inference or timing is performed here.
"""
from pathlib import Path,PurePosixPath
import csv,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
P='protocol/evaluation'; D='results/evaluation'
def require(ok,message):
    if not ok:raise ValueError(message)
def read(path):return json.loads(Path(path).read_text('utf-8-sig'))
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def safe(root,name):
    root=Path(root).resolve();require(isinstance(name,str) and name and '\\' not in name and ':' not in name,'Unsafe science path')
    rel=PurePosixPath(name);require(not rel.is_absolute() and all(x not in ('','.','..') for x in name.split('/')),'Unsafe science path')
    p=root/name;require(p.resolve().is_relative_to(root) and p.exists() and not any(x.is_symlink() or (hasattr(x,'is_junction') and x.is_junction()) for x in [p,*p.parents] if x.is_relative_to(root)),'Missing/linked science path: '+name);return p
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def provenance(root):
    d=read(safe(root,P+'/METADATA_IDENTITIES.json'));require(d.get('schema')=='diream-public-metadata-identities-v1','Wrong metadata identity schema')
    records=d['files'];require(len(records)==len({r['path'] for r in records}),'Duplicate metadata identity')
    for r in records:require(sha(safe(root,r['path']))==r['current_sha256'],'Current metadata changed: '+r['path'])
    return {r['path']:r for r in records}
def protocol_pair(root,freeze_path,protocol_path,index=None):
    root=Path(root).resolve();fpath=Path(freeze_path).resolve();ppath=Path(protocol_path).resolve()
    require(fpath.is_relative_to(root) and ppath.is_relative_to(root),'Scientific protocol must belong to package')
    f=read(fpath);p=read(ppath);m=f.get('reader_metadata',{});q=p.get('reader_metadata',{})
    require(f.get('status')=='FROZEN' and q.get('scope')=='scientific_protocol_projection','Scientific freeze/projection missing')
    require(f['resolved_sha256']==f['protocol_sha256']==q['original_file_sha256']==m['original_protocol_sha256'],'Original scientific protocol identity differs')
    require(m['current_protocol_path']==ppath.relative_to(root).as_posix() and m['current_protocol_sha256']==sha(ppath),'Current protocol binding differs')
    require(p['statistics']['bootstrap_seed']==20261022 and p['statistics']['bootstrap_n']==10000,'Frozen statistics changed')
    return sha(ppath)
def verify(root=ROOT):
    root=Path(root).resolve();identity=provenance(root);pins=read(safe(root,P+'/SAVED_INPUTS.json'))['files']
    for name,h in pins.items():require(sha(safe(root,name))==h,'Saved scientific input changed: '+name)
    groups={}
    audit={r['category']:r for r in read(safe(root,P+'/parent/protocol/source_audit/MEMBERSHIP_AUDIT.json'))['groups']}
    for rel in ('protocol/category_panel.json','protocol/multiview_common.json'):
        for cat,g in read(safe(root,rel))['groups'].items():
            require(cat not in groups,'Duplicated category');groups[cat]=g;a=audit.get(cat);sets=[]
            for role in ('reference','calibration','query'):
                rr=g[role];ids=[r['global_identity'] for r in rr];require(len(ids)==len(set(ids)) and bool(ids),'Role member count/uniqueness differs')
                if a:require(len(ids)==a['counts'][role] and digest(ids)==a['ordered_identity_digests'][role],'Ordered role membership differs')
                sets.append(set(ids))
                require(all(len(r['real_view_boxes'])==(5 if r['source']=='MANTA' else 1) for r in rr),'Real-view geometry differs')
                for r in rr:require(len(r['source_payload_sha256'])==64 and r['rgb_member_bytes']>0,'RGB source pin missing')
            require(all(not sets[i]&sets[j] for i in range(3) for j in range(i+1,3)),'Cross-role leakage')
    require(len(groups)==14 and len(audit)==13 and set(audit)<=set(groups),'Fixed category panel differs')
    states=sorted(safe(root,P+'/state_metadata').glob('*/STATE.json'));require(len(states)==64,'State count changed');state_ids=set()
    for path in states:
        d=read(path);p=d.get('public_projection',{});state_ids.add(d['state_identity_sha256'])
        require(p.get('original_state_identity_sha256')==d['state_identity_sha256'],'Original state identity differs')
        require(p['projection_identity_sha256']==digest({k:v for k,v in d.items() if k!='public_projection'}),'Current state metadata digest differs')
        native=path.with_name('NATIVE_STATE.json')
        if native.exists():require(d['native_state']==read(native),'Native state inner/outer differs')
    evaluations=sorted(safe(root,D+'/e').iterdir());require(len(evaluations)==113,'Evaluation count changed');units_n=0
    configs={c['config_id']:c for c in read(safe(root,P+'/original/RUN_PLAN.json'))['configs']}
    for folder in evaluations:
        c=read(folder/'COMPLETE.json');q=c['config'];require(c['status']=='COMPLETE_EVALUATION' and q['config_id'] in configs,'Unknown/incomplete scientific evaluation')
        require(q['state_sha256'] in state_ids,'Evaluation/state association differs')
        for p in c['outputs']:require(sha(safe(folder,p['file']))==p['sha256'],'Evaluation observation changed')
        u=rows(folder/'PER_UNIT.csv');units_n+=len(u);require([x['global_identity'] for x in u]==[x['global_identity'] for x in groups[q['category']]['query']],'Evaluation ordered query membership differs')
        require(c.get('expected_tables_used_as_inputs') is False,'Evaluation used expected tables as numerical input')
        if c.get('source_kind')=='RAW_RECOMPUTED':require(c.get('GT_opened_only_after_complete_predictions') is True,'Prediction/GT chronology absent')
    require(units_n==19560 and len(rows(safe(root,D+'/RESULTS_CATEGORY.csv')))==113 and len(rows(safe(root,D+'/RESULTS_UNITS.csv')))==19560,'Scientific row count differs')
    b=read(safe(root,D+'/bootstrap/COMPLETE.json'));require(b['status']=='COMPLETE_STATISTICS' and b['seed']==20261022 and b['replications']==10000,'Bootstrap scientific identity changed')
    refs=b['source_evaluations'];require(len(refs)==len({x['path'] for x in refs})==113 and {x['path'] for x in refs}=={p.relative_to(root).as_posix() for p in evaluations},'Bootstrap evaluation membership differs')
    for p in refs:require(sha(safe(root,p['path'])/'COMPLETE.json')==p['sha256'],'Bootstrap current seal binding differs')
    for p in b['outputs']:require(sha(safe(root/D/'bootstrap',p['file']))==p['sha256'],'Bootstrap output changed')
    timing=sorted(safe(root,D+'/t').iterdir());require(len(timing)==25,'Timing count changed');requests=0
    for folder in timing:
        s=read(folder/'TIMING_SUMMARY.json');c=read(folder/'COMPLETE.json');r=read(folder/'REQUESTS.json');requests+=len(r['samples'])
        require(sha(folder/'ENVIRONMENT.json')==s['environment_sha256'] and sha(folder/'TIMING_SUMMARY.json')==c['summary_sha256'] and sha(folder/'REQUESTS.json')==c['requests_sha256'],'Timing byte chain differs')
        require(r['config_id'] in configs and r['state_sha256'] in state_ids,'Timing configuration/state differs')
    require(requests==750,'Request count changed')
    protocol_pair(root,root/P/'original/PROTOCOL_FREEZE.json',root/P/'original/PROTOCOL_RESOLVED.json')
    m=read(safe(root,P+'/METRIC_PATH_MAP.json'));require(len(m)==17 and len({k.replace('\\','/').casefold() for k in m})==17,'Metric logical source collision')
    budgets=read(safe(root,P+'/parent/protocol/source_audit/ACTUAL_BANK_BUDGETS.json'))['states'];require(len(budgets)==17,'Metric source count changed')
    import numpy as np
    for row in budgets:
        pin=m[row['model_path']];path=safe(root,pin['relative']);require(sha(path)==pin['sha256']==row['model_sha256'],'Frozen metric source bytes differ')
        with np.load(path,allow_pickle=False) as data:
            require(set(data.files)=={'center','matrix'},'Metric fields differ')
            for a in row['model_arrays']:require(list(data[a['key']].shape)==a['shape'] and str(data[a['key']].dtype)==a['dtype'] and bool(np.isfinite(data[a['key']]).all()),'Metric geometry/dtype/finite check failed')
    return dict(original_scientific_states=64,current_projection_files=len(identity),current_saved_inputs=len(pins),bootstrap_current_evaluation_pins=113,timing_chains=25,timing_requests=750,historical_protocol_pairs=1,normal_metric_bindings=17,ordered_category_groups=14,evaluation_category_audits=13,scope='Scientific metadata and supplied observation integrity; no fit, inference, bootstrap, new timing, or validation of absent large arrays')
