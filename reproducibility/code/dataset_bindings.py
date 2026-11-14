"""Bind packaged scientific code to explicit resources on the current machine.

Ordered memberships, normal metrics and reference budgets are checked before
use. Source snapshots and device-check evidence have separate identities.
"""
from pathlib import Path
import hashlib,io,json,sys,tarfile

ROOT=Path(__file__).resolve().parents[1]
PARENT=ROOT/'protocol/evaluation/parent'
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(ok,msg):
    if not ok:raise ValueError(msg)
def resolve_external(p,base):
    p=Path(p);return (p if p.is_absolute() else Path(base)/p).resolve()
def defaults(context=None):
    c=dict(context or {});c.setdefault('run_root',str(ROOT/'run_OUTPUT_REQUIRED'))
    c.update(reader_root=str(ROOT),base_reader=str(ROOT/'base'),parent_run_root=str(PARENT))
    c.setdefault('source_root',str(ROOT));c.setdefault('component_cache_mode','raw_rebuild')
    c.setdefault('native_candidate_mode','raw_rebuild');c.setdefault('state_reuse_mode','raw_rebuild')
    c['portable_binding']=True
    return c
def sources():
    from source_identity import verify_sources
    pins=verify_sources(ROOT)['resolved_pins']
    return {'statistics_core':dict(path=str(ROOT/'base/code/reader_statistics_core.py'),sha256=pins['base/code/reader_statistics_core.py']),
            'panel_statistics':dict(path=str(ROOT/'code/panel_statistics.py'),sha256=pins['code/panel_statistics.py'])}
def role_group(config,context=None):
    import matching_backend as b
    c=defaults(context);category=config['category'];b.profile_spec(config['profile_id'])
    require(category in b.VISA+['paddy'],'Category outside the fixed panel')
    name='multiview_common.json' if category=='paddy' else 'category_panel.json'
    path=ROOT/'protocol'/name;pin=__import__('source_identity').verify_sources(ROOT)['resolved_pins'][f'protocol/{name}']
    require(sha(path)==pin,'Ordered-member manifest changed')
    group=read(path)['groups'][category]
    audit=next(r for r in read(PARENT/'protocol/source_audit/MEMBERSHIP_AUDIT.json')['groups'] if r['category']==category)
    out={};sets=[]
    for role in ['reference','calibration','query']:
        rows=group[role];ids=[r['global_identity'] for r in rows]
        require(len(ids)==audit['counts'][role] and b.digest(ids)==audit['ordered_identity_digests'][role],'Role membership/order differs')
        require(len(ids)==len(set(ids)),'Duplicate whole unit');out[role]=rows;sets.append(set(ids))
    require(not any(sets[i]&sets[j] for i in range(3) for j in range(i+1,3)),'Cross-role leakage')
    require(all(len(r['real_view_boxes'])==(5 if category=='paddy' else 1) for role in out.values() for r in role),'Actual-view topology changed')
    out.update(membership_digest=b.digest({r:[x['global_identity'] for x in out[r]] for r in ['reference','calibration','query']}),source_path=str(path),source_sha256=sha(path))
    require(config.get('member_source_sha256') in (None,sha(path)),'Role source SHA mismatch')
    require(config.get('roles_digest') in (None,out['membership_digest'],audit['ordered_identity_digests']),'Role digest mismatch')
    return out
def budget(config,context):
    import matching_backend as b
    rows=sorted([dict(r) for r in read(PARENT/'protocol/source_audit/ACTUAL_BANK_BUDGETS.json')['states'] if r['category']==config['category']],key=lambda r:r['slot_id'])
    mapping=read(ROOT/'protocol/evaluation/METRIC_PATH_MAP.json')
    for row in rows:
        if context.get('state_reuse_mode')=='historical_verified':
            root=context.get('historical_state_roots',{}).get(f"{config['category']}:{row['slot_id']}")
            require(root is not None,'Historical state reuse requires an explicit historical_state_roots category:slot binding; original workstation paths are not executable defaults')
            for key in ['bank_path','indices_path']:
                row[key]=str(Path(root)/row[key].replace('\\','/').rsplit('/',1)[-1])
        original=row['model_path'].replace('\\','/').casefold()
        pin=next(v for k,v in mapping.items() if k.replace('\\','/').casefold()==original);p=ROOT/pin['relative']
        require(sha(p)==pin['sha256']==row['model_sha256'],'Frozen normal metric changed');row['model_path']=str(p)
    values=[int(r['retained_rows']) for r in rows];bid=config.get('budget_id','B');spec=b.profile_spec(config['profile_id'])
    if spec['full']:require(bid=='full','Full-memory profile requires full');target=None
    elif bid=='B':target=values
    elif bid in ['half_B','double_B'] and config['category'] in ['chewinggum','paddy']:target=[v//2 if bid=='half_B' else 2*v for v in values]
    else:raise ValueError('Budget outside the fixed protocol')
    require(config.get('k_per_slot') in (None,target),'Explicit K differs from fixed budget')
    return target,rows
def source_bytes(row,context):
    c=defaults(context);source=c.get('image_sources',{}).get(row['source'])
    require(source is not None,'Bind image_sources.VisA and image_sources.MANTA explicitly; see RESOURCE_BINDINGS.example.json')
    p=Path(source)
    if p.is_dir():
        q=(p/row['rgb_member']).resolve();require(q.is_relative_to(p.resolve()),'RGB member escapes explicit root');data=q.read_bytes()
    elif p.suffix=='.tar':
        with p.open('rb') as f:f.seek(row['rgb_offset_data_uncompressed']);data=f.read(row['rgb_member_bytes'])
    else:
        with tarfile.open(p,'r:*') as t:
            f=t.extractfile(row['rgb_member']);require(f is not None,'RGB member missing');data=f.read()
    require(len(data)==row['rgb_member_bytes'] and hashlib.sha256(data).hexdigest()==row['source_payload_sha256'],'RGB member bytes/SHA differ')
    return data
def verify_source(row,context=None):
    data=source_bytes(row,context);return dict(member=row['rgb_member'],bytes=len(data),sha256=row['source_payload_sha256'])
def decode(row,context=None,verify=True):
    import numpy as np
    from PIL import Image
    # Prevalidation remains outside the timed path; the read/decode remains inside.
    c=defaults(context);source=c.get('image_sources',{}).get(row['source'])
    require(source is not None,'Explicit image source required')
    if verify:data=source_bytes(row,c)
    else:
        p=Path(source)
        if p.is_dir():data=(p/row['rgb_member']).read_bytes()
        elif p.suffix=='.tar':
            with p.open('rb') as f:f.seek(row['rgb_offset_data_uncompressed']);data=f.read(row['rgb_member_bytes'])
        else:
            with tarfile.open(p,'r:*') as t:data=t.extractfile(row['rgb_member']).read()
    with Image.open(io.BytesIO(data)) as im:rgb=np.asarray(im.convert('RGB'))
    views=[rgb[y0:y1,x0:x1].copy() for x0,y0,x1,y1 in row['real_view_boxes']]
    require(all(v.size and v.shape[-1]==3 for v in views),'Invalid actual-view crop');return views
def source_pins(context):
    from source_identity import verify_sources
    pins=verify_sources(ROOT)['resolved_pins']
    result={str((ROOT/path).resolve()):value for path,value in pins.items()}
    if context.get('resources_json'):
        p=Path(context['resources_json']).resolve();result[str(p)]=sha(p)
    return result
def validate_runtime(context,formal=True,timing=False):
    """Validate current raw-rebuild resources and bind the scientific protocol."""
    require(context.get('rebuild_evidence'), 'Supply the current raw-rebuild bundle')
    from public_rebuild_evidence import validate_runtime as validate_rebuild
    return validate_rebuild(context,formal=formal,timing=timing)
def patch_backend(module):
    module.context_defaults=defaults;module.roles=role_group;module._budget=budget
    module.verify_source=verify_source;module.decode_unit=decode;module._parent_pins=source_pins
    # Historical regression is optional and explicit; raw prediction never
    # discovers a workstation prediction cache by guessing its directory.
    def baseline(config,context):
        pin=context.get('regression_baselines',{}).get(config['config_id'])
        if pin is None:return None
        require(sha(pin['path'])==pin['sha256'],'Explicit regression baseline changed')
        return Path(pin['path'])
    module._baseline=baseline
    module._native_baseline=lambda config,context,members:None
