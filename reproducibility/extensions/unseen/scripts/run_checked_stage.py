"""Resume the fixed raw-image pipeline while verifying existing artifacts.

Preserves incomplete encoded roles, fit slots and prediction categories before
restarting them. Complete artifacts require hash verification. An annotation
barrier is reusable only when both category prediction seals are unchanged.
"""
from __future__ import annotations
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,hashlib,importlib.util,json,re,sys,shutil,uuid
from pathlib import Path
sys.dont_write_bytecode=True

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def read(path):return json.loads(Path(path).read_text('utf-8-sig'))
def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if read(path)!=value:raise ValueError('Recovery receipt differs: '+str(path))
        return
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')

def archive_partial(module,paths,reason):
    """Preserve partial bytes inside the explicitly allowed evidence root."""
    root=module.OUT.resolve();paths=[Path(path) for path in paths if Path(path).exists()]
    if not paths:return
    destination=root/'interrupted_attempts'/uuid.uuid4().hex
    if not destination.resolve().is_relative_to(root):raise ValueError('Archive target escapes evidence')
    for source in paths:
        resolved=source.resolve()
        if resolved==root or not resolved.is_relative_to(root) or source.is_symlink():raise ValueError('Unsafe partial source')
        for ancestor in source.parents:
            if ancestor==root:break
            if ancestor.exists() and ancestor.is_symlink():raise ValueError('Symlink ancestor in partial source')
    destination.mkdir(parents=True,exist_ok=False);records=[]
    for source in paths:
        files=[source] if source.is_file() else sorted(path for path in source.rglob('*') if path.is_file())
        if any(path.is_symlink() or not path.resolve().is_relative_to(root) for path in files):raise ValueError('Unsafe nested partial file')
        pins=[{'path':path.relative_to(root).as_posix(),'bytes':path.stat().st_size,'sha256':sha(path)} for path in files]
        target=destination/source.name
        if not target.resolve().is_relative_to(root) or target.exists():raise ValueError('Unsafe archive target')
        shutil.move(str(source),str(target))
        for pin,path in zip(pins,files,strict=True):
            moved=target if len(files)==1 and source.suffix and target.is_file() else target/path.relative_to(source)
            if sha(moved)!=pin['sha256']:raise ValueError('Moved partial differs; stop')
        records.append({'source':source.relative_to(root).as_posix(),'preserved_at':target.relative_to(root).as_posix(),'files':pins})
    dump(destination/'ARCHIVE_RECORD.json',{'reason':reason,'records':records,'scientific_data_replaced':False,'restart_from_zero':True})

def validate_encode_resume(module,category):
    import numpy as np
    directory=module.OUT/'encoded'/category
    if not directory.exists():return
    members=read(module.OUT/'MEMBERSHIP.json')['rows']
    for role in ('reference','calibration','query'):
        path=directory/(role+'.npy');seal=directory/(role+'.json');progress=directory/(role+'_PROGRESS.json')
        if not seal.exists():
            archive_partial(module,[path,progress],'Unsealed encoded role is preserved and restarted from row zero; progress alone is never trusted.')
            continue
        record=read(seal);ids=[row['global_identity'] for row in members if row['category']==category and row['role']==role]
        if not path.is_file() or sha(path)!=record['sha256'] or record['freeze_sha256']!=sha(module.OUT/'FROZEN.json') or record['global_identities']!=ids:
            raise ValueError('Completed encoded role hash/freeze/identities changed')
        array=np.load(path,mmap_mode='r');expected=(len(ids),5,32,32,4,384)
        if array.shape!=expected or array.dtype!=np.float32 or record['shape']!=list(expected):raise ValueError('Completed encoded cache shape/dtype changed')

def validate_fit_resume(module,category):
    folder=module.OUT/'fit'/category
    if not folder.exists():return
    for slot in sorted(folder.glob('slot_*')):
        if slot.is_dir() and not (slot/'COMPLETE.json').exists():
            archive_partial(module,[slot],'Incomplete normal-fit slot is preserved and recomputed from zero.')
    ref=module.OUT/'encoded'/category/'reference.npy';reference_hash=None
    for marker in sorted(folder.rglob('COMPLETE.json')):
        r=read(marker)
        if 'bank_sha256' in r:
            if reference_hash is None:reference_hash=sha(ref)
            if r.get('reference_cache_sha256')!=reference_hash:raise ValueError('Fit resume reference changed')
            for file,key in (('bank.npy','bank_sha256'),('model.npz','model_sha256'),('SELECTION.json','selection_sha256')):
                if sha(marker.parent/file)!=r[key]:raise ValueError('Changed completed fitted state: '+str(marker.parent/file))
            selection=read(marker.parent/'SELECTION.json')
            if sha(marker.parent/'indices.npy')!=selection['indices_sha256'] or selection['retained_rows']!=module.BUDGET:
                raise ValueError('Fit resume selected indices/count changed')
        else:
            side=marker.parent/'SLOT_ARTIFACT_HASHES.json'
            if not side.exists():raise ValueError('Cannot trust existing fit slot marker without its artifact hashes')
            rr=read(side)
            for record in rr['files']:
                if sha(marker.parent/record['path'])!=record['sha256']:raise ValueError('Fit slot artifact changed')

def validate_prediction_resume(module,category,*,require_complete=False):
    folder=module.OUT/'predictions'/category
    if not folder.exists():
        if require_complete:raise ValueError('Required completed prediction category is missing')
        return
    complete=folder/'COMPLETE.json'
    if not complete.exists():
        if require_complete:raise ValueError('Required final prediction seal is missing')
        archive_partial(module,[folder],'Prediction category lacks its final five-profile seal; preserve the entire partial category and recompute all slots from zero.')
        return
    rec=read(complete)
    if rec['freeze_sha256']!=sha(module.OUT/'FROZEN.json'):raise ValueError('Prediction completion freeze changed')
    if {row['profile'] for row in rec['files']}!=set(module.PROFILES):raise ValueError('Completed profile set differs')
    for file in rec['files']:
        if sha(folder/file['path'])!=file['sha256']:raise ValueError('Completed prediction file changed')

def install(module):
    original=module.write
    def checked_write(path,value,*,replace=False):
        path=Path(path)
        if path==module.OUT/'GT_ACCESS_BARRIER.json' and path.exists():
            old=read(path)
            if {k:v for k,v in old.items() if k!='utc'}!={k:v for k,v in value.items() if k!='utc'}:
                raise ValueError('GT barrier differs; cannot reuse a previous UTC')
            # Require exact two-category seals, verified again against disk.
            expected=[{'category':cat,'sha256':sha(module.OUT/'predictions'/cat/'COMPLETE.json')} for cat in module.CATEGORIES]
            if old.get('prediction_seals')!=expected or old.get('both_categories_all_profiles_sealed') is not True:
                raise ValueError('Existing GT barrier does not pin both current category prediction seals')
            return
        original(path,value,replace=replace)
        if path.name=='COMPLETE.json' and path.parent.parent.parent==module.OUT/'fit':
            # This is a completed fit slot, not a bank's nested marker.
            files=[]
            for file in sorted(path.parent.rglob('*')):
                if file.is_file() and file.name!='SLOT_ARTIFACT_HASHES.json':
                    files.append({'path':file.relative_to(path.parent).as_posix(),'sha256':sha(file)})
            dump(path.parent/'SLOT_ARTIFACT_HASHES.json',{'files':files})
    module.write=checked_write

