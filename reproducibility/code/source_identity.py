"""Verify current public science sources independently of original run identities."""
from pathlib import Path, PurePosixPath
import hashlib, json

def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def require(ok,message):
    if not ok: raise ValueError(message)
def member(root,name):
    root=Path(root).resolve(); require(isinstance(name,str) and name and '\\' not in name and ':' not in name,'Unsafe science member')
    p=PurePosixPath(name); require(not p.is_absolute() and all(x not in ('','.','..') for x in name.split('/')),'Unsafe science member')
    q=root/name; require(q.resolve().is_relative_to(root) and q.is_file(),'Missing science member: '+name)
    require(not any(x.is_symlink() or (hasattr(x,'is_junction') and x.is_junction()) for x in [q,*q.parents] if x.is_relative_to(root)),'Linked science member')
    return q
def projection_sha256(root): return sha(member(root,'protocol/evaluation/SOURCE_PINS.json'))
def verify_sources(root):
    root=Path(root).resolve(); p=member(root,'protocol/evaluation/SOURCE_PINS.json'); d=json.loads(p.read_text('utf-8'))
    require(d.get('schema')=='diream-current-scientific-source-pins-v1','Wrong scientific source inventory')
    pins=d['files'];require(isinstance(pins,dict) and bool(pins),'Empty scientific source inventory')
    require(len({x.casefold() for x in pins})==len(pins),'Duplicate source path')
    for name,h in pins.items():require(sha(member(root,name))==h,'Current scientific source/input changed: '+name)
    h=sha(p)
    return dict(scientific_source_pins=len(pins),reviewed_current_derivations=len(d.get('derivations',[])),portable_source_pins_sha256=h,current_source_projection_sha256=h,resolved_pins=pins,original_source_pins_sha256=d['original_source_pins_sha256'])
