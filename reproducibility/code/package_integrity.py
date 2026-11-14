"""Verify the complete public package and its current scientific source pins."""
from pathlib import Path, PurePosixPath
import hashlib,json
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def verify(root):
    root=Path(root).resolve();d=json.loads((root/'MANIFEST.json').read_text('utf-8'));seen=set()
    for row in d['files']:
        name=row['path'];rel=PurePosixPath(name)
        if rel.is_absolute() or any(x in ('','.','..') for x in name.split('/')) or '\\' in name or ':' in name:raise ValueError('Unsafe package member: '+name)
        if name.casefold() in seen:raise ValueError('Duplicate package member: '+name)
        seen.add(name.casefold());p=root/name
        if not p.resolve().is_relative_to(root) or not p.is_file() or any(x.is_symlink() or (hasattr(x,'is_junction') and x.is_junction()) for x in [p,*p.parents] if x.is_relative_to(root)):raise ValueError('Missing or linked package member: '+name)
        if p.stat().st_size!=row['bytes'] or sha(p)!=row['sha256']:raise ValueError('Changed package member: '+name)
    actual={p.relative_to(root).as_posix().casefold() for p in root.rglob('*') if p.is_file() and p.relative_to(root).as_posix()!='MANIFEST.json'}
    if actual!=seen:raise ValueError('Package inventory differs; extract into an empty directory')
    from source_identity import verify_sources
    s=verify_sources(root)
    return dict(package_files=len(seen),scientific_source_pins=s['scientific_source_pins'],reviewed_current_derivations=s['reviewed_current_derivations'],current_source_projection_sha256=s['current_source_projection_sha256'])
