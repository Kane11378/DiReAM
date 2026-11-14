"""Atomic storage and ordered identity checks for predictions and states."""
from __future__ import annotations
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid


def now():
    return datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def file_sha(path):
    with Path(path).open('rb') as src:
        return hashlib.file_digest(src, 'sha256').hexdigest()


def within(root, path):
    root, path = Path(root).resolve(), Path(path).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f'Output must be a descendant of the isolated run: {path}')
    return path


def atomic_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise FileExistsError(path)
    temp = path.with_name('.t_'+uuid.uuid4().hex[:8])
    try:
        with temp.open('x', encoding='utf-8', newline='\n') as out:
            json.dump(value, out, ensure_ascii=False, indent=2)
            out.write('\n')
            out.flush()
            os.fsync(out.fileno())
        if replace:
            os.replace(temp, path)
        else:
            # Caller holds a directory job lock; Windows rename refuses an
            # existing destination. No completed artifact is silently replaced.
            if path.exists():
                raise FileExistsError(path)
            os.rename(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_npz(path, arrays):
    import numpy as np
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_name('.t_'+uuid.uuid4().hex[:8])
    try:
        with temp.open('xb') as out:
            np.savez(out, **arrays)
            out.flush()
            os.fsync(out.fileno())
        if path.exists():
            raise FileExistsError(path)
        os.rename(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


@contextmanager
def job_lock(directory):
    """OS lock is released on process death; stale filenames are harmless."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory/'.job.lock'
    stream = lock.open('a+b')
    stream.seek(0)
    if lock.stat().st_size == 0:
        stream.write(b'0')
        stream.flush()
    stream.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        stream.close()
        raise RuntimeError(f'Another process holds this isolated job: {directory}') from exc
    try:
        yield
    finally:
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def member_digest(group, roles=('calibration', 'query')):
    return digest([dict(role=role, global_identity=row['global_identity'],
                        source_payload_sha256=row['source_payload_sha256'],
                        real_view_boxes=row['real_view_boxes'])
                   for role in roles for row in group[role]])


def validate_records(directory, seal, group, config):
    """Validate complete cal/query identity before evaluation can access GT."""
    directory = Path(directory).resolve()
    if seal.get('status') != 'COMPLETE_PREDICTIONS':
        raise ValueError('Predictions have no completed seal')
    for key in ('config_id', 'category', 'cohort_id', 'profile_id', 'budget_id', 'state_sha256'):
        if seal.get(key) != config.get(key):
            raise ValueError(f'Prediction seal identity mismatch: {key}')
    expected = [(role, row) for role in ('calibration', 'query') for row in group[role]]
    records = seal.get('records', [])
    if seal.get('membership_digest') != member_digest(group) or len(records) != len(expected):
        raise ValueError('Prediction member count/digest mismatch')
    for record, (role, row) in zip(records, expected):
        if record['role'] != role or record['global_identity'] != row['global_identity']:
            raise ValueError('Prediction order or member differs')
        path = within(directory, directory/record['file'])
        if record.get('actual_views') != len(row['real_view_boxes']) or file_sha(path) != record['sha256']:
            raise ValueError('Prediction payload/views differ')
    return seal
