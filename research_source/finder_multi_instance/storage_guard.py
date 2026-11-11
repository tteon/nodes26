#!/usr/bin/env python3
"""Fail closed before a new run writes to an undersized filesystem. No cleanup."""
import argparse
import json
from pathlib import Path
import shutil

GIB=1024**3


def check_space(storage,root_min_gib=10,stage_min_gib=20):
    storage=Path(storage).resolve(strict=True)
    if not storage.is_dir():raise ValueError('staging path must be an existing directory')
    checks=[]
    for role,path,minimum in [('system',Path('/'),root_min_gib),('stage',storage,stage_min_gib)]:
        if minimum<=0:raise ValueError('free-space minima must be positive')
        usage=shutil.disk_usage(path)
        checks.append({'role':role,'path':str(path),'filesystem_device':path.stat().st_dev,
                       'total_bytes':usage.total,'free_bytes':usage.free,
                       'minimum_free_bytes':int(minimum*GIB),'passed':usage.free>=minimum*GIB})
    return {'passed':all(c['passed'] for c in checks),'checks':checks,
            'meaning':'capacity gate only; disk medium and workload isolation require separate provenance'}


def require_space(storage,root_min_gib=10,stage_min_gib=20):
    result=check_space(storage,root_min_gib,stage_min_gib)
    if not result['passed']:raise RuntimeError('insufficient free storage; no new run admitted: '+json.dumps(result))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--storage',type=Path,required=True)
    p.add_argument('--root-min-gib',type=float,default=10);p.add_argument('--stage-min-gib',type=float,default=20)
    a=p.parse_args();result=check_space(a.storage,a.root_min_gib,a.stage_min_gib)
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['passed'] else 2)
