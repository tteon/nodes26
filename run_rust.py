#!/usr/bin/env python3
"""Run the unchanged research Rust reader against the synthetic fixture."""
import argparse,json,os,subprocess
from pathlib import Path
from demo import inputs,verify_samples
p=argparse.ArgumentParser();p.add_argument('--binary',type=Path,default=Path(__file__).resolve().parent/'rust/target/release/ogkf-driver-compare');p.add_argument('--mode',default='single',choices=['single','sequential','parallel']);p.add_argument('--concurrency',type=int,default=1,choices=[1,4]);p.add_argument('--count',type=int,default=4);a=p.parse_args()
i=inputs(a.mode,a.concurrency,a.count);env=dict(os.environ)
# The unchanged reader retains its historical variable names. Values stay private.
for old,new in [('DOZER_URI','NEO4J_URI'),('DOZER_USER','NEO4J_USERNAME'),('DOZER_PASSWORD','NEO4J_PASSWORD')]:
    if not env.get(new):raise SystemExit('Set '+new+' in the environment')
    env[old]=env[new]
r=subprocess.run([str(a.binary.resolve())],input=json.dumps(i),text=True,capture_output=True,env=env)
if r.returncode:raise SystemExit('Rust reader failed; credentials and process error text omitted')
print(json.dumps(verify_samples(json.loads(r.stdout)),ensure_ascii=False,indent=2))
