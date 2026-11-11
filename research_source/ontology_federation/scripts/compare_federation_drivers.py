#!/usr/bin/env python3
"""Bounded DB-only matched Python/neo4rs experiment on copied frozen assertions."""
import collections, concurrent.futures as cf, hashlib, itertools, json, os, platform, random, subprocess, sys, time
from pathlib import Path
import numpy as np
from neo4j import GraphDatabase
import neo4j
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ogkf.environment import load_project_environment
from ogkf.local_runtime import local_dozer
OUT=ROOT/'reports/research/context_overlap_v1/driver_comparison'
QUERY='''MATCH (i:BenchIssuer {cik:$cik})-[:CONTAINS]->(e:BenchEntity)
 -[:HAS_ASSERTION]->(a:BenchAssertion)-[:SUPPORTED_BY]->(s:BenchSource)
 WHERE a.context IN $contexts RETURN a.payload AS payload ORDER BY a.id LIMIT 65'''
LOAD='''UNWIND $rows AS row
MERGE (i:BenchIssuer {cik:row.cik}) MERGE (e:BenchEntity {key:row.key})
MERGE (a:BenchAssertion {id:row.id}) SET a.context=row.context,a.payload=row.payload
MERGE (s:BenchSource {id:row.source_id}) MERGE (i)-[:CONTAINS]->(e)
MERGE (e)-[:HAS_ASSERTION]->(a) MERGE (a)-[:SUPPORTED_BY]->(s)'''
def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def hashrows(rows):return hashlib.sha256(json.dumps(sorted(rows),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def run_python(driver,inputs):
    epoch=time.perf_counter_ns()
    with cf.ThreadPoolExecutor(max_workers=8) as inner:
        def branch(db,cats):
            start=time.perf_counter_ns()-epoch
            with driver.session(database=db,fetch_size=128) as s:
                with s.begin_transaction() as tx:
                    res=tx.run(inputs['query'],cik=inputs['cik'],contexts=cats)
                    rows=[r['payload'] for r in res];res.consume();tx.commit()
            return dict(database=db,start_ns=start,end_ns=time.perf_counter_ns()-epoch,rows=rows)
        def request(n):
            start=time.perf_counter_ns()-epoch
            targets=[('neo4j',inputs['contexts'])] if inputs['mode']=='single' else [('ctxa',[inputs['contexts'][0]]),('ctxb',[inputs['contexts'][1]])]
            if inputs['mode']=='parallel':
                jobs=[inner.submit(branch,*t) for t in targets];branches=[j.result() for j in jobs]
            else:branches=[branch(*t) for t in targets]
            rows=sorted(r for b in branches for r in b['rows'])
            return dict(id=n,start_ns=start,end_ns=time.perf_counter_ns()-epoch,branches=branches,rows=rows)
        for n in range(8):request(n)
        started=time.perf_counter_ns()-epoch
        with cf.ThreadPoolExecutor(max_workers=inputs['concurrency']) as outer:
            samples=list(outer.map(request,range(inputs['count'])))
        return dict(samples=samples,batch_start_ns=started,batch_end_ns=time.perf_counter_ns()-epoch)
def flatten_plan(plan):
    return [{'operator':plan['operatorType'],'db_hits':plan.get('dbHits'),'rows':plan.get('rows'),'args':plan.get('args')}]+[p for c in plan.get('children',[]) for p in flatten_plan(c)]
def main():
    if (OUT/'summary.json').exists():raise RuntimeError('immutable completed comparison; choose a new version to rerun')
    OUT.mkdir(parents=True,exist_ok=True);load_project_environment()
    candidates=json.loads((OUT.parent/'cross_category_assertion_candidates.json').read_text())
    witness=next(x for x in candidates if x['profile']=='GX' and x['issuer_subject_surface_key']=='0000037785|fmc')
    cats=sorted({r['category'] for r in witness['assertions']});assert cats==['Accounting','Legal']
    # Actual model proposals, not semantic gold. QA category assigned only for this execution fixture.
    rows=[]
    for a in witness['assertions']:
        payload=json.dumps(a,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        rows.append({'cik':'0000037785','key':a['entity_key'],'id':a['id'],'source_id':a['source_id'],'context':a['category'],'payload':payload})
    assert len(rows)<=64 and len({r['id'] for r in rows})==len(rows)
    save(OUT/'fixture.json',{'source':str(OUT.parent/'cross_category_assertion_candidates.json'),'rows':rows,'contexts':cats,'selection':'first frozen GX FMC exact subject candidate; two native QA categories, no organizational ownership claim'})
    expected=hashrows([r['payload'] for r in rows]);cells=[];profiles=[];manifest={}
    binary=Path('/tmp/ogkf-driver-target/release/ogkf-driver-compare');assert binary.exists()
    save(OUT/'protocol.json',{'query':QUERY,'cells':'2 drivers x 3 modes x concurrency1/4 x 3 shuffled blocks x16 requests','logical_requests':576,'warmups_per_cell':8,'seed':20261110,'pool_max':8,'fetch_size':128,'rust_threads':4,'transactions':'explicit BEGIN/RUN/full consume/COMMIT, no managed retries','profiling':'Python PROFILE gates outside timed cells; neo4rs0.8 discards summary metadata','model_calls':0,'claim':'local warm small-result client execution comparison, not saturation or GIL attribution','queue':'sample starts on admitted worker/task; outer queue absent from request duration but present in batch throughput','cold_connections':'8 sequential warmups do not preopen8 pool connections; burst connection expansion included','expected_payload_sha256':expected})
    try:
      with local_dozer() as (env,manifest):
        driver=GraphDatabase.driver(env['DOZER_URI'],auth=(env['DOZER_USER'],env['DOZER_PASSWORD']),max_connection_pool_size=8,max_transaction_retry_time=0)
        try:
          with driver.session(database='system') as s:
            for db in ['ctxa','ctxb']:s.run('CREATE DATABASE '+db+' IF NOT EXISTS').consume()
          deadline=time.monotonic()+60
          while True:
            with driver.session(database='system') as s:states={r['name']:r['currentStatus'] for r in s.run('SHOW DATABASES')}
            if all(states.get(db)=='online' for db in ['neo4j','ctxa','ctxb']):break
            if time.monotonic()>deadline:raise TimeoutError('database readiness')
            time.sleep(.5)
          for db,subset in [('neo4j',rows),('ctxa',[r for r in rows if r['context']==cats[0]]),('ctxb',[r for r in rows if r['context']==cats[1]])]:
            with driver.session(database=db) as s:
              for label,key in [('BenchIssuer','cik'),('BenchEntity','key'),('BenchAssertion','id'),('BenchSource','id')]:s.run(f'CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE').consume()
              s.run('CALL db.awaitIndexes(30)').consume();s.run(LOAD,rows=subset).consume()
              result=s.run('PROFILE '+QUERY,cik='0000037785',contexts=cats if db=='neo4j' else [cats[0] if db=='ctxa' else cats[1]])
              got=[r['payload'] for r in result];summary=result.consume()
              assert hashrows(got)==hashrows([r['payload'] for r in subset])
              profiles.append({'database':db,'query':QUERY,'rows':len(got),'operators':flatten_plan(summary.profile),'result_available_after_ms':summary.result_available_after,'result_consumed_after_ms':summary.result_consumed_after})
          save(OUT/'profiles.json',profiles)
          matrix=list(itertools.product(['python','neo4rs'],['single','sequential','parallel'],[1,4]));rng=random.Random(20261110)
          for block in range(3):
            rng.shuffle(matrix)
            for language,mode,concurrency in matrix:
              inputs={'query':QUERY,'cik':'0000037785','contexts':cats,'mode':mode,'concurrency':concurrency,'count':16}
              started=time.time_ns();cpu=time.process_time()
              if language=='python':
                # New driver per cell matches Rust process/pool lifetime; initialization outside timing.
                cell_driver=GraphDatabase.driver(env['DOZER_URI'],auth=(env['DOZER_USER'],env['DOZER_PASSWORD']),max_connection_pool_size=8,max_transaction_retry_time=0)
                try:result=run_python(cell_driver,inputs)
                finally:cell_driver.close()
              else:
                p=subprocess.run([str(binary)],input=json.dumps(inputs),text=True,capture_output=True,timeout=60,env={**os.environ,**env})
                if p.returncode:raise RuntimeError('neo4rs process failed; credentials omitted from exception')
                result=json.loads(p.stdout)
              for sample in result['samples']:
                assert len(sample['rows'])==len(rows) and hashrows(sample['rows'])==expected
                sample['rows_sha256']=hashrows(sample.pop('rows'))
                for b in sample['branches']:b['rows_sha256']=hashrows(b.pop('rows'))
              cell={'driver':language,'mode':mode,'concurrency':concurrency,'block':block,'unix_start_ns':started,**result}
              cells.append(cell);save(OUT/'samples.json',cells)
              print(json.dumps({'cell':len(cells),'driver':language,'mode':mode,'concurrency':concurrency}),flush=True)
        finally:driver.close()
    finally:save(OUT/'runtime.json',manifest)
    aggregates=[]
    for language,mode,c in itertools.product(['python','neo4rs'],['single','sequential','parallel'],[1,4]):
        group=[x for x in cells if (x['driver'],x['mode'],x['concurrency'])==(language,mode,c)]
        ms=[(s['end_ns']-s['start_ns'])/1e6 for x in group for s in x['samples']]
        aggregates.append({'driver':language,'mode':mode,'concurrency':c,'requests':len(ms),'p50_ms':float(np.percentile(ms,50)),'p95_ms':float(np.percentile(ms,95)),'throughput_requests_s':sum(len(x['samples']) for x in group)/sum((x['batch_end_ns']-x['batch_start_ns'])/1e9 for x in group),'per_block_p50_ms':[float(np.median([(s['end_ns']-s['start_ns'])/1e6 for s in x['samples']])) for x in group]})
    save(OUT/'summary.json',{'logical_requests':sum(len(x['samples']) for x in cells),'payload_parity_all':True,'rows_per_request':len(rows),'python':platform.python_version(),'python_driver':neo4j.__version__,'rust_driver':'neo4rs0.8.0','cells':aggregates,'runtime_teardown':manifest.get('teardown_success'),'errors':0,'profile_metadata_rust':None,'gil_overhead':None})
if __name__=='__main__':main()
