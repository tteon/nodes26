#!/usr/bin/env python3
"""Synthetic context-preserving read demo. Default action is offline."""
import argparse,hashlib,json,os,time
from pathlib import Path
from python_reader import QUERY,run_python
ROOT=Path(__file__).resolve().parent
CONTEXTS=['Accounting','Legal']
LOAD='''UNWIND $rows AS row
MERGE (i:BenchIssuer {cik:row.cik}) MERGE (e:BenchEntity {key:row.key})
MERGE (a:BenchAssertion {id:row.id}) SET a.context=row.context,a.payload=row.payload
MERGE (s:BenchSource {id:row.source_id}) MERGE (i)-[:CONTAINS]->(e)
MERGE (e)-[:HAS_ASSERTION]->(a) MERGE (a)-[:SUPPORTED_BY]->(s)'''
def fixture():return json.loads((ROOT/'fixtures/synthetic.json').read_text())
def fingerprint(payloads):return hashlib.sha256(json.dumps(sorted(payloads),ensure_ascii=False,separators=(',',':')).encode()).hexdigest()
def merge_payloads(payloads):
    kept={}
    for payload in payloads:
        row=json.loads(payload);key=(row['build_id'],row['id'])
        if key in kept and kept[key]!=payload:raise ValueError('Conflicting assertion in the same build; no automatic survivor')
        kept[key]=payload
    return sorted(kept.values())
def offline():
    rows=fixture();single=merge_payloads([r['payload'] for r in rows]);branches=[[r['payload'] for r in rows if r['context']==c] for c in CONTEXTS];federated=merge_payloads([p for b in branches for p in b]);assert single==federated
    records=[json.loads(p) for p in single]
    for r in records:assert r['source_text'][r['source_start']:r['source_end']]==r['source_text']
    return {'synthetic_fixture':True,'assertions':len(records),'distinct_entities':len({r['entity_key'] for r in records}),'contexts':CONTEXTS,'single_vs_partitioned_payload_equal':True,'payload_sha256':fingerprint(single),'model_calls':0,'database_calls':0,'latency_measured':False,'evidence':records}
def inputs(mode,concurrency,count):
    if mode not in {'single','sequential','parallel'} or concurrency not in {1,4} or not 1<=count<=32:raise ValueError('Bounded modes/concurrency/count required')
    return {'query':QUERY,'cik':'DEMO0001','contexts':CONTEXTS,'mode':mode,'concurrency':concurrency,'count':count}
def connect():
    from neo4j import GraphDatabase
    required=['NEO4J_URI','NEO4J_USERNAME','NEO4J_PASSWORD']
    if any(not os.environ.get(k) for k in required):raise ValueError('Set NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD in the environment')
    return GraphDatabase.driver(os.environ['NEO4J_URI'],auth=(os.environ['NEO4J_USERNAME'],os.environ['NEO4J_PASSWORD']),max_connection_pool_size=8,max_transaction_retry_time=0)
def ensure_empty(driver,db):
    with driver.session(database=db) as session:
        if session.run('MATCH (n) RETURN count(n) AS n').single()['n']!=0:raise ValueError('Synthetic loader refuses a nonempty database')
def load(driver):
    # Refuse existing data before creating anything or loading any rows.
    ensure_empty(driver,'neo4j')
    with driver.session(database='system') as session:names={r['name'] for r in session.run('SHOW DATABASES')}
    for db in ['ctxa','ctxb']:
        if db in names:ensure_empty(driver,db)
    with driver.session(database='system') as session:
        for db in ['ctxa','ctxb']:
            if db not in names:session.run('CREATE DATABASE '+db).consume()
    deadline=time.monotonic()+30
    while True:
        with driver.session(database='system') as session:states={r['name']:r['currentStatus'] for r in session.run('SHOW DATABASES')}
        if all(states.get(db)=='online' for db in ['neo4j','ctxa','ctxb']):break
        if time.monotonic()>deadline:raise TimeoutError('Database readiness')
        time.sleep(.2)
    rows=fixture()
    for db,context in [('neo4j',None),('ctxa','Accounting'),('ctxb','Legal')]:
        subset=[r for r in rows if context is None or r['context']==context]
        with driver.session(database=db) as session:
            for label,key in [('BenchIssuer','cik'),('BenchEntity','key'),('BenchAssertion','id'),('BenchSource','id')]:session.run(f'CREATE CONSTRAINT IF NOT EXISTS FOR (n:{label}) REQUIRE n.{key} IS UNIQUE').consume()
            session.run('CALL db.awaitIndexes(30)').consume();session.run(LOAD,rows=subset).consume()
    return {'synthetic_fixture_loaded':True,'databases':['neo4j','ctxa','ctxb'],'model_calls':0,'note':'No cleanup/delete command is executed; use an owned disposable DBMS.'}
def verify_samples(result):
    expected=fingerprint([r['payload'] for r in fixture()])
    if not all(fingerprint(r['rows'])==expected for r in result['samples']):raise ValueError('Incomplete or different payload; timing is not comparable')
    result.update(synthetic_fixture=True,payload_parity_all=True,model_calls=0,original_benchmark_reproduced=False)
    return result
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',nargs='?',default='offline',choices=['offline','request','read','load-synthetic']);p.add_argument('--mode',default='single',choices=['single','sequential','parallel']);p.add_argument('--concurrency',type=int,default=1,choices=[1,4]);p.add_argument('--count',type=int,default=4);p.add_argument('--confirm-empty-demo-dbms',action='store_true');a=p.parse_args()
    if not 1<=a.count<=32:p.error('count must be between1and32')
    if a.action=='offline':result=offline()
    elif a.action=='request':result=inputs(a.mode,a.concurrency,a.count)
    else:
        if a.action=='load-synthetic' and not a.confirm_empty_demo_dbms:p.error('Synthetic loading requires --confirm-empty-demo-dbms and an owned disposable DBMS')
        try:
            driver=connect()
            try:result=load(driver) if a.action=='load-synthetic' else verify_samples(run_python(driver,inputs(a.mode,a.concurrency,a.count)))
            finally:driver.close()
        except Exception as error:raise SystemExit('Live operation failed ('+type(error).__name__+'); credentials omitted') from None
    print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
