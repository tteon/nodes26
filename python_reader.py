"""Exact measured Python scheduling function, copied for inspection and reuse."""
import concurrent.futures as cf
import time

QUERY = 'MATCH (i:BenchIssuer {cik:$cik})-[:CONTAINS]->(e:BenchEntity)\n -[:HAS_ASSERTION]->(a:BenchAssertion)-[:SUPPORTED_BY]->(s:BenchSource)\n WHERE a.context IN $contexts RETURN a.payload AS payload ORDER BY a.id LIMIT 65'

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
