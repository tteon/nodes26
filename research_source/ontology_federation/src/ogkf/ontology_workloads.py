"""Bounded, source-pinned ontology formation/reception and native read experiment.

This development protocol measures behavior and exact locator binding, not human
semantic accuracy. It does not relax any other protocol's independent gold gate.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import contextvars
from contextlib import nullcontext
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import html
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

import jsonschema
from neo4j import GraphDatabase

from .environment import load_project_environment
from .enterprise.profiles import load_bundle
from .context_frame.semantic_review import segments
from .local_runtime import local_dozer

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT.parent / 'finder_multi_instance'
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))
from telemetry import Telemetry, audit as trace_audit, encoded
from mara import Mara
from acceptance import extract_message
from storage_guard import require_space

VERSION = 'ontology-workloads-live-v1'
MODEL = 'gpt-oss-120b'
CASE_IDS = ('align-combined-obligations', 'align-segment-only-unavailable',
            'amd-client-revenue', 'amd-client-operating-loss',
            'wynn-required-context-join', 'wynn-board-premise-unavailable')
CIKS = ('0001097149', '0000002488', '0001174922')
CONTEXTS = ('organization', 'performance', 'risk')
PROFILE_PATH = ROOT / 'ontology/fibo_context_v2/profiles.json'
DEFAULT_DATA = Path('<LOCAL_EXPERIMENT_RUNS>/runs/matched-context-scoped-evidence-20261027-v1/dataset.private.json')
DEFAULT_RUNS = DEFAULT_DATA.parent.parent
QUERY = '''MATCH (i:MeaningIssuer {namespace:$namespace,cik:$cik})
MATCH (i)-[:CONTAINS]->(e:MeaningEntity)-[:HAS_ASSERTION]->(a:MeaningAssertion)-[:SUPPORTED_BY]->(s:MeaningSource)
WHERE a.context IN $contexts
RETURN a.payload AS payload, s.id AS source_id, s.text AS source_text
ORDER BY a.id LIMIT 65'''


def sha(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def write(path, value):
    path = Path(path)
    with path.open('x') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n')
    path.chmod(0o600)


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()] if Path(path).exists() else []


def utc():
    return datetime.now(timezone.utc).isoformat()


def dataset(path):
    raw = json.loads(Path(path).read_text())
    sources = sorted((s for s in raw['sources'] if s['cik'] in CIKS), key=lambda s:s['id'])
    facts = sorted((f for f in raw['facts'] if f['cik'] in CIKS), key=lambda f:f['id'])
    cases = sorted((c for c in raw['cases'] if c['id'] in CASE_IDS), key=lambda c:CASE_IDS.index(c['id']))
    assert (len(sources), len(facts), len(cases)) == (9, 43, 6)
    source_map = {s['id']:s for s in sources}
    for s in sources:
        assert s['id'] == 'finder-ref-' + hashlib.sha256(s['text'].encode()).hexdigest()
    for f in facts:
        assert source_map[f['source_id']]['text'][f['quote_start']:f['quote_end']] == f['quote']
        assert source_map[f['source_id']]['cik'] == f['cik']
    fact_ids = {f['id'] for f in facts}
    assert all(set(g) <= fact_ids for c in cases for g in c['required_groups'])
    return {'sources':sources, 'facts':facts, 'cases':cases, 'exposure':'previously exposed constructed development questions; not native FinDER gold',
            'source_projection':'manual reference projection; not model extraction or independent semantic gold'}


def manual_record(f):
    return {**f, 'context':f['collection'], 'modality':'source_assertion',
            'concept_key':'value:MoneyQuantity' if 'USD' in f['unit'] else 'unmapped',
            'support_spans':[{'source_id':f['source_id'], 'start':f['quote_start'], 'end':f['quote_end'], 'text':f['quote']}],
            'admission':'manual_development_projection_not_independent_gold'}


def profile(arm):
    return load_bundle(PROFILE_PATH).prompt_payload(arm, 'competent_generic')


def source_spans(sources):
    lookup = {}
    for s in sorted(sources, key=lambda s:s['id']):
        for part in segments(s['text']):
            alias = f's{len(lookup)}'
            lookup[alias] = {'source_id':s['id'], 'cik':s['cik'], 'context':s['collection'],
                             'accession':s['accession'], **part}
        assert ''.join(v['text'] for v in lookup.values() if v['source_id'] == s['id']) == s['text']
    return lookup


def obj(properties, required=None):
    return {'type':'object','additionalProperties':False,'properties':properties,
            'required':list(properties) if required is None else required}


def index_schema(lookup, bundle):
    fields = {k:{'type':'string'} for k in ('subject','attribute','value','period','scope','unit','basis','modality')}
    fields['concept_key'] = {'type':'string','enum':[t.concept_key for t in bundle.terms]+['unmapped']}
    fields['source_span_ids'] = {'type':'array','items':{'type':'string','enum':list(lookup)},'minItems':1,'maxItems':4,'uniqueItems':True}
    return obj({'observations':{'type':'array','items':obj(fields),'maxItems':24}})


def decode_index(value, schema, lookup, namespace, arm, bundle):
    # A malformed contract is retained as a failed extraction, never repaired.
    jsonschema.validate(value, schema)
    admitted, rejected = [], []
    terms = {t.concept_key:(t.fibo_id if arm == 'FX' else t.generic_id) for t in bundle.terms}
    for n, row in enumerate(value['observations']):
        spans = [lookup[x] for x in row['source_span_ids']]
        if len({s['source_id'] for s in spans}) != 1 or not row['subject'].strip() or not row['attribute'].strip():
            rejected.append({'ordinal':n, 'row':row,'reason':'cross_source_assertion_or_empty_identity'})
            continue
        first = spans[0]
        entity = 'E' + sha([first['cik'], first['context'], ' '.join(row['subject'].casefold().split())])[:24]
        admitted.append({**row,'id':'X'+sha([namespace,first['cik'],n])[:24], 'entity_key':entity,
                         'source_id':first['source_id'],'cik':first['cik'],'context':first['context'],
                         'collection':first['context'],'accession':first['accession'],
                         'support_spans':[{k:s[k] for k in ('source_id','start','end','text')} for s in spans],
                         'term_id':terms.get(row['concept_key']), 'index_profile':arm,
                         'admission':'model_proposal_exact_span_binding_only'})
    return admitted, rejected


def tree_plan(value):
    if value is None:
        return None
    return {k:([tree_plan(c) for c in v] if k == 'children' else v)
            for k,v in value.items()}


def operators(plan):
    if not plan:
        return []
    return [plan.get('operatorType','')] + [x for child in plan.get('children',[]) for x in operators(child)]


class Store:
    def __init__(self, env, telemetry):
        self.driver = GraphDatabase.driver(env['DOZER_URI'], auth=(env['DOZER_USER'],env['DOZER_PASSWORD']),
                                          max_transaction_retry_time=0, connection_timeout=3)
        self.telemetry = telemetry
        self.databases = {'single':'neo4j', **{c:'meaning'+c for c in CONTEXTS}}

    def setup(self):
        with self.driver.session(database='system') as s:
            for c in CONTEXTS:
                s.run('CREATE DATABASE '+self.databases[c]+' IF NOT EXISTS').consume()
        deadline = time.monotonic()+60
        while True:
            with self.driver.session(database='system') as s:
                states = {r['name']:r['currentStatus'] for r in s.run('SHOW DATABASES')}
            if all(states.get(d) == 'online' for d in self.databases.values()):break
            if time.monotonic()>deadline:raise TimeoutError('database readiness')
            time.sleep(.5)
        for db in self.databases.values():
            with self.driver.session(database=db) as s:
                for label, field in (('MeaningIssuer','cik'),('MeaningEntity','key'),('MeaningAssertion','id'),('MeaningSource','id')):
                    s.run(f'CREATE CONSTRAINT {label.lower()} IF NOT EXISTS FOR (n:{label}) REQUIRE (n.namespace,n.{field}) IS UNIQUE').consume()
                s.run('CALL db.awaitIndexes(30)').consume()

    def load(self, namespace, records, sources, *, partitioned=False, episode=None):
        source_map = {s['id']:s for s in sources}
        groups = {'single':records}
        if partitioned:groups.update({c:[r for r in records if r['context']==c] for c in CONTEXTS})
        for target, rows in groups.items():
            payload = [{'namespace':namespace,'cik':r['cik'],'key':r['entity_key'], 'id':r['id'],
                        'source_id':r['source_id'],'text':source_map[r['source_id']]['text'],
                        'context':r['context'],'payload':encoded(r).decode()} for r in rows]
            query='''UNWIND $rows AS row
MERGE (i:MeaningIssuer {namespace:row.namespace,cik:row.cik})
MERGE (e:MeaningEntity {namespace:row.namespace,key:row.key})
MERGE (a:MeaningAssertion {namespace:row.namespace,id:row.id}) SET a.context=row.context,a.payload=row.payload
MERGE (v:MeaningSource {namespace:row.namespace,id:row.source_id}) SET v.text=row.text
MERGE (i)-[:CONTAINS]->(e) MERGE (e)-[:HAS_ASSERTION]->(a) MERGE (a)-[:SUPPORTED_BY]->(v)'''
            with self.telemetry.span('index_write',episode) if episode else nullcontext(None) as span:
                started=time.perf_counter_ns()
                with self.driver.session(database=self.databases[target]) as s:
                    summary=s.run(query,rows=payload).consume()
                elapsed=(time.perf_counter_ns()-started)/1e6
                if span is not None:
                    span['receipt_sha256']=self.telemetry.receipt({'database':self.databases[target],
                        'query':query,'parameters':{'rows':payload},'namespace':namespace,'record_count':len(rows),
                        'native_write_ms':elapsed,'nodes_created':summary.counters.nodes_created,
                        'relationships_created':summary.counters.relationships_created,
                        'properties_set':summary.counters.properties_set,'native_transactions':1})

    def branch(self, target, namespace, cik, contexts, mode=''):
        started_unix_ns, started = time.time_ns(), time.perf_counter_ns()
        with self.driver.session(database=self.databases[target], default_access_mode='READ') as s:
            result = s.run((mode+' ' if mode else '')+QUERY, namespace=namespace,cik=cik,contexts=contexts)
            rows = [dict(r) for r in result]
            summary = result.consume()
        ended = time.perf_counter_ns()
        if len(rows)>64:raise RuntimeError('global fact cap exceeded; truncation is not admitted')
        records = [json.loads(r['payload']) for r in rows]
        for row, record in zip(rows, records):
            assert record['source_id']==row['source_id'] and record['cik']==cik and record['context'] in contexts
            for witness in record['support_spans']:
                assert witness['source_id']==row['source_id']
                assert row['source_text'][witness['start']:witness['end']]==witness['text']
        return {'database':self.databases[target], 'query':QUERY,'query_mode':mode,'parameters':{'namespace':namespace,'cik':cik,'contexts':contexts},
                'started_unix_ns':started_unix_ns,'started_monotonic_ns':started,'ended_monotonic_ns':ended,
                'duration_ms':(ended-started)/1e6,'rows':records,'row_count':len(rows),
                'result_available_after_ms':summary.result_available_after,'result_consumed_after_ms':summary.result_consumed_after,
                'plan':tree_plan(summary.profile or summary.plan), 'connection_acquisition_ms':None,
                'transaction':'one native read autocommit per branch; no global transaction',
                'row_payload_sha256':sha(records), 'source_text_sha256':{r['source_id']:hashlib.sha256(r['source_text'].encode()).hexdigest() for r in rows}}

    def read(self, episode, namespace, cik, contexts, arm='single', mode=''):
        targets = ['single'] if arm=='single' else list(contexts)
        with self.telemetry.span('graph_read',episode) as span:
            submitted = time.perf_counter_ns()
            if arm=='parallel':
                with ThreadPoolExecutor(max_workers=2) as pool:
                    jobs = [pool.submit(contextvars.copy_context().run,self.branch,t,namespace,cik,[t],mode) for t in targets]
                    branches = [j.result() for j in jobs]
            else:
                branches = [self.branch(t,namespace,cik,contexts if t=='single' else [t],mode) for t in targets]
            returned = time.perf_counter_ns()
            records = sorted((r for b in branches for r in b['rows']),key=lambda r:r['id'])
            assert len(records)<=64 and len({r['id'] for r in records})==len(records)
            overlap_ns = sum(max(0,min(a['ended_monotonic_ns'],b['ended_monotonic_ns'])-max(a['started_monotonic_ns'],b['started_monotonic_ns']))
                             for i,a in enumerate(branches) for b in branches[i+1:])
            receipt = {'arm':arm,'namespace':namespace,'cik':cik,'contexts':contexts,'mode':mode,'records':records,
                       'records_sha256':sha(records),'branches':branches,'bolt_transactions':len(branches),
                       'query_wall_ms':(returned-submitted)/1e6,'sum_branch_ms':sum(b['duration_ms'] for b in branches),
                       'pairwise_branch_overlap_ms':overlap_ns/1e6,'timing':'native driver execute/consume including session; receipt fsync excluded',
                       'payload_bytes':len(encoded(records)),'graph_hops':'issuer -> contextual entity -> assertion -> source; no OWL entailment'}
            for b in branches:
                receipt_sha = self.telemetry.receipt(b)
                self.telemetry.external_span('native_read',episode,b['database'],b['started_unix_ns'],b['ended_monotonic_ns']-b['started_monotonic_ns'],receipt_sha,
                                             timestamp_origin='python_time_time_ns_at_branch_start',duration_origin='python_perf_counter_ns')
            span['receipt_sha256']=self.telemetry.receipt(receipt)
            return receipt

    def close(self):self.driver.close()


def qualify(store, data):
    records=[manual_record(f) for f in data['facts']]
    store.load('reference',records,data['sources'],partitioned=True)
    checks=[]
    for case in data['cases']:
        ep='qualification-'+case['id']
        expected=sorted([r for r in records if r['cik']==case['cik']],key=lambda r:r['id'])
        with store.telemetry.span('qualification',ep):
            for arm in ('single','sequential','parallel'):
                result=store.read(ep,'reference',case['cik'],CONTEXTS,arm)
                assert result['records']==expected
            plan=store.read(ep,'reference',case['cik'],CONTEXTS,'single','EXPLAIN')
            assert any('Index' in name and 'Seek' in name for b in plan['branches'] for name in operators(b['plan']))
            absent=store.read(ep,'reference','0000000000',CONTEXTS)
            assert not absent['records']
        checks.append({'case_id':case['id'],'native_reference_parity':True,'source_span_binding':True,'indexed_anchor':True,'unknown_issuer_empty':True})
    return checks


def db_study(store, telemetry):
    requests=[('wynn-all','0001174922',('organization','risk')),('wynn-risk','0001174922',('risk',)),
              ('align-performance','0001097149',('performance',)),('align-cross','0001097149',('organization','performance')),
              ('amd-performance','0000002488',('performance',)),('amd-cross','0000002488',('organization','performance'))]
    assignments=[]
    for request,cik,contexts in requests:
        for mode in ('PROFILE','warmup','timed'):
            repeats=1 if mode=='PROFILE' else 3 if mode=='warmup' else 10
            for repeat in range(repeats):
                arms=['single','sequential','parallel'];random.Random(f'{VERSION}:{request}:{mode}:{repeat}').shuffle(arms)
                assignments.extend({'request':request,'cik':cik,'contexts':contexts,'phase':mode,'repeat':repeat,'arm':a} for a in arms)
    write(telemetry.directory/'assignments.json',assignments)
    expected={}
    for n, job in enumerate(assignments):
        ep=f'db-{n:04}'
        with telemetry.span('db_episode',ep):
            result=store.read(ep,'reference',job['cik'],job['contexts'],job['arm'],'PROFILE' if job['phase']=='PROFILE' else '')
            if job['request'] not in expected:expected[job['request']]=result['records_sha256']
            assert expected[job['request']]==result['records_sha256']
            telemetry.append('db-outcomes.jsonl',{'episode_id':ep,**job,**result})
    telemetry.flush_metrics()


class BoundedMara(Mara):
    def _request(self,*args,**kwargs):
        # Called while the legacy global flock is already held. Coordinate with
        # the maintained gateway's unknown-outcome marker too.
        lock=Path(os.environ.get('MARA_ADMISSION_LOCK',str(self.telemetry.directory.parent/'.mara-global.lock')))
        if lock.resolve() != (self.telemetry.directory.parent/'.mara-global.lock').resolve():
            raise RuntimeError('injected admission lock differs from this frozen protocol')
        if lock.with_suffix('.pending').exists():raise RuntimeError('shared gateway unresolved request')
        spans=read_rows(self.telemetry.directory/'spans.jsonl')
        attempts=[s for s in spans if s['component']=='mara']
        known=sum((s['measurements'].get(k) or 0) for s in attempts for k in ('input_tokens','output_tokens'))
        if len(attempts)+kwargs.get('max_attempts',1)>400 or known>=2_000_000:
            raise RuntimeError('campaign admission budget exhausted')
        result=super()._request(*args,**kwargs)
        if result.get('unresolved'):raise RuntimeError('unknown provider outcome; entire campaign halted without retry')
        return result


def body(messages, name, schema, cap):
    value={'model':MODEL,'temperature':0,'max_tokens':cap,'messages':messages,'parallel_tool_calls':False,
           'tools':[{'type':'function','function':{'name':name,'description':'Submit the exact requested structured result.','parameters':schema}}],
           'tool_choice':{'type':'function','function':{'name':name}}}
    if len(encoded(value))>65536:raise ValueError('request byte cap')
    return value


def call(client, ep, request_id, messages, name, schema, cap):
    request=body(messages,name,schema,cap)
    response=client.request(ep,request_id,request,max_attempts=2)
    value=extract_message(response,name)
    error=None
    try:jsonschema.validate(value,schema)
    except jsonschema.ValidationError as e:error=e.validator
    return {'request':request,'response':response,'value':value,'schema_error':error}


def catalog(records):
    grouped=defaultdict(list)
    for r in records:grouped[r['entity_key']].append(r)
    return [{'entity_key':key,'subject':rows[0]['subject'],'contexts':sorted({r['context'] for r in rows}),
             'attributes':sorted({r['attribute'] for r in rows})} for key,rows in sorted(grouped.items())]


def packet(records):
    spans={}
    evidence=[]
    for record in records:
        pointers=[]
        for witness in record['support_spans']:
            pointer='W'+sha(witness)[:24]
            spans[pointer]=witness
            pointers.append(pointer)
        # Lossless witness interning avoids repeating a whole financial table
        # for each row. Qualifiers and assertion IDs are left unchanged.
        evidence.append({k:v for k,v in record.items()
                         if k not in {'support_spans','quote','quote_start','quote_end','source_span_ids'}} | {'witness_ids':pointers})
    value={'evidence':evidence,'source_spans':spans,'scope':'Only admitted source spans and their qualified assertion proposals are available. Missing evidence is unknown, not globally false.',
           'contract':'entity_key is domain-local; shared CIK supports an explicit issuer join, not substitution of entities or relations.',
           'source_support':'Exact locator binding does not establish entailment. Compare subjects, attributes, units, periods and source text.'}
    if len(encoded(value))>49152:raise ValueError('evidence packet byte cap')
    return value


def answer_schema(records):
    return obj({'value':{'type':['string','null']},'abstain':{'type':'boolean'},'issuer_cik':{'type':'string'},
                'evidence_ids':{'type':'array','items':{'type':'string','enum':[r['id'] for r in records] or ['no_evidence']},'uniqueItems':True},
                'numerator':{'type':['string','null']},'denominator':{'type':['string','null']},
                'subject_used':{'type':'string'},'relation_used':{'type':'string'},'explanation':{'type':'string'}})


def number_equal(a,b):
    try:return Decimal(str(a).replace(',','').strip())==Decimal(str(b).replace(',','').strip())
    except (InvalidOperation,ValueError):return False


def interval_covered(target, spans):
    cursor=target['quote_start']
    for lo,hi in sorted((s['start'],s['end']) for s in spans if s['source_id']==target['source_id']):
        if lo>cursor:break
        cursor=max(cursor,hi)
    return cursor>=target['quote_end']


def score(case, answer, records, reference):
    if not isinstance(answer,dict):return {'atomic_target_match':False,'citation_locator_coverage':False,'independent_semantic_appropriateness':None}
    byid={r['id']:r for r in records};cited=[byid[x] for x in answer.get('evidence_ids',[]) if x in byid]
    spans=[s for r in cited for s in r['support_spans']]
    targets={r['id']:r for r in reference}
    coverage=all(any(interval_covered(targets[f],spans) for f in group) for group in case['required_groups'])
    expected=case['expected']
    match=(answer.get('abstain') is True and answer.get('value') is None) if expected is None else (answer.get('abstain') is False and number_equal(answer.get('value'),expected))
    operands=True
    if expected is not None and 'numerator' in case:
        operands=number_equal(answer.get('numerator'),case['numerator']) and number_equal(answer.get('denominator'),case['denominator'])
    return {'atomic_target_match':match and operands and answer.get('issuer_cik')==case['cik'],
            'citation_locator_coverage':coverage,'citation_ids_valid':all(x in byid for x in answer.get('evidence_ids',[])),
            'required_group_count':len(case['required_groups']),'ratio_operands_match':operands,
            'independent_semantic_appropriateness':None,'index_semantic_precision':None,
            'warning':'Numeric match and locator coverage are separate mechanical diagnostics; neither proves semantic entailment.'}


def agent_episode(store, client, telemetry, data, job):
    case=next(c for c in data['cases'] if c['id']==job['case_id'])
    ep=job['episode_id'];namespace=job['namespace']
    with telemetry.span('agent_episode',ep):
        telemetry.event('agent_bindings',{'case_id':case['id'],'index_model':MODEL if job['workload']=='index_to_answer' else 'not_model_extracted',
            'query_model':MODEL,'representation':job['query_profile'],'topology':'qualified_single','graph_run':namespace,
            'ontology_condition':{'package':job['query_profile'],'role':'system','position':'after_task'}})
        # A pre-model public catalog query uses no private case target values.
        catalog_read=store.read(ep,namespace,case['cik'],CONTEXTS)
        records=catalog_read['records']
        entities=catalog(records)
        schema=obj({'entity_key':{'type':'string','enum':[e['entity_key'] for e in entities]+['unresolved']},'attribute':{'type':'string'}})
        messages=[{'role':'system','content':'Answer the public financial question using the attached graph evidence. Select a contextual starting entity. Request evidence once, then submit a result; abstain on unsupported premises. Do not infer facts from ontology definitions. For ratios cite premises and operands; return numerator and denominator.\nSemantic reference package:\n'+json.dumps(profile(job['query_profile']),ensure_ascii=False)},
                  {'role':'user','content':json.dumps({'question':case['question'],'issuer_cik':case['cik'],'entity_catalog':entities},ensure_ascii=False)}]
        intent=call(client,ep,ep+'-intent',messages,'fetch_attached_evidence',schema,1024)
        if intent['schema_error']:
            outcome={'status':'intent_contract_failure','intent':intent,'answer':None,'packet_sha256':None}
        else:
            # Selection is observed, not silently corrected using private gold.
            # The issuer-wide evidence contract is fixed across query profiles.
            actual=store.read(ep,namespace,case['cik'],CONTEXTS,
                              'parallel' if job['workload']=='fixed_evidence' else 'single')
            assert actual['records']==records
            evidence=packet(records)
            original=intent['response']['response']['choices'][0]['message']
            tool=original['tool_calls'][0]
            continuation=messages+[original,{'role':'tool','tool_call_id':tool['id'],'content':encoded(evidence).decode()}]
            answer=call(client,ep,ep+'-answer',continuation,'submit_results',answer_schema(records),2048)
            outcome={'status':'completed' if not answer['schema_error'] else 'answer_contract_failure','intent':intent,'answer':answer,
                     'packet_sha256':sha(evidence),'packet':evidence,'graph_receipt_sha256':telemetry.receipt(actual)}
        submitted=outcome['answer']['value'] if outcome.get('answer') and not outcome['answer']['schema_error'] else None
        outcome.update({'episode_id':ep,**job,'case':case,'model':MODEL,'score':score(case,submitted,records,data['facts']),
                        'delivered_fact_count':len(records),'catalog_receipt_sha256':telemetry.receipt(catalog_read)})
        telemetry.append('outcomes.jsonl',outcome)
        telemetry.flush_metrics()
        print(json.dumps({'episode':ep,'workload':job['workload'],'condition':job['index_profile']+'/'+job['query_profile'],'case':case['id'],'status':outcome['status'],'atomic_target_match':outcome['score']['atomic_target_match']}),flush=True)


def semantic_study(store, client, telemetry, data, bundle):
    with telemetry.span('capability','capability'):
        listing=client.request('capability','models',path='/models',max_attempts=2)
        assert listing['http_status']==200
        ids={r['id'] for r in listing['response'].get('data',[])}
        assert MODEL in ids
        probe=call(client,'capability','tool-probe',[{'role':'user','content':'Return ready=true.'}],'ready',obj({'ready':{'type':'boolean'}}),1024)
        assert not probe['schema_error'] and probe['value']=={'ready':True}
    index_jobs=[]
    for repeat in range(3):
        for arm in ('GX','FX'):
            for cik in CIKS:index_jobs.append({'repeat':repeat,'arm':arm,'cik':cik,'namespace':f'index-{arm}-{repeat}'})
    random.Random(20261104).shuffle(index_jobs)
    episodes=[]
    for repeat in range(3):
        for case in data['cases']:
            for arm in ('GX','FX'):
                episodes.append({'workload':'index_to_answer','case_id':case['id'],'repeat':repeat,'index_profile':arm,'query_profile':'GX','namespace':f'index-{arm}-{repeat}'})
            for arm in ('G','GX','FX'):
                episodes.append({'workload':'fixed_evidence','case_id':case['id'],'repeat':repeat,'index_profile':'manual','query_profile':arm,'namespace':'reference'})
    random.Random(20261105).shuffle(episodes)
    for n, job in enumerate(episodes):job['episode_id']=f'ow-{n:03}'
    write(telemetry.directory/'assignments.json',{'index':index_jobs,'agent':episodes})
    for n,job in enumerate(index_jobs):
        ep=f'index-{n:02}';sources=[s for s in data['sources'] if s['cik']==job['cik']];lookup=source_spans(sources)
        schema=index_schema(lookup,bundle)
        with telemetry.span('index_episode',ep):
            messages=[{'role':'system','content':'Extract qualified financial assertions from every supplied source passage, without seeing downstream questions. Preserve different subject/attribute/period/scope/unit/basis/modality and distinguish membership, reporting and decision authority. Select the most relevant concept_key or unmapped, and cite immutable source span IDs. Extract at most 24 salient observations across all passages; never invent missing facts.\nSemantic reference package:\n'+json.dumps(profile(job['arm']),ensure_ascii=False)},
                      {'role':'user','content':json.dumps({'issuer_cik':job['cik'],'source_spans':lookup},ensure_ascii=False)}]
            result=call(client,ep,ep,messages,'submit_observations',schema,4096)
            records,rejected=[],[]
            if not result['schema_error']:
                records,rejected=decode_index(result['value'],schema,lookup,job['namespace'],job['arm'],bundle)
            store.load(job['namespace'],records,sources,episode=ep)
            native=store.read(ep,job['namespace'],job['cik'],CONTEXTS)
            assert native['records']==sorted(records,key=lambda r:r['id'])
            telemetry.append('index-outcomes.jsonl',{'episode_id':ep,**job,'result':result,'lookup':lookup,'records':records,'rejected':rejected,
                                                  'native_readback_sha256':sha(native['records']),'independent_semantic_precision':None})
            print(json.dumps({'index':ep,'arm':job['arm'],'issuer':job['cik'],'admitted':len(records),'rejected':len(rejected),'schema_error':result['schema_error']}),flush=True)
    for job in episodes:agent_episode(store,client,telemetry,data,job)


def execute(kind, directory, data_path):
    load_project_environment()
    require_space(directory.parent.parent,root_min_gib=10,stage_min_gib=20)
    if directory.exists():raise RuntimeError('run directory already exists; no implicit replay')
    directory.mkdir(parents=True);directory.chmod(0o700)
    (directory/'source-code.py').write_bytes(Path(__file__).read_bytes())
    data=dataset(data_path);bundle=load_bundle(PROFILE_PATH)
    manifest={'protocol':VERSION,'kind':kind,'started_utc':utc(),'model':MODEL,'model_families_added':0,
              'input_sha256':sha(data),'profile_hash':bundle.profile_hash,'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'cases':list(CASE_IDS),'issuer_families':list(CIKS),'independent_sample_count':3,'repeats':3,
              'initial_http_plan':200 if kind=='mara' else 0,'max_attempts':400,'known_token_stop':2000000,
              'index_max_output_tokens':4096,'intent_max_output_tokens':1024,'answer_max_output_tokens':2048,
              'independent_semantic_appropriateness':None,'mapping_review_status':bundle.mapping_review_status,'reasoner':'none',
              'contrast':'G->GX: application guidance; GX->FX: definitions+identifiers at fixed guidance',
              'dbms':'same owned DozerDB; isolated standard databases; application fanout; no native Composite/global transaction',
              'sampling':'all six predeclared exposed constructed questions from three selected source families; no heldout/general lift claim',
              'context_owner_labels':'manual organization/performance/risk projection, not measured real teams',
              'query_delivery':'fixed_evidence: actual multi-DB parallel reads with qualified single-DB parity; index_to_answer: actual single-DB index-build read',
              'max_logical_concurrency':1,'max_inner_db_workers':2,'stage_storage':str(directory),'status':'running'}
    write(directory/'manifest.json',manifest);write(directory/'dataset.private.json',data)
    telemetry=Telemetry(directory,directory.name)
    runtime=None
    try:
        with local_dozer() as (env,runtime):
            store=Store(env,telemetry)
            try:
                store.setup();gate=qualify(store,data)
                write(directory/'qualification.json',gate);write(directory/'runtime.json',runtime.copy())
                telemetry.host_sample()
                if kind=='db':db_study(store,telemetry)
                else:semantic_study(store,BoundedMara(telemetry,base=os.environ.get('MARA_BASE_URL','https://api.cloud.mara.com/v1')),telemetry,data,bundle)
            finally:store.close()
        manifest['status']='completed'
    except BaseException as error:
        manifest['status']='halted';manifest['exception_type']=type(error).__name__
        telemetry.event('campaign_halted',{'exception_type':type(error).__name__,'message':str(error)[:500]})
        raise
    finally:
        telemetry.host_sample();telemetry.flush_metrics()
        manifest['finished_utc']=utc();manifest['runtime_final']=runtime
        write(directory/'terminal.json',manifest)
        write(directory/'trace-audit.json',trace_audit(directory))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('kind',choices=['db','mara']);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--data',type=Path,default=DEFAULT_DATA)
    a=p.parse_args();execute(a.kind,a.directory,a.data)

if __name__=='__main__':main()
