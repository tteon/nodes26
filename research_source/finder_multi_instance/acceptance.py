#!/usr/bin/env python3
"""Real MARA extraction -> DozerDB projection -> retrieval -> MARA answer, narrow diagnostics."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re
import time
from neo4j import GraphDatabase,Query
from mara import Mara
from telemetry import Telemetry,audit,encoded
from evidence import source_units,bind_units
from scoring import strict_value_match
import copy

EXTRACT={'type':'function','function':{'name':'submit_facts','description':'Submit source-backed facts, preserving distinct subjects, roles, scopes, periods and measurement bases.','parameters':{'type':'object','properties':{'facts':{'type':'array','maxItems':20,'items':{'type':'object','properties':{k:{'type':'string'} for k in ['subject','relation','value','period','scope','unit','quote']},'required':['subject','relation','value','period','scope','unit','quote'],'additionalProperties':False}}},'required':['facts'],'additionalProperties':False}}}
EVIDENCE_EXTRACT=copy.deepcopy(EXTRACT)
EVIDENCE_EXTRACT['function']['parameters']['properties']['facts']['items']['properties'].pop('quote')
EVIDENCE_EXTRACT['function']['parameters']['properties']['facts']['items']['properties']['evidence_span_ids']={'type':'array','items':{'type':'string'},'minItems':1,'maxItems':4}
EVIDENCE_EXTRACT['function']['parameters']['properties']['facts']['items']['required'].remove('quote')
EVIDENCE_EXTRACT['function']['parameters']['properties']['facts']['items']['required'].append('evidence_span_ids')
ANSWER={'type':'function','function':{'name':'submit_answer','description':'Answer only from returned evidence or abstain. Value should be the requested atomic value without explanation.','parameters':{'type':'object','properties':{'value':{'type':'string'},'source_ids':{'type':'array','items':{'type':'string'}},'abstain':{'type':'boolean'}},'required':['value','source_ids','abstain'],'additionalProperties':False}}}
INDEX_MAX_OUTPUT_TOKENS=4096
INDEX_MAX_FACTS=20
PORTS={'treasury':19089,'risk':19090,'governance':19091}


def normalized(value):return re.sub(r'[^a-z0-9.]','',str(value).lower())
def extract_message(result,name):
    if result['http_status']!=200:return None
    message=((result.get('response') or {}).get('choices') or [{}])[0].get('message') or {}
    calls=message.get('tool_calls') or []
    if len(calls)!=1 or calls[0].get('function',{}).get('name')!=name:return None
    try:return json.loads(calls[0]['function']['arguments'])
    except (KeyError,ValueError,TypeError):return None

def body(model,messages,tool):
    data={'model':model,'messages':messages,'tools':[tool],'tool_choice':{'type':'function','function':{'name':tool['function']['name']}},'temperature':0,'max_tokens':INDEX_MAX_OUTPUT_TOKENS if tool['function']['name']=='submit_facts' else 512}
    if model=='MiniMax-M3':data['chat_template_kwargs']={'enable_thinking':False}
    return data


def driver(storage,context):
    auth=json.loads((storage/'private/db-credentials.json').read_text())
    return GraphDatabase.driver('bolt://127.0.0.1:'+str(PORTS[context]),auth=(auth['username'],auth['password']),connection_timeout=5,max_connection_pool_size=4)


def statement(t,storage,episode,context,query,parameters):
    with t.span('bolt',episode,context) as span:
        with driver(storage,context) as d:
            with d.session(database='neo4j') as session:
                result=session.run(Query(query,timeout=30),parameters=parameters);rows=result.data();summary=result.consume()
        span['receipt_sha256']=t.receipt({'query':query,'parameters':parameters,'rows':rows,'server':str(summary.server.address),
             'result_available_after_ms':summary.result_available_after,'result_consumed_after_ms':summary.result_consumed_after})
        return rows


def create_index(storage,run_id,credential_file,evidence_mode='quote'):
    destination=storage/'runs'/run_id;destination.mkdir(parents=True,exist_ok=False)
    dataset=storage/'acceptance-v1';sources=json.loads((dataset/'sources.json').read_text())
    t=Telemetry(destination,run_id);client=Mara(t,credential_file)
    manifest={'run_id':run_id,'mode':'source_derived_acceptance_not_native_qa','source_count':len(sources),
              'source_input_sha256':hashlib.sha256((dataset/'sources.json').read_bytes()).hexdigest(),
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'support_code_sha256':{name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['mara.py','telemetry.py','evidence.py','financial_gold.py','quantities.py']},'question_or_gold_in_indexer_input':False,
              'models':['MiniMax-M3','gpt-oss-120b'],'packages':['G'],'qualification_max_output_tokens':INDEX_MAX_OUTPUT_TOKENS,'qualification_max_facts':INDEX_MAX_FACTS,'not_main_work_budget':True,'evidence_mode':evidence_mode,'complete':False}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    jobs=[]
    for model in manifest['models']:
        for i,source in enumerate(sources):jobs.append({'id':f'index-{model.replace("-","_")}-{i:02d}','model':model,'source':source,'state':'assigned'})
    with (destination/'index-assignments.jsonl').open('x') as f:
        for job in jobs:f.write(json.dumps({'id':job['id'],'model':job['model'],'source_id':job['source']['source_id'],'state':'assigned'})+'\n')
    outcomes=[]
    for job in jobs:
        episode=job['id'];source=job['source'];model=job['model'];accepted=[];rejected=[]
        units=source_units(source['source_id'],source['text'])
        index_input={k:source[k] for k in ['text','issuer','cik','accession']}
        extraction_instruction='Each fact must include an exact contiguous source quote. A quote may include table headers plus rows.'
        if evidence_mode=='span_ids':
            index_input.pop('text');index_input['source_units']=units
            extraction_instruction='Cite evidence_span_ids from the supplied units. Never copy or invent source quotes. Preserve table header context in cited units.'
        with t.span('index_episode',episode):
            # Connectivity probe is separate from the evaluator-only financial Gold gate below.
            rows=statement(t,storage,episode,source['context'],'RETURN 1 AS ok',{})
            if rows!=[{'ok':1}]:raise RuntimeError('DB connectivity probe failed')
            with t.span('source_ingest',episode) as span:
                span['receipt_sha256']=t.receipt(source)
            from financial_gold import require_gold
            require_gold(storage,t,episode,source['source_id'])
            with t.span('index_extract',episode):
                result=client.request(episode,episode,body(model,[{'role':'system','content':f'Extract up to {INDEX_MAX_FACTS} useful atomic facts from the entire source, independently of any question. Preserve who does what to whom and numeric period, scope, unit and basis. '+('Prioritize substantive roles/relationships over standalone age facts. ' if INDEX_MAX_FACTS<20 else '')+extraction_instruction+' Never infer unsupported effects. Empty context strings mean not stated. Do not conflate parent executives with subsidiary executives or totals with subsets.'},
                    {'role':'user','content':json.dumps(index_input)}],EXTRACT if evidence_mode=='quote' else EVIDENCE_EXTRACT))
            data=extract_message(result,'submit_facts')
            with t.span('index_validate',episode):
                if not isinstance(data,dict) or not isinstance(data.get('facts'),list):rejected.append({'reason':'invalid_extraction_contract'})
                else:
                    for index,fact in enumerate(data['facts']):
                        span_ids=[]
                        if evidence_mode=='span_ids' and isinstance(fact,dict):
                            fact=dict(fact);span_ids=fact.pop('evidence_span_ids',None)
                            try:fact['quote']=bind_units(source['text'],units,span_ids)
                            except ValueError as error:
                                rejected.append({'reason':str(error),'index':index,'fact':fact});continue
                        if not isinstance(fact,dict) or set(fact)!=set(['subject','relation','value','period','scope','unit','quote']) or not all(isinstance(v,str) for v in fact.values()):
                            rejected.append({'reason':'invalid_fact_contract','index':index});continue
                        quote=fact['quote'];offset=source['text'].find(quote)
                        if not quote or offset<0:
                            rejected.append({'reason':'quote_not_exact','index':index,'fact':fact});continue
                        accepted.append({**fact,'evidence_span_ids':span_ids,'quote_start':offset,'quote_end':offset+len(quote),'id':hashlib.sha256(encoded([run_id,model,source['source_id'],index,fact])).hexdigest()})
            with t.span('graph_project',episode):
                statement(t,storage,episode,source['context'],
                    'CREATE CONSTRAINT finder_acceptance_source IF NOT EXISTS FOR (s:FinderAcceptanceSource) REQUIRE s.key IS UNIQUE',{})
                statement(t,storage,episode,source['context'],
                    'CREATE CONSTRAINT finder_acceptance_fact IF NOT EXISTS FOR (f:FinderAcceptanceFact) REQUIRE f.id IS UNIQUE',{})
                key=f"{run_id}:{model}:{source['source_id']}"
                statement(t,storage,episode,source['context'],
                    'MERGE (s:FinderAcceptanceSource {key:$key}) SET s.source_id=$source_id,s.build=$build,s.model=$model,s.text=$text,s.issuer=$issuer,s.cik=$cik,s.context=$context '
                    'WITH s UNWIND $facts AS row MERGE (f:FinderAcceptanceFact {id:row.id}) SET f += row MERGE (s)-[:HAS_FACT]->(f) RETURN count(f) AS facts',
                    {'key':key,'source_id':source['source_id'],'build':run_id,'model':model,'text':source['text'],'issuer':source['issuer'],'cik':source['cik'],'context':source['context'],'facts':accepted})
            row={'id':episode,'model':model,'source_id':source['source_id'],'context':source['context'],'accepted':len(accepted),'rejected':len(rejected),
                 'status':'completed' if data is not None else 'failed','receipt_sha256':t.receipt({'accepted':accepted,'rejected':rejected})}
            t.append('index-outcomes.jsonl',row);outcomes.append(row)
            t.host_sample();t.flush_metrics()
    manifest.update(complete=True,accepted_facts=sum(r['accepted'] for r in outcomes),rejected_facts=sum(r['rejected'] for r in outcomes))
    (destination/'completed.json').write_text(json.dumps({'manifest':manifest,'trace_audit':audit(destination)},indent=2)+'\n')
    print(json.dumps({'directory':str(destination),'complete':True,'jobs':len(outcomes),'accepted_facts':manifest['accepted_facts'],'rejected_facts':manifest['rejected_facts']},indent=2))


def query_existing(storage,run_id,build,credential_file):
    directory=storage/'runs'/run_id;directory.mkdir(parents=True,exist_ok=False)
    t=Telemetry(directory,run_id);client=Mara(t,credential_file)
    dataset=storage/'acceptance-v1'
    questions=json.loads((dataset/'questions.public.json').read_text())
    gold={x['id']:x for x in json.loads((dataset/'gold.private.json').read_text())}
    outcomes=[]
    assignment=[{'episode_id':q['id']+'-'+m.replace('-','_')+'-'+arm,'case_id':q['id'],'model':m,'arm':arm,'state':'assigned'}
        for m in ['MiniMax-M3','gpt-oss-120b'] for q in questions for arm in ['bm25_text','bm25_graph_facts']]
    (directory/'assignments.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in assignment))
    (directory/'manifest.json').write_text(json.dumps({'run_id':run_id,'build_id':build,'assignment_count':len(assignment),
        'scope':'source_derived_narrow_acceptance','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'financial_gold_contract_sha256':hashlib.sha256((Path(__file__).parent/'financial_gold.py').read_bytes()).hexdigest(),'questions_sha256':hashlib.sha256((dataset/'questions.public.json').read_bytes()).hexdigest(),'scorer_sha256':hashlib.sha256((Path(__file__).parent/'scoring.py').read_bytes()).hexdigest()},indent=2)+'\n')
    for model in ['MiniMax-M3','gpt-oss-120b']:
        # Deterministic source/fact projection receipt; no private gold IDs in retrieval.
        packets=[]
        for context in PORTS:
            packets.extend(statement(t,storage,'catalog',context,
                'MATCH (s:FinderAcceptanceSource {build:$build,model:$model}) OPTIONAL MATCH (s)-[:HAS_FACT]->(f) '
                'RETURN s.source_id AS source_id,s.text AS text,s.issuer AS issuer,s.context AS context,collect(properties(f)) AS facts',
                {'build':build,'model':model}))
        if len(packets)!=8:raise RuntimeError('source inventory Gold failed; no paid queries')
        import importlib.util
        spec=importlib.util.spec_from_file_location('context_analyze',Path(__file__).parent/'context_audit/analyze.py');lex=importlib.util.module_from_spec(spec);spec.loader.exec_module(lex)
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizer=CountVectorizer(analyzer=lex.tokenize);counts=vectorizer.fit_transform([p['text'] for p in packets]);weights=lex.bm25_matrix(counts)
        for question in questions:
            seed=vectorizer.transform([question['question']]);seed.data[:]=1
            scores=(seed@weights.T).toarray().ravel();indices=lex.ranking(scores)[:3]
            for arm in ['bm25_text','bm25_graph_facts']:
                episode=question['id']+'-'+model.replace('-','_')+'-'+arm
                started=time.perf_counter_ns()
                with t.span('query_episode',episode):
                    with t.span('lexical_seed',episode):
                        selected=[packets[int(i)] for i in indices]
                        chosen=[]
                        t.event('retrieval_selection',{'source_ids':[p['source_id'] for p in selected],'scores':[float(scores[i]) for i in indices],'category_used':False})
                    with t.span('graph_expand' if arm=='bm25_graph_facts' else 'source_fetch',episode):
                        fetched=[]
                        for context in sorted({p['context'] for p in selected}):
                            identifiers=[p['source_id'] for p in selected if p['context']==context]
                            fetched.extend(statement(t,storage,episode,context,
                                'MATCH (s:FinderAcceptanceSource {build:$build,model:$model}) WHERE s.source_id IN $source_ids '
                                'OPTIONAL MATCH (s)-[:HAS_FACT]->(f) RETURN s.source_id AS source_id,s.text AS text,s.issuer AS issuer,s.context AS context,collect(properties(f)) AS facts',
                                {'build':build,'model':model,'source_ids':identifiers}))
                        fetched_by_id={p['source_id']:p for p in fetched}
                        chosen=[fetched_by_id[p['source_id']] for p in selected if p['source_id'] in fetched_by_id]
                    with t.span('context_pack',episode) as span:
                        if arm=='bm25_text':evidence=[{k:p[k] for k in ['source_id','issuer','text']} for p in chosen]
                        else:evidence=[{k:p[k] for k in ['source_id','issuer','context','facts']} for p in chosen]
                        span['receipt_sha256']=t.receipt(evidence)
                    from financial_gold import require_gold
                    require_gold(storage,t,episode)
                    with t.span('answer',episode):
                        result=client.request(episode,episode,body(model,[{'role':'system','content':'Use only retrieved source evidence. Distinguish entity role, scope, measurement basis and year. Return the requested atomic value using submit_answer; numbers should be plain digits with optional decimal, no units or commas. Include supporting source IDs. If required facts are absent, abstain. Graph facts can be incomplete or incorrect; use their exact source quotes to check.'},
                            {'role':'user','content':json.dumps({'question':question['question'],'evidence':evidence})}],ANSWER))
                    answer=extract_message(result,'submit_answer')
                    with t.span('verify',episode):
                        expected=gold[question['id']]
                        valid=isinstance(answer,dict) and isinstance(answer.get('value'),str) and isinstance(answer.get('source_ids'),list) and type(answer.get('abstain')) is bool
                        value_correct=valid and strict_value_match(answer['value'],expected['answer']) and not answer['abstain']
                        cited=valid and expected['source_id'] in answer['source_ids']
                        delivered=expected['source_id'] in {p['source_id'] for p in chosen}
                        row={'episode_id':episode,'case_id':question['id'],'model':model,'arm':arm,'build_id':build,'category':question['category'],
                             'status':'completed' if valid else 'failed','value_correct':bool(value_correct),'gold_source_cited':bool(cited),
                             'provided_source_delivered':delivered,'answer_receipt_sha256':t.receipt(answer),'latency_ms':(time.perf_counter_ns()-started)/1e6,
                             'scope':'narrow_source_derived_atomic_value_and_citation_check_not_native_qa_or_full_entailment'}
                        t.append('outcomes.jsonl',row);outcomes.append(row)
                    t.flush_metrics()
    summary={'episodes':len(outcomes),'value_correct':sum(r['value_correct'] for r in outcomes),'valid':sum(r['status']=='completed' for r in outcomes),
             'scope':'acceptance diagnostic; text and graph evidence are not token/information matched; no causal graph benefit claim','trace_audit':audit(directory)}
    (directory/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['index','query']);p.add_argument('--storage',type=Path,required=True);p.add_argument('--run-id',required=True);p.add_argument('--build');p.add_argument('--max-facts',type=int,choices=[10,20],default=20);p.add_argument('--index-max-output-tokens',type=int,choices=[4096,8192],default=4096);p.add_argument('--evidence-mode',choices=['quote','span_ids'],default='quote');p.add_argument('--credential-file',type=Path,required=True);a=p.parse_args()
    INDEX_MAX_OUTPUT_TOKENS=a.index_max_output_tokens
    INDEX_MAX_FACTS=a.max_facts
    for tool in [EXTRACT,EVIDENCE_EXTRACT]:tool['function']['parameters']['properties']['facts']['maxItems']=INDEX_MAX_FACTS
    if a.action=='index':create_index(a.storage,a.run_id,a.credential_file,a.evidence_mode)
    else:
        if not a.build:p.error('--build required for query')
        query_existing(a.storage,a.run_id,a.build,a.credential_file)
