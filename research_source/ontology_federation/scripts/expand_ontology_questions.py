#!/usr/bin/env python3
"""Versioned native-question extension. No private labels or answers enter requests."""
from pathlib import Path
import argparse,collections,hashlib,json,os,random,re,sys,time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from ogkf import ontology_workloads as ow
from ogkf.environment import load_project_environment
from ogkf.local_runtime import local_dozer
import shutil
import jsonschema
import logging
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

SEED=20261109
INDEX_SYSTEM='Extract qualified financial assertions from every supplied source passage, without seeing downstream questions. Preserve different subject/attribute/period/scope/unit/basis/modality and distinguish membership, reporting and decision authority. Select the most relevant concept_key or unmapped, and cite immutable source span IDs. Extract at most 24 salient observations across all passages; never invent missing facts.\nSemantic reference package:\n'
QUERY_SYSTEM='Answer the public financial question using the attached graph evidence. Select a contextual starting entity. Request evidence once, then submit a result; abstain on unsupported premises. Do not infer facts from ontology definitions. For ratios cite premises and operands; return numerator and denominator.\nSemantic reference package:\n'

def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def rows(p):return [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []

class ExtensionMara(ow.Mara):
 def _request(self,*args,**kwargs):
  r=self.telemetry.directory.parent;lock=Path(os.environ.get('MARA_ADMISSION_LOCK',str(r/'.mara-global.lock')))
  if lock.resolve()!=(r/'.mara-global.lock').resolve() or lock.with_suffix('.pending').exists():raise RuntimeError('shared gateway admission guard')
  spans=[s for s in rows(self.telemetry.directory/'spans.jsonl') if s['component']=='mara']
  wave_limit=int(os.environ.get('OGKF_WAVE_MAX_HTTP','300'));campaign_limit=int(os.environ.get('OGKF_MAX_CAMPAIGN_HTTP','2400'));token_limit=int(os.environ.get('OGKF_MAX_CAMPAIGN_KNOWN_TOKENS','12000000'))
  if not (1<=wave_limit<=300 and 1<=campaign_limit<=2400 and 1<=token_limit<=12_000_000):raise RuntimeError('invalid bounded extension budget')
  known=sum((s['measurements'].get(k) or 0) for s in spans for k in ('input_tokens','output_tokens'))
  if len(spans)>=wave_limit or known>=3_000_000:raise RuntimeError('extension budget exhausted')
  if len(spans)+int(os.environ.get('OGKF_COMPLETED_HTTP_ATTEMPTS','0'))>=campaign_limit or known+int(os.environ.get('OGKF_COMPLETED_KNOWN_TOKENS','0'))>=token_limit:raise RuntimeError('global extension budget exhausted')
  v=super()._request(*args,**kwargs)
  if v.get('unresolved'):raise RuntimeError('unresolved transport outcome; no retry')
  if v.get('http_status') in {401,403,404,410}:raise RuntimeError('provider authentication or endpoint unavailable; halt without retries')
  return v

def prepare(out):
 out.mkdir(parents=True,exist_ok=False)
 r=ROOT/'reports/finder_candidates_v1';f=json.loads((r/'expanded_frame/frame.private.json').read_text());docs={d['document_id']:d for d in f['documents']};cases={x['query']['query_id']:x for x in f['cases']}
 candidates=rows(r/'bound_review.private.jsonl');selected=[];used=set();rng=random.Random(SEED)
 for category in sorted({q['category'] for q in candidates}):
  pool=[q for q in candidates if q['category']==category and q['query_id'] in cases and len(q['verified_source_issuer_ids'])==1 and all(x in docs for x in q['source_ids']) and sum(len(docs[x]['text'].encode()) for x in q['source_ids'])<=16000]
  rng.shuffle(pool);pick=[]
  for q in sorted(pool,key=lambda q:q['native_type_raw']=='None'):
   if q['shared_reference_component_id'] not in used:
    pick.append(q);used.add(q['shared_reference_component_id'])
   if len(pick)==3:break
  if len(pick)<3:raise RuntimeError('insufficient distinct source components for '+category)
  selected+=pick
 public=[];private=[];sources={}
 for q in selected:
  sid=q['query_id'];case=cases[sid];cik=str(q['verified_source_issuer_ids'][0]).removeprefix('cik:').zfill(10)
  for ident in q['source_ids']:
   d=docs[ident];context={'finance_reporting':'performance','legal_governance':'organization','business_risk':'risk'}.get(d['domain_id'])
   if context is None:raise ValueError('unknown source domain '+d['domain_id'])
   # Retain original reference text losslessly; source ownership is not subject identity.
   sources[ident]={'id':ident,'cik':cik,'collection':context,'accession':d['source_uri'],'text':d['text']}
  public.append({'id':sid,'question':q['question'],'cik':cik,'source_ids':q['source_ids']})
  private.append({**q,'original_answer':case['original_answer'],'original_gold_hash':case['original_gold_hash']})
 manifest={'protocol':'ontology-native-question-extension-v1','seed':SEED,'questions':len(public),'category_counts':dict(collections.Counter(q['category'] for q in private)),'native_type_counts':dict(collections.Counter(q['native_type_raw'] for q in private)),'distinct_source_components':len(used),'unique_source_documents':len(sources),'question_repeats':1,'model':'gpt-oss-120b','index_profiles':['GX','FX'],'query_fixed_source_profiles':['G','GX','FX'],'index_to_answer_query_profile':'GX','maximum_campaign_http_attempts':300,'known_token_stop':3000000,'planned_inference_requests':len(public)*12,'max_logical_concurrency':1,'max_db_workers':2,'max_source_bytes_per_question':16000,'semantic_accuracy':None,'exposure':'development: original questions and references were previously inventoried/reviewed; not heldout','fixed_evidence':'literal source-reference records, not independently adjudicated business assertions; different from historical 43-assertion manual view','causal_boundary':'compare profiles within workload; source view and extracted graph differ; not a federation placement treatment','native_gold_role':'evaluator_only','federation':'source-view single/parallel readback parity; application-level, independent transactions','ontology_coverage':'same frozen 9 terms for all categories; incomplete coverage retained as a limitation','inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [r/'bound_review.private.jsonl',r/'expanded_frame/frame.private.json',ow.PROFILE_PATH]},'profile_hash':ow.load_bundle(ow.PROFILE_PATH).profile_hash}
 manifest.update(source_selection='Original native reference documents selected per question; reference-conditioned diagnostic, not open-corpus retrieval evaluation',new_independent_semantic_gold=0,fixed_selection_limitation='literal source-document catalog differs from extracted entity catalog; compare arms only within each workload')
 dump(out/'questions.agent.json',public);dump(out/'questions.evaluator-only.json',private);dump(out/'sources.json',list(sources.values()));dump(out/'protocol.json',manifest)
 print(json.dumps(manifest),flush=True)

def literal_records(sources):
 return [{'id':'R'+ow.sha(s['id'])[:24],'entity_key':'D'+ow.sha(s['id'])[:24],'subject':'Source document '+s['id'],'attribute':'source_reference_text','value':'See immutable source span','period':'unknown','scope':'source_reference','unit':'unknown','basis':'unknown','modality':'source_text','concept_key':'unmapped','source_id':s['id'],'cik':s['cik'],'context':s['collection'],'collection':s['collection'],'support_spans':[{'source_id':s['id'],'start':0,'end':len(s['text']),'text':s['text']}],'admission':'literal_reference_not_business_fact'} for s in sources]

def infer(client,t,ep,messages,name,schema,cap):
 request=ow.body(messages,name,schema,cap);response=client.request(ep.removesuffix('-intent').removesuffix('-answer'),ep,request,max_attempts=1)
 value=ow.extract_message(response,name);error=None
 try:jsonschema.validate(value,schema)
 except jsonschema.ValidationError as e:error=e.validator
 return {'value':value,'schema_error':error,'response':response}

def episode(store,client,t,q,namespace,profile,workload,ep):
 with t.span('agent_episode',ep):
  before=store.read(ep,namespace,q['cik'],ow.CONTEXTS);records=before['records'];cat=ow.catalog(records)
  schema=ow.obj({'entity_key':{'type':'string','enum':[e['entity_key'] for e in cat]+['unresolved']},'attribute':{'type':'string'}})
  messages=[{'role':'system','content':QUERY_SYSTEM+json.dumps(ow.profile(profile),ensure_ascii=False)},{'role':'user','content':json.dumps({'question':q['question'],'issuer_cik':q['cik'],'entity_catalog':cat},ensure_ascii=False)}]
  intent=infer(client,t,ep+'-intent',messages,'fetch_attached_evidence',schema,1024);answer=None;evidence=None
  if not intent['schema_error']:
   actual=store.read(ep,namespace,q['cik'],ow.CONTEXTS,'parallel' if workload=='fixed_source' else 'single');assert actual['records']==records
   evidence=ow.packet(records);original=intent['response']['response']['choices'][0]['message'];tool=original['tool_calls'][0]
   answer=infer(client,t,ep+'-answer',messages+[original,{'role':'tool','tool_call_id':tool['id'],'content':ow.encoded(evidence).decode()}],'submit_results',ow.answer_schema(records),2048)
  outcome={'episode_id':ep,'question_id':q['id'],'workload':workload,'profile':profile,'namespace':namespace,'status':'intent_contract_failure' if intent['schema_error'] else 'answer_contract_failure' if answer['schema_error'] else 'completed','intent':intent,'answer':answer,'packet_sha256':ow.sha(evidence) if evidence else None,'records':len(records),'semantic_accuracy':None}
  t.append('outcomes.jsonl',outcome);t.flush_metrics();print(json.dumps({k:outcome[k] for k in ['episode_id','question_id','workload','profile','status','records']}),flush=True)

def run(config,directory,reuse=None):
 load_project_environment();protocol=json.loads((config/'protocol.json').read_text());questions=json.loads((config/'questions.agent.json').read_text());source_map={s['id']:s for s in json.loads((config/'sources.json').read_text())}
 if directory.exists():raise RuntimeError('fresh run directory required')
 directory.mkdir(parents=True,mode=0o700);t=ow.Telemetry(directory,directory.name);dump(directory/'protocol.json',protocol);(directory/'source-code.py').write_bytes(Path(__file__).read_bytes());client=ExtensionMara(t);bundle=ow.load_bundle(ow.PROFILE_PATH)
 if reuse:
  reused=[]
  for previous in (reuse/'http-state').glob('*.json'):
   result=json.loads(previous.read_text());digest=result['receipt_sha256'];receipt=reuse/'receipts'/(digest+'.json');assert hashlib.sha256(receipt.read_bytes()).hexdigest()==digest
   shutil.copyfile(previous,client.state/previous.name);shutil.copyfile(receipt,directory/'receipts'/receipt.name);reused.append({'request_id':previous.stem,'receipt_sha256':digest})
  dump(directory/'reused-http-receipts.json',{'source_run':str(reuse),'requests':reused,'source_spans':str(reuse/'spans.jsonl'),'new_inference_for_these_requests':0})
 state={'status':'running' ,'started_utc':ow.utc(),'completed_episodes':0};runtime=None
 try:
  with local_dozer() as (env,runtime):
   store=ow.Store(env,t)
   try:
    store.setup();dump(directory/'runtime.json',runtime)
    # All frozen source views pass actual readback/PROFILE/parity before any paid extraction.
    qualification=[]
    for q in questions:
     src=[source_map[s] for s in q['source_ids']];records=literal_records(src);ns='ref-'+q['id'];store.load(ns,records,src,partitioned=True,episode='write-'+q['id'])
     one=store.read('gate-'+q['id'],ns,q['cik'],ow.CONTEXTS,'single','PROFILE');many=store.read('gate-'+q['id'],ns,q['cik'],ow.CONTEXTS,'parallel','PROFILE');assert one['records']==many['records']==sorted(records,key=lambda r:r['id']);assert any(x.split('@')[0]=='NodeUniqueIndexSeek' for x in ow.operators(one['branches'][0]['plan']))
     qualification.append({'question_id':q['id'],'record_count':len(records),'single_multi_parity':True,'source_sha256':{s['id']:hashlib.sha256(s['text'].encode()).hexdigest() for s in src},'single_profile':one['branches'][0]['plan']})
    dump(directory/'qualification.json',qualification)
    # Randomized question order; one repeat favors breadth over repeated exposed questions.
    random.Random(SEED).shuffle(questions)
    for n,q in enumerate(questions):
     src=[source_map[s] for s in q['source_ids']];lookup=ow.source_spans(src);schema=ow.index_schema(lookup,bundle)
     profiles=['GX','FX'];random.Random(SEED+n).shuffle(profiles)
     for arm in profiles:
      ns='idx-'+q['id']+'-'+arm;ep=ns
      with t.span('index_episode',ep):
       result=infer(client,t,ep,[{'role':'system','content':INDEX_SYSTEM+json.dumps(ow.profile(arm),ensure_ascii=False)},{'role':'user','content':json.dumps({'issuer_cik':q['cik'],'source_spans':lookup},ensure_ascii=False)}],'submit_observations',schema,4096)
       admitted,rejected=(ow.decode_index(result['value'],schema,lookup,ns,arm,bundle) if not result['schema_error'] else ([],[]))
       t.append('index-outcomes.jsonl',{'question_id':q['id'],'profile':arm,'result':result,'admitted':admitted,'rejected':rejected})
       store.load(ns,admitted,src,episode=ep)
       readback=store.read(ep,ns,q['cik'],ow.CONTEXTS);assert len(readback['records'])==len(admitted)
      episode(store,client,t,q,ns,'GX','index_to_answer','xa-'+q['id']+'-'+arm);state['completed_episodes']+=1
     profiles=['G','GX','FX'];random.Random(SEED+100+n).shuffle(profiles)
     for arm in profiles:
      episode(store,client,t,q,'ref-'+q['id'],arm,'fixed_source','xf-'+q['id']+'-'+arm);state['completed_episodes']+=1
     dump(directory/'progress.json',state)
   finally:store.close()
  state['status']='completed'
 except BaseException as e:
  state.update(status='halted',exception_type=type(e).__name__);raise
 finally:
  t.flush_metrics();state.update(finished_utc=ow.utc(),runtime_final=runtime);dump(directory/'terminal.json',state);dump(directory/'trace-audit.json',ow.trace_audit(directory));print(json.dumps(state),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','run']);p.add_argument('--config',type=Path,required=True);p.add_argument('--directory',type=Path);p.add_argument('--reuse-http-run',type=Path);a=p.parse_args();prepare(a.config) if a.mode=='prepare' else run(a.config,a.directory,a.reuse_http_run)
