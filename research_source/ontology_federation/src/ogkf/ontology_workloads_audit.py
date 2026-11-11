"""Offline receipt reconstruction and audience casebook for ontology workloads."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path
import statistics

from .ontology_workloads import (MODEL, VERSION, encoded, packet, read_rows, score, sha, trace_audit)


def load(path):return json.loads(path.read_text())


def terminal_cause(row):
    for stage in ('intent','answer'):
        item=row.get(stage)
        if item and item['response']['http_status']!=200:
            return 'provider_http_'+str(item['response']['http_status'])
    return row['status']


def format_tolerant_target(row):
    """Secondary posthoc diagnostic; the frozen strict scorer stays unchanged.

    A literal percent suffix in an explicitly percent-valued question is a
    rendering issue. This normalization does not repair raw outputs or citations.
    """
    item=row.get('answer')
    if not item or item['schema_error']:return False
    value=item['value']
    if not isinstance(value,dict):return False
    normalized=dict(value)
    if row['case'].get('expected_unit')=='percent' and isinstance(normalized.get('value'),str):
        normalized['value']=normalized['value'].strip().removesuffix('%').strip()
    return score(row['case'],normalized,[],[])['atomic_target_match'] if not row['case']['required_groups'] else numeric_only(row['case'],normalized)


def numeric_only(case,value):
    # Empty groups avoid presenting this scalar check as evidence support.
    return score({**case,'required_groups':[]},value,[],[])['atomic_target_match']


def scalar_only(row):
    """Descriptive value diagnostic, deliberately excludes operand/citation tests."""
    item=row.get('answer')
    if not item or item['schema_error'] or not isinstance(item['value'],dict):return False
    value=dict(item['value']);case=dict(row['case'])
    if case.get('expected_unit')=='percent' and isinstance(value.get('value'),str):
        value['value']=value['value'].strip().removesuffix('%').strip()
    case.pop('numerator',None);case.pop('denominator',None)
    return numeric_only(case,value)


def audit_run(directory):
    manifest=load(directory/'manifest.json');data=load(directory/'dataset.private.json')
    assert sha(data)==manifest['input_sha256']
    assert hashlib.sha256((directory/'source-code.py').read_bytes()).hexdigest()==manifest['code_sha256']
    ledger=trace_audit(directory)
    assert ledger['valid'],ledger
    receipts={p.stem:load(p) for p in (directory/'receipts').glob('*.json')}
    spans=read_rows(directory/'spans.jsonl')
    http=[s for s in spans if s['component']=='mara']
    provider=[]
    for span in http:
        receipt=receipts[span['receipt_sha256']]
        assert sha(receipt)==span['receipt_sha256']
        assert receipt['request_sha256']==sha({'path':receipt['path'],'body':receipt['request']})
        provider.append(receipt)
    result={'protocol':manifest['protocol'],'trace_audit':ledger,'http_attempts':len(provider),
            'native_bolt_reads_including_qualification':sum(s['component']=='native_read' for s in spans),
            'http_statuses':dict(Counter(str(r['http_status']) for r in provider)),
            'known_input_tokens':sum(r['usage'].get('prompt_tokens') or 0 for r in provider),
            'known_output_tokens':sum(r['usage'].get('completion_tokens') or 0 for r in provider),
            'missing_usage_attempts':sum(not all(isinstance(r['usage'].get(k),int) for k in ('prompt_tokens','completion_tokens')) for r in provider),
            'missing_model_usage_attempts':sum(r['path']=='/chat/completions' and not all(isinstance(r['usage'].get(k),int) for k in ('prompt_tokens','completion_tokens')) for r in provider),
            'actual_billed_usd':None,'semantic_appropriateness':None,'independent_human_reviews':0,
            'returned_model_ids':dict(Counter(r['response'].get('model','not_returned') for r in provider if isinstance(r['response'],dict))),
            'unknown_transport_outcomes':len(list((directory/'http-state').glob('*.pending'))) if (directory/'http-state').exists() else 0}
    db=read_rows(directory/'db-outcomes.jsonl')
    if manifest['kind']=='db':
        assignments=load(directory/'assignments.json')
        assert len(assignments)==len(db)==252
        expected={}
        for row in db:
            assert sha(row['records'])==row['records_sha256']
            expected.setdefault(row['request'],row['records_sha256'])
            assert expected[row['request']]==row['records_sha256']
            assert row['bolt_transactions']==len(row['branches'])
        result.update(db_logical_requests=len(db),timed_logical_requests=sum(r['phase']=='timed' for r in db),
                      bolt_transactions=sum(r['bolt_transactions'] for r in db),matched_result_parity=True,
                      timed_native_work=sum(r['bolt_transactions'] for r in db if r['phase']=='timed'))
    index=read_rows(directory/'index-outcomes.jsonl');outcomes=read_rows(directory/'outcomes.jsonl')
    if manifest['kind']=='mara':
        assignments=load(directory/'assignments.json')
        result.update(index_assignments=len(assignments['index']),index_terminal=len(index),
                      agent_assignments=len(assignments['agent']),agent_terminal=len(outcomes))
        assert len({r['episode_id'] for r in outcomes})==len(outcomes)
        assert {r['episode_id'] for r in outcomes}<={r['episode_id'] for r in assignments['agent']}
        jobs={r['episode_id']:r for r in assignments['agent']}
        assert all(all(row[k]==jobs[row['episode_id']][k] for k in jobs[row['episode_id']]) for row in outcomes)
        assert len({(r['repeat'],r['arm'],r['cik']) for r in index})==len(index)
        request_bodies={r['request_sha256']:r for r in provider}
        for row in index:
            response=row['result']['response'];receipt=receipts[response['receipt_sha256']]
            assert receipt['request']==row['result']['request']
            assert receipt['response']==response['response']
            assert sha(sorted(row['records'],key=lambda r:r['id']))==row['native_readback_sha256']
            for record in row['records']:
                source=next(s for s in data['sources'] if s['id']==record['source_id'])
                for witness in record['support_spans']:
                    assert source['text'][witness['start']:witness['end']]==witness['text']
        packet_hashes=defaultdict(set)
        for row in outcomes:
            assert row['model']==MODEL
            for stage in ('intent','answer'):
                if not row.get(stage):continue
                item=row[stage];receipt=receipts[item['response']['receipt_sha256']]
                assert item['request']==receipt['request'] and item['response']['response']==receipt['response']
            if row.get('packet'):
                read=receipts[row['graph_receipt_sha256']]
                assert packet(read['records'])==row['packet']
                assert sha(row['packet'])==row['packet_sha256']
                messages=row['answer']['request']['messages']
                assert messages[:-2]==row['intent']['request']['messages']
                assert row['answer']['request']['model']==row['intent']['request']['model']==MODEL
                assert row['answer']['request']['temperature']==row['intent']['request']['temperature']==0
                assert json.loads(messages[-1]['content'])==row['packet']
                assert messages[-1]['tool_call_id']==messages[-2]['tool_calls'][0]['id']
                assert messages[-2]==row['intent']['response']['response']['choices'][0]['message']
                if row['workload']=='fixed_evidence':
                    packet_hashes[row['case_id']].add(row['packet_sha256'])
                    assert read['arm']=='parallel' and read['bolt_transactions']==3
                value=row['answer']['value'] if not row['answer']['schema_error'] else None
                assert score(row['case'],value,read['records'],data['facts'])==row['score']
        assert all(len(values)==1 for values in packet_hashes.values())
        result['fixed_evidence_packet_parity']=True
        result['completed_all_assigned']=len(index)==18 and len(outcomes)==90
    terminal=load(directory/'terminal.json') if (directory/'terminal.json').exists() else {}
    result['campaign_status']=terminal.get('status','running')
    result['owned_runtime_teardown_success']=(terminal.get('runtime_final') or {}).get('teardown_success')
    result['receipt_reconstruction_valid']=True
    if (directory/'tempo-readback.json').exists():
        sink=load(directory/'tempo-readback.json')
        result['tempo']={k:sink[k] for k in ('total_spans','collector_acknowledged','trace_count','fully_read_back','all_spans_read_back')}
    if (directory/'observability-incident.json').exists():
        result['observability_incident']=load(directory/'observability-incident.json')
    for filename,key in [('observability-recovery.json','observability_recovery'),('canonical-span-repair.json','canonical_span_repair')]:
        if (directory/filename).exists():result[key]=load(directory/filename)
    result['manifest_sha256']=sha(manifest)
    result['input_sha256']=manifest['input_sha256']
    result['code_sha256']=manifest['code_sha256']
    return result


def plan_totals(plan):
    if not plan:return {'db_hits':None,'page_cache_hits':None,'page_cache_misses':None}
    nodes=[]
    def walk(p):
        nodes.append(p)
        for c in p.get('children',[]):walk(c)
    walk(plan)
    return {name:sum(p.get(key,0) for p in nodes) if any(key in p for p in nodes) else None
            for name,key in [('db_hits','dbHits'),('page_cache_hits','pageCacheHits'),('page_cache_misses','pageCacheMisses')]}


def selected_subjects(row,directory):
    selected=row['intent']['value']
    if not isinstance(selected,dict):return []
    native=load(directory/'receipts'/f"{row['catalog_receipt_sha256']}.json")
    return sorted({r['subject'] for r in native['records'] if r['entity_key']==selected.get('entity_key')})


def audit_replay(directory,source):
    manifest=load(directory/'manifest.json');terminal=load(directory/'terminal.json')
    assert manifest['protocol']=='ontology-workloads-write-replay-v1'
    assert Path(manifest['source_run']).resolve()==source.resolve()
    for name in ('ontology_workloads_replay.py','ontology_workloads.py'):
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==manifest[name+'_sha256']
    integrity=trace_audit(directory);assert integrity['valid']
    spans=read_rows(directory/'spans.jsonl')
    assert not any(s['component']=='mara' for s in spans)
    jobs={r['episode_id']:r for r in read_rows(source/'index-outcomes.jsonl')}
    writes=[s for s in spans if s['component']=='index_write'];assert len(writes)==18
    for span in writes:
        actual=load(directory/'receipts'/f"{span['receipt_sha256']}.json")
        proposed=jobs[span['episode_id']]['records']
        assert [json.loads(r['payload']) for r in actual['parameters']['rows']]==proposed
        assert actual['record_count']==len(proposed) and actual['native_transactions']==1
    for span in (s for s in spans if s['component']=='graph_read'):
        actual=load(directory/'receipts'/f"{span['receipt_sha256']}.json")
        assert sha(actual['records'])==jobs[span['episode_id']]['native_readback_sha256']
    readback=load(directory/'tempo-readback.json') if (directory/'tempo-readback.json').exists() else {}
    return {'protocol':manifest['protocol'],'http_requests':0,'replayed_index_jobs':18,'observed_write_transactions':18,
            'write_payload_and_native_readback_parity':True,'trace_audit':integrity,
            'original_write_timing_available':False,'write_timings_are':'replay measurements only',
            'owned_runtime_teardown_success':terminal['runtime_final']['teardown_success'],
            'tempo':{k:v for k,v in readback.items() if k!='traces'}}


def summarize(db_dir,mara_dir,replay_dir=None):
    audits={'db':audit_run(db_dir),'mara':audit_run(mara_dir)}
    if replay_dir is not None:audits['write_replay']=audit_replay(replay_dir,mara_dir)
    db=read_rows(db_dir/'db-outcomes.jsonl');outcomes=read_rows(mara_dir/'outcomes.jsonl');index=read_rows(mara_dir/'index-outcomes.jsonl')
    groups=defaultdict(list)
    for row in db:
        if row['phase']=='timed':groups[(row['request'],row['arm'])].append(row)
    db_table=[]
    for (request,arm),rows in sorted(groups.items()):
        values=sorted(r['query_wall_ms'] for r in rows)
        plans=[r for r in db if r['phase']=='PROFILE' and r['request']==request and r['arm']==arm]
        totals=[plan_totals(b['plan']) for r in plans for b in r['branches']]
        db_table.append({'request':request,'arm':arm,'n':len(rows),'p50_ms':statistics.median(values),'min_ms':min(values),'max_ms':max(values),
                         'mean_overlap_ms':statistics.mean(r['pairwise_branch_overlap_ms'] for r in rows),
                         'bolt_transactions_per_request':rows[0]['bolt_transactions'],'fact_count':len(rows[0]['records']),
                         'profile_db_hits':sum(t['db_hits'] for t in totals) if all(t['db_hits'] is not None for t in totals) else None,
                         'profile_page_cache_hits':sum(t['page_cache_hits'] for t in totals) if all(t['page_cache_hits'] is not None for t in totals) else None,
                         'profile_page_cache_misses':sum(t['page_cache_misses'] for t in totals) if all(t['page_cache_misses'] is not None for t in totals) else None})
    index_table=[]
    for row in index:
        records=row['records']
        index_table.append({'issuer':row['cik'],'arm':row['arm'],'repeat':row['repeat'],'fact_proposals':len(records),
                            'contextual_subjects':len({r['entity_key'] for r in records}),
                            'attribute_strings':len({r['attribute'] for r in records}),
                            'concept_keys':dict(Counter(r['concept_key'] for r in records)),
                            'schema_error':row['result']['schema_error'],'independent_semantic_precision':None})
    groups=defaultdict(list)
    for row in outcomes:groups[(row['workload'],row['index_profile'],row['query_profile'])].append(row)
    model_table=[]
    for key,rows in sorted(groups.items()):
        positives=[r for r in rows if r['case']['expected'] is not None]
        negatives=[r for r in rows if r['case']['expected'] is None]
        model_table.append({'workload':key[0],'index_profile':key[1],'query_profile':key[2],'assigned':len(rows),
                            'terminal_causes':dict(Counter(terminal_cause(r) for r in rows)),
                            'atomic_target_matches':sum(r['score']['atomic_target_match'] for r in rows),
                            'secondary_format_tolerant_matches':sum(format_tolerant_target(r) for r in rows),
                            'secondary_scalar_only_matches':sum(scalar_only(r) for r in rows),
                            'positive_atomic_matches':sum(r['score']['atomic_target_match'] for r in positives),'positive_n':len(positives),
                            'insufficient_abstentions':sum(r['score']['atomic_target_match'] for r in negatives),'insufficient_n':len(negatives),
                            'positive_locator_coverage':sum(r['score']['citation_locator_coverage'] for r in positives),
                            'independent_semantic_appropriateness':None})
    case_table=[]
    grouped=defaultdict(list)
    for row in outcomes:grouped[(row['case_id'],row['workload'],row['index_profile'],row['query_profile'])].append(row)
    for key,rows in sorted(grouped.items()):
        answers=[r['answer']['value'] for r in rows if r.get('answer') and not r['answer']['schema_error']]
        case_table.append({'case':key[0],'workload':key[1],'index_profile':key[2],'query_profile':key[3],
                           'assigned':len(rows),'strict_scalar_matches':sum(r['score']['atomic_target_match'] for r in rows),
                           'secondary_format_tolerant_matches':sum(format_tolerant_target(r) for r in rows),
                           'secondary_scalar_only_matches':sum(scalar_only(r) for r in rows),
                           'observed_scalar_abstention_pairs':sorted({json.dumps([a['value'],a['abstain']],ensure_ascii=False) for a in answers}),
                           'selected_subject_surfaces':sorted({s for r in rows for s in selected_subjects(r,mara_dir)}),
                           'terminal_causes':dict(Counter(terminal_cause(r) for r in rows))})
    rationale={'case':'wynn-board-premise-unavailable','observations':{},'independently_adjudicated':False}
    for arm in ('G','GX','FX'):
        selected=[r for r in outcomes if r['workload']=='fixed_evidence' and r['case_id']==rationale['case'] and r['query_profile']==arm]
        rationale['observations'][arm]=[{'episode_id':r['episode_id'],'selected_subjects':selected_subjects(r,mara_dir),
                                        'answer':r['answer']['value'] if r.get('answer') else None} for r in selected]
    fx=rationale['observations']['FX']
    individual_test=bool(fx) and all(isinstance(r['answer'],dict) and 'CFO individually' in r['answer'].get('explanation','') for r in fx)
    rationale['fx_individual_authority_rationale_in_all_repeats']=individual_test
    rationale['interpretation']='Matched action does not certify the intended reasoning. Question wording and rationale require independent review.'
    if individual_test:
        rationale['interpretation']+=' FX explanations introduce CFO-individual decision authority; the preparer target uses the requested body. This is a different stated reason, not independent evidence of semantic improvement.'
    result={'audits':audits,'db':db_table,'index':index_table,'model':model_table,'by_case':case_table,
            'rationale_inspection':rationale,
            'limitations':['Three exposed issuer families and six constructed development questions; repeats are not independent samples.',
                           'FIBO package changes definition text and identifiers; it is not a pure label or size test.',
                           'Reference packages differ in text length; identical evidence and output caps do not mean identical total input tokens.',
                           'G to GX changes application context guidance, not FIBO.',
                           'Model extraction uses a 24-observation budget per issuer; missing evidence may reflect compression.',
                           'KG topology is fixed within each semantic comparison; execution topology is a separate matched-result study.',
                           'One application facade request has independent read transactions, not a global ACID snapshot or native Composite.',
                           'Small warm shared-host reads do not measure saturation, Python GIL causality, cache tuning or enterprise scaling.',
                           'Exact source locators and numeric targets are mechanical diagnostics, not semantic entailment or human quality.',
                           'Strict scores remain frozen. A separately labelled posthoc percent-suffix diagnostic reports format-tolerant scalar matching without repairing outputs.',
                           'No independent mapping or answer adjudication; semantic accuracy remains unknown.']}
    result['limitations'].append('Original index writes were inside the indexing root but lacked separate write spans/timers. The zero-model replay traces actual writes separately; it does not backfill original clocks or setup/write costs.')
    return result


def figure(result, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'GX':'#2b6cb0','FX':'#d97706','G':'#55736b'}
    fig,axes=plt.subplots(1,3,figsize=(16,5.2),layout='constrained')
    issuers=['0001097149','0000002488','0001174922']
    names=['Align','AMD','Wynn']
    for arm,shift in [('GX',-.13),('FX',.13)]:
        rows=[r for r in result['index'] if r['arm']==arm]
        axes[0].scatter([issuers.index(r['issuer'])+shift+(r['repeat']-1)*.025 for r in rows],
                        [r['fact_proposals'] for r in rows],color=colors[arm],label=arm,s=50,alpha=.8)
    axes[0].set(xticks=range(3),xticklabels=names,ylabel='Bound proposals per extraction',title='Formation: count is not quality',ylim=(-1,26))
    axes[0].legend();axes[0].grid(axis='y',alpha=.2)
    query=[next(r for r in result['by_case'] if r['case']=='wynn-board-premise-unavailable' and r['workload']=='fixed_evidence' and r['query_profile']==arm) for arm in ('G','GX','FX')]
    for x,row in enumerate(query):
        axes[1].bar(x,row['strict_scalar_matches'],color=colors[row['query_profile']])
        axes[1].text(x,row['strict_scalar_matches']+.08,f"{row['strict_scalar_matches']}/{row['assigned']}",ha='center')
    axes[1].set(xticks=range(len(query)),xticklabels=[r['query_profile'] for r in query],ylabel='Observed Board-start abstentions',
                title='Reception: matched action ≠ matched reason',ylim=(0,3.6))
    note='FX adds a CFO-individual-authority test' if result['rationale_inspection']['fx_individual_authority_rationale_in_all_repeats'] else 'Inspect stated reasons in the full casebook'
    axes[1].text(.5,.98,note+'\nSelected exposed case; no semantic accuracy claim',transform=axes[1].transAxes,ha='center',va='top',fontsize=8.5)
    axes[1].grid(axis='y',alpha=.2)
    cross=['align-cross','amd-cross','wynn-all']
    for arm,shift,color in [('single',-.16,'#55736b'),('sequential',0,'#9c6471'),('parallel',.16,'#2b6cb0')]:
        rows=[next(r for r in result['db'] if r['request']==case and r['arm']==arm) for case in cross]
        axes[2].scatter([i+shift for i in range(3)],[r['p50_ms'] for r in rows],color=color,label=arm,s=50)
    axes[2].set(xticks=range(3),xticklabels=['Align','AMD','Wynn'],ylabel='Median native read wall time (ms)',title='Execution: matched two-target reads')
    axes[2].legend();axes[2].grid(axis='y',alpha=.2)
    fig.suptitle('Exposed development cohort · 3 issuer families · 3 repeats · no independent semantic accuracy claim',fontsize=12)
    fig.savefig(output/'overview.png',dpi=170)
    fig.savefig(output/'overview.svg')
    svg=output/'overview.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig)


def report(db_dir,mara_dir,output,replay_dir=None):
    result=summarize(db_dir,mara_dir,replay_dir)
    output.mkdir(parents=True,exist_ok=True)
    (output/'summary.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    figure(result,output)
    escape=lambda v:html.escape(str(v))
    def table(rows):
        if not rows:return '<p>No terminal observations.</p>'
        keys=list(rows[0])
        return '<table><tr>'+''.join('<th>'+escape(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+escape(r[k])+'</td>' for k in keys)+'</tr>' for r in rows)+'</table>'
    body='<h1>Ontology formation → agent reception</h1><p>Executed development evidence; unchanged answers and failures are results. No BM25 superiority or independent semantic accuracy claim.</p>'
    body+='<p><strong>Interpretation supplement:</strong> <a href="MECHANISM_INTERPRETATION.md">Actual plans, fixed hops and indexing failures</a> distinguishes empty-target overhead, native work versus scheduling, literal numeric placeholders, lost segment scope and source-span/tuple attribution. Read it before interpreting proposal counts or timing differences as quality or scaling gains.</p>'
    body+='<img src="overview.png" alt="Formation, reception and matched native query measurements" style="width:100%">'
    body+='<h2>Receipt and completeness audit</h2><pre>'+escape(json.dumps(result['audits'],indent=2))+'</pre>'
    body+='<p>The original paid cohort preserves extraction-through-answer readback, but its writes lacked separate child spans. The separately identified zero-model replay, when present above, checks the frozen write payloads and resulting graph packets with actual write receipts. Its clocks are replay measurements; original clocks remain unavailable.</p>'
    body+='<h2>1. Formation: same source, different reference package</h2><p>GX: competent generic + local rules. FX: pinned FIBO definitions + the same rules. No downstream question or target entered indexing. Counts describe proposals, not quality.</p>'+table(result['index'])
    for row in read_rows(mara_dir/'index-outcomes.jsonl'):
        body+='<details><summary>'+escape(row['episode_id']+' '+row['cik']+' '+row['arm']+' repeat '+str(row['repeat']))+'</summary><pre>'+escape(json.dumps({'native_fact_proposals':row['records'],'rejected':row['rejected'],
                  'http_receipt_sha256':row['result']['response']['receipt_sha256'],'schema_error':row['result']['schema_error']},indent=2,ensure_ascii=False))+'</pre></details>'
    body+='<h2>2. Reception: actual two-round MARA episodes</h2><p>Each episode selects an entity, retrieves native graph evidence and generates an answer. Fixed-evidence query arms receive byte-identical assertion/witness packets; their system reference packages vary. Index-derived answers use constant GX query guidance.</p>'+table(result['model'])
    body+='<p><strong>A matched abstention is not proof of the intended reasoning.</strong> '+escape(result['rationale_inspection']['interpretation'])+' The middle figure is a selected illustration; all six cases remain below.</p><details><summary>Actual Board-start entity selections and rationales</summary><pre>'+escape(json.dumps(result['rationale_inspection'],indent=2,ensure_ascii=False))+'</pre></details>'
    body+='<h2>3. Query execution: same complete records</h2><p>Independent read autocommits, one DBMS, 2 CPU/2 GiB appliance, inner parallelism at most 2. Native execution/consumption timers exclude receipt fsync. PROFILE is separate from timed samples. Page-cache counters are Neo4j operator counters, not OS disk faults.</p>'+table(result['db'])
    for row in read_rows(db_dir/'db-outcomes.jsonl'):
        if row['phase']!='PROFILE':continue
        body+='<details><summary>Actual PROFILE: '+escape(row['request']+' / '+row['arm'])+'</summary>'
        for branch in row['branches']:
            body+='<h3>'+escape(branch['database'])+'</h3><pre>'+escape((branch['plan'] or {}).get('args',{}).get('string-representation',json.dumps(branch['plan'],indent=2)))+'</pre>'
        body+='</details>'
    body+='<h2>Every case and condition</h2><p>Strict scalar matches retain the original scorer. The secondary format-tolerant diagnostic only accepts an explicit percent suffix for percent questions; it was added after observing output formatting and is not confirmatory. HTTP failures remain distinct from model contract failures.</p>'+table(result['by_case'])
    body+='<h2>Complete answer casebook</h2>'
    roots={s['episode_id']:s['trace_id'] for s in read_rows(mara_dir/'spans.jsonl') if s['component']=='agent_episode'}
    for row in read_rows(mara_dir/'outcomes.jsonl'):
        answer=row['answer']['value'] if row.get('answer') else None
        case={'case':row['case'],'condition':{k:row[k] for k in ('workload','index_profile','query_profile','repeat','namespace')},
              'entity_selection':row['intent']['value'],'delivered_fact_count':row['delivered_fact_count'],'answer':answer,
              'selected_subject_surfaces':selected_subjects(row,mara_dir),
              'status':row['status'],'mechanical_diagnostics':row['score'],'trace_id':roots[row['episode_id']],
              'packet_sha256':row['packet_sha256'],'intent_http_receipt':row['intent']['response']['receipt_sha256'],
              'delivered_evidence':row.get('packet'),
              'answer_http_receipt':row['answer']['response']['receipt_sha256'] if row.get('answer') else None}
        body+='<details><summary>'+escape(row['episode_id']+' '+row['case_id']+' '+row['index_profile']+'/'+row['query_profile']+' '+row['status'])+'</summary><pre>'+escape(json.dumps(case,indent=2,ensure_ascii=False))+'</pre></details>'
    body+='<h2>Interpretation boundaries</h2><ul>'+''.join('<li>'+escape(s)+'</li>' for s in result['limitations'])+'</ul>'
    document='<!doctype html><meta charset="utf-8"><title>Ontology workload results</title><style>body{font:16px system-ui;margin:36px;max-width:1500px;color:#142230}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{padding:8px;border:1px solid #ccd7e0;text-align:left}th{background:#edf3f7}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{margin:10px 0;padding:10px;background:#f4f7fa}h2{margin-top:36px}</style>'+body
    (output/'report.html').write_text(document)
    queue=[{'episode_id':r['episode_id'],'case_id':r['case_id'],'graph_receipt_sha256':r.get('graph_receipt_sha256'),
            'response_receipt_sha256':r['answer']['response']['receipt_sha256'] if r.get('answer') else None,
            'review_status':'pending_independent_human_review','reviewer':None,'semantic_appropriateness':None}
           for r in read_rows(mara_dir/'outcomes.jsonl')]
    (output/'human-review-queue.json').write_text(json.dumps(queue,indent=2)+'\n')
    index_queue=[{'episode_id':r['episode_id'],'cik':r['cik'],'arm':r['arm'],
                  'response_receipt_sha256':r['result']['response']['receipt_sha256'],
                  'admitted_fact_ids':[f['id'] for f in r['records']],
                  'review_status':'pending_independent_human_review','semantic_precision':None,'semantic_recall':None}
                 for r in read_rows(mara_dir/'index-outcomes.jsonl')]
    (output/'index-review-queue.json').write_text(json.dumps(index_queue,indent=2)+'\n')
    print(json.dumps(result['audits'],indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--db',type=Path,required=True);p.add_argument('--mara',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--replay',type=Path)
    a=p.parse_args();report(a.db,a.mara,a.output,a.replay)

if __name__=='__main__':main()
