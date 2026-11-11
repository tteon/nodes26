"""Build a portable, source-backed briefing for a separate presentation agent."""
import argparse
from collections import Counter, defaultdict
import csv
import html
import json
from pathlib import Path
import re
import shutil
from statistics import mean
import subprocess
import zipfile

import pyarrow.parquet as pq

from .context_frame.compiler import file_hash
from .models import digest

NATIVE = '3a844b72-a5f9-41e1-870c-3b961ecf1a2c'
ORIGINAL = 'c7d13f90-2fcb-4be8-ba23-35fbb566563a'
CONTINUATION = '4c2fbce8-b105-4320-92cd-c8ebf7e92d5d'
SOURCES = [
 ('S01','reports/FINAL_REVIEW.md','current review'),
 ('S02','reports/DS01_RESULTS.md','native pilot results'),
 ('S03','reports/finder_design_review.md','historical design; not execution receipt'),
 ('S04','reports/DS01_IMPLEMENTATION.md','DS01 methods'),
 ('S05','reports/campaign.md','historical service/fixture implementation campaign'),
 ('S06','reports/phase0.md','historical mock checkpoint'),
 ('S07','reports/finder_candidates_v1/summary.json','candidate census checkpoint'),
 ('S08','reports/finder_candidates_v1/census.json','native category/reasoning counts'),
 ('S09','reports/final_review_v1/audit.json','frozen readiness checkpoint'),
 ('S10','reports/ds01_campaign/campaign.json','native campaign aggregate'),
 ('S11','reports/final_review_v1/cached-analysis-v3.json','cached exploratory paired analysis'),
 ('S12',f'reports/ds01_campaign/runs/{NATIVE}/retrieval_traces.parquet','latest native retrieval receipts'),
 ('S13',f'reports/ds01_campaign/runs/{NATIVE}/evaluation_results.rescored-v2.parquet','cached reference proxies'),
 ('S14',f'reports/ds01_campaign/runs/{NATIVE}/distribution_measurements.parquet','observed shift and drift'),
 ('S15',f'reports/ds01_campaign/runs/{NATIVE}/manifest.json','latest native frozen configuration'),
 ('S16',f'reports/ds01_campaign/runs/{ORIGINAL}/agent_evaluation.parquet','original E3 assignments'),
 ('S17',f'reports/ds01_campaign/runs/{CONTINUATION}/agent_evaluation.parquet','continuations of never-dispatched assignments'),
 ('S18','reports/campaign/a319bd85-6161-4530-979f-f440ceade456/summary.json','final real extraction smoke'),
 ('S19','reports/context_frame_v1/annotations/summary.json','previous 24-case silver frame'),
 ('S20','reports/finder_candidates_v1/expanded_frame/summary.json','expanded 329-case silver frame'),
 ('S21','reports/final_review_v1/verification.json','140-test historical checkpoint'),
 ('S22','reports/ds01_campaign/input.evaluator-only.json','frozen native source and reference lineage'),
 ('S23','reports/ds01_campaign/packing-failure-witness.json','older packing witness; not latest-v2 sample'),
 ('S24','reports/ds01_campaign/actual-routing-example.json','actual public plan and reuse receipt'),
 ('S25','contracts/metric_contract.json','metric definitions; not measured results'),
 ('S26',f'reports/ds01_campaign/runs/{NATIVE}/graph-capabilities.json','native DB capabilities/parity'),
 ('S27','reports/graph-capabilities.verified.json','earlier isolated DB capability probes'),
 ('S28',f'reports/ds01_campaign/runs/{NATIVE}/corpus_manifests.parquet','native corpus membership'),
 ('S29',f'reports/ds01_campaign/runs/{ORIGINAL}/llm_request_traces.parquet','original HTTP attempt trace lineage'),
 ('S30',f'reports/ds01_campaign/runs/{CONTINUATION}/llm_request_traces.parquet','continuation HTTP attempt trace lineage'),
]
ASSETS = [
 ('A01','reports/ds01_architecture','methods diagram','Public/private paths; application-level federation. Rebuild as editable diagram where possible.'),
 ('A02','reports/finder_candidates_v1/category_reasoning','actual census','Full native and source-bound populations have independent axes; do not equate sample sizes.'),
 ('A03','reports/ds01_campaign/strategy_quality','actual native proxy','2 questions × 3 seeds/cell. Provided-reference proxy, not semantic quality. All strategies tie.'),
 ('A04','reports/ds01_campaign/observed_shifts','actual measured features','Middle panel is aggregate relation-count JSD; D04 also provides target-entity neighborhood JSD. Do not relabel one as the other.'),
 ('A05','reports/campaign/fixture-comparison','historical synthetic software diagnostic','Synthetic structured claims, not native FinDER quality or causal evidence.'),
 ('A06','reports/verification/fixture-calibration','historical mock diagnostic','No independent human calibration; never use as research-quality finding.'),
]


def split_row(line):
    line = line.strip()
    if line.startswith('|'):
        line = line[1:]
    if line.endswith('|'):
        line = line[:-1]
    return [s.strip().replace(r'\|','|') for s in re.split(r'(?<!\\)\|',line)]


def extract_tables(text):
    lines = text.splitlines()
    output, heading, i = [], '', 0
    while i < len(lines):
        if lines[i].startswith('#'):
            heading = lines[i].lstrip('# ').strip()
        if i+1 < len(lines) and lines[i].strip().startswith('|'):
            headers, separator = split_row(lines[i]), split_row(lines[i+1])
            if len(headers) == len(separator) and all(re.fullmatch(r':?-{3,}:?',s) for s in separator):
                start = i
                rows = []
                i += 2
                while i < len(lines) and lines[i].strip().startswith('|'):
                    row = split_row(lines[i])
                    if len(row) != len(headers):
                        raise ValueError(f'table column mismatch at line {i+1}')
                    rows.append(row)
                    i += 1
                output.append(dict(heading=heading,line_start=start+1,line_end=i,
                                   headers=headers,rows=rows,original_markdown='\n'.join(lines[start:i])))
                continue
        i += 1
    return output


def read_payloads(path):
    return [json.loads(r['payload_json']) for r in pq.ParquetFile(path).read().to_pylist()]


def merge_continuations(original, continued):
    originals = {r['sample_id']: r for r in original}
    if len(originals) != len(original):
        raise ValueError('duplicate original assignment')
    effective = dict(originals)
    seen = set()
    for row in continued:
        parent = row.get('parent_sample_id')
        if parent not in originals or parent in seen:
            raise ValueError('unknown/duplicate continuation parent')
        if 'Unresolved MARA outcome' not in (originals[parent].get('error') or ''):
            raise ValueError('continuation is not a never-dispatched blocked assignment')
        if row['query_id'] != originals[parent]['query_id'] or row['kind'] != originals[parent]['kind']:
            raise ValueError('continuation changed assignment')
        effective[parent] = row
        seen.add(parent)
    return [effective[k] for k in sorted(effective)]


def json_write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def csv_write(path,headers,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream,lineterminator='\n')
        writer.writerow(headers)
        writer.writerows([[json.dumps(c,ensure_ascii=False) if isinstance(c,(list,dict)) else c for c in row] for row in rows])


def build(package, output):
    package, output = Path(package).resolve(), Path(output).resolve()
    if output.exists() or output.with_suffix('.zip').exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    sources = []
    for sid,rel,scope in SOURCES:
        src=package/rel
        target=Path('sources')/(sid+'--'+src.name)
        if not src.is_file():
            raise ValueError('missing source '+rel)
        (output/target).parent.mkdir(exist_ok=True)
        shutil.copyfile(src,output/target)
        sources.append(dict(source_id=sid,original_path=rel,bundle_path=str(target),sha256=file_hash(src),scope=scope))
    source_map={s['source_id']:output/s['bundle_path'] for s in sources}
    load=lambda sid:json.loads(source_map[sid].read_text())
    config_path=package/'configs/presentation_handoff/storyboard.v1.json'
    story=json.loads(config_path.read_text())
    if story['recommended_main_slides']!=len(story['slides']):
        raise ValueError('storyboard slide count mismatch')
    json_write(output/'storyboard.json',story)
    source_ids=set(source_map)
    tables=[]
    appendix=0

    def table(tid,title,headers,rows,scope,ids,locator=None):
        nonlocal appendix
        if not set(ids)<=source_ids:
            raise ValueError('table cites unknown source')
        item=dict(table_id=tid,title=title,headers=headers,rows=rows,scope=scope,source_ids=ids,
                  source_locator=locator,csv_path=f'tables/{tid}.csv',json_path=f'tables/{tid}.json',appendix_pages=[])
        csv_write(output/item['csv_path'],headers,rows)
        for offset in range(0,len(rows),8):
            appendix+=1
            item['appendix_pages'].append(dict(appendix_id=f'AP{appendix:02}',row_start=offset+1,row_end=min(offset+8,len(rows))))
        json_write(output/item['json_path'],item)
        tables.append(item)
    for source in sources:
        if source['original_path'].endswith('.md'):
            for item in extract_tables((output/source['bundle_path']).read_text()):
                tid=f'T{len([t for t in tables if t["table_id"].startswith("T")])+1:02}'
                table(tid,item['heading'],item['headers'],item['rows'],source['scope'],[source['source_id']],
                      dict(line_start=item['line_start'],line_end=item['line_end']))
    audit,campaign,census=load('S09'),load('S10'),load('S08')
    manifest=load('S15');cfg=manifest['config']
    metrics=read_payloads(source_map['S13'])
    shifts=read_payloads(source_map['S14'])
    lookup={r['condition_id']:r for r in shifts}
    traces=read_payloads(source_map['S12'])
    trace_by_sample={r['sample_id']:r for r in traces}
    if len(metrics)!=len(trace_by_sample) or {r['sample_id'] for r in metrics}!=set(trace_by_sample):
        raise ValueError('metric/retrieval sample lineage mismatch')
    chronology=[
      ['01','Phase0','offline/mock','one-document mock smoke and18-call ontology serialization','No real quality/capability conclusion','S06'],
      ['02','Service qualification + real extraction','live engineering','actual probes; final2/2 samples accepted,5 mechanically span-valid assertions','No independent fact gold;54-cell factorial not run','S05,S18'],
      ['03','Frozen identity/routing fixture','synthetic','400 memory episodes;40 native-DB fixture episodes','Not FinDER quality;V0 loss is intentional ablation','S05'],
      ['04','FinDER design and audit','design','native QA/reference contract;8/80-question proposed budgets','Those planned8/80-question blocks are not completed runs','S03'],
      ['05','Native DS01 A and B/C','live engineering','initial native504 episodes;bounded E3 assignments22','Initial sampler, source-metadata graph;not independent semantic qualification','S02,S10,S16'],
      ['06','Continuation','live engineering','4 originally never-dispatched assignments continued','Not retries of ambiguous timeout;not independent replications','S17'],
      ['07','Sampler v2 A/B/C','native DB','36conditions×2queries×7strategies=504 latest episodes','31snapshots;E3 not rerun on this latest sampler','S12,S13,S14'],
      ['08','Context frame','annotation draft','24silver cases, then329 preserving all24','0independent context gold;305additional drafts','S19,S20'],
      ['09','Final code/design review','offline','140tests checkpoint;178artifact hashes and parity/lineage validated','No new HTTP/DB calls, no semantic success estimate','S01,S09,S21']]
    table('D01','실제 실행 순서',['order','stage','class','observed','qualification','sources'],chronology,
          'actual chronology',sorted({sid for row in chronology for sid in row[-1].split(',')}))
    baselines=[['B0','단일 DB 전체 검색','metadata 선택 없음','맥락과 provenance 보존'],
        ['B1','단일 DB domain filter','B3와 같은 plan','planning과 물리적 배치 효과 분리'],
        ['B2','멀티 DB broadcast','전체 domain 조회','최종 전역 top-k 동일'],
        ['B3','멀티 DB metadata routing','전체 structured registry','같은 graph에서 reduced와 비교'],
        ['B3_reduced','멀티 DB metadata routing','축소 registry','현재 native proxy는 다른 전략과 같음'],
        ['B4','멀티 DB adaptive','공개 obligation 부족 시 유한 확장','공개 검사는 semantic 충분성 증명이 아님'],
        ['B5','멀티 DB provider oracle','evaluator가 reference 소재 provider 선택','gold evidence를 직접 주는 reasoning oracle과 구별']]
    table('D02','DS01 실행 baseline',['strategy','placement','selection','control'],baselines,'native DS01 definitions',['S04','S15'])
    full,bound=census['full'],census['source_bound_annotation_candidates']
    true_counts={r['category']:r['count'] for r in full['category_reasoning'] if r['reasoning']}
    table('D03','FinDER category/Reasoning/source coverage',['category','native_count','reasoning_true','source_bound_candidates'],
          [[c,n,true_counts.get(c,0),bound['categories'].get(c,0)] for c,n in full['categories'].items()], 'actual native census',['S08'])
    observed=[]
    for r in sorted(shifts,key=lambda r:(r['intervention'],r['seed'],r['severity'])):
        observed.append([r['condition_id'],r['intervention'],r['seed'],r['severity'],
            r['observed']['entity_tf'].get(cfg['target_entity_id']),
            r['covariate_changes']['relation_jsd'],r['covariate_changes']['neighborhood_jsd'].get(cfg['target_entity_id']),
            r['covariate_changes']['domain_jsd'],r['observed']['domain_document_counts'].get(cfg['target_domain'],0)/cfg['document_budget'],
            r['observed']['token_count'],r['covariate_changes']['token_relative_change'],r['causal_isolation'],r['research_qualified']])
    table('D04','Observed distribution shifts',['condition_id','intervention','seed','severity','target_entity_tf','aggregate_relation_jsd','target_entity_neighborhood_jsd','domain_jsd','target_domain_document_share','token_count','token_relative_change','causal_isolation','research_qualified'],observed,'actual measured covariates',['S14','S15'])
    cells=defaultdict(list)
    for r in metrics:
        s=lookup[r['condition_id']]
        cells[s['intervention'],s['severity'],r['strategy']].append(r)
    quality=[]
    for key,rows in sorted(cells.items()):
        if any(r['evidence_sufficiency_at_k'] is not None for r in rows):
            raise ValueError('semantic labels changed: update presentation claims before building')
        quality.append([*key,len(rows),len({r['query_id'] for r in rows}),len({r['seed'] for r in rows}),
                        mean(r['provided_reference_sufficiency_proxy'] for r in rows),None,
                        sum(not r['execution_success'] for r in rows)])
    table('D05','Native reference sufficiency by strategy',['intervention','severity','strategy','episodes','queries','seeds','provided_reference_sufficiency_proxy','semantic_evidence_sufficiency_at_k','execution_failures'],quality,'actual native reference proxy',['S12','S13','S14'])
    # Reconstruct the witness from the latest native run, never silently relabel the older receipt.
    native_input=load('S22');gold={g['query_id']:g for g in native_input['gold']}
    docmap={d['document_id']:d for d in native_input['documents']}
    corpus={r['condition_id']:r['corpus'] for r in read_payloads(source_map['S28'])}
    witnesses=[]
    for row in traces:
        shift=lookup[row['condition_id']]
        if shift['intervention']!='DS-01B' or row['strategy']!='B0' or row['error']:
            continue
        required={eid for group in gold[row['query_id']]['required_groups'] for alt in group for eid in alt}
        missing=required-set(row['evidence_ids'])
        rejected={r['document_id'] for r in row['packing_rejections'] if r['reason']=='context_byte_budget'}
        matching=[d for d in row['candidate_document_ids'] if docmap[d]['evidence_id'] in missing and d in rejected]
        if matching:
            witnesses.append((row,shift,matching,required))
    if not witnesses:
        raise ValueError('latest native packing witness unavailable')
    row,shift,missing_docs,required=sorted(witnesses,key=lambda v:(-v[1]['severity'],v[0]['seed'],v[0]['sample_id']))[0]
    available={docmap[d]['evidence_id'] for d in corpus[row['condition_id']]['document_ids']}
    if not required<=available:
        raise ValueError('witness lost gold at corpus stage')
    witness=dict(experiment_run_id=NATIVE,condition_id=row['condition_id'],sample_id=row['sample_id'],trace_id=row['trace_id'],
        graph_snapshot_id=row['graph_snapshot_id'],query_id=row['query_id'],seed=row['seed'],severity=shift['severity'],strategy=row['strategy'],
        gold_evidence_ids=sorted(required),gold_present_in_corpus=True,gold_in_candidates=True,
        rejected_gold_document_ids=missing_docs,packing_rejection_reason='context_byte_budget',
        global_top_k=cfg['global_top_k'],context_byte_budget=cfg['context_byte_budget'],actual_context_bytes=row['context_bytes'],
        delivered_evidence_ids=row['evidence_ids'],retrieval_error=row['error'],semantic_leakage_conclusion=None,
        source_ids=['S12','S14','S15','S22','S28'])
    json_write(output/'latest-native-packing-witness.json',witness)
    table('D06','Actual evidence delivery failure',['stage','observed','source_sample'],[
        ['gold preserved in corpus',True,row['sample_id']],['gold in database candidates',True,row['sample_id']],
        ['gold rejected by byte packing',True,row['sample_id']],['DB retrieval error',False,row['sample_id']],
        ['semantic context leakage established',None,row['sample_id']]],'latest native witness',['S12','S14','S15','S22','S28'])
    original,continued=read_payloads(source_map['S16']),read_payloads(source_map['S17'])
    public_questions={q['query_id']:q['text'] for q in native_input['queries']}
    original_by_sample={r['sample_id']:r for r in original}
    attempted_traces={r['trace_id'] for r in read_payloads(source_map['S29'])}
    if any(original_by_sample[r['parent_sample_id']]['trace_id'] in attempted_traces for r in continued):
        raise ValueError('continuation parent already has an HTTP attempt')
    effective=merge_continuations(original,continued)
    def e3row(r):
        judge=r.get('judge_result') or {}
        answer=r.get('generated_answer') or {}
        flag='manual_review_abstention_rubric_conflict' if r['kind']=='empty_evidence_control' and judge.get('correctness')==1 else ''
        label=next((ticker for ticker in ('CBOE','PANW') if ticker.casefold() in public_questions[r['query_id']].casefold()),r['query_id'])
        return [r['query_id'],label,r['kind'],r.get('strategy'),r['condition_id'],r['experiment_run_id'],r['sample_id'],r.get('parent_sample_id'),bool(answer),answer.get('status'),
                judge.get('correctness'),judge.get('faithfulness'),judge.get('reference_defect'),r.get('judge_calibrated'),r.get('error'),flag]
    e3headers=['query_id','query_label','kind','strategy','condition_id','experiment_run_id','sample_id','parent_sample_id','valid_generated_answer','answer_status','automatic_correctness','automatic_faithfulness','reference_defect_flag','judge_calibrated','execution_error','manual_review_flag']
    table('D07','Fixed-evidence outcomes after lineage-aware continuation',e3headers,
          [e3row(r) for r in effective if r['kind']!='end_to_end'],'uncalibrated automatic judge;8 original assignments',['S16','S17'])
    table('D08','DS01 campaign telemetry',['measure','value','scope'],[
        ['http_attempts',campaign['http_attempts'],'DS01 only; catalog2 included'],['inference_http_attempts',campaign['inference_http_attempts'],'not logical call count'],
        ['known_tokens',campaign['known_tokens'],'incomplete known-only sum'],['unknown_inference_usage_attempts',campaign['unknown_inference_usage_attempts'],'unknown, not zero'],
        ['http_200',campaign['http_statuses']['200'],'HTTP attempt status'],['http_502',campaign['http_statuses']['502'],'HTTP attempt status'],
        ['http_429',campaign['http_statuses']['429'],'HTTP attempt status'],['read_timeout',campaign['http_statuses']['None'],'ambiguous outcome not resubmitted']],
          'actual DS01 attempt accounting',['S10'])
    paired=load('S11')['paired_changes']
    table('D09','Exploratory paired effect and uncertainty',['intervention','severity','strategy','metric','paired_mean_change','ci95_low','ci95_high','components','seeds','pairs','inferentially_qualified'],
          [[r['intervention'],r['severity'],r['strategy'],r['metric'],r['paired_mean_change'],*r['ci95'],r['company_or_source_clusters'],r['corpus_seed_clusters'],r['pairs'],r['inferentially_qualified']] for r in paired],
          'paired proxy changes;not population or noninferiority proof',['S11'])
    table('D10','End-to-end effective assignments',e3headers,[e3row(r) for r in effective if r['kind']=='end_to_end'],
          '14 original E2E assignments;judge completion is not human-qualified accuracy',['S16','S17'])
    csv_write(output/'raw-e3-receipts.csv',e3headers,[e3row(r) for r in original+continued])
    claims=[
      ('F01','observed_dataset','FinDER는 5,703문항·8개 category이며 Reasoning=True 1,085개가 두 category에 집중된다','Category × Reasoning을 균형 있는 인과 비교로 해석하지 않는다',['S07','S08']),
      ('F02','observed_pilot','최신 2문항·3 seed 파일럿에서 7개 전략의 reference sufficiency proxy는 동일했다','A/C 1.0, B baseline 1.0 → shifted 0.667; semantic primary는 null이다. Federation 무효·동등성·noninferiority로 일반화할 수 없다',['S10','S11','S12','S13']),
      ('F03','observed_failure','정답 근거가 corpus와 retrieval candidate에 있어도 packing에서 제외될 수 있었다','최신 native 실행의 실제 sample로 검증했다. Relation JSD의 순수 인과효과나 answer leakage로 단정할 수 없다',['S12','S14','S15','S22','S28']),
      ('F05','uncalibrated_diagnostic','PANW는 oracle 근거 조건에서도 자동 correctness 0.5였다','원래 정답 요구의 불완전한 충족을 나타내는 진단이다. Empty 조건의 1.0/reference_defect는 rubric 불일치 검토 대상이며 정답성 개선의 증거가 아니다',['S16','S17']),
      ('F06','observed_telemetry','DS01 HTTP 73회와 known tokens 255,600을 보존했다','추론 5개 attempt의 usage가 미상이다. 전체 비용·billing 총계가 아니며 평가 비용과 배포 비용을 분리한다',['S10']),
      ('F07','observed_capability','Standard multi-database와 single/multi 내용 parity를 검증했고 application-level federation을 사용했다','Native Composite federation·최소 권한 ACL·성능 우위는 검증되지 않았다',['S09','S26','S27']),
      ('F08','engineering_smoke','실제 두 모델 추출 smoke에서 2/2 accepted와 normalized assertion 5개를 기록했다','기계적 span 검증이다. 독립 fact gold 기준 Grounded Fact F1이나 ontology 효과를 나타내지 않는다',['S05','S18']),
      ('F09','synthetic_diagnostic','Synthetic fixture에서 flat 1, broadcast 3, metadata/adaptive 평균 1.25 graph calls/question이었다','40개 agent-authored 질문이다. 실제 FinDER 품질·latency·scaling 결론이 아니며 단일 DB보다 호출 수가 적지도 않았다',['S05']),
      ('F10','readiness','329개 source-bound frame은 silver이며 독립 context/sufficient gold는 0개다','전체 표본 160개 중 11개의 binding이 완료됐고 149개는 대기 중이다. 문항 초안의 증가가 본 실험 완료를 뜻하지 않는다',['S07','S09','S20'])]
    findings=[dict(claim_id=cid,status=status,allowed_wording=wording,qualification=qualification,source_ids=ids)
              for cid,status,wording,qualification,ids in claims]
    json_write(output/'findings.json',findings)
    claim_ids={c['claim_id'] for c in findings}
    for slide in story['slides']:
        if not set(slide['source_ids'])<=source_ids or not set(slide['claim_ids'])<=claim_ids:
            raise ValueError('unresolved storyboard citation')
    assets=[]
    for aid,stem,scope,caption in ASSETS:
        for ext in ('.svg','.png','.mmd'):
            src=package/(stem+ext)
            if src.exists():
                rel=Path('assets')/(aid+'--'+src.name)
                (output/rel).parent.mkdir(exist_ok=True)
                shutil.copyfile(src,output/rel)
                assets.append(dict(asset_id=aid,path=str(rel),original_path=str(src.relative_to(package)),sha256=file_hash(src),scope=scope,caption=caption))
    json_write(output/'sources.json',sources)
    json_write(output/'assets.json',assets)
    json_write(output/'tables.json',[{k:v for k,v in r.items() if k!='rows'} for r in tables])
    json_write(output/'protocol-precedence.json',dict(
        priority=['Actual raw/cached run receipts for observed results','Latest final review/audit for scope','Historical design for proposed methods only','Mock/synthetic checkpoint for software behavior only'],
        collisions=[
          dict(field='B0',historical='no-evidence answer in early design',native='single-DB global retrieval in DS01',rule='Always prefix with protocol;no silent crosswalk'),
          dict(field='top_k',historical=10,native=5,rule='Actual DS01 uses top5 plus12000UTF8 bytes;do not re-label existing outcomes'),
          dict(field='corpus',historical='5832-passage full reference union planned',native='346-passagesourcepool;16docs/corpus;2queries',rule='No claim of full FinDER retrieval experiment'),
          dict(field='E3',historical='original A sampler/live assignments22',native='latestv2 ABC retrieval504;E3 not rerun',rule='Never present these as identical frozen end-to-end campaign'),
          dict(field='counts',historical='24frame,25/54tests and46HTTP',native='329silver,140testcheckpoint;DS01HTTP73',rule='Scopes/checkpoints differ;do not sum test counts or costs'),
          dict(field='gold',historical='native QA/reference gold available',native='extra context/sufficient gold0',rule='Do not erase native gold or promote silver'),
          dict(field='empty-evidence judge',historical='correctness rubric:abstention0',native='PANW empty automatic1.0 withreference_defect=true',rule='Keep raw receipt;manual-review flag;no quality-success claim')]))
    briefing=['# FinDER 발표 제작 인계','',
      '이 패키지는 연구 목적·방법·실제 findings·모든 원본 표와 재그리기 데이터를 전달한다. 최종 PPTX를 만든 것으로 주장하지 않는다. 권장 구성은 연구 검토용 한국어20장 본편과 전체 표 부록이다. 원하면 발표 길이에 맞춰 본편을 압축하되 부록과 출처는 보존한다.','',
      '먼저 AGENT_PROMPT.md를 읽고, storyboard.json과 findings.json을 기준으로 제작한다. sources.json은 파일 hash·원래 경로·현재 패키지 경로를 연결한다. protocol-precedence.json의 이름/예산/버전 충돌 규칙을 적용한다.','',
      f'원본 보고서 표 {sum(t["table_id"].startswith("T") for t in tables)}개와 파생 결과 표 {sum(t["table_id"].startswith("D") for t in tables)}개를 모두 CSV/JSON으로 보존했다. 표를8행씩 나누는 경우 부록은 총{appendix}페이지다. 부록은 참고 분량이며 본편 발표 시간에 포함하지 않아도 된다.','',
      '## 자료와 실행 범위','',
      '| 구분 | 내용 |','| --- | --- |','| 원본 | 5,703문항 / QA가능5,696 / category8 |','| 주석 | 329silver / 독립context gold0 |','| 최신native조회 | 36conditions×2queries×7strategies=504;36conditions=3interventions×4severity×3seed |','| 실측 지표 | provided-reference proxy; semantic metric은 null |','| 파일럿 관측 | 전략별 provided-reference sufficiency proxy가 같았다. Semantic 품질 차이는 아직 평가하지 못했다 |','| E3 | 원래22배정,4continuation;중복제거후22배정.미calibrated자동평가 |','',
      '## 전달 파일','',
      '| 파일 | 역할 |','| --- | --- |','| AGENT_PROMPT.md | 그대로 전달할 발표 제작 지시문 |','| STORYBOARD.md / storyboard.json | 20개슬라이드 제목·메시지·시각자료·발표자notes·출처 |','| TABLES.md / tables/ | 모든원본표와파생표;편집가능CSV/JSON |','| findings.json | 허용하는findings와해석한계 |','| assets/ / assets.json | 구조도·실제chart·historicalfixture그림의분류 |','| sources/ / sources.json | 고정된source사본·hash·scope |','| latest-native-packing-witness.json | 최신native실패사례의exactlineage |','| raw-e3-receipts.csv | blocked/continued를모두보존한26receipt의요약 |','| protocol-precedence.json | 같은이름/다른실행버전의오독방지 |','',
      '## 핵심 findings','']
    for f in findings:
        briefing.extend([f"- **{f['claim_id']} · {f['status']}**: {f['allowed_wording']}. {f['qualification']}",''])
    (output/'START_HERE.md').write_text('\n'.join(briefing).rstrip()+'\n')
    prompt='''# 발표 제작 에이전트 작업 지시

당신은 연구 발표를 만드는 에이전트다. 이 폴더를 작업의 근거로 사용해 실험 목적 → 순서/방법 → 실제 결과 → findings → 한계/다음 검증을 한국어 presentation으로 제작하라. 이 요청은 새 실험 실행을 포함하지 않는다.

1. START_HERE.md, protocol-precedence.json, findings.json, storyboard.json 순서로 읽는다. sources.json과 해당 source snapshot으로 모든 숫자의 단위·분모·scope를 확인한다. 저장소 접근이 없어도 이 패키지로 데이터와 그림을 확인할 수 있다.
2. 권장 본편20장과 전체 표 부록을 만든다. 사용자에게 별도 slide count가 주어지면 우선한다. tables.json의 모든 T/D 표를 부록에 누락 없이 포함하고 표ID·원본행범위·sourceID를 notes에 기록한다. 긴 표는 행/열을 나누되 열·행·정밀도는 보존한다. CSV/JSON은 편집 가능한 native 표/chart로 만든다. 대형 표를 작은 글자의 screenshot으로 넣지 않는다.
3. 슬라이드마다 한 메시지를 유지한다. storyboard의 key_message는 근거가 있는 주장인지 확인한다. 데이터 준비/설계/실측/미calibrated진단/synthetic결과를 구분한다. 표시가 필요한 제한은 해당 그림 옆에 두고 authoring IDs나 버전충돌 설명은 notes/부록에 둔다.
4. A01 구조도는 가능하면 editable diagram으로 재작성한다. D03/D04/D05/D09에서 chart를 재그릴 수 있다. A03은 reference proxy heatmap이고 A04 가운데 패널은 aggregate relation-count JSD다. Target-neighborhood JSD와 혼용하지 않는다. 같은 전략은 같은 색을 사용한다. 미래실험의 예상곡선/효과크기/검증된gold수치를 만들지 않는다.
5. 5,703개QA목록,329개silver주석,최신2문항DB실험을 구별한다. 504episode를504독립문항이라고 하지 않는다. 31snapshot/36conditions/3seed를 구별한다. Primary semantic sufficient/groundedQA/null을0이나proxy로 바꾸지 않는다.
6. DS01 B0(global단일DB)와초기design B0(no-evidence)를 혼동하지 않는다. 같은plan B1↔B3로placement, 같은graph B3↔reduced로metadata효과를 설명한다. 현재멀티DB는application-level federation이다. NativeComposite,reviewedbusinessfactgraph,자유Cypher,graphmulti-hop우위는입증되지않았다.
7. 실측findings를우선한다: 전략간proxy차이미관측;B의낮은값에는contextpacking손실이포함;A/Cceiling;oracle근거제공후에도PANW자동점수0.5. 모두작은pilot/미calibrated한계를붙인다. CBOEemptyReadTimeout은unknown이고재전송하지않았다. PANWempty자동1.0/reference_defect는abstentionrubric과충돌하므로검토대상이며정답개선사례로쓰지않는다.
8. Original22배정과continuation4개는26독립episode가아니다. 원래4개는전송전에차단됐고최종22배정중한번씩수행됐다. raw-e3-receipts.csv와D07/D10의effective결과를구분한다. E3는이전A/sampler실행이고최신v2ABC504retrieval의E3재실행이아니다.
9. DS01HTTP73/known255,600tokens는그campaign범위의불완전합계다. 이전campaign46HTTP와무심코합산하지않는다. 누락usage를0으로채우지않는다. Evaluator비용을배포routing/retrieval/synthesis와분리한다. GPU/HBM/KV-cache성능추정은금지한다.
10. 적용가능한presentation skill을따른다. 플랫폼/디자인미지정시그환경의기본동작을사용하되이자료의편집가능성과출처를보존한다. Speaker notes에sourceID,원본경로/JSONpointer또는표ID,해당run/condition/sampleID를필요한경우기록한다. 외부논문을설명할때는본문을실제로확인한뒤인용한다. 원논문의성과를우리측정결과로표현하지않는다.

결과물은 presentation, 발표자notes, 전체표coverage map(표ID→slide번호/행범위), 핵심claim→slide/source map이다. 모든슬라이드를render하여가독성·겹침·잘림·표누락을검토한다. 이원고에없는새로운실험이나외부공유/배포는실행하지않는다. 승인요청을추가하지말고이범위의제작을완료한다.
'''
    (output/'AGENT_PROMPT.md').write_text(prompt)
    notes=['# 발표 구성과 발표자 노트','']
    for slide in story['slides']:
        notes.extend([f"## {slide['slide_id']} {slide['title']}",'',f"**핵심 메시지:** {slide['key_message']}",''])
        notes.extend(['- '+b for b in slide['body']])
        notes.extend(['',f"**시각자료:** {slide['visual']}",'',f"**발표자 노트:** {slide['speaker_notes']}",'',
                      f"**근거:** {', '.join(slide['source_ids'])}; claims: {', '.join(slide['claim_ids']) or 'design/method'}",''])
    (output/'STORYBOARD.md').write_text('\n'.join(notes).rstrip()+'\n')
    tablebook=['# 전체 표와 결과 데이터','',
               '| 표 ID | 제목 | 상태/범위 | 행 수 | 편집 데이터 | 부록 분할 |','| --- | --- | --- | ---: | --- | --- |']
    for t in tables:
        tablebook.append(f"| {t['table_id']} | {t['title']} | {t['scope']} | {len(t['rows'])} | [{t['table_id']}.csv]({t['csv_path']}) | {', '.join(p['appendix_id'] for p in t['appendix_pages'])} |")
    for t in tables:
        tablebook.extend(['',f"## {t['table_id']} {t['title']}",'',f"Scope: {t['scope']}; sources: {', '.join(t['source_ids'])}",'',
                          '| '+' | '.join(h.replace('|',r'\|') for h in t['headers'])+' |',
                          '| '+' | '.join('---' for _ in t['headers'])+' |'])
        for row in t['rows']:
            tablebook.append('| '+' | '.join(('unknown' if c is None else str(c)).replace('|',r'\|').replace('\n',' ') for c in row)+' |')
    (output/'TABLES.md').write_text('\n'.join(tablebook)+'\n')
    csv_write(output/'table-slide-coverage.template.csv',
        ['table_id','source_ids','proposed_appendix_page','first_row','last_row','actual_deck_slide','coverage_status'],
        [[t['table_id'],t['source_ids'],p['appendix_id'],p['row_start'],p['row_end'],'','pending_production']
         for t in tables for p in t['appendix_pages']])
    csv_write(output/'claim-slide-coverage.template.csv',
        ['claim_id','source_ids','proposed_slide_ids','actual_deck_slide','coverage_status'],
        [[f['claim_id'],f['source_ids'],[s['slide_id'] for s in story['slides'] if f['claim_id'] in s['claim_ids']],
          '','pending_production'] for f in findings])
    page=["<!doctype html><html lang='ko'><meta charset='utf-8'><title>FinDER 발표 인계</title><style>body{max-width:1150px;margin:2rem auto;padding:1rem;font:16px/1.65 sans-serif;color:#17334a}article{border-top:2px solid #bccbd5;margin-top:2rem}table{border-collapse:collapse;width:100%}td,th{border:1px solid #b8cbd3;padding:.5rem;text-align:left}.scope{background:#fff1d2;padding:.7rem}a{color:#175da8}img{max-width:100%}</style>",
          '<h1>FinDER 발표 제작 인계</h1><p>20장 본편과 전체 표 부록을 위한 원고·편집 데이터·출처 패키지입니다.</p>',
          '<p><a href="START_HERE.md">시작 안내</a> · <a href="AGENT_PROMPT.md">전달할 지시문</a> · <a href="TABLES.md">전체 표</a></p>',
          '<p class="scope">2문항 native pilot · 329 silver frame · 독립 context gold 0 · Federation 우위 미입증</p><h2>핵심 findings</h2>']
    for f in findings:
        page.append('<p><b>'+f['claim_id']+'</b> '+html.escape(f['allowed_wording'])+'<br><small>'+html.escape(f['qualification'])+'</small></p>')
    page.append('<h2>발표 순서</h2><table><tr><th>슬라이드</th><th>제목</th><th>핵심 메시지</th><th>근거</th></tr>')
    for s in story['slides']:
        page.append('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in [s['slide_id'],s['title'],s['key_message'],', '.join(s['source_ids'])])+'</tr>')
    page.append('</table><h2>사용할 그림</h2>')
    for a in assets:
        if a['path'].endswith('.png'):
            page.append('<article><h3>'+a['asset_id']+' · '+html.escape(a['scope'])+'</h3><p>'+html.escape(a['caption'])+'</p><img src="'+a['path']+'"></article>')
    (output/'index.html').write_text('\n'.join(page)+'\n')
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=package,check=True,capture_output=True,text=True).stdout.strip()
    build_manifest=dict(version='finder-presentation-handoff-v1',source_commit=commit,
        package_id=digest([[s['source_id'],s['sha256']] for s in sources]+[file_hash(config_path),file_hash(Path(__file__))]),
        source_reports=6,original_tables=sum(t['table_id'].startswith('T') for t in tables),derived_tables=sum(t['table_id'].startswith('D') for t in tables),
        original_table_rows=sum(len(t['rows']) for t in tables if t['table_id'].startswith('T')),
        main_slides=len(story['slides']),appendix_pages_at_8_rows=appendix,source_files=len(sources),asset_files=len(assets),
        effective_E3_assignments=len(effective),raw_E3_receipts=len(original)+len(continued),new_http_calls=0,new_graph_calls=0,
        production_agent_dispatched=False,delivery_scope='Source-backed authoring handoff, not a completed presentation.',
        storyboard_config_sha256=file_hash(config_path),builder_sha256=file_hash(Path(__file__)))
    json_write(output/'manifest.json',build_manifest)
    verify(output)
    json_write(output/'sha256.json',{str(p.relative_to(output)):file_hash(p) for p in sorted(output.rglob('*')) if p.is_file()})
    with zipfile.ZipFile(output.with_suffix('.zip'),'w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(output.rglob('*')):
            if p.is_file():
                archive.write(p,str(Path(output.name)/p.relative_to(output)))
    return build_manifest


def verify(output):
    output=Path(output)
    sources=json.loads((output/'sources.json').read_text())
    for source in sources:
        if file_hash(output/source['bundle_path'])!=source['sha256']:
            raise ValueError('copied source hash mismatch')
    tables=json.loads((output/'tables.json').read_text())
    for t in tables:
        data=json.loads((output/t['json_path']).read_text())
        if t['table_id'].startswith('T'):
            origin=next(s for s in sources if s['source_id']==t['source_ids'][0])
            originals=extract_tables((output/origin['bundle_path']).read_text())
            match=next((r for r in originals if r['line_start']==t['source_locator']['line_start']),None)
            if match is None or match['headers']!=data['headers'] or match['rows']!=data['rows']:
                raise ValueError('source table content changed')
        with (output/t['csv_path']).open(encoding='utf-8-sig',newline='') as stream:
            rows=list(csv.reader(stream))
        expected_rows=[[json.dumps(c,ensure_ascii=False) if isinstance(c,(list,dict)) else ('' if c is None else str(c)) for c in row] for row in data['rows']]
        if rows[0]!=data['headers'] or rows[1:]!=expected_rows:
            raise ValueError('table CSV readback mismatch')
        pages=data['appendix_pages']
        flattened=[i for p in pages for i in range(p['row_start'],p['row_end']+1)]
        if flattened!=list(range(1,len(data['rows'])+1)):
            raise ValueError('appendix table coverage mismatch')
    expected=sum(len(extract_tables((output/s['bundle_path']).read_text())) for s in sources if s['original_path'].endswith('.md'))
    if expected!=sum(t['table_id'].startswith('T') for t in tables):
        raise ValueError('omitted source table')
    return True


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--package',type=Path,default=Path.cwd())
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    print(json.dumps(build(args.package,args.output),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
