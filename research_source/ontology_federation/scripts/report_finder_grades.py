#!/usr/bin/env python3
"""Receipt-bound automated grading report; no network or new benchmark answers."""
import collections,csv,hashlib,html,importlib.util,json,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];path=ROOT/'scripts/grade_finder_answers.py';spec=importlib.util.spec_from_file_location('grader',path);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
REPORT=g.REPORT;RUN=g.RUN

def main():
 terminal=g.read(RUN/'terminal.json')
 if terminal['status']!='completed':raise RuntimeError('final report requires completed grading; preserve incomplete run separately')
 g.summarize();first=g.read(REPORT/'summary.json');original=g.lines(RUN/'grades.jsonl');repairs=g.lines(RUN/'repair-grades.jsonl')
 if (RUN/'repair-protocol.json').exists():
  if not (RUN/'repair-terminal.json').exists() or g.read(RUN/'repair-terminal.json')['status']!='completed':raise RuntimeError('repair not complete')
  repair_protocol=g.read(RUN/'repair-protocol.json');first['actual_http_attempts_including_preflight']=repair_protocol['first_pass_http_attempts'];first['known_usage']=repair_protocol['first_pass_known_usage'];first['http_statuses']=repair_protocol['first_pass_http_statuses'];first['usage_unknown_requests']=repair_protocol['first_pass_unknown_usage_requests']
 g.dump(REPORT/'first_pass_summary.json',first);(REPORT/'first_pass_scores.csv').write_bytes((REPORT/'scores.csv').read_bytes())
 selected={(r['item_id'],r['stage']):r for r in original}
 for r in repairs:
  key=(r['item_id'],r['stage']);assert selected[key]['status']=='judge_failure','only failed first-pass contracts may be recovered';selected[key]=r
 derived=RUN/'derived-score-view';derived.mkdir(exist_ok=True)
 for name in ['items.evaluator-only.json','spans.jsonl','terminal.json']:
  target=derived/name
  if not target.exists():target.symlink_to(RUN/name)
 (derived/'grades.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected.values()))
 try:g.RUN=derived;g.summarize()
 finally:g.RUN=RUN
 summary=g.read(REPORT/'summary.json');summary.update(run=str(RUN),grade_selection='first pass plus one typed-contract recovery only for failed judge outputs',first_pass_judge_failures=sum(r['status']=='judge_failure' for r in original),repair_attempted=len(repairs),repair_graded=sum(r['status']=='graded' for r in repairs));g.dump(REPORT/'summary.json',summary)
 items=g.read(RUN/'items.evaluator-only.json');imap={i['item_id']:i for i in items};grades=list(selected.values());by=selected;assert len([k for k in by if k[0] in imap])==3170
 rows=list(csv.DictReader((REPORT/'scores.csv').open()));groups=summary['groups'];question_rows=collections.defaultdict(dict)
 for row in rows:
  for stage in ['correctness','faithfulness']:
   r=by.get((row['item_id'],stage),{});row[stage+'_status']=r.get('status','not_dispatched');row[stage+'_schema_error']=r.get('schema_error') or ''
 with (REPORT/'scores.csv').open('w') as stream:
  w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (REPORT/'judge_failures.csv').open('w') as stream:
  w=csv.writer(stream);w.writerow(['item_id','question_id','stage','phase','http_status','schema_error','receipt_sha256'])
  for r in grades:
   if r['status']=='judge_failure':w.writerow([r['item_id'],imap[r['item_id']]['question_id'],r['stage'],r['phase'],r.get('http_status'),r.get('schema_error'),r.get('receipt_sha256')])
 for row in rows:question_rows[row['question_id']][(row['workload'],row['condition'])]=row
 pairs=[]
 for workload,contrasts in [('fixed_source',[('G','GX'),('G','FX'),('GX','FX')]),('index_to_answer',[('GX','FX')])]:
  for a,b in contrasts:
   diffs=[]
   for qid,qs in question_rows.items():
    x,y=qs.get((workload,a)),qs.get((workload,b))
    if x and y and x['correctness']!='' and y['correctness']!='':diffs.append(float(y['correctness'])-float(x['correctness']))
   pairs.append({'workload':workload,'baseline':a,'comparison':b,'paired_questions':len(diffs),'comparison_higher':sum(x>0 for x in diffs),'comparison_lower':sum(x<0 for x in diffs),'equal':sum(x==0 for x in diffs),'mean_score_difference_on_known_pairs':statistics.mean(diffs) if diffs else None,'interpretation':'same-model automated development comparison; judge error and selection bias uncalibrated'})
 g.dump(REPORT/'paired_comparisons.json',pairs)
 bycat=collections.defaultdict(list)
 for row in rows:bycat[(row['workload'],row['condition'],row['category'])].append(row)
 with (REPORT/'category_scores.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['workload','condition','category','assigned','scored','mean_correctness_on_known','judge_missing'])
  for (wk,co,cat),rs in sorted(bycat.items()):
   values=[float(r['correctness']) for r in rs if r['correctness']!=''];w.writerow([wk,co,cat,len(rs),len(values),statistics.mean(values) if values else '',len(rs)-len(values)])
 # A review queue is not a fabricated human calibration receipt.
 pilot=set(g.read(RUN/'protocol.json')['pilot_question_ids'])
 with (REPORT/'human_review_pending.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['item_id','question_id','pilot','original_answer_receipt','reviewer_id','human_correctness','human_source_support','rationale','review_status'])
  for item in items:w.writerow([item['item_id'],item['question_id'],item['question_id'] in pilot,item['original_answer_receipt'],'','','','','pending'])
 # Cross-tabs keep native agreement, supported claims and abstention distinct.
 behavior=[]
 for group in groups:
  rs=[r for r in rows if (r['workload'],r['condition'])==(group['workload'],group['condition'])]
  abstaining=[r for r in rs if imap[r['item_id']]['correctness_input']['candidate_answer'] and imap[r['item_id']]['correctness_input']['candidate_answer'].get('abstain') is True]
  known_faith=[r for r in rs if r['faithfulness_status']=='graded']
  behavior.append({'workload':group['workload'],'condition':group['condition'],'assigned':len(rs),'explicit_abstentions':len(abstaining),'abstention_appropriate':sum(r['abstention_assessment']=='appropriate' for r in abstaining),'abstention_not_appropriate':sum(r['abstention_assessment']=='not_appropriate' for r in abstaining),'abstention_unclear':sum(r['abstention_assessment']=='unclear' for r in abstaining),'abstention_unassessed_or_not_applicable':sum(r['abstention_assessment'] not in {'appropriate','not_appropriate','unclear'} for r in abstaining),'faithfulness_graded':len(known_faith),'no_material_claims':sum(r['material_claims']=='0' for r in known_faith),'incomplete_nonempty_claim_lists':sum(r['claims_complete']=='False' and int(r['material_claims'] or 0)>0 for r in known_faith),'answers_with_unsupported_claims':sum(int(r['unsupported_claims'] or 0)>0 for r in known_faith),'fully_correct_with_unsupported_claims':sum(r['correctness']!='' and float(r['correctness'])==1 and int(r['unsupported_claims'] or 0)>0 for r in known_faith),'answers_with_unclear_claims':sum(int(r['unclear_claims'] or 0)>0 for r in known_faith)})
 with (REPORT/'behavior_summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(behavior[0]));w.writeheader();w.writerows(behavior)
 # Hash all actual evaluator receipts and bind each scored judgment to its own request.
 receipt_index=[];known_tokens=collections.Counter();http=collections.Counter();roles=collections.Counter()
 spans=g.lines(RUN/'spans.jsonl');spanmap={s['receipt_sha256']:s for s in spans if s['component']=='mara' and s.get('receipt_sha256')}
 for p in sorted((RUN/'receipts').glob('*.json')):
  raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==p.stem;d=json.loads(raw)
  assert g.ow.sha({'path':d['path'],'body':d['request']})==d['request_sha256'];body=d.get('request');usage=d.get('usage') or {};http[str(d['http_status'])]+=1
  stage='catalog' if d['path']=='/models' else 'preflight' if d['request_id']=='grading-tool-probe' else 'correctness' if 'correctness' in d['request_id'] else 'faithfulness' if 'faithfulness' in d['request_id'] else 'other'
  roles[stage]+=1
  for key in ['prompt_tokens','completion_tokens']:known_tokens[key]+=usage.get(key) or 0
  if stage in {'correctness','faithfulness'}:
   payload=json.loads(body['messages'][1]['content'])
   assert all(('published_answer' in x)==(stage=='correctness') and ('delivered_evidence' in x)==(stage=='faithfulness') for x in payload['items'])
   real=[imap[x['item_id']] for x in payload['items'] if x['item_id'] in imap];assert len({x['question_id'] for x in real})==len(real)
   if stage=='faithfulness':assert '"index_profile"' not in body['messages'][1]['content'] and '"namespace"' not in body['messages'][1]['content']
  assert p.stem in spanmap,'actual HTTP span missing';actual_span=spanmap[p.stem]
  receipt_index.append({'trace_id':actual_span['trace_id'],'span_id':actual_span['span_id'],'actual_http_ms':actual_span['duration_ms'],'request_id':d['request_id'],'stage':stage,'http_status':d['http_status'],'input_tokens':usage.get('prompt_tokens'),'output_tokens':usage.get('completion_tokens'),'request_sha256':d['request_sha256'],'receipt_sha256':p.stem,'receipt_path':str(p),'model':(body or {}).get('model')})
 with (REPORT/'request_index.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(receipt_index[0]));w.writeheader();w.writerows(receipt_index)
 assert not list((RUN/'http-state').glob('*.pending')),'unresolved requests remain'
 refs={r['receipt_sha256'] for r in receipt_index}
 for r in grades:
  if r.get('receipt_sha256'):assert r['receipt_sha256'] in refs
 g.dump(REPORT/'audit.json',{'status':'passed','report_generation_code_sha256':g.sha_file(Path(__file__)),'original_answer_paths':1585,'terminal_real_item_stage_pairs':3170,'original_answers_regenerated':0,'request_and_receipt_hashes_verified':len(receipt_index),'strict_gold_evidence_input_separation':True,'no_same_question_conditions_in_a_batch':True,'same_model_judge':True,'human_review_count':0,'no_unresolved_requests':True,'first_pass_judge_failures':summary['first_pass_judge_failures'],'repair_attempted':len(repairs),'repair_graded':summary['repair_graded'],'synthetic_controls':g.read(RUN/'controls.json'),'pilot':g.read(RUN/'pilot-audit.json'),'request_roles':dict(roles),'canonical_span_count':len(spans),'every_HTTP_receipt_bound_to_actual_span':True,'output_tokens_known':known_tokens['completion_tokens'],'input_tokens_known':known_tokens['prompt_tokens']})
 # Selected actual episodes, balanced across agreement, support gaps, and stable cases.
 priority=['55bd477c','8454865b','83c09c7f','61e10a82','a4154d22','bf2ba2ab'];cards=[]
 for qid in priority:
  selected=[r for r in rows if r['question_id']==qid]
  item=next(i for i in items if i['question_id']==qid);body=f'<h2>{html.escape(qid)}</h2><p>{html.escape(item["correctness_input"]["question"])}</p><details><summary>Published FinDER answer</summary><pre>{html.escape(item["correctness_input"]["published_answer"])}</pre></details>'
  for row in selected:
   i=imap[row['item_id']];f=by[(row['item_id'],'faithfulness')];body+=f'<h3>{row["workload"]} / {row["condition"]}</h3><p>Gold-agreement score: {row["correctness"] or "UNKNOWN"}; claim support: {row["faithfulness"] or "not assessable"}</p><pre>{html.escape(json.dumps(i["correctness_input"]["candidate_answer"],ensure_ascii=False,indent=2))}</pre><p>{html.escape(row["judge_correctness_reason"])}</p><details><summary>Actually delivered evidence and automated claim judgments</summary><pre>{html.escape(json.dumps({"evidence":i["faithfulness_input"]["delivered_evidence"],"judgment":f.get("value")},ensure_ascii=False,indent=2))}</pre></details>'
  cards.append(body)
 (REPORT/'cases.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>FinDER automated grading cases</title><style>body{max-width:1100px;margin:40px auto;font:17px/1.6 system-ui;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f6f9;padding:18px}h2{margin-top:70px}summary{cursor:pointer}</style><h1>Published-answer agreement and evidence support</h1><p>Same-model automated judgments; no independent human calibration. Source and response records are preserved.</p>'+''.join(cards)+'</html>')
 table=['| Workload | Condition | Assigned | Scored | Mean gold-agreement score | Fully correct | Assessable claim lists | Mean supported-claim fraction |','| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
 for x in groups:
  score=f'{x["mean_correctness"]:.3f}' if x['mean_correctness'] is not None else f'{x["correctness_lower_bound"]:.3f}–{x["correctness_upper_bound"]:.3f} (bounds)';faith=f'{x["automatic_faithfulness_mean_on_assessable_claim_lists"]:.3f}' if x['automatic_faithfulness_mean_on_assessable_claim_lists'] is not None else 'UNKNOWN'
  table.append(f'| {x["workload"]} | {x["condition"]} | {x["assigned"]} | {x["correctness_scored"]} | {score} | {x["full_correct_count"]} | {x["faithfulness_assessable"]} | {faith} |')
 paired_lines=[]
 for pair in pairs:
  paired_lines.append(f"- {pair['workload']}, {pair['baseline']} → {pair['comparison']}: {pair['comparison_higher']} higher, {pair['comparison_lower']} lower and {pair['equal']} equal scores among {pair['paired_questions']} known pairs.")
 text='''# Automated grading of 317 FinDER questions

The existing 317 questions and 1,585 assigned answer paths have been assessed without regenerating benchmark answers. This report separates agreement with the published FinDER answer from support in the evidence actually delivered to the model. These are automated judgments by gpt-oss-120b, the same model family as the generator, without independent human calibration.

## Results

'''+ '\n'.join(table)+f'''

Each condition has 317 assigned questions. Original generation/contract failures and explicit abstentions score 0 under the frozen benchmark rule. A judge failure is UNKNOWN rather than an incorrect benchmark answer; score bounds include that uncertainty. The fully-correct count is the number of 1 scores, not all nonzero scores.

Faithfulness is the supported fraction of the material claims listed by the automated judge, averaged only over complete, nonempty claim lists. Answers with no material claims have undefined faithfulness, not 1. Unknown-support claims remain in the denominator; incomplete or failed judge lists are not treated as complete. Assessment coverage must be read with the mean. These rows can cover different subsets of answers, so their means are not a matched causal comparison.

## Same-question comparisons

'''+ '\n'.join(paired_lines)+f'''

These are changes in same-model automated rubric scores, not independently verified wins or losses. The paired counts show whether a condition helps some questions while hurting others; they must not be replaced by a blanket claim that ontology improves every answer. No significance test or population-generalization claim is made.

## Reading the five conditions

The fixed-source lane has G (generic definitions), GX (generic definitions plus local rules), and FX (FIBO-derived definitions plus the same local rules). The source evidence is fixed in this lane. The index-to-answer lane compares graphs extracted with GX or FX; both use GX at query time. Each lane answers a different question, so these five rows are not a single leaderboard.

A score of 1 means all material requested conclusions are correct against the published answer; 0.5 means substantively correct but incomplete without a critical contradiction; 0 means wrong, a material mismatch, explicit abstention, or original output failure. Period, issuer, business scope, unit, sign and modality matter, as do the explanation and operands. The score is a rubric score, not a probability of truth.

[behavior_summary.csv](behavior_summary.csv) separately counts appropriate abstentions, answers with unsupported claims, and fully gold-correct answers containing an unsupported claim. These are overlapping automated labels, not independent human findings. An answer's average claim support is not an all-or-nothing correctness rate.

## Execution and controls

Actual HTTP attempts including catalog, probe and synthetic controls: {len(receipt_index)}. Known input tokens: {known_tokens['prompt_tokens']:,}; output tokens: {known_tokens['completion_tokens']:,}. Unknown usage is retained in the summary; catalog calls have no inference usage. All requests use the shared MARA admission lock and one concurrent call. No original graph writes, extraction runs or answer generation were repeated.

The 40-question diagnostic sample was scored first, after synthetic unit/sign/abstention/source-support controls. Known-response judge contract failures receive at most one single-item recovery using exact item/citation ID enums and unchanged semantic prompts. This changes batching context, so first-pass scores and recovered scores are preserved separately. Transport-ambiguous requests are never replayed. Format coverage gates are protocol checks, not human calibration. There are 0 independent human review submissions. Up to 4 unrelated questions share a request; the same question’s conditions are never together. This reduces cost but is not fully isolated evaluation.

Correctness requests contain question, published answer and full candidate response, but no delivered evidence. Faithfulness requests contain question, candidate and delivered evidence, but no published answer. Treatment names are removed from packet metadata while semantic qualifiers and source text remain. Numeric scale normalization is diagnostic and does not select the nearest convenient number from a long gold answer.

## Interpretation

Use within-workload paired comparisons. Index-to-answer conditions change extraction and graph content while query guidance remains GX. Fixed-source G/GX/FX changes query guidance with the supplied source evidence fixed. These are source-bound development cohorts, not an unseen holdout or open retrieval over all FinDER references. This grading does not compare retrieval quality against BM25. Do not interpret cross-workload differences as a pure federation or ontology effect.

An evidence-appropriate abstention can still receive 0 under the frozen published-answer benchmark rubric. That does not make it an incorrect evidence-grounded decision. Likewise, an answer that agrees with published interpretation can contain a source-unsupported claim. Keep benchmark agreement, source support and abstention assessment separate.

Published answers may contain hypothetical examples or explanatory assumptions. Their agreement score and evidence support may differ. A suspected reference defect is flagged without repairing the original gold or changing the denominator. Same-model grading can reproduce the generator’s preferences and mistakes. Automated scores are not independently established semantic truth or official FinDER evaluation reproduction.

[Selected source-linked cases](cases.html) retain both successful and failed/partial behavior, including Align, FMC, PFG, Lilly and NVR. [scores.csv](scores.csv) contains all 1,585 answer-path rows after contract recovery. [first_pass_scores.csv](first_pass_scores.csv) and [first_pass_summary.json](first_pass_summary.json) preserve the unrecovered first-pass view. [paired_comparisons.json](paired_comparisons.json) retains known-pair higher/lower/equal counts. [category_scores.csv](category_scores.csv) reports coverage per native category; tiny strata such as Risk must not be generalized.

## Reproduction

[protocol.json](protocol.json) freezes prompts, model, bounds and source hashes. [request_index.csv](request_index.csv) binds every HTTP request to its immutable receipt in /data. [audit.json](audit.json) verifies hashes, input separation, completion and unresolved guards. [human_review_pending.csv](human_review_pending.csv) is a pending review queue, not fabricated annotations.

The original run is `{RUN}`. Its append-only grades.jsonl contains all item-level verdicts and claims; receipts, events, spans and metrics are stored alongside. The original 317 question results and independent-semantic-correctness UNKNOWN fields remain unchanged; this report is a separate automated-grading artifact.
'''
 (REPORT/'RESULTS_EN.md').write_text(text)
 # Repository text uses LF; preserve parsed CSV values while normalizing rows.
 for file in REPORT.glob('*.csv'):
  with file.open(newline='') as source:csv_rows=list(csv.reader(source))
  with file.open('w',newline='') as target:csv.writer(target,lineterminator='\n').writerows(csv_rows)
 print(json.dumps({'graded_rows':len(rows),'receipts_verified':len(receipt_index),'groups':len(groups),'human_reviews':0}))
if __name__=='__main__':main()
