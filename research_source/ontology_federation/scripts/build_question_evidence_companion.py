#!/usr/bin/env python3
"""Join frozen question/gold, indexing, delivered evidence and grades offline.

Evaluator-only output. No API calls, graph writes or replacement judgments.
"""
import collections
import csv
import gzip
import hashlib
import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'reports/research/ontology_question_extension_remaining_v1/combined/input'
RUN = Path('<LOCAL_EXPERIMENT_RUNS>/runs/ontology-question-extension-cumulative317-final-20261110-v1')
JUDGE = RUN.parent / 'finder-answer-grading-20261110-v1'
REPORT = ROOT / 'reports/research/finder_answer_grading_v1'
OUT = RUN.parent.parent / 'reports/presentation/kg_construction_story_v2/question_evidence'


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def pretty(value):
    return '<pre>' + html.escape(json.dumps(value, ensure_ascii=False, indent=2)) + '</pre>'


def details(label, value):
    return '<details><summary>' + html.escape(label) + '</summary>' + pretty(value) + '</details>'


STYLE = '''<style>
body{font:16px/1.5 system-ui,sans-serif;max-width:1200px;margin:40px auto;padding:0 20px;color:#17212b}
a{color:#075985}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f3f6f8;padding:16px}
table{border-collapse:collapse;width:100%;margin:16px 0}td,th{border:1px solid #ccd5df;padding:8px;text-align:left;vertical-align:top}
details{border:1px solid #ccd5df;padding:12px;margin:12px 0}summary{cursor:pointer;font-weight:600}
.notice{background:#fff3cd;padding:16px}.gold{background:#eaf3ff;padding:16px}input{padding:10px;width:95%}
</style>'''


def projection(assertions):
    entities = {a['entity_key']: {'id': a['entity_key'], 'subject': a['subject']} for a in assertions}
    sources = {a['source_id'] for a in assertions}
    return {
        'status': 'derived visualization of admitted payloads; not a fresh database dump',
        'entity_nodes': list(entities.values()),
        'assertion_nodes': assertions,
        'source_nodes': sorted(sources),
        'edges': [edge for a in assertions for edge in [
            {'from': a['entity_key'], 'type': 'HAS_ASSERTION', 'to': a['id']},
            {'from': a['id'], 'type': 'SUPPORTED_BY', 'to': a['source_id']}]],
        'business_relationship_representation': 'attribute, value and qualifiers reside in assertion payloads; physical edges are storage relationships',
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'cases').mkdir(exist_ok=True)
    questions = {q['query_id']: q for q in read(INPUT / 'questions.evaluator-only.json')}
    sources = {s['id']: s for s in read(INPUT / 'sources.json')}
    indexes = {(r['question_id'], r['profile']): r for r in lines(RUN / 'index-outcomes.jsonl')}
    outcomes = {o['episode_id']: o for o in lines(RUN / 'outcomes.jsonl')}
    items = read(JUDGE / 'items.evaluator-only.json')
    scores = {r['item_id']: r for r in csv.DictReader((REPORT / 'scores.csv').open())}
    grades = {(r['item_id'], r['stage']): r for r in lines(JUDGE / 'grades.jsonl')}
    for r in lines(JUDGE / 'repair-grades.jsonl'):
        key = r['item_id'], r['stage']
        assert grades[key]['status'] == 'judge_failure'
        grades[key] = r
    grouped = collections.defaultdict(list)
    for item in items:
        grouped[item['question_id']].append(item)
    spans = collections.defaultdict(list)
    for s in lines(RUN / 'spans.jsonl'):
        spans[s['episode_id']].append(s)
    checked = {}

    def receipt(h):
        if not h:
            return None
        p = RUN / 'receipts' / (h + '.json')
        if h not in checked:
            assert sha(p) == h, h
            checked[h] = str(p)
        return read(p)

    cards = []
    totals = collections.Counter()
    for qid, q in sorted(questions.items()):
        assert len(grouped[qid]) == 5
        card = {
            'question_id': qid, 'question': q['question'], 'category': q['category'],
            'native_reasoning': q['native_reasoning'], 'native_type': q['native_type_raw'],
            'published_ground_truth': {
                'answer': q['original_answer'], 'original_gold_hash': q['original_gold_hash'],
                'role': 'evaluator-only; unchanged published answer, not independent source-entailment gold',
            },
            'selected_references': [sources[s] for s in q['source_ids']],
            'extraction_views': {}, 'answer_paths': [],
            'independent_human_semantic_review': 'not performed',
        }
        for arm in ['GX', 'FX']:
            idx = indexes[qid, arm]
            result = idx['result']
            response = result.get('response') or {}
            h = response.get('receipt_sha256')
            raw = receipt(h)
            card['extraction_views'][arm] = {
                'model_structured_output': result.get('value'),
                'schema_error': result.get('schema_error'),
                'http_status': response.get('http_status'),
                'usage': raw.get('usage') if raw else None,
                'admitted': idx['admitted'], 'rejected': idx['rejected'],
                'graph_projection': projection(idx['admitted']),
                'receipt_sha256': h, 'receipt_path': checked.get(h),
                'admission_boundary': 'exact span/contract binding, not independently validated semantic truth',
            }
            totals['extraction_views'] += 1
            totals[arm + '_admitted'] += len(idx['admitted'])
            totals[arm + '_rejected'] += len(idx['rejected'])
        for item in sorted(grouped[qid], key=lambda i: (i['workload'], i['condition'])):
            o = outcomes[item['episode_id']]
            assert item['correctness_input']['published_answer'] == q['original_answer']
            answer_response = (o.get('answer') or {}).get('response') or {}
            answer_receipt = receipt(answer_response.get('receipt_sha256'))
            packet = None
            if answer_receipt:
                messages = [m for m in answer_receipt['request']['messages'] if m['role'] == 'tool']
                assert len(messages) == 1
                packet = json.loads(messages[0]['content'])
                assert canonical_sha(packet) == o['packet_sha256']
                totals['verified_delivered_packets'] += 1
            else:
                assert o['status'] != 'completed'
            intent = o.get('intent') or {}
            intent_response = intent.get('response') or {}
            receipt(intent_response.get('receipt_sha256'))
            db_reads = []
            for span in spans[item['episode_id']]:
                if span['component'] != 'graph_read':
                    continue
                h = span.get('receipt_sha256')
                r = receipt(h)
                if r:
                    branches = [{k: v for k, v in b.items() if k != 'rows'} for b in r.get('branches', [])]
                    db_reads.append({'receipt_sha256': h, 'receipt_path': checked[h],
                                     'arm': r.get('arm'), 'records': r.get('records'),
                                     'query_wall_ms': r.get('query_wall_ms'), 'branches': branches})
            path = {
                'episode_id': item['episode_id'], 'workload': item['workload'],
                'condition': item['condition'], 'query_profile': o['profile'],
                'status': o['status'], 'namespace': o['namespace'],
                'model_intent': intent.get('value'),
                'actual_answer': item['correctness_input']['candidate_answer'],
                'answer_schema_error': (o.get('answer') or {}).get('schema_error'),
                'delivered_evidence': packet, 'packet_sha256': o.get('packet_sha256'),
                'graph_reads': db_reads,
                'automated_score': scores[item['item_id']],
                'automated_judgments': {s: grades[item['item_id'], s] for s in ['correctness', 'faithfulness']},
                'intent_receipt_sha256': intent_response.get('receipt_sha256'),
                'answer_receipt_sha256': answer_response.get('receipt_sha256'),
                'answer_receipt_path': checked.get(answer_response.get('receipt_sha256')),
                'intent_usage': (intent_response.get('response') or {}).get('usage'),
                'answer_usage': (answer_response.get('response') or {}).get('usage'),
                'trace_spans': spans[item['episode_id']],
            }
            card['answer_paths'].append(path)
            totals['answer_paths'] += 1
        cards.append(card)

    assert len(cards) == 317 and totals['extraction_views'] == 634 and totals['answer_paths'] == 1585
    assert totals['GX_admitted'] == 2274 and totals['FX_admitted'] == 2330
    # Deterministic gzip, retaining a complete machine-readable evaluator-only join.
    with (OUT / 'questions_with_gold_graphs_and_answers.jsonl.gz').open('wb') as target:
        with gzip.GzipFile(fileobj=target, mode='wb', mtime=0) as stream:
            for card in cards:
                stream.write((json.dumps(card, ensure_ascii=False, separators=(',', ':')) + '\n').encode())

    index_rows = []
    summary_rows = []
    for card in cards:
        qid = card['question_id']
        page = ['<!doctype html><meta charset="utf-8"><title>' + qid + '</title>', STYLE,
                '<a href="../index.html">All 317 questions</a>',
                '<h1>' + html.escape(card['question']) + '</h1>',
                '<p>' + html.escape(qid + ' · ' + card['category']) + '</p>',
                '<p class="notice">Evaluator-only review. Gold is the published FinDER answer, not independently validated source-entailment truth. Graph proposals and automated judgments can be wrong. No new model calls.</p>',
                '<h2>1. Published ground truth</h2><div class="gold">' + html.escape(card['published_ground_truth']['answer']).replace('\n', '<br>') + '</div>',
                '<h2>2. Original selected references</h2>']
        for ref in card['selected_references']:
            page.append(details(ref['id'] + ' · ' + ref.get('collection', ''), ref))
        page.append('<h2>3. Graph indexing: actual GX and FX results</h2><p>Physical storage: Issuer → CONTAINS → Entity → HAS_ASSERTION → Assertion → SUPPORTED_BY → Source. Business predicates and qualifiers reside inside assertion payloads. The projection below derives from admitted records; actual readback is in each answer path.</p>')
        for arm, view in card['extraction_views'].items():
            page.append('<h3>' + arm + ': ' + str(len(view['admitted'])) + ' admitted / ' + str(len(view['rejected'])) + ' rejected</h3>')
            columns = ['id', 'entity_key', 'subject', 'attribute', 'value', 'period', 'scope', 'unit', 'concept_key']
            page.append('<table><thead><tr>' + ''.join('<th>' + c + '</th>' for c in columns) + '</tr></thead><tbody>')
            for a in view['admitted']:
                page.append('<tr>' + ''.join('<td>' + html.escape(str(a.get(c, ''))) + '</td>' for c in columns) + '</tr>')
            page.append('</tbody></table>')
            page.append(details(arm + ' full output, rejected proposals, support spans, graph projection and receipt', view))
        page.append('<h2>4. Five answer paths: actual tool evidence, output and grading</h2>')
        for path in card['answer_paths']:
            label = path['workload'] + ' / ' + path['condition']
            score = path['automated_score']
            page.append('<h3>' + label + '</h3><p>Status: ' + html.escape(path['status']) + ' · Native correctness: ' + html.escape(score['correctness'] or 'UNKNOWN') + ' · Evidence faithfulness: ' + html.escape(score['faithfulness'] or 'UNDEFINED / UNKNOWN') + '</p>')
            page.append(pretty(path['actual_answer']))
            page.append(details('Exact delivered evidence packet', path['delivered_evidence']))
            page.append(details('Automated judgments and claim-level reasons', path['automated_judgments']))
            page.append(details('Actual queries, parameters, usage, trace IDs and receipt locators', {k: v for k, v in path.items() if k not in {'delivered_evidence', 'actual_answer', 'automated_judgments'}}))
            summary_rows.append({'question_id': qid, 'category': card['category'], 'question': card['question'],
                                 'published_ground_truth': card['published_ground_truth']['answer'],
                                 'workload': path['workload'], 'condition': path['condition'], 'status': path['status'],
                                 'GX_admitted': len(card['extraction_views']['GX']['admitted']),
                                 'FX_admitted': len(card['extraction_views']['FX']['admitted']),
                                 'candidate_answer_json': json.dumps(path['actual_answer'], ensure_ascii=False),
                                 'native_correctness': score['correctness'], 'evidence_faithfulness': score['faithfulness'],
                                 'review_card': 'cases/' + qid + '.html'})
        (OUT / 'cases' / (qid + '.html')).write_text('\n'.join(page))
        index_rows.append('<tr><td><a href="cases/' + qid + '.html">' + qid + '</a></td><td>' + html.escape(card['category']) + '</td><td>' + html.escape(card['question']) + '</td><td>' + str(len(card['selected_references'])) + '</td></tr>')
    (OUT / 'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>FinDER: question-to-evidence review</title>''' + STYLE + '''
<h1>317 questions: ground truth, graphs and answers</h1>
<p class="notice">Evaluator-only companion. Each card joins the unchanged published answer, full selected references, both extraction views, all five answer paths, actual delivered tool packets, automated judgments and trace provenance. Independent human semantic reviews: 0. Not distributed in the public code repository.</p>
<p>Use the browser's search to find an issuer, question or category. Graph views are actual admitted proposals, including failures; they are not extraction gold. No new inference or DB benchmark was run.</p>
<table><thead><tr><th>ID</th><th>Category</th><th>Original question</th><th>References</th></tr></thead><tbody>''' + '\n'.join(index_rows) + '</tbody></table>')
    with (OUT / 'question_answer_summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(summary_rows)
    inputs = [INPUT / 'questions.evaluator-only.json', INPUT / 'sources.json', RUN / 'index-outcomes.jsonl',
              RUN / 'outcomes.jsonl', RUN / 'spans.jsonl', JUDGE / 'items.evaluator-only.json',
              JUDGE / 'grades.jsonl', JUDGE / 'repair-grades.jsonl', REPORT / 'scores.csv']
    audit = {'questions': len(cards), **dict(totals), 'verified_receipt_files': len(checked),
             'new_model_calls': 0, 'new_database_calls': 0, 'publicly_exported': False,
             'independent_human_reviews': 0, 'input_sha256': {str(p): sha(p) for p in inputs},
             'output_sha256': {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file() and p.name != 'audit.json'}}
    (OUT / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: v for k, v in audit.items() if not k.endswith('sha256')}))


if __name__ == '__main__':
    main()
