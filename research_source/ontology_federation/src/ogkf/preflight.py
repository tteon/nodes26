"""Offline artifact audit and cached analysis. Never admits paid experiment runs."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
from typing import Literal
import uuid

import pyarrow as pa
import pyarrow.parquet as pq

from .models import Strict
from .context_frame.artifacts import write_json
from .context_frame.candidates import Candidate
from .context_frame.compiler import file_hash
from .context_frame.lifecycle import verify_origins
from .context_frame.models import Bundle
from .ds01.contracts import Dataset, ShiftManifest
from .ds01.corpus import validate_manifest
from .ds01.evaluation import dependence_groups, paired_report, strategy_shift_interaction


class ReadinessCheck(Strict):
    check_id: str
    scope: str
    status: Literal['ready', 'limited', 'blocked']
    observation: str
    implication: str


def verify_inventory(root, name):
    root = Path(root).resolve()
    records = json.loads((root/name).read_text())
    if not records:
        raise ValueError('empty artifact inventory')
    for rel, expected in records.items():
        path = (root/rel).resolve()
        if not path.is_relative_to(root):
            raise ValueError('inventory path escapes artifact root')
        if not path.is_file() or file_hash(path) != expected:
            raise ValueError('artifact checksum mismatch: '+rel)
    return len(records)


def payloads(path):
    return [json.loads(r['payload_json']) for r in pq.ParquetFile(path).read().to_pylist()]


def derive_checks(facts, env):
    def check(key, scope, status, observed, implication):
        return ReadinessCheck(check_id=key, scope=scope, status=status,
                              observation=observed, implication=implication)
    return [
        check('native_qa', 'published-answer comparison', 'ready',
              f"{facts['qa_candidates']} nonblank native QA candidates; answers and references preserved",
              'Native QA gold remains usable; automated semantic agreement needs a fixed calibrated comparator.'),
        check('source_coverage', 'population context claims',
              'ready' if facts['binding_backlog'] == 0 else 'limited',
              f"{facts['global_sample_ready']}/{facts['global_sample']} population-sampled questions source-bound; {facts['binding_backlog']} pending",
              'The source-ready cohort is a selected population; do not generalize its weights to all FinDER.'),
        check('context_gold', 'semantic evidence sufficiency / contextual leakage',
              'ready' if facts['semantic_gold_cases'] == facts['frame_cases'] and facts['frame_cases'] else 'blocked',
              f"{facts['semantic_gold_cases']}/{facts['frame_cases']} cases have independently reviewed sufficient sets",
              'Rule-generated frames are silver. Source support, scope and allowed composition need independent adjudication.'),
        check('native_reasoning', 'reasoning subgroup comparison', 'limited',
              f"Reasoning=True occurs in {facts['reasoning_categories']}/{facts['categories']} categories",
              'Compare within supported categories; no fully crossed category-by-reasoning causal claim.'),
        check('snapshot_parity', 'placement and routing pilot', 'ready',
              f"{facts['corpus_conditions']} corpora preserve gold; {facts['snapshot_parity']} native snapshot parity receipts",
              'Application-level federation, fixed executor/index and one global evidence budget.'),
        check('intervention_isolation', 'isolated distribution effects',
              'limited' if facts['imperfect_conditions'] else 'ready',
              f"{facts['imperfect_conditions']}/{facts['corpus_conditions']} interventions have measured nuisance drift",
              'Report intervention assignment contrasts and drift; measured divergence alone does not identify causality.'),
        check('inferential_scope', 'population effects / noninferiority', 'limited',
              f"Native pilot: {facts['pilot_questions']} queries, {facts['dependence_components']} known company/source components, {facts['seeds']} corpus seeds",
              'These are engineering observations. More questions sharing sources do not create independent clusters; predefine margins and a power/sensitivity analysis.'),
        check('graph_semantics', 'business-fact / multi-hop graph claim', 'limited',
              'Current DS01 uses source-metadata assertions, Python BM25 ranking and fixed database fetches',
              'This tests retrieval/routing/placement. Add reviewed business assertions and join witnesses before a business-neighborhood or multi-hop claim.'),
        check('mara_environment', 'new live MARA calls', 'ready' if env.get('MARA_API_KEY') else 'blocked',
              'Environment credential is present' if env.get('MARA_API_KEY') else 'Environment credential is absent',
              'Refresh same-account capabilities and bounded admission settings before live calls; credential values are never recorded.'),
        check('calibration', 'automated semantic quality conclusions', 'blocked',
              'No independent evaluator-calibration receipt is supplied to this audit',
              'Validate frozen evaluator against human judgments; no semantic success claim from automatic judges alone.'),
    ]


def audit(candidates, campaign, output, *, environ=None):
    candidates, campaign, output = Path(candidates), Path(campaign), Path(output)
    if output.exists():
        raise FileExistsError(output)
    inventory_count = verify_inventory(candidates, 'delivery-sha256.json')
    campaign_inventory_count = verify_inventory(campaign, 'delivery-sha256.json')
    summary = json.loads((candidates/'summary.json').read_text())
    catalog = [Candidate.model_validate(r) for r in
               pq.ParquetFile(candidates/'candidate_catalog.private.parquet').read().to_pylist()]
    if len(catalog) != len({r.query_id for r in catalog}) or len(catalog) != summary['native_questions']:
        raise ValueError('catalog population mismatch')
    frame = candidates/'expanded_frame'
    bundle = Bundle.model_validate_json((frame/'frame.private.json').read_text())
    cohort = Dataset.model_validate_json((frame/'cohort.dataset.private.json').read_text())
    origin_checks = verify_origins(bundle, cohort)
    public = json.loads((frame/'public/queries.json').read_text())
    if public != [q.model_dump() for q in cohort.queries]:
        raise ValueError('public input boundary mismatch')
    overview = json.loads((campaign/'campaign.json').read_text())
    # Campaign order is explicit provenance; select its last recorded native run.
    native = [r for r in overview['runs'] if (campaign/'runs'/r['run_id']/'graph-capabilities.json').exists()]
    if not native:
        raise ValueError('no recorded native campaign run')
    run = campaign/'runs'/native[-1]['run_id']
    old_manifest = json.loads((run/'manifest.json').read_text())
    config = old_manifest['config']
    dataset = Dataset.model_validate_json((campaign/'input.evaluator-only.json').read_text())
    if dataset.dataset_hash != old_manifest['dataset_hash']:
        raise ValueError('native frozen dataset hash mismatch')
    corpora = {r['condition_id']: r for r in payloads(run/'corpus_manifests.parquet')}
    shifts = []
    for row in payloads(run/'distribution_measurements.parquet'):
        row = {**row, 'corpus': corpora[row['condition_id']]['corpus']}
        shift = ShiftManifest.model_validate({k: row[k] for k in ShiftManifest.model_fields})
        validate_manifest(dataset, shift.corpus, config['document_budget'])
        shifts.append(shift)
    lookup = {s.condition_id: s for s in shifts}
    traces = payloads(run/'retrieval_traces.parquet')
    scores = payloads(run/'evaluation_results.rescored-v2.parquet')
    by_sample = {r['sample_id']: r for r in traces}
    if (len(by_sample) != len(traces) or len(scores) != len(traces)
            or len({r['sample_id'] for r in scores}) != len(scores)
            or {r['sample_id'] for r in scores} != set(by_sample)):
        raise ValueError('retrieval/score lineage cardinality mismatch')
    for row in scores:
        parent = by_sample[row['sample_id']]
        for key in ('query_id', 'condition_id', 'trace_id', 'strategy', 'seed', 'graph_snapshot_id'):
            if row[key] != parent[key]:
                raise ValueError('retrieval/score lineage mismatch')
    for row in traces:
        shift = lookup[row['condition_id']]
        if (row['graph_snapshot_id'] != shift.corpus.graph_snapshot_id
                or len(row['evidence_ids']) > config['global_top_k']
                or row['context_bytes'] > config['context_byte_budget']):
            raise ValueError('retrieval snapshot/budget mismatch')
    capabilities = json.loads((run/'graph-capabilities.json').read_text())
    parities = capabilities['snapshot_parity']
    if (not all(r['exact_readback_parity'] for r in parities)
            or {r['graph_snapshot_id'] for r in parities} != {s.corpus.graph_snapshot_id for s in shifts}):
        raise ValueError('native snapshot parity incomplete')
    analysis = {'parent_experiment_run_id': old_manifest['experiment_run_id'],
        'source_evaluation_file_sha256': file_hash(run/'evaluation_results.rescored-v2.parquet'),
        'new_inference_calls': 0, 'new_graph_calls': 0,
        'paired_changes': paired_report(scores, shifts, dataset),
        'strategy_shift_model': strategy_shift_interaction(scores, shifts, dataset, config['target_entity_id'])}
    qa_candidates = sum(bool(r.question.strip()) and r.has_native_answer for r in catalog)
    if qa_candidates != summary['qa_candidates']:
        raise ValueError('native QA eligibility count mismatch')
    facts = {'native_questions': len(catalog), 'qa_candidates': qa_candidates,
        'categories': len({r.category for r in catalog}),
        'reasoning_true': sum(r.native_reasoning for r in catalog),
        'reasoning_categories': len({r.category for r in catalog if r.native_reasoning}),
        'frame_cases': len(bundle.cases),
        'semantic_gold_cases': sum(c.annotation_status == 'reviewed_gold' and c.support_status == 'independently_reviewed' for c in bundle.cases),
        'global_sample': summary['global_review_sample'], 'global_sample_ready': summary['global_sample_source_bound_ready'],
        'binding_backlog': sum(1 for line in (candidates/'binding_backlog.private.jsonl').read_text().splitlines() if line.strip()),
        'pilot_questions': len(dataset.queries), 'dependence_components': len(set(dependence_groups(dataset).values())),
        'seeds': len({s.seed for s in shifts}), 'corpus_conditions': len(shifts),
        'imperfect_conditions': sum(s.causal_isolation == 'imperfect' for s in shifts),
        'snapshot_parity': len(parities), 'native_retrieval_episodes': len(traces),
        'retrieval_failures': sum(bool(r['error']) for r in traces),
        'verified_inventory_files': inventory_count, 'verified_campaign_files': campaign_inventory_count,
        'origin_checks': origin_checks}
    checks = derive_checks(facts, os.environ if environ is None else environ)
    run_id, trace_id = str(uuid.uuid4()), str(uuid.uuid4())
    code = {str(p.relative_to(Path(__file__).parent)): file_hash(p) for p in sorted(Path(__file__).parent.rglob('*.py'))}
    report = {'version': 'finder-final-preflight-v1', 'experiment_run_id': run_id, 'trace_id': trace_id,
        'created_at': datetime.now(timezone.utc).isoformat(), 'facts': facts,
        'checks': [c.model_dump() for c in checks], 'status_counts': dict(Counter(c.status for c in checks)),
        'new_http_calls': 0, 'new_graph_calls': 0, 'code_hashes': code,
        'input_hashes': {'candidate_delivery': file_hash(candidates/'delivery-sha256.json'),
                         'campaign': file_hash(campaign/'campaign.json'), 'native_manifest': file_hash(run/'manifest.json')},
        'scope': 'Offline readiness evidence, not live-run admission or human calibration certification.'}
    output.mkdir(parents=True)
    write_json(output/'audit.json', report)
    write_json(output/'cached-analysis-v3.json', analysis)
    rows = [{'experiment_run_id': run_id, 'trace_id': trace_id, **c.model_dump()} for c in checks]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output/'readiness.parquet')
    if pq.ParquetFile(output/'readiness.parquet').read().to_pylist() != rows:
        raise ValueError('readiness Parquet readback failed')
    esc = html.escape
    body = ''.join('<tr>'+''.join('<td>'+esc(str(c[k]))+'</td>' for k in
        ('scope', 'status', 'observation', 'implication'))+'</tr>' for c in report['checks'])
    page = "<!doctype html><html lang='ko'><meta charset='utf-8'><title>FinDER 최종 실행 점검</title><style>body{max-width:1200px;margin:2rem auto;font:16px/1.6 sans-serif;padding:1rem;color:#17334a}table{border-collapse:collapse}td,th{border:1px solid #bccbd5;padding:.7rem;text-align:left;vertical-align:top}</style><h1>FinDER 실행 전 점검</h1><p>검증된 실행 기록과 독립 gold 준비 상태를 구분합니다. 새 HTTP·DB 호출은 0회입니다.</p><table><tr><th>범위</th><th>상태</th><th>확인한 사실</th><th>해석·보완</th></tr>"+body+'</table></html>'
    (output/'index.html').write_text(page)
    write_json(output/'sha256.json', {p.name: file_hash(p) for p in sorted(output.iterdir()) if p.is_file()})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidates', type=Path, default=Path('reports/finder_candidates_v1'))
    parser.add_argument('--campaign', type=Path, default=Path('reports/ds01_campaign'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.candidates, args.campaign, args.output)
    print(json.dumps({'output': str(args.output), 'facts': result['facts'], 'status_counts': result['status_counts']}, indent=2))


if __name__ == '__main__':
    main()
