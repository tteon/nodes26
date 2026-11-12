"""Count frozen graph builds offline and reconcile with native write counters."""
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = Path(os.environ['NODES26_FROZEN_RUN'])
OUT = Path(os.environ.get('NODES26_AUDIT_OUT', str(ROOT / 'results/local-graph-audit')))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lines(path):
    return [json.loads(s) for s in path.open() if s.strip()]


def main():
    jobs = lines(RUN / 'index-outcomes.jsonl'); assert len(jobs) == 634
    namespace_jobs = {'idx-' + j['question_id'] + '-' + j['profile']: j for j in jobs}
    assert len(namespace_jobs) == 634
    observed = {}; write_hashes = []
    for span in lines(RUN / 'spans.jsonl'):
        if span.get('component') != 'index_write':
            continue
        h = span['receipt_sha256']; path = RUN / 'receipts' / (h + '.json')
        assert sha(path) == h
        receipt = json.loads(path.read_text())
        if receipt['namespace'] not in namespace_jobs:
            continue
        assert receipt['database'] == 'neo4j' and receipt['namespace'] not in observed
        observed[receipt['namespace']] = receipt; write_hashes.append(h)
    assert len(observed) == 634
    rows = []; conditions = {}
    all_keys = set()
    for profile, label in [('GX', 'General definitions + context rules'), ('FX', 'Financial-ontology definitions + context rules')]:
        total = Counter(); unique_keys = set(); subjects = set(); source_ids = set(); issuers = set()
        for job in jobs:
            if job['profile'] != profile:
                continue
            records = job['admitted']; namespace = 'idx-' + job['question_id'] + '-' + profile
            entities = {r['entity_key'] for r in records}; companies = {r['cik'] for r in records}
            sources = {r['source_id'] for r in records}; ids = {r['id'] for r in records}
            assert len(companies) <= 1 and len(ids) == len(records)
            counts = {'attempted_builds': 1, 'nonempty_builds': int(bool(records)),
                'empty_builds': int(not records), 'issuer_nodes': len(companies),
                'entity_nodes': len(entities), 'assertion_nodes': len(ids), 'source_nodes': len(sources),
                'CONTAINS': len({(r['cik'], r['entity_key']) for r in records}),
                'HAS_ASSERTION': len({(r['entity_key'], r['id']) for r in records}),
                'SUPPORTED_BY': len({(r['id'], r['source_id']) for r in records})}
            counts['total_nodes'] = sum(counts[k] for k in ['issuer_nodes', 'entity_nodes', 'assertion_nodes', 'source_nodes'])
            counts['total_relationships'] = sum(counts[k] for k in ['CONTAINS', 'HAS_ASSERTION', 'SUPPORTED_BY'])
            receipt = observed[namespace]
            assert counts['total_nodes'] == receipt['nodes_created']
            assert counts['total_relationships'] == receipt['relationships_created']
            assert len(records) == receipt['record_count']
            total.update(counts); unique_keys |= entities; all_keys |= entities; source_ids |= sources; issuers |= companies
            subjects |= {(r['cik'], ' '.join(r['subject'].casefold().split())) for r in records}
            rows.append({'question_id': job['question_id'], 'condition': label, 'namespace': namespace, **counts})
        assert total['attempted_builds'] == 317
        conditions[profile] = {'condition': label, **dict(total),
            'distinct_contextual_entity_keys_without_namespace': len(unique_keys),
            'distinct_cik_normalized_subject_surface_keys': len(subjects),
            'distinct_referenced_sources_with_admitted_assertions': len(source_ids),
            'distinct_issuers_with_admitted_assertions': len(issuers)}
    totals = {k: conditions['GX'][k] + conditions['FX'][k] for k in ['attempted_builds', 'nonempty_builds', 'empty_builds', 'entity_nodes', 'issuer_nodes', 'assertion_nodes', 'source_nodes', 'CONTAINS', 'HAS_ASSERTION', 'SUPPORTED_BY', 'total_nodes', 'total_relationships']}
    assert totals['total_nodes'] == 6693 and totals['total_relationships'] == 10364
    assert totals['nonempty_builds'] == 459 and totals['assertion_nodes'] == 4604
    report = {'scope': 'Main317-question extracted-graph campaign only; historical run totals, not current live DB inventory.',
        'questions': 317, 'input_reference_passages': 329, 'input_issuers': 42,
        'subgraph_definition': 'One question × one extraction condition × its namespace. Nonempty builds are connected through their issuer anchor. Empty/failed builds are retained.',
        'entity_identity': 'Physical entity identity is namespace + entity_key. The entity_key hashes CIK + stored context + casefolded whitespace-normalized subject; this is not independently resolved real-world identity.',
        'relationship_definition': 'Native storage edges CONTAINS/HAS_ASSERTION/SUPPORTED_BY. Business attributes and qualifiers are assertion payload fields, not distinct native relationship types.',
        'conditions': conditions, 'totals_with_namespace_isolation': totals,
        'distinct_contextual_entity_keys_union_across_conditions': len(all_keys),
        'native_write_receipts_verified': 634, 'every_build_matches_native_node_and_relationship_creation_counters': True,
        'index_ledger_sha256': sha(RUN / 'index-outcomes.jsonl'),
        'verified_write_receipt_hash_list_sha256': hashlib.sha256('\n'.join(sorted(write_hashes)).encode()).hexdigest(),
        'copies_excluded': ['Original-reference graph views and their physical context copies', 'Earlier formation pilots', 'Copied driver fixture', 'Later screenshot/native-continuity reconstructions'],
        'same_project_global_canonical_entity_total': None, 'new_model_calls': 0, 'new_database_calls': 0}
    OUT.mkdir(exist_ok=True)
    (OUT / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    with (OUT / 'per_build.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fields = [('Graph construction attempts', 'attempted_builds'), ('Nonempty graph builds', 'nonempty_builds'),
        ('Empty graph builds', 'empty_builds'), ('Entity nodes', 'entity_nodes'), ('Issuer anchor nodes', 'issuer_nodes'),
        ('Admitted assertion nodes', 'assertion_nodes'), ('Source nodes', 'source_nodes'), ('All nodes', 'total_nodes'),
        ('CONTAINS edges', 'CONTAINS'), ('HAS_ASSERTION edges', 'HAS_ASSERTION'), ('SUPPORTED_BY edges', 'SUPPORTED_BY'), ('All native relationships', 'total_relationships')]
    doc = ['# Graph size: a fixed denominator and native write receipts', '', report['scope'], '',
        'The main cohort has317unchanged questions,329selected reference passages and42issuers. The two extraction conditions produce634assigned graph builds. Counts below preserve namespace isolation; repeated aliases across questions remain separate physical nodes.', '',
        '| Count | General definitions + rules | Financial ontology + rules | Total |', '| --- | ---: | ---: | ---: |']
    for label, key in fields:
        doc.append(f"| {label} | {conditions['GX'][key]:,} | {conditions['FX'][key]:,} | {totals[key]:,} |")
    doc += ['', '## Identity and relationship meanings', '', report['subgraph_definition'], '', report['entity_identity'], '',
        'Without namespace, distinct contextual entity keys number497and576by condition; their union is767. CIK + normalized subject surface keys number457and532. These are different keys, not conflicting counts and not canonical entity-resolution results.', '',
        report['relationship_definition'], '', 'A semantic entity is not the same counting unit as every graph node. Issuer, assertion and source nodes also occupy the database. More admitted assertions do not establish more correct facts.', '',
        '## Verification and scope', '', 'All634original index-write receipts were hash-verified. Every build exactly matches the recorded nodes_created, relationships_created and record_count values. No model or database call was needed for this audit. Disposable historical databases were torn down; these are historical construction totals, not a claim that6693nodes are currently live in the capture DB.', '',
        'The earlier two-model pilot materialized8model/profile views with267exact-span proposals and remains a separate cohort. The FMC screenshot fixture is one later reconstruction with18nodes/29edges; the selected cited view has7nodes/7edges. Do not sum these copies into the main benchmark or label them unique world facts.', '',
        '## What the FMC screenshot demonstrates', '',
        'The same FMC entity links to separate dated legal-claim and shipping/handling accounting statements. Three selected statements cite two original passages. Conditions and source lineage survive in explicit assertion nodes and payloads, allowing a reader to inspect the evidence that reached the model. The picture alone does not demonstrate federation, a contamination reduction, semantic correctness or a faster query. Actual model-input parity and runtime comparisons have separate receipts.', '']
    (OUT / 'RESULTS_EN.md').write_text('\n'.join(doc))
    print(json.dumps({'conditions': conditions, 'totals': totals, 'verified_native_writes': 634}, indent=2))


if __name__ == '__main__':
    main()
