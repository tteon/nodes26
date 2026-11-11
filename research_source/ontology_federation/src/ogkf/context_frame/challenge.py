"""Source-only graph challenge expansion; native gold never enters extraction.

GraphRAG-Bench-inspired candidates are silver, not new semantic ground truth.
This module reuses the native census, stratified sampler and source-unit parser.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field

from ..ds01.contracts import Dataset, Document, Entity, Query, text_hash, tokens
from ..ds01.retrieval import CONCEPTS
from ..models import Strict, digest
from .candidates import ExpansionConfig, census, stratified_sample
from .compiler import document_units
from .models import Span

VERSION = "finder-graph-challenge-v1"
ONTOLOGY = "source-observation-proposals-v1"
PREDICATES = ["QUOTED_STATEMENT", "TABLE_ROW_OBSERVATION"]
PAPER = "https://arxiv.org/html/2506.02404v3"


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def select_ids(rows, identifiers, key):
    """An explicit engineering cohort; never infer selection from answer values."""
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("duplicate selected identifier")
    lookup = {key(row): row for row in rows}
    if len(lookup) != len(rows) or not set(identifiers) <= lookup.keys():
        raise ValueError("unknown or nonunique selected identifier")
    return [lookup[identifier] for identifier in identifiers]


def load_lines(path):
    rows = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    if len({x["id"] for x in rows}) != len(rows):
        raise ValueError("duplicate input identifier")
    return {x["id"]: x for x in rows}


class Proposal(Strict):
    assertion_id: str
    subject_id: str
    predicate: str
    object_value: str
    document_id: str
    evidence_id: str
    source_hash: str
    start: int
    end: int
    quote: str
    domain_id: str
    evidence_span_ids: list[str] = Field(min_length=1)
    table_row: dict | None = None
    extraction_method: str = "deterministic_source_units_v1"
    validation_status: str = "exact_spans_only"
    semantic_support: str = "UNKNOWN"
    entity_identity_status: str = "source_local_literal_not_real_world_identity"
    temporal_scope: None = None
    business_scope: None = None
    unit: None = None


class BusinessProposal(Proposal):
    qualifier_proposals: dict[str, str | None]


def domain_for(text):
    """Source-text partition only. FinDER categories and answers are not inputs."""
    words = Counter(tokens(text))
    scores = {d: sum(words[w] for w in concepts) for d, concepts in CONCEPTS.items()}
    # Explicit lexical partition, not a gold business-domain judgment.
    return max(sorted(scores), key=lambda d: (scores[d], d == "finance_reporting"))


def extract_source(document, max_rows=8, max_statements=2):
    spans, cues, rows, _ = document_units(document)
    for span in spans:
        span.verify(document)
    by_id = {s.span_id: s for s in spans}
    entities, proposals = {}, []
    for row in rows[:max_rows]:
        label = by_id[row.label_span_id]
        # A row label is a local literal concept, not the source publisher/company.
        eid = digest([ONTOLOGY, document.document_id, "row_label", label.quote])
        entities[eid] = Entity(
            entity_id=eid,
            canonical_name=label.quote,
            aliases=[label.quote],
            identity_status="lexical_only",
        )
        ids = [row.label_span_id, *row.header_span_ids, *row.value_span_ids]
        start, end = min(by_id[s].start for s in ids), max(by_id[s].end for s in ids)
        proposals.append(
            Proposal(
                assertion_id=digest([ONTOLOGY, row.row_id]),
                subject_id=eid,
                predicate="TABLE_ROW_OBSERVATION",
                object_value=json.dumps(row.values),
                document_id=document.document_id,
                evidence_id=document.evidence_id,
                source_hash=document.content_hash,
                start=start,
                end=end,
                quote=document.text[start:end],
                domain_id=document.domain_id,
                evidence_span_ids=ids,
                table_row=row.model_dump(),
            )
        )
    for span in [s for s in spans if s.kind == "statement"][:max_statements]:
        eid = digest([ONTOLOGY, document.document_id, "source_unit"])
        entities[eid] = Entity(
            entity_id=eid,
            canonical_name="Source text unit",
            aliases=[],
            identity_status="lexical_only",
        )
        proposals.append(
            Proposal(
                assertion_id=digest([ONTOLOGY, span.span_id]),
                subject_id=eid,
                predicate="QUOTED_STATEMENT",
                object_value=span.quote,
                document_id=document.document_id,
                evidence_id=document.evidence_id,
                source_hash=document.content_hash,
                start=span.start,
                end=span.end,
                quote=span.quote,
                domain_id=document.domain_id,
                evidence_span_ids=[span.span_id],
            )
        )
    return list(entities.values()), proposals, spans


def verify_proposals(documents, proposals, spans):
    docs = {d.document_id: d for d in documents}
    by_id = {s.span_id: s for s in spans}
    if len(by_id) != len(spans) or len({a.assertion_id for a in proposals}) != len(
        proposals
    ):
        raise ValueError("duplicate extraction identifier")
    for span in spans:
        span.verify(docs[span.document_id])
    for a in proposals:
        d = docs[a.document_id]
        if (
            not 0 <= a.start < a.end <= len(d.text)
            or a.source_hash != d.content_hash
            or a.quote != d.text[a.start : a.end]
            or a.domain_id != d.domain_id
            or a.semantic_support != "UNKNOWN"
            or any(by_id[s].document_id != d.document_id for s in a.evidence_span_ids)
        ):
            raise ValueError("source proposal escaped its evidence or review status")


def generated_questions(selected, native, proposals, spans):
    by_doc = defaultdict(list)
    span_map = {s.span_id: s for s in spans}
    for a in proposals:
        if a.table_row:
            by_doc[a.document_id].append(a)
    used_rows, result = set(), []
    for item in selected:
        parent = native[item["query_id"]]
        candidates = [a for d in parent["source_ids"] for a in by_doc[d]]
        for a in sorted(candidates, key=lambda x: x.assertion_id):
            row = a.table_row
            years = row["year_candidates"]
            if (
                a.assertion_id in used_rows
                or len(years) < 2
                or len(set(years)) != len(years)
                or len(years) != len(row["values"])
            ):
                continue
            used_rows.add(a.assertion_id)
            label = span_map[row["label_span_id"]].quote
            text = (
                f"Use the entity and reporting context of this inquiry: {parent['question']} "
                f"Now compare the source table's '{label}' observations for {years[0]} and {years[-1]}. "
                "State each value with its period, unit and scope. Calculate the difference only if "
                "the column alignment, unit and scope are compatible; otherwise explain the missing information."
            )
            result.append(
                {
                    "query_id": "challenge-"
                    + digest([VERSION, parent["id"], a.assertion_id, text])[:24],
                    "text": text,
                    "origin": "generated_table_context_candidate",
                    "parent_ids": [parent["id"]],
                    "operation_candidate": "compare_then_conditional_difference",
                    "question_type": "open_ended",
                    "source_ids": [a.document_id],
                    "assertion_ids": [a.assertion_id],
                    "annotation_status": "silver",
                    "semantic_answer": None,
                    "sufficient_evidence_gold": None,
                    "graph_necessity": "unverified",
                    "review_reason": "Source row is literal; column alignment, entity, units and task semantics need independent review.",
                }
            )
            break
    # Pair different native inquiries sharing an actual reference. This adds an
    # evidence-integration task, not an invented fact or an asserted causal join.
    owners = defaultdict(list)
    for item in selected:
        for rid in set(native[item["query_id"]]["source_ids"]):
            owners[rid].append(item["query_id"])
    used_pairs = set()
    for rid in sorted(owners):
        ids = sorted(set(owners[rid]))
        if len(ids) < 2:
            continue
        pair = tuple(ids[:2])
        if pair in used_pairs:
            continue
        used_pairs.add(pair)
        result.append(
            {
                "query_id": "challenge-" + digest([VERSION, "paired", pair])[:24],
                "text": "Answer both inquiries separately, preserving each question's entity, period and scope. "
                "Then state whether their evidence establishes a comparable basis; do not infer a causal relationship. "
                + " ".join(
                    f"({i + 1}) {native[q]['question']}" for i, q in enumerate(pair)
                ),
                "origin": "generated_paired_scope_candidate",
                "parent_ids": list(pair),
                "operation_candidate": "independent_answers_then_compatibility",
                "question_type": "open_ended",
                "source_ids": sorted(
                    {r for q in pair for r in native[q]["source_ids"]}
                ),
                "assertion_ids": [],
                "annotation_status": "silver",
                "semantic_answer": None,
                "sufficient_evidence_gold": None,
                "graph_necessity": "unverified",
                "review_reason": "Published component QA is preserved; the added comparison has no independent context/answer gold.",
            }
        )
    issuers = defaultdict(list)
    for item in selected:
        ids = item.get("verified_source_issuer_ids", [])
        if len(ids) == 1:
            issuers[ids[0]].append(item)
    for issuer, items in sorted(issuers.items()):
        ordered = sorted(items, key=lambda x: x["query_id"])
        pairs = [
            (a, b)
            for i, a in enumerate(ordered)
            for b in ordered[i + 1 :]
            if a["category"] != b["category"]
            and set(a["source_ids"]).isdisjoint(b["source_ids"])
        ]
        if not pairs:
            continue
        a, b = pairs[0]
        pair = [a["query_id"], b["query_id"]]
        text = (
            "These inquiries have source references from the same reporting issuer; "
            "this does not make every subject or scope identical. Answer each inquiry separately, "
            "then explain which entity, period and business-scope conditions would be needed "
            "to combine their evidence. Do not infer causality or legal authority from shared issuer identity. "
            + " ".join(f"({i + 1}) {native[q]['question']}" for i, q in enumerate(pair))
        )
        result.append(
            {
                "query_id": "challenge-" + digest([VERSION, issuer, pair, text])[:24],
                "text": text,
                "origin": "generated_cross_context_candidate",
                "parent_ids": pair,
                "operation_candidate": "multi_source_answers_then_context_compatibility",
                "question_type": "open_ended",
                "source_ids": sorted(set(a["source_ids"] + b["source_ids"])),
                "assertion_ids": [],
                "annotation_status": "silver",
                "semantic_answer": None,
                "sufficient_evidence_gold": None,
                "graph_necessity": "unverified",
                "review_reason": "Common source issuer is verified; assertion identity, compatibility and added-answer semantics remain unreviewed.",
            }
        )
    return result


def prepare(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    paths = {
        k: Path(getattr(args, k))
        for k in ("parquet", "native", "public", "references", "pool", "bindings")
    }
    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()
    }
    native, public, references = (
        load_lines(paths[k]) for k in ("native", "public", "references")
    )
    pool = Dataset.model_validate_json(paths["pool"].read_text())
    bindings = json.loads(paths["bindings"].read_text())["sources"]
    candidates, population = census(
        pq.read_table(paths["parquet"]).to_pylist(),
        native,
        public,
        references,
        pool,
        bindings,
    )
    eligible = [
        c
        for c in candidates
        if c.has_native_answer
        and c.source_ids
        and all(references[r]["text"].strip() for r in c.source_ids)
    ]
    config = ExpansionConfig(
        seed=args.seed,
        review_per_category=args.per_category,
        reasoning_true_target_per_category=args.per_category // 2,
        max_annotation_cases=6000,
        same_issuer_candidates=0,
        other_issuer_candidates=0,
        max_rows_per_case=8,
    )
    if getattr(args, "all_native", False):
        selected = [
            {
                **c.model_dump(),
                "selection_scope": "all_reference_ready_native",
                "inclusion_probability": 1.0,
                "analysis_weight": 1.0,
            }
            for c in sorted(eligible, key=lambda c: c.query_id)
        ]
        shortfalls = []
    else:
        selected, shortfalls = stratified_sample(
            eligible, config, scope="global_primary_review"
        )
    docs, entities, proposals, spans = [], {}, [], []
    bound_docs = {d.document_id: d for d in pool.documents}
    for rid, raw in sorted(references.items()):
        text = raw["text"]
        if not text.strip():
            continue
        sha = text_hash(text)
        previous = bound_docs.get(rid)
        d = Document(
            document_id=rid,
            document_version=sha,
            text=text,
            content_hash=sha,
            source_publisher=previous.source_publisher if previous else None,
            source_uri=previous.source_uri if previous else None,
            domain_id=domain_for(text),
            evidence_id=digest([rid, sha, 0, len(text)]),
            entity_mentions={},
            relation_counts={},
            token_count=len(tokens(text)),
            metadata_status="source-only lexical partition; benchmark-reference text; no reviewed fact semantics",
        )
        es, assertions, source_spans = extract_source(d)
        entities.update({e.entity_id: e for e in es})
        proposals.extend(assertions)
        used = {s for a in assertions for s in a.evidence_span_ids}
        spans.extend(s for s in source_spans if s.span_id in used)
        d = d.model_copy(
            update={"relation_counts": dict(Counter(a.predicate for a in assertions))}
        )
        docs.append(d)
    verify_proposals(docs, proposals, spans)
    generated = generated_questions(selected, native, proposals, spans)
    private = []
    for s in selected:
        n = native[s["query_id"]]
        private.append(
            {
                "query_id": n["id"],
                "text": n["question"],
                "origin": "native_unchanged",
                "parent_ids": [n["id"]],
                "source_ids": n["source_ids"],
                "native_answer": n["answer"],
                "category": n["category"],
                "native_reasoning": n["native_reasoning"],
                "native_type": n["native_type"],
                "shared_reference_component_id": s["shared_reference_component_id"],
                "inclusion_probability": s["inclusion_probability"],
                "analysis_weight": s["analysis_weight"],
                "annotation_status": "published_QA_only",
            }
        )
    private.extend(generated)
    public_rows = [
        Query(query_id=q["query_id"], text=q["text"]).model_dump() for q in private
    ]
    population["reference_ready_qa"] = len(eligible)
    manifest = {
        "version": VERSION,
        "input_hashes": hashes,
        "seed": args.seed,
        "framework": PAPER,
        "adaptation": "hierarchy / task / explicit evidence-operation obligations; not official benchmark compatibility",
        "population": population,
        "sample_categories": dict(Counter(x["category"] for x in selected)),
        "sample_reasoning": dict(Counter(str(x["native_reasoning"]) for x in selected)),
        "selection_mode": "all_reference_ready_native"
        if getattr(args, "all_native", False)
        else "stratified_category_sample",
        "selected_native_questions": len(selected),
        "generated_silver_questions": len(generated),
        "sample_shared_reference_components": len(
            {x["shared_reference_component_id"] for x in selected}
        ),
        "source_documents": len(docs),
        "source_only_proposals": len(proposals),
        "proposal_predicates": dict(Counter(a.predicate for a in proposals)),
        "source_bound_publishers": len(
            {d.source_publisher for d in docs if d.source_publisher}
        ),
        "independently_reviewed_business_facts": 0,
        "independently_reviewed_generated_questions": 0,
        "new_http_calls": 0,
        "shortfalls": shortfalls,
        "ontology_version": ONTOLOGY,
        "source_only_indexer": True,
        "native_qa_labels_in_public_inputs": False,
        "unknown_semantics_remain_unknown": True,
        "limits": [
            "Reference-ready retrospective benchmark, not an unseen holdout.",
            "Generated variants share source/parent dependence and are not independent population expansion.",
            "Literal statements/table rows are extraction proposals, not validated enterprise triples or Grounded Fact F1.",
            "Lexical provider partitions are experimental domains, not native category or provider-selection gold.",
        ],
    }
    jsonl(output / "documents.jsonl", [d.model_dump() for d in docs])
    jsonl(output / "entities.jsonl", [e.model_dump() for e in entities.values()])
    jsonl(output / "proposals.jsonl", [a.model_dump() for a in proposals])
    jsonl(output / "spans.jsonl", [s.model_dump() for s in spans])
    jsonl(output / "queries.public.jsonl", public_rows)
    jsonl(output / "queries.private.jsonl", private)
    jsonl(output / "review.pending.jsonl", generated)
    save(output / "manifest.json", manifest)
    snapshot = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in output.glob("*.jsonl")
    }
    save(output / "snapshot-hashes.json", snapshot)
    review = "<!doctype html><meta charset='utf-8'><title>FinDER graph challenge review</title><h1>Silver candidates: independent review pending</h1>"
    for q in generated:
        review += (
            "<article><h2>"
            + html.escape(q["query_id"])
            + "</h2><p>"
            + html.escape(q["text"])
            + "</p><p>Parents: "
            + html.escape(", ".join(q["parent_ids"]))
            + "</p><details><summary>Source references</summary>"
        )
        for rid in q["source_ids"]:
            review += (
                "<h3>"
                + html.escape(rid)
                + "</h3><pre style='white-space:pre-wrap'>"
                + html.escape(references[rid]["text"])
                + "</pre>"
            )
        review += (
            "</details><p>UNKNOWN: "
            + html.escape(q["review_reason"])
            + "</p></article>"
        )
    (output / "review.html").write_text(review)
    if any(
        hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p)]
        for p in paths.values()
    ):
        raise ValueError("input changed during preparation")
    return {
        k: manifest[k]
        for k in (
            "version",
            "selected_native_questions",
            "generated_silver_questions",
            "source_documents",
            "source_only_proposals",
            "new_http_calls",
        )
    }


class ChallengeLedger:
    """Append receipt journal before proceeding; batch Parquet without relabeling DS01."""

    def __init__(self, root, run_id):
        self.root, self.run_id, self.rows = Path(root), run_id, {}
        self.date = datetime.now(timezone.utc).date().isoformat()

    def emit(self, table, **row):
        # Canonicalize once before retaining the row: JSON turns tuples into
        # lists, and caller-owned nested objects must not mutate saved receipts.
        row = json.loads(
            json.dumps(
                {"experiment_run_id": self.run_id, **row},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        self.rows.setdefault(table, []).append(row)
        with (self.root / (table + ".jsonl")).open("a") as stream:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        return json.loads(json.dumps(row, ensure_ascii=False))

    def verify(self):
        counts = {}
        for table, rows in self.rows.items():
            raw = [
                json.loads(x)
                for x in (self.root / (table + ".jsonl")).read_text().splitlines()
            ]
            if raw != rows:
                raise ValueError("journal readback mismatch")
            folder = (
                self.root
                / "parquet"
                / "experiment=finder_graph_challenge"
                / ("date=" + self.date)
            )
            folder.mkdir(parents=True, exist_ok=True)
            records = [
                {
                    "experiment_run_id": self.run_id,
                    "query_id": r.get("query_id"),
                    "trace_id": r.get("trace_id"),
                    "payload_json": json.dumps(r, ensure_ascii=False, sort_keys=True),
                }
                for r in rows
            ]
            schema = pa.schema(
                [
                    (k, pa.string())
                    for k in (
                        "experiment_run_id",
                        "query_id",
                        "trace_id",
                        "payload_json",
                    )
                ]
            )
            path = folder / (table + ".parquet")
            pq.write_table(pa.Table.from_pylist(records, schema=schema), path)
            if [
                json.loads(r["payload_json"])
                for r in pq.ParquetFile(path).read().to_pylist()
            ] != rows:
                raise ValueError("Parquet readback mismatch")
            counts[table] = len(rows)
        return counts


def load_snapshot(root):
    root = Path(root)
    hashes = json.loads((root / "snapshot-hashes.json").read_text())
    for name, sha in hashes.items():
        if (
            Path(name).name != name
            or hashlib.sha256((root / name).read_bytes()).hexdigest() != sha
        ):
            raise ValueError("snapshot changed or invalid file path")

    def lines(name):
        return [json.loads(x) for x in (root / name).read_text().splitlines()]

    docs = [Document.model_validate(r) for r in lines("documents.jsonl")]
    proposals = [
        (
            BusinessProposal
            if r.get("extraction_method") == "llm_business_relation_draft"
            else Proposal
        ).model_validate(r)
        for r in lines("proposals.jsonl")
    ]
    entities = [Entity.model_validate(r) for r in lines("entities.jsonl")]
    verify_proposals(
        docs, proposals, [Span.model_validate(r) for r in lines("spans.jsonl")]
    )
    snapshot = digest(hashes)
    dataset = SimpleNamespace(
        documents=docs,
        assertions=proposals,
        entities=entities,
        ontology_version=ONTOLOGY,
        ontology_relations=sorted({a.predicate for a in proposals}),
        dataset_hash=snapshot,
    )
    return dataset, lines("queries.private.jsonl"), snapshot


def admission():
    return {
        "MARA_API_KEY_present": bool(os.environ.get("MARA_API_KEY")),
        "MARA_ADMISSION_LOCK_present": bool(os.environ.get("MARA_ADMISSION_LOCK")),
        "new_http_calls": 0,
        "model_capabilities": "current_probe_required",
        "semantic_quality_claim": "independent review and evaluator calibration required",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for name in (
        "parquet",
        "native",
        "public",
        "references",
        "pool",
        "bindings",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--all-native", action="store_true")
    p.add_argument("--per-category", type=int, choices=range(1, 101), default=100)
    p.add_argument("--seed", type=int, default=20261031)
    r = sub.add_parser("retrieve")
    for name in ("snapshot", "output"):
        r.add_argument("--" + name, type=Path, required=True)
    r.add_argument("--limit", type=int, default=16)
    r.add_argument("--graph", choices=("memory", "local-dozer"), default="memory")
    r.add_argument("--top-k", type=int, choices=range(1, 11), default=5)
    r.add_argument("--context-bytes", type=int, default=24000)
    r.add_argument("--query-id", dest="query_ids", action="append")
    live = sub.add_parser("llm")
    live.add_argument("--snapshot", type=Path, required=True)
    live.add_argument("--output", type=Path, required=True)
    live.add_argument("--stage", choices=("index", "answer"), required=True)
    live.add_argument("--model", required=True)
    live.add_argument("--max-calls", type=int, required=True)
    live.add_argument("--limit", type=int, default=2)
    live.add_argument("--offset", type=int, default=0)
    live.add_argument("--coverage-contract", action="store_true")
    live.add_argument(
        "--span-mode", choices=("offsets", "unique-quote"), default="offsets"
    )
    live.add_argument("--max-facts", type=int, choices=range(1, 17), default=16)
    live.add_argument(
        "--index-max-tokens", type=int, choices=(2048, 3072, 4096), default=2048
    )
    live.add_argument("--retrieval-run", type=Path)
    live.add_argument(
        "--capability-run",
        type=Path,
        help="reuse verified live probe receipts, at most1hour old",
    )
    live.add_argument("--document-id", dest="document_ids", action="append")
    live.add_argument("--query-id", dest="query_ids", action="append")
    attach = sub.add_parser("attach-index")
    for name in ("snapshot", "index-run", "output"):
        attach.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args)
    elif args.command == "retrieve":
        from .challenge_graph import run_retrieval

        result = run_retrieval(args)
    elif args.command == "llm":
        from ..environment import load_project_environment
        from .challenge_llm import run_live

        load_project_environment()
        result = run_live(args)
    else:
        from .challenge_llm import attach_index

        result = attach_index(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
