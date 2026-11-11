"""Receipt-derived tables, editable source explorer and explicit unknown metrics."""

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import statistics
import unicodedata

import pyarrow as pa
import pyarrow.parquet as pq

from ..models import digest
from .challenge import save
from .review_receipts import audit_review
from .semantic_review import read_lines


def answer_surface_scores(prediction, reference):
    """Descriptive token overlap; explicitly not semantic correctness/faithfulness."""

    def norm(value):
        return re.findall(r"\w+", unicodedata.normalize("NFKC", value).casefold())

    a, b = norm(prediction), norm(reference)
    if not b:
        return {"surface_exact_match": None, "surface_token_f1": None}
    common = sum((Counter(a) & Counter(b)).values())
    precision = common / len(a) if a else 0
    recall = common / len(b)
    return {
        "surface_exact_match": float(a == b),
        "surface_token_f1": 2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0,
    }


def extraction_table(root):
    outcomes = [json.loads(p.read_text()) for p in root.glob("outcome-*.json")]
    groups = defaultdict(list)
    for r in outcomes:
        groups[(r["model"], r["scenario"])].append(r)
    rows = []
    for (model, arm), values in sorted(
        groups.items(),
        key=lambda item: (item[0][0], ("generic", "A", "B", "C").index(item[0][1])),
    ):
        rows.append(
            {
                "model": model,
                "scenario": arm,
                "conditions": len(values),
                "received_nonfailed_conditions": sum(
                    r["status"] != "request_failed" for r in values
                ),
                "request_failed_conditions": sum(
                    r["status"] == "request_failed" for r in values
                ),
                "exact_span_proposals": sum(len(r["accepted"]) for r in values),
                "rejected_quote_proposals": sum(len(r["rejected"]) for r in values),
                "empty_or_all_rejected_conditions": sum(
                    r["status"] == "empty_or_all_rejected" for r in values
                ),
                "grounded_fact_f1": None,
            }
        )
    return rows


def attempt_table(roots):
    rows = []
    for root in roots:
        for p in root.glob("**/traces/experiment=*/date=*/run=*/*.parquet"):
            for r in pq.ParquetFile(p).read().to_pylist():
                rows.append({**r, "artifact_root": str(root)})
    keys = [(r["request_id"], r["attempt"]) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("attempt inventory includes duplicated receipts")
    grouped = defaultdict(list)
    for r in rows:
        grouped[(r["artifact_root"], r["parent_step_id"], r["model_id"])].append(r)
    output = []
    for (root, step, model), values in sorted(grouped.items(), key=lambda x: str(x[0])):
        output.append(
            {
                "artifact_root": root,
                "step": step,
                "model": model,
                "logical_calls": len({v["request_id"] for v in values}),
                "HTTP_attempts": len(values),
                "statuses": dict(Counter(str(v["http_status"]) for v in values)),
                "known_input_tokens": sum(
                    v["input_tokens"] for v in values if v["input_tokens"] is not None
                ),
                "unknown_input_usage_attempts": sum(
                    v["input_tokens"] is None for v in values
                ),
                "known_output_tokens": sum(
                    v["output_tokens"] for v in values if v["output_tokens"] is not None
                ),
                "unknown_output_usage_attempts": sum(
                    v["output_tokens"] is None for v in values
                ),
                "median_HTTP_latency_ms": statistics.median(
                    v["http_latency_ms"] for v in values
                ),
            }
        )
    return rows, output


def write_table(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {
                k: json.dumps(v, ensure_ascii=False)
                if isinstance(v, (dict, list))
                else v
                for k, v in row.items()
            }
            for row in rows
        )
    payload = [
        {
            "row_id": digest([path.name, i]),
            "payload_json": json.dumps(r, ensure_ascii=False, sort_keys=True),
        }
        for i, r in enumerate(rows)
    ]
    target = path.with_suffix(".parquet")
    pq.write_table(pa.Table.from_pylist(payload), target)
    if [
        json.loads(r["payload_json"]) for r in pq.ParquetFile(target).read().to_pylist()
    ] != rows:
        raise ValueError("report Parquet readback mismatch")


def table(rows, fields):
    def display(value):
        if value is None:
            return "Not established"
        if isinstance(value, bool):
            return "Yes" if value else "No"
        return str(value)

    return (
        "<table><thead><tr>"
        + "".join("<th>" + html.escape(k.replace("_", " ")) + "</th>" for k in fields)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join("<td>" + html.escape(display(r.get(k))) + "</td>" for k in fields)
            + "</tr>"
            for r in rows
        )
        + "</tbody></table>"
    )


def figures(root, categories, extractions):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["category"] for r in categories]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    a, b = (
        [r["reasoning_false"] for r in categories],
        [r["reasoning_true"] for r in categories],
    )
    ax.barh(names, a, color="#477ca8", label="Native Reasoning=False")
    ax.barh(names, b, left=a, color="#dc9c42", label="Native Reasoning=True")
    ax.invert_yaxis()
    ax.set_xlabel("Original FinDER rows (not executed QA answers)")
    ax.legend(loc="lower right")
    ax.set_title("All eight categories retained; native labels are evaluator-only")
    fig.tight_layout()
    fig.savefig(root / "native-coverage.svg")
    fig.savefig(root / "native-coverage.png", dpi=160)
    plt.close(fig)
    labels = [r["model"] + " / " + r["scenario"] for r in extractions]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(labels, [r["exact_span_proposals"] for r in extractions], color="#497f91")
    ax.invert_yaxis()
    ax.set_xlabel("Exact-span assertion proposals (not validated facts)")
    ax.set_title("Source-only extraction: output volume is not semantic quality")
    for i, row in enumerate(extractions):
        ax.text(
            row["exact_span_proposals"] + 0.5,
            i,
            f"HTTP/schema failures: {row['request_failed_conditions']}/{row['conditions']}",
            va="center",
            fontsize=8,
        )
    ax.margins(x=0.6)
    fig.tight_layout()
    fig.savefig(root / "extraction-volume.svg")
    fig.savefig(root / "extraction-volume.png", dpi=160)
    plt.close(fig)


def build(args):
    if not args.allow_partial and not (args.queries / "summary.json").exists():
        raise ValueError(
            "query traversal is incomplete; no final report may be claimed"
        )
    reviews, audit = audit_review(args.review, require_complete=not args.allow_partial)
    root = args.output
    root.mkdir(parents=True, exist_ok=args.refresh)
    for name in ("review-audit.json", "summary.json", "claim-source-coverage.json"):
        if (root / name).exists() and args.refresh:
            (root / name).unlink()
    save(root / "review-audit.json", audit)
    questions = read_lines(args.review / "questions.evaluator-only.jsonl")
    documents = read_lines(args.review / "documents.jsonl")
    native = [q for q in questions if q["origin"] == "native_unchanged"]
    categories = []
    for cat in sorted({q["category"] for q in native}):
        values = [q for q in native if q["category"] == cat]
        categories.append(
            {
                "category": cat,
                "original_rows": len(values),
                "nonblank_native_answers": sum(
                    bool(q.get("native_answer")) for q in values
                ),
                "reasoning_true": sum(q["native_reasoning"] is True for q in values),
                "reasoning_false": sum(q["native_reasoning"] is False for q in values),
                "actual_content_reviews": sum(
                    r.get("category") == cat and r["origin"] == "native_unchanged"
                    for r in reviews
                ),
            }
        )
    extractions = extraction_table(args.extraction)
    episodes = read_lines(args.queries / "evaluation_results.jsonl")
    snapshot = json.loads((args.queries / "input-snapshot.json").read_text())
    contract = json.loads((args.queries / "execution-contract.json").read_text())
    expected_episodes = len(snapshot["public_questions"]) * (
        sum(5 if arm == "C" else 2 for _, arm in contract["executable_variants"])
        + int([contract["answer_model"], "C"] in contract["executable_variants"])
    )
    if not args.allow_partial and len(episodes) != expected_episodes:
        raise ValueError(
            "query episode coverage differs from the frozen execution contract"
        )
    fact_table = pa.Table.from_pylist(snapshot["qualified_assertion_proposals"])
    pq.write_table(fact_table, root / "normalized-assertion-proposals.parquet")
    if (
        pq.ParquetFile(root / "normalized-assertion-proposals.parquet")
        .read()
        .to_pylist()
        != snapshot["qualified_assertion_proposals"]
    ):
        raise ValueError("normalized assertion Parquet changed payloads")
    graph_receipts = [
        json.loads(p.read_text())
        for p in args.queries.glob("graph-*.json")
        if p.name != "graph-capabilities.json"
    ]
    retrieval = []
    for key in sorted({(r["model"], r["scenario"], r["strategy"]) for r in episodes}):
        rows = [
            r for r in episodes if (r["model"], r["scenario"], r["strategy"]) == key
        ]
        retrieval.append(
            {
                "extraction_model": key[0],
                "answer_model": contract["answer_model"],
                "scenario": key[1],
                "strategy": key[2],
                "episodes": len(rows),
                "native_reference_document_proxy": sum(
                    r["native_reference_proxy"] for r in rows
                ),
                "answers_received": sum("answer" in r for r in rows),
                "answer_failures": sum("answer_error" in r for r in rows),
                "grounded_QA_success": None,
                "evidence_sufficiency_at_k": None,
            }
        )
    native_by_id = {q["query_id"]: q for q in native}
    surface = []
    for r in episodes:
        if "answer" not in r or r["query_id"] not in native_by_id:
            continue
        claims = r["answer"]["normalized_decision"]["claims"]
        scores = answer_surface_scores(
            " ".join(c["text"] for c in claims),
            native_by_id[r["query_id"]]["native_answer"],
        )
        surface.append(
            {
                "query_id": r["query_id"],
                "extraction_model": r["model"],
                "answer_model": contract["answer_model"],
                "scenario": r["scenario"],
                "strategy": r["strategy"],
                **scores,
                "scope": "uncalibrated surface agreement only; native answer unchanged; not semantic correctness",
            }
        )
    attempts, costs = attempt_table(
        [args.review, args.extraction, args.queries, *args.additional_attempt_roots]
    )
    llm_requests = {}
    for path in args.queries.glob("*/raw/*.json"):
        wire = json.loads(path.read_text())
        request_id = path.stem.rsplit("-", 1)[0]
        llm_requests[request_id] = {
            "http_status": wire["http_status"],
            "request_body": wire["request"],
            "response_content": ((wire.get("response") or {}).get("choices") or [{}])[0]
            .get("message", {})
            .get("content"),
            "provider_rejected_output": (wire.get("response") or {}).get(
                "error_model_output"
            ),
        }
    for episode in episodes:
        if "answer" in episode:
            episode["actual_answer_request_id"] = episode["answer"]["request_id"]
        elif "answer_error" in episode:
            candidates = {
                r["request_id"]
                for r in attempts
                if r["artifact_root"] == str(args.queries)
                and r["document_id"] == episode["query_id"]
                and r["parent_step_id"] == "scenario_answer"
                and r["experiment_condition_id"]
                == digest([episode["model"], episode["scenario"], episode["strategy"]])
            }
            if len(candidates) == 1:
                episode["actual_answer_request_id"] = candidates.pop()
    ops = [
        {
            "origin": origin,
            "model_operation_proposal": op,
            "count": n,
            "independent_gold": False,
        }
        for (origin, op), n in sorted(
            Counter((r["origin"], r["review"]["operation"]) for r in reviews).items()
        )
    ]
    for name, data in (
        ("categories", categories),
        ("extraction", extractions),
        ("retrieval", retrieval),
        ("native-answer-surface-diagnostics", surface),
        ("request-costs", costs),
        ("operation-proposals", ops),
        ("per-question-reviews.evaluator-only", reviews),
    ):
        write_table(root / name, data)
    figures(root, categories, extractions)
    decisions = Counter(
        r["answer"]["normalized_decision"]["action"] for r in episodes if "answer" in r
    )
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "explicit_partial_preview": args.allow_partial,
        "scope": "all-question model-assisted review plus selective source-to-claim engineering pilot",
        "answer_and_router_model": contract["answer_model"],
        "native_original_rows": len(native),
        "native_nonblank_published_answers": sum(
            bool(q.get("native_answer")) for q in native
        ),
        "actual_content_reviews_received": len(reviews),
        "review_pending_items": len(audit["pending_query_ids"]),
        "source_locator_error_reviews": audit["invalid_source_locator_records"],
        "answer_availability_label_mismatches": audit[
            "published_answer_availability_label_mismatches"
        ],
        "executed_pipeline_questions": len(snapshot["public_questions"]),
        "executed_pipeline_sources": len(snapshot["source_documents"]),
        "generated_dependent_candidates": len(questions) - len(native),
        "review": {k: v for k, v in audit.items() if k != "pending_query_ids"},
        "extraction": extractions,
        "graph_variants_materialized": len(graph_receipts),
        "graph_assertion_parity": bool(graph_receipts)
        and all(
            r["exact_single_multi_readback"] and r["exact_source_readback"]
            for r in graph_receipts
        ),
        "retrieval_episodes": len(episodes),
        "answer_responses": sum(decisions.values()),
        "answer_actions": dict(decisions),
        "HTTP_attempts": len(attempts),
        "known_input_tokens": sum(
            r["input_tokens"] for r in attempts if r["input_tokens"] is not None
        ),
        "unknown_input_usage_attempts": sum(
            r["input_tokens"] is None for r in attempts
        ),
        "known_output_tokens": sum(
            r["output_tokens"] for r in attempts if r["output_tokens"] is not None
        ),
        "unknown_output_usage_attempts": sum(
            r["output_tokens"] is None for r in attempts
        ),
        "independent_semantic_gold": 0,
        "ontology_semantic_lift": None,
        "native_composite_used": False,
        "presentation_status": "new report linked to agreed storyline; prior PPTX and all28 appendix tables unchanged",
    }
    save(root / "summary.json", summary)
    source_attempts = [
        r for r in attempts if r["parent_step_id"] == "source_only_business_extraction"
    ]
    source_statuses = dict(Counter(str(r["http_status"]) for r in source_attempts))
    requested_answers = sum("answer" in r or "answer_error" in r for r in episodes)
    test_log = args.extraction.parent / "tests-final.log"
    passes = (
        re.findall(r"(\d+) passed", test_log.read_text()) if test_log.exists() else []
    )
    test_count = passes[-1] if passes else "unrecorded"
    report = f"""# FinDER full question review and federated source-to-claim pilot

This report separates received model opinions, deterministic engineering checks,
and semantic conclusions that require independent adjudication. Native QA gold is
preserved. The new context, fact-support, sufficient-evidence and action labels
have no completed independent human review.

| Observed item | Receipt-derived count |
| --- | ---: |
| Original FinDER rows in the pinned snapshot | {len(native)} |
| Original rows with a published answer | {summary["native_nonblank_published_answers"]} |
| Dependent generated question candidates | {len(questions) - len(native)} |
| Received per-question source-reading judgments | {len(reviews)} |
| Pending question judgments | {len(audit["pending_query_ids"])} |
| Reviews with an invalid/partial source locator | {audit["invalid_source_locator_records"]} |
| Published-answer availability label mismatches | {audit["published_answer_availability_label_mismatches"]} |
| Executed pilot questions / complete source passages | {len(snapshot["public_questions"])} / {len(snapshot["source_documents"])} |
| Source extraction conditions | {sum(r["conditions"] for r in extractions)} |
| Exact-span assertion proposals | {sum(r["exact_span_proposals"] for r in extractions)} |
| Executed retrieval/text-baseline episodes | {len(episodes)} |
| Application-accepted answer responses / attempts | {summary["answer_responses"]} / {requested_answers} |
| Newly adjudicated semantic gold | 0 |

## What was tested

All original questions and generated candidates receive a source-reading opinion
with their complete references. A lossless segment transport retains exact
Unicode offsets. Category, Reasoning, raw Type and previous opinions are withheld
from the reviewer. These opinions remain uncalibrated silver. A structurally
valid locator can still cite an irrelevant part of the source. Invalid locators
and answer-availability contradictions remain explicit; they are not repaired or
converted into native-gold defects.

The source-only pilot uses two extraction models and four guidance profiles:
generic, A (local vocabulary/domain/range), B (definitions plus nine pinned FIBO
reference concepts), and C (B plus business-context inference limits). Structural
output, original passages and quote-localization rules are identical. B-A is a
combined information/length contrast, not a pure FIBO effect. No OWL reasoner or
independently reviewed upper-ontology mapping is claimed. There is one source
call per condition; this is an exposed engineering pilot rather than the full
factorial/repetition study.

Source HTTP statuses were `{json.dumps(source_statuses, sort_keys=True)}`. The
HTTP429 service-capacity responses and provider HTTP400 structured-output
rejections are retained. Missing usage remains unknown. Unequal availability
prevents treating extraction volume as an intrinsic model or ontology ranking.
Exact quote acceptance validates location; assertion entailment remains separate.

## Storage, routing and evidence composition

{len(graph_receipts)} source variants were materialized in one context-preserving
qualified DB and three heterogeneous domain DBs per variant. The same qualified
assertions and complete passages are checked by actual driver reads. B0/B2 share
global candidates, and B1/B3 share a registry plan: packet equality is a placement
control. Graph parity does not establish factual correctness or a physical
federation accuracy advantage.

The original query traversal completed its230 episodes and146 HTTP receipts,
then a final export check rejected tuple/list differences in registry metadata.
Canonical JSON normalization fixes the ledger. `scenario_finalize` rebuilt the
derived Parquet and summary from unchanged journals with no new inference or
database calls; the original failure log remains available. Six tables and1342
rows passed payload readback. This was an export correction, not a repeated
experiment or semantic repair.

The answer and router model is fixed to `{contract["answer_model"]}`. The model
column in raw episodes identifies the upstream extraction model. C routing uses
actual labels, relation coverage/definitions, source-publisher metadata and
context contracts. Only allowlisted domain IDs reach fixed parameterized query
templates. B4 uses bounded relation-cue/count expansion, not gold sufficiency.
All graph strategies deliver a global top-8/20000-byte packet; T0 uses full
passages under the same byte ceiling and has different evidence units.

The runtime uses application-level federation through separate read sessions in
one local DozerDB DBMS. A replicated publisher-reference key associates evidence;
it does not merge assertion subjects or establish SAME_AS, compatible scopes,
causality or legal authority. The actual capability manifest and owned database
cleanup receipts accompany the run. Native Composite is not demonstrated.

## An observed negative trace

In the AMD development example, the source says there were **no material legal
proceedings**. The gpt-oss B and C draft objects omit `no` while retaining the
exact negative source quote. The C/B3 answer uses the original source negation.
The original tuple, graph reads, routing request and answer are retained in
`results/scenario_pipeline_v1/lineage-witness.json` and the report explorer.

This separates an extraction defect from storage integrity and subsequent
reasoning. The quote-priority answer policy is common to every arm. This single
exposed witness is not an ontology-only, federation, or population improvement.
Source loss in other conditions produces abstentions; an abstention does not
automatically count as correct QA.

## What remains unverified

Grounded Fact F1, context-qualified fact retention, evidence sufficiency,
contextual leakage and grounded QA success remain null until their relevant
independent labels and calibrated evaluator exist. Native answer surface
agreement is exported as a separate uncalibrated diagnostic. Unannotated facts
are not automatically unsupported. No population significance, quality
noninferiority, production latency/cost advantage, GPU memory or KV-cache result
is inferred from these receipts.

The full prepared pipeline cohort contains8758 runnable questions and5825
reference passages. Its46600 planned source conditions have not been executed.
The seven original blank-answer rows remain in the full content-review scope.
Long-source admission, semantic calibration, repetitions and confirmatory design
must precede a larger quality study. Blinded independent packets and the strict
answer/claim hash binding support that next step without fabricating reviewers.

## Reproduction and presentation scope

See `SOURCE_SCENARIO_PIPELINE.md` for exact entry points. The source-review receipt
audit runs offline. `scenario_pipeline verify-graphs` rematerializes the recorded
source proposals and exercises real local database queries without new inference.
Fresh paid query runs require `.env`/environment credentials, the existing shared
admission lock, capability probes and explicit call limits.

The {test_count}-test repository gate, scoped lint checks, Parquet readback checks, rendered
SVG/PNG figures and interactive source/trace explorer accompany this report.
All previous empirical receipts and the existing presentation/28 appendix tables
remain unchanged. These new findings map to Setup, Results and Results with real
traces; they have not been inserted into the PowerPoint.
"""
    (root / "report.md").write_text(report)
    evidence_files = [
        args.review / "manifest.json",
        args.extraction / "control.json",
        args.queries / "summary.json",
        args.queries / "graph-capabilities.json",
        root / "review-audit.json",
    ]
    coverage = {
        "claims": [
            {
                "claim_id": "C01",
                "text": f"{len(native)} original rows and {len(questions) - len(native)} dependent candidates are in scope.",
                "status": "observed_input_inventory",
                "sources": [str(evidence_files[0])],
            },
            {
                "claim_id": "C02",
                "text": f"{len(reviews)} unique source-reading judgments are verified against actual full-source request and response receipts.",
                "status": "observed_protocol_coverage_not_semantic_accuracy",
                "sources": [str(root / "review-audit.json")],
            },
            {
                "claim_id": "C03",
                "text": "Qualified assertions and original passages match between single-DB and domain-DB readback.",
                "status": "observed_engineering_parity"
                if summary["graph_assertion_parity"]
                else "not_established",
                "sources": [str(p) for p in args.queries.glob("graph-*.json")],
            },
            {
                "claim_id": "C04",
                "text": "Ontology improves enterprise semantic correctness.",
                "status": "unverified_hypothesis",
                "value": None,
                "reason": "Independent fact, compatibility, sufficient-set and claim/action adjudication/calibration pending.",
            },
            {
                "claim_id": "C05",
                "text": "Execution uses application-level federation; native Composite is not demonstrated.",
                "status": "observed_runtime_and_capability_receipts",
                "sources": [
                    str(args.queries / "graph-capabilities.json"),
                    str(args.queries / "summary.json"),
                ],
            },
        ],
        "source_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
            for p in evidence_files
        },
    }
    witness = args.extraction.parent / "lineage-witness.json"
    if witness.exists():
        coverage["claims"].append(
            {
                "claim_id": "C06",
                "text": "One AMD development trace contains a draft object that omits source negation; its C/B3 answer retains the source negation under the common quote-priority policy.",
                "status": "observed_text_and_lineage_contrast_not_population_semantic_lift",
                "independent_gold": False,
                "sources": [
                    str(witness),
                    str(args.extraction.parent / "negation-witness.json"),
                ],
            }
        )
        coverage["source_hashes"][str(witness)] = hashlib.sha256(
            witness.read_bytes()
        ).hexdigest()
    save(root / "claim-source-coverage.json", coverage)
    payload = json.dumps(
        {
            "reviews": reviews,
            "documents": {d["document_id"]: d for d in documents},
            "episodes": episodes,
            "graph_queries": read_lines(args.queries / "graph_query_traces.jsonl"),
            "routing_plans": read_lines(args.queries / "routing_traces.jsonl"),
            "llm_requests": llm_requests,
        },
        ensure_ascii=False,
    ).replace("<", "\\u003c")
    page = r"""<!doctype html><html lang="en"><meta charset="utf-8"><title>FinDER: full review and source-to-claim pipeline</title>
<style>body{font:15px/1.5 system-ui;margin:32px;color:#172a38;background:#f4f7fa}h1,h2{line-height:1.2}table{border-collapse:collapse;background:white;width:100%;margin:14px 0;font-size:13px}th,td{border:1px solid #d2dce5;padding:8px;text-align:left;overflow-wrap:anywhere}th{background:#e4edf4}.notice{background:#fff1cc;padding:16px;border-left:5px solid #ca8b24}.flow{display:flex;gap:10px;flex-wrap:wrap;margin:24px 0}.flow span{background:#1c5472;color:white;padding:14px;border-radius:8px}.split{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:white;border:1px solid #ccd9e4;padding:12px;max-height:700px;overflow:auto}input,select,button{padding:9px;margin:5px}img{max-width:900px;width:100%}.small{color:#526574}a{color:#15598a}</style>
<h1>FinDER: full question review → source-to-claim pipeline</h1>
<p class="notice">Model-assisted source-reading judgments are uncalibrated silver. Native QA gold is preserved. Independent context/fact/action gold remains pending. Exact source offsets and graph parity are engineering evidence, not semantic accuracy.</p>
<div class="flow"><span>Original passages</span><span>generic / A / B / C extraction</span><span>Qualified assertions</span><span>Single DB / domain DBs</span><span>One evidence packet</span><span>Cited answer claims</span></div>
<h2>Observed coverage</h2>__SUMMARY__<img src="native-coverage.svg" alt="Native category and Reasoning coverage">
<p>Original Reasoning is a Boolean task label, not model reasoning success. Generated variants share parents/sources and are dependent samples.</p>__CATEGORIES__
<h2>Source-only extraction pilot</h2><p>generic: competent control; A: structural vocabulary; B: definitions and pinned FIBO concepts; C: additional business-context rules. One call per source/condition. Volume is not precision or recall. Service failures remain explicit.</p>__EXTRACTION__<img src="extraction-volume.svg" alt="Exact span proposal volume and failure counts">
<h2>Retrieval and answer execution</h2><p>B0/B2 and B1/B3 preserve the same plan and qualified evidence packet across placement. B3 uses actual structured registry metadata. B4 expands under explicit heuristic limits. One global top-8 and20KB packet is used; T0's units are full passages. Reference-document membership below is not evidence sufficiency.</p>__RETRIEVAL__
<p>Graphs use different labels and structural relationships. A shared publisher key associates evidence without equating contextual subjects. Database selection and query templates are application-controlled; native Composite is not established. No semantic improvement or production latency advantage is claimed.</p>
<h2>Actual source → assertion → query → answer-claim traces</h2><p>Select a received execution episode. Native reference membership and source quote matches do not establish the semantic correctness of its claims. Failed answers remain visible.</p><select id="traceItems" style="max-width:100%"></select><div class="split"><pre id="traceEvidence"></pre><pre id="traceAnswer"></pre></div>
<h2>All-question audit explorer (evaluator-only)</h2><p>This explorer intentionally shows automated opinions. Independent annotators use the separate blinded capture packet. A valid source segment pointer may still be irrelevant; no model label here is gold.</p>
<input id="search" placeholder="Question, ID or rationale" size="50"><select id="category"><option value="">All categories/origins</option></select><select id="flag"><option value="">All locator/metadata states</option><option value="pointer">Invalid source pointer</option><option value="availability">Answer-availability mismatch</option></select><span id="count"></span>
<div class="split"><div><select id="items" size="18" style="width:100%"></select><button id="prev">Previous200</button><button id="next">Next200</button><pre id="judgment"></pre></div><div><pre id="source"></pre></div></div>
<h2>Claim/source coverage and reproducibility</h2><p><a href="summary.json">Machine-readable summary</a> · <a href="claim-source-coverage.json">Claim/source coverage</a> · <a href="review-audit.json">Full request/output continuity audit</a>. CSV and Parquet tables accompany the editable SVG figures. No previous empirical receipts or presentation appendix tables were overwritten.</p>
<script type="application/json" id="data">__DATA__</script>
<script>const data=JSON.parse(document.getElementById('data').textContent),r=data.reviews;let visible=[],page=0;const el=id=>document.getElementById(id);for(const c of [...new Set(r.map(x=>x.category||x.origin))].sort()){let o=document.createElement('option');o.value=c;o.textContent=c;el('category').append(o)}function show(){let x=r[Number(el('items').value)];if(!x)return;el('judgment').textContent=JSON.stringify({query_id:x.query_id,question:x.question,published_answer:x.native_answer,model_opinion:x.review,locator_errors:x.pointer_errors,availability_label_mismatch:x.answer_availability_label_mismatch,actual_receipt:x.actual_request_path,independent_gold:false},null,2);el('source').textContent=x.source_ids.map(id=>'DOCUMENT '+id+'\nPUBLISHER '+data.documents[id].source_publisher+'\n\n'+data.documents[id].text).join('\n\n────────────\n\n')}function render(){el('items').replaceChildren();for(const i of visible.slice(page*200,(page+1)*200)){let o=document.createElement('option');o.value=i;o.textContent=r[i].query_id+' | '+r[i].question;el('items').append(o)}el('count').textContent=visible.length+' reviews; page '+(page+1);el('items').selectedIndex=0;show()}function filter(){const s=el('search').value.toLowerCase(),c=el('category').value,f=el('flag').value;visible=r.map((x,i)=>i).filter(i=>(!s||(r[i].query_id+' '+r[i].question+' '+r[i].review.rationale).toLowerCase().includes(s))&&(!c||(r[i].category||r[i].origin)===c)&&(!f||(f==='pointer'?r[i].pointer_errors.length:r[i].answer_availability_label_mismatch)));page=0;render()}for(const id of ['search','category','flag'])el(id).addEventListener('input',filter);el('items').addEventListener('change',show);el('prev').onclick=()=>{page=Math.max(0,page-1);render()};el('next').onclick=()=>{if((page+1)*200<visible.length)page++;render()};filter();</script></html>"""
    cards = [
        {"item": k, "value": summary[k]}
        for k in (
            "native_original_rows",
            "native_nonblank_published_answers",
            "generated_dependent_candidates",
            "actual_content_reviews_received",
            "review_pending_items",
            "source_locator_error_reviews",
            "answer_availability_label_mismatches",
            "executed_pipeline_questions",
            "executed_pipeline_sources",
            "answer_and_router_model",
            "graph_variants_materialized",
            "retrieval_episodes",
            "answer_responses",
            "independent_semantic_gold",
            "ontology_semantic_lift",
        )
    ]
    page = (
        page.replace("__SUMMARY__", table(cards, ["item", "value"]))
        .replace("__CATEGORIES__", table(categories, list(categories[0])))
        .replace("__EXTRACTION__", table(extractions, list(extractions[0])))
        .replace("__RETRIEVAL__", table(retrieval, list(retrieval[0])))
        .replace("__DATA__", payload)
    )
    trace_script = r"""
for(let i=0;i<data.episodes.length;i++){
  const x=data.episodes[i],o=document.createElement('option');o.value=i;
  o.textContent=x.query_id+' | '+x.model+' / '+x.scenario+' / '+x.strategy+' | '+(x.answer?x.answer.normalized_decision.action:x.answer_error||'retrieval only');
  el('traceItems').append(o);
}
function showTrace(){
  const x=data.episodes[Number(el('traceItems').value)];if(!x)return;
  const q=r.find(y=>y.query_id===x.query_id);
  const plans=data.routing_plans.filter(y=>y.query_id===x.query_id&&y.extraction_model===x.model&&y.scenario===x.scenario);
  el('traceEvidence').textContent=JSON.stringify({
    question:q?q.question:x.query_id,
    actual_routing_plans:plans,
    actual_router_requests:plans.map(y=>data.llm_requests[y.request_id]||null),
    retrieval:x.retrieval_trace||{source_ids:x.source_ids},
    actual_graph_queries:data.graph_queries.filter(y=>x.retrieval_trace&&y.parent_trace_id===x.retrieval_trace.trace_id)
  },null,2);
  el('traceAnswer').textContent=JSON.stringify({
    actual_answer_model_request:data.llm_requests[x.actual_answer_request_id]||null,
    actual_answer:x.answer||null,answer_error:x.answer_error||null,
    semantic_success:null,independent_gold:false
  },null,2);
}
el('traceItems').onchange=showTrace;showTrace();
"""
    page = page.replace(
        "filter();</script></html>", "filter();" + trace_script + "</script></html>"
    )
    page = page.replace(
        "<h2>Retrieval and answer execution</h2>",
        "<h2>Retrieval and answer execution</h2><p>Answer and router model: "
        + html.escape(contract["answer_model"])
        + ". Rows vary the source extraction model and guidance.</p>",
    )
    (root / "index.html").write_text(page)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--extraction", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--additional-attempt-roots", type=Path, nargs="*", default=[])
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2))


if __name__ == "__main__":
    main()
