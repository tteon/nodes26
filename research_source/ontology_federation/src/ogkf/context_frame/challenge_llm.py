"""Bounded MARA source indexing or answer capture; no automatic semantic gold."""

import json
import hashlib
import os
import re
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

from pydantic import Field

from ..ds01.contracts import Query
from ..ds01.retrieval import answer
from ..gateway import Gateway
from ..models import Strict, digest
from ..observability import Recorder
from ..probes import load_probe_receipts, probe_mara
from .challenge import (
    BusinessProposal,
    admission,
    jsonl,
    load_snapshot,
    save,
    select_ids,
)
from .compiler import make_span


class QuoteFact(Strict):
    subject: str = Field(min_length=1)
    predicate: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,79}$")
    object_value: str
    period: str | None
    scope: str | None
    unit: str | None
    modality: str | None
    quote: str = Field(min_length=1)


class DraftFact(QuoteFact):
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class DraftExtraction(Strict):
    facts: list[DraftFact] = Field(max_length=16)


class QuoteExtraction(Strict):
    facts: list[QuoteFact] = Field(max_length=16)


def locate_quote(fact, document):
    """Declared exact localization, never fuzzy repair or semantic validation."""
    start = document.text.find(fact.quote)
    if start < 0 or document.text.find(fact.quote, start + 1) >= 0:
        raise ValueError("quote must occur exactly once in original source")
    return DraftFact(**fact.model_dump(), start=start, end=start + len(fact.quote))


def source_messages(
    document, max_facts=16, span_mode="offsets", coverage_contract=False
):
    coverage = (
        "Coverage contract v1: extract distinct explicit assertions, including negative beliefs and materiality expectations. "
        "EXPECTS_NO_MATERIAL_EFFECT means a reported belief about material effect, not a guarantee of no risk. "
        "ESTIMATES_FUTURE_PAYMENT means an estimated payment amount for a stated year, not a recognized liability. "
        "period is a calendar date/year/range only; ordinary course of business is not a period. "
        "Preserve the subject expression in the source (including subsidiaries). "
        "Quote the complete supporting sentence, retaining estimation, as-of date and scope; multiple assertions may share a sentence. "
        if coverage_contract
        else ""
    )
    return [
        {
            "role": "system",
            "content": f"Extract at most{max_facts} enterprise relationship assertions explicitly stated in this source. "
            "Use subject,typed predicate,object and stated period/scope/unit/modality. Preserve unknowns as null. "
            "Use business predicates such as REPORTS_REVENUE,HOLDS_ROLE,MEMBER_OF,OWES,PARTY_TO,COMMITS_TO "
            "only if the source actually supports that meaning. A commitment is not achievement or a legal obligation. "
            "Do not infer that the source publisher is the subject of every assertion. "
            "Keep separate and consolidated values, target and realization, subsidiaries and parents distinct. "
            + (
                "Quote one exact contiguous source span; start/end are zero-based Unicode character offsets. "
                if span_mode == "offsets"
                else "span-location-unique-quote-v1: quote one exact contiguous unique source span. "
                "The application locates its exact characters; do not generate offsets. "
            )
            + coverage
            + "Table evidence must include needed headers,scale and scope. Return JSON schema: "
            + json.dumps(
                (
                    DraftExtraction if span_mode == "offsets" else QuoteExtraction
                ).model_json_schema()
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"source": document.model_dump()}, ensure_ascii=False
            ),
        },
    ]


def normalize_draft(fact, document, request_id, model):
    if (
        not 0 <= fact.start < fact.end <= len(document.text)
        or document.text[fact.start : fact.end] != fact.quote
    ):
        raise ValueError("LLM source quote/offset mismatch; no silent repair")
    return {
        **fact.model_dump(),
        "assertion_id": digest([document.document_id, request_id, fact.model_dump()]),
        "document_id": document.document_id,
        "document_version": document.document_version,
        "source_hash": document.content_hash,
        "evidence_id": digest(
            [document.document_id, document.document_version, fact.start, fact.end]
        ),
        "request_id": request_id,
        "model": model,
        "source_publisher": document.source_publisher,
        "graph_provider": None,
        "llm_provider": "mara",
        "entity_identity_status": "source_local_proposal",
        "semantic_support": "UNKNOWN",
        "validation_status": "exact_span_only",
        "extraction_method": "llm_business_relation_draft",
    }


def live_inputs(args, dataset, questions, snapshot):
    """Validate source/query selection before spending any capability calls."""
    offset = getattr(args, "offset", 0)
    document_ids = getattr(args, "document_ids", None)
    query_ids = getattr(args, "query_ids", None)
    if (document_ids or query_ids) and offset:
        raise ValueError("explicit selection cannot be combined with offset")
    if args.stage == "index":
        if query_ids:
            raise ValueError("index selection requires document IDs, not query IDs")
        rows = sorted(
            (d for d in dataset.documents if len(d.text) <= 12000),
            key=lambda d: digest([snapshot, "source-only-index", d.document_id]),
        )
        rows = (
            select_ids(rows, document_ids, lambda d: d.document_id)
            if document_ids
            else rows[offset : offset + args.limit]
        )
    else:
        if document_ids or args.retrieval_run is None:
            raise ValueError(
                "answer capture requires native queries and an immutable retrieval run"
            )
        protocol = json.loads((args.retrieval_run / "protocol.json").read_text())
        if protocol["snapshot_hash"] != snapshot:
            raise ValueError("answer capture snapshot does not match retrieval")
        native = {q["query_id"] for q in questions if q["origin"] == "native_unchanged"}
        rows = [
            json.loads(x)
            for x in (args.retrieval_run / "retrieval_traces.jsonl")
            .read_text()
            .splitlines()
        ]
        rows = [r for r in rows if r["query_id"] in native and r["strategy"] == "B3"]
        rows = (
            select_ids(rows, query_ids, lambda r: r["query_id"])
            if query_ids
            else rows[offset : offset + args.limit]
        )
    if not rows or len(rows) > args.limit:
        raise ValueError("empty selection or selection exceeds the declared limit")
    return rows


def run_live(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    offset = getattr(args, "offset", 0)
    capability_run = getattr(args, "capability_run", None)
    probe_calls = 0 if capability_run else 6
    max_facts = getattr(args, "max_facts", 16)
    index_max_tokens = getattr(args, "index_max_tokens", 2048)
    span_mode = getattr(args, "span_mode", "offsets")
    extraction_type = DraftExtraction if span_mode == "offsets" else QuoteExtraction
    if (
        offset < 0
        or span_mode not in {"offsets", "unique-quote"}
        or not 1 <= args.limit <= 32
        or not args.limit + probe_calls <= args.max_calls <= 100
        or not 1 <= max_facts <= 16
        or index_max_tokens not in (2048, 3072, 4096)
    ):
        raise ValueError(
            "declare at most100 calls including any six new capability calls and all stage calls"
        )
    gate = admission()
    gate.update(
        stage=args.stage,
        span_mode=span_mode,
        coverage_contract=getattr(args, "coverage_contract", False),
        requested_model=args.model,
        max_logical_calls=args.max_calls,
        max_output_tokens=4096 if args.stage == "answer" else index_max_tokens,
        max_fact_proposals_per_document=max_facts if args.stage == "index" else None,
        source_or_answer_limit=args.limit,
        offset=offset,
        semantic_metrics=None,
        automatic_semantic_gold=False,
    )
    if not gate["MARA_API_KEY_present"] or not gate["MARA_ADMISSION_LOCK_present"]:
        gate["status"] = "BLOCKED_CREDENTIALS"
        save(root / "admission.json", gate)
        return gate
    dataset, questions, snapshot = load_snapshot(args.snapshot)
    gate["source_snapshot_hash"] = snapshot
    try:
        selected = live_inputs(args, dataset, questions, snapshot)
        cached = (
            load_probe_receipts(
                capability_run,
                args.model,
                os.getenv("MARA_BASE_URL", "https://api.cloud.mara.com/v1").rstrip("/"),
            )
            if capability_run
            else None
        )
    except Exception as exc:
        gate.update(status="INVALID_PREFLIGHT", error_type=type(exc).__name__)
        save(root / "admission.json", gate)
        raise
    gate["selected_document_ids"] = (
        [d.document_id for d in selected] if args.stage == "index" else None
    )
    gate["selected_query_ids"] = (
        [r["query_id"] for r in selected] if args.stage == "answer" else None
    )
    gate["selection_mode"] = (
        "explicit_engineering_cohort"
        if getattr(args, "document_ids", None) or getattr(args, "query_ids", None)
        else "default_batch"
    )
    gate["capability_receipts_reused_from"] = (
        str(capability_run) if capability_run else None
    )
    gate["new_capability_call_budget"] = probe_calls
    gate["capability_manifest_hash"] = (
        hashlib.sha256(
            (capability_run / "mara-capabilities.json").read_bytes()
        ).hexdigest()
        if capability_run
        else None
    )
    recorder = Recorder(root, "live", experiment="finder_graph_challenge")
    config = SimpleNamespace(
        max_logical_calls=args.max_calls,
        max_attempts=1,
        timeout_seconds=60,
        min_interval_seconds=1,
        temperature=0,
        models=[args.model],
    )
    gateway = Gateway(recorder, config)
    outcomes = []
    gate["status"] = "DISPATCHING"
    try:
        capabilities = cached or probe_mara(gateway, [args.model])
        save(root / "mara-capabilities.json", capabilities)
        observed = capabilities["models"].get(args.model, {})
        if not observed.get("listed") or not observed.get("basic"):
            gate["status"] = "BLOCKED_MODEL_CAPABILITY"
            return gate
        gateway.capabilities = capabilities["models"]
        if args.stage == "index":
            for d in selected:
                trace = str(uuid.uuid4())
                result = gateway.generate_structured(
                    source_messages(
                        d,
                        max_facts,
                        span_mode,
                        getattr(args, "coverage_contract", False),
                    ),
                    extraction_type.model_json_schema(),
                    args.model,
                    {
                        "max_tokens": index_max_tokens,
                        **(
                            {"temperature": 0}
                            if observed.get("temperature") is True
                            else {}
                        ),
                    },
                    {
                        "trace_id": trace,
                        "parent_step_id": "business_extraction",
                        "document_id": d.document_id,
                        "prompt_version": f"business-source-draft-v4-{span_mode}-coverage{getattr(args, 'coverage_contract', False)}-max{max_facts}",
                    },
                )
                draft = extraction_type.model_validate(result["value"])
                if len(draft.facts) > max_facts:
                    raise ValueError("extraction exceeded the declared fact budget")
                accepted, rejected = [], []
                for i, fact in enumerate(draft.facts):
                    try:
                        if span_mode == "unique-quote":
                            fact = locate_quote(fact, d)
                        accepted.append(
                            normalize_draft(fact, d, result["request_id"], args.model)
                        )
                    except ValueError as exc:
                        rejected.append(
                            {
                                "position": i,
                                "reason": str(exc),
                                "raw": fact.model_dump(),
                            }
                        )
                recorder.write(
                    "normalized/" + result["request_id"] + ".json",
                    {"accepted": accepted, "rejected": rejected},
                )
                outcomes.append(
                    {
                        "document_id": d.document_id,
                        "trace_id": trace,
                        "accepted_span_proposals": accepted,
                        "rejected": rejected,
                        "semantic_quality": None,
                    }
                )
        else:
            public = {
                q["query_id"]: Query(query_id=q["query_id"], text=q["text"])
                for q in questions
            }
            for r in selected:
                value = answer(
                    public[r["query_id"]],
                    r["evidence"],
                    gateway,
                    args.model,
                    {
                        "trace_id": str(uuid.uuid4()),
                        "experiment_condition_id": r["condition_id"],
                    },
                )
                outcomes.append(
                    {
                        "query_id": r["query_id"],
                        "retrieval_trace_id": r["trace_id"],
                        "plan_reused": True,
                        "routing_mode": r["plan_mode"],
                        "business_assertion_ids_delivered": sorted(
                            {
                                binding["assertion"]["assertion_id"]
                                for evidence in r["evidence"]
                                for binding in evidence.get("graph_bindings", [])
                                if binding["assertion"].get("extraction_method")
                                == "llm_business_relation_draft"
                            }
                        ),
                        "answer": value.model_dump(),
                        "semantic_quality": None,
                        "native_answer_agreement": "not_scored_without_registered_comparator",
                    }
                )
        jsonl(root / "outcomes.jsonl", outcomes)
        if args.stage == "index":
            gate["business_smoke_acceptance"] = (
                "SPAN_ONLY_PENDING_GRAPH_AND_CLAIM"
                if outcomes and all(r["accepted_span_proposals"] for r in outcomes)
                else "NOT_MET_EMPTY_OR_REJECTED"
            )
        gate.update(
            status="EXECUTED_UNSCORED",
            stage_outcomes=len(outcomes),
            actual_http_attempts=len(recorder.rows),
            new_http_calls=len(recorder.rows),
        )
        return gate
    except Exception as exc:
        gate.update(status="EXECUTION_FAILED", error_type=type(exc).__name__)
        raise
    finally:
        gateway.close()
        gate["new_http_calls"] = len(recorder.rows)
        gate["actual_http_attempts"] = len(recorder.rows)
        gate["trace_rows"] = len(recorder.readback())
        gate["logical_calls"] = gateway.logical_calls
        save(root / "admission.json", gate)


def attach_index(args):
    """Freeze real span-valid LLM drafts into a new queryable graph snapshot.

    This is structural admission, never independent semantic approval. The base
    corpus and every native QA record remain unchanged.
    """
    dataset, _, snapshot = load_snapshot(args.snapshot)
    admission_record = json.loads((args.index_run / "admission.json").read_text())
    if (
        admission_record.get("status") != "EXECUTED_UNSCORED"
        or admission_record.get("stage") != "index"
        or admission_record.get("source_snapshot_hash") != snapshot
    ):
        raise ValueError("real source-bound index execution required")
    records = [
        json.loads(x)
        for x in (args.index_run / "outcomes.jsonl").read_text().splitlines()
    ]
    docs = {d.document_id: d for d in dataset.documents}
    assertions = {a.assertion_id: a.model_dump() for a in dataset.assertions}
    entities = {e.entity_id: e.model_dump() for e in dataset.entities}
    spans = {
        x["span_id"]: x
        for x in (
            json.loads(line)
            for line in (args.snapshot / "spans.jsonl").read_text().splitlines()
        )
    }
    added = 0
    for record in records:
        for raw in record["accepted_span_proposals"]:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", raw["request_id"]):
                raise ValueError("invalid request receipt identifier")
            d = docs[raw["document_id"]]
            # Revalidate provenance at ingestion rather than trusting status text.
            fact = DraftFact.model_validate({k: raw[k] for k in DraftFact.model_fields})
            accepted_paths = list(
                args.index_run.glob("*/accepted/" + raw["request_id"] + ".json")
            )
            if len(accepted_paths) != 1:
                raise ValueError("missing unique structured-output receipt")
            recorder_root = accepted_paths[0].parent.parent
            http_paths = list(
                (recorder_root / "raw").glob(raw["request_id"] + "-*.json")
            )
            if len(http_paths) != 1:
                raise ValueError("missing unique no-retry HTTP receipt")
            response = json.loads(http_paths[0].read_text())
            body = response["request"]
            delivered = json.loads(body["messages"][-1]["content"])["source"]
            returned = json.loads(
                response["response"]["choices"][0]["message"]["content"]
            )
            receipt_facts = returned["facts"]
            if admission_record.get("span_mode") == "unique-quote":
                if (
                    "span-location-unique-quote-v1"
                    not in body["messages"][0]["content"]
                ):
                    raise ValueError(
                        "missing declared exact quote localization protocol"
                    )
                receipt_facts = [
                    locate_quote(QuoteFact.model_validate(f), d).model_dump()
                    for f in receipt_facts
                ]
            if (
                not 200 <= response["http_status"] < 300
                or body["model"] != raw["model"]
                or delivered != d.model_dump()
                or fact.model_dump() not in receipt_facts
                or returned != json.loads(accepted_paths[0].read_text())
            ):
                raise ValueError(
                    "business draft absent from actual source request/response"
                )
            normalized = normalize_draft(fact, d, raw["request_id"], raw["model"])
            if normalized != raw or raw["assertion_id"] in assertions:
                raise ValueError("changed or duplicate business draft")
            span = make_span(d, fact.start, fact.end, "statement")
            spans.setdefault(span.span_id, span.model_dump())
            entity_id = digest(
                ["llm-source-local-subject", d.document_id, fact.subject]
            )
            entities[entity_id] = {
                "entity_id": entity_id,
                "canonical_name": fact.subject,
                "aliases": [fact.subject],
                "identity_status": "lexical_only",
            }
            proposal = BusinessProposal(
                assertion_id=raw["assertion_id"],
                subject_id=entity_id,
                predicate=fact.predicate,
                object_value=fact.object_value,
                document_id=d.document_id,
                evidence_id=d.evidence_id,
                source_hash=d.content_hash,
                start=fact.start,
                end=fact.end,
                quote=fact.quote,
                domain_id=d.domain_id,
                evidence_span_ids=[span.span_id],
                extraction_method="llm_business_relation_draft",
                qualifier_proposals={
                    k: getattr(fact, k) for k in ("period", "scope", "unit", "modality")
                },
            )
            assertions[proposal.assertion_id] = proposal.model_dump()
            added += 1
    if added == 0:
        raise ValueError(
            "EMPTY_BUSINESS_EXTRACTION: no connected business assertion; execution is not smoke acceptance"
        )
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    for name in (
        "documents.jsonl",
        "queries.public.jsonl",
        "queries.private.jsonl",
        "review.pending.jsonl",
        "review.html",
    ):
        shutil.copyfile(args.snapshot / name, root / name)
    jsonl(root / "entities.jsonl", list(entities.values()))
    jsonl(root / "proposals.jsonl", list(assertions.values()))
    jsonl(root / "spans.jsonl", list(spans.values()))
    save(
        root / "manifest.json",
        {
            "version": "finder-business-draft-snapshot-v1",
            "base_snapshot": snapshot,
            "index_run": str(args.index_run),
            "added_LLM_proposals": added,
            "independently_reviewed_business_facts": 0,
            "source_text_and_native_QA_unchanged": True,
        },
    )
    save(
        root / "snapshot-hashes.json",
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.glob("*.jsonl")
        },
    )
    load_snapshot(root)
    return {
        "added_LLM_proposals": added,
        "new_snapshot": str(root),
        "semantic_gold_promoted": False,
    }
