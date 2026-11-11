"""Synthetic statements and software expectations; never empirical reviewed gold."""

from ..models import digest
from ..ds01.contracts import text_hash
from .contracts import Snapshot, InteractionCase, AgentResponse


def synthetic_fixture(ontology):
    texts = [
        (
            "fin",
            "finance",
            "Consolidated FY2025 revenue of Alpha Energy was USD 800 million; parent standalone FY2025 revenue was USD 500 million.",
        ),
        (
            "esg",
            "sustainability",
            "Alpha Energy commits to a net-zero target for 2030. This is a voluntary planned target. Realized FY2025 emissions reduction was 12 percent within its operational boundary.",
        ),
        (
            "sub",
            "finance",
            "Alpha Energy Holdings is a distinct legal subsidiary. Its standalone FY2025 revenue was USD 120 million.",
        ),
        (
            "other",
            "general",
            "Beta Logistics reports FY2025 revenue of USD 70 million.",
        ),
        (
            "legal",
            "legal",
            "Synthetic jurisdiction X statute L establishes a separate annual reporting obligation for Alpha Energy effective 2025-01-01; it does not make the voluntary 2030 target enforceable.",
        ),
        (
            "homonym",
            "finance",
            "A different legal entity also named Alpha Energy is registered in synthetic jurisdiction Y. Its standalone FY2025 revenue was USD 55 million.",
        ),
    ]
    publishers = [
        dict(publisher_id="pub-" + k, name="Synthetic " + k + " publisher")
        for k, _, _ in texts
    ]
    docs = []
    spans = []
    for key, kind, text in texts:
        h = text_hash(text)
        docid = "syn-doc-" + key
        v = "synthetic-source-v1"
        docs.append(
            dict(
                document_id=docid,
                document_version=v,
                publisher_id="pub-" + key,
                source_uri=None,
                text=text,
                content_hash=h,
                source_kind=kind,
                synthetic=True,
            )
        )
        spans.append(
            dict(
                span_id=digest([docid, v, 0, len(text)]),
                document_id=docid,
                document_version=v,
                source_hash=h,
                start=0,
                end=len(text),
                quote=text,
            )
        )
    entities = [
        dict(
            entity_id=k,
            canonical_name=n,
            canonical_identifier=None,
            identity_status="synthetic_fixture",
        )
        for k, n in [
            ("syn-alpha", "Alpha Energy"),
            ("syn-holdings", "Alpha Energy Holdings"),
            ("syn-beta", "Beta Logistics"),
            ("syn-homonym", "Alpha Energy"),
        ]
    ]
    reps = [
        dict(representation_id="rep-" + k, entity_id=e, domain_id=d, business_role=r)
        for k, e, d, r in [
            ("alpha-fin", "syn-alpha", "finance", "reporting issuer"),
            ("alpha-esg", "syn-alpha", "sustainability", "operational boundary"),
            ("alpha-legal", "syn-alpha", "legal", "statutory reporting subject"),
            ("holdings-fin", "syn-holdings", "finance", "standalone subsidiary"),
            ("beta", "syn-beta", "finance", "reporting issuer"),
            (
                "homonym",
                "syn-homonym",
                "finance",
                "different jurisdiction reporting entity",
            ),
        ]
    ]

    def a(
        aid,
        rep,
        pred,
        obj,
        typ,
        source,
        mod,
        basis,
        scope,
        unit=None,
        period="FY2025",
        jurisdiction=None,
        valid_time=None,
    ):
        return dict(
            assertion_id=aid,
            subject_representation_id="rep-" + rep,
            predicate=pred,
            object_type=typ,
            object_value=obj,
            qualifiers=dict(
                period=period,
                reporting_basis=basis,
                scope=scope,
                unit=unit,
                jurisdiction=jurisdiction,
                modality=mod,
                valid_time=valid_time,
                assertion_kind="source_assertion",
                ontology_version=ontology.version,
            ),
            evidence_span_ids=[spans[source]["span_id"]],
            source_publisher_id=docs[source]["publisher_id"],
            source_provider_id="synthetic-source-track",
            extraction_run_id="authored-fixture-v1",
            support_status="synthetic_fixture",
        )

    assertions = [
        a(
            "syn-a-consolidated",
            "alpha-fin",
            "REPORTS_REVENUE",
            "800",
            "RevenueMeasure",
            0,
            "reported",
            "consolidated",
            "group",
            "USD million",
        ),
        a(
            "syn-a-separate",
            "alpha-fin",
            "REPORTS_REVENUE",
            "500",
            "RevenueMeasure",
            0,
            "reported",
            "separate",
            "parent",
            "USD million",
        ),
        a(
            "syn-a-pledge",
            "alpha-esg",
            "COMMITS_TO",
            "net-zero",
            "ReductionTarget",
            1,
            "planned",
            None,
            "operational boundary",
            period="2030",
        ),
        a(
            "syn-a-achieved",
            "alpha-esg",
            "ACHIEVED",
            "12",
            "ReductionResult",
            1,
            "realized",
            None,
            "operational boundary",
            "percent",
        ),
        a(
            "syn-a-subsidiary",
            "holdings-fin",
            "REPORTS_REVENUE",
            "120",
            "RevenueMeasure",
            2,
            "reported",
            "separate",
            "subsidiary",
            "USD million",
        ),
        a(
            "syn-a-beta",
            "beta",
            "REPORTS_REVENUE",
            "70",
            "RevenueMeasure",
            3,
            "reported",
            "consolidated",
            "group",
            "USD million",
        ),
        a(
            "syn-a-obligation",
            "alpha-legal",
            "LEGALLY_OBLIGATED_TO",
            "annual reporting",
            "LegalObligation",
            4,
            "legally_required",
            None,
            "statutory reporting",
            jurisdiction="Synthetic jurisdiction X",
            valid_time="2025-01-01 onward",
        ),
        a(
            "syn-a-homonym",
            "homonym",
            "REPORTS_REVENUE",
            "55",
            "RevenueMeasure",
            5,
            "reported",
            "separate",
            "entity in synthetic jurisdiction Y",
            "USD million",
            jurisdiction="Synthetic jurisdiction Y",
        ),
    ]
    graph = Snapshot(
        version="synthetic-enterprise-graph-v1",
        publishers=publishers,
        documents=docs,
        spans=spans,
        entities=entities,
        representations=reps,
        assertions=assertions,
        synthetic=True,
    )
    graph.validate_ontology(ontology)
    spec = [
        (
            "clear",
            "What was Alpha Energy FY2025 consolidated revenue?",
            ["syn-a-consolidated", "syn-a-separate"],
            ["ANSWER_WITH_CITATIONS"],
            [],
            ["REPORTS_REVENUE"],
        ),
        (
            "clarify",
            "What was Alpha Energy FY2025 revenue?",
            ["syn-a-consolidated", "syn-a-separate"],
            ["ASK_CLARIFICATION", "COMPARE_SCOPED_ANSWERS"],
            ["reporting_basis"],
            ["REPORTS_REVENUE"],
        ),
        (
            "compare",
            "Compare the announced target with realized FY2025 emissions reduction.",
            ["syn-a-pledge", "syn-a-achieved"],
            ["COMPARE_SCOPED_ANSWERS"],
            [],
            ["COMMITS_TO", "ACHIEVED"],
        ),
        (
            "abstain",
            "Is the voluntary net-zero target legally enforceable?",
            ["syn-a-pledge"],
            ["ABSTAIN_INSUFFICIENT_EVIDENCE"],
            [],
            ["LEGALLY_OBLIGATED_TO"],
        ),
        (
            "identity",
            "What was Alpha Energy Holdings standalone FY2025 revenue?",
            ["syn-a-consolidated", "syn-a-subsidiary"],
            ["ANSWER_WITH_CITATIONS"],
            [],
            ["REPORTS_REVENUE"],
        ),
        (
            "homonym",
            "What was standalone FY2025 revenue for the Alpha Energy registered in synthetic jurisdiction Y?",
            ["syn-a-consolidated", "syn-a-homonym"],
            ["ANSWER_WITH_CITATIONS"],
            [],
            ["REPORTS_REVENUE"],
        ),
    ]
    byid = {a.assertion_id: a for a in graph.assertions}
    cases = []
    for name, q, ids, actions, missing, pred in spec:
        evidence = sorted({s for i in ids for s in byid[i].evidence_span_ids})
        cases.append(
            InteractionCase(
                case_id="SYN-" + name,
                parent_case_id=None,
                protocol_id="SYN-ENTERPRISE-INTERACTION-01",
                question=q,
                synthetic=True,
                context_assertion_ids=ids,
                allowed_actions=actions,
                material_missing_dimensions=missing,
                required_predicates=pred,
                sufficient_span_sets=[] if name == "abstain" else [evidence],
                rationale="Authored software expectation. No human adjudication or empirical model evaluation.",
                counterfactual_kind="identity_context"
                if name in {"identity", "homonym"}
                else "baseline",
                annotation_status="synthetic_expectation",
            )
        )
    # Irrelevant-source addition preserves the expected answer/action and parent dependence.
    cases.append(
        cases[0].model_copy(
            update=dict(
                case_id="SYN-clear-irrelevant",
                parent_case_id="SYN-clear",
                context_assertion_ids=cases[0].context_assertion_ids + ["syn-a-beta"],
                counterfactual_kind="irrelevant_addition",
            )
        )
    )
    return graph, cases


def mock_response(case, snapshot):
    """Scripted adapter using fixture expectations, not a model comprehension score."""
    if case.case_id.startswith("SYN-clarify"):
        return AgentResponse(
            action="ASK_CLARIFICATION",
            answer=None,
            clarification_question="Consolidated group revenue or parent standalone revenue?",
            compared_scopes=[],
            claims=[],
        )
    if case.case_id == "SYN-abstain":
        return AgentResponse(
            action="ABSTAIN_INSUFFICIENT_EVIDENCE",
            answer="The delivered evidence does not establish enforceability.",
            clarification_question=None,
            compared_scopes=[],
            claims=[],
        )
    ids = (
        case.context_assertion_ids[:2]
        if case.case_id == "SYN-compare"
        else [
            "syn-a-subsidiary"
            if case.case_id == "SYN-identity"
            else (
                "syn-a-homonym"
                if case.case_id == "SYN-homonym"
                else "syn-a-consolidated"
            )
        ]
    )
    a = {r.assertion_id: r for r in snapshot.assertions}
    return AgentResponse(
        action="COMPARE_SCOPED_ANSWERS"
        if case.case_id == "SYN-compare"
        else "ANSWER_WITH_CITATIONS",
        answer="Synthetic scripted output; see qualified assertions.",
        clarification_question=None,
        compared_scopes=["planned target 2030", "realized performance FY2025"]
        if case.case_id == "SYN-compare"
        else [],
        claims=[
            dict(
                claim_id="claim-" + i,
                text=a[i].predicate + ": " + a[i].object_value,
                assertion_ids=[i],
                evidence_span_ids=a[i].evidence_span_ids,
                source_publisher_ids=[a[i].source_publisher_id],
            )
            for i in ids
        ],
    )


def graph_records(snapshot, policy):
    """First-class assertion representation; identity equality never flattens qualifiers."""
    if policy not in {"global_lossy", "single_qualified", "multi_qualified"}:
        raise ValueError("unknown identity policy")
    entities = {r.representation_id: r.entity_id for r in snapshot.representations}
    rows = []
    for a in snapshot.assertions:
        raw = a.model_dump()
        row = dict(
            assertion=raw,
            entity_id=entities[a.subject_representation_id],
            database="single"
            if policy != "multi_qualified"
            else next(
                r.domain_id
                for r in snapshot.representations
                if r.representation_id == a.subject_representation_id
            ),
            qualified_payload_hash=digest(raw),
        )
        if policy == "global_lossy":
            row["retrieval_view"] = {
                k: raw[k] for k in ["predicate", "object_value", "evidence_span_ids"]
            }
        else:
            row["retrieval_view"] = raw
        rows.append(row)
    return rows


def relevant_counterfactual(snapshot, ontology):
    """New synthetic source version only; the original snapshot stays immutable."""
    payload = snapshot.model_dump()
    doc = next(d for d in payload["documents"] if d["document_id"] == "syn-doc-fin")
    old_span = next(
        s for s in payload["spans"] if s["document_id"] == doc["document_id"]
    )
    old_id = old_span["span_id"]
    doc["text"] = doc["text"].replace("USD 800 million", "USD 810 million")
    doc["document_version"] = "synthetic-relevant-change-v2"
    doc["content_hash"] = text_hash(doc["text"])
    old_span.update(
        document_version=doc["document_version"],
        source_hash=doc["content_hash"],
        quote=doc["text"],
        end=len(doc["text"]),
    )
    old_span["span_id"] = digest(
        [doc["document_id"], doc["document_version"], 0, len(doc["text"])]
    )
    for a in payload["assertions"]:
        a["evidence_span_ids"] = [
            old_span["span_id"] if s == old_id else s for s in a["evidence_span_ids"]
        ]
        if a["assertion_id"] == "syn-a-consolidated":
            a["object_value"] = "810"
    payload["version"] = "synthetic-relevant-change-v2"
    changed = Snapshot.model_validate(payload)
    changed.validate_ontology(ontology)
    return changed
