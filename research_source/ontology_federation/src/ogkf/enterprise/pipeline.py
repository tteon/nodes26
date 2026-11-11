"""Offline enterprise research audit, pending review infrastructure and smoke runs."""

import argparse
from collections import Counter
import html
import json
import math
import os
from pathlib import Path
from statistics import NormalDist

from ..models import digest
from ..context_frame.compiler import file_hash
from ..context_frame.models import Bundle
from ..presentation_handoff import verify as verify_handoff
from .contracts import (
    Ontology,
    RelationDefinition,
    Snapshot,
    QualifiedAssertion,
    ExtractionResult,
    InteractionCase,
    AgentResponse,
    DIMENSIONS,
)
from .annotation import (
    AnnotationPacket,
    ReviewTarget,
    EntityIdentityGold,
    ContextCompatibilityGold,
    FactSupportGold,
    MinimalSufficientEvidenceSets,
    AnswerClaimGold,
    ReviewerSubmission,
    AdjudicationReceipt,
    pending_submission,
    adjudicate,
)
from .matrices import (
    MatrixConfig,
    budget,
    qualify_live,
    run_mock,
    run_live,
    parse_config,
)
from .profiles import load_bundle
from .fixtures import synthetic_fixture, graph_records, relevant_counterfactual


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")


def build_packet(bundle, selected_ids):
    chosen = [c for c in bundle.cases if c.query.query_id in selected_ids]
    if len(chosen) != len(selected_ids):
        raise ValueError("selected review question absent from source-bound frame")
    frozen_hash = bundle.bundle_hash
    docs = {d.evidence_id: d for d in bundle.documents}
    sources = {}
    targets = []
    for case in chosen:
        ev = case.reference_evidence_ids + case.candidate_evidence_ids
        available = []
        for eid in ev:
            d = docs[eid]
            sid = digest([d.document_id, d.document_version, 0, len(d.text)])
            sources[sid] = dict(
                span_id=sid,
                evidence_id=eid,
                document_id=d.document_id,
                document_version=d.document_version,
                source_hash=d.content_hash,
                start=0,
                end=len(d.text),
                quote=d.text,
                source_publisher=d.source_publisher,
            )
            available.append(sid)
        available = sorted(set(available))
        qid = case.query.query_id

        def add(kind, payload):
            tid = digest(["enterprise-review-target-v1", frozen_hash, qid, kind])
            common = dict(target_id=tid, source_span_ids=[], rationale="")
            model = {
                "entity_identity_gold": EntityIdentityGold,
                "context_compatibility_gold": ContextCompatibilityGold,
                "fact_support_gold": FactSupportGold,
                "minimal_sufficient_evidence_sets": MinimalSufficientEvidenceSets,
                "answer_claim_gold": AnswerClaimGold,
            }[kind]
            proposal = model(kind=kind, **common, **payload)
            targets.append(
                ReviewTarget(
                    target_id=tid,
                    query_id=qid,
                    kind=kind,
                    source_span_ids=available,
                    proposal=proposal,
                )
            )

        add(
            "entity_identity_gold",
            dict(
                left_mention_id="unresolved-source-mention:" + ev[0] + ":left",
                right_mention_id="unresolved-source-mention:" + ev[-1] + ":right",
                canonical_entity_ids=[],
                verdict="UNKNOWN",
            ),
        )
        add(
            "context_compatibility_gold",
            dict(
                query_id=qid,
                operation=None,
                dimensions={d: "UNKNOWN" for d in DIMENSIONS},
            ),
        )
        add(
            "fact_support_gold",
            dict(
                subject=None,
                predicate=None,
                object_value=None,
                qualifiers=None,
                verdict="UNKNOWN",
            ),
        )
        add(
            "minimal_sufficient_evidence_sets",
            dict(query_id=qid, operation=None, evidence_sets=[], verdict="UNKNOWN"),
        )
        add(
            "answer_claim_gold",
            dict(
                query_id=qid,
                claim_id=None,
                answer_hash=None,
                allowed_actions=[],
                answer_correct=None,
                source_supported=None,
                context_compatible=None,
                composition_valid=None,
                citation_binding_correct=None,
                material_missing_dimensions=[],
            ),
        )
    return AnnotationPacket(
        version="enterprise-independent-packet-v1",
        source_bundle_hash=frozen_hash,
        sources=list(sources.values()),
        targets=targets,
        synthetic=False,
        native_gold_policy="Existing native QA/reference answers preserved in the frozen frame. These new judgments are UNKNOWN. Publisher identity is not assertion-subject identity. Resolve actual entity mentions before deciding identity.",
    )


def export_review_forms(packet, bundle, root):
    save(root / "packet.private.json", packet.model_dump())
    questions = {c.query.query_id: c.query.text for c in bundle.cases}
    source = {s["span_id"]: s for s in packet.sources}
    for slot in ["A", "B"]:
        submission = pending_submission(packet, "")
        save(root / f"reviewer_{slot}.pending.json", submission.model_dump())
        articles = []
        byquery = {}
        for t in packet.targets:
            byquery.setdefault(t.query_id, []).append(t)
        for qid, rows in byquery.items():
            articles.append(
                "<article><h2>"
                + html.escape(questions[qid])
                + "</h2><p>Question ID: "
                + qid
                + "</p>"
            )
            sids = sorted({s for r in rows for s in r.source_span_ids})
            for sid in sids:
                s = source[sid]
                articles.append(
                    "<details><summary>"
                    + html.escape(s["document_id"])
                    + "</summary><p>Source span "
                    + sid
                    + "</p><pre>"
                    + html.escape(s["quote"])
                    + "</pre></details>"
                )
            articles.append("</article>")
        # Each reviewer gets only their own blank capture, never the peer's decisions.
        template = json.dumps(submission.model_dump(), ensure_ascii=False).replace(
            "</", "<\\/"
        )
        page = """<!doctype html><html lang='en'><meta charset='utf-8'><title>Independent enterprise review</title>
<style>body{font:16px/1.6 sans-serif;max-width:1100px;margin:2rem auto;padding:1rem}pre{white-space:pre-wrap;background:#f1f5f9;padding:1rem}textarea{width:100%;height:500px}article{border-top:2px solid #ccd;margin:2rem 0}</style>
<h1>Independent reviewer SLOT</h1><p>Do not inspect the other reviewer's submission. Record actual reviewer identity, source-linked rationale and external review record. UNKNOWN is allowed. No generated candidate is reviewed gold. Resolve entity mention identity before marking SAME/DIFFERENT; publisher is not automatically the assertion subject.</p>
<p>Save your JSON capture and send it for adjudication. No server or network submission occurs.</p><textarea id='capture'></textarea><button id='save'>Download my submission</button>ARTICLES
<script>const initial=TEMPLATE;document.getElementById('capture').value=JSON.stringify(initial,null,2);
document.getElementById('save').onclick=()=>{const value=JSON.parse(document.getElementById('capture').value);const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json'}));a.download='reviewer-SLOT.json';a.click();};</script></html>"""
        page = (
            page.replace("SLOT", slot)
            .replace("ARTICLES", "".join(articles))
            .replace("TEMPLATE", template)
        )
        (root / f"reviewer_{slot}.html").write_text(page)


def audit_frozen(package, handoff, output):
    frozen = handoff / "frozen_handoff/presentation_handoff_v1"
    verify_handoff(frozen)
    before = {
        str(p.relative_to(package)): file_hash(p)
        for base in [
            "reports/ds01_campaign",
            "reports/finder_candidates_v1",
            "reports/final_review_v1",
        ]
        for p in (package / base).rglob("*")
        if p.is_file()
    }
    save(output / "frozen-hashes.before.json", before)
    inputs = {
        p["source_id"]: frozen / p["bundle_path"]
        for p in json.loads((frozen / "sources.json").read_text())
    }
    facts = json.loads(inputs["S09"].read_text())["facts"]
    native = json.loads(inputs["S10"].read_text())
    checks = dict(
        original_questions=facts["native_questions"],
        nonblank_qa=facts["qa_candidates"],
        native_categories=facts["categories"],
        silver_frames=facts["frame_cases"],
        independent_context_gold=facts["semantic_gold_cases"],
        source_bound_global_sample=facts["global_sample_ready"],
        global_sample=facts["global_sample"],
        pending_bindings=facts["binding_backlog"],
        native_questions=facts["pilot_questions"],
        native_issuers=facts["dependence_components"],
        seeds=facts["seeds"],
        conditions=facts["corpus_conditions"],
        retrieval_episodes=facts["native_retrieval_episodes"],
        unique_snapshots=facts["snapshot_parity"],
        semantic_primary=None,
        proxy_by_strategy=native["native_strategy_metrics"],
        http_attempts=native["http_attempts"],
        known_tokens=native["known_tokens"],
        missing_inference_usage=native["unknown_inference_usage_attempts"],
        federation="application-level only",
        native_composite_status="NOT VERIFIED:reported standard;constituent aliases unsupported",
        assertion_scope="source metadata,not reviewed enterprise business-fact graph",
    )
    save(output / "frozen-observations.json", checks)
    save(
        output / "source-of-truth.json",
        dict(
            upload=str(handoff),
            frozen_sources={k: str(v.relative_to(package)) for k, v in inputs.items()},
            new_directive=str(
                (handoff / "CODEX_MASTER_DIRECTIVE.md").relative_to(package)
            ),
            visual_storyboard=str(
                (handoff / "VISUAL_STORYBOARD.md").relative_to(package)
            ),
        ),
    )
    save(
        output / "environment-capabilities.json",
        dict(
            credentials={
                k: bool(os.environ.get(k))
                for k in [
                    "MARA_API_KEY",
                    "MARA_ADMISSION_LOCK",
                    "DOZER_URI",
                    "DOZER_PASSWORD",
                ]
            },
            new_live_calls=0,
            live_status="BLOCKED_CREDENTIAL_AND_REVIEW_GATES",
            native_composite="UNVERIFIED",
            renderer="cached artifact-tool2.8.99 available;cached LibreOffice needs hostGLIBC2.38 and is unavailable",
            no_gpu_or_kv_claims=True,
        ),
    )
    text = [
        "# Enterprise semantics gap audit",
        "",
        "Frozen empirical observations are preserved. New offline engineering and synthetic software expectations do not qualify semantic model effects.",
        "",
        "| Observation | Value | Source / scope |",
        "| --- | --- | --- |",
        f"| Original QA / nonblank / categories | {checks['original_questions']} / {checks['nonblank_qa']} / {checks['native_categories']} | S07–S09;native corpus |",
        "| Source-bound silver / independent semantic gold | 329 / 0 | S20/S09;annotation candidates,not executed model questions |",
        "| Stratified sample bindings | 11/160;149pending | S07;no replacement with convenient sources |",
        "| Latest native DS01 | 2questions,2issuers,3seeds;36conditions;504episodes;31snapshots | S09/S12–S15 |",
        "| Reference proxy | A/C1.0;B1.0→0.667;all strategies tie | S10/S11/S13;semantic primary null |",
        "| Packing | Gold in corpus/candidates;excluded by top5/12000UTF8byte packing | latest native witness,S12/S22/S28;answer leakage unmeasured |",
        "| Older E3 | PANWoracle/candidate0.5;empty1.0/reference_defect;CBOEemptytimeout | S16/S17;uncalibrated,not latest sampler rerun |",
        "| Telemetry | 73HTTP incl2catalog;255600known tokens;5inference usage missing | S10;not the earlier46-call campaign |",
        "| Dozer | StandardmultiDB/parity;Composite type standard,aliases unsupported;existence constraint failed | S26/S27;application-level federation |",
        "| Extraction smoke | 2/2accepted;5normalized span-valid facts | S05/S18;not GroundedFactF1 |",
        "",
        "## Missing research evidence",
        "",
        "Independent identity/context/fact/sufficiency/claim labels,reviewed distractors and frozen human-calibrated evaluator are missing. Separate MatrixE and MatrixA preregistrations and synthetic tests can proceed. No broad paid matrix or ontology-benefit claim is qualified. FinDER filing-internal domains are not distinct ESG/legal providers. Auxiliary datasets remain separate tracks,not enterprise gold.",
        "",
        "## Protocol and presentation risks",
        "",
        "Old B0(no-evidence) versus DS01 B0(singleDBglobal),old top10 design versus actualtop5,oldfull5832passage plan versus actual346sourcepool/16docs are distinct. E3olderA/sampler is not latestv2ABC end-to-end. Aggregate relation JSD is not target neighborhood JSD. Unknown is never0. Preserve old120-slide deck and all28table cells. A derived import copy may normalize unsupported table anchor mid→ctr without touching source bytes.",
    ]
    (output / "GAP_AUDIT.md").write_text("\n".join(text) + "\n")
    return before, inputs


def prepare(package, handoff, output):
    package, handoff, output = (
        Path(package).resolve(),
        Path(handoff).resolve(),
        Path(output).resolve(),
    )
    output.mkdir(parents=True, exist_ok=True)
    before, inputs = audit_frozen(package, handoff, output)
    ontology = Ontology.model_validate_json(
        (package / "ontology/enterprise_relations.yaml").read_text()
    )
    snapshot, cases = synthetic_fixture(ontology)
    save(output / "synthetic-graph.json", snapshot.model_dump())
    save(output / "synthetic-interactions.json", [c.model_dump() for c in cases])
    changed = relevant_counterfactual(snapshot, ontology)
    save(
        output / "synthetic-relevant-counterfactual.json",
        dict(
            protocol_id="SYN-COUNTERFACTUAL-01",
            synthetic=True,
            parent_snapshot_id=snapshot.snapshot_id,
            changed_snapshot=changed.model_dump(),
            changed_assertion_id="syn-a-consolidated",
            expected_value_before="800",
            expected_value_after="810",
            semantic_model_success=None,
            scope="Authored source-version intervention;not a real filing or empirical model result.",
        ),
    )
    # No migration maps legacy MENTIONS_* metadata into enterprise predicates.
    views = {
        p: graph_records(snapshot, p)
        for p in ["global_lossy", "single_qualified", "multi_qualified"]
    }
    raw = [r["qualified_payload_hash"] for r in views["single_qualified"]]
    if any(
        [r["qualified_payload_hash"] for r in rows] != raw for rows in views.values()
    ):
        raise ValueError("identity policy changed frozen assertion ledger")
    save(
        output / "synthetic-identity-parity.json",
        dict(
            protocol_id="SYN-IDCTX-01",
            synthetic=True,
            graph_snapshot_id=snapshot.snapshot_id,
            variants=views,
            qualified_payload_parity=True,
            semantic_fact_retention=None,
            scope="Authored software fixture;lossy retrieval view is intentional,not empirical benefit.",
        ),
    )
    bundle = Bundle.model_validate_json(
        (
            package / "reports/finder_candidates_v1/expanded_frame/frame.private.json"
        ).read_text()
    )
    population = [
        json.loads(line)
        for line in (
            package / "reports/finder_candidates_v1/bound_review.private.jsonl"
        )
        .read_text()
        .splitlines()
    ]
    selected = {r["query_id"] for r in population} | {"b33fcee7", "ea48df6d"}
    packet = build_packet(bundle, selected)
    reviewroot = output / "annotation"
    reviewroot.mkdir()
    export_review_forms(packet, bundle, reviewroot)
    coverage = dict(
        original_qa=5703,
        qa_nonblank=5696,
        silver329=len(bundle.cases),
        independent_gold=0,
        annotation_cases=len(selected),
        targets=len(packet.targets),
        target_kinds=dict(Counter(t.kind for t in packet.targets)),
        global_sample=160,
        global_bound=11,
        pending_bindings=149,
        bound_risk_candidates=4,
        reviewer_submissions_received=0,
        adjudicated_targets=0,
        cohen_kappa=None,
        source_binding_backlog="reports/finder_candidates_v1/binding_backlog.private.jsonl",
        interpretation="Separate populations,not one nested sampling funnel. No independent review or agreement score was generated.",
    )
    save(output / "annotation-coverage.json", coverage)
    for cls in [
        Ontology,
        RelationDefinition,
        QualifiedAssertion,
        ExtractionResult,
        Snapshot,
        InteractionCase,
        AgentResponse,
        EntityIdentityGold,
        ContextCompatibilityGold,
        FactSupportGold,
        MinimalSufficientEvidenceSets,
        AnswerClaimGold,
        AnnotationPacket,
        ReviewerSubmission,
        AdjudicationReceipt,
        MatrixConfig,
    ]:
        path = (
            package
            / "schemas"
            / (
                ("qualified_assertion" if cls is QualifiedAssertion else cls.__name__)
                + ".schema.json"
            )
        )
        # These are new versioned engineering schemas, not frozen empirical files.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cls.model_json_schema(), indent=2) + "\n")
    summaries = {}
    for study in ["E", "A"]:
        config = MatrixConfig.model_validate_json(
            (package / f"configs/enterprise/matrix_{study}.json").read_text()
        )
        summaries[study] = run_mock(
            config,
            snapshot,
            cases,
            ontology,
            output / f"mock_matrix_{study}",
            fail_first=True,
        )
        units = 1 if study == "E" else len(cases)
        gate = qualify_live(
            config,
            gold=None,
            calibration=None,
            capabilities=None,
            environ=os.environ,
            requested_calls=budget(config, units)["logical_generation_calls"],
        )
        save(output / f"matrix_{study}_admission.json", gate)
    save(output / "matrix-budget-summary.json", summaries)
    precision = dict(
        version="enterprise-confirmation-design-v1",
        primary_contrasts=[
            "E:O2-O1",
            "A:A1-A0",
            "A:A2-A1",
            "placement:B3-B1",
            "metadata:B3full-B3reduced",
        ],
        noninferiority_margin_absolute=0.02,
        decision_rule="lower95%CI of signed candidate-minus-control primary success exceeds-0.02",
        independent_unit="issuer/shared-source connected component;crossed corpus seed",
        variance_estimate=None,
        minimum_actual_independent_clusters=None,
        qualified_sample_size=None,
        planning_only_normal_halfwidth_example=math.ceil(
            NormalDist().inv_cdf(0.975) ** 2 * 1 / 0.02**2
        ),
        planning_assumptions="Worst-case paired difference variance1;independent normal approximation;precision illustration,not cluster-aware power or selected sample size.",
        status="BLOCKED_REVIEWED_PILOT_VARIANCE_AND_CLUSTER_DESIGN",
    )
    save(output / "confirmation-preregistration.json", precision)
    after = {
        str(p.relative_to(package)): file_hash(p)
        for base in [
            "reports/ds01_campaign",
            "reports/finder_candidates_v1",
            "reports/final_review_v1",
        ]
        for p in (package / base).rglob("*")
        if p.is_file()
    }
    if after != before:
        raise ValueError("frozen empirical files changed")
    save(
        output / "offline-delivery.json",
        dict(
            frozen_files_verified=len(before),
            frozen_unchanged=True,
            synthetic_assertions=len(snapshot.assertions),
            synthetic_cases=len(cases),
            annotation=coverage,
            mock=summaries,
            semantic_primary=None,
            new_http_calls=0,
            new_graph_calls=0,
            gold_fabricated=False,
        ),
    )
    return dict(
        synthetic_assertions=len(snapshot.assertions),
        review_cases=len(selected),
        targets=len(packet.targets),
        mock_runs=summaries,
        frozen_unchanged=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--package", type=Path, default=Path.cwd())
    prep.add_argument("--handoff", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    review = sub.add_parser("adjudicate")
    review.add_argument("--packet", type=Path, required=True)
    review.add_argument("--reviewer-a", type=Path, required=True)
    review.add_argument("--reviewer-b", type=Path, required=True)
    review.add_argument("--adjudication", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    matrix = sub.add_parser("matrix")
    matrix.add_argument("--config", type=Path, required=True)
    matrix.add_argument("--snapshot", type=Path, required=True)
    matrix.add_argument("--cases", type=Path, required=True)
    matrix.add_argument("--ontology", type=Path, required=True)
    matrix.add_argument("--output", type=Path, required=True)
    matrix.add_argument("--live", action="store_true")
    matrix.add_argument("--package", type=Path, default=Path.cwd())
    for key in ["gold", "calibration", "capabilities", "profile-review"]:
        matrix.add_argument("--" + key, type=Path)
    args = p.parse_args()
    if args.command == "prepare":
        result = prepare(args.package, args.handoff, args.output)
    elif args.command == "adjudicate":
        packet = AnnotationPacket.model_validate_json(args.packet.read_text())
        a = ReviewerSubmission.model_validate_json(args.reviewer_a.read_text())
        b = ReviewerSubmission.model_validate_json(args.reviewer_b.read_text())
        receipt = AdjudicationReceipt.model_validate_json(args.adjudication.read_text())
        result = adjudicate(packet, a, b, receipt)
        save(args.output, result)
    else:
        config = parse_config(args.config.read_text())
        profiles = (
            load_bundle(args.package / config.ontology_profile_path)
            if config.version == "enterprise-matrix-v2"
            else None
        )
        snapshot = Snapshot.model_validate_json(args.snapshot.read_text())
        cases = [
            InteractionCase.model_validate(c)
            for c in json.loads(args.cases.read_text())
        ]
        ontology = Ontology.model_validate_json(args.ontology.read_text())
        snapshot.validate_ontology(ontology)
        if args.live:

            def load(p):
                return json.loads(p.read_text()) if p else None

            result = run_live(
                config,
                snapshot,
                cases,
                ontology,
                args.output,
                gold=load(args.gold),
                calibration=load(args.calibration),
                capabilities=load(args.capabilities),
                environ=os.environ,
                profiles=profiles,
                profile_review=load(args.profile_review),
            )
        else:
            result = run_mock(
                config,
                snapshot,
                cases,
                ontology,
                args.output,
                fail_first=True,
                profiles=profiles,
            )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
