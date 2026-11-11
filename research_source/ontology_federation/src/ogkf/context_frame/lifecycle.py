"""Versioned review and cached E2/E3 evaluation; never rewrites an earlier run."""
from ..models import digest
from ..ds01.semantic import AnswerJudgment, score_claims
from .compiler import calculations, normalized
from .models import Bundle, Case, CaseReview, ReviewedFields, ReviewProvenance, ReviewReceipt


def verify_origins(bundle, cohort):
    if bundle.cohort_dataset_hash != cohort.dataset_hash:
        raise ValueError("frame is for another frozen cohort dataset")
    queries, gold = {q.query_id:q for q in cohort.queries}, {g.query_id:g for g in cohort.gold}
    docs = {d.document_id:d for d in cohort.documents}
    if set(queries) != {c.query.query_id for c in bundle.cases}:
        raise ValueError("cohort/case coverage mismatch")
    for d in bundle.documents:
        if d.document_id not in docs or d != docs[d.document_id]:
            raise ValueError("annotation changed an immutable source document")
    for c in bundle.cases:
        if c.query != queries[c.query.query_id] or c.original_gold != gold[c.query.query_id]:
            raise ValueError("annotation changed native question/answer/gold")
    by_doc = {d.document_id:d for d in bundle.documents}
    for h in bundle.hierarchy:
        if normalized(h.filing_quote) != normalized(by_doc[h.document_id].text):
            raise ValueError("filing/source locator normalization mismatch")
    return {"native_qa_preserved": True, "source_documents_preserved": True,
        "span_count": len(bundle.spans), "hierarchy_locators": len(bundle.hierarchy),
        "cases":len(bundle.cases), "gold_cases":sum(c.annotation_status=="reviewed_gold" for c in bundle.cases)}


def pending_review(bundle):
    fields = set(ReviewedFields.model_fields)-{"sufficiency_reviewed"}
    return ReviewReceipt(bundle_hash=bundle.bundle_hash, independent_review_complete=False,
        reviewer_ids=[], adjudication_record=None,
        cases=[CaseReview(case_id=c.case_id, judgments=ReviewedFields(
            **{k:c.model_dump()[k] for k in fields}, sufficiency_reviewed=False)) for c in bundle.cases])


def apply_review(bundle, receipt):
    if receipt.bundle_hash != bundle.bundle_hash:
        raise ValueError("review bound to another bundle")
    if (not receipt.independent_review_complete or len({r.strip() for r in receipt.reviewer_ids})<2
            or not all(r.strip() for r in receipt.reviewer_ids) or not receipt.adjudication_record
            or not receipt.adjudication_record.strip()
            or not receipt.cases):
        raise ValueError("actual independent review and adjudication required; pending is not gold")
    ids = [r.case_id for r in receipt.cases]
    if len(ids)!=len(set(ids)) or not set(ids)<={c.case_id for c in bundle.cases}:
        raise ValueError("duplicate/unknown review case")
    review_hash = digest(receipt.model_dump())
    changes = {r.case_id:r.judgments for r in receipt.cases}
    result = bundle.model_dump()
    for i, c in enumerate(bundle.cases):
        if c.case_id not in changes:
            continue
        j = changes[c.case_id]
        if not j.rationale_steps or not j.compatibility_rules or not all(r.rationale.strip() for r in j.compatibility_rules):
            raise ValueError("review lacks source-based rules or rationale obligations")
        if not any(r.decision!="UNKNOWN" for r in j.compatibility_rules) or not any(d.decision!="UNKNOWN" for d in j.evidence_decisions):
            raise ValueError("unchanged pending decisions cannot attest a completed review")
        if any(d.decision!="UNKNOWN" and not d.rationale.strip() for d in j.evidence_decisions):
            raise ValueError("reviewed eligibility needs a source-based rationale")
        if j.sufficiency_reviewed:
            needed = {e for group in j.support_groups for alt in group for e in alt}
            decisions = {d.evidence_id:d.decision for d in j.evidence_decisions}
            if not needed or any(decisions.get(e) not in {"DIRECT_SUPPORT","PERMITTED_COMPOSITION"} for e in needed):
                raise ValueError("reviewed sufficient groups need explicitly eligible supporting evidence")
            if any(r.decision=="UNKNOWN" for r in j.compatibility_rules):
                raise ValueError("context/composition rules not fully adjudicated")
        payload = {**c.model_dump(), **j.model_dump(exclude={"sufficiency_reviewed"}),
            "annotation_status":"reviewed_gold", "author_method":"independent_review", "review_record":review_hash,
            "support_status":"independently_reviewed" if j.sufficiency_reviewed else "provided_reference_proxy"}
        # Reviewers may remove an irrelevant draft row. Recompute conditional arithmetic
        # from the retained source cells instead of leaving orphaned draft candidates.
        payload["calculation_candidates"]=[r.model_dump() for r in calculations(c.query.text,j.operation_candidate,
            j.table_rows,{s.span_id:s for s in bundle.spans})]
        result["cases"][i] = Case.model_validate(payload).model_dump()
    result["review_provenance"].append(ReviewProvenance(input_bundle_hash=bundle.bundle_hash,
        receipt_hash=review_hash, reviewer_ids=receipt.reviewer_ids, adjudication_record=receipt.adjudication_record).model_dump())
    return Bundle.model_validate(result)


def groups_satisfied(groups, delivered):
    return bool(groups) and all(any(set(alt)<=delivered for alt in group) for group in groups)


class FrameEvaluator:
    """Freeze a private frame once; do not hash megabytes again for every sample."""
    def __init__(self,bundle):
        self._bundle=Bundle.model_validate(bundle.model_dump())
        self._fingerprint=self._bundle.bundle_hash

    def evaluate(self,episode,*,stage,claim_review=None):
        return _evaluate_episode(self._bundle,episode,stage=stage,claim_review=claim_review,
                                 fingerprint=self._fingerprint)


def evaluate_episode(bundle,episode,*,stage,claim_review=None):
    return FrameEvaluator(bundle).evaluate(episode,stage=stage,claim_review=claim_review)


def _evaluate_episode(bundle, episode, *, stage, claim_review, fingerprint):
    case = next((c for c in bundle.cases if c.query.query_id==episode.get("query_id")), None)
    if case is None:
        raise ValueError("episode has no context frame")
    if stage not in {"retrieval", "agent"}:
        raise ValueError("unsupported stage")
    key = "evidence_ids" if stage=="retrieval" else "context_evidence_ids"
    if key not in episode:
        raise ValueError("delivered evidence lineage missing")
    delivered = set(episode[key])
    all_evidence = {d.evidence_id for d in bundle.documents}
    # Historical retrieval may contain other verified pool documents outside this review packet.
    unknown_review_evidence = sorted(delivered-all_evidence)
    independent = case.annotation_status=="reviewed_gold"
    semantic = independent and case.support_status=="independently_reviewed"
    judgments = {d.evidence_id:d.decision for d in case.evidence_decisions}
    incompatible = {e for e,d in judgments.items() if d=="INCOMPATIBLE"}
    complete_context = independent and all(judgments.get(e,"UNKNOWN")!="UNKNOWN" for e in delivered)
    parent = episode["sample_id"]
    result = {"evaluation_id":digest([VERSION_KEY, fingerprint, stage, episode.get("experiment_run_id"), parent]),
        "evaluator_version":VERSION_KEY, "frame_bundle_hash":fingerprint, "case_id":case.case_id,
        "query_id":case.query.query_id, "parent_sample_id":parent, "stage":stage,
        "parent_experiment_run_id":episode.get("experiment_run_id"),
        **{k:episode.get(k) for k in ("condition_id","graph_snapshot_id","trace_id","seed","strategy","kind")},
        "annotation_status":case.annotation_status, "original_execution_error":episode.get("error"),
        "provided_reference_sufficiency_proxy":groups_satisfied(case.original_required_groups,delivered),
        "evidence_sufficiency_at_k":groups_satisfied(case.support_groups,delivered) if semantic else None,
        "context_incompatible_evidence_exposure_ratio":len(delivered & incompatible)/len(delivered)
            if complete_context and delivered else None,
        "context_review_coverage":sum(judgments.get(e,"UNKNOWN")!="UNKNOWN" for e in delivered)/len(delivered)
            if independent and delivered else (0. if delivered else None),
        "outside_frame_evidence_ids":unknown_review_evidence,
        "context_leakage_rate":None, "grounded_qa_success":None, "rationale_score":None,
        "rationale_status":"Needs independent claim/operation judgment or calibrated evaluator; no text-similarity substitute",
        "delivered_evidence_ids":sorted(delivered)}
    if stage=="agent":
        claims = score_claims(episode, result, claim_review)
        result.update({k:claims[k] for k in ("context_leakage_rate","grounded_qa_success","evidence_attribution_accuracy","claim_review_coverage")})
        result["prior_automatic_judge_result"] = episode.get("judge_result")
        result["prior_judge_calibrated"] = episode.get("judge_calibrated",False)
    return result


VERSION_KEY = "finder-context-frame-evaluator-v1"


def pending_answer_review(episode):
    answer = episode.get("generated_answer")
    if not answer:
        return None
    return AnswerJudgment(version="ds01-independent-claims-v1", sample_id=episode["sample_id"],
        answer_hash=digest(answer), answer_correct=None, reviewer_ids=[], review_record=None,
        status="pending", claims=[dict(claim_id=c["claim_id"], source_supported=None,
            context_compatible=None, composition_valid=None, status="pending",
            rationale="Judge the cited source, question scope, operation and conclusion independently.") for c in answer["claims"]])
