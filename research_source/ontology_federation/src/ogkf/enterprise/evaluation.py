"""Versioned, review-dependent metric primitives. Missing labels are not zero."""

VERSION = "enterprise-evaluator-v1"


def evidence_sufficiency(delivered_ids, alternatives, *, reviewed=False):
    if not reviewed or alternatives is None:
        return None
    if not alternatives or any(not s for s in alternatives):
        raise ValueError("complete minimal sets required")
    delivered = set(delivered_ids)
    return any(set(s) <= delivered for s in alternatives)


def grounded_qa_success(*, answer_correct, sufficient, claims):
    """Each material claim must have reviewed support, context, composition and citation."""
    fields = [
        "source_supported",
        "context_compatible",
        "composition_valid",
        "citation_binding_correct",
    ]
    if answer_correct is None or sufficient is None or claims is None:
        return None
    if any(c.get(k) is None for c in claims for k in fields):
        return None
    if answer_correct and not claims:
        raise ValueError("a substantive correct answer requires material claims")
    return bool(
        answer_correct and sufficient and all(all(c[k] for k in fields) for c in claims)
    )


def grounded_fact_f1(
    *, matched_supported_gold, reviewed_gold_total, prediction_statuses
):
    """Use independent match/support adjudications, never closed-world gold absence.

    prediction_statuses: each prediction is TP, FP, or UNKNOWN after review.
    Unresolved predictions make the primary unqualified, with coverage exposed.
    matched_supported_gold counts distinct matched gold IDs (not duplicate predictions).
    """
    if prediction_statuses is None or reviewed_gold_total is None:
        return dict(precision=None, recall=None, f1=None, review_coverage=None)
    if reviewed_gold_total <= 0:
        raise ValueError("positive independently reviewed gold denominator required")
    if any(s not in {"TP", "FP", "UNKNOWN"} for s in prediction_statuses):
        raise ValueError("unknown adjudication status")
    if not 0 <= matched_supported_gold <= reviewed_gold_total:
        raise ValueError("invalid unique gold matches")
    total = len(prediction_statuses)
    known = sum(s != "UNKNOWN" for s in prediction_statuses)
    coverage = known / total if total else 1.0
    if "UNKNOWN" in prediction_statuses:
        return dict(precision=None, recall=None, f1=None, review_coverage=coverage)
    tp = prediction_statuses.count("TP")
    if tp != matched_supported_gold:
        raise ValueError(
            "TP predictions require one-to-one distinct reviewed gold matches"
        )
    precision = tp / total if total else 0.0
    recall = matched_supported_gold / reviewed_gold_total
    return dict(
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        review_coverage=coverage,
    )


def context_retention(
    expected_reviewed_ids, retained_context_compatible_ids, *, reviewed=False
):
    if not reviewed or expected_reviewed_ids is None:
        return None
    expected = set(expected_reviewed_ids)
    if not expected:
        raise ValueError("empty reviewed retention denominator")
    return len(expected & set(retained_context_compatible_ids)) / len(expected)
