"""Versioned exact-qualified-fact metric; incomplete gold never implies falsehood."""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import validate

from .models import digest

VERSION = "qualified-exact-v1"
KEYS = ("source_document_id", "source_version", "subject", "predicate", "literal_value",
        "domain_id", "temporal_scope", "business_scope", "unit")


def fact_key(fact):
    return tuple(fact[k] for k in KEYS)


def evaluate(predictions, gold):
    """Gold positives require independent reviewed labels; unknowns are unscored.

    This narrow evaluator deliberately requires exact canonical strings. It does
    not claim general semantic evaluation or score a source quote as entailment.
    """
    base = Path(__file__).resolve().parents[2]
    validate(gold, json.loads((base / "contracts/gold_schema.json").read_text()))
    if gold["evaluator_version"] != VERSION:
        raise ValueError("evaluator version mismatch")
    positives = {fact_key(f): f for f in gold["assertions"] if f["annotation_status"] == "supported"}
    negatives = {fact_key(f) for f in gold["assertions"] if f["annotation_status"] == "unsupported"}
    if positives.keys() & negatives:
        raise ValueError("contradictory gold labels")
    emitted = {}
    for prediction in predictions:
        emitted.setdefault(fact_key(prediction), []).append(prediction)
    # Matching a fact requires an exact approved supporting span as well.
    tp = sum(k in positives and any(p["evidence_start"] == s["start"] and p["evidence_end"] == s["end"]
                                   and p["evidence_span"] == s["quote"]
                                   for p in values for s in positives[k]["approved_spans"])
             for k, values in emitted.items())
    fp = sum(k in negatives for k in emitted)
    unknown = len(emitted) - tp - fp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / len(positives) if positives else None
    complete = unknown == 0 and gold["prediction_review_complete"]
    f1 = (2 * tp / (len(positives) + tp + fp)) if complete and positives else None
    return {"evaluator_version": VERSION, "grounded_fact_f1": f1,
            "reviewed_precision": precision, "gold_recall": recall,
            "true_positive": tp, "reviewed_false_positive": fp, "unknown_predictions": unknown,
            "annotation_coverage": (len(emitted) - unknown) / len(emitted) if emitted else None,
            "annotation_qualified": bool(complete and gold["independent_review_complete"]),
            "qualified_for_hypothesis_testing": False,
            "scope": "exact qualified fact + approved source span; development calibration only"}


def calibration_check(receipt_path: str | None, dataset_path: Path):
    if not receipt_path:
        raise ValueError("human calibration receipt required before live ontology comparison or matrix")
    receipt = json.loads(Path(receipt_path).read_text())
    expected = {"dataset_sha256": digest(json.loads(dataset_path.read_text())),
                "evaluator_version": VERSION, "independent_review_complete": True,
                "human_evaluator_agreement": True,
                "evaluator_code_sha256": digest(Path(__file__).read_text()),
                "metric_contract_sha256": digest(json.loads((Path(__file__).resolve().parents[2] / "contracts/metric_contract.json").read_text()))}
    if not receipt.get("reviewer") or not receipt.get("adjudication_record"):
        raise ValueError("reviewer and adjudication record required")
    if any(receipt.get(k) != v for k, v in expected.items()):
        raise ValueError("calibration receipt does not qualify this dataset/evaluator")
    # The receipt is a human attestation, not cryptographic proof of independence.
    return receipt
