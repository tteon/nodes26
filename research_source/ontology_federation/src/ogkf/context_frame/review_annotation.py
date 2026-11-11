"""Source-bound independent capture for the complete question population.

The export deliberately excludes automated review labels and rationales. Existing
enterprise annotation validation/adjudication applies unchanged; no gold is made.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from ..enterprise.annotation import (
    AnnotationPacket,
    AnswerClaimGold,
    ContextCompatibilityGold,
    MinimalSufficientEvidenceSets,
    ReviewTarget,
    pending_submission,
)
from ..enterprise.contracts import DIMENSIONS
from ..models import digest
from .challenge import jsonl, save
from .semantic_review import read_lines


def make_packet(questions, documents):
    sources, lookup = [], {}
    for doc in documents:
        sid = digest([doc["document_id"], doc["content_hash"], 0, len(doc["text"])])
        lookup[doc["document_id"]] = sid
        sources.append(
            {
                "span_id": sid,
                "document_id": doc["document_id"],
                "document_version": doc["content_hash"],
                "source_hash": doc["content_hash"],
                "start": 0,
                "end": len(doc["text"]),
                "quote": doc["text"],
                "evidence_id": doc["evidence_id"],
            }
        )
    targets = []
    for q in questions:
        common = {
            "source_span_ids": [lookup[r] for r in q["source_ids"]],
            "rationale": "",
            "query_id": q["query_id"],
        }
        for kind in (
            "context_compatibility_gold",
            "minimal_sufficient_evidence_sets",
            "answer_claim_gold",
        ):
            tid = digest(["full-question-independent-capture-v1", q["query_id"], kind])
            base = {**common, "kind": kind, "target_id": tid}
            if kind == "context_compatibility_gold":
                proposal = ContextCompatibilityGold(
                    **base,
                    operation=None,
                    dimensions={d: "UNKNOWN" for d in DIMENSIONS},
                )
            elif kind == "minimal_sufficient_evidence_sets":
                proposal = MinimalSufficientEvidenceSets(
                    **base, operation=None, evidence_sets=[], verdict="UNKNOWN"
                )
            else:
                proposal = AnswerClaimGold(
                    **base,
                    claim_id=None,
                    answer_hash=digest(q["native_answer"])
                    if q.get("native_answer")
                    else None,
                    allowed_actions=[],
                    answer_correct=None,
                    source_supported=None,
                    context_compatible=None,
                    composition_valid=None,
                    citation_binding_correct=None,
                    material_missing_dimensions=[],
                )
            targets.append(
                ReviewTarget(
                    target_id=tid,
                    query_id=q["query_id"],
                    kind=kind,
                    source_span_ids=common["source_span_ids"],
                    proposal=proposal,
                )
            )
    return AnnotationPacket(
        version="finder-all-question-independent-capture-v1",
        source_bundle_hash=digest(
            {
                "documents": sorted(
                    (d["document_id"], d["content_hash"]) for d in documents
                ),
                "questions": [
                    (q["query_id"], q["text"], q.get("native_answer"))
                    for q in questions
                ],
            }
        ),
        sources=sources,
        targets=targets,
        native_gold_policy="Published native QA remains unchanged and usable separately. Empty and generated answers remain missing; context/action/sufficient-set labels require actual independent review. Model-review judgments are withheld from this packet.",
        synthetic=False,
    )


def export(review, output):
    manifest = json.loads((review / "manifest.json").read_text())
    for name, key in (
        ("questions.evaluator-only.jsonl", "question_data_sha256"),
        ("documents.jsonl", "document_data_sha256"),
    ):
        if hashlib.sha256((review / name).read_bytes()).hexdigest() != manifest[key]:
            raise ValueError("frozen independent-review inputs changed")
    questions = read_lines(review / "questions.evaluator-only.jsonl")
    documents = read_lines(review / "documents.jsonl")
    packet = make_packet(questions, documents)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "annotation.packet.json", packet.model_dump())
    # One blank template can be copied by each real reviewer; it is not a submission.
    save(
        output / "reviewer.template.PENDING.json",
        pending_submission(packet, "REPLACE_WITH_ACTUAL_REVIEWER_ID").model_dump(),
    )
    jsonl(
        output / "questions.agent.jsonl",
        [{"query_id": q["query_id"], "text": q["text"]} for q in questions],
    )
    jsonl(
        output / "native.answers.evaluator-only.jsonl",
        [
            {
                "query_id": q["query_id"],
                "published_answer": q.get("native_answer"),
                "published_answer_hash": digest(q["native_answer"])
                if q.get("native_answer")
                else None,
            }
            for q in questions
        ],
    )
    result = {
        "questions": len(questions),
        "sources": len(documents),
        "targets": len(packet.targets),
        "targets_by_kind": dict(Counter(t.kind for t in packet.targets)),
        "packet_hash": packet.packet_hash,
        "independent_reviews_received": 0,
        "adjudicated_targets": 0,
        "model_labels_and_rationales_withheld": True,
        "source_domain_partition_is_gold": False,
        "partial_real_submissions_supported": True,
        "file_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir()
            if p.is_file()
        },
    }
    save(output / "manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.review, args.output), indent=2))


if __name__ == "__main__":
    main()
