"""Blinded source-fact and answer-claim capture for an actual scenario run."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from ..enterprise.annotation import (
    AnnotationPacket,
    AnswerClaimGold,
    FactSupportGold,
    ReviewTarget,
    pending_submission,
)
from ..models import digest
from .challenge import jsonl, save
from .semantic_review import read_lines


def export(query_run, output):
    snapshot = json.loads((query_run / "input-snapshot.json").read_text())
    sources, lookup = [], {}
    for d in snapshot["source_documents"]:
        sid = digest([d["document_id"], d["content_hash"], 0, len(d["text"])])
        lookup[d["document_id"]] = sid
        sources.append(
            {
                "span_id": sid,
                "document_id": d["document_id"],
                "document_version": d["content_hash"],
                "source_hash": d["content_hash"],
                "start": 0,
                "end": len(d["text"]),
                "quote": d["text"],
                "evidence_id": d["evidence_id"],
            }
        )
    targets, drafts, private_map = [], [], []
    seen = set()
    for f in snapshot["qualified_assertion_proposals"]:
        if f["fact_id"] in seen:
            continue
        seen.add(f["fact_id"])
        tid = digest(["source-scenario-fact-review-v1", f["fact_id"]])
        proposal = FactSupportGold(
            kind="fact_support_gold",
            target_id=tid,
            source_span_ids=[lookup[f["document_id"]]],
            rationale="",
            subject=f["subject"],
            predicate=f["predicate"],
            object_value=f["object_value"],
            qualifiers=None,
            verdict="UNKNOWN",
        )
        targets.append(
            ReviewTarget(
                target_id=tid,
                query_id="source-fact:" + f["document_id"],
                kind=proposal.kind,
                source_span_ids=proposal.source_span_ids,
                proposal=proposal,
            )
        )
        drafts.append(
            {
                "target_id": tid,
                "subject": f["subject"],
                "predicate": f["predicate"],
                "object_value": f["object_value"],
                "subject_type": f["subject_type"],
                "object_type": f["object_type"],
                "period": f["period"],
                "scope": f["scope"],
                "unit": f["unit"],
                "modality": f["modality"],
                "source_quote": f["quote"],
                "source_start": f["start"],
                "source_end": f["end"],
                "document_id": f["document_id"],
            }
        )
        private_map.append(
            {
                "target_id": tid,
                "fact_id": f["fact_id"],
                "extraction_model": f["model"],
                "scenario": f["scenario"],
                "request_id": f["request_id"],
            }
        )
    for row in read_lines(query_run / "evaluation_results.jsonl"):
        if "answer" not in row:
            continue
        answer = row["answer"]
        decision = answer["normalized_decision"]
        # The action is reviewed as well as each individual claim, including abstentions.
        for claim in [None, *decision["claims"]]:
            tid = digest(
                [
                    "source-scenario-answer-review-v1",
                    answer["request_id"],
                    claim["claim_id"] if claim else None,
                ]
            )
            proposal = AnswerClaimGold(
                kind="answer_claim_gold",
                target_id=tid,
                query_id=row["query_id"],
                claim_id=claim["claim_id"] if claim else None,
                answer_hash=digest(decision),
                source_span_ids=sorted(lookup.values()),
                rationale="",
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
                    query_id=row["query_id"],
                    kind=proposal.kind,
                    source_span_ids=proposal.source_span_ids,
                    proposal=proposal,
                )
            )
            drafts.append(
                {
                    "target_id": tid,
                    "query_id": row["query_id"],
                    "claim": claim,
                    "actual_decision_to_judge": decision,
                    "delivered_citations": answer["claim_source_links"],
                    "target_scope": "individual claim"
                    if claim
                    else "whole response and action",
                }
            )
            private_map.append(
                {
                    "target_id": tid,
                    "extraction_model": row["model"],
                    "scenario": row["scenario"],
                    "strategy": row["strategy"],
                    "answer_request_id": answer["request_id"],
                }
            )
    packet = AnnotationPacket(
        version="source-scenario-independent-review-v1",
        source_bundle_hash=digest(snapshot),
        sources=sources,
        targets=targets,
        native_gold_policy="Native answers unchanged and withheld here. Source-fact and actual answer/action correctness require actual independent source review; model identities, scenario arms and strategy assignments are withheld from the blinded draft file.",
        synthetic=False,
    )
    output.mkdir(parents=True, exist_ok=False)
    save(output / "annotation.packet.json", packet.model_dump())
    save(
        output / "reviewer.template.PENDING.json",
        pending_submission(packet, "REPLACE_WITH_ACTUAL_REVIEWER_ID").model_dump(),
    )
    jsonl(output / "drafts.blinded.jsonl", drafts)
    jsonl(output / "assignment.evaluator-only.jsonl", private_map)
    jsonl(output / "questions.agent.jsonl", snapshot["public_questions"])
    manifest = {
        "packet_hash": packet.packet_hash,
        "targets": len(targets),
        "targets_by_kind": dict(Counter(t.kind for t in targets)),
        "independent_reviews_received": 0,
        "adjudicated_targets": 0,
        "models_scenarios_strategies_withheld_from_blinded_drafts": True,
        "source_snapshot_hash": digest(snapshot),
        "file_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.iterdir()
            if p.is_file()
        },
    }
    save(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.query_run, args.output), indent=2))


if __name__ == "__main__":
    main()
