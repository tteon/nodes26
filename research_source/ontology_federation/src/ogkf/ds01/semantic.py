"""Independent claim evaluation; absent judgments stay unknown, never synthetic gold."""
import argparse
import json
from pathlib import Path
from typing import Literal

from ..models import Strict, digest


class ClaimJudgment(Strict):
    claim_id: str
    source_supported: bool | None
    context_compatible: bool | None
    composition_valid: bool | None
    status: Literal["reviewed", "pending"]
    rationale: str


class AnswerJudgment(Strict):
    version: str
    sample_id: str
    answer_hash: str
    answer_correct: bool | None
    reviewer_ids: list[str]
    review_record: str | None
    status: Literal["reviewed", "pending"]
    claims: list[ClaimJudgment]


def score_claims(episode, retrieval_score, review: AnswerJudgment | None):
    output = {"version": "ds01-independent-claims-v1", "sample_id": episode["sample_id"],
              "grounded_qa_success": None, "evidence_attribution_accuracy": None,
              "context_leakage_rate": None, "claim_review_coverage": None, "independent_review": False}
    answer = episode.get("generated_answer")
    if answer and answer["claims"]:
        output["claim_review_coverage"] = 0.
    if review is None or answer is None:
        return output
    if review.sample_id != episode["sample_id"] or review.answer_hash != digest(answer):
        raise ValueError("review bound to another sample/answer")
    if (review.status != "reviewed" or len({r.strip() for r in review.reviewer_ids}) < 2
            or not all(r.strip() for r in review.reviewer_ids)
            or not review.review_record or not review.review_record.strip()):
        return output
    if "context_evidence_ids" not in episode:
        raise ValueError("delivered evidence lineage missing")
    delivered = set(episode["context_evidence_ids"])
    citations_valid = all(c["evidence_ids"] and set(c["evidence_ids"] + c["operand_evidence_ids"]) <= delivered
                          for c in answer["claims"])
    output["citation_ids_valid"] = citations_valid
    ids = [c["claim_id"] for c in answer["claims"]]
    reviewed_ids = [c.claim_id for c in review.claims]
    if len(ids) != len(set(ids)) or len(reviewed_ids) != len(set(reviewed_ids)) or not set(reviewed_ids) <= set(ids):
        raise ValueError("duplicate/unknown claim identity")
    known = [c for c in review.claims if c.status == "reviewed" and all(v is not None for v in
             (c.source_supported, c.context_compatible, c.composition_valid))]
    if any(not c.rationale.strip() for c in known):
        raise ValueError("reviewed claims need a source-based rationale")
    output["independent_review"] = True
    output["claim_review_coverage"] = len(known) / len(ids) if ids else None
    if known:
        output["evidence_attribution_accuracy"] = sum(
            c.source_supported and c.context_compatible and c.composition_valid for c in known) / len(known)
        # Unsupported invention and source-supported misattribution are separate errors.
        context_known = [c for c in known if c.source_supported]
        output["context_leakage_rate"] = (sum(not c.context_compatible or not c.composition_valid for c in context_known) /
                                           len(context_known)) if context_known else None
    sufficient = retrieval_score.get("evidence_sufficiency_at_k")
    if sufficient is not None and review.answer_correct is not None and len(known) == len(ids):
        output["grounded_qa_success"] = bool(sufficient and review.answer_correct and ids and citations_valid
            and all(c.source_supported and c.context_compatible and c.composition_valid for c in known))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--retrieval-score", type=Path, required=True)
    p.add_argument("--review", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    review = AnswerJudgment.model_validate_json(args.review.read_text()) if args.review else None
    result = score_claims(json.loads(args.episode.read_text()), json.loads(args.retrieval_score.read_text()), review)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
