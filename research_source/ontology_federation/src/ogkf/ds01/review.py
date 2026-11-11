"""Build pending review queues and apply explicit independent review receipts.

No automated command here can author or attest human judgments.
"""
import argparse
import html
import json
from pathlib import Path
from typing import Literal

from pydantic import Field
from ..models import Strict, digest
from .contracts import Dataset


class NegativeReview(Strict):
    document_id: str
    kind: Literal["lexical", "relational", "temporal", "scope", "identity", "unclassified"]
    decision: Literal["PENDING", "NONANSWER", "POSSIBLY_RELEVANT"]
    rationale: str | None


class ContextReview(Strict):
    evidence_id: str
    decision: Literal["COMPATIBLE", "INCOMPATIBLE", "UNKNOWN"]
    rationale: str


class CaseReview(Strict):
    query_id: str
    operation: str
    support_status: Literal["PENDING", "REVIEWED"]
    sufficient_groups: list[list[list[str]]]
    negative_documents: list[NegativeReview]
    context_labels: list[ContextReview]
    rationale: str | None


class ReviewReceipt(Strict):
    version: str
    dataset_hash: str
    reviewer_ids: list[str]
    independent_review_complete: bool
    adjudication_record: str | None
    cases: list[CaseReview]


def pending_queue(dataset, max_documents=24):
    docs = {d.document_id: d for d in dataset.documents}
    cases = []
    for q in dataset.queries:
        g = next(g for g in dataset.gold if g.query_id == q.query_id)
        candidates = sorted((label for label in dataset.distractors if label.query_id == q.query_id),
                            key=lambda x: (x.kind, x.document_id))[:max_documents]
        cases.append(CaseReview(query_id=q.query_id, operation="PENDING_QUESTION_RELATIVE_OPERATION",
            support_status="PENDING", sufficient_groups=g.required_groups, context_labels=[], rationale=None,
            negative_documents=[NegativeReview(document_id=c.document_id, kind=c.kind, decision="PENDING", rationale=None)
                                for c in candidates]))
    return ReviewReceipt(version="ds01-independent-review-v1", dataset_hash=dataset.dataset_hash,
        reviewer_ids=[], independent_review_complete=False, adjudication_record=None, cases=cases)


def apply_review(dataset, receipt):
    if receipt.dataset_hash != dataset.dataset_hash:
        raise ValueError("review is for another frozen dataset")
    if (not receipt.independent_review_complete or len({r.strip().casefold() for r in receipt.reviewer_ids}) < 2
            or any(not r.strip() for r in receipt.reviewer_ids) or not receipt.adjudication_record
            or not receipt.adjudication_record.strip()):
        raise ValueError("actual independent review/adjudication records required; pending is not gold")
    public = {q.query_id: q for q in dataset.queries}
    gold = {g.query_id: g.model_dump() for g in dataset.gold}
    labels = {(r.query_id, r.document_id): r.model_dump() for r in dataset.distractors}
    evidence = {d.evidence_id for d in dataset.documents}
    if len({c.query_id for c in receipt.cases}) != len(receipt.cases):
        raise ValueError("duplicate case review")
    record = "review:" + digest(receipt.model_dump())
    for case in receipt.cases:
        if case.query_id not in public or case.operation.startswith("PENDING"):
            raise ValueError("unknown question or unreviewed operation")
        if case.support_status == "REVIEWED":
            union = {e for group in case.sufficient_groups for alternative in group for e in alternative}
            if not dataset.gold[[g.query_id for g in dataset.gold].index(case.query_id)].required_ids() <= union:
                raise ValueError("review cannot discard original provided evidence")
            gold[case.query_id].update(required_groups=case.sufficient_groups,
                sufficient_set_status="independently_reviewed", review_record=record)
        for row in case.negative_documents:
            key = case.query_id, row.document_id
            if key not in labels:
                raise ValueError("review names an unknown candidate")
            if row.decision == "NONANSWER":
                if not row.rationale:
                    raise ValueError("nonanswer decision requires source-based rationale")
                labels[key].update(kind=row.kind, status="independently_reviewed_nonanswer", review_record=record)
            elif row.decision == "POSSIBLY_RELEVANT":
                labels[key]["kind"] = "unclassified"
        for row in case.context_labels:
            if row.evidence_id not in evidence:
                raise ValueError("context review names unknown evidence")
        gold[case.query_id]["compatible_evidence_ids"] = [r.evidence_id for r in case.context_labels if r.decision == "COMPATIBLE"]
        gold[case.query_id]["incompatible_evidence_ids"] = [r.evidence_id for r in case.context_labels if r.decision == "INCOMPATIBLE"]
    payload = dataset.model_dump()
    payload.update(version=dataset.version + "-review-" + digest(receipt.model_dump())[:12],
        gold=list(gold.values()), distractors=list(labels.values()), annotation_status="independently_reviewed",
        input_hashes={**dataset.input_hashes, "independent_review_receipt": digest(receipt.model_dump())})
    result = Dataset.model_validate(payload)
    # Review can enrich evidence judgments, never rewrite questions, answers or facts.
    if (result.documents != dataset.documents or result.assertions != dataset.assertions
            or result.entities != dataset.entities or result.queries != dataset.queries
            or {g.query_id: g.answer for g in result.gold} != {g.query_id: g.answer for g in dataset.gold}):
        raise ValueError("review changed an immutable experimental variable")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["queue", "apply"])
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--receipt", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    dataset = Dataset.model_validate_json(args.dataset.read_text())
    if args.command == "apply":
        if args.receipt is None:
            raise ValueError("--receipt required")
        value = apply_review(dataset, ReviewReceipt.model_validate_json(args.receipt.read_text())).model_dump()
    else:
        value = pending_queue(dataset).model_dump()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        import os
        os.chmod(args.output, 0o600)
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    if args.command == "queue":
        refs = {d.document_id: d for d in dataset.documents}
        queries = {q.query_id: q for q in dataset.queries}
        cards = []
        for case in value["cases"]:
            cards.append("<h2>" + html.escape(queries[case["query_id"]].text) + "</h2>")
            for row in case["negative_documents"]:
                d = refs[row["document_id"]]
                cards.append("<article><h3>" + html.escape(d.document_id) + "</h3><p>Candidate "
                    + html.escape(row["kind"]) + "; NONANSWER UNREVIEWED</p><pre>" + html.escape(d.text) + "</pre></article>")
        args.output.with_suffix(".html").write_text("<!doctype html><meta charset='utf-8'><title>DS01 independent review queue</title>"
            "<style>body{max-width:1000px;margin:2em auto;font:16px sans-serif}pre{white-space:pre-wrap}"
            "article{border-top:1px solid #aaa;margin:2em 0}</style><h1>Pending source-based review</h1>"
            "<p>Judge against the question and operation. Nonprovided does not mean irrelevant. "
            "Review source support, context, permitted compositions, alternative evidence and disagreement. "
            "No independent review has been attested by this generated queue.</p>" + "".join(cards))
    print(json.dumps({"output": str(args.output), "independent_review_fabricated": False}))


if __name__ == "__main__":
    main()
