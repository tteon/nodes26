"""Audit received review content against the exact original wire requests."""

import argparse
import hashlib
import json
from pathlib import Path

from .semantic_review import pointer_messages, read_lines, validate_pointer_review


def audit_review(root, *, require_complete=True):
    manifest = json.loads((root / "manifest.json").read_text())
    for name, key in (
        ("questions.evaluator-only.jsonl", "question_data_sha256"),
        ("documents.jsonl", "document_data_sha256"),
    ):
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != manifest[key]:
            raise ValueError("review source/question input hash mismatch")
    questions = {
        q["query_id"]: q for q in read_lines(root / "questions.evaluator-only.jsonl")
    }
    docs = {d["document_id"]: d for d in read_lines(root / "documents.jsonl")}
    plans = {b["batch_id"]: b for b in manifest["batches"]}
    for p in root.glob("segment-*/recovery-plan.json"):
        plans.update({b["batch_id"]: b for b in json.loads(p.read_text())["batches"]})
    output, seen, received_requests = [], set(), set()
    failed = []
    for p in sorted((root / "batches").glob("*.json")):
        batch = json.loads(p.read_text())
        if not batch.get("records"):
            failed.append(p.name)
            continue
        batch_id = p.stem
        rows = [questions[q] for q in plans[batch_id]["query_ids"]]
        request_ids = {r["request_id"] for r in batch["records"]}
        if len(request_ids) != 1:
            raise ValueError("multiple response requests within a review batch")
        request_id = request_ids.pop()
        raw = list(root.glob("segment-*/*/raw/" + request_id + "-*.json"))
        accepted = list(root.glob("segment-*/*/accepted/" + request_id + ".json"))
        if len(raw) != 1 or len(accepted) != 1:
            raise ValueError("a unique actual review response is required")
        wire, value = (
            json.loads(raw[0].read_text()),
            json.loads(accepted[0].read_text()),
        )
        messages, source_ids = pointer_messages(rows, docs)
        if (
            wire["http_status"] != 200
            or wire["request"]["messages"] != messages
            or wire["request"]["model"] != batch["reviewer_model"]
            or json.loads(wire["response"]["choices"][0]["message"]["content"]) != value
        ):
            raise ValueError(
                "review source/question/model/wire response continuity mismatch"
            )
        expected = {
            r["query_id"]: r
            for r in validate_pointer_review(
                value, rows, docs, source_ids, request_id, batch_id, allow_partial=True
            )
        }
        if {r["query_id"] for r in batch["records"]} != set(expected):
            raise ValueError("received unique review items were lost or invented")
        for record in batch["records"]:
            qid = record["query_id"]
            if qid in seen:
                raise ValueError("duplicate authoritative question review")
            for key in (
                "query_id",
                "origin",
                "parent_ids",
                "review",
                "evidence_pointers",
                "pointer_errors",
            ):
                if record.get(key) != expected[qid].get(key):
                    raise ValueError(
                        "normalized review changed actual model output: " + key
                    )
            seen.add(qid)
            q = questions[qid]
            output.append(
                {
                    **expected[qid],
                    "question": q["text"],
                    "native_answer": q.get("native_answer"),
                    "source_ids": q["source_ids"],
                    "actual_request_path": str(raw[0]),
                    "input_coverage_verified": True,
                }
            )
        received_requests.add(request_id)
    missing = sorted(questions.keys() - seen)
    if require_complete and missing:
        raise ValueError(
            f"full review is incomplete: {len(missing)} questions lack received judgments"
        )
    audit = {
        "expected_questions": len(questions),
        "actual_received_question_reviews": len(seen),
        "all_question_content_reviews_received": not missing,
        "pending_query_ids": missing,
        "verified_complete_source_requests": len(received_requests),
        "source_clipping": False,
        "input_and_output_continuity": True,
        "independent_human_gold": 0,
        "model_review_semantic_calibration": "not established",
        "failed_or_empty_batch_events": failed,
        "valid_source_locator_records": sum(not r["pointer_errors"] for r in output),
        "invalid_source_locator_records": sum(
            bool(r["pointer_errors"]) for r in output
        ),
        "published_answer_availability_label_mismatches": sum(
            r["answer_availability_label_mismatch"] for r in output
        ),
        "source_locator_semantic_support": "not judged by this protocol check",
    }
    return output, audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    _, audit = audit_review(args.review, require_complete=not args.allow_partial)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
