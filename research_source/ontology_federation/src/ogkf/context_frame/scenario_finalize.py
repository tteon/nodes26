"""Rebuild derived query Parquet/summary from immutable journals; no inference."""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from .challenge import ChallengeLedger, save
from .semantic_review import read_lines

TABLES = (
    "graph_operation_traces",
    "graph_query_traces",
    "retrieval_traces",
    "evaluation_results",
    "evidence_attribution",
    "routing_traces",
)


def finalize(root):
    rows = {
        name: read_lines(root / (name + ".jsonl"))
        for name in TABLES
        if (root / (name + ".jsonl")).exists()
    }
    run_ids = {r["experiment_run_id"] for values in rows.values() for r in values}
    if len(run_ids) != 1:
        raise ValueError("one original graph/agent run required")
    ledger = ChallengeLedger(root, run_ids.pop())
    ledger.rows = rows
    attempts, dates = [], set()
    for path in root.glob("*/traces/experiment=*/date=*/run=*/*.parquet"):
        attempts.extend(pq.ParquetFile(path).read().to_pylist())
        dates.update(
            p.name.removeprefix("date=")
            for p in path.parents
            if p.name.startswith("date=")
        )
    if len(dates) != 1:
        raise ValueError("one original run-date partition required")
    ledger.date = dates.pop()
    counts = ledger.verify()
    episodes = rows["evaluation_results"]
    receipts = [
        json.loads(p.read_text())
        for p in root.glob("graph-*.json")
        if p.name != "graph-capabilities.json"
    ]
    contract = json.loads((root / "execution-contract.json").read_text())
    snapshot = json.loads((root / "input-snapshot.json").read_text())
    expected = len(snapshot["public_questions"]) * (
        sum(5 if arm == "C" else 2 for _, arm in contract["executable_variants"])
        + int([contract["answer_model"], "C"] in contract["executable_variants"])
    )
    result = {
        "variants_materialized": len(receipts),
        "retrieval_episodes": len(episodes),
        "registered_query_traversal_complete": len(episodes) == expected,
        "answers_received": sum("answer" in r for r in episodes),
        "answer_errors": sum("answer_error" in r for r in episodes),
        "new_LLM_HTTP_attempts": len(attempts),
        "same_logical_assertions_single_multi": bool(receipts)
        and all(
            r["exact_single_multi_readback"] and r["exact_source_readback"]
            for r in receipts
        ),
        "parquet": counts,
        "semantic_quality": None,
        "independent_gold": False,
        "federation": "application_level_heterogeneous_domain_graphs",
        "native_composite_used": False,
        "summary_source": "original JSON journals and HTTP Parquet receipts; no inference or database re-execution",
    }
    target = root / "summary.json"
    if target.exists():
        previous = json.loads(target.read_text())
        if any(previous.get(k) != v for k, v in result.items() if k in previous):
            raise ValueError(
                "existing derived summary disagrees with original journals"
            )
    else:
        save(target, result)
    recovery = root / "offline-export-recovery.json"
    if not recovery.exists():
        save(
            recovery,
            {
                "reason": "in-memory tuple/list mismatch in registry metadata; canonical JSON readback restores the original recorded values",
                "original_journals_modified": False,
                "new_inference_calls": 0,
                "new_database_calls": 0,
                "parquet_readback": counts,
            },
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.run), indent=2))


if __name__ == "__main__":
    main()
