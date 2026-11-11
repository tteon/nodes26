"""Continue only E3 jobs with proof that no HTTP request was dispatched.

An ambiguous timeout is permanently abandoned, never replayed. Original outcomes
remain immutable and every continuation points to its original sample.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

import pyarrow.parquet as pq
from ..models import digest
from ..observability import Recorder
from .contracts import Dataset, PilotConfig
from .runner import code_hashes, preflight_gateway, record_generation, sync_llm, write
from .storage import MemoryExecutor
from .trace import Ledger


def table(root, name):
    return [json.loads(row["payload_json"]) for p in (root / "parquet").glob("**/" + name + "/*.parquet")
            for row in pq.ParquetFile(p).read().to_pylist()]


def abandon_owned(parent, lock_path):
    manifest = json.loads((parent / "manifest.json").read_text())
    summary = json.loads((parent / "summary.json").read_text())
    if summary["experiment_run_id"] != manifest["experiment_run_id"]:
        raise ValueError("parent client terminal receipt missing/mismatched")
    pending = lock_path.with_suffix(".pending")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if not pending.exists():
                return {"status": "no_pending_guard"}
            state = json.loads(pending.read_text())
            if state["run_id"] != manifest["experiment_run_id"]:
                raise ValueError("pending guard belongs to another experiment; untouched")
            raw = json.loads((parent / "raw" / (state["request_id"] + "-" + str(state["attempt"]) + ".json")).read_text())
            if raw["http_status"] is not None or raw["error"] != "ReadTimeout":
                raise ValueError("only an owned terminal client ReadTimeout can be abandoned here")
            if digest({"path": "/chat/completions", "body": raw["request"]}) != state["request_hash"]:
                raise ValueError("pending/raw request hash mismatch")
            receipt = {"status": "permanently_abandoned_without_resubmission",
                "parent_run_id": state["run_id"], "request_id": state["request_id"],
                "request_hash": state["request_hash"], "provider_outcome": "unknown",
                "usage": None, "client_process_completed": True,
                "policy": "Only never-dispatched independent jobs may continue; timed-out request is excluded."}
            write(parent / "abandoned.pending.json", state)
            write(parent / "timeout-resolution.json", receipt)
            pending.unlink()
            return receipt
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def continuation_jobs(parent):
    attempts = table(parent, "llm_request_traces")
    sent = {row["trace_id"] for row in attempts}
    contexts = {row["sample_id"]: row for row in table(parent, "reasoning_contexts")}
    jobs = []
    for episode in table(parent, "agent_evaluation"):
        if "Unresolved MARA outcome" not in (episode.get("error") or ""):
            continue
        if episode["trace_id"] in sent or episode["sample_id"] not in contexts:
            raise ValueError("continuation eligibility has an HTTP attempt or missing context")
        jobs.append((episode, contexts[episode["sample_id"]]))
    logical_sent = len({row["request_id"] for row in attempts})
    order = {"oracle_evidence": 0, "lexical_context_swap": 1, "relational_context_swap": 2, "empty_evidence_control": 3}
    return sorted(jobs, key=lambda pair: (pair[0]["query_id"], order[pair[1]["kind"]])), logical_sent + len(jobs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--capabilities", type=Path, required=True)
    args = p.parse_args()
    parent = args.parent
    original = json.loads((parent / "manifest.json").read_text())
    config = PilotConfig.model_validate(original["config"])
    jobs, consumed = continuation_jobs(parent)
    remaining = config.max_logical_calls - consumed
    bound = 1 + 3 * len(jobs)
    if not jobs or bound > remaining:
        raise ValueError("no eligible jobs or insufficient original campaign budget")
    resolution = abandon_owned(parent, Path(os.environ["MARA_ADMISSION_LOCK"]))
    recorder = Recorder(args.output, "live", experiment="ds01")
    ledger = Ledger(recorder.root, recorder.run_id)
    write(recorder.root / "manifest.json", {"experiment_run_id": recorder.run_id,
        "parent_run_id": original["experiment_run_id"], "parent_path": str(parent),
        "code_hashes": code_hashes(), "config": config.model_dump(),
        "scope": "Only never-dispatched fixed-evidence E3 jobs; not new independent repetitions",
        "parent_logical_attempts_including_admission_rejections": consumed,
        "parent_original_logical_cap": config.max_logical_calls, "continuation_logical_cap": bound,
        "timeout_resolution": resolution,
        "jobs": [{"parent_sample_id": e["sample_id"], "parent_trace_id": e["trace_id"]} for e, _ in jobs]})
    dataset = Dataset.model_validate_json((parent / "dataset.private.json").read_text())
    docs = MemoryExecutor(dataset, ledger).docs
    gold = {g.query_id: g for g in dataset.gold}
    queries = {q.query_id: q for q in dataset.queries}
    gateway = None
    try:
        gateway, capability = preflight_gateway(recorder, config.model_copy(update={"max_logical_calls": bound}), args.capabilities)
        write(recorder.root / "mara-capability-admission.json", capability)
        for episode, context in jobs:
            query = queries[episode["query_id"]]
            by_evidence = {d["evidence_id"]: d for d in docs.values()}
            evidence = [by_evidence[e] for e in context["evidence_ids"]]
            expected = gold[query.query_id]
            if context["kind"] == "empty_evidence_control":
                expected = expected.model_copy(update={"answer": "Insufficient evidence in the delivered empty context."})
            common = {k: episode[k] for k in ("condition_id", "seed", "graph_snapshot_id", "query_id", "strategy")}
            common.update(sample_id=digest([recorder.run_id, episode["sample_id"]]), trace_id=str(uuid.uuid4()),
                parent_sample_id=episode["sample_id"], parent_trace_id=episode["trace_id"])
            ledger.emit("reasoning_contexts", **common, kind=context["kind"], evidence_ids=context["evidence_ids"],
                treatment_applied=context["treatment_applied"], execution_source="frozen evidence; no new graph query",
                native_gold_unchanged=True)
            result = record_generation(query, expected, evidence, common, config, gateway, ledger, context["kind"], time.perf_counter())
            print(json.dumps({"phase": "E3_never_dispatched_continuation", "kind": context["kind"],
                "logical_calls": gateway.logical_calls, "error": result["error"]}), flush=True)
    finally:
        if gateway:
            gateway.close()
        sync_llm(recorder, ledger)
        recorder.readback()
        results = ledger.rows.get("agent_evaluation", [])
        write(recorder.root / "summary.json", {"experiment_run_id": recorder.run_id,
            "parent_run_id": original["experiment_run_id"], "continued_jobs": len(results),
            "failed_jobs": sum(r["error"] is not None for r in results),
            "logical_calls": gateway.logical_calls if gateway else None, "http_attempts": len(recorder.rows),
            "known_tokens": sum((r["input_tokens"] or 0) + (r["output_tokens"] or 0) for r in recorder.rows),
            "unknown_usage_attempts": sum(r["input_tokens"] is None or r["output_tokens"] is None for r in recorder.rows),
            "parquet_readback": ledger.verify(), "ambiguous_request_replayed": False})
    print(json.dumps({"recovery_path": str(recorder.root)}))


if __name__ == "__main__":
    main()
