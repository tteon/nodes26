"""Reproducible DS-01 CLI: schemas, source preparation, gated E2/E3 pilot."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import uuid

from ..gateway import Gateway, GatewayError
from ..local_runtime import local_dozer
from ..models import digest
from ..observability import Recorder
from .contracts import (Assertion, CorpusManifest, Dataset, Document, Entity, GoldEvidence,
    PilotConfig, Query, RoutePlan, ShiftManifest)
from .corpus import generate
from .evaluation import judge, paired_report, retrieval_metrics, strategy_shift_interaction
from .finder import import_finder
from .retrieval import FixedIndex, Planner, answer, pack, registry, retrieve
from .storage import BoltExecutor, MemoryExecutor
from .trace import Ledger

BASE = Path(__file__).resolve().parents[3]


class BoundedGateway(Gateway):
    def request(self, body, metadata, *, path="/chat/completions"):
        if body is not None and len(json.dumps(body, ensure_ascii=False).encode()) > 64000:
            self.recorder.write("admission-rejections/" + str(uuid.uuid4()) + ".json",
                {"reason": "input_body_byte_cap", "trace_id": metadata.get("trace_id")})
            raise GatewayError("DS01 input body exceeds declared64KB bound; no HTTP request sent")
        return super().request(body, metadata, path=path)


def sync_llm(recorder, ledger):
    known = {(r["request_id"], r["attempt"]) for r in ledger.rows.get("llm_request_traces", [])}
    for row in recorder.rows:
        if (row["request_id"], row["attempt"]) not in known:
            ledger.emit("llm_request_traces", **row)


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def code_hashes():
    return {str(p.relative_to(BASE)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((BASE / "src/ogkf").rglob("*.py"))}


def export_schemas(directory):
    directory.mkdir(parents=True, exist_ok=False)
    for model in (Query, GoldEvidence, Entity, Assertion, Document, Dataset, CorpusManifest,
                  ShiftManifest, PilotConfig, RoutePlan):
        write(directory / (model.__name__ + ".schema.json"), model.model_json_schema())


def preflight_gateway(recorder, config, capability_path):
    observed = json.loads(capability_path.read_text())
    timestamp = datetime.fromisoformat(observed["observed_at"].replace("Z", "+00:00"))
    if (datetime.now(timezone.utc) - timestamp).total_seconds() > 86400:
        raise ValueError("MARA capability manifest expired; rerun explicit bounded probe")
    if observed.get("base_url", "").rstrip("/") != os.getenv("MARA_BASE_URL", "https://api.cloud.mara.com/v1").rstrip("/"):
        raise ValueError("capability endpoint differs")
    for model in {config.router_model, config.answer_model, config.judge_model}:
        if observed["models"].get(model, {}).get("json_schema") is not True:
            raise ValueError("model structured output not observed: " + model)
    gateway = BoundedGateway(recorder, SimpleNamespace(timeout_seconds=60, min_interval_seconds=1.,
                      max_logical_calls=config.max_logical_calls, max_attempts=config.max_attempts),
                      capabilities=observed["models"])
    catalog = gateway.request(None, {"parent_step_id": "capability_catalog"}, path="/models")
    ids = {row["id"] for row in catalog["response"].get("data", [])}
    if not {config.router_model, config.answer_model, config.judge_model} <= ids:
        gateway.close()
        raise ValueError("declared model absent from live same-account catalog")
    return gateway, {"capability_manifest_sha256": hashlib.sha256(capability_path.read_bytes()).hexdigest(),
                     "catalog_request_id": catalog["request_id"], "available_model_ids": sorted(ids)}


def record_generation(query, gold, evidence, common, config, gateway, ledger, kind, started):
    generated, scored, error = None, None, None
    inference_ms = None
    try:
        generated = answer(query, evidence, gateway, config.answer_model,
            {"trace_id": common["trace_id"], "experiment_condition_id": common["condition_id"],
             "ontology_version": "ds01-source-ontology-v1"})
        inference_ms = (time.perf_counter() - started) * 1000
        scored = judge(query, gold, generated, evidence, gateway, config.judge_model,
            {"trace_id": common["trace_id"], "experiment_condition_id": common["condition_id"]})
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
        inference_ms = inference_ms if inference_ms is not None else (time.perf_counter() - started) * 1000
    known = {d["evidence_id"] for d in evidence}
    for index, claim in enumerate(generated.claims if generated else []):
        ledger.emit("evidence_attribution", **common,
            claim_id=digest([common["sample_id"], index, claim.text, claim.evidence_ids]),
            model_claim_id=claim.claim_id, claim=claim.model_dump(),
            citation_ids_delivered=bool(claim.evidence_ids) and set(claim.evidence_ids + claim.operand_evidence_ids) <= known,
            semantic_attribution_status="judge_assisted_uncalibrated" if scored else "unknown")
    result = ledger.emit("agent_evaluation", **common, kind=kind,
        generated_answer=generated.model_dump() if generated else None,
        judge_result=scored.model_dump() if scored else None,
        judge_calibrated=False, grounded_qa_success=None,
        operational_completed=generated is not None and error is None and common.get("retrieval_error") is None,
        inference_end_to_end_ms=inference_ms, error=error,
        expected_label_scope="native_published_QA" if kind != "empty_evidence_control" else "derived_context_policy_control",
        context_evidence_ids=sorted(known), gold_rewritten=False)
    return result


def run_live(dataset, shifts, config, executor, index, recorder, ledger, capability_path):
    gateway, capability = preflight_gateway(recorder, config, capability_path)
    write(recorder.root / "mara-capability-admission.json", capability)
    planner = Planner(dataset, gateway, config.router_model)
    gold = {g.query_id: g for g in dataset.gold}
    first_seed = config.seeds[0]
    primary = [s for s in shifts if s.seed == first_seed and s.intervention == config.interventions[0]
               and s.severity in {0, max(config.severities)}]
    public_query = dataset.queries[0]
    plan_cache = {}
    completed = []
    try:
        for shift in primary:
            for strategy in config.strategies:
                if len(completed) >= config.live_end_to_end_limit:
                    break
                started, trace = time.perf_counter(), str(uuid.uuid4())
                plan = None
                origin = None
                key = shift.condition_id, public_query.query_id, strategy == "B3_reduced"
                # Freeze the same decision for B1/B3/B4 placement/planning contrasts.
                if strategy in {"B1", "B3", "B4", "B3_reduced"} and key in plan_cache:
                    plan, origin = plan_cache[key]
                row = retrieve(public_query, shift, dataset, config, strategy, executor, planner, index, ledger,
                    plan=plan, oracle=gold[public_query.query_id] if strategy == "B5" else None,
                    phase="E3_end_to_end", trace_id=trace)
                if row["plan"] and key not in plan_cache:
                    plan_cache[key] = (RoutePlan.model_validate(row["plan"]), trace)
                common = {k: row[k] for k in ("condition_id", "sample_id", "seed", "graph_snapshot_id",
                                               "query_id", "trace_id", "strategy")}
                common["retrieval_error"] = row["error"]
                ledger.emit("routing_decisions", **common, decision=row["plan"],
                    origin_trace_id=origin or trace, plan_reused=origin is not None,
                    latency_scope="actual current calls; reused decisions exclude original generation latency")
                result = record_generation(public_query, gold[public_query.query_id], row["evidence"], common,
                    config, gateway, ledger, "end_to_end", started)
                completed.append(result)
                print(json.dumps({"phase": "E3_end_to_end", "episodes": len(completed),
                    "logical_calls": gateway.logical_calls, "error": result["error"]}), flush=True)
        reference = primary[0]
        docs = executor.docs
        for query in dataset.queries:
            expected = gold[query.query_id]
            required = [d for d in docs.values() if d["evidence_id"] in expected.required_ids()]
            candidates = [docs[r] for r in reference.corpus.document_ids if docs[r]["evidence_id"] not in expected.required_ids()]
            candidates.sort(key=lambda d: (-len(set(query.text.lower().split()) & set(d["text"].lower().split())), d["document_id"]))
            lexical = candidates[:1]
            relational = [d for d in candidates if d not in lexical][:1]
            for kind, extras in (("oracle_evidence", []), ("lexical_context_swap", lexical),
                                 ("relational_context_swap", relational), ("empty_evidence_control", [])):
                trace = str(uuid.uuid4())
                common = dict(condition_id=reference.condition_id, seed=reference.seed,
                    graph_snapshot_id=reference.corpus.graph_snapshot_id, query_id=query.query_id,
                    trace_id=trace, strategy="E3_fixed", sample_id=digest([recorder.run_id, query.query_id, kind]))
                packet = required + extras
                scores = {d["document_id"]: len(packet) - i for i, d in enumerate(packet)}
                evidence, rejected, _ = pack(packet, scores, config.global_top_k, config.context_byte_budget)
                label = expected
                if kind == "empty_evidence_control":
                    evidence = []
                    # A policy-control label, never overwrite original native gold.
                    label = expected.model_copy(update={"answer": "Insufficient evidence in the delivered empty context."})
                treatment_applied = (kind not in {"lexical_context_swap", "relational_context_swap"}
                                     or any(d["evidence_id"] in {e["evidence_id"] for e in evidence} for d in extras))
                ledger.emit("reasoning_contexts", **common, kind=kind, evidence_ids=[d["evidence_id"] for d in evidence],
                    treatment_applied=treatment_applied, packing_rejections=rejected,
                    distractor_status="candidate_not_independently_reviewed_nonanswer",
                    expected_relation_to_base="insufficient" if kind == "empty_evidence_control" else "unchanged_provisionally",
                    oracle_diagnostic=True, native_gold_unchanged=True)
                result = record_generation(query, label, evidence, common, config, gateway, ledger, kind, time.perf_counter())
                completed.append(result)
                print(json.dumps({"phase": "E3_fixed", "kind": kind, "query": query.query_id,
                    "logical_calls": gateway.logical_calls, "error": result["error"]}), flush=True)
    finally:
        gateway.close()
        sync_llm(recorder, ledger)
        recorder.readback()
    return completed


def figures(root, shifts, evaluations):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return {"status": "matplotlib_unavailable"}
    fig, ax = plt.subplots(figsize=(9, 5))
    from collections import defaultdict
    groups = defaultdict(list)
    lookup = {s.condition_id: s for s in shifts}
    for row in evaluations:
        s = lookup[row["condition_id"]]
        groups[s.intervention, row["strategy"], s.severity].append(float(row["provided_reference_sufficiency_proxy"]))
    for intervention, strategy in sorted({(i, a) for i, a, _ in groups}):
        x = sorted(s for i, a, s in groups if i == intervention and a == strategy)
        ax.plot(x, [sum(groups[intervention, strategy, s]) / len(groups[intervention, strategy, s]) for s in x],
                marker="o", label=intervention + " " + strategy)
    ax.set(xlabel="Assigned sampling severity", ylabel="Provided-reference sufficiency proxy",
           title="DS-01 native FinDER pilot: exploratory retrieval", ylim=(-.05, 1.05))
    ax.legend(ncol=2, fontsize=8)
    fig.text(.02, .01, "Source-valid corpus; nonanswer/context labels pending. This is not semantic Evidence Sufficiency.", fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, 1))
    path = root / "retrieval_response.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"status": "created", "path": path.name}


def run(dataset, config, output, *, graph="memory", live=False, capability_path=None, test_gate=True):
    recorder = Recorder(output, "live" if live else "mock", experiment="ds01")
    ledger = Ledger(recorder.root, recorder.run_id)
    write(recorder.root / "manifest.json", {
        "experiment_run_id": recorder.run_id, "dataset_hash": dataset.dataset_hash, "config": config.model_dump(),
        "code_hashes": code_hashes(), "scope": "Native FinDER source-bound constructed-corpus pilot, retrospectively exposed",
        "graph_mode": graph, "live_llm_requested": live, "native_queries_unchanged": True,
        "gold_answers_unchanged": True, "human_calibration_receipt": None,
        "maximum_http_attempts": config.max_logical_calls * config.max_attempts,
        "maximum_input_body_bytes_per_request": 64000,
        "body_bound_status": "Conservative full input bound checked by DS01 gateway wrapper before each HTTP call",
        "maximum_output_tokens_per_request": 4096,
        "observability_retention": "Public benchmark/filing inputs only; private modes0600;30-day manual retention",
        "pipeline_status_at_manifest": "admission"})
    write(recorder.root / "dataset.private.json", dataset.model_dump())
    write(recorder.root / "public/queries.json", [q.model_dump() for q in dataset.queries])
    runtime = None
    try:
        if test_gate:
            check = subprocess.run([sys.executable, "-m", "pytest", "-q", str(BASE / "tests/test_ds01.py")],
                                   cwd=BASE, capture_output=True, text=True, timeout=120)
            write(recorder.root / "generator-test-gate.json", {"exit_code": check.returncode,
                "stdout": check.stdout, "stderr": check.stderr, "code_hashes": code_hashes()})
            if check.returncode:
                raise ValueError("generator/gold/metric tests failed; no matrix admitted")
        shifts = generate(dataset, config)
        write(recorder.root / "shifts.json", [s.model_dump() for s in shifts])
        for s in shifts:
            common = dict(condition_id=s.condition_id, seed=s.seed, graph_snapshot_id=s.corpus.graph_snapshot_id)
            ledger.emit("corpus_manifests", **common, corpus=s.corpus.model_dump(), target=s.target)
            ledger.emit("distribution_measurements", **common, intervention=s.intervention, severity=s.severity,
                reference_corpus_id=s.reference_corpus_id, observed=s.observed,
                target=s.target, covariate_changes=s.covariate_changes, causal_isolation=s.causal_isolation,
                structural_valid=s.structural_valid, research_qualified=s.research_qualified, limitations=s.limitations)
        environment_context = local_dozer() if graph == "local-dozer" else nullcontext((dict(os.environ), {}))
        evaluations, live_results = [], []
        with environment_context as (environment, runtime):
            executor = BoltExecutor(dataset, ledger, environment) if graph != "memory" else MemoryExecutor(dataset, ledger)
            try:
                capabilities = executor.build(shifts) if graph != "memory" else {
                    "backend": "memory", "external_database_tested": False, "native_federation": False}
                write(recorder.root / "graph-capabilities.json", capabilities)
                if graph != "memory":
                    write(recorder.root / "query-profile-diagnostics.json", executor.profile(shifts[0].corpus))
                index, planner = FixedIndex(dataset), Planner(dataset)
                expected = {g.query_id: g for g in dataset.gold}
                for number, shift in enumerate(shifts):
                    for query in dataset.queries:
                        frozen_plan = planner.plan(query, "offline", shift.condition_id)
                        for strategy in config.strategies:
                            row = retrieve(query, shift, dataset, config, strategy, executor, planner, index, ledger,
                                plan=frozen_plan if strategy in {"B1", "B3", "B4"} else None,
                                oracle=expected[query.query_id] if strategy == "B5" else None)
                            score = retrieval_metrics(row, expected[query.query_id], dataset)
                            evaluations.append(ledger.emit("evaluation_results",
                                **{k: row[k] for k in ("condition_id", "sample_id", "seed", "graph_snapshot_id",
                                                      "query_id", "trace_id", "strategy")}, **score))
                    print(json.dumps({"phase": "E2_retrieval", "conditions_completed": number + 1,
                        "conditions_total": len(shifts), "episodes": len(evaluations)}), flush=True)
                if live:
                    if capability_path is None:
                        raise ValueError("live MARA requires same-account capability manifest")
                    live_results = run_live(dataset, shifts, config, executor, index, recorder, ledger, capability_path)
            finally:
                if graph != "memory":
                    write(recorder.root / "owned-database-teardown.json", executor.close())
            if graph == "local-dozer":
                # Context manager updates teardown result on exit.
                pass
        if graph == "local-dozer":
            write(recorder.root / "local-runtime.json", runtime)
        contrasts = paired_report(evaluations, shifts, dataset)
        analysis = {"paired_changes": contrasts,
                    "strategy_shift_model": strategy_shift_interaction(evaluations, shifts, dataset, config.target_entity_id)}
        write(recorder.root / "analysis.json", analysis)
        figures_result = figures(recorder.root, shifts, evaluations)
        if recorder.rows:
            known_tokens = sum((r["input_tokens"] or 0) + (r["output_tokens"] or 0) for r in recorder.rows)
            usage_unknown = sum(r["input_tokens"] is None or r["output_tokens"] is None for r in recorder.rows)
        else:
            known_tokens, usage_unknown = None, 0
        summary = {"experiment_run_id": recorder.run_id, "run_path": str(recorder.root),
            "queries": len(dataset.queries), "source_pool_documents": len(dataset.documents),
            "conditions": len(shifts), "unique_snapshots": len({s.corpus.corpus_id for s in shifts}),
            "retrieval_episodes": len(evaluations), "retrieval_execution_failures": sum(not r["execution_success"] for r in evaluations),
            "E3_agent_episodes": len(live_results), "E3_failed_episodes": sum(r["error"] is not None for r in live_results),
            "structural_corpus_checks_passed": all(s.structural_valid for s in shifts),
            "imperfect_interventions": sum(s.causal_isolation == "imperfect" for s in shifts),
            "semantic_research_qualified_conditions": sum(s.research_qualified for s in shifts),
            "http_attempts": len(recorder.rows), "http_statuses": {
                str(status): sum(r["http_status"] == status for r in recorder.rows) for status in {r["http_status"] for r in recorder.rows}},
            "known_tokens": known_tokens, "usage_unknown_attempts": usage_unknown,
            "usage_complete": bool(recorder.rows) and usage_unknown == 0,
            "figure": figures_result, "parquet_readback": ledger.verify(),
            "limitations": ["Provided-reference sufficiency is a proxy; primary semantic metric null without labels.",
                "Distractor nonanswer/context semantics and human evaluator calibration remain unqualified.",
                "Source-metadata relation shifts do not establish business-fact neighborhood effects.",
                "Small retrospectively exposed cohort; no unseen-issuer or noninferiority claim.",
                "Shared route plans support placement controls, not independent deployment latency comparisons."]}
        write(recorder.root / "summary.json", summary)
        return summary
    except Exception as exc:
        sync_llm(recorder, ledger)
        if runtime is not None and graph == "local-dozer" and not (recorder.root / "local-runtime.json").exists():
            write(recorder.root / "local-runtime.json", runtime)
        write(recorder.root / "run-error.json", {"type": type(exc).__name__, "message": str(exc),
              "partial_parquet_counts": ledger.verify(), "http_attempts": len(recorder.rows)})
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    schemas = sub.add_parser("schemas")
    schemas.add_argument("--output", type=Path, required=True)
    prepare = sub.add_parser("prepare")
    for field in ("public", "gold", "references", "metadata", "filings", "output"):
        prepare.add_argument("--" + field, type=Path, required=True)
    prepare.add_argument("--queries", nargs="+", default=["b33fcee7", "ea48df6d"])
    pilot = sub.add_parser("run")
    pilot.add_argument("--dataset", type=Path, required=True)
    pilot.add_argument("--config", type=Path, default=BASE / "configs/ds01.pilot.json")
    pilot.add_argument("--output", type=Path, default=BASE / "results/ds01")
    pilot.add_argument("--graph", choices=["memory", "local-dozer", "dozer"], default="memory")
    pilot.add_argument("--live", action="store_true")
    pilot.add_argument("--capabilities", type=Path)
    pilot.add_argument("--all-shifts", action="store_true")
    args = p.parse_args()
    if args.command == "schemas":
        export_schemas(args.output)
    elif args.command == "prepare":
        dataset = import_finder(args.public, args.gold, args.references, args.metadata, args.filings, args.queries)
        write(args.output, dataset.model_dump())
        print(json.dumps({"dataset_path": str(args.output), "hash": dataset.dataset_hash,
            "queries": len(dataset.queries), "source_valid_documents": len(dataset.documents),
            "assertions": len(dataset.assertions), "semantic_nonanswer_pool_validated": False}))
    else:
        dataset = Dataset.model_validate_json(args.dataset.read_text())
        config = PilotConfig.model_validate_json(args.config.read_text())
        if args.all_shifts:
            config = config.model_copy(update={"interventions": ["DS-01A", "DS-01B", "DS-01C"]})
        summary = run(dataset, config, args.output, graph=args.graph, live=args.live, capability_path=args.capabilities)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
