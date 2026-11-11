from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import uuid

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from .evaluation import calibration_check, evaluate
from .extraction import messages, normalize, repair_unique_quote, ONTOLOGIES, PROMPTS, PROMPT_VERSION
from .gateway import Gateway, GatewayError
from .models import Config, Extraction, Source, digest
from .observability import Recorder
from .probes import probe_graph, probe_mara

BASE = Path(__file__).resolve().parents[2]


def load_config(path):
    return Config.model_validate_json(path.read_text())


def mock_client():
    """Synthetic responder reads only public source; no gold is accessible here.

    Deliberately emits a fixed supported fixture fact. Identical mock outputs
    across arms cannot demonstrate model differences or ontology effects.
    """
    def respond(request):
        if request.method == "GET":
            return httpx.Response(200, json={"data": [{"id": "mock-a"}, {"id": "mock-b"}]})
        body = json.loads(request.content)
        if "capability_probe" in body["messages"][-1]["content"]:
            message = {"content": '{"ok":true}'}
            if body.get("tools"):
                message = {"content": None, "tool_calls": [{"id": "fixture-tool", "type": "function", "function": {
                    "name": "capability_probe", "arguments": '{"ok":true}'}}]}
        else:
            source = json.loads(body["messages"][-1]["content"])["source"]
            quote = source["text"].split("\n")[0]
            fact = {"subject": "ExampleCo", "subject_type": "ReportingEntity", "predicate": "reports_revenue",
                    "literal_value": "10", "domain_id": source["domain_id"], "temporal_scope": "2025",
                    "business_scope": "consolidated group", "unit": "USD million", "table_headers": [],
                    "evidence_start": 0, "evidence_end": len(quote), "evidence_span": quote}
            message = {"content": json.dumps({"facts": [fact]})}
        # No fabricated token accounting. Mock and missing usage are explicit.
        return httpx.Response(200, json={"choices": [{"message": message}]})
    return httpx.Client(transport=httpx.MockTransport(respond))


def runtime_manifest():
    return {"python": platform.python_version(), "dependencies": {
        p: importlib.metadata.version(p) for p in ("httpx", "pydantic", "pyarrow", "neo4j", "jsonschema")},
        "code_hashes": {p.name: digest(p.read_text()) for p in Path(__file__).parent.glob("*.py")},
        "prompt_hash": digest(PROMPTS), "ontology_hash": digest(ONTOLOGIES)}


def run_extraction(args):
    config = load_config(args.config)
    sources = [Source.model_validate(d) for d in json.loads((BASE / config.dataset).read_text())]
    conditions = config.conditions()
    count = len(sources) * len(conditions)
    if count > config.max_logical_calls:
        raise ValueError("condition matrix exceeds configured call budget")
    if args.mode == "mock" and any(not m.startswith("mock-") for m in config.models):
        raise ValueError("mock mode requires explicitly named mock models")
    if args.mode == "live":
        if not args.capabilities:
            raise ValueError("live run requires --capabilities from a live probe")
        cap = json.loads(args.capabilities.read_text())
        if cap.get("mode") != "live" or cap.get("base_url") != os.getenv("MARA_BASE_URL", "https://api.cloud.mara.com/v1").rstrip("/"):
            raise ValueError("capabilities do not match live provider endpoint")
        age = datetime.now(timezone.utc) - datetime.fromisoformat(cap["observed_at"])
        if age.total_seconds() < 0 or age.total_seconds() > 86400:
            raise ValueError("capability manifest expired; re-probe within 24 hours")
        for model in config.models:
            entry = cap.get("models", {}).get(model, {})
            if not any(entry.get(feature) is True for feature in ("basic", "json_schema", "json_object")):
                raise ValueError("model has no verified completion format: " + model)
        # Identical transport format across all cells, based on intersection.
        common = {k: all(cap["models"][m].get(k) is True for m in config.models)
                  for k in ("json_schema", "json_object", "temperature")}
        caps = {m: common for m in config.models}
        if len(config.ontologies) > 1 or count > 2 or len(sources) != 1:
            calibration_check(config.human_calibration_receipt, BASE / config.dataset)
            if not config.successful_smoke_summary:
                raise ValueError("successful live two-model smoke summary required before expansion")
            smoke_path = Path(config.successful_smoke_summary)
            smoke = json.loads(smoke_path.read_text())
            smoke_manifest = json.loads(smoke_path.with_name("manifest.json").read_text())
            if (smoke.get("mode") != "live" or smoke.get("samples") != 2 or
                    smoke.get("successful_samples") != 2 or smoke_manifest["config"]["models"] != config.models or
                    smoke_manifest.get("code_hashes") != runtime_manifest()["code_hashes"]):
                raise ValueError("prior smoke must qualify both models on current code")
    else:
        caps = {m: {"json_schema": True, "temperature": True} for m in config.models}
    recorder = Recorder(args.output, args.mode)
    recorder.write("manifest.json", {"scope": "pipeline smoke; no hypothesis claims", "config": config.model_dump(),
        "config_hash": digest(config.model_dump()), "sources_hash": digest([s.model_dump() for s in sources]),
        "effective_capabilities": caps, "capability_manifest_hash": digest(cap) if args.mode == "live" else None,
        "planned_logical_calls": count, "max_http_attempts": count * config.max_attempts,
        "mode": args.mode, "experiment_run_id": recorder.run_id, **runtime_manifest(),
        "retention_policy": "public/synthetic inputs only; private bodies 30 days, manual deletion; see README"})
    gateway = Gateway(recorder, config, client=mock_client() if args.mode == "mock" else None, capabilities=caps)
    outcomes, normalized, repaired_count = [], [], 0
    try:
        for source in sources:
            for condition in conditions:
                trace_id = str(uuid.uuid4())
                sample_id = digest([recorder.run_id, source.document_id, condition])[:24]
                meta = {"trace_id": trace_id, "parent_step_id": sample_id,
                        "experiment_condition_id": condition["condition_id"],
                        "prompt_version": PROMPT_VERSION+"/" + condition["prompt"],
                        "ontology_version": "ontology-v1/" + condition["ontology"],
                        "document_id": source.document_id, "source_publisher": source.source_publisher}
                parameters = {"max_tokens": config.max_output_tokens}
                if caps[condition["model"]].get("temperature") is True:
                    parameters["temperature"] = config.temperature
                prompt = messages(source, condition["prompt"], condition["ontology"])
                outcome = {**condition, "sample_id": sample_id, "trace_id": trace_id,
                           "document_id": source.document_id, "prompt_characters": sum(len(m["content"]) for m in prompt),
                           "decoding_parameters": parameters, "unsupported_parameters": [] if "temperature" in parameters else ["temperature"],
                           "status": "failed", "facts_accepted": 0, "facts_rejected": 0}
                try:
                    result = gateway.generate_structured(prompt, Extraction.model_json_schema(), condition["model"], parameters, meta)
                    extracted = Extraction.model_validate(result["value"])
                    facts, rejected = [], []
                    for fact_index, fact in enumerate(extracted.facts):
                        try:
                            repaired = False
                            try:
                                assertion = normalize(fact, source, recorder.run_id, condition, result["request_id"])
                            except ValueError:
                                if config.span_repair_policy != "unique_exact_quote_v1":
                                    raise
                                fixed = repair_unique_quote(fact, source)
                                assertion = normalize(fixed, source, recorder.run_id, condition, result["request_id"])
                                recorder.write(f"repaired/{sample_id}-{fact_index}.json", {
                                    "policy": config.span_repair_policy, "original": fact.model_dump(), "repaired": fixed.model_dump()})
                                repaired, repaired_count = True, repaired_count + 1
                            facts.append({**assertion, "span_repaired": repaired,
                                          "trace_id": trace_id, "llm_provider": "mara" if args.mode == "live" else "mock",
                                          "graph_provider": None})
                        except ValueError as exc:
                            rejected.append({"fact": fact.model_dump(), "reason": str(exc)})
                    recorder.write("normalized/" + sample_id + ".json", facts)
                    if rejected:
                        recorder.write("rejected/spans-" + sample_id + ".json", rejected)
                    normalized.extend(facts)
                    outcome.update(status="accepted" if not rejected else "partially_rejected", facts_accepted=len(facts),
                                   facts_rejected=len(rejected), request_id=result["request_id"])
                except GatewayError as exc:
                    outcome["error"] = str(exc)
                outcomes.append(outcome)
                recorder.write("samples/" + sample_id + ".json", outcome)
        recorder.write("normalized/facts.json", normalized)
        if normalized:
            pq.write_table(pa.Table.from_pylist(normalized), recorder.root / "normalized/facts.parquet")
        rows = recorder.readback()
        summary = {"mode": args.mode, "experiment_run_id": recorder.run_id, "samples": len(outcomes),
                   "successful_samples": sum(o["status"] == "accepted" for o in outcomes),
                   "logical_calls": gateway.logical_calls, "http_attempts": len(rows),
                   "normalized_facts": len(normalized), "usage_missing_attempts": sum(r["input_tokens"] is None for r in rows),
                   "human_calibration": "not_performed_by_agent", "quality_metrics": None,
                   "hypotheses_evaluated": False, "repaired_outputs": repaired_count}
        recorder.write("summary.json", summary)
        print(json.dumps({"run_directory": str(recorder.root), **summary}, indent=2))
        return 0 if summary["successful_samples"] == count else 1
    finally:
        gateway.close()


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", type=Path, default=BASE / "configs/smoke.mock.json")
    run.add_argument("--mode", choices=["mock", "live"], default="mock")
    run.add_argument("--capabilities", type=Path)
    run.add_argument("--output", type=Path, default=BASE / "results")
    mara = sub.add_parser("probe-mara")
    mara.add_argument("--config", type=Path, default=BASE / "configs/probe.live.json")
    mara.add_argument("--mode", choices=["mock", "live"], default="live")
    mara.add_argument("--output", type=Path, default=BASE / "results")
    mara.add_argument("--models",nargs="+")
    graph = sub.add_parser("probe-graph")
    graph.add_argument("--mutate", action="store_true")
    graph.add_argument("--output", type=Path, default=BASE / "results")
    matrix = sub.add_parser("matrix")
    matrix.add_argument("--config", type=Path, default=BASE / "configs/factorial.live.json")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--run", type=Path, required=True)
    evaluation.add_argument("--gold", type=Path, default=BASE / "datasets/synthetic.gold.v1.json")
    study=sub.add_parser("study")
    study.add_argument("--output",type=Path,default=BASE/"results")
    study.add_argument("--question-limit",type=int,default=40)
    score=sub.add_parser("score-study")
    score.add_argument("--run",type=Path,required=True)
    score.add_argument("--output",type=Path,default=BASE/"results")
    score.add_argument("--gold",type=Path)
    args = parser.parse_args()
    if args.command == "run":
        return run_extraction(args)
    if args.command == "study":
        from .study import run_development_study
        print(json.dumps(run_development_study(args.output,question_limit=args.question_limit),indent=2))
        return 0
    if args.command == "score-study":
        from .study import score_saved_study
        print(json.dumps(score_saved_study(args.run,args.output,args.gold),indent=2))
        return 0
    if args.command == "matrix":
        config = load_config(args.config)
        print(json.dumps({"conditions_per_document": len(config.conditions()), "conditions": config.conditions()}, indent=2))
        return 0
    if args.command == "evaluate":
        gold = json.loads(args.gold.read_text())
        scores = []
        for sample in sorted((args.run / "samples").glob("*.json")):
            outcome = json.loads(sample.read_text())
            facts_path = args.run / "normalized" / (sample.stem + ".json")
            predictions = json.loads(facts_path.read_text()) if facts_path.exists() else []
            scoped_gold = {**gold, "assertions": [f for f in gold["assertions"]
                                                  if f["source_document_id"] == outcome["document_id"]]}
            score = evaluate(predictions, scoped_gold)
            scores.append({"sample_id": sample.stem, "condition_id": outcome["condition_id"],
                           "execution_status": outcome["status"], **score})
        metrics = {"gold_hash": digest(gold), "scope": "software fixture calibration; no human judgment supplied",
                   "independent_review_complete": gold["independent_review_complete"], "samples": scores}
        with (args.run / "fixture_metrics.json").open("x") as f:
            json.dump(metrics, f, indent=2)
        print(json.dumps(metrics, indent=2))
        return 0
    recorder = Recorder(args.output, getattr(args, "mode", "live"))
    if args.command == "probe-graph":
        manifest = probe_graph(mutate=args.mutate)
    else:
        config = load_config(args.config)
        try:
            gateway = Gateway(recorder, config, client=mock_client() if args.mode == "mock" else None)
        except GatewayError as exc:
            manifest = {"manifest_version": "mara-capabilities-v1", "mode": args.mode,
                        "models": {}, "blockers": [str(exc)], "http_attempts": 0}
        else:
            try:
                manifest = probe_mara(gateway,models=args.models)
                manifest["base_url"] = gateway.base
                manifest["http_attempts"] = len(recorder.readback())
            finally:
                gateway.close()
    recorder.write("capabilities.json", {**manifest, **runtime_manifest()})
    print(json.dumps({"run_directory": str(recorder.root), **manifest}, indent=2))
    return 1 if manifest.get("blockers") else 0


if __name__ == "__main__":
    raise SystemExit(main())
