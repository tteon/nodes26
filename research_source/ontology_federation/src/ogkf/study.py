"""Reproducible frozen RQ2/RQ3 study. Gold is used only after episode execution."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import statistics

import pyarrow as pa
import pyarrow.parquet as pq

from .agent import ExtractiveSynthesizer, run_agent
from .analysis import clustered_difference
from .fixtures import development_study
from .identity import Snapshot, build_variant, verify_variants
from .metrics import StudyGold, cost_summary, identity_metrics, qa_metrics
from .models import digest
from .observability import AGENT_SCHEMA, GRAPH_SCHEMA, Recorder
from .routing import MetadataRouter
from .storage import MemoryStore, QueryTrace

ARMS=[("V0","R0",False,True),("V0q","R0",False,True),
      ("V1","R1",False,True),("V1","R2",False,True),("V1","R3",False,True),
      ("V2","R1",False,True),("V2","R2",False,True),("V2","R3",False,True),
      ("V1","R2",True,True),("V1","R2",False,False)]


def arm_id(policy,strategy,reduced,provenance):
    return f"{policy}-{strategy}-{'reduced' if reduced else 'full'}-{'provenance' if provenance else 'no-provenance'}"


def write_table(path,rows,schema=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows,schema=schema),path)


def score_saved_study(run:Path,output:Path,gold_path:Path|None=None):
    """Recompute from immutable episode receipts without any LLM or DB calls.

    Useful for a corrected evaluator or interrupted reporting stage; prior
    receipts/reports are never overwritten. Evaluation gets a separate run ID.
    """
    from .cli import runtime_manifest
    snapshot=Snapshot.model_validate_json((run/"frozen/snapshot.json").read_text())
    gold=StudyGold.model_validate_json((run/"private/evaluation-gold.json").read_text())
    manifest=json.loads((run/"manifest.json").read_text())
    registry=json.loads((run/"frozen/registry.json").read_text())
    public=json.loads((run/"frozen/questions.public.json").read_text())
    if (snapshot.snapshot_hash!=manifest["snapshot_hash"] or digest(registry)!=manifest["registry_hash"] or
            digest(public)!=manifest["public_question_hash"] or digest(gold.model_dump())!=manifest["evaluator_gold_hash"]):
        raise ValueError("saved study frozen input hash mismatch")
    original_gold_hash=digest(gold.model_dump())
    if gold_path:
        gold=StudyGold.model_validate_json(gold_path.read_text())
    episodes=[json.loads(path.read_text()) for path in sorted((run/"episodes").glob("*.json"))]
    labels={q.question_id:q for q in gold.questions}
    scores=[]
    for episode in episodes:
        if episode["snapshot_hash"]!=snapshot.snapshot_hash or episode["registry_hash"]!=digest(registry):
            raise ValueError("episode belongs to another frozen graph or registry")
        scores.append({**qa_metrics(episode,labels[episode["question_id"]],snapshot,
            independent=gold.independent_review_complete and gold.origin=="independent_annotation"),"arm_id":episode["arm_id"]})
    evaluated=Recorder(output,"evaluation_only")
    identity=[identity_metrics(snapshot,build_variant(snapshot,p),gold) for p in ("V0","V0q","V1","V2")]
    summary={"input_run":str(run),"evaluation_run":str(evaluated.root),"origin":snapshot.origin,
        "original_gold_hash":original_gold_hash,"evaluated_gold_hash":digest(gold.model_dump()),
        "evaluator_inputs_hash":digest([snapshot.snapshot_hash,digest(gold.model_dump()),digest(episodes)]),
        "episodes":len(episodes),"episode_errors":sum(e["error"] is not None for e in episodes),
        "exact_claim_successes":sum(s["evidence_complete_grounded_qa_success"] is True for s in scores),
        "identity_metrics":identity,"costs":cost_summary(episodes,scores),"research_qualified":False,**runtime_manifest()}
    evaluated.write("summary.json",summary)
    evaluated.write("evaluation-gold.json",gold.model_dump())
    write_table(evaluated.root/"qa.parquet",scores)
    return summary


def run_development_study(output: Path, *, question_limit=40, stores=None, router=None, synthesizer=None,
                          arms=None, recorder=None, registry_override=None):
    from .cli import runtime_manifest
    snapshot,registry,questions,gold=development_study()
    registry=registry_override or registry
    if not 1<=question_limit<=40:
        raise ValueError("development question limit must be 1..40")
    recorder=recorder or Recorder(output,"offline_fixture")
    variants={p:build_variant(snapshot,p) for p in ("V0","V0q","V1","V2")}
    parity=verify_variants(snapshot,list(variants.values()))
    router,synthesizer=router or MetadataRouter(),synthesizer or ExtractiveSynthesizer()
    if stores is None:
        stores={p:MemoryStore(snapshot,v,{d:(next(e.database_name for e in registry.entries if e.domain_id==d)
                    if d!="flat" else "fixture"+p.lower()+"flat") for d in v.placement},QueryTrace(recorder))
                for p,v in variants.items()}
    if any(store.variant.snapshot_hash!=snapshot.snapshot_hash for store in stores.values()):
        raise ValueError("provided store snapshot differs from frozen study")
    expected_domains={e.domain_id:e.database_name for e in registry.entries}
    if any(store.databases!=expected_domains for store in stores.values() if store.variant.policy in {"V1","V2"}):
        raise ValueError("registry and executor physical database names differ")
    recorder.write("frozen/snapshot.json",snapshot.model_dump())
    recorder.write("frozen/registry.json",registry.model_dump())
    recorder.write("frozen/questions.public.json",[q.model_dump() for q in questions])
    recorder.write("private/evaluation-gold.json",gold.model_dump())
    recorder.write("manifest.json",{"experiment_run_id":recorder.run_id,"study_version":"frozen-study-v1",
        "origin":snapshot.origin,"snapshot_hash":snapshot.snapshot_hash,"registry_hash":registry.registry_hash,
        "public_question_hash":digest([q.model_dump() for q in questions]),"evaluator_gold_hash":digest(gold.model_dump()),
        "router":router.version,"synthesizer":synthesizer.version,"arms":arms or ARMS,
        "question_limit":question_limit,"independent_review_complete":False,"assertion_parity":parity,**runtime_manifest()})
    episodes=[]
    for policy,strategy,reduced,provenance in arms or ARMS:
        arm=arm_id(policy,strategy,reduced,provenance)
        for question in questions[:question_limit]:
            episode=run_agent(question,strategy,stores[policy],registry,router,synthesizer,
                              reduced_metadata=reduced,provenance=provenance)
            episode["arm_id"]=arm
            recorder.write("episodes/"+episode["trace_id"]+".json",episode)
            episodes.append(episode)
    # Only the evaluator receives gold, after all public episodes have terminated.
    labels={q.question_id:q for q in gold.questions}
    scores=[{**qa_metrics(e,labels[e["question_id"]],snapshot,independent=False),"arm_id":e["arm_id"]}
            for e in episodes]
    identity=[identity_metrics(snapshot,v,gold) for v in variants.values()]
    by_arm=defaultdict(list)
    for e,s in zip(episodes,scores):
        by_arm[e["arm_id"]].append((e,s))
    comparisons=[]
    baseline=arm_id("V1","R1",False,True)
    if baseline in by_arm:
        for alternative in (arm_id("V1","R2",False,True),arm_id("V1","R3",False,True)):
            if alternative in by_arm:
                comparisons.append(clustered_difference(scores,left_arm=baseline,right_arm=alternative,
                    metric="evidence_complete_grounded_qa_success"))
    costs={arm:cost_summary([p[0] for p in pairs],[p[1] for p in pairs]) for arm,pairs in by_arm.items()}
    summary={"run_directory":str(recorder.root),"origin":"synthetic_fixture","research_qualified":False,
        "sources":len(snapshot.sources),"assertions":len(snapshot.assertions),"entities":len(snapshot.entities),
        "identity_pairs":len(gold.identity_pairs),"questions":min(len(questions),question_limit),
        "episodes":len(episodes),"episode_errors":sum(e["error"] is not None for e in episodes),
        "assertion_parity":parity,"identity_metrics":identity,"paired_contrasts":comparisons,"costs":costs,
        "arm_success_counts":{arm:sum(p[1]["evidence_complete_grounded_qa_success"] is True for p in pairs)
                              for arm,pairs in by_arm.items()},
        "interpretation":"software fixture outcomes only; labels/answers are agent-authored; no supported or rejected research hypotheses"}
    recorder.write("metrics/summary.json",summary)
    write_table(recorder.root/"metrics/qa.parquet",scores)
    write_table(recorder.root/"metrics/identity.parquet",identity)
    graph_rows=[r for store in stores.values() for r in store.telemetry.rows]
    if graph_rows:
        # Rows are a fixed operation schema; nested execution plans remain null unless profiled.
        write_table(recorder.root/f"traces/experiment=federation/date={recorder.date}/run={recorder.run_id}/graph.parquet",graph_rows,GRAPH_SCHEMA)
    agent_rows=[{k:v for k,v in e.items() if k in {"trace_id","question_id","arm_id","strategy","identity_policy",
        "graph_provider","tool_call_count","routing_latency_ms","synthesis_latency_ms","end_to_end_latency_ms",
        "stop_reason","error","row_cap_reached"}} for e in episodes]
    write_table(recorder.root/f"traces/experiment=federation/date={recorder.date}/run={recorder.run_id}/agents.parquet",agent_rows,AGENT_SCHEMA)
    # Verify persisted row counts rather than trusting only in-memory serialization.
    if pq.ParquetFile(recorder.root/"metrics/qa.parquet").metadata.num_rows!=len(episodes):
        raise ValueError("study Parquet readback mismatch")
    return summary
