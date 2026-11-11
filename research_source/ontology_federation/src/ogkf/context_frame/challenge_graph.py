"""Execute frozen graph reads across identical single/multi-DB snapshots."""

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace
import uuid

from neo4j import Query as CypherQuery, READ_ACCESS

from ..ds01.contracts import DOMAINS, Query
from ..ds01.retrieval import FixedIndex, Planner, retrieve
from ..ds01.storage import BoltExecutor, MemoryExecutor
from ..local_runtime import local_dozer
from ..models import digest
from .challenge import ChallengeLedger, admission, load_snapshot, save

GRAPH_QUERY = """MATCH (s:DS01Snapshot {snapshot_id:$sid})-[:INCLUDES]->(d:DS01Document)
WHERE d.document_id IN $ids AND d.domain_id IN $domains
CALL (d) {
  OPTIONAL MATCH (d)<-[:FROM_SOURCE]-(a:DS01Assertion)-[:SUBJECT]->(e:DS01Entity)
  WITH a,e WHERE a IS NOT NULL ORDER BY a.assertion_id LIMIT $assertion_cap
  RETURN collect({assertion:a.payload,subject:e.payload}) AS bindings
}
RETURN d.payload AS payload,bindings LIMIT $limit"""


def payloads(dataset, cap=16):
    by_doc = defaultdict(list)
    entities = {e.entity_id: e.model_dump() for e in dataset.entities}
    for a in dataset.assertions:
        by_doc[a.document_id].append(a)
    docs, bindings = {}, {}
    for d in dataset.documents:
        ordered = sorted(by_doc[d.document_id], key=lambda a: a.assertion_id)
        docs[d.document_id] = {
            **d.model_dump(),
            "assertion_ids": [a.assertion_id for a in ordered],
        }
        bindings[d.document_id] = [
            {"assertion": a.model_dump(), "subject": entities[a.subject_id]}
            for a in ordered[:cap]
        ]
    return docs, bindings


class ChallengeMemoryExecutor(MemoryExecutor):
    version = "source-proposal-graph-read-v1"

    def __init__(self, dataset, ledger):
        self.dataset, self.ledger = dataset, ledger
        self.docs, self.bindings = payloads(dataset)

    def read(self, corpus, ids, domains, *, flat, common):
        rows = super().read(corpus, ids, domains, flat=flat, common=common)
        return [{**r, "graph_bindings": self.bindings[r["document_id"]]} for r in rows]


class ChallengeBoltExecutor(BoltExecutor):
    version = "source-proposal-graph-read-v1"

    def __init__(self, dataset, ledger, environment):
        # Reuse owned DB materialization/parity without quadratic assertion scans.
        proxy = SimpleNamespace(documents=dataset.documents, assertions=[])
        super().__init__(proxy, ledger, environment)
        self.dataset = dataset
        self.docs, self.bindings = payloads(dataset)
        prefix = "challenge-" + uuid.uuid4().hex[:8]
        suffix = {
            "flat": "flat",
            "finance_reporting": "finance",
            "legal_governance": "legal",
            "business_risk": "risk",
        }
        self.names = {k: prefix + "-" + v for k, v in suffix.items()}

    def _owned(self, name):
        if name not in self.names.values() or not re.fullmatch(
            r"challenge-[a-f0-9]{8}-(flat|finance|legal|risk)", name
        ):
            raise ValueError("database outside owned challenge allowlist")
        return name

    def read(self, corpus, ids, domains, *, flat, common):
        if (
            not set(ids) <= set(corpus.document_ids)
            or len(ids) > 20
            or not domains
            or not set(domains) <= set(DOMAINS)
            or (not flat and len(domains) != 1)
        ):
            raise ValueError("challenge graph read scope escape")
        name = self._owned(self.names["flat" if flat else domains[0]])
        params = {
            "sid": corpus.graph_snapshot_id,
            "ids": ids,
            "domains": domains,
            "assertion_cap": 16,
            "limit": len(ids),
        }
        start = time.perf_counter()
        rows = []
        error = None
        try:
            with self.driver.session(
                database=name, default_access_mode=READ_ACCESS
            ) as session:
                result = session.run(CypherQuery(GRAPH_QUERY, timeout=10), params)
                for r in result:
                    doc = json.loads(r["payload"])
                    links = [
                        {
                            "assertion": json.loads(b["assertion"]),
                            "subject": json.loads(b["subject"]),
                        }
                        for b in r["bindings"]
                    ]
                    if (
                        doc != self.docs[doc["document_id"]]
                        or links != self.bindings[doc["document_id"]]
                    ):
                        raise ValueError(
                            "graph path differs from frozen source/subject/assertion/evidence"
                        )
                    rows.append({**doc, "graph_bindings": links})
                result.consume()
            expected = {rid for rid in ids if self.docs[rid]["domain_id"] in domains}
            if {r["document_id"] for r in rows} != expected:
                raise ValueError("graph read returned incomplete document inventory")
            return rows
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            self.ledger.emit(
                "graph_query_traces",
                **common,
                provider_id="flat" if flat else domains[0],
                database_name=name,
                graph_provider="dozerdb",
                query_template_id=self.version,
                query_parameters_hash=digest(params),
                query_latency_ms=(time.perf_counter() - start) * 1000,
                records_returned=len(rows),
                assertion_bindings_returned=sum(len(r["graph_bindings"]) for r in rows),
                db_hits=None,
                error=error,
            )


class CachedIndex(FixedIndex):
    """Reuse identical public-query ranking across placement strategies only."""

    def rank(self, query, corpus):
        key = digest([query.model_dump(), corpus.graph_snapshot_id])
        if getattr(self, "last_key", None) != key:
            self.last_rank = super().rank(query, corpus)
            self.last_key = key
        return self.last_rank


def execute(dataset, questions, snapshot, args, root, ledger, executor, shift):
    planner, index = Planner(dataset), CachedIndex(dataset)
    config = SimpleNamespace(
        global_top_k=args.top_k, context_byte_budget=args.context_bytes
    )
    selected = questions[: args.limit] if args.limit else questions
    if not selected or args.limit < 0 or not 1000 <= args.context_bytes <= 200000:
        raise ValueError("invalid query/context budget")
    counts = defaultdict(lambda: Counter())
    failures = []
    witness = None
    for position, q in enumerate(selected):
        query = Query(query_id=q["query_id"], text=q["text"])
        # These receive only the public Query and source-derived registry.
        plan = planner.plan(query, str(uuid.uuid4()), shift.condition_id)
        paired_results = {}
        for strategy in ("B0", "B1", "B2", "B3", "B4"):
            result = retrieve(
                query,
                shift,
                dataset,
                config,
                strategy,
                executor,
                planner,
                index,
                ledger,
                plan=plan if strategy in {"B1", "B3", "B4"} else None,
                phase="challenge_retrieval_only",
            )
            required = set(q["source_ids"])
            candidates = set(result["candidate_document_ids"])
            delivered = {r["document_id"] for r in result["evidence"]}
            score = bool(required <= delivered) if required else None
            scope = (
                "native_provided_reference_proxy"
                if q["origin"] == "native_unchanged"
                else "assigned_source_coverage_not_semantic_gold"
            )
            record = ledger.emit(
                "evaluation_results",
                query_id=query.query_id,
                trace_id=result["trace_id"],
                graph_snapshot_id=shift.corpus.graph_snapshot_id,
                dataset_hash=snapshot,
                strategy=strategy,
                origin=q["origin"],
                parent_ids=q["parent_ids"],
                metric_scope=scope,
                required_source_ids=sorted(required),
                candidate_source_recall=len(required & candidates) / len(required),
                delivered_reference_recall=len(required & delivered) / len(required),
                all_assigned_references_delivered=score,
                grounded_QA_success=None,
                context_leakage=None,
                provider_selection_precision=None,
                provider_selection_recall=None,
                tool_calls=result["tool_calls"],
                context_bytes=result["context_bytes"],
                retrieval_latency_ms=result["end_to_end_ms"],
                answer_generation="not_executed",
                error=result["error"],
            )
            key = q["origin"] + "/" + strategy
            counts[key]["episodes"] += 1
            counts[key]["reference_proxy_passes"] += int(bool(score))
            counts[key]["errors"] += int(bool(result["error"]))
            counts[key]["tool_calls"] += result["tool_calls"]
            paired_results[strategy] = sorted(delivered)
            if result["error"]:
                failures.append(record)
                raise RuntimeError(
                    "graph query failed; preserve trace and stop expansion"
                )
            if witness is None and result["evidence"]:
                witness = {
                    "public_question": query.model_dump(),
                    "plan": plan.model_dump(),
                    "routing_mode": "deterministic_metadata_not_LLM",
                    "graph_query_template": GRAPH_QUERY,
                    "trace_id": result["trace_id"],
                    "strategy": strategy,
                    "delivered_evidence": result["evidence"],
                    "answer": None,
                }
        if paired_results["B1"] != paired_results["B3"]:
            raise ValueError(
                "same-plan single/multi placement delivered different evidence"
            )
        if (position + 1) % 50 == 0:
            print(
                json.dumps(
                    {
                        "queries_completed": position + 1,
                        "total_queries": len(selected),
                        "graph_provider": executor.provider,
                    }
                ),
                flush=True,
            )
    if witness:
        save(root / "routing-and-query-witness.json", witness)
    return {
        "queries": len(selected),
        "native_queries": sum(q["origin"] == "native_unchanged" for q in selected),
        "generated_silver_queries": sum(
            q["origin"] != "native_unchanged" for q in selected
        ),
        "by_origin_strategy": dict(counts),
        "execution_failures": len(failures),
        "same_plan_B1_B3_packet_parity": True,
        "answer_model_calls": 0,
        "semantic_effect": None,
        "federation_mode": "application_level"
        if executor.provider == "dozerdb"
        else "memory",
        "query_mode": "fixed_parameterized_graph_traversal_not_LLM_generated_Cypher",
    }


def run_retrieval(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    dataset, questions, snapshot = load_snapshot(args.snapshot)
    query_ids = getattr(args, "query_ids", None)
    if query_ids:
        from .challenge import select_ids

        if args.limit and len(query_ids) > args.limit:
            raise ValueError("explicit queries exceed the declared limit")
        questions = select_ids(questions, query_ids, lambda q: q["query_id"])
    # Interleave origins, so a bounded pilot does not silently omit generated questions.
    groups = defaultdict(list)
    for q in questions:
        groups[q["origin"]].append(q)
    questions = []
    for i in range(max(map(len, groups.values()))):
        questions.extend(rows[i] for _, rows in sorted(groups.items()) if i < len(rows))
    ledger = ChallengeLedger(root, str(uuid.uuid4()))
    logical = digest(
        [d.model_dump() for d in dataset.documents]
        + [a.model_dump() for a in dataset.assertions]
    )
    corpus = SimpleNamespace(
        corpus_id=snapshot,
        graph_snapshot_id="challenge-" + snapshot,
        document_ids=[d.document_id for d in dataset.documents],
        assertion_ids=[a.assertion_id for a in dataset.assertions],
        logical_payload_hash=logical,
    )
    shift = SimpleNamespace(
        corpus=corpus,
        seed=20261031,
        severity=0,
        intervention="none",
        condition_id=digest([snapshot, args.top_k, args.context_bytes]),
    )
    save(root / "admission.json", admission())
    save(
        root / "protocol.json",
        {
            "snapshot_hash": snapshot,
            "source_snapshot": str(args.snapshot),
            "graph": args.graph,
            "global_top_k": args.top_k,
            "context_byte_budget": args.context_bytes,
            "limit": args.limit,
            "selected_query_ids": query_ids,
            "selection_mode": "explicit_engineering_cohort"
            if query_ids
            else "interleaved_origins",
            "ranking": "fixed source BM25; same index/scores in every strategy",
            "question_categories_used_by_router": False,
            "native_answer_used_by_extractor": False,
            "LLM_routing": False,
            "free_Cypher_generation": False,
            "primary_eligible_metric": "provided_reference_coverage_proxy; semantic outcomes unreviewed",
            "queries": GRAPH_QUERY,
            "ontology_version": dataset.ontology_version,
        },
    )
    if args.graph == "local-dozer":
        with local_dozer() as (environment, runtime):
            executor = ChallengeBoltExecutor(dataset, ledger, environment)
            try:
                capabilities = executor.build([shift])
                save(root / "capabilities.json", capabilities)
                summary = execute(
                    dataset, questions, snapshot, args, root, ledger, executor, shift
                )
                save(root / "profile-diagnostics.json", executor.profile(corpus))
            finally:
                save(root / "database-cleanup.json", executor.close())
        save(root / "local-runtime.json", runtime)
    else:
        executor = ChallengeMemoryExecutor(dataset, ledger)
        summary = execute(
            dataset, questions, snapshot, args, root, ledger, executor, shift
        )
    summary["parquet_readback"] = ledger.verify()
    save(root / "summary.json", summary)
    return summary
