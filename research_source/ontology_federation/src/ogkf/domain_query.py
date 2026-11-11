"""Synthetic related-entity predicate checks, never a semantic or scaling benchmark."""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading
import time
import uuid

from neo4j import GraphDatabase, Query, READ_ACCESS
from pydantic import Field

from .context_frame.challenge import ChallengeLedger, save
from .context_frame.scenario_store import inspect_owned_runtime
from .local_runtime import local_dozer
from .models import Strict, digest

DATA_KEYS = ("movies", "productions", "people", "credits")
DOMAINS = ("catalog", "production", "talent")
DOMAIN_OF = dict(zip(DATA_KEYS, ("catalog", "production", "talent", "talent")))
PERSON = "p.person_id=$person_id AND c.role=$role AND c.snapshot_date=$snapshot"
PRODUCTION = "p.phase=$phase AND p.snapshot_date=$snapshot"
FILTERS = {
    "production": f"MATCH(p:Production)-[:FOR_MOVIE]->(m:MovieRef) WHERE {PRODUCTION} RETURN m.movie_id AS id,p.payload AS production",
    "talent": f"MATCH(p:Person)-[c:PARTICIPATED_IN]->(m:MovieRef) WHERE {PERSON} RETURN m.movie_id AS id,c.payload AS credit,p.payload AS person",
    "catalog": "MATCH(m:Movie) WHERE m.movie_id IN $ids AND m.snapshot_date=$snapshot RETURN m.movie_id AS id,m.title AS title,m.payload AS catalog",
}
FLAT = """MATCH(p:Production)-[:FOR_MOVIE]->(ref:MovieRef)
MATCH(person:Person)-[c:PARTICIPATED_IN]->(ref)
MATCH(m:Movie) WHERE m.movie_id=ref.movie_id
AND p.phase=$phase AND person.person_id=$person_id AND c.role=$role
AND p.snapshot_date=$snapshot AND c.snapshot_date=$snapshot AND m.snapshot_date=$snapshot
RETURN m.movie_id AS id,m.title AS title,m.payload AS catalog,
p.payload AS production,c.payload AS credit,person.payload AS person"""


class BusinessQuery(Strict):
    query_id: str
    text: str
    person_id: str | None
    phase: str = Field(pattern=r"^(PHOTOGRAPHY|POST_PRODUCTION)$")
    role: str = Field(pattern=r"^(ACTOR|PRODUCER|DIRECTOR)$")
    snapshot_date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


def load_fixture(path):
    data = json.loads(path.read_text())
    if data.get("synthetic") is not True:
        raise ValueError(
            "this executor is qualified only for explicit synthetic fixtures"
        )
    movies = {r["movie_id"] for r in data["movies"]}
    people = {r["person_id"] for r in data["people"]}
    for key, field in zip(
        DATA_KEYS, ("movie_id", "production_id", "person_id", "credit_id")
    ):
        if len({r[field] for r in data[key]}) != len(data[key]):
            raise ValueError("duplicate fixture identifiers")
    for row in data["productions"] + data["credits"]:
        if row["movie_id"] not in movies:
            raise ValueError("unresolved fixture movie reference")
    if any(r["person_id"] not in people for r in data["credits"]):
        raise ValueError("unresolved fixture person reference")
    if any(r["role"] not in {"ACTOR", "PRODUCER", "DIRECTOR"} for r in data["credits"]):
        raise ValueError("invalid fixture role")
    snapshot = digest({k: data[k] for k in (*DATA_KEYS, "snapshot_date")})
    records = {}
    for key in DATA_KEYS:
        for i, row in enumerate(data[key]):
            eid = digest([snapshot, key, i, row])
            records[eid] = {
                "evidence_id": eid,
                "domain_id": DOMAIN_OF[key],
                "source_pointer": str(path) + f"#/{key}/{i}",
                "document_version": hashlib.sha256(path.read_bytes()).hexdigest(),
                "snapshot_date": data["snapshot_date"],
                "synthetic": True,
                "record": row,
            }
    return data, snapshot, records


def guard(query, data):
    if query.person_id not in {p["person_id"] for p in data["people"]}:
        return (
            "CLARIFY",
            "A fixture person ID must be selected; names may be ambiguous.",
        )
    if query.snapshot_date != data["snapshot_date"]:
        return "ABSTAIN", "The requested snapshot is not available in this fixture."
    return None


class Projection:
    """A frozen SQLite predicate projection, not an Elasticsearch implementation."""

    def __init__(self, data, records):
        self.db = sqlite3.connect(":memory:")
        self.snapshot_date = data["snapshot_date"]
        self.records = records
        self.by_row = {digest(v["record"]): k for k, v in records.items()}
        self.db.execute(
            "CREATE TABLE view(movie_id TEXT,person_id TEXT,role TEXT,phase TEXT,production TEXT,credit TEXT,person TEXT)"
        )
        people = {p["person_id"]: p for p in data["people"]}
        for p in data["productions"]:
            for c in data["credits"]:
                if c["movie_id"] == p["movie_id"]:
                    self.db.execute(
                        "INSERT INTO view VALUES(?,?,?,?,?,?,?)",
                        (
                            p["movie_id"],
                            c["person_id"],
                            c["role"],
                            p["phase"],
                            json.dumps(p),
                            json.dumps(c),
                            json.dumps(people[c["person_id"]]),
                        ),
                    )
        self.db.execute("CREATE INDEX predicates ON view(person_id,role,phase)")

    def select(self, query):
        if query.snapshot_date != self.snapshot_date:
            raise ValueError("projection snapshot mismatch")
        rows = self.db.execute(
            "SELECT movie_id,production,credit,person FROM view WHERE person_id=? AND role=? AND phase=?",
            (query.person_id, query.role, query.phase),
        ).fetchall()
        return [
            {"id": r[0], "production": r[1], "credit": r[2], "person": r[3]}
            for r in rows
        ]

    def close(self):
        self.db.close()


def packet(query, rows, records, snapshot):
    by_row = {digest(v["record"]): k for k, v in records.items()}
    groups = defaultdict(set)
    titles = {}
    for row in rows:
        titles[row["id"]] = row["title"]
        for field in ("catalog", "production", "credit", "person"):
            value = row[field]
            value = json.loads(value) if isinstance(value, str) else value
            groups[row["id"]].add(by_row[digest(value)])
    results = [
        {"movie_id": key, "title": titles[key], "evidence_ids": sorted(ids)}
        for key, ids in sorted(groups.items())
    ]
    selected = sorted({eid for row in results for eid in row["evidence_ids"]})
    return {
        "question": query.model_dump(),
        "snapshot_id": snapshot,
        "synthetic": True,
        "action": "ANSWER",
        "results": results,
        "evidence": [records[eid] for eid in selected],
        "semantic_correctness": None,
        "LLM_calls": 0,
    }


class Store:
    def __init__(self, environment, data, records, ledger):
        self.driver = GraphDatabase.driver(
            environment["DOZER_URI"],
            auth=(environment["DOZER_USER"], environment["DOZER_PASSWORD"]),
            max_transaction_retry_time=0,
        )
        self.names = {d: "ogkf" + uuid.uuid4().hex[:12] + d for d in (*DOMAINS, "flat")}
        self.created = []
        self.data, self.records, self.ledger = data, records, ledger
        self.lock = threading.Lock()

    def run(self, db, cypher, parameters, *, read=False, sample=None, template=None):
        if db != "system" and db not in self.names.values():
            raise ValueError("database outside owned allowlist")
        if read:
            cypher = "CALL { " + cypher + " } RETURN * LIMIT $record_cap"
            parameters = {**parameters, "record_cap": 101}
        start, error, rows = time.perf_counter(), None, []
        try:
            with self.driver.session(
                database=db, default_access_mode=READ_ACCESS if read else "WRITE"
            ) as session:
                rows = session.run(Query(cypher, timeout=20), parameters).data()
                if len(rows) > 100:
                    raise ValueError("bounded query result cap exceeded")
            return rows
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            with self.lock:
                self.ledger.emit(
                    "graph_query_traces" if read else "graph_operation_traces",
                    sample_id=sample,
                    trace_id=str(uuid.uuid4()),
                    database_name=db,
                    query_template_id=template,
                    query=cypher,
                    query_parameters=parameters,
                    query_parameters_hash=digest(parameters),
                    query_latency_ms=(time.perf_counter() - start) * 1000,
                    records_returned=len(rows),
                    query_error=error,
                    db_hits=None,
                )

    def build(self):
        payloads = {
            digest(v["record"]): json.dumps(v["record"], sort_keys=True)
            for v in self.records.values()
        }
        for domain, db in self.names.items():
            if not re.fullmatch(r"ogkf[a-z0-9]+", db):
                raise ValueError("unsafe owned database")
            self.run("system", f"CREATE DATABASE `{db}`", {})
            self.created.append(db)
            for key, label, identity in (
                ("movies", "Movie", "movie_id"),
                ("productions", "Production", "production_id"),
                ("people", "Person", "person_id"),
            ):
                if domain != "flat" and DOMAIN_OF[key] != domain:
                    continue
                rows = [
                    {
                        **r,
                        "payload": payloads[digest(r)],
                        "snapshot_date": self.data["snapshot_date"],
                    }
                    for r in self.data[key]
                ]
                self.run(
                    db,
                    f"UNWIND $rows AS row CREATE(n:{label}) SET n=row",
                    {"rows": rows},
                )
                self.run(
                    db,
                    f"CREATE CONSTRAINT fixture_{label} FOR(n:{label}) REQUIRE n.{identity} IS UNIQUE",
                    {},
                )
            if domain in {"production", "flat"}:
                self.run(
                    db,
                    "MATCH(p:Production) MERGE(m:MovieRef {movie_id:p.movie_id}) MERGE(p)-[:FOR_MOVIE]->(m)",
                    {},
                )
            if domain in {"talent", "flat"}:
                rows = [
                    {
                        **r,
                        "payload": payloads[digest(r)],
                        "snapshot_date": self.data["snapshot_date"],
                    }
                    for r in self.data["credits"]
                ]
                self.run(
                    db,
                    "UNWIND $rows AS row MATCH(p:Person {person_id:row.person_id}) MERGE(m:MovieRef {movie_id:row.movie_id}) CREATE(p)-[c:PARTICIPATED_IN]->(m) SET c=row",
                    {"rows": rows},
                )
        readback = {}
        for domain, db in self.names.items():
            rows = self.run(
                db,
                "MATCH(n) WHERE n.payload IS NOT NULL RETURN n.payload AS payload UNION ALL MATCH()-[c:PARTICIPATED_IN]->() RETURN c.payload AS payload",
                {},
                read=True,
                template="snapshot_readback",
            )
            readback[domain] = sorted(r["payload"] for r in rows)
        expected = sorted(payloads.values())
        if (
            readback["flat"] != expected
            or sorted(r for d in DOMAINS for r in readback[d]) != expected
        ):
            raise ValueError(
                "single/multi graph does not preserve the same fixture records"
            )
        return {
            "single_multi_payload_parity": True,
            "records": len(expected),
            "domains": list(DOMAINS),
            "native_composite": False,
        }

    def execute(self, query, arm, projection, sample):
        params = {
            "phase": query.phase,
            "person_id": query.person_id,
            "role": query.role,
            "snapshot": query.snapshot_date,
        }
        if arm == "single":
            return self.run(
                self.names["flat"],
                FLAT,
                params,
                read=True,
                sample=sample,
                template="qualified_single_join",
            )
        if arm == "federated":
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    d: pool.submit(
                        self.run,
                        self.names[d],
                        FILTERS[d],
                        params,
                        read=True,
                        sample=sample,
                        template=d + "_predicate",
                    )
                    for d in ("production", "talent")
                }
                found = {d: f.result() for d, f in futures.items()}
            candidates = [
                {**a, **b}
                for a in found["production"]
                for b in found["talent"]
                if a["id"] == b["id"]
            ]
        elif arm == "projection":
            start = time.perf_counter()
            candidates = projection.select(query)
            self.ledger.emit(
                "projection_query_traces",
                sample_id=sample,
                provider="sqlite_frozen_projection",
                query_latency_ms=(time.perf_counter() - start) * 1000,
                records_returned=len(candidates),
                snapshot_date=query.snapshot_date,
            )
        else:
            raise ValueError("unknown strategy")
        ids = sorted({r["id"] for r in candidates})
        catalogs = self.run(
            self.names["catalog"],
            FILTERS["catalog"],
            {**params, "ids": ids},
            read=True,
            sample=sample,
            template="catalog_hydration",
        )
        return [{**a, **b} for a in candidates for b in catalogs if a["id"] == b["id"]]

    def close(self):
        dropped = []
        try:
            for name in reversed(self.created):
                self.run("system", f"DROP DATABASE `{name}` DESTROY DATA", {})
                dropped.append(name)
        finally:
            self.driver.close()
        return {"owned_databases_dropped": dropped}


def run(args):
    data, snapshot, records = load_fixture(args.data)
    queries = [
        BusinessQuery.model_validate_json(line)
        for line in args.questions.read_text().splitlines()
        if line
    ]
    expected = json.loads(args.expected.read_text())["cases"]
    if set(expected) != {q.query_id for q in queries}:
        raise ValueError("expectations and public question IDs differ")
    args.output.mkdir(parents=True, exist_ok=False)
    run_id = str(uuid.uuid4())
    ledger = ChallengeLedger(args.output, run_id)
    save(
        args.output / "input-snapshot.json",
        {
            "data": data,
            "source_records": list(records.values()),
            "public_questions": [q.model_dump() for q in queries],
            "snapshot_id": snapshot,
            "expected_sidecar_not_agent_input": True,
            "implementation_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            "input_file_hashes": {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (args.data, args.questions, args.expected)
            },
        },
    )
    save(args.output / "expected.evaluator-only.json", expected)
    outcomes = []
    projection = Projection(data, records)
    runtime = None
    try:
        with local_dozer() as (environment, runtime):
            save(
                args.output / "graph-capabilities.json",
                inspect_owned_runtime(environment),
            )
            store = Store(environment, data, records, ledger)
            try:
                parity = store.build()
                save(args.output / "graph-parity.json", parity)
                for q in queries:
                    for arm in ("single", "federated", "projection"):
                        sample = digest([run_id, snapshot, q.query_id, arm])
                        refusal = guard(q, data)
                        if refusal:
                            result = {
                                "action": refusal[0],
                                "reason": refusal[1],
                                "results": [],
                                "evidence": [],
                                "synthetic": True,
                                "LLM_calls": 0,
                                "semantic_correctness": None,
                            }
                        else:
                            result = packet(
                                q,
                                store.execute(q, arm, projection, sample),
                                records,
                                snapshot,
                            )
                        ids = sorted(r["movie_id"] for r in result["results"])
                        matches = (
                            result["action"] == expected[q.query_id]["action"]
                            and ids == expected[q.query_id]["movie_ids"]
                        )
                        row = ledger.emit(
                            "query_results",
                            sample_id=sample,
                            trace_id=str(uuid.uuid4()),
                            snapshot_id=snapshot,
                            query_id=q.query_id,
                            strategy=arm,
                            packet=result,
                            expected_fixture_agreement=matches,
                            independent_semantic_gold=False,
                        )
                        outcomes.append(row)
            finally:
                save(args.output / "cleanup.json", store.close())
    finally:
        if runtime is not None:
            save(args.output / "local-runtime.json", runtime)
        save(args.output / "parquet-readback.json", ledger.verify())
        projection.close()
    counts = ledger.verify()
    summary = {
        "synthetic": True,
        "public_queries": len(queries),
        "strategies": ["single", "federated", "projection"],
        "episodes": len(outcomes),
        "declared_fixture_expectations_matched": sum(
            r["expected_fixture_agreement"] for r in outcomes
        ),
        "same_logical_records": parity["single_multi_payload_parity"],
        "parquet": counts,
        "LLM_calls": 0,
        "independent_semantic_gold": 0,
        "native_composite_executed": False,
        "scaling_benchmark": False,
        "ontology_lift": None,
        "latency_advantage": None,
        "cold_local_engineering_run": True,
    }
    save(args.output / "summary.json", summary)
    if not all(r["expected_fixture_agreement"] for r in outcomes):
        raise ValueError(
            "fixture expectation mismatch; preserve receipts and investigate"
        )
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--questions", type=Path, required=True)
    p.add_argument("--expected", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(p.parse_args()), indent=2))


if __name__ == "__main__":
    main()
