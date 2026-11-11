"""Owned local native-DB synthetic assertion roundtrip;no LLM or semantic score."""

import argparse
import json
import re
import time
import uuid
from pathlib import Path
from neo4j import GraphDatabase, Query, READ_ACCESS
import pyarrow as pa
import pyarrow.parquet as pq
from ..local_runtime import local_dozer
from ..models import digest
from .contracts import Ontology, Snapshot
from .fixtures import synthetic_fixture
from .storage import batches


def domain_snapshot(snapshot, domain):
    p = snapshot.model_dump()
    reps = {
        r.representation_id for r in snapshot.representations if r.domain_id == domain
    }
    p["assertions"] = [
        a for a in p["assertions"] if a["subject_representation_id"] in reps
    ]
    span_ids = {s for a in p["assertions"] for s in a["evidence_span_ids"]}
    p["spans"] = [s for s in p["spans"] if s["span_id"] in span_ids]
    doc_ids = {s["document_id"] for s in p["spans"]}
    p["documents"] = [d for d in p["documents"] if d["document_id"] in doc_ids]
    pub_ids = {d["publisher_id"] for d in p["documents"]}
    p["publishers"] = [r for r in p["publishers"] if r["publisher_id"] in pub_ids]
    p["representations"] = [
        r for r in p["representations"] if r["representation_id"] in reps
    ]
    entity_ids = {r["entity_id"] for r in p["representations"]}
    p["entities"] = [e for e in p["entities"] if e["entity_id"] in entity_ids]
    return Snapshot.model_validate(p)


def run(ontology_path, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    ontology = Ontology.model_validate_json(Path(ontology_path).read_text())
    snapshot, _ = synthetic_fixture(ontology)
    prefix = "ogkfent" + uuid.uuid4().hex[:12]
    names = {k: prefix + k for k in ["flat", "finance", "sustainability", "legal"]}
    created = []
    operations = []
    runtime = None
    report = dict(
        protocol_id="SYN-NATIVE-ENTERPRISE-01",
        experiment_run_id=str(uuid.uuid4()),
        trace_id=str(uuid.uuid4()),
        synthetic=True,
        source_snapshot_id=snapshot.snapshot_id,
        federation_mode="application_level",
        native_composite_used=False,
        semantic_primary=None,
        llm_calls=0,
        server_acl="ephemeral admin;READ sessions do not establish least-privilege ACL",
        created_databases=created,
    )

    def save(name, value):
        with (output / name).open("x") as f:
            json.dump(value, f, indent=2)
            f.write("\n")

    try:
        with local_dozer() as (environment, runtime):
            driver = GraphDatabase.driver(
                environment["DOZER_URI"],
                auth=("neo4j", environment["DOZER_PASSWORD"]),
                connection_timeout=5,
                max_transaction_retry_time=0,
            )

            def query(db, template, parameters=None, *, read=False):
                if db != "system" and db not in names.values():
                    raise ValueError("DB outside owned allowlist")
                start = time.perf_counter()
                error = None
                rows = None
                try:
                    with driver.session(
                        database=db,
                        default_access_mode=READ_ACCESS if read else "WRITE",
                    ) as session:
                        result = session.run(
                            Query(template, timeout=10), parameters or {}
                        )
                        rows = result.data()
                        result.consume()
                except Exception as exc:
                    error = type(exc).__name__
                    raise
                finally:
                    operations.append(
                        dict(
                            experiment_run_id=report["experiment_run_id"],
                            trace_id=report["trace_id"],
                            operation_id=str(uuid.uuid4()),
                            graph_snapshot_id=snapshot.snapshot_id,
                            database_name=db,
                            query_template_id=digest(template),
                            query_parameters_hash=digest(parameters or {}),
                            query_latency_ms=(time.perf_counter() - start) * 1000,
                            records_returned=len(rows) if rows is not None else None,
                            read_only=read,
                            query_error=error,
                        )
                    )
                return rows

            try:
                report["components"] = query(
                    "system",
                    "CALL dbms.components() YIELD name,versions,edition RETURN name,versions,edition",
                    read=True,
                )
                ddl = (
                    Path(ontology_path).resolve().parents[1]
                    / "migrations/001_enterprise_qualified_assertions.cypher"
                ).read_text()
                # Statements are fixed source DDL;never generated or user supplied Cypher.
                constraints = [
                    s.strip() + ";"
                    for s in re.sub(r"//[^\n]*", "", ddl).split(";")
                    if s.strip()
                ]
                for key, db in names.items():
                    if not re.fullmatch(
                        r"ogkfent[a-f0-9]{12}(flat|finance|sustainability|legal)", db
                    ):
                        raise ValueError("invalid owned DB")
                    query("system", f"CREATE DATABASE `{db}`")
                    created.append(db)
                    deadline = time.monotonic() + 30
                    while True:
                        try:
                            query(db, "RETURN 1 AS ok", read=True)
                            break
                        except Exception:
                            if time.monotonic() > deadline:
                                raise
                            time.sleep(0.5)
                    for statement in constraints:
                        query(db, statement)
                    part = snapshot if key == "flat" else domain_snapshot(snapshot, key)
                    for batch in batches(part, ontology):
                        query(db, batch["query_template"], batch["parameters"])
                loaded = {}
                for key, db in names.items():
                    rows = query(
                        db,
                        "MATCH (r:ContextualRepresentation)-[:SUBJECT_OF]->(a:QualifiedAssertion)-[:SUPPORTED_BY]->(s:EvidenceSpan)-[:IN_DOCUMENT]->(d:SourceDocument)-[:PUBLISHED_BY]->(p:SourcePublisher) RETURN a.assertion_id AS assertion_id,a.canonical_payload_json AS payload,collect(s.span_id) AS spans,collect(p.publisher_id) AS publishers",
                        read=True,
                    )
                    loaded[key] = {}
                    for row in rows:
                        raw = json.loads(row["payload"])
                        if set(row["spans"]) != set(raw["evidence_span_ids"]) or set(
                            row["publishers"]
                        ) != {raw["source_publisher_id"]}:
                            raise ValueError("native provenance path mismatch")
                        loaded[key][row["assertion_id"]] = raw
                expected = {a.assertion_id: a.model_dump() for a in snapshot.assertions}
                multi = {
                    i: v
                    for domain, rows in loaded.items()
                    if domain != "flat"
                    for i, v in rows.items()
                }
                if loaded["flat"] != expected or multi != expected:
                    raise ValueError("single/multi canonical assertion mismatch")
                report.update(
                    status="PASS",
                    flat_assertions=len(loaded["flat"]),
                    multi_assertions=len(multi),
                    domain_counts={d: len(v) for d, v in loaded.items()},
                    qualified_assertion_parity=True,
                    source_publisher_span_paths_verified=True,
                    qualified_values_and_qualifiers_preserved=True,
                )
                save("readback.private.json", loaded)
            finally:
                cleanup = []
                for db in reversed(created):
                    try:
                        query("system", f"DROP DATABASE `{db}` DESTROY DATA")
                        cleanup.append(dict(database=db, success=True))
                    except Exception as exc:
                        cleanup.append(
                            dict(
                                database=db,
                                success=False,
                                error_type=type(exc).__name__,
                            )
                        )
                report["database_cleanup"] = cleanup
                driver.close()
    except Exception as exc:
        report.update(status="FAILED_WITH_RECEIPT", error_type=type(exc).__name__)
    finally:
        report["runtime"] = runtime
        if runtime and not runtime.get("teardown_success"):
            report["status"] = "CLEANUP_FAILED"
        report["graph_operations"] = len(operations)
        save("graph-operations.json", operations)
        schema = pa.schema(
            [
                (k, pa.string())
                for k in [
                    "experiment_run_id",
                    "trace_id",
                    "operation_id",
                    "graph_snapshot_id",
                    "database_name",
                    "query_template_id",
                    "query_parameters_hash",
                    "query_error",
                ]
            ]
            + [
                ("query_latency_ms", pa.float64()),
                ("records_returned", pa.int64()),
                ("read_only", pa.bool_()),
            ]
        )
        pq.write_table(
            pa.Table.from_pylist(operations, schema=schema),
            output / "graph-operations.parquet",
        )
        if pq.ParquetFile(output / "graph-operations.parquet").read().num_rows != len(
            operations
        ):
            raise ValueError("native graph trace Parquet mismatch")
        save("summary.json", report)
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ontology", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = run(a.ontology, a.output)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
