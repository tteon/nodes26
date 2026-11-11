"""Allowlisted fixed-template graph access; no generated Cypher interface."""
from __future__ import annotations

import json
import re
import time
import uuid

from neo4j import Query, READ_ACCESS

from .identity import GraphVariant, Snapshot
from .models import digest

READ = """MATCH (f:OGKFFact {snapshot:$snapshot})
WHERE f.canonical_identifier IN $identifiers
AND (size($predicates)=0 OR f.predicate IN $predicates)
AND ($temporal='' OR f.temporal_scope=$temporal)
RETURN f.payload AS payload ORDER BY f.fact_id LIMIT $limit"""


class QueryTrace:
    def __init__(self, recorder=None):
        self.recorder = recorder
        self.rows = []

    def append(self, **row):
        row = {"operation_id": str(uuid.uuid4()), "execution_plan_metrics": None, **row}
        self.rows.append(row)
        if self.recorder:
            self.recorder.write("graph/" + row["operation_id"] + ".json", row)


class MemoryStore:
    def __init__(self, snapshot: Snapshot, variant: GraphVariant, databases: dict, telemetry=None):
        self.snapshot, self.variant = snapshot, variant
        self.databases, self.telemetry = dict(databases), telemetry or QueryTrace()
        if set(databases) != set(variant.placement) or len(set(databases.values())) != len(databases):
            raise ValueError("database allowlist must exactly match variant placements")
        self.reverse = {v: k for k, v in databases.items()}
        self.by_entity = {e.entity_id: e for e in snapshot.entities}
        self.by_fact = {f["fact_id"]:f for f in variant.projections}
        self.provider = "memory_fixture"

    def identifiers(self, canonical_identifier):
        identifiers = {canonical_identifier}
        # Reviewed SAME_AS expands identifiers without merging assertions/scopes.
        if self.variant.policy == "V2":
            changed = True
            while changed:
                changed = False
                for mapping in self.variant.mappings:
                    if mapping.review_status != "reviewed" or mapping.mapping_type != "SAME_AS":
                        continue
                    left = self.by_entity[mapping.left_entity_id].canonical_identifier
                    right = self.by_entity[mapping.right_entity_id].canonical_identifier
                    if left in identifiers or right in identifiers:
                        before = len(identifiers)
                        identifiers.update([left, right])
                        changed |= len(identifiers) != before
        return sorted(identifiers)

    def parameters(self, database, identifier, predicates, temporal, limit):
        if database not in self.reverse:
            raise ValueError("database is outside explicit allowlist")
        if not isinstance(limit, int) or not 1 <= limit <= 100:
            raise ValueError("row limit must be 1..100")
        if len(predicates) > 32 or any(not isinstance(p, str) for p in predicates):
            raise ValueError("invalid predicate parameters")
        return {"snapshot": self.variant.snapshot_hash, "identifiers": self.identifiers(identifier),
                "predicates": list(predicates), "temporal": temporal, "limit": limit}

    def read(self, database, identifier, predicates, temporal, limit, trace_id):
        params = self.parameters(database, identifier, predicates, temporal, limit)
        start = time.perf_counter()
        placed = set(self.variant.placement[self.reverse[database]])
        rows = [f for f in self.variant.projections if f["fact_id"] in placed
                and self.by_entity[f["subject_id"]].canonical_identifier in params["identifiers"]
                and (not predicates or f["predicate"] in predicates)
                and (not temporal or f["temporal_scope"] == temporal)]
        rows = sorted(rows, key=lambda f: f["fact_id"])[:limit]
        self.telemetry.append(trace_id=trace_id, database_name=database,
            selected_domain=self.reverse[database], query_template_id="qualified_fact_read_v1",
            query_parameters_hash=digest(params), query_latency_ms=(time.perf_counter()-start)*1000,
            records_returned=len(rows), query_error=None, graph_provider=self.provider)
        return [dict(row) for row in rows]


class BoltStore(MemoryStore):
    """Driver sessions default to READ; v1 accepts only the constant READ template.

    Prefer a separately provisioned least-privilege read account. READ mode alone
    is not server authorization; admin fallback is disclosed by the live runner.
    """
    def __init__(self, snapshot, variant, databases, driver, telemetry=None):
        super().__init__(snapshot, variant, databases, telemetry)
        self.driver, self.provider = driver, "dozerdb"

    def read(self, database, identifier, predicates, temporal, limit, trace_id):
        params = self.parameters(database, identifier, predicates, temporal, limit)
        start, rows, error = time.perf_counter(), [], None
        try:
            with self.driver.session(database=database, default_access_mode=READ_ACCESS) as session:
                result = session.run(Query(READ, timeout=5), params)
                rows = [json.loads(row["payload"]) for row in result]
                result.consume()
            placed=set(self.variant.placement[self.reverse[database]])
            if any(row.get("fact_id") not in placed or row!=self.by_fact.get(row.get("fact_id")) for row in rows):
                raise ValueError("database evidence differs from frozen assertion payload")
            return rows
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            self.telemetry.append(trace_id=trace_id, database_name=database,
                selected_domain=self.reverse[database], query_template_id="qualified_fact_read_v1",
                query_parameters_hash=digest(params), query_latency_ms=(time.perf_counter()-start)*1000,
                records_returned=len(rows), query_error=error, graph_provider=self.provider)

    def diagnostic_profile(self,database,identifier,predicates,temporal,limit):
        params=self.parameters(database,identifier,predicates,temporal,limit)
        with self.driver.session(database=database,default_access_mode=READ_ACCESS) as session:
            result=session.run(Query("PROFILE "+READ,timeout=5),params)
            result.data()
            summary=result.consume()
        return {"database_name":database,"query_template_id":"qualified_fact_read_v1",
                "query_parameters_hash":digest(params),"execution_plan_metrics":summary.profile,
                "scope":"separate read-only diagnostic, excluded from episode timing/call counts"}


class GraphDeployment:
    """Own only freshly created named DBs; cleanup never uses an arbitrary name."""
    def __init__(self, driver):
        self.driver, self.created = driver, []
        self.prefix = "ogkf" + uuid.uuid4().hex[:10]
        self.cleanup = []

    def create_load(self, snapshot, variant, *, reuse=None):
        if reuse:
            if variant.policy != "V2" or set(reuse) != set(variant.placement) or any(db not in self.created for db in reuse.values()):
                raise ValueError("only qualified V1 placements may be reused by V2")
            rows=[{"mapping_id":digest(m.model_dump()),"payload":json.dumps(m.model_dump())} for m in variant.mappings]
            for db in reuse.values():
                with self.driver.session(database=db) as session:
                    session.run(Query("UNWIND $rows AS row MERGE(m:OGKFIdentityMapping {mapping_id:row.mapping_id}) SET m.payload=row.payload",timeout=10),{"rows":rows}).consume()
            return dict(reuse)
        databases = {}
        by_entity = {e.entity_id: e for e in snapshot.entities}
        by_source={(s.document_id,s.document_version):s for s in snapshot.sources}
        for domain, ids in variant.placement.items():
            db = self.prefix + variant.policy.lower() + domain
            if not re.fullmatch(r"ogkf[a-f0-9]{10}v(0q?|1|2)(flat|finance|legal|sustainability)", db):
                raise ValueError("invalid experiment DB name")
            with self.driver.session(database="system") as session:
                session.run(Query(f"CREATE DATABASE `{db}`", timeout=15)).consume()
            self.created.append(db)
            databases[domain] = db
            with self.driver.session(database=db) as session:
                for label, key in (("OGKFFact", "fact_id"), ("OGKFEntity", "entity_id"), ("OGKFSource", "document_key"), ("OGKFEvidence", "evidence_id")):
                    session.run(Query(f"CREATE CONSTRAINT {label.lower()}_key FOR (n:{label}) REQUIRE n.{key} IS UNIQUE", timeout=10)).consume()
                session.run(Query("CREATE INDEX ogkf_fact_scope FOR (f:OGKFFact) ON (f.snapshot,f.canonical_identifier,f.temporal_scope)",timeout=10)).consume()
                payloads = []
                for fact in variant.projections:
                    if fact["fact_id"] not in ids:
                        continue
                    entity = by_entity[fact["subject_id"]]
                    payloads.append({"fact_id": fact["fact_id"], "snapshot": variant.snapshot_hash,
                        "predicate": fact["predicate"], "temporal_scope": fact["temporal_scope"],
                        "canonical_identifier": entity.canonical_identifier, "payload": json.dumps(fact),
                        "entity_id": variant.entities[fact["subject_id"]], "entity_payload": json.dumps(entity.model_dump()),
                        "document_key": digest([fact["source_document_id"], fact["source_version"]]),
                        "source_payload":json.dumps(by_source[fact["source_document_id"],fact["source_version"]].model_dump()),
                        "evidence_id": fact["evidence_id"], "quote": fact["evidence_span"]})
                session.run(Query("""UNWIND $rows AS row
MERGE (e:OGKFEntity {entity_id:row.entity_id})
SET e.canonical_identifier=row.canonical_identifier
MERGE (s:OGKFSource {document_key:row.document_key}) SET s.payload=row.source_payload
MERGE (v:OGKFEvidence {evidence_id:row.evidence_id}) SET v.quote=row.quote
MERGE (f:OGKFFact {fact_id:row.fact_id}) SET f.snapshot=row.snapshot, f.predicate=row.predicate,
 f.temporal_scope=row.temporal_scope, f.canonical_identifier=row.canonical_identifier,f.payload=row.payload
MERGE (f)-[:SUBJECT]->(e) MERGE (f)-[:SUPPORTED_BY]->(v) MERGE (v)-[:FROM_SOURCE]->(s)""", timeout=15), {"rows": payloads}).consume()
                rows = session.run(Query("MATCH(f:OGKFFact) RETURN f.payload AS payload ORDER BY f.fact_id", timeout=10)).data()
                actual = [json.loads(r["payload"]) for r in rows]
                expected = sorted([f for f in variant.projections if f["fact_id"] in ids], key=lambda f: f["fact_id"])
                if actual != expected:
                    raise ValueError("graph load exact projection parity failed")
                mapping_rows=[{"mapping_id":digest(m.model_dump()),"payload":json.dumps(m.model_dump())}
                              for m in variant.mappings]
                if mapping_rows:
                    session.run(Query("UNWIND $rows AS row MERGE(m:OGKFIdentityMapping {mapping_id:row.mapping_id}) SET m.payload=row.payload",timeout=10),{"rows":mapping_rows}).consume()
        return databases

    def close(self):
        for db in reversed(self.created):
            try:
                with self.driver.session(database="system") as session:
                    session.run(Query(f"DROP DATABASE `{db}` DESTROY DATA", timeout=15)).consume()
                self.cleanup.append({"database": db, "success": True})
            except Exception as exc:
                self.cleanup.append({"database": db, "success": False, "error_type": type(exc).__name__})
