"""Owned immutable snapshots; generated text cannot choose DBs or Cypher."""
import json
import re
import time
import uuid

from neo4j import GraphDatabase, READ_ACCESS, Query
from ..models import digest


class MemoryExecutor:
    version = "ds01-scoped-read-v1"
    provider = "memory"
    def __init__(self, dataset, ledger):
        self.dataset, self.ledger = dataset, ledger
        self.docs = {d.document_id: {**d.model_dump(), "assertion_ids": sorted(
            a.assertion_id for a in dataset.assertions if a.document_id == d.document_id)} for d in dataset.documents}

    def read(self, corpus, ids, domains, *, flat, common):
        start = time.perf_counter()
        if not set(ids) <= set(corpus.document_ids) or len(ids) > 20:
            raise ValueError("candidate/limit scope escape")
        rows = [self.docs[r] for r in ids if self.docs[r]["domain_id"] in domains]
        self.ledger.emit("graph_query_traces", **common, provider_id="flat" if flat else domains[0],
            database_name="memory", graph_provider=self.provider, query_template_id=self.version,
            query_parameters_hash=digest([corpus.graph_snapshot_id, ids, domains]),
            query_latency_ms=(time.perf_counter() - start) * 1000, records_returned=len(rows),
            db_hits=None, plan=None, error=None)
        return rows


class BoltExecutor(MemoryExecutor):
    provider = "dozerdb"
    def __init__(self, dataset, ledger, environment):
        super().__init__(dataset, ledger)
        self.driver = GraphDatabase.driver(environment["DOZER_URI"],
            auth=(environment.get("DOZER_USER", "neo4j"), environment["DOZER_PASSWORD"]),
            max_transaction_retry_time=0, connection_timeout=5)
        suffix = {"flat": "flat", "finance_reporting": "finance", "legal_governance": "legal", "business_risk": "risk"}
        prefix = "ds01-" + uuid.uuid4().hex[:8]
        self.names = {k: prefix + "-" + v for k, v in suffix.items()}
        self.created = []
        self.capabilities = {"native_composite_used": False, "federation_mode": "application_level",
                             "server_read_acl": "ephemeral admin; READ sessions enforce driver intent, not server ACL"}

    def _owned(self, name):
        if name not in self.names.values() or not re.fullmatch(r"ds01-[a-f0-9]{8}-(flat|finance|legal|risk)", name):
            raise ValueError("database outside owned allowlist")
        return name

    def _quoted(self, name):
        return chr(96) + self._owned(name) + chr(96)

    def build(self, shifts):
        with self.driver.session(database="system") as session:
            info = session.run("CALL dbms.components() YIELD name, versions RETURN name, versions").data()
            self.capabilities["components"] = info
        for key, name in self.names.items():
            self._owned(name)
            with self.driver.session(database="system") as session:
                session.run(f"CREATE DATABASE {self._quoted(name)}").consume()
                self.created.append(name)
            deadline = time.monotonic() + 30
            while True:
                try:
                    with self.driver.session(database=name) as session:
                        session.run("RETURN 1").consume()
                    break
                except Exception:
                    if time.monotonic() > deadline:
                        raise RuntimeError("owned DB startup timeout")
                    time.sleep(.2)
            docs = [d for d in self.dataset.documents if key == "flat" or d.domain_id == key]
            ids = {d.document_id for d in docs}
            assertions = [a for a in self.dataset.assertions if a.document_id in ids]
            with self.driver.session(database=name) as session:
                for label, prop in (("DS01Document", "document_id"), ("DS01Assertion", "assertion_id"),
                                    ("DS01Entity", "entity_id"), ("DS01Snapshot", "snapshot_id")):
                    session.run(f"CREATE CONSTRAINT {label.lower()}_pk FOR (n:{label}) REQUIRE n.{prop} IS UNIQUE").consume()
                session.run("UNWIND $rows AS row CREATE (e:DS01Entity {entity_id:row.entity_id, payload:row.payload})",
                    rows=[{"entity_id": e.entity_id, "payload": json.dumps(e.model_dump(), sort_keys=True)}
                          for e in self.dataset.entities]).consume()
                for offset in range(0, len(docs), 100):
                    session.run("UNWIND $rows AS row CREATE (d:DS01Document {document_id:row.id, domain_id:row.domain, payload:row.payload})",
                        rows=[{"id": d.document_id, "domain": d.domain_id,
                               "payload": json.dumps(self.docs[d.document_id], sort_keys=True)}
                              for d in docs[offset:offset + 100]]).consume()
                for offset in range(0, len(assertions), 200):
                    session.run("UNWIND $rows AS row MATCH (d:DS01Document {document_id:row.document_id}), "
                        "(e:DS01Entity {entity_id:row.subject_id}) "
                        "CREATE (a:DS01Assertion {assertion_id:row.id, payload:row.payload}) "
                        "CREATE (a)-[:FROM_SOURCE]->(d) CREATE (a)-[:SUBJECT]->(e)",
                        rows=[{"id": a.assertion_id, "document_id": a.document_id, "subject_id": a.subject_id,
                               "payload": json.dumps(a.model_dump(), sort_keys=True)}
                              for a in assertions[offset:offset + 200]]).consume()
                unique = {s.corpus.corpus_id: s.corpus for s in shifts}
                for corpus in unique.values():
                    included = sorted(ids & set(corpus.document_ids))
                    session.run("CREATE (s:DS01Snapshot {snapshot_id:$sid, logical_hash:$hash})",
                                sid=corpus.graph_snapshot_id, hash=corpus.logical_payload_hash).consume()
                    session.run("MATCH (s:DS01Snapshot {snapshot_id:$sid}) UNWIND $ids AS id "
                        "MATCH (d:DS01Document {document_id:id}) CREATE (s)-[:INCLUDES]->(d)",
                        sid=corpus.graph_snapshot_id, ids=included).consume()
                session.run("CALL db.awaitIndexes(20)").consume()
                constraints = session.run("SHOW CONSTRAINTS YIELD name, type RETURN name, type").data()
                self.capabilities.setdefault("constraints", {})[name] = constraints
        self.capabilities.update(multi_database=True, database_selection=True, uniqueness_constraints=True,
                                 databases=list(self.created))
        return self.verify(shifts)

    def verify(self, shifts):
        receipts = []
        for corpus in {s.corpus.corpus_id: s.corpus for s in shifts}.values():
            physical = {}
            for key, name in self.names.items():
                with self.driver.session(database=name, default_access_mode=READ_ACCESS) as session:
                    docs = session.run("MATCH (s:DS01Snapshot {snapshot_id:$sid})-[:INCLUDES]->(d) RETURN d.payload AS p",
                                       sid=corpus.graph_snapshot_id).data()
                    facts = session.run("MATCH (s:DS01Snapshot {snapshot_id:$sid})-[:INCLUDES]->(d)<-[:FROM_SOURCE]-(a) RETURN a.payload AS p",
                                        sid=corpus.graph_snapshot_id).data()
                physical[key] = ({json.loads(r["p"])["document_id"]: json.loads(r["p"]) for r in docs},
                                 {json.loads(r["p"])["assertion_id"]: json.loads(r["p"]) for r in facts})
            for ids in (0, 1):
                union = {}
                for key in self.names:
                    if key != "flat":
                        union.update(physical[key][ids])
                if physical["flat"][ids] != union:
                    raise ValueError("single/multi logical payload parity failed")
            expected_docs = {rid: self.docs[rid] for rid in corpus.document_ids}
            expected_facts = {a.assertion_id: a.model_dump() for a in self.dataset.assertions
                              if a.assertion_id in corpus.assertion_ids}
            if physical["flat"] != (expected_docs, expected_facts):
                raise ValueError("DB readback differs from frozen source/assertion snapshot")
            receipts.append({"graph_snapshot_id": corpus.graph_snapshot_id, "corpus_id": corpus.corpus_id,
                "logical_payload_hash": corpus.logical_payload_hash, "source_documents": len(expected_docs),
                "qualified_assertions": len(expected_facts), "exact_readback_parity": True})
        self.capabilities["snapshot_parity"] = receipts
        return self.capabilities

    def read(self, corpus, ids, domains, *, flat, common):
        if (not set(ids) <= set(corpus.document_ids) or len(ids) > 20 or not domains
                or not set(domains) <= set(self.names) - {"flat"}):
            raise ValueError("scope/limit escape")
        if not flat and len(domains) != 1:
            raise ValueError("multi-domain read must fan out separately")
        name = self._owned(self.names["flat" if flat else domains[0]])
        parameters = {"sid": corpus.graph_snapshot_id, "ids": ids, "domains": domains, "limit": len(ids)}
        cypher = ("MATCH (s:DS01Snapshot {snapshot_id:$sid})-[:INCLUDES]->(d:DS01Document) "
                  "WHERE d.document_id IN $ids AND d.domain_id IN $domains RETURN d.payload AS payload LIMIT $limit")
        start = time.perf_counter()
        rows, error = [], None
        try:
            with self.driver.session(database=name, default_access_mode=READ_ACCESS) as session:
                rows = [json.loads(r["payload"]) for r in session.run(Query(cypher, timeout=10), **parameters)]
            expected = {r for r in ids if self.docs[r]["domain_id"] in domains}
            if {r["document_id"] for r in rows} != expected:
                raise ValueError("native read inventory incomplete")
            if any(r != self.docs[r["document_id"]] for r in rows):
                raise ValueError("native payload changed")
            return rows
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            self.ledger.emit("graph_query_traces", **common, provider_id="flat" if flat else domains[0],
                database_name=name, graph_provider=self.provider, query_template_id=self.version,
                query_parameters_hash=digest(parameters), query_latency_ms=(time.perf_counter() - start) * 1000,
                records_returned=len(rows), db_hits=None, plan=None, error=error)

    def profile(self, corpus):
        result = []
        for key, name in self.names.items():
            with self.driver.session(database=name, default_access_mode=READ_ACCESS) as session:
                query = session.run("PROFILE MATCH (s:DS01Snapshot {snapshot_id:$sid})-[:INCLUDES]->(d:DS01Document) RETURN count(d)",
                                    sid=corpus.graph_snapshot_id)
                query.data()
                summary = query.consume()
                plan = summary.profile
                result.append({"database": name, "provider_id": key, "graph_snapshot_id": corpus.graph_snapshot_id,
                    "plan": json.loads(json.dumps(plan, default=str)), "diagnostic_only": True})
        return result

    def close(self):
        errors = []
        for name in reversed(self.created):
            try:
                with self.driver.session(database="system") as session:
                    session.run(f"DROP DATABASE {self._quoted(name)} DESTROY DATA").consume()
            except Exception as exc:
                errors.append({"database": name, "error": type(exc).__name__})
        self.driver.close()
        return {"owned_databases_dropped": len(self.created) - len(errors), "errors": errors}
