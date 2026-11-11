"""Parameterized batches; caller owns isolated DB and capability/allowlist checks.
No connection, automatic DDL, or global name uniqueness is introduced here.
"""

import json
from ..models import digest


def batches(snapshot, ontology):
    snapshot.validate_ontology(ontology)
    namespace = snapshot.snapshot_id

    def key(kind, value):
        return digest([namespace, kind, value])

    rows = []

    def add(query, **params):
        rows.append(
            dict(query_template=query, parameters={**params, "snapshot_id": namespace})
        )

    for p in snapshot.publishers:
        add(
            "MERGE (n:SourcePublisher {id:$id}) SET n += $props",
            id=key("publisher", p.publisher_id),
            props=p.model_dump(),
        )
    for d in snapshot.documents:
        add(
            "MATCH (p:SourcePublisher {id:$publisher}) MERGE (d:SourceDocument {id:$id}) SET d += $props MERGE (d)-[:PUBLISHED_BY]->(p)",
            id=key("document", d.document_id),
            publisher=key("publisher", d.publisher_id),
            props=d.model_dump(),
        )
    for s in snapshot.spans:
        props = s.model_dump()
        props["table_locator"] = (
            json.dumps(props["table_locator"], sort_keys=True)
            if props["table_locator"]
            else None
        )
        add(
            "MATCH (d:SourceDocument {id:$document}) MERGE (s:EvidenceSpan {id:$id}) SET s += $props MERGE (s)-[:IN_DOCUMENT]->(d)",
            id=key("span", s.span_id),
            document=key("document", s.document_id),
            props=props,
        )
    for e in snapshot.entities:
        add(
            "MERGE (e:RealWorldEntity {id:$id}) SET e += $props",
            id=key("entity", e.entity_id),
            props=e.model_dump(),
        )
    for r in snapshot.representations:
        add(
            "MATCH (e:RealWorldEntity {id:$entity}) MERGE (r:ContextualRepresentation {id:$id}) SET r += $props MERGE (r)-[:REPRESENTS]->(e)",
            id=key("representation", r.representation_id),
            entity=key("entity", r.entity_id),
            props=r.model_dump(),
        )
    for a in snapshot.assertions:
        props = a.model_dump(exclude={"qualifiers", "evidence_span_ids"})
        props.update(a.qualifiers.model_dump())
        props["canonical_payload_json"] = json.dumps(a.model_dump(), sort_keys=True)
        add(
            "MATCH (r:ContextualRepresentation {id:$representation}) MERGE (a:QualifiedAssertion {id:$id}) SET a += $props MERGE (r)-[:SUBJECT_OF]->(a)",
            id=key("assertion", a.assertion_id),
            representation=key("representation", a.subject_representation_id),
            props=props,
        )
        for sid in a.evidence_span_ids:
            add(
                "MATCH (a:QualifiedAssertion {id:$assertion}), (s:EvidenceSpan {id:$span}) MERGE (a)-[:SUPPORTED_BY]->(s)",
                assertion=key("assertion", a.assertion_id),
                span=key("span", sid),
            )
    return rows
