"""Frozen qualified assertions and explicit, separately measured identity policies."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .models import Source, Strict, digest

Domain = Literal["finance", "legal", "sustainability"]


class ContextEntity(Strict):
    entity_id: str
    canonical_identifier: str
    canonical_name: str
    entity_type: str
    domain_id: Domain
    provider_id: str
    business_scope: str
    identity_scope: Literal["domain_local"] = "domain_local"


class IdentityMapping(Strict):
    left_entity_id: str
    right_entity_id: str
    mapping_type: Literal["SAME_AS", "RELATED_TO", "DIFFERENT_FROM", "UNRESOLVED"]
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(min_length=1)
    review_status: Literal["reviewed", "pending"]


class Snapshot(Strict):
    version: str
    origin: Literal["synthetic_fixture", "source_extraction"]
    sources: list[Source]
    entities: list[ContextEntity]
    assertions: list[dict]
    mappings: list[IdentityMapping]

    @model_validator(mode="after")
    def verify_lineage(self):
        entities = {e.entity_id: e for e in self.entities}
        sources = {(s.document_id, s.document_version): s for s in self.sources}
        if len(entities) != len(self.entities) or len(sources) != len(self.sources):
            raise ValueError("duplicate entity/source IDs")
        seen, evidence_ids = set(), set()
        for fact in self.assertions:
            required = {"fact_id", "subject_id", "predicate", "literal_value", "domain_id", "temporal_scope",
                        "business_scope", "unit", "source_document_id", "source_version", "source_publisher",
                        "source_content_hash", "evidence_id", "evidence_start", "evidence_end", "evidence_span"}
            if not required <= fact.keys() or fact["fact_id"] in seen:
                raise ValueError("incomplete/duplicate frozen assertion")
            entity = entities[fact["subject_id"]]
            source = sources[fact["source_document_id"], fact["source_version"]]
            if (entity.domain_id != fact["domain_id"] or source.domain_id != fact["domain_id"] or
                    entity.business_scope != fact["business_scope"] or source.source_publisher != fact["source_publisher"] or
                    source.content_hash != fact["source_content_hash"]):
                raise ValueError("assertion context/source mismatch")
            start, end = fact["evidence_start"], fact["evidence_end"]
            if not 0 <= start < end <= len(source.text) or source.text[start:end] != fact["evidence_span"]:
                raise ValueError("invalid frozen source span")
            if fact["evidence_id"] != digest([source.document_id, source.document_version, source.content_hash, start, end]):
                raise ValueError("unstable evidence ID")
            seen.add(fact["fact_id"])
            evidence_ids.add(fact["evidence_id"])
        for mapping in self.mappings:
            if (mapping.left_entity_id not in entities or mapping.right_entity_id not in entities or
                    mapping.left_entity_id == mapping.right_entity_id or not set(mapping.evidence_ids) <= evidence_ids):
                raise ValueError("invalid cross-context mapping lineage")
        return self

    @property
    def snapshot_hash(self):
        return digest(self.model_dump())


class GraphVariant(Strict):
    version: str = "identity-policy-v1"
    policy: Literal["V0", "V0q", "V1", "V2"]
    snapshot_hash: str
    assertion_hash: str
    entities: dict[str, str]
    placement: dict[str, list[str]]
    assertions: list[dict]
    projections: list[dict]
    mappings: list[IdentityMapping]


def assertion_hash(assertions):
    return digest(sorted(assertions, key=lambda f: f["fact_id"]))


def build_variant(snapshot: Snapshot, policy: str) -> GraphVariant:
    if policy not in {"V0", "V0q", "V1", "V2"}:
        raise ValueError("unknown identity policy")
    assignment = {e.entity_id: digest(["global", e.canonical_identifier]) if policy == "V0" else e.entity_id
                  for e in snapshot.entities}
    placement = {d: [] for d in (["flat"] if policy in {"V0", "V0q"} else ["finance", "legal", "sustainability"])}
    projections = []
    for fact in snapshot.assertions:
        placement["flat" if policy in {"V0", "V0q"} else fact["domain_id"]].append(fact["fact_id"])
        projection = {**fact, "representation_id": assignment[fact["subject_id"]]}
        if policy == "V0":
            # Explicit lossy retrieval view; original qualified assertions remain immutable.
            projection["domain_id"], projection["business_scope"] = None, None
        projections.append(projection)
    return GraphVariant(policy=policy, snapshot_hash=snapshot.snapshot_hash,
                        assertion_hash=assertion_hash(snapshot.assertions), entities=assignment,
                        placement=placement, assertions=snapshot.assertions, projections=projections,
                        mappings=snapshot.mappings if policy == "V2" else [])


def verify_variants(snapshot, variants):
    expected = assertion_hash(snapshot.assertions)
    for variant in variants:
        if variant.snapshot_hash != snapshot.snapshot_hash or variant.assertion_hash != expected or assertion_hash(variant.assertions) != expected:
            raise ValueError("frozen assertion parity violated")
        ids = [i for values in variant.placement.values() for i in values]
        if sorted(ids) != sorted(f["fact_id"] for f in snapshot.assertions):
            raise ValueError("placement lost or duplicated facts")
        if variant.policy != "V0":
            qualified = [{k: v for k, v in f.items() if k != "representation_id"} for f in variant.projections]
            if assertion_hash(qualified) != expected:
                raise ValueError("qualified retrieval projection differs")
    return {"qualified_assertion_parity": True, "assertion_hash": expected,
            "lossy_view": "V0 domain/business scope erased explicitly; raw qualified assertions retained"}
