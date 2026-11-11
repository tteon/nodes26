"""Compact provider registry and an ontology revision ledger; no QA gold here."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .identity import Domain
from .models import Strict, digest


class ProviderMetadata(Strict):
    provider_id: str
    domain_id: Domain
    database_name: str
    ontology_version: str
    supported_entity_types: list[str]
    supported_relation_types: list[str]
    supported_question_categories: list[str]
    available_time_range: tuple[str, str]
    source_document_types: list[str]
    provenance_policy: str
    schema_summary: str
    database_capabilities: dict


class Registry(Strict):
    version: str
    entries: list[ProviderMetadata]

    @model_validator(mode="after")
    def unique_routes(self):
        if len({e.domain_id for e in self.entries}) != len(self.entries):
            raise ValueError("one explicit route per domain required in v1")
        if len({e.database_name for e in self.entries}) != len(self.entries):
            raise ValueError("isolated registry database names must be unique")
        return self

    def compact(self, reduced=False):
        fields = {"provider_id", "domain_id", "supported_entity_types", "supported_relation_types",
                  "supported_question_categories", "available_time_range", "provenance_policy", "ontology_version"}
        if reduced:
            fields = {"provider_id", "domain_id", "supported_question_categories"}
        return [{k: v for k, v in e.model_dump().items() if k in fields} for e in self.entries]

    @property
    def registry_hash(self):
        return digest(self.model_dump())


class PublicQuestion(Strict):
    question_id: str
    text: str
    canonical_identifier: str
    temporal_scope: str


class QueryPlan(Strict):
    question_type: str
    selected_domains: list[Domain] = Field(max_length=3)
    required_evidence: list[str]
    execution_strategy: Literal["parallel_read", "bounded_sequential"]
    max_tool_calls: int = Field(ge=1, le=3)
    rationale: str

    @model_validator(mode="after")
    def unique_domains(self):
        if len(set(self.selected_domains)) != len(self.selected_domains) or len(set(self.required_evidence)) != len(self.required_evidence):
            raise ValueError("duplicate planned domain/evidence")
        return self


class OntologyRevision(Strict):
    ontology_version: str
    parent_version: str | None
    structural: dict
    semantic: dict
    content_hash: str

    @model_validator(mode="after")
    def verify_hash(self):
        if self.content_hash != digest([self.structural, self.semantic]):
            raise ValueError("ontology content hash mismatch")
        return self
