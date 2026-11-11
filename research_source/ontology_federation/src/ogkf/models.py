from __future__ import annotations

import hashlib
import itertools
import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Source(Strict):
    document_id: str
    document_version: str
    company_id: str
    source_publisher: str
    source_uri: str
    section_id: str
    source_timestamp: str
    domain_id: Literal["finance", "legal", "sustainability"]
    text: str
    content_hash: str

    @model_validator(mode="after")
    def content_matches(self):
        if self.content_hash != hashlib.sha256(self.text.encode()).hexdigest():
            raise ValueError("source content hash mismatch")
        return self


class ExtractedFact(Strict):
    subject: str
    subject_type: str
    predicate: str
    literal_value: str
    domain_id: Literal["finance", "legal", "sustainability"]
    temporal_scope: str
    business_scope: str
    unit: str | None
    table_headers: list[str]
    evidence_start: int = Field(ge=0)
    evidence_end: int = Field(ge=1)
    evidence_span: str


class Extraction(Strict):
    facts: list[ExtractedFact]


class Config(Strict):
    version: str
    models: list[str] = Field(min_length=2)
    prompts: list[Literal["minimal", "schema_guided", "ontology_guided"]] = Field(min_length=1)
    ontologies: list[Literal["none", "lightweight", "rich"]] = Field(min_length=1)
    repetitions: int = Field(ge=1, le=10)
    max_logical_calls: int = Field(ge=1, le=100)
    max_attempts: int = Field(default=2, ge=1, le=4)
    max_output_tokens: int = Field(default=1024, ge=1, le=4096)
    timeout_seconds: float = Field(default=30, gt=0, le=60)
    min_interval_seconds: float = Field(default=1, ge=0)
    temperature: float = 0
    dataset: str
    human_calibration_receipt: str | None = None
    successful_smoke_summary: str | None = None
    span_repair_policy: Literal["none", "unique_exact_quote_v1"] = "none"

    @model_validator(mode="after")
    def no_duplicate_levels(self):
        for key in ("models", "prompts", "ontologies"):
            levels = getattr(self, key)
            if len(set(levels)) != len(levels):
                raise ValueError(f"duplicate {key} levels")
        return self

    def conditions(self):
        return [dict(model=m, prompt=p, ontology=o, repetition=r,
                     condition_id=digest([self.version, m, p, o, r])[:20])
                for m, p, o, r in itertools.product(self.models, self.prompts, self.ontologies,
                                                   range(self.repetitions))]
