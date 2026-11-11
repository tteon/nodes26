from __future__ import annotations

from typing import Literal
from pydantic import Field, model_validator
from ..models import Strict, digest
import hashlib
import re
from collections import Counter, defaultdict

VERSION = "ds01-v1"
DOMAINS = ("finance_reporting", "legal_governance", "business_risk")


def tokens(text):
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.casefold())


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


class Query(Strict):
    query_id: str
    text: str


class GoldEvidence(Strict):
    query_id: str
    answer: str
    # AND across groups; OR across alternative evidence sets within each group.
    required_groups: list[list[list[str]]]
    sufficient_set_status: Literal["provided_reference_proxy", "independently_reviewed"]
    relevance: dict[str, float] = Field(default_factory=dict)
    irrelevant_evidence_ids: list[str] = Field(default_factory=list)
    compatible_evidence_ids: list[str] = Field(default_factory=list)
    incompatible_evidence_ids: list[str] = Field(default_factory=list)
    company_id: str | None = None
    category: str | None = None
    review_record: str | None = None

    @model_validator(mode="after")
    def meaningful_groups(self):
        if not self.required_groups or any(not g or any(not alt for alt in g) for g in self.required_groups):
            raise ValueError("empty support group/alternative")
        if self.sufficient_set_status == "independently_reviewed" and not self.review_record:
            raise ValueError("independent sufficient evidence needs review record")
        if set(self.compatible_evidence_ids) & set(self.incompatible_evidence_ids):
            raise ValueError("contradictory context labels")
        return self

    def required_ids(self):
        return {e for group in self.required_groups for alternative in group for e in alternative}


class Entity(Strict):
    entity_id: str
    canonical_name: str
    aliases: list[str]
    identity_status: Literal["source_bound_issuer", "reviewed", "lexical_only"]


class Document(Strict):
    document_id: str
    document_version: str
    source_publisher: str | None
    source_uri: str | None
    text: str
    content_hash: str
    domain_id: str
    evidence_id: str
    entity_mentions: dict[str, int]
    relation_counts: dict[str, int]
    token_count: int
    temporal_scope: str | None = None
    business_scope: str | None = None
    metadata_status: str

    @model_validator(mode="after")
    def verify(self):
        if self.content_hash != text_hash(self.text) or self.token_count != len(tokens(self.text)):
            raise ValueError("document text/token hash mismatch")
        if self.token_count < 1 or any(v < 0 for v in self.entity_mentions.values()):
            raise ValueError("invalid source counts")
        if self.evidence_id != digest([self.document_id, self.document_version, 0, len(self.text)]):
            raise ValueError("unstable whole-passage evidence ID")
        return self


class Assertion(Strict):
    assertion_id: str
    subject_id: str
    predicate: str
    object_value: str
    document_id: str
    evidence_id: str
    source_hash: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str
    assertion_kind: Literal["source_metadata", "source_assertion"]
    validation_status: Literal["mechanical_source", "independently_reviewed"]
    domain_id: str
    temporal_scope: str | None
    business_scope: str | None
    unit: str | None
    review_record: str | None = None


class DistractorLabel(Strict):
    query_id: str
    document_id: str
    kind: Literal["lexical", "relational", "temporal", "scope", "identity", "unclassified"]
    status: Literal["candidate_not_gold", "independently_reviewed_nonanswer"]
    review_record: str | None = None

    @model_validator(mode="after")
    def reviewed(self):
        if self.status == "independently_reviewed_nonanswer" and not self.review_record:
            raise ValueError("nonanswer assertion needs independent review record")
        return self


class Dataset(Strict):
    version: str
    queries: list[Query]
    gold: list[GoldEvidence]
    entities: list[Entity]
    documents: list[Document]
    assertions: list[Assertion]
    distractors: list[DistractorLabel]
    ontology_version: str
    ontology_relations: list[str]
    ontology_kind: str
    input_hashes: dict[str, str]
    annotation_status: str

    @model_validator(mode="after")
    def integrity(self):
        for rows, key in ((self.queries, "query_id"), (self.gold, "query_id"),
                          (self.entities, "entity_id"), (self.documents, "document_id"),
                          (self.assertions, "assertion_id")):
            ids = [getattr(r, key) for r in rows]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate " + key)
        docs = {d.document_id: d for d in self.documents}
        evidence = {d.evidence_id for d in self.documents}
        entities = {e.entity_id for e in self.entities}
        if {q.query_id for q in self.queries} != {g.query_id for g in self.gold}:
            raise ValueError("query/gold mismatch")
        if len(evidence) != len(docs):
            raise ValueError("duplicate evidence IDs")
        for g in self.gold:
            if not g.required_ids() <= evidence:
                raise ValueError("gold references missing from corpus")
        for a in self.assertions:
            d = docs[a.document_id]
            if (a.subject_id not in entities or a.evidence_id != d.evidence_id or a.source_hash != d.content_hash
                    or not 0 <= a.start < a.end <= len(d.text) or d.text[a.start:a.end] != a.quote
                    or a.predicate not in self.ontology_relations):
                raise ValueError("assertion/source/ontology mismatch")
            if a.validation_status == "independently_reviewed" and not a.review_record:
                raise ValueError("independent assertion lacks review record")
        counted_relations = defaultdict(Counter)
        for a in self.assertions:
            counted_relations[a.document_id][a.predicate] += 1
        for d in self.documents:
            if Counter(d.relation_counts) != counted_relations[d.document_id]:
                raise ValueError("document relation counts do not match actual assertions")
        for label in self.distractors:
            if label.query_id not in {q.query_id for q in self.queries} or label.document_id not in docs:
                raise ValueError("unknown distractor label ID")
            g = next(g for g in self.gold if g.query_id == label.query_id)
            if docs[label.document_id].evidence_id in g.required_ids():
                raise ValueError("gold evidence labeled distractor")
        return self

    @property
    def dataset_hash(self):
        return digest(self.model_dump())


class CorpusManifest(Strict):
    version: str = VERSION
    corpus_id: str
    graph_snapshot_id: str
    dataset_hash: str
    document_ids: list[str]
    assertion_ids: list[str]
    logical_payload_hash: str
    ontology_version: str
    document_count: int
    token_count: int
    gold_evidence_ids: list[str]


class ShiftManifest(Strict):
    condition_id: str
    intervention: Literal["DS-01A", "DS-01B", "DS-01C"]
    severity: float = Field(ge=0, le=1)
    seed: int
    reference_corpus_id: str
    corpus: CorpusManifest
    target: dict
    observed: dict
    covariate_changes: dict
    structural_valid: bool
    causal_isolation: Literal["matched_measured_covariates", "imperfect"]
    research_qualified: bool
    limitations: list[str]


class Provider(Strict):
    provider_id: str
    domain_id: str
    supported_entity_types: list[str]
    supported_relations: list[str]
    supported_question_concepts: list[str]
    time_range: list[str | None]
    provenance_policy: str
    ontology_version: str
    schema_summary: str


class RoutePlan(Strict):
    selected_providers: list[str] = Field(max_length=3)
    required_concepts: list[str]
    entity_mentions: list[str]
    periods: list[str]
    scope_constraints: list[str]
    required_operations: list[str]
    uncertainties: list[str]
    max_tool_calls: int = Field(ge=1, le=3)
    rationale: str


class NativeClaim(Strict):
    claim_id: str
    text: str
    evidence_ids: list[str]
    formula: str | None
    operand_evidence_ids: list[str]


class NativeAnswer(Strict):
    status: Literal["answered", "insufficient", "conflicting"]
    text: str
    claims: list[NativeClaim]


class JudgeResult(Strict):
    correctness: Literal[0.0, 0.5, 1.0]
    faithfulness: float | None = Field(ge=0, le=1)
    context_leakage: bool | None
    attribution_accuracy: float | None = Field(ge=0, le=1)
    rationale: str
    reference_defect: bool


class PilotConfig(Strict):
    version: str = VERSION
    interventions: list[Literal["DS-01A", "DS-01B", "DS-01C"]]
    severities: list[float]
    seeds: list[int]
    document_budget: int = Field(ge=4, le=256)
    global_top_k: int = Field(ge=1, le=20)
    target_entity_id: str
    target_relation: str
    target_domain: str
    strategies: list[str]
    context_byte_budget: int = Field(ge=1000, le=48000)
    answer_model: str
    router_model: str
    judge_model: str
    max_logical_calls: int = Field(ge=1, le=200)
    max_attempts: int = Field(ge=1, le=2)
    live_end_to_end_limit: int = Field(ge=1, le=40)
    distractor_kinds: list[Literal["lexical", "relational", "temporal", "scope", "identity"]] = Field(
        default_factory=lambda: ["lexical", "relational"])
    require_reviewed_distractors: bool = True

    @model_validator(mode="after")
    def bounded_matrix(self):
        if (len(set(self.severities)) < 4 or 0 not in self.severities
                or len(set(self.seeds)) < 2 or any(not 0 <= s <= 1 for s in self.severities)):
            raise ValueError("baseline + at least three severities, multiple seeds required")
        if not set(self.strategies) <= {"B0", "B1", "B2", "B3", "B4", "B5", "B3_reduced"}:
            raise ValueError("unknown strategy")
        return self
