from __future__ import annotations
from typing import Literal
from pydantic import Field, StrictBool, model_validator
from ..models import Strict, digest
from ..ds01.contracts import text_hash

Action = Literal[
    "ANSWER_WITH_CITATIONS",
    "ASK_CLARIFICATION",
    "COMPARE_SCOPED_ANSWERS",
    "ABSTAIN_INSUFFICIENT_EVIDENCE",
]
DIMENSIONS = (
    "identity",
    "period",
    "business_scope",
    "unit",
    "definition",
    "modality",
    "composition",
    "jurisdiction",
)


class Qualifiers(Strict):
    period: str | None
    reporting_basis: str | None
    scope: str | None
    unit: str | None
    jurisdiction: str | None
    modality: Literal[
        "reported", "estimated", "planned", "realized", "legally_required", "unknown"
    ]
    valid_time: str | None
    assertion_kind: Literal["source_assertion", "derived_inference"]
    ontology_version: str


class SourcePublisher(Strict):
    publisher_id: str
    name: str


class Document(Strict):
    document_id: str
    document_version: str
    publisher_id: str
    source_uri: str | None
    text: str
    content_hash: str
    source_kind: Literal["finance", "sustainability", "legal", "general", "synthetic"]
    synthetic: StrictBool

    @model_validator(mode="after")
    def hashes(self):
        if self.content_hash != text_hash(self.text):
            raise ValueError("document hash mismatch")
        return self


class EvidenceSpan(Strict):
    span_id: str
    document_id: str
    document_version: str
    source_hash: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str
    table_locator: dict[str, str] | None = None

    def verify(self, doc):
        if (
            self.document_version != doc.document_version
            or self.source_hash != doc.content_hash
            or not 0 <= self.start < self.end <= len(doc.text)
            or doc.text[self.start : self.end] != self.quote
            or self.span_id
            != digest([self.document_id, self.document_version, self.start, self.end])
        ):
            raise ValueError("exact evidence locator mismatch")


class RealWorldEntity(Strict):
    entity_id: str
    canonical_name: str
    canonical_identifier: str | None
    identity_status: Literal["synthetic_fixture", "independently_reviewed", "unknown"]


class ContextualRepresentation(Strict):
    representation_id: str
    entity_id: str
    domain_id: str
    business_role: str
    identity_scope: Literal["domain_local"] = "domain_local"


class QualifiedAssertion(Strict):
    assertion_id: str
    subject_representation_id: str
    predicate: str
    object_type: str
    object_value: str
    qualifiers: Qualifiers
    evidence_span_ids: list[str] = Field(min_length=1)
    source_publisher_id: str
    source_provider_id: str
    extraction_run_id: str
    support_status: Literal["proposal", "synthetic_fixture", "independently_reviewed"]
    derivation_rule: str | None = None


class RelationDefinition(Strict):
    relation_id: str
    definition: str
    domain: str
    range: list[str] = Field(min_length=1)
    permitted_qualifiers: list[str]
    required_qualifiers: list[str]
    allowed_modalities: list[str]
    non_implications: list[str]
    inference_limits: list[str] = Field(min_length=1)
    temporal_rule: str
    provider_mappings: list[dict[str, str]]
    required_source_kind: str | None = None


class Ontology(Strict):
    version: str
    classes: list[str]
    relations: list[RelationDefinition]
    compatibility_policy: str

    @model_validator(mode="after")
    def stable_vocabulary(self):
        ids = [r.relation_id for r in self.relations]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate relation ID")
        for r in self.relations:
            if (
                r.domain not in self.classes
                or not set(r.range) <= set(self.classes)
                or not set(r.required_qualifiers) <= set(r.permitted_qualifiers)
                or not set(r.permitted_qualifiers) <= set(Qualifiers.model_fields)
                or not set(r.non_implications) <= set(ids)
            ):
                raise ValueError(
                    "invalid relation domain/range/qualifier/non-implication contract"
                )
        return self


class Snapshot(Strict):
    version: str
    publishers: list[SourcePublisher]
    documents: list[Document]
    spans: list[EvidenceSpan]
    entities: list[RealWorldEntity]
    representations: list[ContextualRepresentation]
    assertions: list[QualifiedAssertion]
    synthetic: StrictBool

    @model_validator(mode="after")
    def integrity(self):
        collections = [
            (self.publishers, "publisher_id"),
            (self.documents, "document_id"),
            (self.spans, "span_id"),
            (self.entities, "entity_id"),
            (self.representations, "representation_id"),
            (self.assertions, "assertion_id"),
        ]
        for rows, key in collections:
            ids = [getattr(r, key) for r in rows]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate snapshot identifier")
        pubs = {p.publisher_id for p in self.publishers}
        docs = {d.document_id: d for d in self.documents}
        spans = {s.span_id: s for s in self.spans}
        entities = {e.entity_id for e in self.entities}
        reps = {r.representation_id for r in self.representations}
        if any(d.publisher_id not in pubs for d in self.documents):
            raise ValueError("unknown publisher")
        if any(d.synthetic != self.synthetic for d in self.documents):
            raise ValueError("mixed synthetic provenance")
        for s in self.spans:
            if s.document_id not in docs:
                raise ValueError("unknown span document")
            s.verify(docs[s.document_id])
        if any(r.entity_id not in entities for r in self.representations):
            raise ValueError("unknown real entity")
        for a in self.assertions:
            if a.subject_representation_id not in reps or not set(
                a.evidence_span_ids
            ) <= set(spans):
                raise ValueError("unknown assertion subject/evidence")
            if any(
                docs[spans[s].document_id].publisher_id != a.source_publisher_id
                for s in a.evidence_span_ids
            ):
                raise ValueError("assertion/source publisher mismatch")
            if self.synthetic and a.support_status == "independently_reviewed":
                raise ValueError(
                    "synthetic assertions cannot become empirical reviewed gold"
                )
        return self

    @property
    def snapshot_id(self):
        return digest(self.model_dump())

    def validate_ontology(self, ontology):
        rels = {r.relation_id: r for r in ontology.relations}
        for a in self.assertions:
            r = rels.get(a.predicate)
            if (
                r is None
                or a.object_type not in r.range
                or a.qualifiers.ontology_version != ontology.version
            ):
                raise ValueError("assertion ontology membership mismatch")
            q = a.qualifiers.model_dump()
            if a.qualifiers.modality not in r.allowed_modalities or any(
                q[k] is None for k in r.required_qualifiers
            ):
                raise ValueError("relation modality/required qualifier violation")
            if (
                a.qualifiers.assertion_kind == "derived_inference"
                and not a.derivation_rule
            ):
                raise ValueError("derived inference requires explicit rule")
            if r.required_source_kind:
                by_span = {s.span_id: s for s in self.spans}
                by_doc = {d.document_id: d for d in self.documents}
                if not any(
                    by_doc[by_span[s].document_id].source_kind == r.required_source_kind
                    for s in a.evidence_span_ids
                ):
                    raise ValueError(
                        "required source-kind witness absent;ontology conformance alone is not entailment"
                    )
        return True


class ExtractionResult(Strict):
    assertions: list[QualifiedAssertion]


class InteractionCase(Strict):
    case_id: str
    parent_case_id: str | None
    protocol_id: str
    question: str
    synthetic: StrictBool
    context_assertion_ids: list[str]
    allowed_actions: list[Action] = Field(min_length=1)
    material_missing_dimensions: list[str]
    required_predicates: list[str]
    sufficient_span_sets: list[list[str]]
    rationale: str
    counterfactual_kind: Literal[
        "baseline", "irrelevant_addition", "relevant_fact_change", "identity_context"
    ]
    annotation_status: Literal[
        "synthetic_expectation", "proposal", "independently_reviewed"
    ]


class AnswerClaim(Strict):
    claim_id: str
    text: str
    assertion_ids: list[str]
    evidence_span_ids: list[str]
    source_publisher_ids: list[str]


class AgentResponse(Strict):
    action: Action
    answer: str | None
    clarification_question: str | None
    compared_scopes: list[str]
    claims: list[AnswerClaim]

    @model_validator(mode="after")
    def action_shape(self):
        if self.action == "ASK_CLARIFICATION":
            if (
                not self.clarification_question
                or not self.clarification_question.strip()
                or self.claims
            ):
                raise ValueError(
                    "clarification requires specific question and no definitive claims"
                )
        elif self.action == "ABSTAIN_INSUFFICIENT_EVIDENCE":
            if self.claims:
                raise ValueError(
                    "insufficient evidence action cannot carry definitive claims"
                )
        elif not self.claims or not self.answer:
            raise ValueError("answer/compare requires cited claims")
        if (
            self.action == "COMPARE_SCOPED_ANSWERS"
            and len(set(self.compared_scopes)) < 2
        ):
            raise ValueError("comparison must preserve distinct scopes/modalities")
        return self


def validate_claim_bindings(response, case, snapshot):
    """Check delivered provenance links, not semantic entailment of claim text."""
    delivered = {
        a.assertion_id: a
        for a in snapshot.assertions
        if a.assertion_id in case.context_assertion_ids
    }
    for claim in response.claims:
        if not claim.assertion_ids or not set(claim.assertion_ids) <= set(delivered):
            raise ValueError("claim cites absent/undelivered assertions")
        assertions = [delivered[i] for i in claim.assertion_ids]
        spans = {s for a in assertions for s in a.evidence_span_ids}
        publishers = {a.source_publisher_id for a in assertions}
        if not claim.evidence_span_ids or not set(claim.evidence_span_ids) <= spans:
            raise ValueError("claim evidence is not linked to cited assertions")
        if set(claim.source_publisher_ids) != publishers:
            raise ValueError("claim publisher binding mismatch")
    return True
