from __future__ import annotations

from decimal import Decimal
from typing import Literal
from pydantic import Field, model_validator

from ..models import Strict, digest
from ..ds01.contracts import Document, GoldEvidence, Query
from . import VERSION


class FrameConfig(Strict):
    version: Literal["finder-context-frame-pilot-v1"]
    cohort_size: int = Field(ge=1, le=40)
    selection_seed: int
    include_query_ids: list[str]
    same_issuer_candidates: int = Field(ge=0, le=5)
    other_issuer_candidates: int = Field(ge=0, le=5)
    max_rows_per_case: int = Field(ge=1, le=20)
    author_method: Literal["deterministic_rule_draft"]
    review_policy: str
    framework: Literal["https://arxiv.org/pdf/2506.02404v3"]
    agent_input_policy: str

    @model_validator(mode="after")
    def unique_ids(self):
        if len(self.include_query_ids)!=len(set(self.include_query_ids)):
            raise ValueError("duplicate mandatory cohort query")
        return self


class Span(Strict):
    span_id: str
    document_id: str
    evidence_id: str
    source_hash: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    quote: str
    kind: Literal["passage", "heading", "period", "unit", "scope", "modality", "number", "row_label", "statement"]

    def verify(self, document):
        expected = digest([document.document_id, document.document_version, self.start, self.end])
        if (self.document_id != document.document_id or self.evidence_id != document.evidence_id
                or self.source_hash != document.content_hash or self.span_id != expected
                or not 0 <= self.start < self.end <= len(document.text)
                or document.text[self.start:self.end] != self.quote):
            raise ValueError("span/source/ID mismatch")


class Hierarchy(Strict):
    document_id: str
    report_id: str
    issuer_id: str
    accession: str
    filing_sha256: str
    filing_date: str | None
    report_date: str | None
    filing_start: int
    filing_end: int
    filing_quote: str
    normalized_match_count: int = Field(ge=1)
    binding_method: Literal["pinned_filing_normalized_token_match"]
    section_label_candidate: str | None
    section_span_id: str | None
    content_unit_id: str
    caveat: str = "Report dates are not fact periods; headings and table scopes need semantic review."


class Cue(Strict):
    field: Literal["period", "unit", "scope", "modality", "topic"]
    value: str
    span_ids: list[str] = Field(min_length=1)
    status: Literal["literal_source_cue"] = "literal_source_cue"


class TableRow(Strict):
    row_id: str
    document_id: str
    label_span_id: str
    value_span_ids: list[str] = Field(min_length=1)
    header_span_ids: list[str]
    year_candidates: list[str]
    values: list[str]
    column_alignment: Literal["unreviewed", "reviewed"] = "unreviewed"
    caveat: str = "Order and signs are preserved. Year-to-cell mapping and units are not independently reviewed."


class ContextTarget(Strict):
    issuer_ids: list[str]
    public_period_candidates: list[str]
    scope_candidates: list[str]
    requested_metric_candidates: list[str]
    source_status: str = "Question cues and source-level issuer bindings; assertion subjects need review."


class CalculationCandidate(Strict):
    operation: Literal["subtract", "percentage_change"]
    row_id: str
    older_period: str
    newer_period: str
    older_span_id: str
    newer_span_id: str
    result: str
    status: Literal["conditional_on_column_alignment"] = "conditional_on_column_alignment"
    preconditions: list[str] = Field(default_factory=lambda:[
        "Row meaning matches the requested metric; lexical similarity is not semantic verification.",
        "Year/header-to-cell alignment, entity, scope, unit and definition are compatible."])


class EvidenceDecision(Strict):
    evidence_id: str
    decision: Literal["UNKNOWN", "DIRECT_SUPPORT", "PERMITTED_COMPOSITION", "CONTRAST_ONLY", "NONANSWER", "INCOMPATIBLE"]
    rationale: str


class ReasoningStep(Strict):
    step_id: str
    operation: Literal["identify", "lookup", "compare", "subtract", "percentage_change", "ratio", "aggregate", "qualitative_assessment"]
    instruction: str
    evidence_ids: list[str]
    span_ids: list[str]
    prerequisites: list[str]
    formula: str | None
    unresolved: list[str]


class CompatibilityRule(Strict):
    rule_id: str
    dimension: Literal["identity", "period", "organizational_scope", "unit", "definition", "modality", "composition"]
    requirement: str
    evidence_ids: list[str]
    decision: Literal["UNKNOWN", "PERMITTED", "PROHIBITED"]
    rationale: str


class Case(Strict):
    case_id: str
    query: Query
    original_answer: str
    original_reference_ids: list[str] = Field(min_length=1)
    original_required_groups: list[list[list[str]]]
    original_gold: GoldEvidence
    original_gold_hash: str
    topic_level_1_candidate: str
    topic_level_2_candidate: str
    operation_candidate: str
    target: ContextTarget
    reference_evidence_ids: list[str]
    candidate_evidence_ids: list[str]
    cues: list[Cue]
    table_rows: list[TableRow]
    calculation_candidates: list[CalculationCandidate]
    source_statement_span_ids: list[str] = Field(default_factory=list)
    rationale_steps: list[ReasoningStep]
    compatibility_rules: list[CompatibilityRule]
    support_groups: list[list[list[str]]]
    support_status: Literal["provided_reference_proxy", "independently_reviewed"]
    evidence_decisions: list[EvidenceDecision]
    annotation_status: Literal["silver", "reviewed_gold"]
    author_method: Literal["deterministic_rule_draft", "llm_draft", "independent_review"]
    review_record: str | None
    limitations: list[str]

    @model_validator(mode="after")
    def shape(self):
        if (self.original_gold_hash != digest(self.original_gold.model_dump())
                or self.original_gold.query_id != self.query.query_id
                or self.original_answer != self.original_gold.answer
                or self.original_required_groups != self.original_gold.required_groups
                or self.original_gold.required_ids() != set(self.reference_evidence_ids)):
            raise ValueError("original gold/answer/reference mismatch")
        if not self.support_groups or any(not group or any(not alt or len(alt) != len(set(alt)) for alt in group)
                                          for group in self.support_groups):
            raise ValueError("empty/duplicate sufficient evidence alternative")
        ids = [s.step_id for s in self.rationale_steps]
        prior = set()
        for step in self.rationale_steps:
            if step.step_id in prior or not set(step.prerequisites) <= prior:
                raise ValueError("reasoning steps must form an ordered acyclic dependency graph")
            prior.add(step.step_id)
        if not ids:
            raise ValueError("missing rationale obligations")
        dimensions=[r.dimension for r in self.compatibility_rules]
        if len(dimensions)!=7 or len(set(dimensions))!=7:
            raise ValueError("all seven context dimensions must remain explicit")
        if self.annotation_status == "silver":
            if self.support_status != "provided_reference_proxy" or self.review_record or self.author_method == "independent_review":
                raise ValueError("silver cannot attest independent support or review")
            if any(r.decision != "UNKNOWN" for r in self.compatibility_rules) or any(
                    d.decision != "UNKNOWN" for d in self.evidence_decisions):
                raise ValueError("unreviewed context decisions must remain unknown")
            if any(r.column_alignment != "unreviewed" for r in self.table_rows):
                raise ValueError("silver cannot attest table alignment review")
        elif not self.review_record or self.author_method != "independent_review":
            raise ValueError("reviewed case lacks provenance")
        return self


class ReviewProvenance(Strict):
    input_bundle_hash: str
    receipt_hash: str
    reviewer_ids: list[str]
    adjudication_record: str


class Bundle(Strict):
    version: str = VERSION
    role: Literal["evaluator_only"] = "evaluator_only"
    framework_sources: dict[str, str]
    input_hashes: dict[str, str]
    source_dataset_hash: str
    cohort_dataset_hash: str
    documents: list[Document]
    hierarchy: list[Hierarchy]
    spans: list[Span]
    cases: list[Case]
    review_provenance: list[ReviewProvenance] = Field(default_factory=list)

    @property
    def bundle_hash(self):
        return digest(self.model_dump())

    @model_validator(mode="after")
    def integrity(self):
        for values in ([d.document_id for d in self.documents], [s.span_id for s in self.spans],
                       [h.document_id for h in self.hierarchy], [c.query.query_id for c in self.cases],
                       [c.case_id for c in self.cases]):
            if len(values) != len(set(values)):
                raise ValueError("duplicate annotation identity")
        docs = {d.document_id: d for d in self.documents}
        spans = {s.span_id: s for s in self.spans}
        evidence = {d.evidence_id for d in self.documents}
        if set(docs) != {h.document_id for h in self.hierarchy}:
            raise ValueError("hierarchy/document mismatch")
        for span in self.spans:
            if span.document_id not in docs:
                raise ValueError("unknown span document")
            span.verify(docs[span.document_id])
        for h in self.hierarchy:
            if h.filing_start < 0 or h.filing_end <= h.filing_start or len(h.filing_quote) != h.filing_end-h.filing_start:
                raise ValueError("invalid filing locator")
            if h.section_span_id and (h.section_span_id not in spans or spans[h.section_span_id].document_id != h.document_id):
                raise ValueError("section anchor mismatch")
        for c in self.cases:
            wanted = set(c.reference_evidence_ids + c.candidate_evidence_ids)
            if (not wanted <= evidence or len(wanted) != len(c.reference_evidence_ids + c.candidate_evidence_ids)
                    or {docs[r].evidence_id for r in c.original_reference_ids} != set(c.reference_evidence_ids)):
                raise ValueError("case/evidence roster mismatch")
            if c.case_id != digest([VERSION, c.query.model_dump(), c.original_gold_hash]):
                raise ValueError("unstable case identity")
            used = set(c.source_statement_span_ids)
            for cue in c.cues:
                used.update(cue.span_ids)
            for row in c.table_rows:
                used.update([row.label_span_id, *row.value_span_ids, *row.header_span_ids])
                if row.document_id not in docs or len(row.values) != len(row.value_span_ids):
                    raise ValueError("invalid table row")
                if [spans[s].quote for s in row.value_span_ids] != row.values:
                    raise ValueError("table values changed source strings")
                if any(spans[s].document_id != row.document_id for s in
                       [row.label_span_id, *row.value_span_ids, *row.header_span_ids]):
                    raise ValueError("table row borrowed another document's header or values")
                if row.year_candidates != [spans[s].quote for s in row.header_span_ids]:
                    raise ValueError("table periods changed header strings")
                if row.row_id != digest([row.label_span_id, row.value_span_ids]):
                    raise ValueError("unstable table row identity")
            row_map = {r.row_id: r for r in c.table_rows}
            for calc in c.calculation_candidates:
                if calc.row_id not in row_map:
                    raise ValueError("calculation cites unknown table row")
                row = row_map[calc.row_id]
                if (len(set(row.year_candidates)) != len(row.year_candidates)
                        or len(row.year_candidates) != len(row.values)
                        or calc.older_period not in row.year_candidates or calc.newer_period not in row.year_candidates):
                    raise ValueError("ambiguous conditional calculation columns")
                oi, ni = row.year_candidates.index(calc.older_period), row.year_candidates.index(calc.newer_period)
                if calc.older_span_id != row.value_span_ids[oi] or calc.newer_span_id != row.value_span_ids[ni]:
                    raise ValueError("calculation borrowed another cell")
                def number(raw):
                    raw = raw.replace(",", "").replace("%", "").replace("−", "-").strip()
                    return Decimal("-" + raw[1:-1].strip() if raw.startswith("(") and raw.endswith(")") else raw)
                older, newer = number(row.values[oi]), number(row.values[ni])
                if calc.operation == "percentage_change" and not older:
                    raise ValueError("zero-base percentage calculation")
                expected = newer-older if calc.operation == "subtract" else (newer-older)/older*100
                if Decimal(calc.result) != expected:
                    raise ValueError("conditional arithmetic changed")
            for step in c.rationale_steps:
                used.update(step.span_ids)
                if not set(step.evidence_ids) <= wanted:
                    raise ValueError("reasoning cites unknown evidence")
            if any(s not in spans or spans[s].evidence_id not in wanted for s in used):
                raise ValueError("case cites unrelated or unknown source span")
            for groups in (c.original_required_groups, c.support_groups):
                if not {e for g in groups for alt in g for e in alt} <= wanted:
                    raise ValueError("support group cites unknown evidence")
            decisions = [d.evidence_id for d in c.evidence_decisions]
            if len(decisions) != len(set(decisions)) or set(decisions) != wanted:
                raise ValueError("each evidence item needs one explicit decision")
            if any(not set(r.evidence_ids) <= wanted for r in c.compatibility_rules):
                raise ValueError("rule cites unknown evidence")
            if any(not row.evidence_id for row in c.evidence_decisions):
                raise ValueError("missing evidence decision identity")
            if c.annotation_status == "reviewed_gold" and c.review_record not in {r.receipt_hash for r in self.review_provenance}:
                raise ValueError("reviewed case missing bundle review provenance")
        return self


class ReviewedFields(Strict):
    target: ContextTarget
    topic_level_1_candidate: str
    topic_level_2_candidate: str
    operation_candidate: str
    table_rows: list[TableRow]
    rationale_steps: list[ReasoningStep]
    compatibility_rules: list[CompatibilityRule]
    support_groups: list[list[list[str]]]
    sufficiency_reviewed: bool
    evidence_decisions: list[EvidenceDecision]


class CaseReview(Strict):
    case_id: str
    judgments: ReviewedFields


class ReviewReceipt(Strict):
    version: str = "finder-context-review-v1"
    bundle_hash: str
    independent_review_complete: bool
    reviewer_ids: list[str]
    adjudication_record: str | None
    cases: list[CaseReview]
