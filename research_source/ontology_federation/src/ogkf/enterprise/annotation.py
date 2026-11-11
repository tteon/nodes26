"""Blind capture, typed source-bound judgments and immutable adjudication.

These protocol checks cannot prove human identity or independence. No command
here authors a human judgment or upgrades generated candidates automatically.
"""

from collections import Counter, defaultdict
import json
from typing import Literal, Union, Annotated
from pydantic import Field, StrictBool, model_validator
from ..models import Strict, digest
from ..ds01.contracts import text_hash
from .contracts import DIMENSIONS, Action, Qualifiers


class JudgmentBase(Strict):
    target_id: str
    source_span_ids: list[str]
    rationale: str


class EntityIdentityGold(JudgmentBase):
    kind: Literal["entity_identity_gold"]
    left_mention_id: str
    right_mention_id: str
    canonical_entity_ids: list[str]
    verdict: Literal["SAME", "DIFFERENT", "UNKNOWN"]
    mention_locators: list[dict] = Field(default_factory=list)


class ContextCompatibilityGold(JudgmentBase):
    kind: Literal["context_compatibility_gold"]
    query_id: str
    operation: str | None
    dimensions: dict[str, Literal["COMPATIBLE", "INCOMPATIBLE", "UNKNOWN"]]

    @model_validator(mode="after")
    def all_dimensions(self):
        if set(self.dimensions) != set(DIMENSIONS):
            raise ValueError("all eight context dimensions required")
        return self


class FactSupportGold(JudgmentBase):
    kind: Literal["fact_support_gold"]
    subject: str | None
    predicate: str | None
    object_value: str | None
    qualifiers: Qualifiers | None
    verdict: Literal["SUPPORTED", "UNSUPPORTED", "UNKNOWN"]


class MinimalSufficientEvidenceSets(JudgmentBase):
    kind: Literal["minimal_sufficient_evidence_sets"]
    query_id: str
    operation: str | None
    evidence_sets: list[list[str]]
    verdict: Literal["SUFFICIENT", "NO_SUFFICIENT_SET", "UNKNOWN"]

    @model_validator(mode="after")
    def minimal_sets(self):
        sets = [frozenset(x) for x in self.evidence_sets]
        if any(not x for x in sets) or len(sets) != len(set(sets)):
            raise ValueError("empty or duplicate sufficient evidence alternatives")
        if any(a < b for a in sets for b in sets):
            raise ValueError(
                "sufficient evidence alternatives must be minimal, no dominated supersets"
            )
        if self.verdict == "SUFFICIENT" and not sets:
            raise ValueError("sufficient verdict needs alternatives")
        if self.verdict == "NO_SUFFICIENT_SET" and sets:
            raise ValueError("no-sufficient verdict contradicts alternatives")
        return self


class AnswerClaimGold(JudgmentBase):
    kind: Literal["answer_claim_gold"]
    query_id: str
    claim_id: str | None
    answer_hash: str | None
    allowed_actions: list[Action]
    answer_correct: StrictBool | None
    source_supported: StrictBool | None
    context_compatible: StrictBool | None
    composition_valid: StrictBool | None
    citation_binding_correct: StrictBool | None
    material_missing_dimensions: list[str]


Judgment = Annotated[
    Union[
        EntityIdentityGold,
        ContextCompatibilityGold,
        FactSupportGold,
        MinimalSufficientEvidenceSets,
        AnswerClaimGold,
    ],
    Field(discriminator="kind"),
]


class ReviewTarget(Strict):
    target_id: str
    query_id: str
    kind: str
    source_span_ids: list[str]
    proposal: Judgment


class AnnotationPacket(Strict):
    version: str
    source_bundle_hash: str
    sources: list[dict]
    targets: list[ReviewTarget]
    native_gold_policy: str
    synthetic: StrictBool = False

    @model_validator(mode="after")
    def references(self):
        spans = {r["span_id"] for r in self.sources}
        for r in self.sources:
            if (
                r["start"] != 0
                or r["end"] != len(r["quote"])
                or text_hash(r["quote"]) != r["source_hash"]
                or r["span_id"]
                != digest(
                    [r["document_id"], r["document_version"], r["start"], r["end"]]
                )
            ):
                raise ValueError("packet exact full-passage locator/hash mismatch")
        ids = [t.target_id for t in self.targets]
        if len(ids) != len(set(ids)) or len(spans) != len(self.sources):
            raise ValueError("duplicate packet source/target")
        for t in self.targets:
            if (
                t.target_id != t.proposal.target_id
                or t.kind != t.proposal.kind
                or not set(t.source_span_ids) <= spans
            ):
                raise ValueError("packet target/source mismatch")
        return self

    @property
    def packet_hash(self):
        return digest(self.model_dump())


class ReviewerSubmission(Strict):
    version: str
    packet_hash: str
    reviewer_id: str
    author_role: Literal["human_reviewer", "automated_proposal"]
    external_review_record: str | None
    judgments: list[Judgment]


class AdjudicationReceipt(Strict):
    packet_hash: str
    reviewer_submission_hashes: list[str]
    adjudicator_id: str
    external_adjudication_record: str
    decisions: list[Judgment]


def reviewer_key(text):
    import unicodedata

    return unicodedata.normalize("NFKC", text).strip().casefold()


def pending_submission(packet, reviewer_slot):
    return ReviewerSubmission(
        version="independent-review-capture-v1",
        packet_hash=packet.packet_hash,
        reviewer_id=reviewer_slot,
        author_role="human_reviewer",
        external_review_record=None,
        judgments=[t.proposal.model_copy(deep=True) for t in packet.targets],
    )


def signature(judgment):
    value = judgment.model_dump(exclude={"rationale", "source_span_ids", "target_id"})
    if isinstance(judgment, MinimalSufficientEvidenceSets):
        value["evidence_sets"] = sorted(sorted(s) for s in value["evidence_sets"])
    if isinstance(judgment, AnswerClaimGold):
        value["allowed_actions"] = sorted(value["allowed_actions"])
    return digest(value)


def validate_submission(packet, submission, *, require_human=True):
    if submission.packet_hash != packet.packet_hash:
        raise ValueError("review refers to another frozen packet")
    if (
        not reviewer_key(submission.reviewer_id)
        or not submission.external_review_record
        or not submission.external_review_record.strip()
    ):
        raise ValueError("actual external review record required")
    if require_human and submission.author_role != "human_reviewer":
        raise ValueError("automated proposals are not independent human gold")
    targets = {t.target_id: t for t in packet.targets}
    seen = set()
    sources = {s["span_id"] for s in packet.sources}
    for j in submission.judgments:
        if (
            j.target_id in seen
            or j.target_id not in targets
            or j.kind != targets[j.target_id].kind
        ):
            raise ValueError("duplicate/unknown/mismatched target")
        seen.add(j.target_id)
        if (
            not j.rationale.strip()
            or not j.source_span_ids
            or not set(j.source_span_ids) <= sources
        ):
            raise ValueError("review requires source-linked rationale")
        if not set(j.source_span_ids) <= set(targets[j.target_id].source_span_ids):
            raise ValueError(
                "review borrows sources outside target;version packet to extend evidence"
            )
        if hasattr(j, "query_id") and j.query_id != targets[j.target_id].query_id:
            raise ValueError("question changed")
        if isinstance(j, AnswerClaimGold):
            original = targets[j.target_id].proposal
            if (j.claim_id, j.answer_hash) != (original.claim_id, original.answer_hash):
                raise ValueError(
                    "answer/claim changed; version the review packet for a new response"
                )
        if isinstance(j, EntityIdentityGold):
            p = targets[j.target_id].proposal
            if (j.left_mention_id, j.right_mention_id) != (
                p.left_mention_id,
                p.right_mention_id,
            ):
                raise ValueError("identity mentions changed")
            if j.verdict != "UNKNOWN" and not j.canonical_entity_ids:
                raise ValueError("identity verdict requires canonical IDs")
            if j.verdict == "SAME" and len(set(j.canonical_entity_ids)) != 1:
                raise ValueError("SAME identity needs one canonical entity")
            if j.verdict == "DIFFERENT" and len(set(j.canonical_entity_ids)) < 2:
                raise ValueError("DIFFERENT identity needs distinct canonical entities")
            if j.verdict != "UNKNOWN":
                locators = {m.get("mention_id"): m for m in j.mention_locators}
                if set(locators) != {j.left_mention_id, j.right_mention_id}:
                    raise ValueError(
                        "resolved identity needs exact locators for both mentions"
                    )
                for m in locators.values():
                    if m.get("source_span_id") not in j.source_span_ids:
                        raise ValueError("mention source not reviewed")
                    source = next(
                        s for s in packet.sources if s["span_id"] == m["source_span_id"]
                    )
                    start, end = m.get("start"), m.get("end")
                    if (
                        not isinstance(start, int)
                        or not isinstance(end, int)
                        or not 0 <= start < end <= len(source["quote"])
                        or source["quote"][start:end] != m.get("quote")
                    ):
                        raise ValueError("mention offset/quote mismatch")
        if isinstance(j, FactSupportGold) and j.verdict != "UNKNOWN":
            if (
                not j.subject
                or not j.predicate
                or j.object_value is None
                or j.qualifiers is None
            ):
                raise ValueError(
                    "fact support requires normalized assertion and qualifiers"
                )
        if isinstance(j, MinimalSufficientEvidenceSets):
            allowed_evidence = {
                s["evidence_id"]
                for s in packet.sources
                if s["span_id"] in targets[j.target_id].source_span_ids
            }
            if any(not set(s) <= allowed_evidence for s in j.evidence_sets):
                raise ValueError("unknown evidence in sufficient set")
    return True


def agreement(packet, left, right):
    validate_submission(packet, left)
    validate_submission(packet, right)
    if reviewer_key(left.reviewer_id) == reviewer_key(right.reviewer_id):
        raise ValueError("reviewers must be distinct after normalization")
    a = {j.target_id: j for j in left.judgments}
    b = {j.target_id: j for j in right.judgments}
    cells = defaultdict(list)

    def label(j):
        if hasattr(j, "verdict"):
            return j.verdict
        if isinstance(j, ContextCompatibilityGold):
            return json.dumps(j.dimensions, sort_keys=True)
        return json.dumps(
            {
                k: v
                for k, v in j.model_dump().items()
                if k
                in {
                    "allowed_actions",
                    "answer_correct",
                    "source_supported",
                    "context_compatible",
                    "composition_valid",
                    "citation_binding_correct",
                }
            },
            sort_keys=True,
        )

    for key in sorted(a.keys() & b.keys()):
        cells[a[key].kind].append(
            (signature(a[key]), signature(b[key]), key, label(a[key]), label(b[key]))
        )
    results = []
    for kind, rows in sorted(cells.items()):
        ca, cb = (
            Counter(x for _, _, _, x, _ in rows),
            Counter(y for _, _, _, _, y in rows),
        )
        n = len(rows)
        observed = sum(x == y for _, _, _, x, y in rows) / n
        expected = sum(ca[k] * cb[k] for k in ca.keys() | cb.keys()) / (n * n)
        results.append(
            dict(
                kind=kind,
                paired_targets=n,
                decision_agreement=observed,
                full_judgment_agreement=sum(x == y for x, y, _, _, _ in rows) / n,
                cohen_kappa=(observed - expected) / (1 - expected)
                if expected < 1
                else None,
                class_prevalence_left={k: v / n for k, v in ca.items()},
                class_prevalence_right={k: v / n for k, v in cb.items()},
                disagreements=[key for x, y, key, _, _ in rows if x != y],
                scope="Before adjudication;class prevalence uses semantic decision labels,not target IDs or rationale;targets are not independent questions.",
            )
        )
    return dict(
        packet_hash=packet.packet_hash,
        reviewer_ids=[left.reviewer_id, right.reviewer_id],
        paired_target_count=len(a.keys() & b.keys()),
        eligible_target_count=len(packet.targets),
        by_kind=results,
    )


def adjudicate(packet, left, right, receipt):
    stats = agreement(packet, left, right)
    hashes = sorted([digest(left.model_dump()), digest(right.model_dump())])
    if (
        receipt.packet_hash != packet.packet_hash
        or sorted(receipt.reviewer_submission_hashes) != hashes
        or not receipt.external_adjudication_record.strip()
        or not reviewer_key(receipt.adjudicator_id)
        or reviewer_key(receipt.adjudicator_id)
        in {reviewer_key(left.reviewer_id), reviewer_key(right.reviewer_id)}
    ):
        raise ValueError(
            "adjudication must bind both original independent submissions and a distinct adjudicator"
        )
    approved = ReviewerSubmission(
        version="adjudicated-capture-v1",
        packet_hash=packet.packet_hash,
        reviewer_id=receipt.adjudicator_id,
        author_role="human_reviewer",
        external_review_record=receipt.external_adjudication_record,
        judgments=receipt.decisions,
    )
    validate_submission(packet, approved)
    left_ids = {j.target_id for j in left.judgments}
    right_ids = {j.target_id for j in right.judgments}
    if any(j.target_id not in left_ids & right_ids for j in receipt.decisions):
        raise ValueError(
            "both reviewers must independently label each adjudicated target"
        )
    # Synthetic expectations may exercise the machinery, never populate empirical gold.
    resolved = sum(
        j.verdict != "UNKNOWN"
        if hasattr(j, "verdict")
        else (
            all(v != "UNKNOWN" for v in j.dimensions.values())
            if isinstance(j, ContextCompatibilityGold)
            else bool(j.allowed_actions)
            and all(
                getattr(j, k) is not None
                for k in [
                    "answer_correct",
                    "source_supported",
                    "context_compatible",
                    "composition_valid",
                    "citation_binding_correct",
                ]
            )
        )
        for j in receipt.decisions
    )
    return dict(
        version="adjudicated-enterprise-gold-v1",
        packet_hash=packet.packet_hash,
        source_bundle_hash=packet.source_bundle_hash,
        gold_version=digest(receipt.model_dump()),
        synthetic=packet.synthetic,
        reviewer_submission_hashes=hashes,
        reviewer_ids=[left.reviewer_id, right.reviewer_id],
        adjudicator=receipt.adjudicator_id,
        adjudication_record=receipt.external_adjudication_record,
        agreement_before_adjudication=stats,
        judgments=[j.model_dump() for j in receipt.decisions],
        resolved_targets=resolved,
        empirical_gold_eligible=not packet.synthetic and resolved > 0,
    )


def action_metrics(predicted, allowed_actions):
    if allowed_actions is None or not allowed_actions:
        return dict(
            action_correct=None,
            unnecessary_clarification=None,
            missing_clarification=None,
            correct_abstention=None,
        )
    return dict(
        action_correct=predicted in allowed_actions,
        unnecessary_clarification=predicted == "ASK_CLARIFICATION"
        if "ANSWER_WITH_CITATIONS" in allowed_actions
        and "ASK_CLARIFICATION" not in allowed_actions
        else None,
        missing_clarification=predicted != "ASK_CLARIFICATION"
        if set(allowed_actions) == {"ASK_CLARIFICATION"}
        else None,
        correct_abstention=predicted == "ABSTAIN_INSUFFICIENT_EVIDENCE"
        if set(allowed_actions) == {"ABSTAIN_INSUFFICIENT_EVIDENCE"}
        else None,
    )
