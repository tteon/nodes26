"""Evaluator-only identity/context/QA contracts. Never imported by the router."""
from __future__ import annotations

from collections import defaultdict
from typing import Literal

from .models import Strict

VERSION = "context-grounded-v1"


class IdentityPair(Strict):
    left_entity_id: str
    right_entity_id: str
    same_identity: bool | None
    context_compatible: bool | None
    requires_identity_link: bool
    review_status: Literal["reviewed", "pending"]
    scenario: str
    rationale: str


class QuestionGold(Strict):
    question_id: str
    company_id: str
    category: Literal["single_domain", "cross_domain", "conflicting_evidence", "out_of_domain"]
    relevant_domains: list[str]
    supporting_assertion_groups: list[list[str]]
    acceptable_assertion_sets: list[list[str]]
    expected_status: Literal["supported", "conflicting", "abstain"]
    annotation_status: Literal["reviewed", "pending"]


class StudyGold(Strict):
    version: str
    origin: Literal["synthetic_fixture", "independent_annotation"]
    independent_review_complete: bool
    identity_pairs: list[IdentityPair]
    context_qualified_assertion_ids: list[str]
    questions: list[QuestionGold]


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def identity_metrics(snapshot, variant, gold: StudyGold):
    if gold.version != VERSION:
        raise ValueError("study evaluator version mismatch")
    known = {e.entity_id for e in snapshot.entities}
    parent = dict(variant.entities)
    def find(value):
        while parent.get(value, value) != value:
            value = parent[value]
        return value
    def union(left, right):
        parent[find(left)] = find(right)
    for mapping in variant.mappings:
        if mapping.mapping_type == "SAME_AS" and mapping.review_status == "reviewed":
            union(mapping.left_entity_id, mapping.right_entity_id)
    false_merge_identity = false_merge_context = false_split = comparable = legitimate = 0
    seen_pairs={}
    required_links={tuple(sorted([p.left_entity_id,p.right_entity_id])) for p in gold.identity_pairs
                    if p.review_status=="reviewed" and p.requires_identity_link}
    for pair in gold.identity_pairs:
        if pair.left_entity_id not in known or pair.right_entity_id not in known:
            raise ValueError("gold identity refers to unknown entity")
        if pair.review_status != "reviewed":
            continue
        pair_key=tuple(sorted([pair.left_entity_id,pair.right_entity_id]))
        label=(pair.same_identity,pair.context_compatible)
        if pair_key in seen_pairs:
            if seen_pairs[pair_key]!=label:
                raise ValueError("contradictory identity/context labels for one pair")
            continue
        seen_pairs[pair_key]=label
        comparable += 1
        merged = variant.entities[pair.left_entity_id] == variant.entities[pair.right_entity_id]
        linked = find(pair.left_entity_id) == find(pair.right_entity_id)
        false_merge_identity += merged and pair.same_identity is False
        false_merge_context += merged and pair.context_compatible is False
        false_split += pair_key in required_links and pair.same_identity is True and not linked
        legitimate += pair.same_identity is True and pair.context_compatible is False
    original = {f["fact_id"]: f for f in snapshot.assertions}
    projected = {f["fact_id"]: f for f in variant.projections}
    if not set(gold.context_qualified_assertion_ids) <= original.keys():
        raise ValueError("gold retention denominator outside frozen assertions")
    qualifiers = ("subject_id", "domain_id", "business_scope", "temporal_scope", "source_publisher",
                  "source_document_id", "source_version", "source_content_hash", "evidence_id", "unit")
    retained = sum(fid in projected and all(projected[fid].get(k) == original[fid].get(k) for k in qualifiers)
                   for fid in set(gold.context_qualified_assertion_ids))
    return {"evaluator_version": VERSION, "policy": variant.policy,
        "context_qualified_fact_retention": ratio(retained,len(set(gold.context_qualified_assertion_ids))),
        "false_identity_merges": false_merge_identity, "false_context_merges": false_merge_context,
        "false_identity_splits": false_split, "reviewed_identity_pairs": comparable,
        "identity_pair_coverage": ratio(comparable,len({tuple(sorted([p.left_entity_id,p.right_entity_id])) for p in gold.identity_pairs})),
        "identity_case_annotations":len(gold.identity_pairs),
        "legitimate_context_representations": legitimate,
        "representation_count": len(set(variant.entities.values())),
        "qualified_assertion_count": len(variant.assertions),
        "independent_review_complete": gold.independent_review_complete,
        "research_qualified": gold.origin == "independent_annotation" and gold.independent_review_complete}


def qa_metrics(episode, gold: QuestionGold, snapshot, *, independent=False):
    if episode["question_id"] != gold.question_id:
        raise ValueError("episode/gold mismatch")
    canonical = {f["fact_id"]: f for f in snapshot.assertions}
    expected_ids = {fid for group in gold.supporting_assertion_groups for fid in group}
    acceptable = {fid for group in gold.acceptable_assertion_sets for fid in group}
    if not (expected_ids | acceptable) <= canonical.keys():
        raise ValueError("QA gold cites an unknown assertion")
    evidence = {f["fact_id"]: f for f in episode["evidence"]}
    # A rehashed retrieval record cannot change the frozen source/context payload.
    core_fields = ("evidence_id", "source_document_id", "source_version", "source_content_hash", "source_publisher",
                   "domain_id", "temporal_scope", "business_scope", "predicate", "literal_value", "subject_id")
    valid_ids = {fid for fid, f in evidence.items() if fid in canonical and
                 all(f.get(k) == canonical[fid].get(k) for k in core_fields)}
    claims = episode["answer"]["claims"]
    claimed_ids = {c["assertion_id"] for c in claims}
    citation_valid = []
    for claim in claims:
        fact = canonical.get(claim["assertion_id"])
        citation_valid.append(bool(fact and claim["assertion_id"] in valid_ids and
            claim["value"] == fact["literal_value"] and all(claim.get(k) == fact[k] and claim.get(k) is not None
                for k in ("evidence_id", "source_document_id", "source_publisher", "domain_id", "temporal_scope"))))
    covered = sum(bool(set(group) & valid_ids) for group in gold.supporting_assertion_groups)
    answer_covered = all(set(group) & claimed_ids & valid_ids for group in gold.supporting_assertion_groups)
    expected_status = episode["answer"]["status"] == gold.expected_status
    if gold.expected_status == "abstain":
        correct = expected_status and not claims
        complete = True
    else:
        correct = expected_status and bool(claims) and claimed_ids <= acceptable and any(
            set(option) <= claimed_ids for option in gold.acceptable_assertion_sets)
        complete = covered == len(gold.supporting_assertion_groups) and answer_covered
    success = bool(correct and complete and all(citation_valid) and not episode["error"])
    selected = set(episode["selected_domains"])
    if "flat" in selected:
        selected = {"finance", "legal", "sustainability"}
    relevant = set(gold.relevant_domains)
    scored = gold.annotation_status == "reviewed"
    return {"evaluator_version": VERSION, "question_id": gold.question_id, "company_id": gold.company_id,
        "category": gold.category, "strategy": episode["strategy"], "identity_policy": episode["identity_policy"],
        "evidence_complete_grounded_qa_success": success if scored else None,
        "answer_correct": bool(correct) if scored else None,
        "evidence_group_recall": ratio(covered,len(gold.supporting_assertion_groups)),
        "citation_accuracy": ratio(sum(citation_valid),len(claims)),
        "database_selection_precision": ratio(len(selected & relevant),len(selected)),
        "database_selection_recall": ratio(len(selected & relevant),len(relevant)),
        "irrelevant_retrieval_ratio": ratio(len(set(evidence)-expected_ids),len(evidence)),
        "tool_calls": episode["tool_call_count"], "end_to_end_latency_ms": episode["end_to_end_latency_ms"],
        "research_qualified": bool(scored and independent), "scope": "source-assertion exact fixture QA; free text entailment unscored"}


def cost_summary(episodes, scores):
    successful = sum(s["evidence_complete_grounded_qa_success"] is True for s in scores)
    stages = [s for episode in episodes for s in episode["llm_stages"].values() if s["logical_calls"]]
    known = sum(s["known_input_tokens"] + s["known_output_tokens"] for s in stages)
    complete = bool(stages) and all(s["usage_complete"] for s in stages)
    return {"episodes": len(episodes), "correctly_answered": successful,
        "graph_calls_per_correct_answer": ratio(sum(e["tool_call_count"] for e in episodes), successful),
        "known_llm_tokens": known, "llm_usage_complete": complete,
        "llm_tokens_per_correct_answer": ratio(known,successful) if complete else None,
        "currency_cost_per_correct_answer": None, "pricing": "unverified; no currency estimate"}
