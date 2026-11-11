"""Evaluator-only labels: no imports from this module in routing or executors."""
from collections import defaultdict
import json
import math
import random
from statistics import mean

from ..models import digest
from .contracts import JudgeResult


def divide(numerator, denominator):
    return numerator / denominator if denominator else None


def retrieval_metrics(row, gold, dataset):
    delivered = set(row["evidence_ids"])
    expected = gold.required_ids()
    covered = [any(set(alternative) <= delivered for alternative in group) for group in gold.required_groups]
    complete = all(covered) and not row["error"]
    relevant = {d.domain_id for d in dataset.documents if d.evidence_id in expected}
    selected = set(row["selected_providers"])
    dcg = sum(gold.relevance.get(e, 0) / math.log2(i + 2) for i, e in enumerate(row["evidence_ids"]))
    ideal = sum(v / math.log2(i + 2) for i, v in enumerate(sorted(gold.relevance.values(), reverse=True)[:row["global_top_k"]]))
    reviewed = gold.sufficient_set_status == "independently_reviewed"
    judged_irrelevant = delivered & set(gold.irrelevant_evidence_ids)
    judged = delivered & (set(gold.relevance) | set(gold.irrelevant_evidence_ids))
    return {
        "evidence_sufficiency_at_k": complete if reviewed else None,
        "provided_reference_sufficiency_proxy": complete,
        "required_group_recall": divide(sum(covered), len(covered)),
        "evidence_recall_at_k": divide(len(delivered & expected), len(expected)),
        "ndcg_at_k": divide(dcg, ideal) if reviewed and delivered <= judged else None,
        "annotated_reference_ndcg_proxy": divide(dcg, ideal),
        "provider_selection_precision": divide(len(selected & relevant), len(selected)),
        "provider_selection_recall": divide(len(selected & relevant), len(relevant)),
        "provider_label_scope": "Locations of provided references in frozen logical partitions; not FinDER category/provider truth",
        "irrelevant_evidence_ratio": divide(len(judged_irrelevant), len(delivered)) if delivered and delivered <= judged else None,
        "judged_subset_irrelevant_ratio": divide(len(judged_irrelevant), len(judged)),
        "relevance_judgment_coverage": divide(len(judged), len(delivered)),
        "context_leakage": None, "grounded_qa_success": None,
        "metric_version": "ds01-evidence-v2", "research_qualified": reviewed,
        "execution_success": row["error"] is None
    }


def judge(query, gold, generated, evidence, gateway, model, metadata):
    if generated is None:
        return JudgeResult(correctness=0., faithfulness=None, context_leakage=None,
                           attribution_accuracy=None, rationale="Generation failed.", reference_defect=False)
    # Correctness and source support have different inputs; no strategy names.
    correct_schema = {"type": "object", "additionalProperties": False,
        "properties": {"score": {"type": "number", "enum": [0, .5, 1]},
                       "reason": {"type": "string"}, "reference_defect": {"type": "boolean"}},
        "required": ["score", "reason", "reference_defect"]}
    correct = gateway.generate_structured([
        {"role": "system", "content":
            "Evaluate financial QA against the published answer. Ignore style, citation formatting and harmless wording. "
            "1=all material requested conclusions/quantities correct; .5=substantively correct but incomplete without a critical contradiction; "
            "0=wrong conclusion, critical entity/period/scope/unit/sign/calculation error or abstention. "
            "Flag a suspected reference defect; never repair gold. Output JSON schema: " + json.dumps(correct_schema)},
        {"role": "user", "content": json.dumps({"question": query.text, "published_answer": gold.answer,
                                               "candidate_answer": generated.model_dump()})}],
        correct_schema, model, {"max_tokens": 4096}, {**metadata, "parent_step_id": "evaluate_correctness",
                                                     "document_id": query.query_id})["value"]
    faith_schema = {"type": "object", "additionalProperties": False,
        "properties": {"faithfulness": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "context_leakage": {"type": ["boolean", "null"]},
            "attribution_accuracy": {"type": ["number", "null"], "minimum": 0, "maximum": 1},
            "reason": {"type": "string"}},
        "required": ["faithfulness", "context_leakage", "attribution_accuracy", "reason"]}
    faithful = gateway.generate_structured([
        {"role": "system", "content":
            "Judge every material candidate claim against ACTUALLY DELIVERED evidence. Check cited sources, issuer, period, "
            "organizational scope, modality and calculation operands. Source support and relevance to the question differ. "
            "A true source statement applied to the wrong context is leakage. Missing labels/ambiguous context are null, not false. "
            "No material claims implies null faithfulness/attribution. Do not use outside facts. JSON schema: " + json.dumps(faith_schema)},
        {"role": "user", "content": json.dumps({"question": query.text, "answer": generated.model_dump(), "evidence": evidence})}],
        faith_schema, model, {"max_tokens": 4096}, {**metadata, "parent_step_id": "evaluate_faithfulness",
                                                   "document_id": query.query_id})["value"]
    return JudgeResult(correctness=correct["score"], faithfulness=faithful["faithfulness"],
        context_leakage=faithful["context_leakage"], attribution_accuracy=faithful["attribution_accuracy"],
        rationale=correct["reason"] + " | " + faithful["reason"], reference_defect=correct["reference_defect"])


def dependence_groups(dataset):
    """Join known same-company/shared-source questions before resampling.

    Exact shared evidence, passage content, or filing URI creates dependence even
    when company labels differ. Unknown issuer links remain an explicit limitation.
    """
    parent = {q.query_id: q.query_id for q in dataset.queries}
    owners = {}
    by_evidence = {d.evidence_id: d for d in dataset.documents}

    def root(q):
        while parent[q] != q:
            parent[q] = parent[parent[q]]
            q = parent[q]
        return q

    for g in sorted(dataset.gold, key=lambda g: g.query_id):
        keys = [("company", g.company_id)] if g.company_id else []
        for eid in sorted(g.required_ids()):
            doc = by_evidence[eid]
            keys.extend([("evidence", eid), ("content", doc.content_hash)])
            if doc.source_uri:
                keys.append(("source_uri", doc.source_uri))
        for key in keys:
            if key in owners:
                a, b = sorted([root(g.query_id), root(owners[key])])
                parent[b] = a
            else:
                owners[key] = g.query_id
    members = defaultdict(list)
    for q in sorted(parent):
        members[root(q)].append(q)
    labels = {q: digest(["known-company-source-component-v1", group])
              for group in members.values() for q in group}
    return labels


def measured_shift(shift, target_entity_id=None):
    """Use the intended intervention feature, never a universal JSD proxy."""
    if shift.intervention == "DS-01A":
        return "target_entity_tf", shift.observed.get("entity_tf", {}).get(target_entity_id)
    if shift.intervention == "DS-01B":
        values = shift.covariate_changes.get("neighborhood_jsd", {})
        if target_entity_id is None:
            return "target_entity_neighborhood_jsd", None
        return "target_entity_neighborhood_jsd", values.get(target_entity_id)
    if shift.intervention == "DS-01C":
        return "domain_jsd", shift.covariate_changes.get("domain_jsd")
    raise ValueError("unknown intervention")


def paired_report(evaluations, shifts, dataset, bootstrap=500):
    """Crossed company/source-component and corpus-seed resampling; small-n limits explicit."""
    lookup = {s.condition_id: s for s in shifts}
    groups = dependence_groups(dataset)
    base = {(r["strategy"], r["query_id"], lookup[r["condition_id"]].intervention, r["seed"]): r
            for r in evaluations if lookup[r["condition_id"]].severity == 0}
    cells = defaultdict(list)
    for row in evaluations:
        shift = lookup[row["condition_id"]]
        if shift.severity == 0:
            continue
        reference = base[row["strategy"], row["query_id"], shift.intervention, row["seed"]]
        field = "evidence_sufficiency_at_k"
        if row[field] is None or reference[field] is None:
            field = "provided_reference_sufficiency_proxy"
        cells[shift.intervention, shift.severity, row["strategy"], field].append({
            "company": groups[row["query_id"]], "seed": row["seed"],
            "delta": float(row[field]) - float(reference[field])})
    output = []
    rng = random.Random(20261028)
    for (intervention, severity, strategy, metric), rows in sorted(cells.items()):
        companies = sorted({r["company"] for r in rows})
        seeds = sorted({r["seed"] for r in rows})
        draws = []
        for _ in range(bootstrap):
            sampled_companies = rng.choices(companies, k=len(companies))
            sampled_seeds = rng.choices(seeds, k=len(seeds))
            values = [r["delta"] for c in sampled_companies for seed in sampled_seeds
                      for r in rows if r["company"] == c and r["seed"] == seed]
            if values:
                draws.append(mean(values))
        draws.sort()
        output.append({"intervention": intervention, "severity": severity, "strategy": strategy,
            "metric": metric, "paired_mean_change": mean(r["delta"] for r in rows),
            "ci95": [draws[int(.025 * (len(draws) - 1))], draws[int(.975 * (len(draws) - 1))]],
            "company_or_source_clusters": len(companies), "corpus_seed_clusters": len(seeds),
            "dependence_grouping": "connected known-company/shared-evidence/content/source-URI components",
            "analysis_version": "ds01-paired-v3",
            "pairs": len(rows), "inferentially_qualified": len(companies) >= 10 and len(seeds) >= 5,
            "scope": "Exploratory crossed-cluster bootstrap; not population/noninferiority proof."})
    return output


def strategy_shift_interaction(evaluations, shifts, dataset, target_entity_id=None):
    """Separate descriptive moderation per intervention; never pool unlike shifts.

    Binary-outcome linear mixed models are exploratory sensitivity analyses. The
    paired assigned-condition contrasts, not this measured-covariate regression,
    support intervention claims subject to the recorded isolation limitations.
    """
    lookup = {s.condition_id: s for s in shifts}
    groups = dependence_groups(dataset)
    reports = []
    for intervention in sorted({s.intervention for s in shifts}):
        rows = []
        metric_name = None
        eligible = [r for r in evaluations if lookup[r["condition_id"]].intervention == intervention
                    and r["evidence_sufficiency_at_k"] is not None]
        for r in eligible:
            metric_name, magnitude = measured_shift(lookup[r["condition_id"]], target_entity_id)
            if magnitude is not None and math.isfinite(magnitude):
                rows.append({"outcome": float(r["evidence_sufficiency_at_k"]), "strategy": r["strategy"],
                    "seed": str(r["seed"]), "component": groups[r["query_id"]], "observed_shift": magnitude})
        components = {r["component"] for r in rows}
        seeds = {r["seed"] for r in rows}
        result = {"intervention": intervention, "shift_metric": metric_name,
            "analysis_version": "ds01-moderation-v3", "rows": len(rows),
            "company_source_components": len(components), "seeds": len(seeds),
            "missing_shift_rows": len(eligible)-len(rows),
            "formula": "outcome ~ C(strategy) * observed_shift",
            "scope": "Exploratory linear mixed model of binary outcome; measured-shift association, not an isolated causal effect."}
        if not eligible:
            result["status"] = "semantic_sufficiency_labels_missing"
        elif not rows:
            result["status"] = "intervention_shift_measure_missing"
        elif len(components) < 10 or len(seeds) < 5:
            result["status"] = "insufficient_independent_clusters"
        elif len({r["observed_shift"] for r in rows}) < 2:
            result["status"] = "shift_variation_missing"
        elif len({r["outcome"] for r in rows}) < 2:
            result["status"] = "outcome_variation_missing"
        else:
            result.update(_fit_interaction(rows))
        reports.append(result)
    return {"analysis_version": "ds01-moderation-v3", "interventions": reports}


def _fit_interaction(rows):
    try:
        import pandas as pd
        import statsmodels.formula.api as smf
    except ImportError:
        return {"status": "optional_statsmodels_unavailable", "missing_metrics": None}
    try:
        # Constant top-level group makes all variance-component effects crossed.
        data = pd.DataFrame(rows).assign(all_group=1)
        fit = smf.mixedlm("outcome ~ C(strategy) * observed_shift", data, groups="all_group",
            re_formula="0", vc_formula={"component": "0+C(component)", "seed": "0+C(seed)"}).fit()
        return {"status": "fitted", "parameters": fit.params.to_dict(), "ci95": fit.conf_int().to_dict(),
                "converged": bool(fit.converged), "scope": "Descriptive measured-shift moderation"}
    except Exception as exc:
        return {"status": "fit_failed", "error": type(exc).__name__, "coefficients": None}
