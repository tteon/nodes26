"""Deterministic sampling interventions. Never mutate a document or assertion."""
from collections import Counter, defaultdict
import math
import random
from statistics import mean

from ..models import digest
from .contracts import CorpusManifest, ShiftManifest

GENERATOR_VERSION = "ds01-sampling-v2-distinct-severities"


def normalize(counts):
    total = sum(counts.values())
    return {k: v / total for k, v in sorted(counts.items())} if total else {}


def jsd(left, right):
    p, q = normalize(left), normalize(right)
    if not p or not q:
        return None
    middle = {k: (p.get(k, 0) + q.get(k, 0)) / 2 for k in p.keys() | q.keys()}
    return sum(v * math.log2(v / middle[k]) / 2 for d in (p, q) for k, v in d.items() if v)


def fixed_idf(dataset):
    n = len(dataset.documents)
    df = Counter(e for d in dataset.documents for e, count in d.entity_mentions.items() if count)
    return {e.entity_id: math.log((n + 1) / (df[e.entity_id] + 1)) + 1 for e in dataset.entities}


def measure(dataset, document_ids, idf):
    docs = [d for d in dataset.documents if d.document_id in set(document_ids)]
    mentions, joint, relations, domains, neighborhoods, degree = Counter(), Counter(), Counter(), Counter(), defaultdict(Counter), Counter()
    by_id = {d.document_id: d for d in docs}
    for d in docs:
        domains[d.domain_id] += 1
        mentions.update(d.entity_mentions)
        relations.update(d.relation_counts)
        for entity, count in d.entity_mentions.items():
            if count:
                joint[entity + "|" + d.domain_id] += count
    for a in dataset.assertions:
        if a.document_id in by_id:
            neighborhoods[a.subject_id][a.predicate] += 1
            degree[a.subject_id] += 1
    entropy = {}
    for e in dataset.entities:
        distribution = normalize({domain: count for key, count in joint.items()
                                  for entity, domain in [key.split("|", 1)] if entity == e.entity_id})
        entropy[e.entity_id] = -sum(v * math.log2(v) for v in distribution.values()) if distribution else None
    total_tokens = sum(d.token_count for d in docs)
    tf = {e.entity_id: mentions[e.entity_id] / total_tokens for e in dataset.entities}
    return {
        "document_count": len(docs), "token_count": total_tokens,
        "mean_document_tokens": mean(d.token_count for d in docs) if docs else None,
        "entity_mentions": dict(mentions), "entity_tf": tf,
        "entity_tfidf_fixed": {e: value * idf[e] for e, value in tf.items()},
        "domain_document_counts": dict(domains),
        "entity_domain_joint": normalize(joint), "domain_entropy": entropy,
        "relation_counts": dict(relations),
        "entity_relation_neighborhood": {e: dict(v) for e, v in neighborhoods.items()},
        "entity_degrees": {e.entity_id: degree[e.entity_id] for e in dataset.entities},
        "entity_degree_distribution": dict(Counter(str(degree[e.entity_id]) for e in dataset.entities)),
        "idf_reference": "fixed full source pool; unchanged across interventions and seeds",
        "ontology_kind": dataset.ontology_kind
    }


def manifest(dataset, ids):
    ids = sorted(set(ids))
    docs = sorted((d for d in dataset.documents if d.document_id in ids), key=lambda d: d.document_id)
    assertions = sorted((a for a in dataset.assertions if a.document_id in ids), key=lambda a: a.assertion_id)
    logical_hash = digest({"documents": [d.model_dump() for d in docs],
        "assertions": [a.model_dump() for a in assertions], "entities": [e.model_dump() for e in dataset.entities]})
    cid = digest([dataset.dataset_hash, ids, logical_hash, dataset.ontology_version])
    gold_ids = sorted(set().union(*(g.required_ids() for g in dataset.gold)))
    return CorpusManifest(corpus_id=cid, graph_snapshot_id="ds01-" + cid,
        dataset_hash=dataset.dataset_hash, document_ids=ids,
        assertion_ids=[a.assertion_id for a in assertions], logical_payload_hash=logical_hash,
        ontology_version=dataset.ontology_version, document_count=len(ids),
        token_count=sum(d.token_count for d in docs), gold_evidence_ids=gold_ids)


def validate_manifest(dataset, corpus, budget):
    if corpus.dataset_hash != dataset.dataset_hash:
        raise ValueError("dataset changed after freeze")
    if corpus.document_count != budget or len(corpus.document_ids) != budget:
        raise ValueError("document budget mismatch")
    if len(set(corpus.document_ids)) != len(corpus.document_ids):
        raise ValueError("document duplication")
    if not set(corpus.document_ids) <= {d.document_id for d in dataset.documents}:
        raise ValueError("unknown document")
    if manifest(dataset, corpus.document_ids) != corpus:
        raise ValueError("manifest content/hash mismatch")
    available = {d.evidence_id for d in dataset.documents if d.document_id in corpus.document_ids}
    if not set(corpus.gold_evidence_ids) <= available:
        raise ValueError("gold evidence removed")
    # Validation of exact documents plus Dataset span/hash checks protects all facts.
    return True


def changes(reference, shifted):
    keys = ("domain_document_counts", "relation_counts", "entity_mentions")
    delta = {k: sum(abs(shifted[k].get(x, 0) - reference[k].get(x, 0))
                   for x in reference[k].keys() | shifted[k].keys()) for k in keys}
    neighborhoods = {}
    for e in reference["entity_relation_neighborhood"].keys() | shifted["entity_relation_neighborhood"].keys():
        neighborhoods[e] = jsd(reference["entity_relation_neighborhood"].get(e, {}),
                               shifted["entity_relation_neighborhood"].get(e, {}))
    return {**delta, "token_relative_change": shifted["token_count"] / reference["token_count"] - 1,
        "entity_domain_jsd": jsd(reference["entity_domain_joint"], shifted["entity_domain_joint"]),
        "domain_jsd": jsd(reference["domain_document_counts"], shifted["domain_document_counts"]),
        "relation_jsd": jsd(reference["relation_counts"], shifted["relation_counts"]),
        "neighborhood_jsd": neighborhoods}


def generate(dataset, config):
    docs = {d.document_id: d for d in dataset.documents}
    if config.target_entity_id not in {e.entity_id for e in dataset.entities}:
        raise ValueError("target entity unknown")
    if config.target_relation not in dataset.ontology_relations or config.target_domain not in {d.domain_id for d in docs.values()}:
        raise ValueError("shift target absent from source pool")
    protected_evidence = set().union(*(g.required_ids() for g in dataset.gold))
    core = {d.document_id for d in docs.values() if d.evidence_id in protected_evidence}
    labels = {(x.query_id, x.document_id): x for x in dataset.distractors}
    pool = sorted(rid for rid in set(docs) - core
                  if all((q.query_id, rid) in labels and labels[q.query_id, rid].kind in config.distractor_kinds
                         for q in dataset.queries))
    if config.require_reviewed_distractors:
        pool = [rid for rid in pool if all(labels[q.query_id, rid].status == "independently_reviewed_nonanswer"
                                          for q in dataset.queries)]
        if not pool:
            raise ValueError("independent distractor review missing; strict DS01 cannot run")
    slots = config.document_budget - len(core)
    if not 0 < slots <= len(pool):
        raise ValueError("infeasible corpus budget")
    idf = fixed_idf(dataset)
    feature = {
        "DS-01A": lambda d: d.entity_mentions.get(config.target_entity_id, 0),
        "DS-01B": lambda d: d.relation_counts.get(config.target_relation, 0),
        "DS-01C": lambda d: int(d.domain_id == config.target_domain)
    }
    result = []
    for seed in config.seeds:
        rng = random.Random(seed)
        random_order = {rid: rng.random() for rid in pool}
        initial_base = set(sorted(pool, key=lambda rid: (random_order[rid], rid))[:slots])
        for intervention in config.interventions:
            score = feature[intervention]
            preferred = {rid for rid in pool if score(docs[rid]) > 0}
            base = set(initial_base)
            levels = sorted(s for s in config.severities if s > 0)
            maximum = min(slots, len(preferred))
            minimum = max(0, slots - len(set(pool) - preferred))
            if maximum - minimum < len(levels):
                raise ValueError("insufficient immutable source variety for distinct severity levels: " + intervention)
            # Reserve enough source variety before freezing this intervention's
            # reference. Conditioning the reference is explicit, never a fact edit.
            while maximum - len(base & preferred) < len(levels):
                remove = sorted(base & preferred, key=lambda r: (random_order[r], r))
                add = sorted(set(pool) - preferred - base, key=lambda r: (random_order[r], r))
                old = remove[0]
                new = min(add, key=lambda r: (abs(docs[r].token_count - docs[old].token_count), random_order[r], r))
                base.remove(old)
                base.add(new)
            base_count = len(base & preferred)
            reference = manifest(dataset, core | base)
            ref_measures = measure(dataset, reference.document_ids, idf)
            quota = {0: base_count}
            previous = base_count
            for index, level in enumerate(levels):
                raw = base_count + math.ceil(level * (maximum - base_count))
                desired = max(previous + 1, min(raw, maximum - (len(levels) - index - 1)))
                quota[level] = desired
                previous = desired
            for severity in sorted(config.severities):
                selected = set(base)
                desired = quota[severity]
                while len(selected & preferred) < desired:
                    add = sorted(preferred - selected)
                    remove = sorted(selected - preferred)
                    if not add or not remove:
                        break
                    # Match nuisance domain, relation profile, entity counts and length
                    # where available; never rewrite facts to force an exact match.
                    def nuisance(a, b):
                        x, y = docs[a], docs[b]
                        penalty = abs(x.token_count - y.token_count) / max(1, y.token_count)
                        if intervention != "DS-01C":
                            penalty += 10 * (x.domain_id != y.domain_id)
                        if intervention != "DS-01B":
                            penalty += 5 * (x.relation_counts != y.relation_counts)
                        if intervention != "DS-01A":
                            penalty += 5 * (x.entity_mentions != y.entity_mentions)
                        return penalty, random_order[a], a, b
                    _, _, a, b = min(nuisance(a, b) for a in add for b in remove)
                    selected.remove(b)
                    selected.add(a)
                corpus = manifest(dataset, core | selected)
                validate_manifest(dataset, corpus, config.document_budget)
                observed = measure(dataset, corpus.document_ids, idf)
                cov = changes(ref_measures, observed)
                nuisance_keys = {"DS-01A": ("domain_document_counts", "relation_counts"),
                                 "DS-01B": ("domain_document_counts", "entity_mentions"),
                                 "DS-01C": ("entity_mentions", "relation_counts")}[intervention]
                matched = abs(cov["token_relative_change"]) <= .05 and all(cov[k] == 0 for k in nuisance_keys)
                actual = len(selected & preferred)
                limitations = []
                if maximum == base_count:
                    limitations.append("No available shift headroom under this seed/budget.")
                if not matched:
                    limitations.append("Non-target covariates changed; no perfectly isolated causal feature claim.")
                reviewed = all((q.query_id, rid) in labels and labels[q.query_id, rid].status == "independently_reviewed_nonanswer"
                               for q in dataset.queries for rid in selected)
                if not reviewed:
                    limitations.append("Non-gold passages are source-valid candidates, not independently certified nonanswer distractors.")
                if dataset.ontology_kind != "reviewed_business_relations":
                    limitations.append("Relation neighborhood describes source metadata; business-fact neighborhood effect unqualified.")
                target = {"preferred_document_count": desired, "achieved_preferred_document_count": actual,
                    "feature": {"DS-01A": config.target_entity_id, "DS-01B": config.target_relation,
                                "DS-01C": config.target_domain}[intervention],
                    "document_budget": config.document_budget, "target_satisfied": actual == desired,
                    "effective_shift": desired > base_count, "generator_version": GENERATOR_VERSION,
                    "reference_conditioned_for_distinct_severity_capacity": base != initial_base,
                    "requested_fraction": severity, "observed_quota_fraction": (desired - base_count) / (maximum - base_count)}
                result.append(ShiftManifest(condition_id=digest([GENERATOR_VERSION, config.version, intervention, seed, severity, corpus.corpus_id]),
                    intervention=intervention, severity=severity, seed=seed, reference_corpus_id=reference.corpus_id,
                    corpus=corpus, target=target, observed=observed, covariate_changes=cov,
                    structural_valid=True, causal_isolation="matched_measured_covariates" if matched else "imperfect",
                    research_qualified=reviewed and matched and dataset.annotation_status == "independently_reviewed"
                        and all(g.sufficient_set_status == "independently_reviewed" for g in dataset.gold),
                    limitations=limitations))
    return result
