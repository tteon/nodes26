"""Secondary RQ1 measures; labels and fixed reference ontology are evaluator inputs."""
from __future__ import annotations

from .evaluation import evaluate, fact_key
from .metrics import ratio


def overlap(left, right):
    left,right={fact_key(f) for f in left},{fact_key(f) for f in right}
    return {"jaccard":ratio(len(left & right),len(left | right)),"left_only":len(left-right),
            "right_only":len(right-left),"intersection":len(left & right)}


def extraction_metrics(predictions, gold, reference_ontology, linking_labels=None):
    primary=evaluate(predictions,gold)
    supported=[f for f in gold["assertions"] if f["annotation_status"]=="supported"]
    # Surface entity measures are explicitly scoped; do not call them entity linking.
    predicted_entities={(f["subject"],f["domain_id"],f["business_scope"]) for f in predictions}
    gold_entities={(f["subject"],f["domain_id"],f["business_scope"]) for f in supported}
    tp=len(predicted_entities & gold_entities)
    types=set(reference_ontology.get("entity_types",[]))
    relations=set(reference_ontology.get("relations",{}))
    violations=sum(f.get("subject_type") not in types or f["predicate"] not in relations for f in predictions)
    labels=linking_labels or {}
    labeled=[f for f in predictions if labels.get(f["fact_id"]) is not None]
    return {**primary,"surface_entity_gold_recall":ratio(tp,len(gold_entities)),
        "surface_entity_match_fraction":ratio(tp,len(predicted_entities)),
        "surface_entity_f1":None if predicted_entities-gold_entities else ratio(2*tp,len(gold_entities)+len(predicted_entities)),
        "entity_linking_accuracy":ratio(sum(f.get("subject_id")==labels[f["fact_id"]] for f in labeled),len(labeled)),
        "entity_linking_annotation_coverage":ratio(len(labeled),len(predictions)),
        "reference_ontology_violation_rate":ratio(violations,len(predictions)),
        "reference_ontology_scope":"type/relation membership only, no semantic rule reasoning",
        "type_coverage":ratio(len({f.get('subject_type') for f in predictions} & types),len(types)),
        "relation_coverage":ratio(len({f['predicate'] for f in predictions} & relations),len(relations)),
        "reviewed_unsupported_assertion_rate":ratio(primary["reviewed_false_positive"],
            primary["true_positive"]+primary["reviewed_false_positive"]),
        "evidence_grounded_fraction":ratio(primary["true_positive"],len({fact_key(f) for f in predictions})),
        "unannotated_entity_policy":"unknown; surface match fraction is not semantic entity precision"}
