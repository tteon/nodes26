"""Paired clustered uncertainty and document-grouped factorial contrasts."""
from __future__ import annotations

from collections import defaultdict
import random
import statistics


def percentile(values, fraction):
    values = sorted(values)
    if not values:
        return None
    position = (len(values)-1)*fraction
    lower = int(position)
    upper = min(lower+1,len(values)-1)
    return values[lower]+(values[upper]-values[lower])*(position-lower)


def clustered_difference(rows, *, left_arm, right_arm, metric, seed=20261028, repetitions=2000, margin=.02):
    """Company-cluster resampling with all paired questions retained together.

    Missing pairs are disclosed, never converted to zero. Difference = right-left.
    """
    grouped = defaultdict(dict)
    for row in rows:
        key = (row["company_id"],row["question_id"])
        arm = row["arm_id"]
        if arm in grouped[key]:
            raise ValueError("duplicate paired experimental observation")
        grouped[key][arm] = row.get(metric)
    clusters, dropped = defaultdict(list), 0
    for (company,_), arms in grouped.items():
        left,right=arms.get(left_arm),arms.get(right_arm)
        if left is None or right is None:
            dropped += 1
            continue
        clusters[company].append(float(right)-float(left))
    if not clusters:
        return {"paired_difference":None,"ci95":None,"clusters":0,"unpaired_questions":dropped,
                "noninferiority_supported":False}
    values=[v for group in clusters.values() for v in group]
    point=statistics.mean(values)
    ci=None
    if len(clusters)>=2:
        rng=random.Random(seed)
        ids=sorted(clusters)
        draws=[]
        for _ in range(repetitions):
            sample=[v for company in rng.choices(ids,k=len(ids)) for v in clusters[company]]
            draws.append(statistics.mean(sample))
        ci=[percentile(draws,.025),percentile(draws,.975)]
    qualified=all(r.get("research_qualified") is True for r in rows)
    return {"left_arm":left_arm,"right_arm":right_arm,"paired_difference":point,"ci95":ci,
        "clusters":len(clusters),"paired_questions":len(values),"unpaired_questions":dropped,
        "seed":seed,"bootstrap_repetitions":repetitions,"noninferiority_margin":margin,
        "noninferiority_supported":bool(qualified and ci and ci[0]>-margin),
        "conditional_margin_check":bool(ci and ci[0]>-margin),
        "research_qualified":qualified,
        "interpretation":"cluster bootstrap uncertainty conditional on reviewed labels; synthetic labels are not empirical evidence"}


def factorial_fit(rows, *, metric, cluster_key="company_id", repetitions=1000, seed=20261028):
    """OLS with model/prompt/ontology main effects and all two-way interactions.

    Repetitions are averaged per document/cell first. CIs resample entire company
    clusters. Insufficient/rank-deficient designs return a blocker, not p-values.
    """
    import itertools
    import numpy as np
    factors=["model","prompt","ontology"]
    observed=defaultdict(list)
    company_by_document={}
    for row in rows:
        if row.get(metric) is None:
            continue
        document=row["document_id"]
        company=row[cluster_key]
        if document in company_by_document and company_by_document[document]!=company:
            raise ValueError("document assigned to inconsistent company clusters")
        company_by_document[document]=company
        observed[(document,*[row[f] for f in factors])].append(float(row[metric]))
    if not observed:
        return {"status":"blocked","reason":"no scored observations"}
    levels={f:sorted({key[i+1] for key in observed}) for i,f in enumerate(factors)}
    terms=[("intercept",())]
    indicators=[(f,level) for f in factors for level in levels[f][1:]]
    terms += [(f+"="+level,((f,level),)) for f,level in indicators]
    for left,right in itertools.combinations(indicators,2):
        if left[0]!=right[0]:
            terms.append((left[0]+"="+left[1]+":"+right[0]+"="+right[1],(left,right)))
    samples=[]
    for key,values in sorted(observed.items()):
        document,*cell=key
        cell=dict(zip(factors,cell))
        samples.append((company_by_document[document],
            [float(all(cell[f]==level for f,level in required)) for _,required in terms],statistics.mean(values)))
    x=np.array([s[1] for s in samples]);y=np.array([s[2] for s in samples])
    if len(samples)<=len(terms) or np.linalg.matrix_rank(x)<len(terms):
        return {"status":"blocked","reason":"insufficient or rank-deficient document/cell design",
                "observations":len(samples),"terms":len(terms)}
    coefficients=np.linalg.lstsq(x,y,rcond=None)[0]
    clusters=sorted({s[0] for s in samples})
    if len(clusters)<2:
        return {"status":"blocked","reason":"fewer than two independent company clusters"}
    rng=random.Random(seed);draws=[]
    for _ in range(repetitions):
        selected=[s for c in rng.choices(clusters,k=len(clusters)) for s in samples if s[0]==c]
        bx=np.array([s[1] for s in selected]);by=np.array([s[2] for s in selected])
        if np.linalg.matrix_rank(bx)==len(terms):
            draws.append(np.linalg.lstsq(bx,by,rcond=None)[0])
    return {"status":"estimated","estimator":"document-cell mean OLS; company-cluster bootstrap",
        "observations":len(samples),"clusters":len(clusters),"levels":levels,"seed":seed,
        "bootstrap_valid_draws":len(draws),"coefficients":[{"term":name,"effect":float(coefficients[i]),
            "ci95":[percentile([float(d[i]) for d in draws],.025),percentile([float(d[i]) for d in draws],.975)] if draws else None}
            for i,(name,_) in enumerate(terms)],
        "causal_claim":"Only registered controlled contrasts justify causal interpretation; fit alone does not"}
