"""Deterministic silver annotations. Literal extraction never attests semantic gold."""
from collections import Counter, defaultdict, deque
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re

from ..models import digest
from ..ds01.contracts import Dataset, GoldEvidence, Query, tokens
from ..ds01.finder import NORMAL, normalized
from . import VERSION
from .models import (Bundle, CalculationCandidate, Case, CompatibilityRule, ContextTarget, Cue,
                     EvidenceDecision, Hierarchy, ReasoningStep, Span, TableRow)

FRAMEWORK = {
    "paper": "https://arxiv.org/pdf/2506.02404v3",
    "paper_title": "GraphRAG-Bench: Challenging Domain-Specific Reasoning for Evaluating Graph Retrieval-Augmented Generation",
    "sections": "3.2 hierarchy;3.3 expert-crafted rationale;4 separate answer/rationale evaluation",
    "official_reference": "https://github.com/jeremycp3/GraphRAG-Bench/blob/main/evaluator.py",
    "adaptation": "Our extension adds exact spans, financial scope/time/unit conditions and independent review. No evaluator code copied.",
    "finder": "https://huggingface.co/datasets/Linq-AI-Research/FinDER/tree/c4c1b6454aef7f0bb1c37235c7f52ce644642da0",
    "finder_license": "CC-BY-NC-4.0; retain original Linq AI Research attribution",
}
NUMBER = re.compile(r"(?<![\w.])(?:\([ \t]*)?[-−]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[ \t]*\))?%?")
YEARS = re.compile(r"\b(?:19|20)\d{2}\b")
CUES = {
    "period": YEARS,
    "unit": re.compile(r"\bin (?:thousands|millions|billions)\b|\bexcept (?:per share data|EPS)\b|\bper share\b|\b(?:USD|EUR|GBP)\b|[$€£%]", re.I),
    "scope": re.compile(r"\bconsolidated\b|\bstandalone\b|\bseparate financial\b|\bsegment\b|\bsubsidiar(?:y|ies)\b|\boperational (?:boundary|control)\b", re.I),
    "modality": re.compile(r"\b(?:target|planned|intends?|commits?|commitments?|required|shall|may|could|would)\b", re.I),
}
STOP = set("the a an and of in for from to on what is how with does vs by at year years impact change trend".split())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def operation(question):
    q = question.casefold()
    if re.search(r"(?:%|percent(?:age)?)\s*(?:change|increase|decrease)|growth rate", q):
        return "percentage_change"
    if re.search(r"\bdelta\b|\bdifference\b", q):
        return "subtract"
    if re.search(r"\bratio\b|\bmargin\b|% of|percentage of", q):
        return "ratio"
    if re.search(r"sum of|combined total|aggregate", q):
        return "aggregate"
    if re.search(r"\btrend\b|\bvs\.?\b|versus|compared|prior years|change over", q):
        return "compare"
    if re.search(r"\bobligation|enforceab|contractual|\blegal\b", q):
        return "obligation_verification"
    if re.search(r"\bscope\b|standalone|consolidated", q):
        return "scope_verification"
    if re.search(r"\bimpact\b|\bwhy\b|\bhow\b|strateg|stability|implication", q):
        return "qualitative_assessment"
    return "lookup"


def topic(question):
    q = question.casefold()
    choices = [(r"eps|earnings per share|dilut", "financial_reporting", "earnings_per_share"),
        (r"revenue|\brev\b|sales", "financial_reporting", "revenue"),
        (r"cash flow|capex|capital expenditure|capital alloc", "financial_reporting", "capital_and_cash_flow"),
        (r"debt|liabilit|liquidity|margin|assets?|ebitda|npm", "financial_reporting", "financial_position"),
        (r"emission|carbon|sustainab", "sustainability", "environmental_reporting"),
        (r"governance|leadership|compensation|insider|controller|cfo|board", "governance", "corporate_governance"),
        (r"legal|contract|litig|regulat|obligation", "legal_governance", "obligations"),
        (r"risk|cyber|volatil|competition", "business_risk", "risk_management")]
    return next(((a, b) for pattern, a, b in choices if re.search(pattern, q)), ("business_analysis", "unclassified"))


def question_periods(question):
    result = set(YEARS.findall(question))
    for m in re.finditer(r"\b(20\d{2})\s*[-–]\s*(\d{2})\b", question):
        first, last = int(m[1]), int(m[1][:2] + m[2])
        if 0 <= last-first <= 10:
            result.update(str(y) for y in range(first, last+1))
    return sorted(result)


def select_cohort(pool, native, public, config):
    """Source-binding eligibility, then public-question operation strata; no answer scoring."""
    docs = {d.document_id: d for d in pool.documents}
    eligible = {}
    rejected = Counter()
    for qid, row in native.items():
        if not row.get("answer", "").strip():
            rejected["blank_answer"] += 1
            continue
        if not row.get("source_ids") or not set(row["source_ids"]) <= docs.keys():
            rejected["incomplete_source_binding"] += 1
            continue
        if qid not in public or set(public[qid]) != {"id", "question"} or public[qid]["question"] != row["question"]:
            raise ValueError("native public/private question mismatch")
        eligible[qid] = row
    wanted = config["cohort_size"]
    seeds = config["selection_seed"]
    forced = config.get("include_query_ids", [])
    if not set(forced) <= eligible.keys() or wanted < len(set(forced)) or wanted > len(eligible):
        raise ValueError("unavailable cohort size or mandatory query")
    buckets = defaultdict(list)
    for qid in eligible:
        if qid not in forced:
            buckets[operation(public[qid]["question"])].append(qid)
    queues = {key: deque(sorted(values, key=lambda q: digest([VERSION, seeds, q]))) for key, values in buckets.items()}
    chosen = list(dict.fromkeys(forced))
    while len(chosen) < wanted:
        for name in sorted(queues):
            if queues[name] and len(chosen) < wanted:
                chosen.append(queues[name].popleft())
    queries, gold, labels = [], [], []
    for qid in chosen:
        r = eligible[qid]
        queries.append(Query(query_id=qid, text=r["question"]))
        evidence = [docs[rid].evidence_id for rid in sorted(set(r["source_ids"]))]
        # Existing pilot labels are preserved byte-for-field where present.
        old = next((g for g in pool.gold if g.query_id == qid), None)
        gold.append(old or GoldEvidence(query_id=qid, answer=r["answer"], required_groups=[[[e]] for e in evidence],
            sufficient_set_status="provided_reference_proxy", relevance={e: 1. for e in evidence},
            category=r.get("category")))
    payload = pool.model_dump()
    payload.update(version=VERSION+"-cohort", queries=[q.model_dump() for q in queries],
        gold=[g.model_dump() for g in gold], distractors=labels,
        annotation_status="Source-bound cohort; context/rationale silver pending independent review")
    cohort = Dataset.model_validate(payload)
    return cohort, {"eligible_queries": len(eligible), "exclusions": dict(rejected), "query_ids": chosen,
        "selection_rule": "forced prior-pilot queries then seeded round-robin public-operation strata; no answer-quality selection",
        "operations": dict(Counter(operation(q.text) for q in queries)), "cohort_size": wanted,
        "selection_seed": seeds, "prior_exposure": "Retrospective benchmark development, no certified unseen holdout"}


def make_span(document, start, end, kind):
    return Span(span_id=digest([document.document_id, document.document_version, start, end]),
        document_id=document.document_id, evidence_id=document.evidence_id, source_hash=document.content_hash,
        start=start, end=end, quote=document.text[start:end], kind=kind)


def document_units(document):
    spans = {}
    def add(start, end, kind):
        s = make_span(document, start, end, kind)
        spans.setdefault(s.span_id, s)
        return s.span_id
    add(0, len(document.text), "passage")
    cues = []
    for kind, pattern in CUES.items():
        for m in pattern.finditer(document.text):
            sid = add(m.start(), m.end(), kind)
            cues.append(Cue(field=kind, value=m.group(), span_ids=[sid]))
    lines = []
    for m in re.finditer(r"[^\r\n]+", document.text):
        raw = m.group()
        if raw.strip():
            start = m.start()+len(raw)-len(raw.lstrip())
            lines.append((start, start+len(raw.strip()), raw.strip()))
    # Keep headings literal; a matching phrase is a candidate, not a certified hierarchy label.
    headings = [(a, b, t) for a, b, t in lines if re.search(
        r"^note\s+\d|statements? of|^item\s+\d|^management.s discussion", t, re.I)]
    section = add(headings[0][0], headings[0][1], "heading") if headings else None
    for a,b,t in lines:
        if len(t)>=60 and len(tokens(t))>=10:
            add(a,b,"statement")
    rows, label, values, header_ids, years = [], None, [], [], []
    def flush():
        nonlocal values
        if label and values:
            label_id = add(label[0], label[1], "row_label")
            value_ids = [add(a, b, "number") for a, b in values]
            rows.append(TableRow(row_id=digest([label_id, value_ids]), document_id=document.document_id,
                label_span_id=label_id, value_span_ids=value_ids, header_span_ids=header_ids.copy(),
                year_candidates=years.copy(), values=[spans[s].quote for s in value_ids]))
        values = []
    i = 0
    while i < len(lines):
        start, end, text = lines[i]
        matches = list(NUMBER.finditer(text))
        yr = list(YEARS.finditer(text))
        if yr and (len(yr) >= 2 or YEARS.fullmatch(text)):
            flush(); label = None
            candidates = [(start+m.start(), start+m.end(), m.group()) for m in yr]
            if YEARS.fullmatch(text):
                while i+1 < len(lines) and YEARS.fullmatch(lines[i+1][2]):
                    i += 1
                    a, b, t = lines[i]
                    candidates.append((a, b, t))
            header_ids = [add(a, b, "period") for a, b, t in candidates]
            years = [t for a, b, t in candidates]
        elif not re.search(r"[A-Za-z]", text):
            if label:
                values.extend((start+m.start(), start+m.end()) for m in matches)
        else:
            flush()
            first = matches[0].start() if matches else len(text)
            # A row requires a label before its numbers. Narrative clauses remain candidate rows.
            label_text = text[:first].rstrip(" \t:$")
            if len(label_text) >= 3:
                label = start, start+len(label_text)
                values = [(start+m.start(), start+m.end()) for m in matches]
            else:
                label = None
        i += 1
    flush()
    return list(spans.values()), cues, rows, section


def locate_in_filing(document, binding, filing_text):
    matches = list(NORMAL.finditer(filing_text))
    words = [m.group().lower() for m in matches]
    full = " ".join(words)
    needle = normalized(document.text)
    # Token-aligned matches only: do not bind '2023' to the suffix of another token.
    pattern = re.compile(r"(?<!\S)" + re.escape(needle) + r"(?!\S)")
    found = list(pattern.finditer(full))
    if not found:
        raise ValueError("reference missing from pinned filing")
    token_index = full[:found[0].start()].count(" ")
    count = len(needle.split())
    a, b = matches[token_index].start(), matches[token_index+count-1].end()
    quote = filing_text[a:b]
    if normalized(quote) != needle:
        raise ValueError("filing locator failed roundtrip")
    return a, b, quote, len(found)


def number_value(raw):
    value = raw.replace(",", "").replace("%", "").replace("−", "-").strip()
    if value.startswith("(") and value.endswith(")"):
        value = "-" + value[1:-1].strip()
    return Decimal(value)


def calculations(question, op, rows, spans):
    if op not in {"subtract", "percentage_change"}:
        return []
    periods = question_periods(question)
    if len(periods) < 2:
        return []
    old, new = periods[0], periods[-1]
    terms = set(tokens(question))-STOP
    output = []
    scored = [(len(terms & set(tokens(spans[row.label_span_id].quote))),row) for row in rows]
    best = max((score for score,row in scored),default=0)
    for row in rows:
        label=spans[row.label_span_id].quote.casefold()
        if not best or len(terms & set(tokens(label)))!=best:
            continue
        if "diluted" in question.casefold() and "diluted" not in label:
            continue
        if re.search(r"\beps\b|earnings per share",question,re.I) and (
                not re.search(r"\beps\b|per share",label) or re.search(r"weighted|shares used|shares outstanding",label)):
            continue
        if (len(row.values) != len(row.year_candidates) or len(set(row.year_candidates)) != len(row.year_candidates)
                or old not in row.year_candidates or new not in row.year_candidates):
            continue
        oi, ni = row.year_candidates.index(old), row.year_candidates.index(new)
        older, newer = number_value(row.values[oi]), number_value(row.values[ni])
        if op == "percentage_change" and not older:
            continue
        result = newer-older if op == "subtract" else (newer-older)/older*100
        output.append(CalculationCandidate(operation=op, row_id=row.row_id, older_period=old, newer_period=new,
            older_span_id=row.value_span_ids[oi], newer_span_id=row.value_span_ids[ni], result=str(result)))
    return output


def build_bundle(pool, cohort, bindings, filing_root, input_hashes, config):
    docs = {d.document_id: d for d in cohort.documents}
    gold = {g.query_id: g for g in cohort.gold}
    ref_ids = {qid: sorted(d.document_id for d in cohort.documents if d.evidence_id in g.required_ids())
               for qid, g in gold.items()}
    selected = {}
    for q in cohort.queries:
        wanted = set(ref_ids[q.query_id])
        target_issuers = {bindings[r]["cik"] for r in wanted}
        terms = set(tokens(q.text))-STOP
        candidates = [d for d in cohort.documents if d.document_id not in wanted]
        candidates.sort(key=lambda d: (-len(terms & set(tokens(d.text))), d.document_id))
        same = [d.document_id for d in candidates if bindings[d.document_id]["cik"] in target_issuers]
        other = [d.document_id for d in candidates if bindings[d.document_id]["cik"] not in target_issuers]
        selected[q.query_id] = (same[:config["same_issuer_candidates"]] + other[:config["other_issuer_candidates"]])
    used = set(r for ids in ref_ids.values() for r in ids) | set(r for ids in selected.values() for r in ids)
    cache, unit_map, all_spans, hierarchy = {}, {}, {}, []
    registry = json.loads((Path(filing_root)/"source-registry.json").read_text())
    filings = {r["accession"]: r for r in registry["documents"]}
    verified = {}
    for rid in sorted(used):
        d, binding = docs[rid], bindings[rid]
        accession = binding["accession"]
        f = filings[accession]
        if accession not in cache:
            path = Path(filing_root)/f["text_path"]
            if file_hash(path) != binding["filing_sha256"] or f["text_sha256"] != binding["filing_sha256"]:
                raise ValueError("filing content/version mismatch")
            cache[accession] = path.read_bytes().decode("utf-8")
            verified[str(path)] = file_hash(path)
        a, b, quote, count = locate_in_filing(d, binding, cache[accession])
        spans, cues, rows, section = document_units(d)
        all_spans.update((s.span_id, s) for s in spans)
        unit_map[rid] = cues, rows
        hierarchy.append(Hierarchy(document_id=rid,
            report_id=digest([binding["cik"], accession, binding["filing_sha256"]]),
            issuer_id="cik:"+binding["cik"], accession=accession, filing_sha256=binding["filing_sha256"],
            filing_date=f.get("filing_date"), report_date=f.get("report_date"), filing_start=a, filing_end=b,
            filing_quote=quote, normalized_match_count=count, binding_method="pinned_filing_normalized_token_match",
            section_label_candidate=all_spans[section].quote if section else None, section_span_id=section,
            content_unit_id=d.evidence_id))
    cases = []
    for q in cohort.queries:
        g = gold[q.query_id]
        refs, candidates = ref_ids[q.query_id], selected[q.query_id]
        ref_evidence = [docs[r].evidence_id for r in refs]
        candidate_evidence = [docs[r].evidence_id for r in candidates]
        cues = [c for r in refs for c in unit_map[r][0]]
        terms = set(tokens(q.text))-STOP
        rows = [row for r in refs for row in unit_map[r][1]]
        rows = [row for row in rows if terms & set(tokens(all_spans[row.label_span_id].quote))]
        rows.sort(key=lambda r: (-len(terms & set(tokens(all_spans[r.label_span_id].quote))), r.row_id))
        rows = rows[:config["max_rows_per_case"]]
        statement_candidates=[s for s in all_spans.values() if s.document_id in refs and s.kind=="statement"]
        statement_candidates.sort(key=lambda s:(-len(terms & set(tokens(s.quote))),s.span_id))
        statement_ids=[s.span_id for s in statement_candidates[:3]]
        row_summary="; ".join(all_spans[r.label_span_id].quote+": "+" | ".join(r.values) for r in rows[:4])
        op = operation(q.text)
        step_operation = op if op in {"subtract", "percentage_change", "ratio", "aggregate", "compare", "lookup"} else "qualitative_assessment"
        case_id = digest([VERSION, q.model_dump(), digest(g.model_dump())])
        steps = [ReasoningStep(step_id="identify", operation="identify",
            instruction="질문이 요구한 기업·지표·기간·범위를 확인합니다. 보고서 발행 기업과 개별 주장의 대상 기업을 구분합니다.",
            evidence_ids=ref_evidence, span_ids=[], prerequisites=[], formula=None,
            unresolved=["Source publisher identity does not prove the subject of every assertion."]),
            ReasoningStep(step_id="ground", operation="lookup",
                instruction="원문 문장 또는 표의 값을 연도 헤더·단위·정의·제약조건과 함께 찾습니다."
                    +(" 표의 행 후보: "+row_summary if row_summary else " 서술 근거 후보를 열어 판단 조건을 확인합니다."),
                evidence_ids=ref_evidence,
                span_ids=list(dict.fromkeys(statement_ids+[s for row in rows for s in [row.label_span_id, *row.value_span_ids, *row.header_span_ids]])),
                prerequisites=["identify"], formula=None,
                unresolved=["Table column alignment, row relevance and contextual sufficiency require review."]),
            ReasoningStep(step_id="derive", operation=step_operation,
                instruction="맥락이 맞는 근거에만 필요한 계산·판단을 적용합니다. 숫자의 관측과 인과·법적 결론을 구분합니다.",
                evidence_ids=ref_evidence, span_ids=[], prerequisites=["ground"],
                formula={"subtract":"newer - older", "percentage_change":"(newer - older) / older * 100"}.get(op),
                unresolved=["Do not infer shareholder returns, causation or enforceability from a numeric trend alone.",
                    "Allow alternative valid evidence and derivations after review."])]
        dimensions = {
            "identity":"주장 대상과 모회사·자회사 역할을 확인합니다. 발행 기업을 확인한 것만으로 모든 주장의 대상이 확정되지는 않습니다.",
            "period":"회계기간과 비교·재작성 기준을 맞춥니다. 보고서 제출일과 사실이 관측된 기간을 구분합니다.",
            "organizational_scope":"연결·별도·사업부·운영 범위를 구분하고 서로 결합해도 되는지 확인합니다.",
            "unit":"금액 배수·통화·주당 수치 예외·퍼센트·음수 부호를 확인한 뒤 계산합니다.",
            "definition":"기본·희석 EPS, GAAP·조정 지표 및 지표 정의의 변경 여부를 확인합니다.",
            "modality":"계획·예측·조건부 의무와 이미 실현된 관측을 구분합니다.",
            "composition":"근거 조합이 결론을 뒷받침하는지 확인합니다. 동시 변화만으로 인과관계나 법적 강제력을 확정하지 않습니다."}
        decisions = [EvidenceDecision(evidence_id=e, decision="UNKNOWN",
            rationale="Published reference membership or source similarity is not an independent eligibility judgment.")
            for e in ref_evidence+candidate_evidence]
        l1, l2 = topic(q.text)
        cases.append(Case(case_id=case_id, query=q, original_answer=g.answer,
            original_reference_ids=refs, original_required_groups=g.required_groups, original_gold=g,
            original_gold_hash=digest(g.model_dump()), topic_level_1_candidate=l1, topic_level_2_candidate=l2,
            operation_candidate=op, target=ContextTarget(issuer_ids=sorted({"cik:"+bindings[r]["cik"] for r in refs}),
                public_period_candidates=question_periods(q.text),
                scope_candidates=sorted({c.value for c in cues if c.field=="scope"}),
                requested_metric_candidates=sorted(terms)), reference_evidence_ids=ref_evidence,
            candidate_evidence_ids=candidate_evidence, cues=cues, table_rows=rows,
            source_statement_span_ids=statement_ids,
            calculation_candidates=calculations(q.text, op, rows, all_spans), rationale_steps=steps,
            compatibility_rules=[CompatibilityRule(rule_id=dimension, dimension=dimension,
                requirement=requirement, evidence_ids=ref_evidence, decision="UNKNOWN",
                rationale="Question-operation-relative decision pending independent source review.")
                for dimension, requirement in dimensions.items()], support_groups=g.required_groups,
            support_status="provided_reference_proxy", evidence_decisions=decisions,
            annotation_status="silver", author_method="deterministic_rule_draft", review_record=None,
            limitations=["Rule draft is not an expert-crafted gold rationale.",
                "Topic labels are candidates, not gold provider selections.",
                "Candidate documents may answer part of the question; no nonanswer label has been asserted.",
                "Numeric calculations are conditional on unreviewed header/cell alignment; do not prove semantic correctness."]))
    return Bundle(framework_sources=FRAMEWORK, input_hashes={**input_hashes, **verified},
        source_dataset_hash=pool.dataset_hash, cohort_dataset_hash=cohort.dataset_hash,
        documents=[docs[r] for r in sorted(used)], hierarchy=hierarchy,
        spans=sorted(all_spans.values(), key=lambda s:s.span_id), cases=cases)
