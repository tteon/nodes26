"""Native FinDER import. Published answers stay in evaluator-only GoldEvidence."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from ..models import digest
from .contracts import Assertion, Dataset, DistractorLabel, Document, Entity, GoldEvidence, Query, text_hash, tokens

LEXEMES = {"MENTIONS_REVENUE": r"\brevenue[s]?\b", "MENTIONS_AGREEMENT": r"\b(?:agreement|contract)[s]?\b",
    "MENTIONS_EMPLOYEE": r"\bemployee[s]?\b", "MENTIONS_RISK": r"\brisk[s]?\b",
    "MENTIONS_CAPITAL": r"\bcapital\b"}
RELATIONS = ["PUBLISHED_PASSAGE", *LEXEMES]
NORMAL = re.compile(r"[A-Za-z]+|\d+(?:[,.]\d+)*|[%$€£]")


def normalized(text):
    return " ".join(m.group().lower() for m in NORMAL.finditer(text))


def source_domain(text):
    if re.search(r"\blitigation\b|\bcontract\b|\bagreement\b|collective bargaining|legal proceeding", text, re.I):
        return "legal_governance"
    if re.search(r"\brevenue\b|\bincome\b|\bcash\b|\bearnings\b|\basset|\bdilut", text, re.I):
        return "finance_reporting"
    return "business_risk"


def import_finder(public_path, gold_path, reference_path, metadata_path, filing_root, query_ids):
    paths = list(map(Path, (public_path, gold_path, reference_path, metadata_path)))
    public, gold, refs = ({r["id"]: r for r in map(json.loads, p.open())} for p in paths[:3])
    metadata = json.loads(paths[3].read_text())
    registry_path = Path(filing_root) / "source-registry.json"
    filings = {f["accession"]: f for f in json.loads(registry_path.read_text())["documents"]}
    entities = {}
    for rid, row in metadata.items():
        eid = "cik:" + row["cik"]
        aliases = sorted(set(row["issuer_aliases"]))
        if eid in entities:
            aliases = sorted(set(entities[eid].aliases + aliases))
        entities[eid] = Entity(entity_id=eid, canonical_name=filings[row["accession"]]["company"],
                              aliases=aliases, identity_status="source_bound_issuer")
    compiled = {e: re.compile(r"(?<!\w)(?:" + "|".join(re.escape(a) for a in sorted(v.aliases, key=len, reverse=True)
                if len(a) >= 3) + r")(?!\w)", re.I) for e, v in entities.items()}
    cache, docs, assertions = {}, [], []
    for rid, m in sorted(metadata.items()):
        r, f = refs[rid], filings[m["accession"]]
        if text_hash(r["text"]) != r["sha256"]:
            raise ValueError("FinDER reference hash mismatch")
        if f["accession"] not in cache:
            raw = (Path(filing_root) / f["text_path"]).read_bytes()
            if hashlib.sha256(raw).hexdigest() != m["filing_sha256"] or m["filing_sha256"] != f["text_sha256"]:
                raise ValueError("pinned filing hash mismatch")
            cache[f["accession"]] = normalized(raw.decode())
        if normalized(r["text"]) not in cache[f["accession"]]:
            raise ValueError("reference does not bind to pinned full filing")
        eid = "cik:" + m["cik"]
        version = r["sha256"]
        evidence_id = digest([rid, version, 0, len(r["text"])])
        domain = source_domain(r["text"])
        found = [(predicate, next(re.finditer(pattern, r["text"], re.I), None)) for predicate, pattern in LEXEMES.items()]
        relations = {"PUBLISHED_PASSAGE": 1, **{p: 1 for p, match in found if match}}
        doc = Document(document_id=rid, document_version=version, source_publisher=f["company"],
            source_uri=f["url"], text=r["text"], content_hash=version, domain_id=domain, evidence_id=evidence_id,
            entity_mentions={e: len(rx.findall(r["text"])) for e, rx in compiled.items() if rx.search(r["text"])},
            relation_counts=relations, token_count=len(tokens(r["text"])), temporal_scope=None, business_scope=None,
            metadata_status="full-reference token match and pinned filing hash; period/scope and business-fact meaning unreviewed")
        docs.append(doc)
        spans = [("PUBLISHED_PASSAGE", 0, len(doc.text), doc.document_id)] + [
            (p, match.start(), match.end(), match.group()) for p, match in found if match]
        for predicate, start, end, value in spans:
            payload = dict(subject_id=eid, predicate=predicate, object_value=value, document_id=rid,
                evidence_id=evidence_id, source_hash=version, start=start, end=end, quote=doc.text[start:end],
                assertion_kind="source_metadata", validation_status="mechanical_source", domain_id=domain,
                temporal_scope=None, business_scope=None, unit=None, review_record=None)
            assertions.append(Assertion(assertion_id=digest(payload), **payload))
    by_doc = {d.document_id: d for d in docs}
    queries, labels, distractors = [], [], []
    for qid in query_ids:
        q, g = public[qid], gold[qid]
        if set(q) != {"id", "question"} or q["question"] != g["question"] or not g["answer"].strip():
            raise ValueError("native public/gold query contract violated")
        if not set(g["source_ids"]) <= by_doc.keys():
            raise ValueError("pilot gold not fully bound; choose a source-verified cohort")
        queries.append(Query(query_id=qid, text=q["question"]))
        companies = {metadata[r]["cik"] for r in g["source_ids"]}
        labels.append(GoldEvidence(query_id=qid, answer=g["answer"],
            required_groups=[[[by_doc[r].evidence_id]] for r in sorted(set(g["source_ids"]))],
            sufficient_set_status="provided_reference_proxy",
            relevance={by_doc[r].evidence_id: 1.0 for r in set(g["source_ids"])},
            company_id=next(iter(companies)) if len(companies) == 1 else None, category=g["category"]))
        for doc in docs:
            if doc.document_id not in set(g["source_ids"]):
                distractors.append(DistractorLabel(query_id=qid, document_id=doc.document_id,
                    kind="lexical" if set(tokens(q["question"])) & set(tokens(doc.text)) else "relational",
                    status="candidate_not_gold"))
    return Dataset(version="finder-source-bound-pilot-v1", queries=queries, gold=labels,
        entities=sorted(entities.values(), key=lambda e: e.entity_id), documents=docs, assertions=assertions,
        distractors=distractors, ontology_version="ds01-source-ontology-v1", ontology_relations=RELATIONS,
        ontology_kind="source_metadata_relations_not_reviewed_business_facts",
        input_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths + [registry_path]},
        annotation_status="source_bound; distractor nonanswer/identity/context semantics pending independent review")
