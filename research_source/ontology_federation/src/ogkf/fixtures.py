"""Public synthetic development corpus; generated labels are NOT independent gold."""
from __future__ import annotations

import hashlib

from .extraction import normalize
from .identity import ContextEntity, IdentityMapping, Snapshot
from .metrics import IdentityPair, QuestionGold, StudyGold, VERSION
from .models import ExtractedFact, Source, digest
from .registry import ProviderMetadata, PublicQuestion, Registry

SCENARIOS = ["same_identity_same_context", "same_identity_different_role", "same_name_different_identity",
             "provider_conflict", "time_dependent", "legal_operational_definition"]


def development_study():
    sources, entities, assertions, pairs, mappings, questions, labels = [], [], [], [], [], [], []
    conditions = {"ontology": "rich", "condition_id": "frozen-fixture-v1"}
    for company in range(10):
        issuer = f"urn:synthetic:issuer:{company}"
        by_domain, entity_ids = {}, {}
        for domain, predicate, category in [("finance", "financial_obligation", "financial"),
                                            ("legal", "contractual_enforceability", "contract"),
                                            ("sustainability", "emission_commitment", "carbon")]:
            scope = {"finance": "consolidated reporting group", "legal": "contracting legal entity",
                     "sustainability": "operational emissions boundary"}[domain]
            entity = ContextEntity(entity_id=digest([issuer,domain]), canonical_identifier=issuer,
                canonical_name=f"ExampleCo {company}", entity_type="Organization", domain_id=domain,
                provider_id="synthetic-"+domain, business_scope=scope)
            entities.append(entity)
            entity_ids[domain] = entity.entity_id
            specs = [(predicate, str(company+10), "2025", "unitless"),
                     (predicate, str(company+9), "2024", "unitless"),
                     ("reports_revenue" if domain=="finance" else "governing_law" if domain=="legal" else "emissions_total",
                      str(100+company), "2025", "USD million" if domain=="finance" else "code" if domain=="legal" else "tCO2e"),
                     ("reporting_boundary", scope, "2025", None)]
            lines = [f"ExampleCo {company} | {p} | {value} | period={period} | scope={scope} | unit={unit}."
                     for p,value,period,unit in specs]
            text = "\n".join(lines)
            source = Source(document_id=f"synthetic-{company}-{domain}", document_version="v1",
                company_id=issuer, source_publisher="synthetic-"+domain,
                source_uri=f"urn:ogkf:fixture:{company}:{domain}:v1", section_id=domain,
                source_timestamp="2026-01-01", domain_id=domain, text=text,
                content_hash=hashlib.sha256(text.encode()).hexdigest())
            sources.append(source)
            cursor, by_domain[domain] = 0, []
            for index, ((pred,value,period,unit), line) in enumerate(zip(specs,lines)):
                raw = ExtractedFact(subject=entity.canonical_name, subject_type="Organization", predicate=pred,
                    literal_value=value, domain_id=domain, temporal_scope=period, business_scope=scope, unit=unit,
                    table_headers=["entity","predicate","value","period","scope","unit"],
                    evidence_start=cursor, evidence_end=cursor+len(line), evidence_span=line)
                fact = normalize(raw,source,"synthetic-frozen-v1",conditions,digest([issuer,domain,index]))
                fact.update(subject_id=entity.entity_id, extraction_method="synthetic_fixture_authoring",
                            assertion_kind="source_assertion", semantic_support="synthetic_expected")
                fact["fact_id"] = digest([source.document_id,index,fact["evidence_id"]])
                assertions.append(fact)
                if period=="2025":
                    by_domain[domain].append(fact)
                cursor += len(line)+1
            if company==0 and domain=="legal":
                other = entity.model_copy(update={"entity_id":digest([issuer,domain,"provider-b"]),"provider_id":"synthetic-legal-b"})
                entities.append(other)
                quote=f"ExampleCo 0 | {predicate} | disputed | period=2025 | scope={scope} | unit=unitless."
                other_source=source.model_copy(update={"document_id":"synthetic-0-legal-b","source_publisher":"synthetic-legal-b",
                    "text":quote,"content_hash":hashlib.sha256(quote.encode()).hexdigest(),"source_uri":"urn:ogkf:fixture:0:legal-b:v1"})
                sources.append(other_source)
                raw=ExtractedFact(subject=entity.canonical_name,subject_type="Organization",predicate=predicate,
                    literal_value="disputed",domain_id=domain,temporal_scope="2025",business_scope=scope,unit="unitless",
                    table_headers=[],evidence_start=0,evidence_end=len(quote),evidence_span=quote)
                f=normalize(raw,other_source,"synthetic-frozen-v1",conditions,"provider-b")
                f.update(subject_id=other.entity_id,extraction_method="synthetic_fixture_authoring",assertion_kind="source_assertion",
                         semantic_support="synthetic_expected")
                assertions.append(f)
                by_domain[domain].append(f)
                pairs.append(IdentityPair(left_entity_id=entity.entity_id,right_entity_id=other.entity_id,same_identity=True,
                    context_compatible=True,requires_identity_link=True,review_status="reviewed",scenario="provider_conflict",
                    rationale="Synthetic separate publishers disagree on the same contextual predicate and period"))
                mappings.append(IdentityMapping(left_entity_id=entity.entity_id,right_entity_id=other.entity_id,
                    mapping_type="SAME_AS",confidence=1,evidence_ids=[f["evidence_id"]],review_status="reviewed"))
        for left,right,scenario in [("finance","legal","same_identity_different_role"),
                                    ("legal","sustainability","legal_operational_definition")]:
            pair = IdentityPair(left_entity_id=entity_ids[left],right_entity_id=entity_ids[right],same_identity=True,
                context_compatible=False,requires_identity_link=True,review_status="reviewed",scenario=scenario,
                rationale="Synthetic same legal identity, different reporting/operational context")
            pairs.append(pair)
            mappings.append(IdentityMapping(left_entity_id=pair.left_entity_id,right_entity_id=pair.right_entity_id,
                mapping_type="SAME_AS",confidence=1,evidence_ids=[by_domain[left][0]["evidence_id"]],review_status="reviewed"))
        # Name homonyms must remain distinct even when strings match.
        if company:
            previous = next(e for e in entities if e.canonical_identifier=="urn:synthetic:issuer:0" and e.domain_id=="finance")
            current = next(e for e in entities if e.entity_id==entity_ids["finance"])
            current.canonical_name = previous.canonical_name
            pairs.append(IdentityPair(left_entity_id=previous.entity_id,right_entity_id=current.entity_id,
                same_identity=False,context_compatible=False,requires_identity_link=False,review_status="reviewed",
                scenario="same_name_different_identity",rationale="Synthetic homonyms have distinct canonical identifiers"))
        # Public questions carry no categories, domains or supporting evidence labels.
        definitions = [
            ("single", f"What is the financial obligation for ExampleCo {company} in 2025?", ["finance"],
             [f for f in by_domain["finance"] if f["predicate"]=="financial_obligation"], "single_domain"),
            ("cross", f"Verify the carbon commitment, financial obligation and contract enforceability for ExampleCo {company} in 2025.",
             ["finance","legal","sustainability"], [f for group in by_domain.values() for f in group if f["predicate"] in
                 {"financial_obligation","contractual_enforceability","emission_commitment"}], "conflicting_evidence" if company==0 else "cross_domain"),
            ("legal", f"What is the contract enforceability for ExampleCo {company} in 2025?", ["legal"],
             [f for f in by_domain["legal"] if f["predicate"]=="contractual_enforceability"], "conflicting_evidence" if company==0 else "single_domain"),
            ("ood", f"What is the lunar mineral inventory for ExampleCo {company} in 2025?", [], [], "out_of_domain")]
        for suffix,text,domains,facts,category in definitions:
            qid=f"fixture-{company}-{suffix}"
            questions.append(PublicQuestion(question_id=qid,text=text,canonical_identifier=issuer,temporal_scope="2025"))
            labels.append(QuestionGold(question_id=qid,company_id=issuer,category=category,relevant_domains=domains,
                supporting_assertion_groups=[[f["fact_id"]] for f in facts],
                acceptable_assertion_sets=[[f["fact_id"] for f in facts]] if facts else [],
                expected_status="conflicting" if category=="conflicting_evidence" else "supported" if facts else "abstain",
                annotation_status="reviewed"))
    # Same-context identity alias under a distinct provider: retain representation,
    # link identity explicitly; no assertion is copied or synthesized into a DB.
    original = entities[0]
    alias = original.model_copy(update={"entity_id": digest([original.entity_id,"alias"]), "provider_id":"synthetic-alias"})
    entities.append(alias)
    pairs.append(IdentityPair(left_entity_id=original.entity_id,right_entity_id=alias.entity_id,same_identity=True,
        context_compatible=True,requires_identity_link=True,review_status="reviewed",scenario="same_identity_same_context",
        rationale="Synthetic two provider representations of one contextual entity"))
    mappings.append(IdentityMapping(left_entity_id=original.entity_id,right_entity_id=alias.entity_id,
        mapping_type="SAME_AS",confidence=1,evidence_ids=[assertions[0]["evidence_id"]],review_status="reviewed"))
    pairs.extend([IdentityPair(left_entity_id=entities[1].entity_id,right_entity_id=entities[2].entity_id,
        same_identity=True,context_compatible=True,requires_identity_link=True,review_status="reviewed",scenario=scenario,
        rationale="Synthetic assertions preserve provider disagreement and temporal qualification")
        for scenario in ("provider_conflict","time_dependent")])
    snapshot=Snapshot(version="synthetic-frozen-v1",origin="synthetic_fixture",sources=sources,entities=entities,
                      assertions=assertions,mappings=mappings)
    gold=StudyGold(version=VERSION,origin="synthetic_fixture",independent_review_complete=False,identity_pairs=pairs,
        context_qualified_assertion_ids=[f["fact_id"] for f in assertions],questions=labels)
    registry=Registry(version="registry-v1",entries=[ProviderMetadata(provider_id="synthetic-"+domain,domain_id=domain,
        database_name="fixture"+domain,ontology_version="domain-ontology-v1/"+domain,
        supported_entity_types=["Organization"],supported_relation_types=relations,supported_question_categories=categories,
        available_time_range=("2024","2025"),source_document_types=["synthetic_report"],
        provenance_policy="qualified_source_assertions",schema_summary="OGKFFact -> evidence -> versioned source",
        database_capabilities={"fixed_read_template":True,"native_federation":"unverified"})
        for domain,relations,categories in [
            ("finance",["financial_obligation","reports_revenue","reporting_boundary"],["financial","revenue"]),
            ("legal",["contractual_enforceability","governing_law","reporting_boundary"],["contract","legal"]),
            ("sustainability",["emission_commitment","emissions_total","reporting_boundary"],["carbon","emissions"])]])
    return snapshot,registry,questions,gold
