"""Offline qualification of separate, source-pinned model-by-ontology studies.

No external service is contacted. Scripted outputs test engineering invariants;
semantic metrics and costs remain unknown. Never reuse an output directory.
"""

import argparse
import html
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..models import digest
from .contracts import Ontology
from .fixtures import synthetic_fixture
from .matrices import (
    ProfileMatrixConfig,
    agent_request,
    extraction_request,
    parse_config,
    queue,
    registry_hash,
    run_mock,
    run_live,
)
from .profiles import ProfileBundle, ProfileReview, load_bundle, sha_bytes


CONTRASTS = [
    {
        "id": "reference_package",
        "weights": {"F": 1, "G": -1},
        "interpretation": "Primary mode includes definition-content differences; shared-definition mode isolates terminology/reference identifiers.",
    },
    {
        "id": "context_given_fibo",
        "weights": {"FX": 1, "F": -1},
        "interpretation": "Added context and relationship guidance at fixed FIBO terms.",
    },
    {
        "id": "context_given_generic",
        "weights": {"GX": 1, "G": -1},
        "interpretation": "The same added guidance at fixed generic terms.",
    },
    {
        "id": "reference_context_interaction",
        "weights": {"FX": 1, "F": -1, "GX": -1, "G": 1},
        "interpretation": "Difference in context-guidance effects; no synergy assumed.",
    },
]


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def protected_hashes(package):
    paths = [
        package / p
        for p in (
            "reports/ds01_campaign",
            "reports/finder_candidates_v1",
            "reports/final_review_v1",
            "reports/research/enterprise_semantics_v1",
            "reports/presentation",
            "reports/presentation_handoff_v1",
            "handoffs/enterprise_semantics_v1",
            "configs/enterprise/matrix_E.json",
            "configs/enterprise/matrix_A.json",
        )
    ]
    files = [
        f for p in paths for f in (p.rglob("*") if p.is_dir() else [p]) if f.is_file()
    ]
    return {
        str(f.relative_to(package)): sha_bytes(f.read_bytes())
        for f in sorted(set(files))
    }


def prompt_audit(config, snapshot, cases, ontology, profiles, output):
    units = (
        [snapshot.documents[0].document_id]
        if config.study == "E"
        else [c.case_id for c in cases]
    )
    q = queue(
        config, units, snapshot.snapshot_id, registry_hash(config, ontology, profiles)
    )
    by_case = {c.case_id: c for c in cases}
    rows = []
    seen_requests = set()
    invariants = {}
    for condition in q:
        if config.study == "E":
            messages, schema = extraction_request(
                condition["unit_id"],
                snapshot,
                ontology,
                condition,
                profiles,
                config.definition_control,
            )
            guidance_key = "ontology_information"
        else:
            messages, schema, _ = agent_request(
                by_case[condition["unit_id"]],
                snapshot,
                ontology,
                condition,
                profiles,
                config.definition_control,
            )
            guidance_key = "relation_semantics"
        payload = json.loads(messages[1]["content"])
        guidance = payload.pop(guidance_key)
        fixed_hash = digest([messages[0], payload, schema])
        key = (condition["unit_id"], condition["prompt_strategy"])
        invariants.setdefault(key, set()).add(fixed_hash)
        request_hash = digest(messages)
        if request_hash not in seen_requests:
            path = output / "requests" / (request_hash + ".json")
            request = {"messages": messages, "schema": schema}
            if path.exists():
                if json.loads(path.read_text()) != request:
                    raise ValueError("request hash collision or altered artifact")
            else:
                save(path, request)
            seen_requests.add(request_hash)
        rows.append(
            {
                **condition,
                "study": config.study,
                "definition_control": config.definition_control,
                "profile_hash": profiles.profile_hash,
                "prompt_hash": request_hash,
                "fixed_input_hash": fixed_hash,
                "output_schema_hash": digest(schema),
                "prompt_utf8_bytes": len(json.dumps(messages).encode()),
                "guidance_utf8_bytes": len(json.dumps(guidance).encode()),
                "provider_input_tokens": None,
                "provider_output_tokens": None,
                "semantic_primary": None,
            }
        )
    if any(len(values) != 1 for values in invariants.values()):
        raise ValueError("ontology arms changed non-ontology inputs")
    if len({r["output_schema_hash"] for r in rows}) != 1:
        raise ValueError("output schema changed across arms")
    return rows


def report_html(protocol, coverage, report):
    def esc(value):
        return html.escape(str(value))

    body = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>FIBO and contextual relationship studies v2</title>
<style>body{font:17px system-ui;max-width:1160px;margin:40px auto;padding:16px;color:#173043}table{border-collapse:collapse;width:100%;margin:18px 0}td,th{border-bottom:1px solid #ccd5dd;padding:10px;text-align:left;vertical-align:top}pre{white-space:pre-wrap}a{color:#165b8c}</style>
<h1>FIBO and contextual relationship studies</h1>
<p>Engineering qualification only. Semantic model effects have not been measured.
FIBO supplies reference concepts. Application-owned contracts define contextual applicability and limits on inference.</p>
<p><a href="protocol.json">Experimental protocol</a> · <a href="summary.json">Offline receipts</a> · <a href="profile-review.html">Independent profile review</a> · <a href="presentation-content.json">English presentation text</a></p>
<h2>Controlled conditions</h2><table><tr><th>Arm</th><th>Reference concepts</th><th>Context and relationship guidance</th></tr>
<tr><td>G</td><td>Competent generic definitions</td><td>Absent</td></tr>
<tr><td>F</td><td>Pinned FIBO definitions</td><td>Absent</td></tr>
<tr><td>G+α (GX)</td><td>Same generic definitions</td><td>Present</td></tr>
<tr><td>F+α (FX)</td><td>Same FIBO definitions</td><td>Same guidance as GX</td></tr></table>
<p>All conditions retain identical source evidence, qualifiers, output schemas, response policy and validation. The primary reference-package contrast includes definition differences. A separate shared-definition terminology control is rendered and audited. UTF-8 bytes are recorded; model token matching is unverified.</p>
<h2>Separate experiments</h2><table><tr><th>Study</th><th>Fixed inputs</th><th>Primary outcome</th></tr>
<tr><td>E: extraction</td><td>Document, identity handles, output schema and postprocessing</td><td>Grounded Fact F1, pending independent fact gold</td></tr>
<tr><td>A: reasoning</td><td>Exactly the same qualified assertions and evidence per question</td><td>Evidence-complete Grounded QA Success, pending claim/context/action gold</td></tr></table>
<p>Initial studies use two model slots and three repeats. E has two prompt strategies and four ontology arms (48 cells per document); A has one shared response policy (24 cells per question). Repeats are not independent companies or documents.</p>
<h2>Routing and enforcement</h2><p>Metadata routing and hard evidence gates are later, separate interventions. Hold graph, executor, answer model and composed evidence interface fixed. Compare a single DB, client-batched multi-DB reads and custom CALL federation fairly. The installed DozerDB build does not qualify native USE/Composite federation. A shared validation policy does not establish source entailment.</p>
<h2>Offline execution</h2><pre>"""
    body += (
        esc(json.dumps(report, indent=2))
        + "</pre><h2>Source coverage</h2><table><tr><th>Concept</th><th>FIBO term</th><th>Module</th><th>Application review</th></tr>"
    )
    for row in coverage:
        body += (
            "<tr>"
            + "".join(
                "<td>" + esc(row[k]) + "</td>"
                for k in (
                    "concept_key",
                    "fibo_id",
                    "source_path",
                    "mapping_review_status",
                )
            )
            + "</tr>"
        )
    body += "</table><h2>Analysis</h2><p>Estimate paired differences and reference × guidance × model interactions with company/shared-source dependence. Report effect sizes, cluster-aware confidence intervals, failures and negative findings. No power or noninferiority claim is qualified without reviewed pilot variance and a frozen margin.</p></html>"
    return body


def prepare(package, output):
    package, output = Path(package).resolve(), Path(output)
    before = protected_hashes(package)
    output.mkdir(parents=True, exist_ok=False)
    ontology = Ontology.model_validate_json(
        (package / "ontology/enterprise_relations.yaml").read_text()
    )
    profiles = load_bundle(package / "ontology/fibo_context_v2/profiles.json")
    snapshot, cases = synthetic_fixture(ontology)
    save(output / "synthetic-snapshot.json", snapshot.model_dump())
    save(output / "synthetic-cases.json", [c.model_dump() for c in cases])
    coverage = [
        {
            **t.model_dump(),
            "mapping_review_status": profiles.mapping_review_status,
            "exact_definitions_equal": t.generic_definitions == t.fibo_definitions,
            "source_definition_verified": True,
        }
        for t in profiles.terms
    ]
    save(output / "source-coverage.json", coverage)
    save(
        output / "profile-review.pending.json",
        {
            "version": "ontology-profile-review-v1",
            "profile_hash": profiles.profile_hash,
            "reviewer_ids": [],
            "submission_hashes": [],
            "adjudication_hash": None,
            "approved_concept_keys": [],
            "approved_guidance_hash": None,
            "source_based_rationale": "",
        },
    )
    review = "<!doctype html><html lang='en'><meta charset='utf-8'><title>Independent profile review</title><h1>Pending independent review</h1><p>No reviewers or approvals have been generated. Review each mapping against its intended data facet; a source-pinned definition does not establish application compatibility.</p>"
    for t in profiles.terms:
        review += (
            "<h2>"
            + html.escape(t.concept_key)
            + "</h2><pre>"
            + html.escape(t.model_dump_json(indent=2))
            + "</pre>"
        )
        review += (
            '<p><a href="https://github.com/edmcouncil/fibo/blob/'
            + profiles.fibo_commit
            + "/"
            + html.escape(t.source_path, quote=True)
            + '">Pinned official source module</a></p>'
        )
    review += (
        "<h2>Application-owned guidance for adjudication</h2><pre>"
        + html.escape(
            json.dumps(
                profiles.prompt_payload("FX", "competent_generic")["context_guidance"],
                indent=2,
            )
        )
        + "</pre></html>"
    )
    (output / "profile-review.html").write_text(review)
    rows, summaries, gates = [], {}, {}
    for study in ("E", "A"):
        config = parse_config(
            (package / f"configs/enterprise/matrix_{study}.v2.json").read_text()
        )
        save(output / f"config_{study}.json", config.model_dump())
        rows.extend(prompt_audit(config, snapshot, cases, ontology, profiles, output))
        summaries[study] = run_mock(
            config,
            snapshot,
            cases,
            ontology,
            output / ("mock_" + study),
            fail_first=True,
            profiles=profiles,
        )
        alias = config.model_copy(update={"definition_control": "shared_definitions"})
        save(output / f"config_{study}.alias-control.json", alias.model_dump())
        rows.extend(prompt_audit(alias, snapshot, cases, ontology, profiles, output))
        # The blocked command must return before constructing an HTTP client.
        gates[study] = run_live(
            config,
            snapshot,
            cases,
            ontology,
            output / ("admission_" + study),
            gold=None,
            calibration=None,
            capabilities=None,
            environ={},
            profiles=profiles,
        )
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                (k, pa.string())
                for k in (
                    "study",
                    "definition_control",
                    "condition_id",
                    "sample_id",
                    "unit_id",
                    "model_slot",
                    "prompt_strategy",
                    "ontology_arm",
                    "profile_hash",
                    "prompt_hash",
                    "fixed_input_hash",
                    "output_schema_hash",
                )
            ]
            + [
                (k, pa.int64())
                for k in (
                    "replicate_id",
                    "prompt_utf8_bytes",
                    "guidance_utf8_bytes",
                    "provider_input_tokens",
                    "provider_output_tokens",
                )
            ]
            + [("semantic_primary", pa.float64())]
        ),
    )
    pq.write_table(table, output / "prompt-audit.parquet")
    if (
        pq.ParquetFile(output / "prompt-audit.parquet").read().to_pylist()
        != table.to_pylist()
    ):
        raise ValueError("prompt audit Parquet readback mismatch")
    protocol = {
        "version": "fibo-context-study-v2",
        "profile_hash": profiles.profile_hash,
        "reference_commit": profiles.fibo_commit,
        "contrasts": CONTRASTS,
        "studies": ["E: conditional fact extraction", "A: fixed-evidence reasoning"],
        "main_definition_control": "competent_generic",
        "secondary_definition_control": "shared_definitions",
        "reference_package_isolation": "F-G includes definition differences; only secondary mode equalizes definition text.",
        "constant": [
            "source facts",
            "identity handles",
            "evidence",
            "qualifiers",
            "output schema",
            "response policy",
            "postprocessing",
            "retry policy",
        ],
        "future_separate_interventions": [
            "metadata routing",
            "hard evidence gate",
            "placement",
            "batched vs per-context synthesis",
        ],
        "analysis_unit": "company/shared-source component with paired document/question and model-repeat structure",
        "effects": None,
        "confidence_intervals": None,
        "independent_semantic_gold": 0,
        "reasoner": "none",
        "owl_entailment_or_consistency_claimed": False,
        "length_control": "UTF-8 bytes measured; provider token matching remains unverified",
    }
    save(output / "protocol.json", protocol)
    for cls in (ProfileBundle, ProfileReview, ProfileMatrixConfig):
        save(output / "schemas" / (cls.__name__ + ".json"), cls.model_json_schema())
    after = protected_hashes(package)
    if before != after:
        raise ValueError("existing frozen research artifacts changed")
    save(
        output / "frozen-preservation.json",
        {
            "files": len(before),
            "before_hash": digest(before),
            "after_hash": digest(after),
            "unchanged": True,
        },
    )
    report = {
        "version": "fibo-context-offline-v2",
        "ontology_profile_hash": profiles.profile_hash,
        "source_terms": len(profiles.terms),
        "source_modules": len({t.source_path for t in profiles.terms}),
        "prompt_audit_rows": len(rows),
        "definition_controls": ["competent_generic", "shared_definitions"],
        "mock_E_logical_calls": summaries["E"]["logical_calls"],
        "mock_A_logical_calls": summaries["A"]["logical_calls"],
        "scripted_attempts": sum(s["mock_attempts"] for s in summaries.values()),
        "http_attempts": 0,
        "graph_calls": 0,
        "semantic_primary": None,
        "model_understanding_measured": False,
        "provider_usage": None,
        "arm_dependent_enforcement": False,
        "admission": gates,
        "frozen_unchanged": True,
    }
    save(output / "summary.json", report)
    save(
        output / "presentation-content.json",
        {
            "language": "en",
            "status": "design_and_engineering_receipts",
            "slides": [
                {
                    "title": "FIBO and enterprise context",
                    "text": "FIBO supplies common reference concepts. Application-owned relationship contracts specify contextual applicability and inference limits.",
                    "status": "proposed_approach",
                },
                {
                    "title": "Four controlled ontology conditions",
                    "table": [
                        ["G", "Generic definitions"],
                        ["F", "FIBO definitions"],
                        ["G+α", "Generic + contextual contracts"],
                        ["F+α", "FIBO + the same contracts"],
                    ],
                    "status": "registered_design",
                },
                {
                    "title": "Separate extraction and reasoning studies",
                    "text": "E fixes documents and output schemas. A fixes qualified assertions and evidence. Both cross models with ontology profiles; independent gold is required for semantic outcomes.",
                    "status": "registered_design",
                },
                {
                    "title": "Context-composed retrieval",
                    "text": "A guarded custom CALL can read isolated DozerDB databases and return one connected evidence packet. Single-DB and client-batched controls preserve the same assertions and provenance. Native Composite support is not qualified.",
                    "status": "architecture_existing_receipts_separate_from_ontology_effect",
                },
                {
                    "title": "Offline qualification",
                    "text": f"Verified {len(profiles.terms)} source-pinned concepts, fixed non-ontology inputs and typed Parquet traces. Scripted tests make no model-understanding or performance claim.",
                    "status": "observed_engineering_only",
                },
                {
                    "title": "Hypotheses pending independent evaluation",
                    "text": "Measure grounded facts, context-compatible claims, justified actions and sufficient evidence. Report negative findings and conditions where a context-preserving single DB is sufficient.",
                    "status": "unverified_hypotheses",
                },
            ],
        },
    )
    (output / "report.html").write_text(report_html(protocol, coverage, report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.package, args.output), indent=2))


if __name__ == "__main__":
    main()
