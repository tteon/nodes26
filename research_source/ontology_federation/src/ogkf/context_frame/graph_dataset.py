"""Full FinDER-derived graph-question candidates and simple matched controls."""

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import zipfile

from ..ds01.contracts import Query
from ..models import digest
from .challenge import jsonl, load_snapshot, save

VERSION = "finder-ontology-alpha-candidates-v2"


def expand(dataset, questions, span_records):
    assertions = {a.assertion_id: a for a in dataset.assertions}
    spans = {s["span_id"]: s for s in span_records}
    native = {q["query_id"]: q for q in questions if q["origin"] == "native_unchanged"}
    controls = []
    for q in questions:
        if q["origin"] != "generated_table_context_candidate":
            continue
        a = assertions[q["assertion_ids"][0]]
        row = a.table_row
        if not row or len(row["values"]) != len(row["year_candidates"]):
            raise ValueError("invalid table witness")
        label = spans[row["label_span_id"]]["quote"]
        year = row["year_candidates"][0]
        parent = native[q["parent_ids"][0]]
        text = f"Use the entity and reporting context of this inquiry: {parent['text']} Now retrieve the source table's '{label}' observation for {year}. State its unit and reporting scope if established; otherwise identify the missing qualification. No difference calculation is requested."
        controls.append(
            {
                "query_id": "simple-" + digest([VERSION, q["query_id"], text])[:24],
                "text": text,
                "origin": "generated_simple_lookup_control",
                "parent_ids": q["parent_ids"],
                "matched_comparison_id": q["query_id"],
                "source_ids": q["source_ids"],
                "assertion_ids": q["assertion_ids"],
                "operation_candidate": "single_cell_lookup_with_qualification",
                "annotation_status": "silver",
                "semantic_answer": None,
                "sufficient_evidence_gold": None,
                "graph_necessity": "unverified",
                "parser_value_candidate": row["values"][0],
                "review_reason": "Matched to comparison case; parser year/value alignment and entity/unit/scope remain unreviewed, no automatic semantic gold.",
            }
        )
    return questions + controls


def validate(rows, public):
    ids = [q["query_id"] for q in rows]
    if len(ids) != len(set(ids)) or len(public) != len(rows):
        raise ValueError("duplicate or missing question")
    by_id = {q["query_id"]: q for q in rows}
    for q, p in zip(rows, public, strict=True):
        if (
            set(p) != {"query_id", "text"}
            or p != Query(query_id=q["query_id"], text=q["text"]).model_dump()
        ):
            raise ValueError("private labels leaked to agent")
        if any(
            parent not in by_id or by_id[parent]["origin"] != "native_unchanged"
            for parent in q["parent_ids"]
        ):
            raise ValueError("missing native parent")
        if q["origin"] != "native_unchanged" and (
            q.get("annotation_status") != "silver"
            or q.get("semantic_answer") is not None
        ):
            raise ValueError("unreviewed case promoted to gold")
    return {
        "unique_questions": len(rows),
        "public_fields_only": True,
        "all_parents_preserved": True,
        "no_generated_semantic_gold": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    ds, qs, sha = load_snapshot(args.snapshot)
    spans = [
        json.loads(s) for s in (args.snapshot / "spans.jsonl").read_text().splitlines()
    ]
    rows = expand(ds, qs, spans)
    public = [Query(query_id=q["query_id"], text=q["text"]).model_dump() for q in rows]
    checks = validate(rows, public)
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    jsonl(out / "questions.agent.jsonl", public)
    jsonl(out / "questions.evaluator-only.jsonl", rows)
    generated = [q for q in rows if q["origin"] != "native_unchanged"]
    contracts = {
        "CTX-ENTITY": "A shared publisher or name does not establish assertion-subject identity.",
        "CTX-TIME": "Attach values to the stated calendar period; compare distinct periods without treating them as the same observation.",
        "CTX-SCOPE-UNIT": "Compose operands only when entity, scope, definition and units are compatible; preserve missing qualifiers.",
        "CTX-MODALITY": "Keep estimates, commitments, realized outcomes and legal duties distinct.",
        "CTX-COMPOSE": "Cite every necessary assertion and expose the formula or join; shared issuer identity does not imply causality.",
    }
    save(out / "context-contracts.json", contracts)
    review = [
        {
            "query_id": q["query_id"],
            "parent_ids": q["parent_ids"],
            "source_ids": q["source_ids"],
            "assertion_ids": q["assertion_ids"],
            "context_contract_ids": sorted(contracts),
            "ontology_alignment_status": "UNKNOWN",
            "targets": {
                k: "UNKNOWN"
                for k in (
                    "source_support",
                    "identity_scope",
                    "period_unit_alignment",
                    "permitted_composition",
                    "sufficient_evidence",
                    "expected_action",
                    "graph_necessity",
                )
            },
            "native_gold_unchanged": True,
            "independent_reviews": [],
            "adjudication": None,
        }
        for q in generated
    ]
    jsonl(out / "review.pending.jsonl", review)
    manifest = {
        "version": VERSION,
        "snapshot_hash": sha,
        "source_snapshot": str(args.snapshot),
        "native_questions": sum(q["origin"] == "native_unchanged" for q in rows),
        "generated_silver_questions": len(generated),
        "total_questions": len(rows),
        "origins": dict(Counter(q["origin"] for q in rows)),
        "native_category_counts": dict(
            Counter(q["category"] for q in rows if q["origin"] == "native_unchanged")
        ),
        "source_documents": len(ds.documents),
        "source_only_observations": len(ds.assertions),
        "independent_generated_gold": 0,
        "executed_model_cases_in_this_dataset": 0,
        "checks": checks,
        "population_note": "Pinned raw native census5703; nonblank reference-ready native5696. User-mentioned5173 is not verified as a source revision or subset. All corpus has prior development exposure.",
        "dependence": "Controls and variants share native parents/source components; generated count is not independent sample size.",
        "graph_need": "hypothesis only; compare competent text/equal-evidence controls before claiming graph necessity",
        "result_strata": [
            "generic_and_ontology_success",
            "ontology_only_success",
            "generic_only_success",
            "both_fail",
            "unknown_or_annotation_disagreement",
        ],
        "planned_comparison": "Fixed facts/evidence/answer policy/model: G competent generic, F FIBO reference, GX generic+same context rules, FX FIBO+same context rules; later compare retrieval with identical assertions and budgets. Measure token-length confounding.",
        "context_contracts": "context-contracts.json; reviewer obligations, not validated per-case alignment",
        "ontology_alignment": "No automatic FIBO mapping or OWL reasoning; per-case applicability remains UNKNOWN",
        "native_gold": "Original QA/reference gold preserved for native questions; generated graph tasks need separate reviewed answers/compatibility/support/action labels.",
    }
    save(out / "manifest.json", manifest)
    cards = []
    # Searchable reviewer index, not an independent annotation submission.
    for q in generated:
        cards.append(
            "<tr><td>"
            + html.escape(q["query_id"])
            + "</td><td>"
            + html.escape(q["origin"])
            + "</td><td>"
            + html.escape(q["text"])
            + "</td><td>"
            + html.escape(", ".join(q["source_ids"]))
            + "</td><td>UNKNOWN</td></tr>"
        )
    (out / "review.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Graph-aware candidate review</title><style>body{font:16px/1.5 system-ui;margin:30px}td,th{border:1px solid #ccc;padding:10px;vertical-align:top}table{border-collapse:collapse}input{padding:12px;width:70%}</style><h1>Source-grounded graph-question candidates</h1><p>Silver drafts. UNKNOWN semantic labels. Native parents remain unchanged. IDs resolve to source snapshot documents/proposals/spans. This index is not a reviewer submission.</p><input id="filter" placeholder="Search case, family or question"><table><thead><tr><th>ID</th><th>Family</th><th>Question</th><th>Source witness</th><th>Review</th></tr></thead><tbody>'
        + "".join(cards)
        + '</tbody></table><script>document.getElementById("filter").oninput=e=>{const v=e.target.value.toLowerCase();document.querySelectorAll("tbody tr").forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(v));};</script></html>'
    )
    with zipfile.ZipFile(
        out / "questions-and-review.zip", "w", zipfile.ZIP_DEFLATED
    ) as z:
        for name in (
            "context-contracts.json",
            "questions.agent.jsonl",
            "questions.evaluator-only.jsonl",
            "review.pending.jsonl",
            "manifest.json",
            "review.html",
        ):
            z.write(out / name, name)
        z.write(args.snapshot / "documents.jsonl", "sources/documents.jsonl")
        used = {aid for q in generated for aid in q.get("assertion_ids", [])}
        witness = [a for a in ds.assertions if a.assertion_id in used]
        span_ids = {sid for a in witness for sid in a.evidence_span_ids}
        entity_ids = {a.subject_id for a in witness}
        for name, records in (
            ("assertions.witness.jsonl", [a.model_dump() for a in witness]),
            ("spans.witness.jsonl", [r for r in spans if r["span_id"] in span_ids]),
            (
                "entities.witness.jsonl",
                [e.model_dump() for e in ds.entities if e.entity_id in entity_ids],
            ),
        ):
            z.writestr(
                "sources/" + name,
                "".join(
                    json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                    for r in records
                ),
            )
        z.writestr(
            "sources/scope.json",
            json.dumps(
                {
                    "documents": len(ds.documents),
                    "included_assertion_witnesses": len(witness),
                    "source_snapshot_hash": sha,
                    "assertions_are_literal_silver": True,
                    "independent_semantic_gold": False,
                },
                indent=2,
            ),
        )
    save(
        out / "delivery-sha256.json",
        {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in out.iterdir()
            if f.is_file()
        },
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
