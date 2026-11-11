from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..models import digest
from .compiler import file_hash
from .lifecycle import pending_review, verify_origins


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x", encoding="utf-8") as stream:
        os.chmod(path, 0o600)
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def write_table(root, table, rows, run_id):
    date = datetime.now(timezone.utc).date().isoformat()
    common = ("experiment_run_id","query_id","case_id","span_id","evidence_id","document_id",
              "parent_sample_id","evaluation_id","condition_id","graph_snapshot_id","trace_id","strategy")
    records = []
    for row in rows:
        row = {"experiment_run_id":run_id, **row}
        records.append({**{k:str(row[k]) if row.get(k) is not None else None for k in common},
                        "payload_json":json.dumps(row,ensure_ascii=False,sort_keys=True)})
    path = Path(root)/"parquet/experiment=context_frame"/f"date={date}"/(table+".parquet")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.exists():
        raise FileExistsError(path)
    schema = pa.schema([(k,pa.string()) for k in (*common,"payload_json")])
    pq.write_table(pa.Table.from_pylist(records,schema=schema),path)
    actual = pq.ParquetFile(path).read().to_pylist()
    if actual!=records:
        raise ValueError("context Parquet readback mismatch")
    return len(actual)


def escaped(text):
    # Numeric entities keep exact source whitespace inside pre without git trailing whitespace.
    return html.escape(text).replace("\t","&#9;").replace("\r","&#13;").replace("\n","&#10;")


def review_html(bundle):
    spans = {s.span_id:s for s in bundle.spans}
    docs = {d.evidence_id:d for d in bundle.documents}
    dimensions={"identity":"기업 동일성","period":"기간","organizational_scope":"연결·별도·사업부 범위",
        "unit":"단위·통화","definition":"지표 정의","modality":"계획·실적·의무","composition":"근거 결합"}
    pieces = ["<!doctype html><html lang='ko'><meta charset='utf-8'><title>FinDER 맥락 주석 검토</title>",
        "<style>body{max-width:1100px;margin:2em auto;padding:1em;font:16px/1.6 sans-serif;color:#17334a}"
        "article{border-top:3px solid #b7cbd7;margin:3em 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f1f5f7;padding:1em}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #bccbd5;padding:.5em;text-align:left}"
        ".badge{background:#fff0d0;padding:.3em}.ids{font-size:12px;overflow-wrap:anywhere}summary{cursor:pointer}</style>",
        "<h1>FinDER 맥락·근거·연산 확장 주석</h1><p>GraphRAG-Bench의 hierarchy / gold rationale 프레임을 적용한 검토 자료입니다. "
        "자동 초안은 silver입니다. Source 일치가 semantic 충분성이나 독립 검수를 뜻하지 않습니다. "
        "원래 정답의 수치와 해석도 각각 원문으로 검토합니다. 다른 유효한 근거 조합과 풀이도 허용합니다.</p>"]
    for c in bundle.cases:
        pieces += [f"<article id='{escaped(c.query.query_id)}'><h2>{escaped(c.query.text)}</h2>",
            f"<p class='badge'>{escaped(c.annotation_status)} · {escaped(c.operation_candidate)} · "
            f"{escaped(c.topic_level_1_candidate)} / {escaped(c.topic_level_2_candidate)}</p>",
            f"<p class='ids'>Query: {escaped(c.query.query_id)} | Case: {c.case_id}</p>",
            "<details><summary>원래 FinDER 정답 — 변경하지 않음</summary><pre>"+escaped(c.original_answer)+"</pre></details>",
            "<h3>질문·대상 범위 초안</h3><pre>"+escaped(json.dumps(c.target.model_dump(),ensure_ascii=False,indent=2))+"</pre>",
            "<h3>필요 근거와 판단 절차</h3><ol>"]
        for step in c.rationale_steps:
            pieces.append("<li><b>"+escaped(step.operation)+"</b>: "+escaped(step.instruction)+
                          ("<pre>"+escaped(step.formula)+"</pre>" if step.formula else "")+"</li>")
        pieces += ["</ol><h3>함께 사용할 조건 — 결정은 검토 대기</h3><table><tr><th>차원</th><th>확인할 내용</th><th>판정</th></tr>"]
        for rule in c.compatibility_rules:
            pieces.append("<tr><td>"+escaped(dimensions[rule.dimension])+"</td><td>"+escaped(rule.requirement)+"</td><td>"+rule.decision+"</td></tr>")
        pieces += ["</table><h3>원문 표의 행·값 후보</h3><p>값의 순서·음수 표시는 보존했습니다. 연도와 값의 연결, 단위 적용은 검토 전입니다.</p>",
            "<table><tr><th>행</th><th>연도 헤더 후보</th><th>원문 값</th></tr>"]
        for row in c.table_rows:
            pieces.append("<tr><td>"+escaped(spans[row.label_span_id].quote)+"</td><td>"+escaped(" | ".join(row.year_candidates))+
                          "</td><td>"+escaped(" | ".join(row.values))+"</td></tr>")
        pieces.append("</table>")
        if c.source_statement_span_ids:
            pieces.append("<h3>원문 서술 후보 — 질문에 대한 충분성은 검토 대기</h3>")
            for sid in c.source_statement_span_ids:
                pieces.append("<details><summary>"+escaped(spans[sid].document_id)+"</summary><p class='ids'>Span: "+sid+
                    "</p><pre>"+escaped(spans[sid].quote)+"</pre></details>")
        if c.calculation_candidates:
            pieces.append("<h3>조건부 산술 검사</h3><p>표의 행 의미·기간·범위·단위·열 대응이 질문과 맞는다는 조건에서 계산한 후보입니다. 검수된 정답 판정이 아닙니다.</p><ul>")
            for calc in c.calculation_candidates:
                pieces.append("<li>"+escaped(f"{calc.operation}: {calc.newer_period}({spans[calc.newer_span_id].quote}) / "
                    f"{calc.older_period}({spans[calc.older_span_id].quote}) → {calc.result}")+"</li>")
            pieces.append("</ul>")
        for title, ids in (("FinDER 제공 참조",c.reference_evidence_ids),("비교 후보 — 무관한 근거로 확정하지 않음",c.candidate_evidence_ids)):
            pieces.append("<h3>"+title+"</h3>")
            for eid in ids:
                d=docs[eid]
                pieces.append("<details><summary>"+escaped(d.source_publisher or "UNKNOWN")+" · "+escaped(d.document_id)+
                    "</summary><p class='ids'>Evidence: "+eid+"</p><pre>"+escaped(d.text)+"</pre></details>")
        pieces.append("</article>")
    return "\n".join(pieces)+"\n</html>\n"


def export_bundle(output,bundle,cohort,selection,*,run_id):
    checks = verify_origins(bundle,cohort)
    write_json(output/"cohort.dataset.private.json",cohort.model_dump())
    write_json(output/"frame.private.json",bundle.model_dump())
    write_json(output/"public/queries.json",[q.model_dump() for q in cohort.queries])
    write_json(output/"review.pending.json",pending_review(bundle).model_dump())
    write_json(output/"selection.json",selection)
    with (output/"review.html").open("x",encoding="utf-8") as stream:
        stream.write(review_html(bundle))
    counts = {}
    tables = {
        "context_cases":[c.model_dump()|{"query_id":c.query.query_id} for c in bundle.cases],
        "source_hierarchy":[h.model_dump() for h in bundle.hierarchy],
        "source_spans":[s.model_dump() for s in bundle.spans],
        "reasoning_obligations":[s.model_dump()|{"query_id":c.query.query_id,"case_id":c.case_id} for c in bundle.cases for s in c.rationale_steps],
        "compatibility_rules":[s.model_dump()|{"query_id":c.query.query_id,"case_id":c.case_id} for c in bundle.cases for s in c.compatibility_rules]}
    for name, rows in tables.items():
        counts[name]=write_table(output,name,rows,run_id)
    summary={"version":bundle.version,"bundle_hash":bundle.bundle_hash,"checks":checks,
        "queries":len(bundle.cases),"source_documents":len(bundle.documents),
        "issuers":len({h.issuer_id for h in bundle.hierarchy}),
        "reference_issuers":len({h.issuer_id for h in bundle.hierarchy
            if h.document_id in {r for c in bundle.cases for r in c.original_reference_ids}}),
        "native_reference_documents":len({r for c in bundle.cases for r in c.original_reference_ids}),
        "conditional_calculations":sum(len(c.calculation_candidates) for c in bundle.cases),
        "silver_cases":sum(c.annotation_status=="silver" for c in bundle.cases),
        "reviewed_gold_cases":sum(c.annotation_status=="reviewed_gold" for c in bundle.cases),
        "http_calls":0,"parquet_readback":counts,
        "limitation":"Independent semantic/identity judgments and evaluator calibration are pending; original native QA/reference gold remains usable."}
    write_json(output/"summary.json",summary)
    return summary


def inventory(output):
    write_json(output/"artifact-sha256.json",{str(p.relative_to(output)):file_hash(p)
        for p in sorted(output.rglob("*")) if p.is_file()})
