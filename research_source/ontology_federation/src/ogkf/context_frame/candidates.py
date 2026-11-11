"""Pinned FinDER census and private category/reasoning/type sampling.

Native labels select annotation candidates, never providers or agent operations.
"""
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import html
import json
from pathlib import Path
import platform
from typing import Literal
import uuid

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import Field, StrictBool

from ..models import Strict, digest
from ..ds01.contracts import Dataset, tokens
from .artifacts import escaped, export_bundle, inventory, write_json
from .compiler import build_bundle, file_hash, operation, select_cohort
from .lifecycle import verify_origins
from .models import Bundle

VERSION = "finder-native-candidates-v1"
CATEGORIES = ("Accounting", "Company overview", "Financials", "Footnotes", "Governance", "Legal", "Risk", "Shareholder return")
TYPE_MAP = {"Subtract":"Subtract", "Subtraction":"Subtract", "Addition":"Addition", "Division":"Division",
            "Multiplication":"Multiplication", "Compositional":"Compositional", "None":"None"}
BASE = Path(__file__).resolve().parents[3]


class ExpansionConfig(Strict):
    version: Literal["finder-native-candidates-v1"] = VERSION
    seed: int
    review_per_category: int = Field(ge=1, le=100)
    reasoning_true_target_per_category: int = Field(ge=0, le=100)
    max_annotation_cases: int = Field(ge=24, le=6000)
    same_issuer_candidates: int = Field(ge=0, le=5)
    other_issuer_candidates: int = Field(ge=0, le=5)
    max_rows_per_case: int = Field(ge=1, le=20)
    expand_all_source_bound: Literal[True] = True
    preserve_annotation_status: Literal["silver"] = "silver"


class Candidate(Strict):
    query_id: str
    question: str
    category: str
    native_reasoning: StrictBool
    native_type_raw: str
    type_canonical_for_analysis: str
    public_operation_candidate: str
    native_operation_hint_for_reviewer: str | None
    source_ids: list[str]
    source_count: int
    reference_chars: int
    has_native_answer: bool
    source_readiness: Literal["source_bound_ready", "partial_filing_binding", "dataset_reference_only"]
    annotation_eligible: bool
    verified_source_issuer_ids: list[str]
    verified_source_filing_ids: list[str]
    shared_reference_component_id: str
    flags: list[str]
    labels_role: Literal["evaluator_only"] = "evaluator_only"


def census(raw, native, public, references, pool, bindings):
    """Verify every original field against native files before using a label."""
    expected = {"_id","text","reasoning","category","references","answer","type"}
    if len({r["_id"] for r in raw}) != len(raw) or {r["_id"] for r in raw} != set(native) or set(native) != set(public):
        raise ValueError("original/native/public ID mismatch")
    bound = {d.document_id for d in pool.documents}
    if not bound <= bindings.keys():
        raise ValueError("source-bound pool lacks filing metadata")
    roots = {qid:qid for qid in native}
    def find(x):
        while roots[x] != x:
            roots[x] = roots[roots[x]]; x = roots[x]
        return x
    seen = {}
    for qid in sorted(native):
        for rid in set(native[qid]["source_ids"]):
            if rid not in references:
                raise ValueError("missing original reference")
            if rid in seen:
                a,b=find(qid),find(seen[rid]);roots[max(a,b)]=min(a,b)
            seen[rid]=qid
    groups=defaultdict(list)
    for qid in sorted(native):groups[find(qid)].append(qid)
    group_ids={q:digest([VERSION,"shared-reference",members]) for members in groups.values() for q in members}
    result=[]
    for original in raw:
        if set(original)!=expected or type(original["reasoning"]) is not bool or not isinstance(original["type"],str):
            raise ValueError("unexpected original Reasoning/Type/schema")
        qid=original["_id"]; n=native[qid]; p=public[qid]
        if (set(p)!={"id","question"} or p["question"]!=original["text"] or n["question"]!=original["text"]
                or n["answer"]!=original["answer"] or n["category"]!=original["category"]
                or n["native_reasoning"] is not original["reasoning"] or n["native_type"]!=original["type"]
                or Counter(original["references"])!=Counter(references[r]["text"] for r in n["source_ids"])):
            raise ValueError("original/native question answer category reasoning type or reference mismatch: "+qid)
        ids=list(n["source_ids"]); unique=set(ids)
        matched=unique & bound
        status="source_bound_ready" if unique and unique<=bound else ("partial_filing_binding" if matched else "dataset_reference_only")
        flags=[]
        if original["reasoning"] and original["type"]=="None":flags.append("reasoning_true_type_unspecified")
        if not original["reasoning"] and original["type"]!="None":flags.append("reasoning_false_type_specified")
        if original["type"]=="Subtraction":flags.append("type_spelling_alias_preserved")
        if len(ids)!=len(unique):flags.append("repeated_reference_id_preserved")
        has_answer=bool(original["answer"].strip())
        if not has_answer:flags.append("blank_native_answer")
        hint={"Subtract":"subtract","Subtraction":"subtract","Addition":"aggregate",
              "Division":"ratio","Multiplication":"multiply","Compositional":"composition_review"}.get(original["type"])
        result.append(Candidate(query_id=qid,question=original["text"],category=original["category"],
            native_reasoning=original["reasoning"],native_type_raw=original["type"],
            type_canonical_for_analysis=TYPE_MAP.get(original["type"],"UNMAPPED:"+original["type"]),
            public_operation_candidate=operation(original["text"]),native_operation_hint_for_reviewer=hint,
            source_ids=ids,source_count=len(unique),reference_chars=sum(len(references[r]["text"]) for r in unique),
            has_native_answer=has_answer,source_readiness=status,
            annotation_eligible=has_answer and status=="source_bound_ready",
            verified_source_issuer_ids=sorted({"cik:"+bindings[r]["cik"] for r in matched}),
            verified_source_filing_ids=sorted({bindings[r]["accession"] for r in matched}),
            shared_reference_component_id=group_ids[qid],flags=flags))
    result.sort(key=lambda r:r.query_id)
    def distribution(rows):
        return {"questions":len(rows),"categories":dict(sorted(Counter(r.category for r in rows).items())),
            "reasoning":{str(flag):sum(r.native_reasoning is flag for r in rows) for flag in (False,True)},
            "raw_types":dict(sorted(Counter(r.native_type_raw for r in rows).items())),
            "canonical_types":dict(sorted(Counter(r.type_canonical_for_analysis for r in rows).items())),
            "category_reasoning":[{"category":cat,"reasoning":flag,
                "count":sum(r.category==cat and r.native_reasoning is flag for r in rows)}
                for cat in sorted({r.category for r in result}) for flag in (False,True)]}
    report={"version":VERSION,"full":distribution(result),
        "qa_candidates":distribution([r for r in result if r.has_native_answer]),
        "source_bound_annotation_candidates":distribution([r for r in result if r.annotation_eligible]),
        "reasoning_type_crosstab":[{"reasoning":flag,"type_raw":kind,"count":count}
            for (flag,kind),count in sorted(Counter((r.native_reasoning,r.native_type_raw) for r in result).items())],
        "readiness":dict(Counter(r.source_readiness for r in result)),
        "flags":dict(Counter(f for r in result for f in r.flags)),
        "shared_reference_components":len(groups),"verified_original_rows":len(result),
        "reasoning_interpretation":"Original boolean task label, not a gold rationale, difficulty score or observed reasoning success.",
        "known_limitations":["Native Category is not gold provider selection.",
            "Reasoning=True is absent in several categories; cross-category flag effects are confounded.",
            "Type=None with Reasoning=True is unspecified, not an automatic annotation error.",
            "Source-bound availability is selective; its sample weights do not represent all FinDER.",
            "Original corpus has prior retrieval-development exposure; no new holdout is certified."]}
    return result,report


def type_quotas(rows,n):
    """One per available raw type where budget permits, then proportional deficit."""
    counts=Counter(r.native_type_raw for r in rows)
    allocation={k:0 for k in counts}
    for key in sorted(counts,key=lambda k:(-counts[k],k))[:min(n,len(counts))]:allocation[key]=1
    while sum(allocation.values())<n:
        key=max((k for k in counts if allocation[k]<counts[k]),
            key=lambda k:(Fraction(n*counts[k],len(rows))-allocation[k],k))
        allocation[key]+=1
    return allocation


def stratified_sample(rows,config,*,scope,categories=None):
    """Known stratum inclusion probabilities, no outcome or issuer-based cherry-picking."""
    selected,shortfalls=[],[]
    for category in sorted(categories if categories is not None else {r.category for r in rows}):
        cat=[r for r in rows if r.category==category]
        n=min(config.review_per_category,len(cat))
        available={flag:[r for r in cat if r.native_reasoning is flag] for flag in (False,True)}
        true_n=min(config.reasoning_true_target_per_category,n,len(available[True]))
        false_n=min(n-true_n,len(available[False]))
        true_n+=min(n-true_n-false_n,len(available[True])-true_n)
        quotas={True:true_n,False:false_n}
        if n<config.review_per_category:
            shortfalls.append({"category":category,"requested":config.review_per_category,"available":len(cat),"selected":n})
        for flag,draw in quotas.items():
            subset=available[flag]
            for kind,take in type_quotas(subset,draw).items() if draw else []:
                population=[r for r in subset if r.native_type_raw==kind]
                ordered=sorted(population,key=lambda r:digest([VERSION,config.seed,scope,category,flag,kind,r.query_id]))
                for row in ordered[:take]:
                    selected.append({**row.model_dump(),"selection_scope":scope,
                        "stratum_population":len(population),"selected_in_stratum":take,
                        "inclusion_probability":take/len(population),"analysis_weight":len(population)/take,
                        "weight_population":"All nonblank native QA" if scope=="global_primary_review" else "Locally source-bound nonblank QA only"})
    return sorted(selected,key=lambda r:r["query_id"]),shortfalls


def jsonl(path,rows):
    with path.open("x",encoding="utf-8") as stream:
        for row in rows:stream.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")


def plots(output,report):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    categories=sorted(report["full"]["categories"])
    fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
    for axis,key,title in zip(axes,("full","source_bound_annotation_candidates"),
            ("All native FinDER questions","Locally source-bound annotation candidates")):
        rows=report[key]["category_reasoning"]
        false=[next(r["count"] for r in rows if r["category"]==c and r["reasoning"] is False) for c in categories]
        true=[next(r["count"] for r in rows if r["category"]==c and r["reasoning"] is True) for c in categories]
        axis.barh(categories,false,label="Reasoning=False",color="#7e9cb2")
        axis.barh(categories,true,left=false,label="Reasoning=True",color="#d38b28")
        for i,(a,b) in enumerate(zip(false,true)):axis.text(a+b+max(false+true)*.015,i,str(a+b),va="center",fontsize=8)
        axis.set_title(title);axis.set_xlabel("Questions (independent panel scales)");axis.margins(x=.16);axis.legend(fontsize=8)
    fig.text(.5,.015,"Native boolean task labels; source availability is selective. Counts do not establish semantic validity or causal effects.",ha="center",fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1))
    fig.savefig(output/"category_reasoning.png",dpi=150)
    svg=output/"category_reasoning.svg";fig.savefig(svg);plt.close(fig)
    svg.write_text("\n".join(s.rstrip() for s in svg.read_text().splitlines())+"\n")


def report_html(report,primary,bound,shortfalls):
    pieces=["<!doctype html><html lang='ko'><meta charset='utf-8'><title>FinDER 후보군 확장</title>",
        "<style>body{max-width:1100px;margin:2em auto;font:16px/1.6 sans-serif;padding:1em}table{border-collapse:collapse;width:100%}"
        "td,th{border:1px solid #bbc;padding:.5em;text-align:right}td:first-child{text-align:left}img{max-width:100%}</style>",
        "<h1>FinDER 전체 후보군·Reasoning 분포</h1><p>고정 원본의 Category, reasoning(Boolean), type을 그대로 확인했습니다. "
        "이 표시는 정답 풀이·실제 추론 성공 여부·정답 provider가 아닙니다.</p>",
        "<img src='category_reasoning.png' alt='전체와 원문 연결 후보의 category reasoning 분포'>",
        "<table><tr><th>Category</th><th>전체</th><th>Reasoning=True</th><th>원문 연결 후보</th><th>전체 기반 검토 표본</th><th>연결 후보 검토 표본</th></tr>"]
    for cat,total in report["full"]["categories"].items():
        true=next(r["count"] for r in report["full"]["category_reasoning"] if r["category"]==cat and r["reasoning"] is True)
        pieces.append(f"<tr><td>{html.escape(cat)}</td><td>{total}</td><td>{true}</td><td>{report['source_bound_annotation_candidates']['categories'].get(cat,0)}</td>"
            f"<td>{sum(r['category']==cat for r in primary)}</td><td>{sum(r['category']==cat for r in bound)}</td></tr>")
    pieces += ["</table><h2>사용할 자료</h2><ul><li><a href='expanded_frame/review.html'>확장한 silver 맥락 주석 검토</a></li>"
        "<li><a href='census.json'>전체 분포와 라벨 검토 항목</a></li><li><a href='summary.json'>실행 범위</a></li>"
        "<li><a href='global_review.private.jsonl'>전체 기반 층화 검토 후보</a></li>"
        "<li><a href='bound_review.private.jsonl'>원문 연결 후보의 층화 검토 목록</a></li></ul>",
        "<p>원문 연결 표본의 가중치는 해당 후보군에만 적용합니다. 미연결 문항을 제외한 결과를 전체 FinDER로 일반화할 수 없습니다. "
        "공유 근거·기업·seed 의존성과 독립 gold 검수는 별도로 반영해야 합니다. Reasoning 효과는 해당 flag가 모두 존재하는 카테고리 안에서 비교합니다.</p>",
        "<p>목표 미달 셀: "+escaped(json.dumps(shortfalls,ensure_ascii=False))+"</p></html>\n"]
    return "\n".join(pieces)


def run(args):
    from .__main__ import read_jsonl
    output=args.output;output.mkdir(parents=True,exist_ok=False,mode=0o700)
    config=ExpansionConfig.model_validate_json(args.config.read_text())
    manifest=json.loads(args.native_manifest.read_text())
    if file_hash(args.parquet)!=manifest["parquet_sha256"]:
        raise ValueError("pinned original Parquet hash mismatch")
    for field,key in (("native","questions.private.jsonl"),("public","questions.agent.jsonl"),("references","references.private.jsonl")):
        if file_hash(getattr(args,field))!=manifest["artifacts"][key]:
            raise ValueError("pinned native artifact mismatch: "+field)
    inputs={str(getattr(args,k)):file_hash(getattr(args,k)) for k in
            ("parquet","native_manifest","native","public","references","pool","bindings","config")}
    if args.previous_frame:inputs[str(args.previous_frame)]=file_hash(args.previous_frame)
    code={str(p.relative_to(BASE)):file_hash(p) for p in (BASE/"src/ogkf").rglob("*.py")}
    run_id=str(uuid.uuid4())
    write_json(output/"manifest.json",{"run_id":run_id,"version":VERSION,"input_hashes":inputs,"code_hashes":code,
        "dataset_revision":manifest["revision"],"python":platform.python_version(),
        "dependency_lock_sha256":file_hash(BASE/"uv.lock"),"api_calls":0,"db_calls":0,"role":"evaluator_only"})
    archive=output/"code_archive";archive.mkdir()
    for rel,h in code.items():
        p=archive/(h+".py")
        if not p.exists():p.write_bytes((BASE/rel).read_bytes())
    pool=Dataset.model_validate_json(args.pool.read_text())
    native,public,references=read_jsonl(args.native),read_jsonl(args.public),read_jsonl(args.references)
    bindings=json.loads(args.bindings.read_text())["sources"]
    raw=pq.ParquetFile(args.parquet).read().to_pylist()
    candidates,report=census(raw,native,public,references,pool,bindings)
    if len(candidates)!=manifest["questions"] or set(report["full"]["categories"])!=set(CATEGORIES):
        raise ValueError("unexpected original FinDER population")
    primary,primary_shortfall=stratified_sample([r for r in candidates if r.has_native_answer],config,scope="global_primary_review",categories=CATEGORIES)
    ready=[r for r in candidates if r.annotation_eligible]
    bound,bound_shortfall=stratified_sample(ready,config,scope="source_bound_review",categories=CATEGORIES)
    if len(ready)>config.max_annotation_cases or len(ready)<24:
        raise ValueError("source-bound annotation population outside configured budget")
    rows=[r.model_dump() for r in candidates]
    table=pa.Table.from_pylist(rows)
    pq.write_table(table,output/"candidate_catalog.private.parquet")
    if pq.ParquetFile(output/"candidate_catalog.private.parquet").read().to_pylist()!=rows:
        raise ValueError("candidate Parquet readback mismatch")
    write_json(output/"census.json",report)
    for name,rs in (("global_review.private.jsonl",primary),("bound_review.private.jsonl",bound),
        ("reasoning_priority.private.jsonl",[r.model_dump() for r in candidates if r.native_reasoning or 'reasoning_false_type_specified' in r.flags]),
        ("label_review.private.jsonl",[r.model_dump() for r in candidates if any(f.startswith('reasoning_') for f in r.flags)]),
        ("binding_backlog.private.jsonl",[r for r in primary if not r["annotation_eligible"]])):
        jsonl(output/name,rs)
    annotation_config=dict(cohort_size=len(ready),selection_seed=config.seed,include_query_ids=[],
        same_issuer_candidates=config.same_issuer_candidates,other_issuer_candidates=config.other_issuer_candidates,
        max_rows_per_case=config.max_rows_per_case)
    cohort,selection=select_cohort(pool,native,public,annotation_config)
    selection.update(scope="Census of all locally source-bound nonblank QA, not a representative native sample",
        native_metadata_sidecar="../native_labels.private.jsonl")
    bundle=build_bundle(pool,cohort,bindings,args.filings,inputs,annotation_config)
    frame_hash=bundle.bundle_hash
    previous_count=0
    if args.previous_frame:
        previous=Bundle.model_validate_json(args.previous_frame.read_text())
        current={c.query.query_id:c for c in bundle.cases}
        if previous.source_dataset_hash!=pool.dataset_hash or any(
                c.query.query_id not in current or c!=current[c.query.query_id] for c in previous.cases):
            raise ValueError("expansion changed or dropped a previous immutable case")
        previous_count=len(previous.cases)
    frame_output=output/"expanded_frame";frame_output.mkdir()
    frame_summary=export_bundle(frame_output,bundle,cohort,selection,run_id=run_id)
    by_q={c.query.query_id:c for c in bundle.cases}
    native_labels=[r.model_dump()|{"case_id":by_q[r.query_id].case_id,"frame_bundle_hash":frame_hash} for r in ready]
    jsonl(output/"native_labels.private.jsonl",native_labels)
    page=(frame_output/"review.html").read_text()
    for row in native_labels:
        marker="<article id='"+escaped(row["query_id"])+"'>"
        badge="<p class='badge'>Native Category: "+escaped(row["category"])+" | Reasoning: "+str(row["native_reasoning"])+" | Type: "+escaped(row["native_type_raw"])+" — evaluator-only labels</p>"
        if marker not in page:raise ValueError("review HTML case marker missing")
        page=page.replace(marker,marker+badge)
    (frame_output/"review.html").write_text(page)
    inventory(frame_output)
    plots(output,report)
    (output/"index.html").write_text(report_html(report,primary,bound,bound_shortfall))
    write_json(output/"protocol.json",{
        "selection":config.model_dump(),"full_native_review_scope":"All nonblank published QA; global category-reasoning-raw-type stratified probability sample",
        "source_bound_review_scope":"Separate source-available engineering cohort; no population representativeness claim",
        "weights":"Inverse within-stratum inclusion probabilities; never use source-bound weights for full FinDER",
        "missingness":"Do not estimate native population performance from only the source-ready subset of the global review sample",
        "dependence":"Pair query/condition contrasts and cluster by shared source, verified issuer and corpus seed; issuer grouping incomplete outside source-bound pool",
        "reasoning_contrast":"Within categories with both native flag levels. Structural zeros prevent a fully crossed Category x Reasoning design.",
        "gold":"All automatic context annotations remain silver; native QA/reference gold is retained; context judgments and calibration still require independent review",
        "routing":"Do not pass category, native reasoning, native type, gold evidence IDs or sampling labels to router/synthesizer",
        "exposure":"Retrospective development corpus; no new holdout certified",
        "hypotheses":"More cases improve coverage, not proof of federation benefit or causal isolation"})
    summary={"version":VERSION,"native_questions":len(candidates),"qa_candidates":sum(r.has_native_answer for r in candidates),
        "categories":len(report["full"]["categories"]),"reasoning_true":sum(r.native_reasoning for r in candidates),
        "source_bound_annotation_candidates":len(ready),"global_review_sample":len(primary),
        "previous_cases_preserved":previous_count,"additional_silver_cases":len(ready)-previous_count,
        "global_sample_source_bound_ready":sum(r["annotation_eligible"] for r in primary),
        "bound_review_sample":len(bound),"bound_shortfalls":bound_shortfall,"global_shortfalls":primary_shortfall,
        "expanded_frame":frame_summary,"http_calls":0,"db_calls":0,
        "reasoning_note":"Boolean dataset task flag, not gold rationale or empirically verified reasoning success"}
    write_json(output/"summary.json",summary)
    inventory(output)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


def validate(output):
    for rel,h in json.loads((output/"artifact-sha256.json").read_text()).items():
        if file_hash(output/rel)!=h:raise ValueError("artifact checksum mismatch: "+rel)
    frame=output/"expanded_frame"
    bundle=Bundle.model_validate_json((frame/"frame.private.json").read_text())
    cohort=Dataset.model_validate_json((frame/"cohort.dataset.private.json").read_text())
    checks=verify_origins(bundle,cohort)
    rows=pq.ParquetFile(output/"candidate_catalog.private.parquet").read().to_pylist()
    candidates=[Candidate.model_validate(r) for r in rows]
    if len(candidates)!=len({r.query_id for r in candidates}):raise ValueError("duplicate catalog question")
    print(json.dumps({"checks":checks,"catalog_rows":len(rows),"source_files_rewritten":False},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    cmd=sub.add_parser("run")
    for field in ("parquet","native-manifest","native","public","references","pool","bindings","filings","config","output"):
        cmd.add_argument("--"+field,type=Path,required=True)
    cmd.add_argument("--previous-frame",type=Path)
    check=sub.add_parser("validate");check.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.command=="run":run(args)
    else:validate(args.output)


if __name__=="__main__":main()
