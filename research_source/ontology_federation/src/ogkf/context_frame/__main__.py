import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import uuid

import pyarrow.parquet as pq

from ..ds01.contracts import Dataset
from .artifacts import export_bundle, inventory, write_json, write_table
from .compiler import build_bundle, file_hash, select_cohort
from .lifecycle import FrameEvaluator, apply_review, pending_answer_review, verify_origins
from .models import Bundle, Case, FrameConfig, ReviewReceipt


def read_jsonl(path):
    result={}
    for line in path.read_text().splitlines():
        row=json.loads(line)
        if row["id"] in result:
            raise ValueError("duplicate native question identity")
        result[row["id"]]=row
    return result


def new_output(path):
    path.mkdir(parents=True,exist_ok=False,mode=0o700)
    return path


def replay(bundle,campaign,output,run_id):
    original=Dataset.model_validate_json((campaign/"input.evaluator-only.json").read_text())
    if original.dataset_hash!=bundle.source_dataset_hash:
        raise ValueError("cached campaign source dataset does not match frame")
    qids={c.query.query_id for c in bundle.cases}
    evaluator=FrameEvaluator(bundle)
    rows,pending,inputs,rejected=[],[],{},[]
    for table,stage in (("retrieval_traces","retrieval"),("agent_evaluation","agent")):
        for path in sorted((campaign/"runs").glob("*/"+table+".parquet")):
            inputs[str(path)]=file_hash(path)
            for raw in pq.ParquetFile(path).read().to_pylist():
                episode=json.loads(raw["payload_json"])
                if episode["query_id"] not in qids:
                    continue
                if episode.get("dataset_hash",bundle.source_dataset_hash)!=bundle.source_dataset_hash:
                    raise ValueError("cached retrieval belongs to another source snapshot")
                try:
                    row=evaluator.evaluate(episode,stage=stage)
                except ValueError as exc:
                    rejected.append({"parent_sample_id":episode["sample_id"],"stage":stage,
                        "source_path":str(path),"error":str(exc)})
                    continue
                row.update(experiment_run_id=run_id,source_path=str(path),source_sha256=inputs[str(path)])
                rows.append(row)
                if stage=="agent":
                    review=pending_answer_review(episode)
                    if review:
                        pending.append({"query_id":episode["query_id"],"case_id":row["case_id"],
                            "parent_run_id":episode["experiment_run_id"],"source_path":str(path),
                            "source_sha256":inputs[str(path)],"episode":episode,"judgment":review.model_dump()})
    if not rows:
        raise ValueError("no cached episodes matched frame")
    count=write_table(output,"context_evaluation",rows,run_id)
    write_json(output/"evaluation.json",rows)
    write_json(output/"answer-review.pending.json",pending)
    write_json(output/"rejected-links.json",rejected)
    summary={"frame_bundle_hash":bundle.bundle_hash,"matched_episodes":count,
        "retrieval_episodes":sum(r["stage"]=="retrieval" for r in rows),
        "agent_outcomes_including_preserved_failures":sum(r["stage"]=="agent" for r in rows),
        "answer_review_items":len(pending),"rejected_links":len(rejected),
        "reviewed_sufficiency_scored":sum(r["evidence_sufficiency_at_k"] is not None for r in rows),
        "http_calls":0,"db_calls":0,"cached_source_hashes":inputs,
        "note":"Post-hoc frame-linked evaluation of preserved samples; no new independent repetition or gold promotion."}
    write_json(output/"summary.json",summary)
    return summary


def main():
    p=argparse.ArgumentParser(description="Apply evaluator-only GraphRAG-Bench-inspired context annotations to FinDER")
    sub=p.add_subparsers(dest="command",required=True)
    build=sub.add_parser("build")
    for field in ("pool","public","native","bindings","filings","config","output"):
        build.add_argument("--"+field,type=Path,required=True)
    schema=sub.add_parser("schemas")
    schema.add_argument("--output",type=Path,required=True)
    for name in ("validate","apply","replay"):
        cmd=sub.add_parser(name)
        cmd.add_argument("--frame",type=Path,required=True)
        if name in {"validate","apply"}:
            cmd.add_argument("--cohort",type=Path,required=True)
        if name=="apply":
            cmd.add_argument("--receipt",type=Path,required=True)
        if name=="replay":
            cmd.add_argument("--campaign",type=Path,required=True)
        if name!="validate":
            cmd.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.command=="schemas":
        output=new_output(args.output)
        for model in (Bundle,Case,FrameConfig,ReviewReceipt):
            write_json(output/(model.__name__+".schema.json"),model.model_json_schema())
        return
    if args.command=="validate":
        bundle=Bundle.model_validate_json(args.frame.read_text())
        cohort=Dataset.model_validate_json(args.cohort.read_text())
        print(json.dumps(verify_origins(bundle,cohort),indent=2))
        return
    output=new_output(args.output)
    run_id=str(uuid.uuid4())
    input_fields=("pool","public","native","bindings","config","frame","cohort","receipt")
    inputs={str(getattr(args,key)):file_hash(getattr(args,key)) for key in input_fields if getattr(args,key,None)}
    code={str(path.relative_to(Path(__file__).parents[3])):file_hash(path)
          for path in Path(__file__).parents[1].rglob("*.py")}
    write_json(output/"manifest.json",{"version":"finder-context-frame-run-v1","run_id":run_id,
        "command":args.command,"input_hashes":inputs,"code_hashes":code,"http_budget":0,
        "python":platform.python_version(),"dependency_lock_sha256":file_hash(Path(__file__).parents[3]/"uv.lock"),
        "created_at":datetime.now(timezone.utc).isoformat(),
        "role":"evaluator-only annotation and cached evaluation; no routing gold injection"})
    # Exact source snapshots permit later verification after implementation changes.
    for relative,h in code.items():
        source=Path(__file__).parents[3]/relative
        target=output/"code_archive"/(h+".py")
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            target.write_bytes(source.read_bytes())
    try:
        if args.command=="build":
            pool=Dataset.model_validate_json(args.pool.read_text())
            native,public=read_jsonl(args.native),read_jsonl(args.public)
            config=FrameConfig.model_validate_json(args.config.read_text()).model_dump()
            cohort,selection=select_cohort(pool,native,public,config)
            bindings=json.loads(args.bindings.read_text())["sources"]
            bundle=build_bundle(pool,cohort,bindings,args.filings,inputs,config)
            summary=export_bundle(output,bundle,cohort,selection,run_id=run_id)
        elif args.command=="apply":
            bundle=Bundle.model_validate_json(args.frame.read_text())
            cohort=Dataset.model_validate_json(args.cohort.read_text())
            verify_origins(bundle,cohort)
            reviewed=apply_review(bundle,ReviewReceipt.model_validate_json(args.receipt.read_text()))
            summary=export_bundle(output,reviewed,cohort,{"reviewed_from":bundle.bundle_hash},run_id=run_id)
        else:
            bundle=Bundle.model_validate_json(args.frame.read_text())
            summary=replay(bundle,args.campaign,output,run_id)
        inventory(output)
        print(json.dumps(summary,ensure_ascii=False,indent=2))
    except Exception as exc:
        write_json(output/"error.json",{"type":type(exc).__name__,"message":str(exc),"http_calls":0})
        raise


if __name__=="__main__":
    main()
