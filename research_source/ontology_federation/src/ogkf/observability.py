from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import uuid

import pyarrow as pa
import pyarrow.parquet as pq


ATTEMPT_SCHEMA = pa.schema([
    (key, pa.string()) for key in (
        "experiment_run_id", "trace_id", "request_id", "parent_step_id", "experiment_condition_id",
        "model_id", "llm_provider", "graph_provider", "source_publisher", "prompt_version",
        "ontology_version", "document_id", "request_hash", "raw_path", "error", "validation_status",
        "provider_reported_usage", "rate_limit_headers", "mode")
] + [("attempt", pa.int32()), ("retry_count", pa.int32()), ("http_status", pa.int32()),
     ("input_tokens", pa.int64()), ("output_tokens", pa.int64()), ("http_latency_ms", pa.float64())])

GRAPH_SCHEMA=pa.schema([(k,pa.string()) for k in ("operation_id","trace_id","database_name","selected_domain",
    "query_template_id","query_parameters_hash","query_error","graph_provider","execution_plan_metrics")]+
    [("records_returned",pa.int64()),("query_latency_ms",pa.float64())])
AGENT_SCHEMA=pa.schema([(k,pa.string()) for k in ("trace_id","question_id","arm_id","strategy","identity_policy",
    "graph_provider","stop_reason","error")]+[("tool_call_count",pa.int64()),("routing_latency_ms",pa.float64()),
    ("synthesis_latency_ms",pa.float64()),("end_to_end_latency_ms",pa.float64()),("row_cap_reached",pa.bool_())])


class Recorder:
    """Append receipts before proceeding; private bodies separate from Parquet metadata.

    Retention: synthetic/public experiment inputs only, mode 0700/0600, manual deletion
    after 30 days. Never feed private production material without a new policy.
    """
    def __init__(self, root: Path, mode: str, *, experiment="extraction"):
        if not experiment.replace("_", "").isalnum():
            raise ValueError("invalid experiment partition")
        self.experiment = experiment
        self.run_id = str(uuid.uuid4())
        self.root = root / self.run_id
        self.root.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.mode = mode
        self.rows = []
        self.date = datetime.now(timezone.utc).date().isoformat()

    def write(self, relative: str, value):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open("x", encoding="utf-8") as f:
            os.chmod(path, 0o600)
            json.dump(value, f, indent=2, ensure_ascii=False)
        return str(path.relative_to(self.root))

    def attempt(self, row, receipt):
        path = self.write(f"raw/{row['request_id']}-{row['attempt']}.json", receipt)
        row = {**row, "raw_path": path, "experiment_run_id": self.run_id, "mode": self.mode}
        self.rows.append(row)
        # Separate immutable part per attempt survives later crashes and schema nulls.
        path = self.root / f"traces/experiment={self.experiment}/date={self.date}/run={self.run_id}"
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        pq.write_table(pa.Table.from_pylist([row], schema=ATTEMPT_SCHEMA), path / f"{row['request_id']}-{row['attempt']}.parquet")

    def readback(self):
        rows = []
        for path in sorted(self.root.glob(f"traces/experiment={self.experiment}/**/*.parquet")):
            rows.extend(pq.ParquetFile(path).read().to_pylist())
        if len(rows) != len(self.rows):
            raise RuntimeError("Parquet receipt count mismatch")
        return rows
