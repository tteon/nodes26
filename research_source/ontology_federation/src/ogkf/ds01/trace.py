from datetime import datetime, timezone
import json
from pathlib import Path
import uuid
import pyarrow as pa
import pyarrow.parquet as pq


class Ledger:
    """Durable immutable parts, common lineage columns and a lossless JSON payload."""
    def __init__(self, root, run_id):
        self.root, self.run_id = Path(root), run_id
        self.date = datetime.now(timezone.utc).date().isoformat()
        self.rows = {}

    def emit(self, table, **row):
        row = {"experiment_run_id": self.run_id, **row}
        self.rows.setdefault(table, []).append(row)
        fields = ("experiment_run_id", "condition_id", "sample_id", "graph_snapshot_id", "query_id",
                  "evidence_id", "assertion_id", "provider_id", "trace_id", "strategy")
        record = {k: str(row[k]) if row.get(k) is not None else None for k in fields}
        record["seed"] = row.get("seed")
        record["payload_json"] = json.dumps(row, ensure_ascii=False, sort_keys=True)
        schema = pa.schema([(k, pa.string()) for k in fields] + [("seed", pa.int64()), ("payload_json", pa.string())])
        directory = self.root / "parquet" / "experiment=ds01" / ("date=" + self.date) / table
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / (str(uuid.uuid4()) + ".parquet")
        pq.write_table(pa.Table.from_pylist([record], schema=schema), path)
        return row

    def verify(self):
        for table, rows in self.rows.items():
            files = (self.root / "parquet").glob("**/" + table + "/*.parquet")
            actual = [json.loads(r["payload_json"]) for p in files for r in pq.ParquetFile(p).read().to_pylist()]
            expected = [json.dumps(r, sort_keys=True) for r in rows]
            if sorted(json.dumps(r, sort_keys=True) for r in actual) != sorted(expected):
                raise ValueError("Parquet trace readback mismatch: " + table)
        return {table: len(rows) for table, rows in self.rows.items()}
