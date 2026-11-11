from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import uuid

from jsonschema import Draft202012Validator

from .gateway import GatewayError

PROBE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean", "const": True}},
    "required": ["ok"],
    "additionalProperties": False,
}
DOZER_IMAGE = "graphstack/dozerdb@sha256:a77526ea3918fdc46d1fff70c4aea7d71d3874a26ecec059179d6775845b1247"
PROBE_PROMPT_VERSION = "independent-features-v2"


def probe_message(feature):
    if feature == "tool_calling":
        return "Call capability_probe with ok=true."
    if feature in {"json_object", "json_schema"}:
        return 'Return exactly the JSON object {"ok":true}.'
    return "Reply with the word OK."


def probe_value(response, feature):
    msg = response["choices"][0]["message"]
    if feature == "tool_calling":
        fn = msg["tool_calls"][0]["function"]
        if fn["name"] != "capability_probe":
            raise ValueError("wrong function")
        value = json.loads(fn["arguments"])
    elif feature in {"json_object", "json_schema"}:
        value = json.loads(msg["content"])
    else:
        value = {"ok": bool(msg.get("content"))}
    Draft202012Validator(PROBE_SCHEMA).validate(value)
    return value


def load_probe_receipts(
    run,
    model,
    base,
    *,
    now=None,
    require_basic=True,
    manifest_name="mara-capabilities.json",
):
    """Reuse one recent, source-receipt-checked live probe, never a hand-edited flag."""
    run = Path(run)
    if Path(manifest_name).name != manifest_name:
        raise ValueError("capability manifest must be a local filename")
    manifest = json.loads((run / manifest_name).read_text())
    now = now or datetime.now(timezone.utc)
    observed = datetime.fromisoformat(manifest["observed_at"])
    if (
        manifest.get("mode") != "live"
        or manifest.get("api_base_url") != base
        or manifest.get("probe_prompt_version") != PROBE_PROMPT_VERSION
        or observed.tzinfo is None
        or not 0 <= (now - observed).total_seconds() <= 3600
    ):
        raise ValueError("stale or incompatible capability receipts")

    def receipt(request_id):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", request_id):
            raise ValueError("invalid capability receipt ID")
        paths = list(run.glob("*/raw/" + request_id + "-*.json"))
        if len(paths) != 1:
            raise ValueError("unique capability HTTP receipt required")
        raw = json.loads(paths[0].read_text())
        if raw["http_status"] != 200:
            raise ValueError("unsuccessful capability receipt")
        return raw

    catalog = receipt(manifest["catalog_request_id"])
    actual_ids = [m["id"] for m in catalog["response"]["data"]]
    if (
        catalog["request"] is not None
        or actual_ids != manifest["available_model_ids"]
        or model not in actual_ids
    ):
        raise ValueError("model catalog receipt mismatch")
    entry = manifest["models"][model]
    if (require_basic and not entry.get("basic")) or not (
        entry.get("json_object") or entry.get("json_schema")
    ):
        raise ValueError("working basic and structured capabilities required")
    for feature in (
        "basic",
        "json_object",
        "json_schema",
        "tool_calling",
        "temperature",
    ):
        if not entry.get(feature):
            continue
        raw = receipt(entry["checks"][feature]["request_id"])
        body = raw["request"]
        if body["model"] != model or body["messages"] != [
            {"role": "user", "content": probe_message(feature)}
        ]:
            raise ValueError("capability request/model mismatch")
        if (
            feature in {"json_object", "json_schema"}
            and body.get("response_format", {}).get("type") != feature
        ):
            raise ValueError("structured capability request mismatch")
        if feature == "tool_calling" and not body.get("tools"):
            raise ValueError("tool capability request mismatch")
        probe_value(raw["response"], feature)
    return manifest


def probe_mara(gateway, models=None):
    models = models if models is not None else gateway.config.models
    if not 1 <= len(models) <= 3 or len(set(models)) != len(models):
        raise ValueError("probe must select 1..3 unique model IDs")
    manifest = {
        "manifest_version": "mara-capabilities-v1",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "probe_prompt_version": PROBE_PROMPT_VERSION,
        "api_base_url": gateway.base,
        "mode": gateway.recorder.mode,
        "provider": "mara",
        "available_model_ids": [],
        "models": {},
        "rate_limit_429": "not_observed; no deliberate quota exhaustion",
        "stability": "single probes only; not qualified",
        "blockers": [],
    }
    try:
        result = gateway.request(None, {}, path="/models")
        manifest["catalog_request_id"] = result["request_id"]
        manifest["available_model_ids"] = [m["id"] for m in result["response"]["data"]]
    except (GatewayError, KeyError, TypeError) as exc:
        manifest["blockers"].append(type(exc).__name__)
        return manifest
    for model in models:
        entry = {"listed": model in manifest["available_model_ids"], "checks": {}}
        manifest["models"][model] = entry
        if not entry["listed"]:
            continue
        for feature in (
            "basic",
            "json_object",
            "json_schema",
            "tool_calling",
            "temperature",
        ):
            body = {
                "model": model,
                "messages": [{"role": "user", "content": probe_message(feature)}],
                "max_tokens": 1024,
            }
            if feature == "json_object":
                body["response_format"] = {"type": "json_object"}
            elif feature == "json_schema":
                body["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "capability",
                        "strict": True,
                        "schema": PROBE_SCHEMA,
                    },
                }
            elif feature == "tool_calling":
                body["tools"] = [
                    {
                        "type": "function",
                        "function": {
                            "name": "capability_probe",
                            "parameters": PROBE_SCHEMA,
                        },
                    }
                ]
                body["tool_choice"] = {
                    "type": "function",
                    "function": {"name": "capability_probe"},
                }
            elif feature == "temperature":
                body["temperature"] = gateway.config.temperature
            try:
                result = gateway.request(
                    body, {"model_id": model, "parent_step_id": "capability/" + feature}
                )
                response = result["response"]
                probe_value(response, feature)
                entry[feature] = True
                entry["checks"][feature] = {
                    "request_id": result["request_id"],
                    "usage_available": isinstance(response.get("usage"), dict),
                    "interpretation": "one observed valid response; does not prove server-side enforcement",
                }
            except Exception as exc:
                entry[feature] = False
                entry["checks"][feature] = {
                    "error_type": type(exc).__name__,
                    "support": "unqualified",
                }
        entry["usage_available"] = any(
            c.get("usage_available") for c in entry["checks"].values()
        )
    manifest["rate_limit_headers_observed"] = any(
        json.loads(r["rate_limit_headers"] or "{}") for r in gateway.recorder.rows
    )
    manifest["rate_limit_429"] = (
        "observed"
        if any(r["http_status"] == 429 for r in gateway.recorder.rows)
        else manifest["rate_limit_429"]
    )
    return manifest


def probe_graph(*, mutate=False):
    """No existing databases are modified. Each name must be newly created by us.

    Destructive/DDL probes require explicit --mutate on a configured experiment
    endpoint. Failed cleanup stays in the manifest. Never use IF NOT EXISTS.
    """
    manifest = {
        "manifest_version": "graph-capabilities-v1",
        "observed_at": datetime.now(timezone.utc).isoformat(),
        "graph_provider": "dozerdb",
        "candidate_image": DOZER_IMAGE,
        "image_verified_on_target": False,
        "operations": {},
        "blockers": [],
        "federation_mode": "unverified",
        "created_databases": [],
        "cleanup": [],
    }
    uri, password = os.getenv("DOZER_URI"), os.getenv("DOZER_PASSWORD")
    if not uri or not password:
        manifest["blockers"].append(
            "DOZER_URI/DOZER_PASSWORD unavailable; all capabilities unknown"
        )
        return manifest
    from neo4j import GraphDatabase, Query, READ_ACCESS

    driver = GraphDatabase.driver(
        uri,
        auth=(os.getenv("DOZER_USER", "neo4j"), password),
        connection_timeout=5,
        max_transaction_retry_time=0,
    )
    created, aliases = [], []
    name = "ogkf" + uuid.uuid4().hex[:12]
    domain_names = [name + suffix for suffix in ("finance", "legal", "sustainability")]
    comp, alias = name + "composite", name + "alias"

    def q(label, query, db="system", params=None, *, read=False):
        import time

        start = time.perf_counter()
        try:
            with driver.session(
                database=db, default_access_mode=READ_ACCESS if read else "WRITE"
            ) as session:
                result = session.run(Query(query, timeout=10), params or {})
                rows = result.data()
                result.consume()
            manifest["operations"][label] = {
                "status": "observed_success",
                "query": query,
                "database": db,
                "rows": rows,
                "latency_ms": (time.perf_counter() - start) * 1000,
            }
            return True
        except Exception as exc:
            manifest["operations"][label] = {
                "status": "failed_unqualified",
                "query": query,
                "database": db,
                "error_type": type(exc).__name__,
                "error_code": getattr(exc, "code", None),
                "latency_ms": (time.perf_counter() - start) * 1000,
            }
            return False

    try:
        q(
            "server_version",
            "CALL dbms.components() YIELD name,versions,edition RETURN name,versions,edition",
            "neo4j",
            read=True,
        )
        if not mutate:
            manifest["blockers"].append(
                "DDL probes not run; use --mutate only on an experiment endpoint"
            )
            return manifest
        for db in domain_names:
            if not re.fullmatch(r"ogkf[a-f0-9]{12}(finance|legal|sustainability)", db):
                raise ValueError("invalid probe name")
            if q("create_" + db, f"CREATE DATABASE `{db}`"):
                created.append(db)
        if not created:
            manifest["blockers"].append(
                "No experiment database created; inspect administration errors"
            )
            return manifest
        db = created[0]
        q("database_selection", "RETURN 1 AS ok", db, read=True)
        q(
            "uniqueness",
            "CREATE CONSTRAINT ogkf_id FOR (n:OGKFEntity) REQUIRE n.id IS UNIQUE",
            db,
        )
        # Seed distinct sentinel facts to verify real multi-graph reads, not constant queries.
        for i, target in enumerate(created):
            q(
                "sentinel_" + target,
                "CREATE (:OGKFEntity {id:$id})",
                target,
                {"id": str(i)},
            )
        manifest["database_isolation_verified"] = True
        for i, target in enumerate(created):
            success = q(
                "isolation_" + target,
                "MATCH(n:OGKFEntity) RETURN n.id AS id",
                target,
                read=True,
            )
            if not success or manifest["operations"]["isolation_" + target].get(
                "rows"
            ) != [{"id": str(i)}]:
                manifest["database_isolation_verified"] = False
        q("shutdown", f"STOP DATABASE `{db}`")
        q("startup", f"START DATABASE `{db}`")
        if q("alias", f"CREATE ALIAS `{alias}` FOR DATABASE `{db}`"):
            aliases.append(alias)
            q("alias_selection", "MATCH (n:OGKFEntity) RETURN n.id", alias, read=True)
        if q("composite", f"CREATE COMPOSITE DATABASE `{comp}`"):
            created.append(comp)
            q("database_types", "SHOW DATABASES YIELD name,type RETURN name,type")
            manifest["composite_reported_type"] = next(
                (
                    row["type"]
                    for row in manifest["operations"]["database_types"].get("rows", [])
                    if row["name"] == comp
                ),
                None,
            )
            constituents = []
            for i, target in enumerate(domain_names):
                if target in created:
                    a = comp + ".d" + str(i)
                    if q(
                        "constituent_" + str(i),
                        f"CREATE ALIAS `{comp}`.`d{i}` FOR DATABASE `{target}`",
                    ):
                        aliases.append(a)
                        constituents.append(i)
            if len(constituents) >= 2:
                q(
                    "multi_graph",
                    f"CALL {{ USE `{comp}`.`d0` MATCH(n:OGKFEntity) RETURN n.id AS id UNION ALL USE `{comp}`.`d1` MATCH(n:OGKFEntity) RETURN n.id AS id }} RETURN id",
                    comp,
                    read=True,
                )
                q(
                    "dynamic_graph",
                    "UNWIND $graphs AS graph CALL { WITH graph USE graph.byName(graph) MATCH(n:OGKFEntity) RETURN n.id AS id } RETURN id",
                    comp,
                    {"graphs": [comp + ".d0", comp + ".d1"]},
                    read=True,
                )
        native = manifest["operations"].get("multi_graph", {})
        manifest["federation_mode"] = (
            "native_observed"
            if manifest.get("composite_reported_type") == "composite"
            and sorted(str(r.get("id")) for r in native.get("rows", [])) == ["0", "1"]
            else "application_level_required_or_unqualified"
        )
        # An unsupported constraint can make its DB unavailable on this version.
        # Probe it last in a separately owned DB; never poison the domain probes.
        constraint_db = name + "constraint"
        if q("create_constraint_db", f"CREATE DATABASE `{constraint_db}`"):
            created.append(constraint_db)
            q(
                "property_existence",
                "CREATE CONSTRAINT ogkf_required FOR (n:OGKFEntity) REQUIRE n.id IS NOT NULL",
                constraint_db,
            )
    finally:
        manifest["created_databases"] = list(created)
        for a in reversed(aliases):
            escaped = ".".join("`" + part + "`" for part in a.split("."))
            manifest["cleanup"].append(
                {
                    "alias": a,
                    "success": q(
                        "drop_alias_" + a, f"DROP ALIAS {escaped} FOR DATABASE"
                    ),
                }
            )
        for db in reversed(created):
            manifest["cleanup"].append(
                {
                    "database": db,
                    "success": q("drop_" + db, f"DROP DATABASE `{db}` DESTROY DATA"),
                }
            )
        driver.close()
    return manifest
