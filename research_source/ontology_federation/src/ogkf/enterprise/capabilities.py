"""Separate conservative proof assessment; CREATE success alone is insufficient."""

import argparse
import json
from pathlib import Path


def assess_native_composite(manifest, *, permissions_receipt=None):
    ops = manifest.get("operations", {})

    def successful(name):
        return ops.get(name, {}).get("status") == "observed_success"

    expected = ["0", "1"]

    def values(key):
        return sorted(str(r.get("id")) for r in ops.get(key, {}).get("rows", []))

    checks = dict(
        composite_type=manifest.get("composite_reported_type") == "composite",
        constituent_aliases=successful("constituent_0") and successful("constituent_1"),
        multi_graph=successful("multi_graph") and values("multi_graph") == expected,
        dynamic_allowlisted_read=successful("dynamic_graph")
        and values("dynamic_graph") == expected,
        isolation_readback=manifest.get("database_isolation_verified") is True,
        least_privilege_permissions=bool(
            permissions_receipt
            and permissions_receipt.get("write_denied") is True
            and permissions_receipt.get("nonallowlisted_read_denied") is True
            and permissions_receipt.get("external_record")
        ),
    )
    return dict(
        version="native-composite-proof-v1",
        verified=all(checks.values()),
        checks=checks,
        missing_proof=[k for k, v in checks.items() if not v],
        execution_mode="native_capability_verified"
        if all(checks.values())
        else "application_level_only_or_unverified",
        scope="Capability proof,not semantic benefit. No DDL or network operation performed by this assessor.",
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--permissions", type=Path)
    args = p.parse_args()
    result = assess_native_composite(
        json.loads(args.manifest.read_text()),
        permissions_receipt=json.loads(args.permissions.read_text())
        if args.permissions
        else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
