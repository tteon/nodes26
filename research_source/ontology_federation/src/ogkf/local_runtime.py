"""Disposable, owned local DozerDB appliance. Never touches existing containers."""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
import secrets
import socket
import subprocess
import time
import uuid

from neo4j import GraphDatabase

from .probes import DOZER_IMAGE


@contextmanager
def local_dozer():
    name = "ogkf-local-" + uuid.uuid4().hex[:12]
    password = secrets.token_urlsafe(24)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    environment = {**os.environ, "NEO4J_AUTH": "neo4j/" + password}
    command = ["docker", "run", "-d", "--rm", "--name", name,
               "--label", "experiment.owner=ogkf-local", "--memory", "2g", "--cpus", "2",
               "-p", f"127.0.0.1:{port}:7687", "--env", "NEO4J_AUTH",
               "--env", "NEO4J_server_memory_heap_initial__size=256M",
               "--env", "NEO4J_server_memory_heap_max__size=512M",
               "--env", "NEO4J_server_memory_pagecache_size=128M", DOZER_IMAGE]
    process = subprocess.run(command, env=environment, capture_output=True, text=True, timeout=60)
    if process.returncode:
        raise RuntimeError("owned DozerDB startup failed: " + process.stderr.replace(password, "[REDACTED]"))
    driver = GraphDatabase.driver(f"bolt://127.0.0.1:{port}", auth=("neo4j", password),
                                  connection_timeout=2, max_transaction_retry_time=0)
    manifest = {"container": name, "image": DOZER_IMAGE, "uri": f"bolt://127.0.0.1:{port}",
                "memory_bytes": 2147483648, "cpus": 2, "heap_max_mib": 512,
                "pagecache_mib": 128, "existing_services_modified": False}
    try:
        deadline = time.monotonic() + 90
        while True:
            try:
                driver.verify_connectivity()
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise RuntimeError("owned DozerDB readiness deadline exceeded")
                time.sleep(1)
        image = json.loads(subprocess.run(["docker", "image", "inspect", DOZER_IMAGE,
            "--format", "{{json .RepoDigests}}"], capture_output=True, text=True, check=True).stdout)
        manifest["image_verified"] = DOZER_IMAGE in image
        yield {"DOZER_URI": manifest["uri"], "DOZER_USER": "neo4j", "DOZER_PASSWORD": password}, manifest
    finally:
        driver.close()
        label = subprocess.run(["docker", "inspect", name, "--format", '{{index .Config.Labels "experiment.owner"}}'],
                               capture_output=True, text=True, timeout=10)
        if label.returncode == 0 and label.stdout.strip() == "ogkf-local":
            removed = subprocess.run(["docker", "rm", "-f", "-v", name], capture_output=True, text=True, timeout=30)
            manifest["teardown_success"] = removed.returncode == 0
        else:
            manifest["teardown_success"] = False
