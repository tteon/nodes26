"""Source-pinned FIBO/reference controls and application-owned context guidance.

This is a prompt profile, not an OWL reasoner or independently reviewed fact gold.
The generic control has its own definitions; an additional shared-definition mode
isolates terminology/identifier changes. Neither mode changes source evidence.
"""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile
from typing import Literal
import xml.etree.ElementTree as ET

from pydantic import Field, model_validator

from ..models import Strict, digest
from .contracts import Ontology

ARMS = ("G", "F", "GX", "FX")
DefinitionControl = Literal["competent_generic", "shared_definitions"]
SHA = r"^[a-f0-9]{64}$"
RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
SKOS_DEFINITION = "{http://www.w3.org/2004/02/skos/core#}definition"


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def source_digest(value):
    """The older source registry uses compact canonical JSON, unlike ogkf.digest."""
    return sha_bytes(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    )


class AlignedTerm(Strict):
    concept_key: str
    generic_id: str
    fibo_id: str
    generic_definitions: list[str] = Field(min_length=1)
    fibo_definitions: list[str] = Field(min_length=1)
    kind: list[str] = Field(min_length=1)
    source_path: str
    source_sha256: str = Field(pattern=SHA)
    physical_fields: list[str]
    mapping_admission: str


class ContextExtension(Strict):
    id: str
    definition: str


class ProfileBundle(Strict):
    version: Literal["fibo-context-profiles-v2"]
    fibo_commit: str = Field(pattern=r"^[a-f0-9]{40}$")
    archive_sha256: str = Field(pattern=SHA)
    upstream_packages_sha256: str = Field(pattern=SHA)
    upstream_manifest_sha256: str = Field(pattern=SHA)
    upstream_mappings_sha256: str = Field(pattern=SHA)
    license_sha256: str = Field(pattern=SHA)
    terms: list[AlignedTerm] = Field(min_length=1)
    local_extensions: list[ContextExtension] = Field(min_length=1)
    relationship_contract: Ontology
    application_rules: list[str] = Field(min_length=1)
    mapping_review_status: Literal["pending_independent_review"]
    reasoner: Literal["none"]
    definition_equivalence: Literal["not_independently_reviewed"]

    @model_validator(mode="after")
    def identifiers(self):
        for field in ("concept_key", "generic_id", "fibo_id"):
            ids = [getattr(t, field) for t in self.terms]
            if len(ids) != len(set(ids)):
                raise ValueError("profile mappings must be one-to-one")
        for t in self.terms:
            path = PurePosixPath(t.source_path)
            if path.is_absolute() or ".." in path.parts or not path.parts:
                raise ValueError("unsafe source module path")
            if not t.fibo_id.startswith("https://spec.edmcouncil.org/fibo/ontology/"):
                raise ValueError("reference term is not a FIBO IRI")
            if not t.generic_id.startswith("urn:finder:generic:"):
                raise ValueError("generic identifier must be application-owned")
        if len({x.id for x in self.local_extensions}) != len(self.local_extensions):
            raise ValueError("duplicate context extension")
        return self

    @property
    def profile_hash(self):
        return digest(self.model_dump())

    def verify_sources(self, root):
        root = Path(root).resolve()
        graphs = {}
        for t in self.terms:
            p = (root / t.source_path).resolve()
            if root not in p.parents:
                raise ValueError("source module escapes reference directory")
            raw = p.read_bytes()
            if sha_bytes(raw) != t.source_sha256:
                raise ValueError("FIBO module hash mismatch")
            if t.source_path not in graphs:
                graphs[t.source_path] = ET.fromstring(raw)
            definitions = sorted(
                "".join(child.itertext())
                for node in graphs[t.source_path].iter()
                if node.get(RDF + "about") == t.fibo_id
                for child in node
                if child.tag == SKOS_DEFINITION
            )
            if definitions != sorted(t.fibo_definitions):
                raise ValueError("FIBO definition differs from pinned RDF")
            kinds = set()
            for node in graphs[t.source_path].iter():
                if node.get(RDF + "about") == t.fibo_id:
                    if node.tag.startswith("{http://www.w3.org/2002/07/owl#}"):
                        kinds.add(node.tag[1:].replace("}", ""))
                    kinds.update(
                        child.get(RDF + "resource")
                        for child in node
                        if child.tag == RDF + "type"
                    )
            if not set(t.kind) <= kinds:
                raise ValueError("FIBO kind differs from pinned RDF")
        if sha_bytes((root / "LICENSE").read_bytes()) != self.license_sha256:
            raise ValueError("FIBO license hash mismatch")
        return True

    def prompt_payload(self, arm, definition_control: DefinitionControl):
        if arm not in ARMS:
            raise ValueError("unknown profile arm")
        if definition_control not in ("competent_generic", "shared_definitions"):
            raise ValueError("unknown definition control")
        fibo = arm in {"F", "FX"}
        alpha = arm in {"GX", "FX"}
        return {
            "terms": [
                {
                    "concept_key": t.concept_key,
                    "id": t.fibo_id if fibo else t.generic_id,
                    "definition": t.fibo_definitions
                    if fibo or definition_control == "shared_definitions"
                    else t.generic_definitions,
                    "kind": t.kind,
                    "mapping_admission": t.mapping_admission,
                }
                for t in self.terms
            ],
            "classes": self.relationship_contract.classes,
            "relations": [
                {"relation_id": r.relation_id, "domain": r.domain, "range": r.range}
                for r in self.relationship_contract.relations
            ],
            "context_guidance": {
                "definitions": [x.model_dump() for x in self.local_extensions],
                "relationship_contracts": [
                    r.model_dump() for r in self.relationship_contract.relations
                ],
                "rules": self.application_rules,
            }
            if alpha
            else None,
        }


class ProfileReview(Strict):
    version: Literal["ontology-profile-review-v1"]
    profile_hash: str = Field(pattern=SHA)
    reviewer_ids: list[str] = Field(min_length=2, max_length=2)
    submission_hashes: list[str] = Field(min_length=2, max_length=2)
    adjudication_hash: str = Field(pattern=SHA)
    approved_concept_keys: list[str]
    approved_guidance_hash: str = Field(pattern=SHA)
    source_based_rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_reviewers(self):
        if len({r.strip().casefold() for r in self.reviewer_ids if r.strip()}) != 2:
            raise ValueError("two distinct reviewer identities required")
        if len(set(self.submission_hashes)) != 2 or any(
            len(h) != 64 or any(c not in "0123456789abcdef" for c in h)
            for h in self.submission_hashes
        ):
            raise ValueError("two distinct hash-bound submissions required")
        if not self.source_based_rationale.strip():
            raise ValueError("source-based mapping rationale required")
        return self


def load_bundle(path):
    path = Path(path)
    bundle = ProfileBundle.model_validate_json(path.read_text())
    bundle.verify_sources(path.parent / "reference")
    return bundle


def build(packages, manifest, archive, ontology_path, output):
    """Freeze only three needed upstream modules; never extract arbitrary paths."""
    packages, manifest, archive = map(Path, (packages, manifest, archive))
    source = json.loads(packages.read_text())
    provenance = json.loads(manifest.read_text())
    if sha_bytes(archive.read_bytes()) != provenance["archive_sha256"]:
        raise ValueError("FIBO archive provenance mismatch")
    for arm in ARMS:
        p = dict(source[arm + "-d0"])
        expected = p.pop("content_sha256")
        if source_digest(p) != expected:
            raise ValueError("upstream package hash mismatch")
    fibo = source["F-d0"]
    generic = {t["id"]: t for t in source["G-d0"]["terms"]}
    mappings_path = packages.parent / "application-mappings.json"
    mappings = json.loads(mappings_path.read_text())
    if source["GX-d0"]["local_extensions"] != source["FX-d0"]["local_extensions"]:
        raise ValueError("generic and FIBO alpha definitions differ")
    if any(source[arm + "-d0"]["distractor_terms"] for arm in ARMS):
        raise ValueError("primary profiles must exclude ontology distractors")
    terms = []
    for t in fibo["terms"]:
        matching = [m for m in mappings if m["fibo"] == t["id"]]
        if len(matching) != 1:
            raise ValueError("missing or ambiguous FIBO mapping")
        m = matching[0]
        terms.append(
            AlignedTerm(
                concept_key=m["facet"] + ":" + m["generic"].rsplit(":", 1)[-1],
                generic_id=m["generic"],
                fibo_id=t["id"],
                generic_definitions=[generic[m["generic"]]["definition"]],
                fibo_definitions=t["definition"],
                kind=t["kind"],
                source_path=t["source_path"],
                source_sha256=t["source_sha256"],
                physical_fields=m["physical_fields"],
                mapping_admission=m["admission"],
            )
        )
    terms.sort(key=lambda t: t.concept_key)
    modules = {t.source_path for t in terms}
    root_prefix = "fibo-" + provenance["fibo_commit"] + "/"
    with tarfile.open(archive) as tar:
        files = {
            p: tar.extractfile(root_prefix + p).read() for p in modules | {"LICENSE"}
        }
    bundle = ProfileBundle(
        version="fibo-context-profiles-v2",
        fibo_commit=provenance["fibo_commit"],
        archive_sha256=provenance["archive_sha256"],
        upstream_packages_sha256=sha_bytes(packages.read_bytes()),
        upstream_manifest_sha256=sha_bytes(manifest.read_bytes()),
        upstream_mappings_sha256=sha_bytes(mappings_path.read_bytes()),
        license_sha256=sha_bytes(files["LICENSE"]),
        terms=terms,
        local_extensions=source["FX-d0"]["local_extensions"],
        relationship_contract=Ontology.model_validate_json(
            Path(ontology_path).read_text()
        ),
        application_rules=[
            "A source-supported assertion can be inapplicable to the question's entity, period, basis, scope, unit, modality or jurisdiction.",
            "A shared real-world identifier does not make contextual representations or all their assertions interchangeable.",
            "Reporting to, membership of, oversight of and decision authority are distinct relationships; none implies the others without source evidence.",
            "Approval, intention and reported risk do not by themselves establish execution, achievement or occurrence. They may coexist; non-implication is not disjointness.",
            "Missing evidence in an admitted snapshot is unknown outside that snapshot, not proof of global absence.",
            "Keep source assertion, extracted proposal and derived inference distinct. Vocabulary conformance does not prove source support.",
        ],
        mapping_review_status="pending_independent_review",
        reasoner="none",
        definition_equivalence="not_independently_reviewed",
    )
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for name, raw in files.items():
        p = output / "reference" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
    bundle.verify_sources(output / "reference")
    (output / "profiles.json").write_text(bundle.model_dump_json(indent=2) + "\n")
    return bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    for name in ("packages", "manifest", "archive", "ontology", "output"):
        b.add_argument("--" + name, type=Path, required=True)
    v = sub.add_parser("verify")
    v.add_argument("--profiles", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        bundle = build(
            args.packages, args.manifest, args.archive, args.ontology, args.output
        )
    else:
        bundle = load_bundle(args.profiles)
    print(
        json.dumps(
            {
                "profile_hash": bundle.profile_hash,
                "terms": len(bundle.terms),
                "reasoner": bundle.reasoner,
                "semantic_primary": None,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
