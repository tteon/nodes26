# Results and engineering lessons

**Ontology-Guided Graph Federation for Agentic RAG**
XCENA - Graph Product Engineer - Yitae Jeong
Session track: Data Intelligence

Subtitle: One Entity, Multiple Contexts

## Read the results

- [Two conclusion recaps and Thanks (editable PPTX)](../presentation/Closing_Recaps_and_Thanks_EN.pptx)
- [Closing slides PDF](../presentation/Closing_Recaps_and_Thanks_EN.pdf) and [spoken English](../presentation/CLOSING_SCRIPT_EN.md)
- [Graph inventory and exact counting definitions](GRAPH_INVENTORY.md)
- [Multi-model/profile results](MULTIMODEL_RESULTS.md)
- [Executed grading method and exact instructions](GRADING_METHOD.md)
- [Aggregate result charts](../results/figures/)

## Main 317-question graph inventory

The same 317 native questions link 329 selected reference passages from 42 issuers.
Two indexing conditions assign 634 graph builds. The 634 original write receipts
were hash-verified and reconcile exactly with node/relationship creation counts.

| Count | General definitions + rules | Financial ontology + rules | Total |
| --- | ---: | ---: | ---: |
| Nonempty graph builds | 226 | 233 | 459 |
| Empty/failed graph builds | 91 | 84 | 175 |
| Entity nodes, namespace-qualified | 537 | 619 | 1,156 |
| Admitted statement nodes | 2,274 | 2,330 | 4,604 |
| All nodes including issuer/source anchors | 3,273 | 3,420 | 6,693 |
| Native structural relationships | 5,085 | 5,279 | 10,364 |

A build means one question × one extraction condition × its namespace. These
are historical construction totals across isolated namespaces, not one current
live graph or a deduplicated real-world entity count. Native edges are CONTAINS,
HAS_ASSERTION and SUPPORTED_BY. Business attributes/qualifiers live in statement
payloads. More admitted statements do not establish more correct facts.

## Answer quality: mixed changes, same question set

Financial-ontology graphs score higher on 66 questions, equally on 193 and lower on 58
relative to general-definition graphs. Equal includes 124 Zero-to-Zero pairs.

| Evidence | Full | Partial | Zero | Total |
| --- | ---: | ---: | ---: | ---: |
| Original reference passages + general definitions/rules | 164 | 44 | 109 | 317 |
| General-definition extraction graph | 101 | 37 | 179 | 317 |
| Financial-ontology extraction graph | 92 | 56 | 169 | 317 |

Full=1: all required material parts agree with the published answer.
Partial=0.5: substantively correct but incomplete without critical contradiction.
Zero=0 under the frozen mismatch/no-answer/original-output-failure rules. Judge
failures stay UNKNOWN. Published-answer agreement and delivered-evidence support
use separate inputs. These are study-specific, same-model automated judgments;
independent human reviews remain 0. No general ontology accuracy gain is established.

## Formation and execution are separate studies

The earlier two-model pilot used 12 passages ×2 models ×4 prompt profiles:96 assigned
conditions,25 failed requests and 267 exact-span proposals. Unequal completion and
unvalidated proposal meaning prevent interpreting raw volumes as model ranking.
The later 317-question answer campaign fixes GPT-OSS-120B to inspect the pipeline.

The matched database/client study keeps 14 statements identical across read paths.
At concurrency 1, sequential-to-parallel p50 changes from 3.127 to 2.074 ms in Python
and 2.094 to 1.303 ms in Rust. At concurrency 4, Python/Rust fanout p50 is 7.668/3.433 ms,
while p95 is 14.613/16.832 ms. Single-DB PROFILE has 133 DB hits versus 70+70 in the two
branches. These small-fixture DB/client timings exclude hosted-model generation.
Python already overlaps reads; GIL causality, native Composite and intra-query
runtime effects were not isolated. [Full measurements](../results/driver_summary.json)
and [PROFILE records](../results/driver_profiles.json) preserve all cells.

## Reproduction boundary

The existing [synthetic demo](../README.md#start-offline) runs without model or DB
calls and checks the evidence contract. It does not regenerate the financial
benchmark. To recompute the historical graph inventory, obtain the frozen ledgers
and receipts separately and run the source-linked audit:

```bash
NODES26_FROZEN_RUN=/path/to/frozen/run NODES26_AUDIT_OUT=/tmp/nodes26-audit python3 tools/audit_graph_inventory.py
```

The public supplement contains code, aggregate measurements, source-free closing
slides, common/own judge instructions and independently authored synthetic data.
Native question/gold/reference text, actual model payloads, credentials and
third-party ontology definitions remain excluded under the established publication
scope. Original data and definition licenses/attribution are linked in the README.
No new model or database benchmark was run for this update.
