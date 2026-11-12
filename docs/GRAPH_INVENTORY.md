# Graph size: a fixed denominator and native write receipts

Main317-question extracted-graph campaign only; historical run totals, not current live DB inventory.

The main cohort has317unchanged questions,329selected reference passages and42issuers. The two extraction conditions produce634assigned graph builds. Counts below preserve namespace isolation; repeated aliases across questions remain separate physical nodes.

| Count | General definitions + rules | Financial ontology + rules | Total |
| --- | ---: | ---: | ---: |
| Graph construction attempts | 317 | 317 | 634 |
| Nonempty graph builds | 226 | 233 | 459 |
| Empty graph builds | 91 | 84 | 175 |
| Entity nodes | 537 | 619 | 1,156 |
| Issuer anchor nodes | 226 | 233 | 459 |
| Admitted assertion nodes | 2,274 | 2,330 | 4,604 |
| Source nodes | 236 | 238 | 474 |
| All nodes | 3,273 | 3,420 | 6,693 |
| CONTAINS edges | 537 | 619 | 1,156 |
| HAS_ASSERTION edges | 2,274 | 2,330 | 4,604 |
| SUPPORTED_BY edges | 2,274 | 2,330 | 4,604 |
| All native relationships | 5,085 | 5,279 | 10,364 |

## Identity and relationship meanings

One question × one extraction condition × its namespace. Nonempty builds are connected through their issuer anchor. Empty/failed builds are retained.

Physical entity identity is namespace + entity_key. The entity_key hashes CIK + stored context + casefolded whitespace-normalized subject; this is not independently resolved real-world identity.

Without namespace, distinct contextual entity keys number497and576by condition; their union is767. CIK + normalized subject surface keys number457and532. These are different keys, not conflicting counts and not canonical entity-resolution results.

Native storage edges CONTAINS/HAS_ASSERTION/SUPPORTED_BY. Business attributes and qualifiers are assertion payload fields, not distinct native relationship types.

A semantic entity is not the same counting unit as every graph node. Issuer, assertion and source nodes also occupy the database. More admitted assertions do not establish more correct facts.

## Verification and scope

All634original index-write receipts were hash-verified. Every build exactly matches the recorded nodes_created, relationships_created and record_count values. No model or database call was needed for this audit. Disposable historical databases were torn down; these are historical construction totals, not a claim that6693nodes are currently live in the capture DB.

The earlier two-model pilot materialized8model/profile views with267exact-span proposals and remains a separate cohort. The FMC screenshot fixture is one later reconstruction with18nodes/29edges; the selected cited view has7nodes/7edges. Do not sum these copies into the main benchmark or label them unique world facts.

## What the FMC screenshot demonstrates

The same FMC entity links to separate dated legal-claim and shipping/handling accounting statements. Three selected statements cite two original passages. Conditions and source lineage survive in explicit assertion nodes and payloads, allowing a reader to inspect the evidence that reached the model. The picture alone does not demonstrate federation, a contamination reduction, semantic correctness or a faster query. Actual model-input parity and runtime comparisons have separate receipts.
