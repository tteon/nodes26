# What each experiment establishes

## Financial workload

The source dataset is FinDER, a financial question/reference/answer benchmark. The pinned snapshot has 5,703 rows; our 317 unchanged native questions cover eight categories, 42 issuers and 329 source records. The references were selected for each question: source-bound exposed development, not open retrieval or an unseen holdout. Native answers remain evaluator-only. The public repository contains no native question catalog, gold, reference text or actual API payloads. [Dataset card and terms](https://huggingface.co/datasets/Linq-AI-Research/FinDER/blob/c4c1b6454aef7f0bb1c37235c7f52ce644642da0/README.md).

The 317 questions retain native Reasoning flags: 78 true / 239 false. That flag and operation labels are not graph-hop counts. The earlier two-model extraction pilot and DB-only fixture are separate cohorts, not additional native questions.

## Model, prompt and ontology controls

Historical tags deepseek/gptoss/minimax25/minimax27 include two MiniMax variants. A historical 305 versus 1,200+ proposal comparison uses 16 versus 80 cases and is not a matched improvement estimate. Later extraction uses 12 passages × 2 models × 4 guidance profiles: 96 conditions, 25 failures and 267 exact-span proposals. Models: DeepSeek-V3.1 and gpt-oss-120b; answer/router: gpt-oss-120b. Exact span binding is not entailment or a calibrated model ranking.

The 317 campaign fixes gpt-oss-120b at temperature 0. G has generic definitions; GX adds local rules; FX uses selected FIBO definitions/identifiers with the same rules. Fixed-source query guidance compares G/GX/FX. Index-to-answer changes GX/FX extraction/graph content while querying with GX. Definitions, identifiers and package length change together. Different pretraining corpora are a motivation, not an isolated measured cause.

The semantic package follows the common system instruction. Extraction sees source spans, not downstream gold. The query sees a question/catalog, requests evidence once, receives an actual tool message and returns a structured result. The actual CIK/namespace traversal anchor is harness-owned; selected entity handles did not drive new adaptive traversal in the 317-question campaign. Do not claim that these episodes establish improved entity routing.

## Answer assessment

Existing 317 questions × 5 paths = 1,585 answers. 1,584 native rubric scores are known and one remains UNKNOWN; 14 evidence judgments remain invalid after one received-contract recovery. Original output failures and explicit abstentions score 0 under the frozen native rule. No-material-claim faithfulness is undefined. Complete nonempty claim-list means cover different subsets across conditions. Judge family matches the generator; independent human reviews: 0.

In index-to-answer, FX has 66 higher / 58 lower / 193 equal scores versus GX. Mean rubric scores 0.379 versus 0.377 do not establish reliable ontology gains, and fully correct counts are 92 versus 101. A gold-agreeing answer can still add a source-unsupported explanation. An appropriate abstention can reflect an incomplete delivered packet. The judge itself can return a format-valid but incomplete claim list. Preserve those failures rather than converting them into semantic truth.

## Matched execution

The original 576 measured logical requests return the same 14 copied FMC payloads. Two clients, three modes, concurrency 1/4, three shuffled blocks, 16 requests per cell/block; 288 warmup logical requests separately. Unique issuer anchor, three storage edges, fixed cap; PROFILE 133 single versus 70 + 70 branch hits. Payload parity validates the execution comparison, not answer quality. The synthetic demo here has 4 assertions and does not reproduce that benchmark population.

Python uses threads, Rust uses Tokio. At concurrency 4, parallel p50/p95: Python 7.668/14.613 ms; neo4rs 3.433/16.832 ms. Both single-DB medians are lower than fanout. Python already overlaps branches. GIL, decode, pool wait, network and server contention were not isolated; Rust's slow tail block has no established cause. These are warm local small-result observations, not saturation limits or native Composite performance.

No new API or database experiment was run to prepare this repository. The default demo and tests run offline. Optional live commands produce new observations on your deployment and do not call an LLM.

## Per-question audit companion

The retained evaluator-only companion joins all 317 published answers and references to 634 GX/FX indexing views and all 1,585 answer paths, including actual delivered packets, query/trace provenance and automated claim judgments. The source archive includes `build_question_evidence_companion.py`, an offline join script requiring separately retained licensed inputs and immutable receipts. Source-containing cards and native gold are excluded from this public repository. Graph projections derive from admitted payloads, with actual query readback kept separately; admitted proposals are not extraction ground truth.
