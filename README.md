# Ontology-Guided Graph Federation for Agentic RAG

**XCENA - Graph Product Engineer - Yitae Jeong**

Session track: Data Intelligence

Subtitle: One Entity, Multiple Contexts

**[Results, graph sizes, charts and closing slides](docs/RESULTS.md)**

Code and aggregate evidence supplement for the session.

GraphRAG begins with the graph we construct: text → entity/relationship extraction → explicit ontology guidance → qualified assertions → evidence retrieval → agent answer → runtime evaluation.

This repository includes a portable synthetic demo, the actual Python scheduling function and unchanged neo4rs reader used in the matched read study, aggregate results, query examples, common prompt prefixes and an inspectable research-source archive. It does not redistribute FinDER questions, answers, source passages, actual model request/response receipts, credentials or third-party FIBO definitions.

## Start offline

```bash
git clone https://github.com/tteon/nodes26.git
cd nodes26
python3 demo.py
python3 -m unittest discover -s tests -v
```

Python 3.10 or newer is sufficient for the offline demo. Four authored synthetic assertions concern one ExampleCo entity across Legal/Accounting and segment scopes. The demo checks that qualified single-view and partitioned representations return identical payloads, and refuses silent overwrites of conflicting assertions in one build. It reports **zero database/model calls and no measured latency**. This is a contract demonstration, not new semantic benchmark evidence.

## Optional live reads

Install `requirements.txt`. Supply `NEO4J_URI`, `NEO4J_USERNAME` and `NEO4J_PASSWORD` in the process environment. The runner does not automatically read a local `.env`. Use an owned disposable Neo4j-compatible DBMS with multi-database support; the synthetic loader refuses nonempty target databases and performs no deletion.

```bash
python3 -m pip install -r requirements.txt
python3 demo.py load-synthetic --confirm-empty-demo-dbms
python3 demo.py read --mode single --count 4
python3 demo.py read --mode sequential --count 4
python3 demo.py read --mode parallel --concurrency 4 --count 4
```

The loader checks the default `neo4j` database, creates empty `ctxa`/`ctxb` if absent, and loads copies of the synthetic payloads. Reads consume all rows in explicit transactions. Each reader invocation includes eight warmup logical requests; `count` is bounded to 1–32 and concurrency to 1 or 4. Timings from your environment are new local observations, not reproduction of the published financial benchmark numbers. No paid model API is called.

For the standalone Rust reader:

```bash
cargo build --release --locked --manifest-path rust/Cargo.toml
python3 run_rust.py --mode parallel --concurrency 4 --count 4
```

The adapter maps environment names internally, retaining the original research Rust file unchanged. [Source provenance](SOURCE_PROVENANCE.json) records file hashes. neo4rs 0.8.0 is distinct from the `neo4j-rust-ext` extension for the Python driver; the latter was not benchmarked here.

## Read the research

- [Talk outline](docs/TALK_OUTLINE.md): graph construction first, 16 main slides, 25 + 5 minutes.
- [Methodology and limits](docs/METHODOLOGY.md): separate historical diversity, two-model pilot, fixed-model 317 questions and DB-only execution.
- [Storage/query guide](docs/STORAGE.md): qualified single DB, database partitions and a separately labeled native Composite example.
- [Aggregate quality](results/quality_summary.json), [driver measurements](results/driver_summary.json), [actual PROFILE diagnostics](results/driver_profiles.json).
- [Research source archive](research_source/README.md): source code with local path placeholders; external semantic packages, datasets and paid-run receipts are not included.
- [Prompt prefixes](prompts/README.md): common instructions only, not the complete historical system messages.

The study does not establish universal ontology accuracy gains, superiority over BM25, a GIL bottleneck, native Composite performance or production survivorship. The source-bound 317-question campaign used gpt-oss-120b; the grading judge uses the same model family with zero independent human reviews. A qualified single database is a competent control. All 576 matched database requests returned identical payloads; Python also overlapped reads, and Rust's lower fanout median came with a higher p95 at concurrency four.

## Data and licensing

Repository code follows the research workspace's MIT license declaration. Third-party data and definitions retain their own terms. FinDER is by Linq-AI-Research; use the [pinned official dataset card](https://huggingface.co/datasets/Linq-AI-Research/FinDER/blob/c4c1b6454aef7f0bb1c37235c7f52ce644642da0/README.md), revision `c4c1b6454aef7f0bb1c37235c7f52ce644642da0`, and its CC-BY-NC-4.0 terms and attribution. Our fixture is independently authored synthetic text and is not a renamed copy of the native dataset.

The frozen FIBO-guided research package used commit `fee10a4ebe807f647565f7aa3da8968423826dd1`. Definition files are not bundled; obtain applicable source definitions and their license separately before reconstructing the historical prompts. Generated proposals, exact source locators and schema-valid answers are not automatically semantically correct.
