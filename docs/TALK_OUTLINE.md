# GraphRAG begins with graph construction

14-slide outline,25minutes plus5minutes Q&A. These are notes, not a generated PPTX.

1. **GraphRAG begins before retrieval** (45 seconds)
2. **Text becomes a proposed graph** (105 seconds)
3. **Our financial workload and its questions** (105 seconds)
4. **Ontology makes intended meaning explicit** (90 seconds)
5. **Our ontology design philosophy** (120 seconds)
6. **What we changed and what we held fixed** (120 seconds)
7. **More extracted records do not guarantee preserved meaning** (105 seconds)
8. **Entity identity and assertion identity** (105 seconds)
9. **Storage choices inside a Neo4j architecture** (105 seconds)
10. **One consumer request, multiple graph reads** (120 seconds)
11. **What did the agent do with the evidence?** (150 seconds)
12. **Query plans and overlapping reads** (105 seconds)
13. **The client runtime changes the latency distribution** (135 seconds)
14. **Preserve the conditions under which a claim is useful** (90 seconds)

Sequence: source text → LLM entity/relationship extraction → ontology alignment → controlled formation → preservation of views → multiple-database retrieval → agent evidence assessment → client runtime → lessons.
