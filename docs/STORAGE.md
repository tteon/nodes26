# Preserve claims before choosing a database boundary

One real-world entity can have several representations and many source-qualified assertions. One scalar property per entity cannot retain every period, scope and source without additional structure. Labels classify; they do not by themselves encode claim conditions or enforce all context filters.

A competent single database separates entities, assertions and sources, retaining build/context metadata. Multiple databases provide another management boundary. Separate DBMS instances add deployment/resource boundaries and operating costs. Shared names or publisher IDs alone do not establish SAME_AS or compatible operands. MDM/survivorship and ontology migration remain proposed lifecycle policies, not validated outcomes.

## Measured read pattern

[measured_read.cypher](../queries/measured_read.cypher) contains the original inner MATCH. Single mode targets neo4j with both contexts. Sequential/parallel modes target ctxa Accounting and ctxb Legal. Both consume all required rows and combine results. Each database has its own uniqueness/index setup. The inner MATCH is unchanged; scheduling differs.

The original14-record fixture fits below the branch caps. Equal results there do not establish global top-k parity for arbitrary truncation. The richer property-level schema in [qualified_single_design.cypher](../queries/qualified_single_design.cypher) is illustrative, not the schema whose PROFILE was measured. Its proposed qualifier properties are not automatically indexed when hidden inside JSON payload strings.

## Native Composite reference

A Composite database exposes alias-selected local/remote constituent graphs without storing the graph data itself. Cross-graph associations require explicit keys/proxies. See [Neo4j concepts](https://neo4j.com/docs/operations-manual/current/scalability/composite-databases/concepts/).

[native_composite_example.cypher](../queries/native_composite_example.cypher) is a documentation-based Enterprise example and was not executed or timed here. It uses fixed aliases and returns partition identity plus scalar evidence. Our measured result is application fanout, not a native Composite execution plan. See [official query guide](https://neo4j.com/docs/operations-manual/current/scalability/composite-databases/querying-composite-databases/).

A single agent-facing tool can expose either execution strategy. Its evidence envelope still needs assertion/build ID, source/span ID, subject, attribute, value, period, scope, unit, basis, modality and completeness. Database boundaries do not automatically prevent the model from combining incompatible premises.
