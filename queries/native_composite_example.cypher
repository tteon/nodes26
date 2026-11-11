// Documentation-based Neo4j Enterprise Composite example, not executed/timed here.
// Connect to composite database views, with local aliases views.accounting
// and views.legal targeting the two populated constituent databases.
// Aliases and constituent indexes must already exist.
// Cypher 5-compatible form; graph names are fixed by the trusted application.
CALL {
  USE views.accounting
  MATCH (i:BenchIssuer {cik:$cik})-[:CONTAINS]->(e:BenchEntity)
   -[:HAS_ASSERTION]->(a:BenchAssertion)-[:SUPPORTED_BY]->(s:BenchSource)
  WHERE a.context = 'Accounting'
  RETURN 'Accounting' AS storage_partition, a.id AS assertion_id, a.payload AS payload
  UNION ALL
  USE views.legal
  MATCH (i:BenchIssuer {cik:$cik})-[:CONTAINS]->(e:BenchEntity)
   -[:HAS_ASSERTION]->(a:BenchAssertion)-[:SUPPORTED_BY]->(s:BenchSource)
  WHERE a.context = 'Legal'
  RETURN 'Legal' AS storage_partition, a.id AS assertion_id, a.payload AS payload
}
RETURN storage_partition, assertion_id, payload
ORDER BY assertion_id
