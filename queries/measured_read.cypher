// Exact MATCH used in the completed matched driver study.
// Single: database=neo4j, contexts=['Accounting','Legal'].
// Sequential/parallel: database=ctxa, contexts=['Accounting'];
//                     database=ctxb, contexts=['Legal'].
// Branch dispatch and sorting/combination happen in the client.
MATCH (i:BenchIssuer {cik:$cik})-[:CONTAINS]->(e:BenchEntity)
 -[:HAS_ASSERTION]->(a:BenchAssertion)-[:SUPPORTED_BY]->(s:BenchSource)
 WHERE a.context IN $contexts RETURN a.payload AS payload ORDER BY a.id LIMIT 65
