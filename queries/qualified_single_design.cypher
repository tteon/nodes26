// Proposed logical schema, not the schema whose PROFILE was measured.
// Qualifiers are materialized properties here, unlike the benchmark JSON payload.
CREATE CONSTRAINT assertion_in_build IF NOT EXISTS
FOR (a:Assertion) REQUIRE (a.build_id, a.id) IS UNIQUE;
// Uniqueness alone does not require every property to be present.
// Enforce required fields and source support in the application contract.
MATCH (e:Entity {issuer_cik:$cik})-[:HAS_ASSERTION]->(a:Assertion)
 -[:SUPPORTED_BY]->(s:Source)
WHERE a.build_id IN $allowed_builds
  AND a.context IN $allowed_contexts
  AND a.period = $period
  AND a.scope = $scope
RETURN e.issuer_cik, a.id, a.attribute, a.value, a.period, a.scope,
       a.unit, a.basis, a.modality, s.id AS source_id
ORDER BY a.id
