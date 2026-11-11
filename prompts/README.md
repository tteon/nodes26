# Shared prompt prefixes, not complete historical prompts

`index_system.txt` and `query_system.txt` are copied verbatim from the common instructions of the executed fixed-model campaign. A serialized G/GX/FX semantic package followed each prefix in the system role. Actual source spans/catalogs, schemas, tool calls and continuations varied by episode. Those payloads are not included here.

Extraction output:subject, attribute, value, period, scope, unit, basis, modality, concept_key, source_span_ids. At most 24 observations; 1–4 span IDs per observation. An exact locator does not prove source entailment, and the old contract admitted placeholders or rejected some multi-passage assertions. These failures are not silently fixed in published aggregates.

Query output: value, abstain, issuer_cik, evidence_ids, numerator, denominator, subject_used, relation_used, explanation. Generation did not receive native gold. The 317 campaign's actual query anchor remained harness-owned.

G=generic definitions;GX=generic+local rules;FX=selected FIBO definitions/identifiers+the same rules. The nine concepts concern time, leadership/roles/responsibility and monetary quantity facets. This is prompt guidance, not a reasoner or proof of model alignment. Full historical definition packages must be reconstructed from licensed sources and the frozen version metadata before claiming prompt reproduction.
