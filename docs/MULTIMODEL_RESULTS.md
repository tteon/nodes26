# Multi-model extraction before the fixed-model answer study

Main slide7 shows the same-source two-model comparison and the explicit transition to fixed GPT-OSS-120B. Appendix18 retains every model/profile volume and failure cell; Appendix19 retains all four GPT-OSS variants and the DeepSeek failed cells. These are separate from the 317-question answer campaign.

| Model | Prompt guidance | Assigned passages | Failed requests | Exact-span proposals |
| --- | --- | ---: | ---: | ---: |
| DeepSeek-V3.1 | No supplied ontology vocabulary | 12 | 4 | 41 |
| DeepSeek-V3.1 | Local types | 12 | 8 | 24 |
| DeepSeek-V3.1 | Financial definitions + local types | 12 | 6 | 35 |
| DeepSeek-V3.1 | Financial definitions + context limits | 12 | 5 | 41 |
| gpt-oss-120b | No supplied ontology vocabulary | 12 | 0 | 35 |
| gpt-oss-120b | Local types | 12 | 1 | 30 |
| gpt-oss-120b | Financial definitions + local types | 12 | 0 | 32 |
| gpt-oss-120b | Financial definitions + context limits | 12 | 1 | 29 |

Totals: 96 assigned conditions; 25 failed requests; 267 exact-span proposals. GPT-OSS has 126 proposals / 2 failed requests; DeepSeek has 141 proposals / 23 failed requests. A matching quote location does not certify a correct financial claim. Failed requests and unequal completion prevent interpreting raw proposal volumes as model superiority.

In one inspected same-source AMD example, the two context-limited extraction outputs disagree about negation: one drops the source’s negative qualifier and the other retains it. This is a retained inspected witness, not a population semantic error rate.

**Transition:** “We next hold GPT-OSS-120B fixed to inspect how guidance, extracted statements and delivered evidence relate to downstream answers. This controls model variation in that study; it does not establish that GPT-OSS is the best model.”

Pretraining differences motivate explicit guidance, but were not isolated experimentally. The controlled pilot is two models, not four providers. No new inference was performed for this presentation revision.
