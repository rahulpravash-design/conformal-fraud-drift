# Research-gap assessment

**Verdict (2026-10-18): UNDETERMINED. Gate A is NOT passed. No novelty claim ("first", "novel") is permitted yet.**

## What was and was not done
- Tools available to me reached the open web only through one `WebSearch` result summary per query; **arXiv, Semantic Scholar, Crossref, OpenAlex,
  IEEE Xplore, DBLP, Google Scholar and ijcnn.org were all unreachable** (proxy 403 / DNS) from both shell and fetch tools.
  So **no candidate paper's full text or abstract page was opened.** Titles below come from search-result listings only.
- One search ("conformal prediction fraud detection Bank Account Fraud temporal distribution shift coverage") returned no result that the summary
  tied to BAF + conformal + temporal coverage. That is weak, non-exhaustive evidence of absence — one query, one engine, snippets only.

## Closest candidates (all UNVERIFIED, must be read before any claim)
| ID | Title (as listed) | Why it matters | Fraud? | BAF? | Chronological? | Group cov.? | Adaptation compared? | Review workload? | BAF variants? |
|---|---|---|---|---|---|---|---|---|---|
| L09 | Temporal Graph Prototype-conditioned CP for Fraud Detection | fraud + CP + temporal | yes | ? | ? | ? | ? (mentions ACI) | ? | ? |
| L10 | Audited CP for Classification under Unknown Distribution Shift | classification CP under shift | ? | ? | ? | ? | ? | ? | ? |
| L11 | High probability risk control under covariate shift (Feedzai) | same company as BAF; covariate shift | ? | ? | ? | ? | ? | ? | ? |
| L02/L03/L08 | ACI; CP under covariate shift; CP beyond exchangeability | method foundations (theory) | no | no | – | – | – | – | – |
`?` = not determined because the source was not read. Questions 8–10 of the required per-paper analysis cannot be answered until then.

## Provisional position
The proposed contribution — *an empirical, chronologically ordered measurement of marginal vs label-conditional vs group-conditional conformal
coverage with review-workload accounting on BAF* — is plausibly a modest empirical contribution, not a methodological one. Foundational
theory (L02, L03, L08) already says coverage can degrade under shift; the added value would be **evidence on this specific benchmark**.

## Smallest defensible change if overlap is found (or as a safe default)
1. Frame as an **empirical audit / reproducible benchmark study**, not "first" or "novel method".
2. Make the label-conditional (fraud-class) coverage finding a headline question: marginal LAC at ~1% prevalence is expected to under-cover fraud (hypothesis H1, untested).
3. Add rolling-origin windows to strengthen RQ1 beyond two test months.
4. Keep RQ4 (variants) conditional on schema checks; drop if Types III–V schemas make protocol comparison invalid.

## Action for the researcher (I cannot do this from here)
Run these in Google Scholar / Semantic Scholar and record hits in `literature/literature_matrix.csv`:
`"conformal prediction" "bank account fraud"` · `"conformal" "BAF" fraud fairness` · `"conformal prediction" fraud "temporal" coverage` ·
`"Mondrian" conformal "distribution shift" tabular` · `"group-conditional coverage" "distribution shift"` · `Feedzai conformal`.
Then open L09–L11 and fill the table. Gate A passes only if none answers RQ1–RQ3 on BAF.
