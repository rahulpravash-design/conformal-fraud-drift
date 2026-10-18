# Decision log
| ID | Date | Decision | Evidence | Alternatives |
|---|---|---|---|---|
| D-001 | 2026-10-18 | Python 3.13 venv, pandas 3 / sklearn 1.9 / LightGBM 4.7; versions in `requirements-lock.txt` | `uv pip freeze` | pin older stack |
| D-002 | 2026-10-18 | Keep raw + row-level derived data out of Git | CC BY-NC-ND 4.0 (datasheet Q44) | commit samples (rejected: ND clause) |
| D-003 | 2026-10-18 | Primary split 0-3/4/5/6-7 as requested; benchmark-default 0-5/6-7 kept as planned robustness | datasheet Q11 | adopt default as primary |
| D-004 | 2026-10-18 | Primary group `age>=50`, sensitivity `age>50`; fixed before seeing data | datasheet Q17 says <50/≥50; notebook code uses `>50`; ages rounded to decades | pick after viewing sizes |
| D-005 | 2026-10-18 | Implement label-conditional (Mondrian-on-Y) calibration alongside marginal | ~1% prevalence ⇒ marginal LAC says little about fraud coverage | marginal only |
| D-006 | 2026-10-18 | Treat 2 test months as a small-sample limitation; plan rolling-origin secondary analysis | only months 0–7 exist | claim drift effects from 2 months |
| D-007 | 2026-10-18 | Defer APS, group-Mondrian, ACI, weighted CP | requested scope order; assumptions need literature check | implement now |
| D-008 | 2026-10-18 | `customer_age` stays a model feature (as in benchmark); exclude-age run is a sensitivity analysis | paper's LightGBM setup | exclude by default |
| D-009 | 2026-10-18 | Synthetic fixture used ONLY for mechanics tests; never written to `results/` | — | — |
