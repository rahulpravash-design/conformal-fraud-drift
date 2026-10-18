# Data dictionary (from the official datasheet, https://github.com/feedzai/bank-account-fraud/blob/main/documents/datasheet.pdf)
**Not yet checked against a downloaded file.** Compare with `results/metrics/audit_Base.json` → `overview.columns` after download.

Label `fraud_bool` (binary). Time `month` (0–7). Missing sentinels: −1 (or any negative for `intended_balcon_amount`).

| Column | Type | Documented range / note |
|---|---|---|
| income | numeric | decile form, [0.1, 0.9] |
| name_email_similarity | numeric | [0, 1] |
| prev_address_months_count | numeric | [−1, 380] |
| current_address_months_count | numeric | [−1, 429] |
| customer_age | numeric | rounded to decade, [10, 90]; **protected attribute** |
| days_since_request | numeric | [0, 79] |
| intended_balcon_amount | numeric | [−16, 114], negatives missing |
| payment_type | categorical | 5 anonymised values |
| zip_count_4w | numeric | [1, 6830] |
| velocity_6h / 24h / 4w | numeric | applications per hour windows |
| bank_branch_count_8w | numeric | [0, 2404] |
| date_of_birth_distinct_emails_4w | numeric | [0, 39] |
| employment_status | categorical | 7 anonymised values |
| credit_risk_score | numeric | [−191, 389] |
| email_is_free, phone_home_valid, phone_mobile_valid, has_other_cards, foreign_request, keep_alive_session | binary | |
| housing_status | categorical | 7 anonymised values |
| bank_months_count | numeric | [−1, 32] |
| proposed_credit_limit | numeric | [200, 2000] |
| source | categorical | INTERNET / TELEAPP |
| session_length_in_minutes | numeric | [−1, 107] |
| device_os | categorical | windows, macOS, linux, x11, other |
| device_distinct_emails | numeric | [−1, 2] (datasheet name `device_distinct_emails`; actual column may be `device_distinct_emails_8w` [U]) |
| device_fraud_count | numeric | [0, 1] |
| month | numeric | [0, 7] |
| fraud_bool | binary | label |
Variants Type I–V add bias (group-size, prevalence, separability disparities) and Type IV/V change train-vs-test group behaviour on purpose; they may
add columns (Type III–V "additional columns appended", datasheet) [U].
