## Results (2026-10-02_142519)

_Model: claude-sonnet-5-5. 40 test cases._

| Prompt | Valid JSON | Clean JSON | Schema valid | Field accuracy | Exact match | Hallucination rate |
|---|---|---|---|---|---|---|
| v1_baseline | 100.0% | 0.0% | 0.0% | 40.6% | 0.0% | 0.0% |
| v2_schema | 100.0% | 100.0% | 0.0% | 76.2% | 0.0% | 5.1% |
| v3_rules_fewshot | 100.0% | 97.5% | 100.0% | 98.4% | 87.5% | 1.3% |

### Accuracy by field

| Prompt | business_name | contact_name | email | phone | industry | services | budget_usd | has_website |
|---|---|---|---|---|---|---|---|---|
| v1_baseline | 22.5% | 37.5% | 75.0% | 90.0% | 2.5% | 2.5% | 40.0% | 55.0% |
| v2_schema | 100.0% | 100.0% | 100.0% | 97.5% | 17.5% | 12.5% | 92.5% | 90.0% |
| v3_rules_fewshot | 100.0% | 100.0% | 100.0% | 100.0% | 97.5% | 100.0% | 100.0% | 90.0% |
