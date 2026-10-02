# Prompt Evaluation Harness: LLM Lead Extraction

A small, repeatable way to **measure** whether a prompt change actually makes an LLM better, instead of eyeballing a few outputs.

The task comes from real agency work at Eastworld Technology: turning messy sales notes (voicemails, texts, web forms, half-finished call notes) into clean, structured CRM data. This is the same problem behind the "Smart Paste" auto-fill feature in my Eastworld Prompt Hub app.

## What it does

```
40 labeled test cases ──► prompt v1 / v2 / v3 ──► Claude API ──► parse + validate JSON ──► score every field ──► CSV + Markdown report
```

- **Test set:** 40 realistic, messy notes, each with a hand-labeled correct answer (`data/test_cases.jsonl`). It includes edge cases on purpose: spelled-out emails, phone extensions, budget ranges, "maybe later" services, declined services, distractor numbers, and missing info.
- **Prompt versions** (`prompts/`):
  - `v1_baseline`: a one-line instruction
  - `v2_schema`: adds an explicit JSON schema
  - `v3_rules_fewshot`: adds allowed-value lists, normalization rules, a "never guess" rule, and one worked example
- **Scoring** (`eval.py`):
  - **Valid JSON**: the output can be parsed at all
  - **Clean JSON**: pure JSON with no markdown fences or extra chatter (matters for production pipelines)
  - **Schema valid**: correct keys, types, and allowed values
  - **Field accuracy**: per-field correctness, with lenient normalization so formatting isn't counted as an error
  - **Exact match**: every field in the record is correct
  - **Hallucination rate**: how often the model filled in a field that should have been empty
- **Experiment tracking:** every run appends to `results/history.csv` with a timestamp, model, and a hash of each prompt's text, so any score can be traced back to the exact prompt that produced it.

## Results

_Model: claude-sonnet-5-5 · 40 test cases · Oct 2026_

| Prompt | Valid JSON | Clean JSON | Schema valid | Field accuracy | Exact match | Hallucination rate |
|---|---|---|---|---|---|---|
| v1_baseline | 100.0% | 0.0% | 0.0% | 40.6% | 0.0% | 0.0%* |
| v2_schema | 100.0% | 100.0% | 0.0% | 76.2% | 0.0% | 5.1% |
| **v3_rules_fewshot** | **100.0%** | **97.5%** | **100.0%** | **98.4%** | **87.5%** | **1.3%** |

### Accuracy by field

| Prompt | business_name | contact_name | email | phone | industry | services | budget_usd | has_website |
|---|---|---|---|---|---|---|---|---|
| v1_baseline | 22.5% | 37.5% | 75.0% | 90.0% | 2.5% | 2.5% | 40.0% | 55.0% |
| v2_schema | 100.0% | 100.0% | 100.0% | 97.5% | 17.5% | 12.5% | 92.5% | 90.0% |
| v3_rules_fewshot | 100.0% | 100.0% | 100.0% | 100.0% | 97.5% | 100.0% | 100.0% | 90.0% |

<sub>*v1's 0% hallucination rate is misleading: it invented its own key names, so most fields came back missing rather than filled in. The meaningful comparison is v2 to v3.</sub>

## Key findings

- **Structure beats intelligence.** The same model jumped from 40.6% to 98.4% field accuracy, and from 0% to 87.5% fully correct records, purely through prompt design.
- **A schema fixes field names, but not values.** v2 got names, emails and phones nearly perfect, but scored only 17.5% on industry and 12.5% on services, and failed schema validation on every record, because the model invented its own category labels. Adding fixed allowed-value lists in v3 took those fields to 97.5% and 100%.
- **Explicit "never guess" rules cut hallucinations about 4x** (5.1% to 1.3%), and turned ambiguous inputs ("maybe 2 or 3 thousand", "under 2k", "five grand") into consistent, correct numbers.
- **The weakest field is still `has_website` (90%).** Notes like "needs a booking site" don't say whether a site already exists, so the model has to judge. Next step (v4): add worked examples for these implicit cases and re-measure.

## Run it yourself

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your-key-here        # Windows PowerShell: $env:ANTHROPIC_API_KEY="your-key-here"

python eval.py --dry-run                       # tests the pipeline with fake outputs, no key needed
python eval.py                                 # real run, all prompt versions
python eval.py --prompts v3_rules_fewshot      # one version only
python eval.py --model <model-name> --limit 10 # different model, first 10 cases
```

Each run creates `results/<timestamp>_live/` with:
- `summary.md`: comparison tables
- `summary.csv`: metrics per prompt version
- `details.csv`: every case, every field, plus the raw model output for failure analysis

## Adding a new prompt version

1. Copy the best prompt to `prompts/v4_<idea>.txt` and keep the `{notes}` placeholder.
2. Change **one thing** so you know what caused any difference.
3. Run `python eval.py --prompts v3_rules_fewshot v4_<idea>`.
4. Open `details.csv` and read the failures before deciding what to change next.

## Tech

Python 3.10+, Anthropic Claude API, standard library only (json, csv, re, argparse).
