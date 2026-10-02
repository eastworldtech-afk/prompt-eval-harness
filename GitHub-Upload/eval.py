"""
Prompt Evaluation Harness
Runs several versions of an LLM extraction prompt against a labeled test set,
scores every output, and logs results so prompt changes can be compared over time.

Usage:
  python eval.py --dry-run                 # no API key needed, fake model outputs (tests the pipeline)
  python eval.py                           # real run with Claude, all prompt versions
  python eval.py --prompts v3_rules_fewshot --limit 10
"""

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import random
import re
import time
from pathlib import Path

ROOT = Path(__file__).parent
PROMPTS_DIR = ROOT / "prompts"
DATA_FILE = ROOT / "data" / "test_cases.jsonl"
RESULTS_DIR = ROOT / "results"

FIELDS = ["business_name", "contact_name", "email", "phone",
          "industry", "services", "budget_usd", "has_website"]

INDUSTRIES = {"food_and_beverage", "med_spa", "dental", "healthcare", "law_firm",
              "real_estate", "fitness", "retail", "construction", "home_services",
              "beauty", "auto", "professional_services", "other"}
SERVICES = {"website", "seo", "social_media", "paid_ads", "branding",
            "crm", "it_support", "ai_agent"}

DEFAULT_MODEL = os.environ.get("EVAL_MODEL", "claude-sonnet-4-5")
USE_TEMPERATURE = True


# ---------- loading ----------

def load_cases(limit=None):
    with open(DATA_FILE, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    return cases[:limit] if limit else cases


def load_prompts(names=None):
    prompts = {}
    for p in sorted(PROMPTS_DIR.glob("*.txt")):
        if names and p.stem not in names:
            continue
        prompts[p.stem] = p.read_text(encoding="utf-8")
    if not prompts:
        raise SystemExit(f"No prompts found matching {names}")
    return prompts


def prompt_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()[:10]


# ---------- model calls ----------

def call_claude(client, model, prompt_text):
    global USE_TEMPERATURE
    params = dict(model=model, max_tokens=600, messages=[{"role": "user", "content": prompt_text}])
    start = time.time()
    if USE_TEMPERATURE:
        try:
            msg = client.messages.create(**params, temperature=0)
        except Exception as e:
            # Some SDK versions / models don't accept temperature; fall back without it
            if "temperature" not in str(e).lower():
                raise
            USE_TEMPERATURE = False
            print("\n(note: temperature not supported here, continuing without it)")
            start = time.time()
            msg = client.messages.create(**params)
    else:
        msg = client.messages.create(**params)
    latency = time.time() - start
    text = "".join(block.text for block in msg.content if getattr(block, "type", "") == "text")
    return text, latency, msg.usage.input_tokens, msg.usage.output_tokens


def mock_call(version, case, rng):
    """Fake model for --dry-run. Older prompt versions make more mistakes on purpose."""
    exp = dict(case["expected"])
    error_rate = {"v1": 0.45, "v2": 0.25}.get(version[:2], 0.05)
    out = dict(exp)
    for field in FIELDS:
        if rng.random() < error_rate:
            if out[field] is None:
                out[field] = "unknown" if field != "has_website" else True   # hallucinated value
            elif field == "budget_usd":
                out[field] = f"{out[field] // 1000}k"                          # wrong type
            elif field == "services":
                out[field] = out[field][:-1]
            elif field == "phone" and out[field]:
                out[field] = f"({out[field][:3]}) {out[field][3:6]}-{out[field][6:]}"
    if version.startswith("v1") and rng.random() < 0.3:
        out["name"] = out.pop("contact_name")                                 # off-schema key
    text = json.dumps(out)
    if version.startswith("v1") and rng.random() < 0.6:
        text = "Here is the extracted data:\n```json\n" + text + "\n```"
    if rng.random() < 0.03:
        text = text[:-5]                                                      # truncated / broken JSON
    return text, rng.uniform(0.6, 1.8), 400, 90


# ---------- parsing & scoring ----------

def parse_json(text):
    """Returns (obj or None, clean). clean = the raw output was pure JSON with no extra text."""
    raw = text.strip()
    try:
        return json.loads(raw), True
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0)), False
        except json.JSONDecodeError:
            pass
    return None, False


def schema_errors(obj):
    errors = []
    if not isinstance(obj, dict):
        return ["not an object"]
    missing = [f for f in FIELDS if f not in obj]
    extra = [k for k in obj if k not in FIELDS]
    if missing:
        errors.append(f"missing: {missing}")
    if extra:
        errors.append(f"extra keys: {extra}")
    for f in ["business_name", "contact_name", "email", "phone"]:
        if f in obj and obj[f] is not None and not isinstance(obj[f], str):
            errors.append(f"{f} not string/null")
    if "industry" in obj and obj["industry"] is not None and obj["industry"] not in INDUSTRIES:
        errors.append("industry not in allowed list")
    if "services" in obj:
        if not isinstance(obj["services"], list):
            errors.append("services not a list")
        elif any(s not in SERVICES for s in obj["services"]):
            errors.append("services contain unknown values")
    if "budget_usd" in obj and obj["budget_usd"] is not None and (
            isinstance(obj["budget_usd"], bool) or not isinstance(obj["budget_usd"], (int, float))):
        errors.append("budget_usd not number/null")
    if "has_website" in obj and obj["has_website"] is not None and not isinstance(obj["has_website"], bool):
        errors.append("has_website not bool/null")
    return errors


def normalize(field, value):
    """Lenient comparison so formatting differences aren't counted as wrong answers."""
    if value is None:
        return None
    if field in ("business_name", "contact_name"):
        v = re.sub(r"\b(dr|mr|mrs|ms|dds|md)\b\.?", "", str(value).lower())
        v = v.replace("&", "and")
        return re.sub(r"[^a-z0-9]", "", v) or None
    if field == "email":
        return str(value).strip().lower()
    if field == "phone":
        digits = re.sub(r"\D", "", str(value))
        if len(digits) == 11 and digits.startswith("1"):
            digits = digits[1:]
        return digits or None
    if field == "industry":
        return str(value).strip().lower()
    if field == "services":
        return sorted({str(s).strip().lower() for s in value}) if isinstance(value, list) else "INVALID"
    if field == "budget_usd":
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return "INVALID"
    if field == "has_website":
        return value if isinstance(value, bool) else "INVALID"
    return value


def score_case(pred, expected):
    """Per-field correctness plus hallucination tracking (model filled a field that should be empty)."""
    result = {}
    halluc = 0
    empty_expected = 0
    for f in FIELDS:
        exp = normalize(f, expected[f])
        got = normalize(f, pred.get(f)) if isinstance(pred, dict) else None
        if f == "services":
            exp_empty = exp == []
            got_empty = got in (None, [])
            correct = (got == exp) or (exp_empty and got_empty)
        else:
            exp_empty = exp is None
            got_empty = got is None
            correct = got == exp
        if exp_empty:
            empty_expected += 1
            if not got_empty:
                halluc += 1
        result[f] = correct
    return result, halluc, empty_expected


# ---------- reporting ----------

def summarize(rows):
    n = len(rows)
    field_total = n * len(FIELDS)
    field_correct = sum(sum(r["fields"].values()) for r in rows)
    empty_total = sum(r["empty_expected"] for r in rows)
    halluc_total = sum(r["hallucinations"] for r in rows)
    return {
        "cases": n,
        "json_valid_pct": round(100 * sum(r["json_valid"] for r in rows) / n, 1),
        "clean_json_pct": round(100 * sum(r["clean_json"] for r in rows) / n, 1),
        "schema_valid_pct": round(100 * sum(r["schema_valid"] for r in rows) / n, 1),
        "field_accuracy_pct": round(100 * field_correct / field_total, 1),
        "exact_match_pct": round(100 * sum(all(r["fields"].values()) for r in rows) / n, 1),
        "hallucination_pct": round(100 * halluc_total / empty_total, 1) if empty_total else 0.0,
        "avg_latency_s": round(sum(r["latency"] for r in rows) / n, 2),
        "avg_output_tokens": round(sum(r["out_tokens"] for r in rows) / n, 1),
    }


def field_breakdown(rows):
    return {f: round(100 * sum(r["fields"][f] for r in rows) / len(rows), 1) for f in FIELDS}


def write_outputs(run_dir, all_rows, summaries, breakdowns, meta):
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "details.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt_version", "case_id", "json_valid", "clean_json", "schema_valid",
                    "schema_errors", "hallucinations", *[f"{x}_correct" for x in FIELDS], "raw_output"])
        for r in all_rows:
            w.writerow([r["version"], r["case_id"], r["json_valid"], r["clean_json"], r["schema_valid"],
                        "; ".join(r["schema_errors"]), r["hallucinations"],
                        *[r["fields"][x] for x in FIELDS], r["raw"]])

    keys = list(next(iter(summaries.values())).keys())
    with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prompt_version", *keys])
        for v, s in summaries.items():
            w.writerow([v, *s.values()])

    history = RESULTS_DIR / "history.csv"
    new_file = not history.exists()
    with open(history, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["timestamp", "mode", "model", "prompt_version", "prompt_hash", *keys])
        for v, s in summaries.items():
            w.writerow([meta["timestamp"], meta["mode"], meta["model"], v, meta["hashes"][v], *s.values()])

    label = "DRY RUN (fake model outputs, not real results)" if meta["mode"] == "dry-run" else f"Model: {meta['model']}"
    lines = [f"## Results ({meta['timestamp']})", "", f"_{label}. {summaries[next(iter(summaries))]['cases']} test cases._", "",
             "| Prompt | Valid JSON | Clean JSON | Schema valid | Field accuracy | Exact match | Hallucination rate |",
             "|---|---|---|---|---|---|---|"]
    for v, s in summaries.items():
        lines.append(f"| {v} | {s['json_valid_pct']}% | {s['clean_json_pct']}% | {s['schema_valid_pct']}% | "
                     f"{s['field_accuracy_pct']}% | {s['exact_match_pct']}% | {s['hallucination_pct']}% |")
    lines += ["", "### Accuracy by field", "", "| Prompt | " + " | ".join(FIELDS) + " |",
              "|---|" + "---|" * len(FIELDS)]
    for v, b in breakdowns.items():
        lines.append(f"| {v} | " + " | ".join(f"{b[x]}%" for x in FIELDS) + " |")
    (run_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Evaluate prompt versions against a labeled test set.")
    ap.add_argument("--prompts", nargs="*", help="prompt file names without .txt (default: all)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, help="only run the first N test cases")
    ap.add_argument("--dry-run", action="store_true", help="use fake outputs; no API key needed")
    args = ap.parse_args()

    cases = load_cases(args.limit)
    prompts = load_prompts(args.prompts)

    client = None
    if not args.dry_run:
        try:
            import anthropic
        except ImportError:
            raise SystemExit("Run: pip install -r requirements.txt")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("Set ANTHROPIC_API_KEY first (see README).")
        client = anthropic.Anthropic()

    rng = random.Random(42)
    timestamp = dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    all_rows, summaries, breakdowns = [], {}, {}
    consecutive_errors = 0

    for version, template in prompts.items():
        print(f"\n> {version}  ({len(cases)} cases)")
        rows = []
        for case in cases:
            prompt_text = template.replace("{notes}", case["notes"])
            try:
                if args.dry_run:
                    raw, latency, tin, tout = mock_call(version, case, rng)
                else:
                    raw, latency, tin, tout = call_claude(client, args.model, prompt_text)
            except Exception as e:  # keep going if one call fails, but stop if the API is unreachable
                raw, latency, tin, tout = f"ERROR: {e}", 0.0, 0, 0
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    raise SystemExit(f"\n\nStopped: the last 3 API calls failed. Error from the API:\n{e}\n\n"
                                     "Common fixes: check the model name (--model), your API key, or your credit balance.")
            else:
                consecutive_errors = 0

            obj, clean = parse_json(raw)
            errs = schema_errors(obj) if obj is not None else ["invalid JSON"]
            fields, halluc, empty_exp = score_case(obj, case["expected"])
            rows.append({
                "version": version, "case_id": case["id"], "raw": raw,
                "json_valid": obj is not None, "clean_json": clean and obj is not None,
                "schema_valid": obj is not None and not errs, "schema_errors": errs,
                "fields": fields, "hallucinations": halluc, "empty_expected": empty_exp,
                "latency": latency, "in_tokens": tin, "out_tokens": tout,
            })
            print("." if all(fields.values()) else "x", end="", flush=True)
        summaries[version] = summarize(rows)
        breakdowns[version] = field_breakdown(rows)
        all_rows.extend(rows)

    meta = {"timestamp": timestamp, "mode": "dry-run" if args.dry_run else "live",
            "model": "mock" if args.dry_run else args.model,
            "hashes": {v: prompt_hash(t) for v, t in prompts.items()}}
    run_dir = RESULTS_DIR / f"{timestamp}_{meta['mode']}"
    table = write_outputs(run_dir, all_rows, summaries, breakdowns, meta)
    print("\n\n" + table)
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
