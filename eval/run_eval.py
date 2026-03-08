"""
Evaluation script: runs 20 test queries across models and generates a comparison table.

Usage:
    uv run python eval/run_eval.py                       # all 3 models
    uv run python eval/run_eval.py --models haiku,gpt4o   # specific models
    uv run python eval/run_eval.py --dry-run              # estimate cost only

Output:
    - Markdown comparison table printed to stdout
    - CSV written to eval_results/results_YYYYMMDD_HHMMSS.csv
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root is on sys.path when run as `python eval/run_eval.py`
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.agent.agent import build_agent, query_agent
from src.config import EVAL_RESULTS_DIR, MODEL_PRICING, SUPPORTED_MODELS
from src.observability.token_tracker import TokenTrackingHandler
from src.vectorstore.chroma_store import ChromaStore

from eval.test_cases import TEST_CASES


# ── Scoring ──────────────────────────────────────────────────────────────────

def score_routing(intermediate_steps: list, expected_tools: list[str]) -> float:
    """1.0 if all expected tools were called, 0.0 otherwise."""
    if not expected_tools:
        return 1.0
    called = set()
    for step in intermediate_steps:
        action, _ = step
        called.add(action.tool)
    return 1.0 if all(t in called for t in expected_tools) else 0.0


def score_content(answer: str, expected_contains: list[str]) -> float:
    """Fraction of expected substrings found in the answer (case-insensitive).

    Commas are stripped from both sides before matching so that "93075"
    matches an answer that contains "93,075" (and vice versa).
    """
    if not expected_contains:
        return 1.0
    answer_norm = answer.lower().replace(",", "")
    hits = sum(1 for kw in expected_contains if kw.lower().replace(",", "") in answer_norm)
    return hits / len(expected_contains)


# ── Single model evaluation ──────────────────────────────────────────────────

def evaluate_model(model_key: str) -> dict:
    """Run all test cases against a single model, return aggregate metrics."""
    config = SUPPORTED_MODELS[model_key]
    provider = config["provider"]
    model_name = config["model_name"]

    tracker = TokenTrackingHandler()
    agent = build_agent(provider=provider, model_name=model_name, callbacks=[tracker])

    results = []
    total_latency = 0.0
    errors = 0

    for tc in TEST_CASES:
        tracker.reset()
        start = time.time()

        try:
            result = query_agent(agent, tc["query"], callbacks=[tracker])
            latency = time.time() - start

            intermediate_steps = result.get("intermediate_steps", [])
            routing = score_routing(intermediate_steps, tc["expected_tools"])
            content = score_content(result["answer"], tc["expected_contains"])

            if not result.get("success", True):
                errors += 1

        except Exception as e:
            latency = time.time() - start
            result = {"answer": f"ERROR: {e}", "success": False, "warning": str(e)}
            routing = 0.0
            content = 0.0
            errors += 1

        total_latency += latency
        results.append({
            "id": tc["id"],
            "category": tc["category"],
            "description": tc["description"],
            "routing_score": routing,
            "content_score": content,
            "latency": latency,
            "input_tokens": tracker.input_tokens,
            "output_tokens": tracker.output_tokens,
            "success": result["success"],
            "answer_preview": result["answer"][:200],
        })

    # Aggregate
    n = len(results)
    routing_scores = [r["routing_score"] for r in results]
    content_scores = [r["content_score"] for r in results]
    total_input = sum(r["input_tokens"] for r in results)
    total_output = sum(r["output_tokens"] for r in results)

    pricing = MODEL_PRICING.get(model_name, {"input": 0, "output": 0})
    est_cost = (total_input * pricing["input"] + total_output * pricing["output"]) / 1_000_000

    return {
        "model_key": model_key,
        "model_name": model_name,
        "routing_accuracy": sum(routing_scores) / n if n else 0,
        "content_accuracy": sum(content_scores) / n if n else 0,
        "avg_latency": total_latency / n if n else 0,
        "total_tokens": total_input + total_output,
        "est_cost": est_cost,
        "errors": errors,
        "details": results,
    }


# ── Output formatting ────────────────────────────────────────────────────────

def print_comparison_table(all_results: list[dict]) -> None:
    """Print a markdown comparison table to stdout."""
    headers = ["Metric"] + [r["model_key"] for r in all_results]

    rows = [
        ["Tool Routing Acc."] + [f"{r['routing_accuracy']:.0%}" for r in all_results],
        ["Content Accuracy"] + [f"{r['content_accuracy']:.2f}" for r in all_results],
        ["Avg Latency (s)"] + [f"{r['avg_latency']:.1f}" for r in all_results],
        ["Total Tokens"] + [f"{r['total_tokens']:,}" for r in all_results],
        ["Est. Cost ($)"] + [f"${r['est_cost']:.4f}" for r in all_results],
        ["Errors"] + [str(r["errors"]) for r in all_results],
    ]

    # Calculate column widths
    all_rows = [headers] + rows
    widths = [max(len(row[i]) for row in all_rows) for i in range(len(headers))]

    # Print
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    header_line = "| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |"
    separator = "|-" + "-|-".join("-" * w for w in widths) + "-|"
    print(header_line)
    print(separator)
    for row in rows:
        print("| " + " | ".join(v.ljust(w) for v, w in zip(row, widths)) + " |")

    print()

    # Per-question detail for each model
    for result in all_results:
        print(f"\n--- {result['model_key']} ({result['model_name']}) ---")
        print(f"{'ID':<5} {'Cat':<8} {'Route':>5} {'Content':>7} {'Time':>6}  Description")
        print("-" * 70)
        for d in result["details"]:
            print(
                f"{d['id']:<5} {d['category']:<8} "
                f"{d['routing_score']:>5.0%} {d['content_score']:>7.2f} "
                f"{d['latency']:>5.1f}s  {d['description']}"
            )


def write_csv(all_results: list[dict]) -> Path:
    """Write detailed results to CSV."""
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = EVAL_RESULTS_DIR / f"results_{timestamp}.csv"

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "id", "category", "description",
            "routing_score", "content_score", "latency_s",
            "input_tokens", "output_tokens", "success", "answer_preview",
        ])
        for result in all_results:
            for d in result["details"]:
                writer.writerow([
                    result["model_name"], d["id"], d["category"], d["description"],
                    d["routing_score"], d["content_score"], f"{d['latency']:.2f}",
                    d["input_tokens"], d["output_tokens"], d["success"],
                    d["answer_preview"],
                ])

    return path


# ── Dry run cost estimate ────────────────────────────────────────────────────

def estimate_cost(model_keys: list[str]) -> None:
    """Print estimated cost without running any queries."""
    print("\nEstimated cost (rough, based on ~2K tokens/query):")
    print("-" * 50)
    n_queries = len(TEST_CASES)
    for key in model_keys:
        config = SUPPORTED_MODELS[key]
        model_name = config["model_name"]
        pricing = MODEL_PRICING.get(model_name, {"input": 0, "output": 0})
        # Rough estimate: ~1500 input + ~500 output tokens per query, 2 LLM calls per query
        est_input = n_queries * 1500 * 2
        est_output = n_queries * 500 * 2
        est_cost = (est_input * pricing["input"] + est_output * pricing["output"]) / 1_000_000
        print(f"  {key:>10}: ~${est_cost:.4f} ({n_queries} queries × ~2 LLM calls)")
    print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run agent evaluation across models")
    parser.add_argument(
        "--models",
        default="haiku,gpt4o,deepseek",
        help="Comma-separated model keys (default: haiku,gpt4o,deepseek)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show estimated cost without running queries",
    )
    args = parser.parse_args()

    model_keys = [k.strip() for k in args.models.split(",")]

    # Validate model keys
    for key in model_keys:
        if key not in SUPPORTED_MODELS:
            print(f"Error: unknown model '{key}'. Available: {list(SUPPORTED_MODELS.keys())}")
            return

    if args.dry_run:
        estimate_cost(model_keys)
        return

    # Check documents are ingested
    try:
        store = ChromaStore()
        count = store.count()
        if count == 0:
            print("Error: no documents ingested. Run 'uv run python main.py' first.")
            return
        print(f"Found {count} document chunks in ChromaDB.\n")
    except Exception as e:
        print(f"Error connecting to ChromaDB: {e}")
        return

    # Run eval for each model
    all_results = []
    for key in model_keys:
        print(f"\n{'=' * 70}")
        print(f"Evaluating: {key} ({SUPPORTED_MODELS[key]['model_name']})")
        print(f"{'=' * 70}")

        result = evaluate_model(key)
        all_results.append(result)

        print(f"  Done — routing: {result['routing_accuracy']:.0%}, "
              f"content: {result['content_accuracy']:.2f}, "
              f"cost: ${result['est_cost']:.4f}")

    # Output
    print_comparison_table(all_results)

    csv_path = write_csv(all_results)
    print(f"\nCSV saved to: {csv_path}")


if __name__ == "__main__":
    main()
