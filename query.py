"""
Query CLI for the Document Intelligence Agent.

Usage:
    uv run python query.py "What are the payment terms?"        # single query
    uv run python query.py                                      # interactive REPL
    uv run python query.py --model deepseek "What is 2 + 2?"   # specific model

Exit codes:
    0 — query answered successfully
    1 — agent returned success=False (no answer found)
    2 — bad arguments or startup failure
"""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()

from src.agent.agent import build_agent, query_agent
from src.config import SUPPORTED_MODELS, configure_logging
from src.observability.token_tracker import TokenTrackingHandler

configure_logging()


def _print_result(result: dict, tracker: TokenTrackingHandler, model_key: str) -> None:
    """Print answer to stdout; warnings and token stats to stderr."""
    print(result["answer"])

    warning = result.get("warning")
    if warning and warning not in result["answer"]:
        print(f"\n[Warning: {warning}]", file=sys.stderr)

    tokens = tracker.total_tokens
    if tokens:
        model_name = SUPPORTED_MODELS[model_key]["model_name"]
        cost = tracker.estimate_cost(model_name)
        print(f"\n[{tokens:,} tokens | ${cost:.4f}]", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the Document Intelligence Agent.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  query.py \"What are the payment terms?\"\n"
            "  query.py --model deepseek \"Summarize the key contracts.\"\n"
            "  query.py                    # interactive REPL"
        ),
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Question to ask. Omit to start the interactive REPL.",
    )
    parser.add_argument(
        "--model",
        default="haiku",
        choices=list(SUPPORTED_MODELS),
        help="Model to use (default: haiku).",
    )
    args = parser.parse_args()

    config = SUPPORTED_MODELS[args.model]
    print(f"Building agent ({args.model})...", file=sys.stderr)
    try:
        agent = build_agent(provider=config["provider"], model_name=config["model_name"])
    except Exception as e:
        print(f"Error: could not build agent — {e}", file=sys.stderr)
        sys.exit(2)

    tracker = TokenTrackingHandler()

    if args.query:
        # ── Single-query mode ──────────────────────────────────────────────
        tracker.reset()
        result = query_agent(agent, args.query, callbacks=[tracker])
        _print_result(result, tracker, args.model)
        sys.exit(0 if result["success"] else 1)

    else:
        # ── Interactive REPL ───────────────────────────────────────────────
        print(
            "Document Intelligence Agent — interactive mode. "
            "Type 'exit' or press Ctrl-D to quit.\n",
            file=sys.stderr,
        )
        while True:
            try:
                query = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye.", file=sys.stderr)
                break

            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Goodbye.", file=sys.stderr)
                break

            tracker.reset()
            result = query_agent(agent, query, callbacks=[tracker])
            _print_result(result, tracker, args.model)
            print()


if __name__ == "__main__":
    main()
