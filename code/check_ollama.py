"""Backwards-compatible shim around check_backend for the ollama provider.

Prefer `python check_backend.py --provider ollama ...` for new code.
"""

import argparse
import logging
import sys

from check_backend import check_backend, get_base_url, list_models  # noqa: F401


def check_ollama(
    base_url: str | None = None,
    required_model: str | None = None,
    timeout: float = 10.0,
) -> list[str]:
    return check_backend("ollama", base_url, required_model, timeout)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Ollama endpoint and list models.")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--require-model", default=None)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    try:
        models = check_ollama(args.base_url, args.require_model, args.timeout)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"Ollama OK: {get_base_url('ollama', args.base_url)}")
    for name in models:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
