"""Generic LLM backend reachability check.

Supports:
  - ollama:   GET {base}/api/tags  (native Ollama API)
  - llamacpp: GET {base}/models    (OpenAI-compatible, base includes /v1)
              + optional GET {server_root}/health
  - lmstudio: GET {base}/models    (OpenAI-compatible)
  - openai:   GET {base}/models    (OpenAI-compatible; needs API key for some servers)

Base URLs resolve from explicit args first, then provider env vars:
  ollama   -> OLLAMA_BASE_URL   (default http://127.0.0.1:11434)
  llamacpp -> LLAMA_BASE_URL    (default http://127.0.0.1:8081/v1)
  lmstudio -> LMSTUDIO_BASE_URL (default http://127.0.0.1:1234/v1)
  openai   -> OPENAI_BASE_URL   (default https://api.openai.com/v1)

Usage:
    python check_backend.py --provider llamacpp --require-model daredevil-8b
    python check_backend.py --provider ollama --require-model gemma4:12b-it-qat
"""

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

PROVIDERS = ("ollama", "llamacpp", "lmstudio", "openai")

_DEFAULTS = {
    "ollama": "http://127.0.0.1:11434",
    "llamacpp": "http://127.0.0.1:8081/v1",
    "lmstudio": "http://127.0.0.1:1234/v1",
    "openai": "https://api.openai.com/v1",
}

_ENV_VARS = {
    "ollama": "OLLAMA_BASE_URL",
    "llamacpp": "LLAMA_BASE_URL",
    "lmstudio": "LMSTUDIO_BASE_URL",
    "openai": "OPENAI_BASE_URL",
}


def get_base_url(provider: str, base_url: str | None = None) -> str:
    provider = provider.lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unsupported provider '{provider}'. Supported: {list(PROVIDERS)}")
    url = base_url or os.getenv(_ENV_VARS[provider], _DEFAULTS[provider])
    return url.rstrip("/")


def _auth_headers(provider: str) -> dict:
    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY")
        if key:
            return {"Authorization": f"Bearer {key}"}
    return {}


def list_models(
    provider: str,
    base_url: str | None = None,
    timeout: float = 10.0,
) -> list[str]:
    """Return model names/ids for the given provider. Raises on failure."""
    if requests is None:
        raise RuntimeError("The 'requests' package is required to check backends.")
    provider = provider.lower()
    base = get_base_url(provider, base_url)
    headers = _auth_headers(provider)

    if provider == "ollama":
        url = f"{base}/api/tags"
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return [m.get("name") for m in data.get("models", []) if m.get("name")]

    # OpenAI-compatible: {base}/models
    url = f"{base}/models"
    resp = requests.get(url, timeout=timeout, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    models: list[str] = []
    # llama-server returns {"models": [...], "data": [...]}; OpenAI returns {"data": [...]}
    for key in ("data", "models"):
        items = data.get(key, []) if isinstance(data, dict) else []
        for m in items:
            if isinstance(m, dict):
                name = m.get("id") or m.get("name") or m.get("model")
                if name:
                    models.append(name)
            elif isinstance(m, str):
                models.append(m)
    # de-dupe, preserve order
    seen: set[str] = set()
    unique = [m for m in models if not (m in seen or seen.add(m))]
    return unique


def check_backend(
    provider: str,
    base_url: str | None = None,
    required_model: str | None = None,
    timeout: float = 10.0,
) -> list[str]:
    """Verify the backend is reachable; optionally require a model.

    Returns the list of available model names. Raises RuntimeError with an
    actionable message when unreachable or when ``required_model`` is absent.
    """
    provider = provider.lower()
    base = get_base_url(provider, base_url)
    try:
        models = list_models(provider, base, timeout=timeout)
    except Exception as e:
        hints = {
            "ollama": "Start Ollama (e.g. `ollama serve`) or set OLLAMA_BASE_URL.",
            "llamacpp": "Start llama-server (e.g. `llama-server -m <model.gguf> --port 8081`) or set LLAMA_BASE_URL.",
            "lmstudio": "Start LM Studio server or set LMSTUDIO_BASE_URL.",
            "openai": "Check network/API key or set OPENAI_BASE_URL.",
        }
        raise RuntimeError(
            f"Cannot reach {provider} backend at {base}: {e}. {hints.get(provider, '')}"
        ) from e
    logger.info(f"🤖🔌 {provider} at {base} models: {models or '(none)'}")
    if required_model and required_model not in models:
        raise RuntimeError(
            f"{provider} model '{required_model}' not found at {base}. "
            f"Available: {models or '(none)'}. "
            f"Serve it or set the model to an available one."
        )
    return models


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check LLM backend endpoint and list models.")
    parser.add_argument(
        "--provider",
        default=os.getenv("LLM_PROVIDER", "llamacpp"),
        choices=list(PROVIDERS),
    )
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--require-model", default=None)
    parser.add_argument("--timeout", type=float, default=10.0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    try:
        models = check_backend(args.provider, args.base_url, args.require_model, args.timeout)
    except (RuntimeError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"{args.provider} OK: {get_base_url(args.provider, args.base_url)}")
    for name in models:
        print(f"  - {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
