from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from openai import OpenAI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test an OpenRouter chat completion.")
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--token-param", choices=["max_tokens", "max_completion_tokens", "none"], default="none")
    parser.add_argument("--include-reasoning", action="store_true")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--prompt", default="Answer in exactly one short sentence: what is 2+2?")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env}.")

    client = OpenAI(base_url=args.base_url, api_key=api_key)
    kwargs = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "extra_headers": {"X-OpenRouter-Title": "expected-answer-rag"},
    }
    if args.token_param != "none":
        kwargs[args.token_param] = args.max_output_tokens
    if args.include_reasoning:
        kwargs["include_reasoning"] = True
    if args.reasoning_effort:
        kwargs["reasoning"] = {"effort": args.reasoning_effort}

    completion = client.chat.completions.create(**kwargs)
    print(completion.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
