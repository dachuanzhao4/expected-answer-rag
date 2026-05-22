from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe OpenRouter concurrency with tiny chat requests.")
    parser.add_argument("--model", default="openai/gpt-5-mini")
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--workers", default="1,2,4,8", help="Comma-separated concurrency levels to test.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of batches per concurrency level.")
    parser.add_argument("--sleep", type=float, default=1.5, help="Seconds to sleep between batches.")
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--prompt", default="Reply with exactly OK.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Set {args.api_key_env} before running this probe.")

    worker_counts = parse_worker_counts(args.workers)
    client = OpenRouterProbeClient(
        base_url=args.base_url.rstrip("/"),
        api_key=api_key,
        timeout=args.timeout,
    )
    key_summary = summarize_key(client.get("/key"))
    batches = []
    for workers in worker_counts:
        for repeat in range(args.repeats):
            batches.append(run_batch(client, args.model, args.prompt, workers, repeat))
            time.sleep(args.sleep)
    print(
        json.dumps(
            {
                "model": args.model,
                "key": key_summary,
                "batches": batches,
                "recommendation": recommend_workers(batches),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def parse_worker_counts(value: str) -> list[int]:
    counts = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        counts.append(max(1, int(part)))
    return counts or [1]


class OpenRouterProbeClient:
    def __init__(self, base_url: str, api_key: str, timeout: float) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://local.experiment",
            "X-OpenRouter-Title": "query2doc-mask-rag-rate-probe",
        }

    def get(self, path: str) -> dict[str, Any]:
        return self.request(path, payload=None)

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.request(path, payload=payload)

    def request(self, path: str, payload: dict[str, Any] | None) -> dict[str, Any]:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers=self.headers,
            method="GET" if payload is None else "POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                body = response.read().decode("utf-8", errors="replace")
                return {
                    "ok": True,
                    "status": response.status,
                    "elapsed_s": time.perf_counter() - started,
                    "headers": rate_headers(response.headers.items()),
                    "body": json.loads(body) if body else None,
                }
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                parsed: Any = json.loads(body)
            except json.JSONDecodeError:
                parsed = body[:500]
            return {
                "ok": False,
                "status": exc.code,
                "elapsed_s": time.perf_counter() - started,
                "headers": rate_headers(exc.headers.items()),
                "body": parsed,
            }
        except Exception as exc:  # noqa: BLE001 - probe should summarize failures.
            return {
                "ok": False,
                "status": "exception",
                "elapsed_s": time.perf_counter() - started,
                "headers": {},
                "body": repr(exc),
            }


def rate_headers(items: Any) -> dict[str, str]:
    kept = {}
    for key, value in items:
        lowered = str(key).lower()
        if lowered.startswith(("x-ratelimit", "ratelimit", "retry-after")):
            kept[lowered] = str(value)
    return kept


def summarize_key(result: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "ok": result.get("ok"),
        "status": result.get("status"),
        "rate_headers": result.get("headers", {}),
    }
    if result.get("ok") and isinstance(result.get("body"), dict):
        data = result["body"].get("data", {})
        summary["data"] = {
            "is_free_tier": data.get("is_free_tier"),
            "limit_is_null": data.get("limit") is None,
            "limit_remaining_is_null": data.get("limit_remaining") is None,
            "deprecated_rate_limit": data.get("rate_limit"),
        }
    else:
        summary["error"] = result.get("body")
    return summary


def run_batch(client: OpenRouterProbeClient, model: str, prompt: str, workers: int, repeat: int) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    started = time.perf_counter()
    with futures.ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda _: chat_once(client, payload), range(workers)))
    wall_s = time.perf_counter() - started
    ok_results = [result for result in results if result["ok"]]
    latencies = [float(result["elapsed_s"]) for result in results]
    costs = [
        float(result["usage"]["cost"])
        for result in ok_results
        if isinstance(result.get("usage"), dict) and result["usage"].get("cost") is not None
    ]
    statuses: dict[str, int] = {}
    for result in results:
        status = str(result["status"])
        statuses[status] = statuses.get(status, 0) + 1
    return {
        "workers": workers,
        "repeat": repeat,
        "wall_s": round(wall_s, 3),
        "ok": len(ok_results),
        "statuses": statuses,
        "latency_s": {
            "min": round(min(latencies), 3),
            "median": round(statistics.median(latencies), 3),
            "max": round(max(latencies), 3),
        },
        "total_cost": round(sum(costs), 6) if costs else None,
        "rate_headers_seen": [result["headers"] for result in results if result.get("headers")][:3],
        "errors": [result["error"] for result in results if not result["ok"]][:2],
    }


def chat_once(client: OpenRouterProbeClient, payload: dict[str, Any]) -> dict[str, Any]:
    result = client.post("/chat/completions", payload)
    usage = None
    content = None
    if result.get("ok") and isinstance(result.get("body"), dict):
        body = result["body"]
        usage = body.get("usage")
        try:
            content = body["choices"][0]["message"].get("content")
        except (KeyError, IndexError, TypeError, AttributeError):
            content = None
    return {
        "ok": result.get("ok"),
        "status": result.get("status"),
        "elapsed_s": result.get("elapsed_s"),
        "headers": result.get("headers", {}),
        "content": content,
        "usage": usage,
        "error": result.get("body") if not result.get("ok") else None,
    }


def recommend_workers(batches: list[dict[str, Any]]) -> dict[str, Any]:
    stable = []
    for batch in batches:
        if batch["ok"] == batch["workers"] and not any(status in batch["statuses"] for status in ("429", "402")):
            stable.append(batch)
    if not stable:
        return {"workers": 1, "reason": "No tested concurrency level completed without errors."}
    best = max(stable, key=lambda batch: batch["workers"])
    conservative = max(1, min(best["workers"], 4))
    return {
        "workers": conservative,
        "max_stable_tested": best["workers"],
        "reason": "Use the conservative value for long experiment runs; short probes can overestimate sustainable throughput.",
    }


if __name__ == "__main__":
    main()
