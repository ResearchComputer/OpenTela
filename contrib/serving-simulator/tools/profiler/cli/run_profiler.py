#!/usr/bin/env python3
"""
HTTP Request Profiler for LLM serving endpoints.

This script profiles HTTP requests to localhost:8000 and generates Chrome traces.
"""

import os
import sys
import json
import asyncio
import argparse

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.profiler import HTTPProfiler, RequestSpec

def load_trace(trace_file: str) -> list:
    """Load trace from JSONL file."""
    requests = []

    with open(trace_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                requests.append(req)
            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}")
                continue

    return requests


def build_prompt(target_tokens: int) -> str:
    """Build a simple prompt by repeating 'a ' to simulate token load."""
    return "a " * target_tokens


async def run_profiler(args):
    """Run the HTTP profiler."""
    print(f"Loading trace from: {args.input}")
    trace_data = load_trace(args.input)

    # Separate ignored (ERROR status) from valid requests
    ignored_requests = [r for r in trace_data if r.get("status") == "ERROR"]
    valid_requests = [r for r in trace_data if r.get("status") != "ERROR"]

    if args.limit > 0:
        valid_requests = valid_requests[:args.limit]

    print(f"Loaded {len(trace_data)} requests ({len(valid_requests)} valid, {len(ignored_requests)} ignored with ERROR status)")

    # Build request specifications from valid requests only
    request_specs = []
    for i, req in enumerate(valid_requests):
        # Extract fields from trace
        request_id = req.get("id", f"request_{i}")
        model = req.get("model", "meta-llama/Meta-Llama-3.1-8B-Instruct")

        # Use actual token counts from trace file
        input_tokens = req.get("reported_token_input", 50)
        output_tokens = req.get("reported_token_output",
                               req.get("model_parameters", {}).get("max_tokens", 256))

        # Always use temperature 0.0
        temperature = 0.0

        # Build prompt by repeating "a " to simulate input token load
        prompt = build_prompt(input_tokens)

        spec = RequestSpec(
            request_id=request_id,
            model=model,
            prompt=prompt,
            max_tokens=int(output_tokens),
            min_tokens=int(output_tokens),  # Force exact token generation
            temperature=temperature,
            ignore_eos=True,  # Ignore end-of-sequence to generate exact count
            api_key=args.api_key
        )
        request_specs.append(spec)

        print(f"  Request {i+1}: {input_tokens} input tokens, {output_tokens} output tokens (forced), temp={temperature:.2f}")

    # Create profiler
    profiler = HTTPProfiler(base_url=args.url)

    print(f"\n{'--' * 10} Profiling Started {'--' * 10}")
    print(f"Target URL: {args.url}")
    print(f"Total requests: {len(request_specs)}")
    if args.arrival_rate:
        print(f"Arrival rate: {args.arrival_rate} requests/second")
    else:
        print("Arrival rate: All requests sent immediately")

    # Run profiling
    await profiler.profile_requests(
        request_specs,
        arrival_rate=float(args.arrival_rate) if args.arrival_rate else None
    )

    print(f"{'--' * 10} Profiling Done {'--' * 10}\n")

    # Export results
    os.makedirs(os.path.dirname(args.trace_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.stats_output), exist_ok=True)

    profiler.export_trace(args.trace_output)
    profiler.export_stats(args.stats_output)

    print(f"Trace exported to: {args.trace_output}")
    print(f"Stats exported to: {args.stats_output}")

    # Print summary
    stats = profiler.request_stats
    successful = [s for s in stats if s.get("error") is None]
    failed = [s for s in stats if s.get("error") is not None]

    print(f"\nSummary:")
    print(f"  Successful requests: {len(successful)}/{len(stats)}")
    print(f"  Failed requests: {len(failed)}")
    print(f"  Ignored requests (ERROR status in trace): {len(ignored_requests)}")

    if successful:
        avg_latency = sum(s["total_latency"] for s in successful) / len(successful)
        avg_prefill = sum(s["prefill_time"] for s in successful if s["prefill_time"]) / len([s for s in successful if s["prefill_time"]])
        avg_decode = sum(s["decode_time"] for s in successful if s["decode_time"]) / len([s for s in successful if s["decode_time"]])
        avg_token_lat = sum(s["avg_token_latency"] for s in successful if s["avg_token_latency"]) / len([s for s in successful if s["avg_token_latency"]])
        total_tokens = sum(s["generated_tokens"] for s in successful)

        print(f"  Average total latency: {avg_latency:.3f}s")
        print(f"  Average TTFT (prefill time): {avg_prefill:.3f}s")
        print(f"  Average decode time: {avg_decode:.3f}s")
        print(f"  Average inter-token latency: {avg_token_lat*1000:.2f}ms")
        print(f"  Total tokens generated: {total_tokens}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Profile HTTP requests to LLM serving endpoint"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL trace file"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8000",
        help="Base URL of the LLM server (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=None,
        help="Request arrival rate (requests/second). If not specified, all requests sent immediately"
    )
    parser.add_argument(
        "--trace-output",
        type=str,
        default="output/trace.json",
        help="Output Chrome trace file (default: output/trace.json)"
    )
    parser.add_argument(
        "--stats-output",
        type=str,
        default="output/stats.json",
        help="Output statistics file (default: output/stats.json)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Limit the number of requests to process (default: all)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for authentication"
    )

    args = parser.parse_args()

    # Run async profiler
    asyncio.run(run_profiler(args))
