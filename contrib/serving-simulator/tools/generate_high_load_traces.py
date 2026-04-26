#!/usr/bin/env python3
"""
Generate high-load trace files for scheduler experiments.

Creates 1000-request trace files with different workload characteristics:
1. Decode-heavy: short input, long output
2. Prefill-heavy: long input, short output
3. Balanced: similar input and output lengths
4. Realistic mixed: Pareto distribution (80% short, 20% long)
"""

import json
import random
import argparse
import os
from typing import List, Dict, Any


def generate_request(
    req_id: int,
    input_range: tuple,
    output_range: tuple,
    model: str = "meta-llama/Meta-Llama-3-8B-Instruct"
) -> Dict[str, Any]:
    """Generate a single request with random input/output lengths."""
    input_length = random.randint(*input_range)
    output_length = random.randint(*output_range)

    return {
        "id": f"req_{req_id:04d}",
        "status": "DEFAULT",
        "model": model,
        "model_parameters": {
            "top_p": 1,
            "max_tokens": output_length,
            "temperature": random.choice([0, 0.7, 1.0]),
            "presence_penalty": 0,
            "frequency_penalty": 0
        },
        "reported_token_input": input_length,
        "reported_token_output": output_length
    }


def generate_decode_heavy(num_requests: int = 1000) -> List[Dict[str, Any]]:
    """
    Generate decode-heavy workload (short input, long output).

    Memory-bound workload - should favor BandwidthScheduler and
    InputOutputAdaptive schedulers.
    """
    requests = []
    for i in range(num_requests):
        req = generate_request(
            req_id=i,
            input_range=(50, 200),  # Short prompts
            output_range=(500, 1500)  # Long completions
        )
        requests.append(req)
    return requests


def generate_prefill_heavy(num_requests: int = 1000) -> List[Dict[str, Any]]:
    """
    Generate prefill-heavy workload (long input, short output).

    Compute-bound workload - should favor FLOPsScheduler and
    InputOutputAdaptive schedulers.
    """
    requests = []
    for i in range(num_requests):
        req = generate_request(
            req_id=i,
            input_range=(1000, 3000),  # Long prompts
            output_range=(50, 200)  # Short completions
        )
        requests.append(req)
    return requests


def generate_balanced(num_requests: int = 1000) -> List[Dict[str, Any]]:
    """
    Generate balanced workload (similar input and output).

    Mixed workload - should favor RooflineScheduler and
    InputOutputAdaptive schedulers.
    """
    requests = []
    for i in range(num_requests):
        req = generate_request(
            req_id=i,
            input_range=(200, 800),  # Medium prompts
            output_range=(200, 800)  # Medium completions
        )
        requests.append(req)
    return requests


def generate_realistic_mixed(num_requests: int = 1000) -> List[Dict[str, Any]]:
    """
    Generate realistic mixed workload with Pareto distribution.

    80% short requests, 20% long requests - realistic production workload.
    Should favor adaptive schedulers that handle variance.
    """
    requests = []
    num_short = int(num_requests * 0.8)
    num_long = num_requests - num_short

    # Generate short requests (80%)
    for i in range(num_short):
        req = generate_request(
            req_id=i,
            input_range=(20, 200),  # Short prompts
            output_range=(50, 300)  # Short-medium completions
        )
        requests.append(req)

    # Generate long requests (20%)
    for i in range(num_short, num_requests):
        req = generate_request(
            req_id=i,
            input_range=(500, 2000),  # Long prompts
            output_range=(500, 2000)  # Long completions
        )
        requests.append(req)

    # Shuffle to mix short and long requests
    random.shuffle(requests)

    # Re-assign IDs to maintain order
    for i, req in enumerate(requests):
        req["id"] = f"req_{i:04d}"

    return requests


def save_trace(requests: List[Dict[str, Any]], output_file: str):
    """Save requests to JSONL file (one request per line)."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, 'w') as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')

    print(f"Generated {len(requests)} requests -> {output_file}")


def print_stats(requests: List[Dict[str, Any]], workload_name: str):
    """Print statistics about generated workload."""
    inputs = [r["reported_token_input"] for r in requests]
    outputs = [r["reported_token_output"] for r in requests]

    print(f"\n{workload_name} Statistics:")
    print(f"  Requests: {len(requests)}")
    print(f"  Input tokens:  min={min(inputs):4d}, max={max(inputs):4d}, avg={sum(inputs)//len(inputs):4d}")
    print(f"  Output tokens: min={min(outputs):4d}, max={max(outputs):4d}, avg={sum(outputs)//len(outputs):4d}")
    print(f"  Total tokens: {sum(inputs) + sum(outputs):,}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate high-load trace files for scheduler experiments"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1000,
        help="Number of requests per trace file (default: 1000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="examples",
        help="Output directory for trace files (default: examples)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--workload",
        type=str,
        choices=["all", "decode-heavy", "prefill-heavy", "balanced", "realistic"],
        default="all",
        help="Which workload to generate (default: all)"
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    print(f"Generating high-load trace files (seed={args.seed})...")
    print(f"Each trace will have {args.num_requests} requests")
    print()

    workloads = {
        "decode-heavy": (generate_decode_heavy, "trace_high_load_decode_heavy.jsonl"),
        "prefill-heavy": (generate_prefill_heavy, "trace_high_load_prefill_heavy.jsonl"),
        "balanced": (generate_balanced, "trace_high_load_balanced.jsonl"),
        "realistic": (generate_realistic_mixed, "trace_high_load_realistic.jsonl")
    }

    if args.workload == "all":
        workloads_to_generate = workloads.items()
    else:
        workloads_to_generate = [(args.workload, workloads[args.workload])]

    for workload_name, (generator_func, filename) in workloads_to_generate:
        requests = generator_func(args.num_requests)
        output_file = os.path.join(args.output_dir, filename)
        save_trace(requests, output_file)
        print_stats(requests, workload_name.upper())

    print(f"\n{'='*60}")
    print("All trace files generated successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
