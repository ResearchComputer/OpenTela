import os
import json
import time
from simulator.utils.engine import (
    start_vllm,
    stop_vllm,
    bench_throughput_sync,
)
from simulator.utils.gpu import get_gpu_spec

models = ["meta-llama/Llama-2-13b-hf"]
input_seq_lens = [1]
output_lens = [1024]
tp_sizes = [1]
batch_sizes = [32]

def evaluate(args):
    all_results = []
    max_seq_len = 4096
    for model in models:
        for tp_size in tp_sizes:
            pid, stdout_log = start_vllm(
                model, tp_size=tp_size, max_seq_len=max_seq_len
            )
            for input_len in input_seq_lens:
                for output_len in output_lens:
                    for bsz in batch_sizes:
                        print(
                            f"Benchmarking model {model} with input length {input_len} and output length {output_len} and batch size {bsz}"
                        )
                        results = bench_throughput_sync(
                            model_id=model,
                            input_prompt_len=input_len,
                            output_len=output_len,
                            base_url="http://localhost:8080",
                            batch_size=bsz
                        )
                        results["tp_size"] = tp_size
                        all_results.append(results)
                        time.sleep(20)
            stop_vllm(pid)
    # hostname + timestamp
    output_file = os.path.join(
        args.output_dir,
        f"{os.uname().nodename}_{int(time.time())}_bsz_benchmark_results.json",
    )
    with open(output_file, "w") as f:
        json.dump(
            {
                "benchmarks": all_results,
                "gpu_spec": get_gpu_spec(),
            },
            f,
            indent=2,
        )
    print(f"All results saved to: {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LLM serving benchmark with statistics"
    )
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument(
        "--iterations",
        type=int,
        default=8,
        help="Number of benchmark iterations to run",
    )
    parser.add_argument(
        "--warmup-iterations",
        type=int,
        default=3,
        help="Number of warmup iterations to discard from statistics",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results.json",
        help="Directory to save detailed results",
    )
    args = parser.parse_args()
    evaluate(args)
