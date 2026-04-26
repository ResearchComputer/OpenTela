import os
import json
import time
from simulator.utils.engine import (
    benchmark_openai_compatible_server_with_stats,
    start_vllm,
    stop_vllm,
)
from simulator.utils.gpu import get_gpu_spec


models = [
    'meta-llama/Llama-3.1-8B',
    'meta-llama/Llama-2-7b-hf',
    # 'meta-llama/Llama-2-13b-hf'
]
input_seq_lens = [64, 128, 256, 512, 1024, 2048]
output_lens = [16, 32, 64, 128, 256, 512, 1024]
tp_sizes = [1, 2, 4]

def evaluate(args):
    all_results = []
    max_seq_len = 2* (max(input_seq_lens) + max(output_lens))
    if max_seq_len > 4096:
        max_seq_len = 4096
    for model in models:
        for tp_size in tp_sizes:
            pid, stdout_log = start_vllm(model, tp_size=tp_size, max_seq_len=max_seq_len)
            for input_len in input_seq_lens:
                for output_len in output_lens:
                    print(f"Benchmarking model {model} with input length {input_len} and output length {output_len}")
                    results = benchmark_openai_compatible_server_with_stats(
                    model_id=model,
                    input_prompt_len=input_len,
                    output_len=output_len,
                    base_url="http://localhost:8080/v1",
                    api_key=None,
                    request_timeout=args.timeout,
                    iterations=args.iterations,
                    warmup_iterations=args.warmup_iterations,
                )
                results['tp_size'] = tp_size
                all_results.append(results)
                time.sleep(20)  # wait for a while between benchmarks
            stop_vllm(pid)
    # hostname + timestamp
    output_file = os.path.join(args.output_dir, f"{os.uname().nodename}_{int(time.time())}_benchmark_results.json")
    with open(output_file, 'w') as f:
        json.dump({
            "benchmarks": all_results,
            "gpu_spec": get_gpu_spec(),
            "server_config": {
                "tp_size": 1,
            },
        }, f, indent=2)
    print(f"All results saved to: {output_file}")
                
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run LLM serving benchmark with statistics")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--iterations", type=int, default=8,
                       help="Number of benchmark iterations to run")
    parser.add_argument("--warmup-iterations", type=int, default=3,
                        help="Number of warmup iterations to discard from statistics")
    parser.add_argument("--output-dir", type=str, default="benchmark_results.json",
                        help="Directory to save detailed results")
    args = parser.parse_args()
    evaluate(args)
    