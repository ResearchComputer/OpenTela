import os
import time
import json
import yaml
import argparse
import logging
import requests
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import asyncio
import aiohttp
import numpy as np
from simulator.real.spinup import SpinUpManager, dnt_gpu_mapping
from simulator.utils.engine import benchmark_openai_compatible_server_with_stats
from simulator.core.placement import PlacementDecisionMaker
from simulator.utils.engine import send_single_request, _build_token_like_prompt
from collections import defaultdict

logger = logging.getLogger(__name__)
# Force root logger to INFO to override spinup's config
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO)

class WorkloadRunner:
    def __init__(self, config_path: str, output_file: str = ".local/output/benchmark_results.jsonl"):
        self.config_path = config_path
        self.output_file = output_file
        # Reuse SpinUpManager for config parsing and placement logic
        self.manager = SpinUpManager(config_path)
        
    def run_benchmarks(self, base_url: str, timeout: Optional[float] = None):
        """Run workloads using the specified base URL."""
        timeout_str = f"{timeout}s" if timeout is not None else "unlimited"
        logger.info(f"Starting workloads with base URL: {base_url} (timeout={timeout_str})")
        
        # Helper to sample from distribution tuple (name, params)
        def sample_dist(dist_tuple, min_val=1):
            name, params = dist_tuple
            if name == "Normal":
                val = np.random.normal(params[0], params[1])
            elif name == "Uniform":
                val = np.random.uniform(params[0], params[1])
            elif name == "Constant": 
                val = params[0]
            else:
                val = 100
            return max(min_val, int(val))

        all_request_specs = []
        
        # 1. Initialize requests for all models
        for workload_cfg in self.manager.workloads:
            model = workload_cfg.model_id
            logger.info(f"Preparing workload for {model}")
            logger.info(f"  Duration: {workload_cfg.duration}s")
            logger.info(f"  Arrival Rate: {workload_cfg.arrival_process.rate()} req/s")
            
            # Generate arrival times
            # We use 0 as start for generation, then add to current time for scheduling
            arrival_times = workload_cfg.arrival_process.generate_arrivals(start=0, duration=workload_cfg.duration)
            num_requests = len(arrival_times)
            logger.info(f"  Generated {num_requests} requests for {model}")
            
            for i in range(num_requests):
                input_len = sample_dist(workload_cfg.input_dist, min_val=5)
                output_len = sample_dist(workload_cfg.output_dist, min_val=5)
                # Store spec: (model_id, arrival_offset, input_len, output_len, request_index)
                all_request_specs.append({
                    "model": model,
                    "arrival_offset": arrival_times[i],
                    "input_len": input_len,
                    "output_len": output_len,
                    "request_id": i
                })

        logger.info(f"Total requests generated across all models: {len(all_request_specs)}")

        # 2. Send them all together
        async def run_all_requests():
            connector = aiohttp.TCPConnector(limit=1000)
            async with aiohttp.ClientSession(connector=connector) as session:
                start_time = time.time()
                
                async def schedule_request(spec):
                    # Wait until arrival time
                    now = time.time()
                    target_time = start_time + spec["arrival_offset"]
                    delay = target_time - now
                    if delay > 0:
                        await asyncio.sleep(delay)
                    
                    # Generate prompt content
                    prompt = _build_token_like_prompt(spec["input_len"])
                    
                    result = await send_single_request(
                        session=session,
                        model_id=spec["model"],
                        prompt=prompt,
                        max_tokens=spec["output_len"],
                        request_id=spec["request_id"],
                        base_url=base_url,
                        timeout=timeout
                    )
                    return result, spec["model"]

                tasks = []
                for spec in all_request_specs:
                    tasks.append(asyncio.create_task(schedule_request(spec)))
                
                results = await asyncio.gather(*tasks)
                return results

        try:
            results_with_model = asyncio.run(run_all_requests())
        except Exception as e:
            logger.error(f"Global workload execution failed: {e}")
            import traceback
            traceback.print_exc()
            return

        # Group results by model
        results_by_model = defaultdict(list)
        for res, model in results_with_model:
            results_by_model[model].append(res)

        # Process and write results per model
        # Ensure directory exists
        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        for model, detailed_results in results_by_model.items():
            if not detailed_results:
                continue

            # Calculate stats
            total_requests = len(detailed_results)
            successful_requests = [r for r in detailed_results if r["success"]]
            
            # Calculate duration based on actual timestamps
            if successful_requests:
                start_ts_list = [r.get("start_ts", 0) for r in successful_requests if r.get("start_ts")]
                end_ts_list = [r.get("end_ts", 0) for r in successful_requests if r.get("end_ts")]
                
                if start_ts_list and end_ts_list:
                    model_start_time = min(start_ts_list)
                    model_end_time = max(end_ts_list)
                    duration = model_end_time - model_start_time
                else:
                    duration = 0
            else:
                duration = 0

            throughput = len(successful_requests) / duration if duration > 0 else 0
            
            total_latency = sum(r["latency"] for r in successful_requests)
            avg_latency = total_latency / len(successful_requests) if successful_requests else 0
            
            avg_ttft = sum(r["time_to_first_token_seconds"] for r in successful_requests) / len(successful_requests) if successful_requests else 0
            avg_tbt = sum(r["avg_time_between_tokens_seconds"] for r in successful_requests) / len(successful_requests) if successful_requests else 0
            
            timestamp = int(time.time())
            
            global_info = {
                "overall_throughput_rps": throughput,
                "average_e2e_latency": avg_latency,
                "average_ttft": avg_ttft,
                "average_time_between_tokens": avg_tbt,
                "model": model,
                "base_url": base_url,
                "timestamp": timestamp,
                "total_requests": total_requests,
                "successful_requests": len(successful_requests)
            }
            
            # Append to output file
            with open(self.output_file, "a") as f:
                # First line: global info
                f.write(json.dumps(global_info) + "\n")
                
                # Remaining lines: per-request stats
                for r in detailed_results:
                    # Map send_single_request result keys to desired output keys
                    request_stat = {
                        "input_length": r.get("prompt_tokens", 0),
                        "output_length": r.get("completion_tokens", 0),
                        "generated_tokens": r.get("completion_tokens", 0),
                        "e2e_latency": r["latency"],
                        "ttft": r["time_to_first_token_seconds"],
                        "avg_time_between_tokens": r["avg_time_between_tokens_seconds"],
                        "success": r["success"],
                        "start_time": r.get("start_ts"),
                        "end_time": r.get("end_ts"),
                        "error": r.get("error")
                    }
                    f.write(json.dumps(request_stat) + "\n")
            
            logger.info(f"Statistics written to {self.output_file}")
            
            # Print summary to console
            print(f"\nBenchmark Summary for {model}:")
            print(f"  Total Requests: {total_requests}")
            print(f"  Successful: {len(successful_requests)}")
            print(f"  Throughput: {throughput:.2f} rps")
            print(f"  Avg Latency: {avg_latency:.4f} s")
            print(f"  Avg TTFT: {avg_ttft:.4f} s")
            print(f"  Avg TBT: {avg_tbt:.4f} s")
            print(f"Results saved to {self.output_file}")

def main():
    parser = argparse.ArgumentParser(description="Run workloads on spun-up cluster nodes.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--output-file", type=str, default=".local/output/benchmark_results.jsonl", help="File to save benchmark results.")
    parser.add_argument("--base-url", type=str, default="http://localhost:8190", help="Base URL for the LLM service.")
    parser.add_argument("--timeout", type=float, default=None, help="Request timeout in seconds. Default is unlimited.")
    args = parser.parse_args()
    runner = WorkloadRunner(args.config, args.output_file)
    
    runner.run_benchmarks(args.base_url, timeout=args.timeout)

    # Fetch and print stats
    try:
        stats_url = f"{args.base_url}/v1/stats"
        response = requests.get(stats_url)
        if response.status_code == 200:
            stats = response.json()
            print("\nServer Statistics:")
            print(json.dumps(stats, indent=2))
            
            # Write stats to output file
            with open(args.output_file, "a") as f:
                f.write(json.dumps({"server_stats": stats}) + "\n")
            logging.info(f"Server statistics appended to {args.output_file}")
        else:
            logging.warning(f"Failed to fetch stats: {response.status_code}")
    except Exception as e:
        logging.error(f"Error fetching stats: {e}")

if __name__ == "__main__":
    main()
