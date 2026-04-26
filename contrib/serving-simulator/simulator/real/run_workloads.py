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
        async def run_single_workload(workload_cfg):
            model = workload_cfg.model_id
            logger.info(f"Running workload for {model}")
            logger.info(f"  Duration: {workload_cfg.duration}s")
            logger.info(f"  Arrival Rate: {workload_cfg.arrival_process.rate()} req/s")
            
            # Generate arrival times
            start_time = time.time()
            # We use 0 as start for generation, then add to current time for scheduling
            arrival_times = workload_cfg.arrival_process.generate_arrivals(start=0, duration=workload_cfg.duration)
            num_requests = len(arrival_times)
            logger.info(f"  Generated {num_requests} requests")
            
            # Generate input/output lengths
            input_lens = []
            output_lens = []
            
            # Helper to sample from distribution tuple (name, params)
            def sample_dist(dist_tuple, min_val=1):
                name, params = dist_tuple
                if name == "Normal":
                    val = np.random.normal(params[0], params[1])
                elif name == "Uniform":
                    val = np.random.uniform(params[0], params[1])
                elif name == "Constant": # Assuming Constant exists or fallback
                    val = params[0]
                else:
                    # Fallback or default
                    val = 100
                return max(min_val, int(val))

            for _ in range(num_requests):
                input_lens.append(sample_dist(workload_cfg.input_dist, min_val=1))
                output_lens.append(sample_dist(workload_cfg.output_dist, min_val=5))
                
            # Prepare requests
            tasks = []
            
            # Create a session
            connector = aiohttp.TCPConnector(limit=1000) # High limit for concurrency
            async with aiohttp.ClientSession(connector=connector) as session:
                
                # We need to schedule tasks. 
                # Simple approach: sleep until arrival time.
                
                async def schedule_request(idx, arrival_offset):
                    # Wait until arrival time
                    now = time.time()
                    target_time = start_time + arrival_offset
                    delay = target_time - now
                    if delay > 0:
                        await asyncio.sleep(delay)
                    
                    # Generate prompt content (dummy)
                    prompt = _build_token_like_prompt(input_lens[idx])
                    
                    return await send_single_request(
                        session=session,
                        model_id=model,
                        prompt=prompt,
                        max_tokens=output_lens[idx],
                        request_id=idx,
                        base_url=base_url,
                        timeout=timeout
                    )

                # Create all tasks
                for i in range(num_requests):
                    tasks.append(asyncio.create_task(schedule_request(i, arrival_times[i])))
                
                # Wait for all tasks
                results = await asyncio.gather(*tasks)
                actual_end_time = time.time()
                
            return results, start_time, actual_end_time

        # Run all workloads
        for workload_cfg in self.manager.workloads:
            model = workload_cfg.model_id
            try:
                detailed_results, start_time, end_time = asyncio.run(run_single_workload(workload_cfg))
                
                if not detailed_results:
                    continue

                # Process and write results
                timestamp = int(time.time())
                # Ensure directory exists
                output_dir = os.path.dirname(self.output_file)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                # Calculate global stats
                total_requests = len(detailed_results)
                successful_requests = [r for r in detailed_results if r["success"]]
                
                # Throughput
                # duration = workload_cfg.duration # Old way: based on config
                duration = end_time - start_time # New way: based on actual execution time
                throughput = len(successful_requests) / duration if duration > 0 else 0
                
                # Re-calculating stats from results
                total_latency = sum(r["latency"] for r in successful_requests)
                avg_latency = total_latency / len(successful_requests) if successful_requests else 0
                
                avg_ttft = sum(r["time_to_first_token_seconds"] for r in successful_requests) / len(successful_requests) if successful_requests else 0
                avg_tbt = sum(r["avg_time_between_tokens_seconds"] for r in successful_requests) / len(successful_requests) if successful_requests else 0
                
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
                            "input_length": r.get("prompt_tokens", 0), # Approx, actual tokens used
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

            except Exception as e:
                logger.error(f"Workload execution failed for {model}: {e}")
                import traceback
                traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Run workloads on spun-up cluster nodes.")
    parser.add_argument("--config", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--output-file", type=str, default=".local/output/benchmark_results.jsonl", help="File to save benchmark results.")
    parser.add_argument("--base-url", type=str, default="http://148.187.108.173:8092/v1/service/llm", help="Base URL for the LLM service.")
    parser.add_argument("--timeout", type=float, default=None, help="Request timeout in seconds. Default is unlimited.")
    args = parser.parse_args()
    runner = WorkloadRunner(args.config, args.output_file)
    
    runner.run_benchmarks(args.base_url, timeout=args.timeout)

if __name__ == "__main__":
    main()
