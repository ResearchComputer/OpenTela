import argparse
import asyncio
import logging
import time
import yaml
import re
import aiohttp
import json
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from transformers import AutoTokenizer
from simulator.core.arrival import PoissonProcess, DeterministicProcess, GammaProcess
from simulator.utils.engine import _send_single_request

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

@dataclass
class WorkloadConfig:
    model: str
    arrival_rate: float
    duration: float
    input_dist: Tuple[str, List[float]]
    output_dist: Tuple[str, List[float]]
    arrival_process: str

class RecoveryRunner:
    def __init__(self, config_path: str, base_url: str, duration: Optional[float] = None, output_path: Optional[str] = None):
        self.config_path = config_path
        self.base_url = base_url.rstrip("/")
        self.override_duration = duration
        self.output_path = output_path
        self.workloads: List[WorkloadConfig] = []
        self._parse_config()
        
        # Metrics
        self.completed_requests = 0
        self.total_completion_tokens = 0
        self.start_time = 0
        self.metrics_history = []
        self.current_instances = 0
        self.status_counts = {"2xx": 0, "4xx": 0, "5xx": 0}
        self.persistent_errors = 0

    def _parse_distribution(self, dist_str: str) -> tuple:
        """Parse distribution string like 'Normal(1024, 5)' or 'Poisson(5)'."""
        match = re.match(r"(\w+)\((.*)\)", dist_str)
        if not match:
            # Fallback for simple numbers
            try:
                return "Constant", [float(dist_str)]
            except ValueError:
                raise ValueError(f"Unknown distribution format: {dist_str}")
        
        name = match.group(1)
        params = [float(p.strip()) for p in match.group(2).split(",")]
        return name, params

    def _parse_config(self):
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        
        for w in config.get("workload", []):
            # Parse arrival rate
            arr_name, arr_params = self._parse_distribution(w["arrival_rate"])
            if arr_name == "Poisson":
                rate = arr_params[0]
            elif arr_name == "Constant":
                rate = arr_params[0]
            else:
                logger.warning(f"Unsupported arrival process {arr_name}, defaulting to Poisson with rate 1.0")
                rate = 1.0

            # Parse input/output distributions
            _, input_params = self._parse_distribution(str(w["input"]))
            input_dist = (_, input_params)
            
            _, output_params = self._parse_distribution(str(w["output"]))
            output_dist = (_, output_params)

            self.workloads.append(WorkloadConfig(
                model=w["model"],
                arrival_rate=rate,
                duration=w["duration"],
                input_dist=input_dist,
                output_dist=output_dist,
                arrival_process=arr_name
            ))

    def _get_arrival_process(self, config: WorkloadConfig):
        if config.arrival_process == "Poisson":
            return PoissonProcess(config.arrival_rate)
        elif config.arrival_process == "Deterministic": # or Constant
             return DeterministicProcess(config.arrival_rate)
        else:
            return PoissonProcess(config.arrival_rate)

    async def _monitor_throughput(self):
        """Periodically log throughput."""
        logger.info("Starting throughput monitor...")
        last_time = time.time()
        last_tokens = 0
        last_status_counts = {"2xx": 0, "4xx": 0, "5xx": 0}
        
        # Open file if needed
        f_out = None
        if self.output_path:
            f_out = open(self.output_path, "w")
        
        try:
            while True:
                await asyncio.sleep(1.0)
                current_time = time.time()
                current_tokens = self.total_completion_tokens
                
                elapsed = current_time - last_time
                delta_tokens = current_tokens - last_tokens
                
                throughput = delta_tokens / elapsed if elapsed > 0 else 0
                
                delta_status = {
                    k: self.status_counts[k] - last_status_counts[k] 
                    for k in self.status_counts
                }
                
                metric = {
                    "timestamp": current_time,
                    "instances": self.current_instances,
                    "throughput": throughput,
                    "status_codes": self.status_counts.copy(),
                    "persistent_errors": self.persistent_errors
                }
                
                logger.info(f"Throughput: {throughput:.2f} tokens/s (Total: {current_tokens} tokens) | Status (delta): {delta_status} | Persistent Errors: {self.persistent_errors}")
                
                self.metrics_history.append(metric)
                
                if f_out:
                    f_out.write(json.dumps(metric) + "\n")
                    f_out.flush()
                
                last_time = current_time
                last_tokens = current_tokens
                last_status_counts = self.status_counts.copy()
        finally:
            if f_out:
                f_out.close()

    async def _monitor_instances(self):
        """Periodically check DNT table for online instances."""
        dnt_url = "http://148.187.108.173:8092/v1/dnt/table"
        logger.info(f"Starting instance monitor polling {dnt_url}...")
        
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.get(dnt_url, timeout=2) as response:
                        if response.status == 200:
                            data = await response.json()
                            online_count = 0
                            for node_data in data.values():
                                services = node_data.get("service", [])
                                llm_service = next((s for s in services if s.get("name") == "llm"), None)
                                if llm_service and llm_service.get("status") == "connected":
                                    online_count += 1
                            self.current_instances = online_count
                            logger.info(f"Online Instances: {online_count}")
                        else:
                            logger.warning(f"Failed to fetch DNT table: {response.status}")
                except Exception as e:
                    logger.warning(f"Error checking instances: {e}")
                
                await asyncio.sleep(1.0)

    def _sample_distribution(self, name: str, params: List[float], size: int = 1) -> np.ndarray:
        if name == "Constant":
            val = np.full(size, params[0])
        elif name == "Normal":
            # Normal(mean, std)
            val = np.random.normal(params[0], params[1], size)
        elif name == "Poisson":
            val = np.random.poisson(params[0], size)
        else:
            val = np.full(size, params[0])
        return np.maximum(1, val.astype(int))

    def prepare_workloads(self) -> List[Tuple[float, WorkloadConfig, str, int]]:
        """Pre-generate all requests with exact token counts."""
        logger.info("Preparing workloads and generating prompts...")
        prepared_requests = []
        
        for w in self.workloads:
            logger.info(f"Loading tokenizer for {w.model}...")
            try:
                tokenizer = AutoTokenizer.from_pretrained(w.model)
            except Exception as e:
                logger.warning(f"Failed to load tokenizer for {w.model}: {e}. Using dummy prompt generation.")
                tokenizer = None

            process = self._get_arrival_process(w)
            duration = self.override_duration if self.override_duration else w.duration
            
            logger.info(f"Generating arrivals for model {w.model} (rate={w.arrival_rate}, duration={duration}s)")
            arrival_times = process.generate_arrivals(start=0, duration=duration)
            num_requests = len(arrival_times)
            logger.info(f"Scheduling {num_requests} requests for {w.model}")
            
            if num_requests == 0:
                continue

            # Vectorized sampling
            input_lens = np.maximum(5, self._sample_distribution(w.input_dist[0], w.input_dist[1], size=num_requests))
            output_lens = self._sample_distribution(w.output_dist[0], w.output_dist[1], size=num_requests)
            
            max_input_len = int(np.max(input_lens))
            
            base_text = "word " * (max_input_len * 2)
            base_tokens = []
            if tokenizer:
                base_tokens = tokenizer.encode(base_text)
                if len(base_tokens) < max_input_len:
                     logger.warning(f"Base text too short for {max_input_len} tokens, extending...")
                     base_text = base_text * 2
                     base_tokens = tokenizer.encode(base_text)

            # Process in chunks to save memory and use batch_decode
            chunk_size = 10000
            for i in tqdm(range(0, num_requests, chunk_size)):
                end_idx = min(i + chunk_size, num_requests)
                chunk_arrival_times = arrival_times[i:end_idx]
                chunk_input_lens = input_lens[i:end_idx]
                chunk_output_lens = output_lens[i:end_idx]
                
                if tokenizer:
                    # Prepare list of token lists for batch decode
                    batch_tokens = [base_tokens[:l] for l in chunk_input_lens]
                    prompts = tokenizer.batch_decode(batch_tokens, skip_special_tokens=True)
                else:
                    prompts = [" ".join(["word"] * l) for l in chunk_input_lens]
                
                for j, prompt in enumerate(prompts):
                    prepared_requests.append((
                        chunk_arrival_times[j],
                        w,
                        prompt,
                        chunk_output_lens[j]
                    ))
        
        # Sort by arrival time
        prepared_requests.sort(key=lambda x: x[0])
        return prepared_requests

    async def run(self):
        logger.info(f"Starting workload runner with base URL: {self.base_url}")
        
        # Prepare workloads first
        scheduled_requests = self.prepare_workloads()
        if not scheduled_requests:
            logger.warning("No requests scheduled.")
            return

        self.start_time = time.time()
        
        # Start monitors
        monitor_task = asyncio.create_task(self._monitor_throughput())
        instance_monitor_task = asyncio.create_task(self._monitor_instances())
        
        connector = aiohttp.TCPConnector(limit=1000)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            request_id = 0
            
            base_time = time.time()
            
            for arrival_offset, config, prompt, output_len in scheduled_requests:
                request_id += 1
                
                # Calculate when to fire this request
                target_time = base_time + arrival_offset
                delay = target_time - time.time()
                
                if delay > 0:
                    await asyncio.sleep(delay)
                
                # Spawn request
                task = asyncio.create_task(self._handle_request(
                    session, config, prompt, output_len, request_id
                ))
                tasks.append(task)
            
            # Wait for all requests to finish
            if tasks:
                await asyncio.gather(*tasks)
            else:
                logger.warning("No requests were scheduled!")
                
        monitor_task.cancel()
        instance_monitor_task.cancel()
        try:
            await monitor_task
            await instance_monitor_task
        except asyncio.CancelledError:
            pass
            
        total_duration = time.time() - self.start_time
        avg_throughput = self.total_completion_tokens / total_duration if total_duration > 0 else 0
        logger.info(f"Workload finished. Average throughput: {avg_throughput:.2f} tokens/s")

    async def _handle_request(self, session: aiohttp.ClientSession, config: WorkloadConfig, prompt: str, output_len: int, req_id: int):
        payload = {
            "model": config.model,
            "prompt": prompt,
            "max_tokens": int(output_len),
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True}
        }
        
        # Construct URL: base_url + /v1/completions
        # Ensure base_url doesn't end with /
        url = f"{self.base_url}/v1/completions"
        
        max_retries = 3
        for attempt in range(max_retries + 1):
            try:
                async with session.post(
                    url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=None)
                ) as response:
                    status = response.status
                    if 200 <= status < 300:
                        self.status_counts["2xx"] += 1
                    elif 400 <= status < 500:
                        self.status_counts["4xx"] += 1
                    elif 500 <= status < 600:
                        self.status_counts["5xx"] += 1

                    if response.status != 200:
                        error_text = await response.text()
                        logger.warning(f"Request {req_id} failed (Attempt {attempt+1}/{max_retries+1}): HTTP {response.status}: {error_text}")
                        if attempt < max_retries:
                            await asyncio.sleep(1 * (2 ** attempt)) # Exponential backoff
                            continue
                        else:
                            self.persistent_errors += 1
                            return

                    async for line in response.content:
                        if not line:
                            continue

                        line_str = line.decode('utf-8').strip()
                        if not line_str.startswith('data: '):
                            continue

                        data_str = line_str[6:]  # Remove 'data: ' prefix

                        if data_str == '[DONE]':
                            break

                        try:
                            chunk = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        # Check for token content
                        choices = chunk.get("choices", [])
                        if choices:
                            delta = choices[0]
                            text_piece = delta.get("text", "")
                            if text_piece:
                                self.total_completion_tokens += 1
                                
                    self.completed_requests += 1
                    return # Success
                    
            except Exception as e:
                logger.warning(f"Request {req_id} exception (Attempt {attempt+1}/{max_retries+1}): {e}")
                if attempt < max_retries:
                    await asyncio.sleep(1 * (2 ** attempt))
                else:
                    self.persistent_errors += 1


def main():
    parser = argparse.ArgumentParser(description="Run recovery workload and measure throughput.")
    parser.add_argument("--config", type=str, required=True, help="Path to workload config YAML.")
    parser.add_argument("--base-url", type=str, default="http://148.187.108.173:8092/v1/service/llm", help="Base URL of the API.")
    parser.add_argument("--duration", type=float, default=None, help="Override duration in seconds.")
    parser.add_argument("--output", type=str, default=None, help="Path to output JSONL file.")
    
    args = parser.parse_args()
    
    runner = RecoveryRunner(args.config, args.base_url, args.duration, args.output)
    asyncio.run(runner.run())

if __name__ == "__main__":
    main()
