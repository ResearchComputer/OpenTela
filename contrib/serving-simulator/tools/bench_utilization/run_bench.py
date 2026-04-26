import argparse
import json
import logging
import os
import subprocess
import threading
import time
import requests
import sys
from datetime import datetime

# Configuration
DNT_URL = "http://148.187.108.173:8092/v1/dnt/table"
METRICS_URL_TEMPLATE = "http://148.187.108.173:8092/v1/p2p/{node_id}/v1/_service/llm/metrics"
OPENAI_API_BASE = "http://148.187.108.173:8092/v1/service/llm/v1"

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def get_node_id(model_name):
    """
    Fetch DNT table and find the node_id associated with the given model name.
    """
    try:
        logger.info(f"Fetching DNT table from {DNT_URL}...")
        response = requests.get(DNT_URL)
        response.raise_for_status()
        data = response.json()
        
        for node_id, node_data in data.items():
            # Strip leading slash if present
            if node_id.startswith("/"):
                node_id_clean = node_id[1:]
            else:
                node_id_clean = node_id
                
            services = node_data.get("service", [])
            if not services:
                continue
            for service in services:
                if service.get("name") == "llm":
                    identity_group = service.get("identity_group", [])
                    # Look for model=... in identity_group
                    for identity in identity_group:
                        if identity.startswith("model="):
                            current_model = identity.split("=", 1)[1]
                            if current_model == model_name:
                                logger.info(f"Found node {node_id_clean} for model {model_name}")
                                return node_id_clean
        
        logger.error(f"No node found for model: {model_name}")
        return None

    except Exception as e:
        logger.error(f"Error fetching or parsing DNT table: {e}")
        return None

def collect_metrics(node_id, interval, stop_event, output_file):
    """
    Periodically collect metrics from the node and save to file.
    """
    url = METRICS_URL_TEMPLATE.format(node_id=node_id)
    logger.info(f"Starting metrics collection from {url} every {interval}s")
    
    metrics_buffer = []
    
    while not stop_event.is_set():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                # Prometheus metrics are text-based, not JSON
                metric_text = response.text
                metrics_parsed = {}
                for line in metric_text.splitlines():
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if len(parts) >= 2:
                        key = parts[0]
                        try:
                            val = float(parts[1])
                        except ValueError:
                            val = parts[1]
                        metrics_parsed[key] = val

                # Add local timestamp
                record = {
                    "timestamp": time.time(),
                    "iso_time": datetime.now().isoformat(),
                    "metrics": metrics_parsed
                }
                metrics_buffer.append(record)
            else:
                logger.warning(f"Failed to fetch metrics: {response.status_code}")
        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
        
        # Sleep in small chunks to allow quick stopping
        for _ in range(int(interval * 10)):
            if stop_event.is_set():
                break
            time.sleep(0.1)
            
    logger.info(f"Metrics collection stopped. Writing {len(metrics_buffer)} records to {output_file}...")
    try:
        with open(output_file, 'w') as f:
            for record in metrics_buffer:
                f.write(json.dumps(record) + "\n")
        logger.info("Metrics written successfully.")
    except Exception as e:
        logger.error(f"Error writing metrics to file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Run eval script and collect metrics.")
    parser.add_argument("--model", required=True, help="Model name to benchmark")
    parser.add_argument("--tasks", required=True, help="Tasks string for lm_eval")
    parser.add_argument("--interval", type=float, default=1.0, help="Metrics collection interval in seconds")
    parser.add_argument("--output", default="metrics.jsonl", help="Output file for metrics")
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['OPENAI_API_KEY'] = 'test'
    os.environ['OPENAI_API_BASE'] = OPENAI_API_BASE
    
    # 1. Get Node ID
    node_id = get_node_id(args.model)
    if not node_id:
        logger.error("Could not determine node_id. Exiting.")
        sys.exit(1)
        
    # 2. Start Metrics Collection
    stop_event = threading.Event()
    metrics_thread = threading.Thread(
        target=collect_metrics,
        args=(node_id, args.interval, stop_event, args.output)
    )
    metrics_thread.start()
    
    # 3. Run Eval Script
    # Construct the command
    # Using the template provided:
    # lm_eval --model openai-completions --model_args model= {model_name} --tasks {tasks}
    
    # To handle potential spaces in args properly if passed via shell=True, 
    # but here we construct the string as requested.
    cmd = f"/mnt/scratch/xiayao/mamba/envs/pg/bin/lm_eval --model local-completions --tasks {args.tasks} --model_args model={args.model},base_url={OPENAI_API_BASE}/completions,num_concurrent=128,max_retries=5,tokenized_requests=False,timeout=99999999999,max_length=262144"
    logger.info(f"Running command: {cmd}")
    
    try:
        # Run subprocess and wait
        process = subprocess.run(cmd, shell=True)
        return_code = process.returncode
        if return_code != 0:
            logger.error(f"Eval script failed with return code {return_code}")
        else:
            logger.info("Eval script completed successfully.")
            
    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    except Exception as e:
        logger.error(f"Error running eval script: {e}")
    finally:
        # 4. Stop Metrics Collection
        logger.info("Stopping metrics collection...")
        stop_event.set()
        metrics_thread.join()

if __name__ == "__main__":
    main()