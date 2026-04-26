import time
import requests
import json
from tqdm import tqdm

import yaml
import logging
import os
import paramiko
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

from simulator.core.placement import PlacementDecisionMaker, PhysicalNodeConfig, NodeConfiguration
from simulator.core.config import WorkloadConfig
from simulator.core.arrival import PoissonProcess, GammaProcess, DeterministicProcess

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARN)

cluster_mapping = {
    "NVDA:A100_80G:SXM": "bristen",
    "NVDA:GH200": "clariden",
}

dnt_gpu_mapping = {
    "NVDA:A100_80G:SXM": "NVIDIA A100-SXM4-80GB",
    "NVDA:GH200": "NVIDIA GH200 120GB",
}

class SpinUpManager:
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config_data = yaml.safe_load(f)
            
        self.physical_nodes = self._parse_physical_nodes()
        self.workloads = self._parse_workloads()
        self.placement_strategy = self.config_data.get("placement_strategy", "maximize_replicas")
        self.placement_config = self.config_data.get("placement_config", {})
        
        # Setup Jinja2 environment
        from jinja2 import Environment, FileSystemLoader
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        self.env = Environment(loader=FileSystemLoader(template_dir))
        
    def _parse_physical_nodes(self) -> List[PhysicalNodeConfig]:
        nodes = []
        for node_cfg in self.config_data.get("nodes", []):
            nodes.append(PhysicalNodeConfig(
                gpu_type=node_cfg["gpu"],
                count=node_cfg["count"],
                gpus_per_node=node_cfg["gpus_per_node"],
                cost=node_cfg.get("cost", 0.0)
            ))
        return nodes

    def _parse_workloads(self) -> List[WorkloadConfig]:
        workloads = []
        for wl_cfg in self.config_data.get("workload", []):
            # Parse arrival rate
            arrival_str = wl_cfg["arrival_rate"]
            arrival_process = self._parse_distribution(arrival_str)
            
            # Parse input/output dists
            input_dist = self._parse_dist_tuple(wl_cfg["input"])
            output_dist = self._parse_dist_tuple(wl_cfg["output"])
            
            workloads.append(WorkloadConfig(
                model_id=wl_cfg["model"],
                arrival_process=arrival_process,
                duration=wl_cfg["duration"],
                input_dist=input_dist,
                output_dist=output_dist,
                tensor_parallel_size=wl_cfg.get("tensor-parallel-size")
            ))
        return workloads

    def _parse_distribution(self, dist_str: str):
        # Simple parser for "Type(params)"
        name = dist_str.split("(")[0]
        params_str = dist_str.split("(")[1].strip(")")
        params = [float(x.strip()) for x in params_str.split(",")]
        
        if name == "Poisson":
            return PoissonProcess(params[0])
        elif name == "Gamma":
            return GammaProcess(params[0], params[1])
        elif name == "Deterministic":
            return DeterministicProcess(params[0])
        else:
            raise ValueError(f"Unknown distribution: {name}")

    def _parse_dist_tuple(self, dist_str: str) -> Tuple[str, List[float]]:
        name = dist_str.split("(")[0]
        params_str = dist_str.split("(")[1].strip(")")
        params = [float(x.strip()) for x in params_str.split(",")]
        return (name, params)

    def run(self, dry_run: bool = False):
        logger.info("Running placement...")
        decision_maker = PlacementDecisionMaker(
            strategy_name=self.placement_strategy,
            memory_threshold=self.placement_config.get("memory_threshold", 0.8)
        )
        
        logical_nodes, metadata = decision_maker.place(self.physical_nodes, self.workloads)
        logger.info(f"Placement complete. Generated {len(logical_nodes)} logical nodes.")
        
        # Keep track of what we expect to see
        expected_nodes = []

        # Submit one job per logical node
        for i, ln in enumerate(logical_nodes):
            hardware = ln.hardware
            model = ln.model_id
            tp = ln.parallel_config.tensor_parallel_size
            
            cluster = cluster_mapping.get(hardware)
            if not cluster:
                logger.warning(f"No cluster mapping for hardware {hardware}. Skipping.")
                continue
            
            expected_nodes.append({
                "model": model,
                "hardware": hardware,
                "tp": tp
            })

            logger.info(f"Generating script for {model} on {cluster} (Job {i+1})...")
            
            # Generate script for this single instance
            script_content = self.generate_slurm_script(cluster, model, tp, i)
            
            if dry_run:
                print(f"\n--- Slurm Script for {cluster} (Job {i+1}) ---")
                print(script_content)
                print("----------------------------------\n")
            else:
                self.submit_job(cluster, script_content, i)
        
        if not dry_run:
            self.wait_for_nodes(expected_nodes)

    def wait_for_nodes(self, expected_nodes: List[Dict[str, Any]]):
        url = "http://148.187.108.173:8092/v1/dnt/table"
        total_expected = len(expected_nodes)
        logger.info(f"Waiting for {total_expected} nodes to become available at {url}...")
        
        # Group expected nodes for easier checking
        # (model, hardware) -> count
        required_counts = {}
        for node in expected_nodes:
            key = (node["model"], node["hardware"])
            required_counts[key] = required_counts.get(key, 0) + 1
            
        with tqdm(total=total_expected, desc="Waiting for nodes", unit="node") as pbar:
            while True:
                try:
                    response = requests.get(url)
                    if response.status_code != 200:
                        logger.warning(f"Failed to fetch DNT table: {response.status_code}")
                        time.sleep(5)
                        continue
                    
                    data = response.json()
                    
                    # Count current available nodes
                    current_counts = {}
                    total_found_matching = 0
                    
                    for node_id, node_data in data.items():
                        # Check services
                        services = node_data.get("service", [])
                        if not services:
                            continue
                            
                        # Assume one LLM service per node for now as per our setup
                        llm_service = next((s for s in services if s.get("name") == "llm"), None)
                        if not llm_service:
                            continue
                            
                        if llm_service.get("status") != "connected":
                            continue
                            
                        # Extract model
                        identity_group = llm_service.get("identity_group", [])
                        model_identity = next((i for i in identity_group if i.startswith("model=")), None)
                        if not model_identity:
                            continue
                        model = model_identity.split("=")[1]
                        
                        # Extract hardware
                        # We look at the first GPU name to identify the node type
                        gpus = node_data.get("hardware", {}).get("gpus", [])
                        if not gpus:
                            continue
                        gpu_name = gpus[0].get("name")
                        
                        # Map back to our config hardware names
                        # Invert dnt_gpu_mapping or check values
                        config_hardware = None
                        for k, v in dnt_gpu_mapping.items():
                            if v == gpu_name:
                                config_hardware = k
                                break
                        
                        if not config_hardware:
                            # logger.debug(f"Unknown GPU type {gpu_name}, skipping")
                            continue
                            
                        key = (model, config_hardware)
                        
                        # Only count if we actually need this type of node
                        if key in required_counts:
                            # Don't overcount if we found more than needed (optional, but good for progress bar)
                            current_needed = required_counts[key]
                            current_have = current_counts.get(key, 0)
                            
                            if current_have < current_needed:
                                current_counts[key] = current_have + 1
                                total_found_matching += 1
                    
                    # Update progress bar
                    pbar.n = total_found_matching
                    pbar.refresh()
                    
                    # Check if requirements are met
                    if total_found_matching >= total_expected:
                        logger.info("All required nodes are online!")
                        break
                    
                    time.sleep(10)
                    
                except Exception as e:
                    logger.error(f"Error checking status: {e}")
                    time.sleep(5)

    def generate_slurm_script(self, cluster: str, model: str, tp: int, job_index: int) -> str:
        template = self.env.get_template("slurm_job.j2")
        
        # Determine OCF binary and environment file based on cluster
        if "clariden" in cluster:
            ocf_binary = "/ocfbin/ocf-arm"
            env_file = "/capstor/store/cscs/swissai/infra02/xyao/envs/vllm-clariden.toml"
        elif "bristen" in cluster:
            ocf_binary = "/ocfbin/ocf-amd64"
            env_file = "/capstor/store/cscs/swissai/infra02/xyao/envs/vllm-bristen.toml"
        else:
            # Fallback or error? defaulting to amd64 for now or maybe raise warning
            logger.warning(f"Unknown cluster architecture for {cluster}, defaulting to amd64")
            ocf_binary = "/ocfbin/ocf-amd64"
            env_file = "/capstor/store/cscs/swissai/infra02/xyao/envs/vllm-bristen.toml"

        # Bootstrap address provided by user
        bootstrap_addr = "/ip4/148.187.108.173/tcp/43905/p2p/QmSxh8s3UqmSXBa9SLLREKGAQ6DYCmaeCHeBDdzpJDgn45"

        return template.render(
            job_name=f"llm-sim-{cluster}-{job_index}",
            total_nodes=1,
            time="12:00:00",
            model=model,
            tp_size=tp,
            ocf_binary=ocf_binary,
            bootstrap_addr=bootstrap_addr,
            env_file=env_file
        )

    def submit_job(self, cluster: str, script_content: str, job_index: int):
        # Write to local temp file first
        local_dir = ".local"
        os.makedirs(local_dir, exist_ok=True)
        
        local_filename = f"spinup_script_{cluster}_{job_index}.sh"
        local_path = os.path.join(local_dir, local_filename)
        local_path = os.path.abspath(local_path)
        
        with open(local_path, "w") as f:
            f.write(script_content)
        logger.info(f"Generated local script: {local_path}")

        remote_filename = f"spinup_script_{cluster}_{job_index}.sh"
        # Use relative path which defaults to home, avoiding ~ expansion issues in some SFTP implementations
        remote_path = remote_filename 
        
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        
        jump_client = None

        try:
            # Load system host keys and user config
            client.load_system_host_keys()
            
            ssh_config = paramiko.SSHConfig()
            user_config_file = os.path.expanduser("~/.ssh/config")
            if os.path.exists(user_config_file):
                with open(user_config_file) as f:
                    ssh_config.parse(f)
            
            user_config = ssh_config.lookup(cluster)
            
            connect_args = {
                'hostname': user_config.get('hostname', cluster),
                'username': user_config.get('user'),
                'port': int(user_config.get('port', 22)),
                'key_filename': user_config.get('identityfile')
            }
            
            # Filter None values
            connect_args = {k: v for k, v in connect_args.items() if v is not None}
            
            # Check for ProxyJump
            proxy_jump = user_config.get('proxyjump')
            sock = None
            
            if proxy_jump:
                logger.info(f"Found ProxyJump: {proxy_jump}")
                jump_config = ssh_config.lookup(proxy_jump)
                
                jump_client = paramiko.SSHClient()
                jump_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                jump_client.load_system_host_keys()
                
                jump_args = {
                    'hostname': jump_config.get('hostname', proxy_jump),
                    'username': jump_config.get('user'),
                    'port': int(jump_config.get('port', 22)),
                    'key_filename': jump_config.get('identityfile')
                }
                jump_args = {k: v for k, v in jump_args.items() if v is not None}
                
                logger.info(f"Connecting to jump host {proxy_jump}...")
                jump_client.connect(**jump_args)
                
                dest_addr = (connect_args['hostname'], connect_args['port'])
                sock = jump_client.get_transport().open_channel("direct-tcpip", dest_addr, ('127.0.0.1', 0))

            logger.info(f"Connecting to {cluster} ({connect_args.get('hostname')})...")
            client.connect(**connect_args, sock=sock)
            
            # SFTP script
            sftp = client.open_sftp()
            logger.info(f"Uploading {local_path} to {remote_path}...")
            sftp.put(local_path, remote_path)
            sftp.close()
            logger.info(f"Script uploaded to {remote_path}")
            
            # Submit
            command = f"sbatch {remote_path}"
            logger.info(f"Executing: {command}")
            stdin, stdout, stderr = client.exec_command(command)
            
            exit_status = stdout.channel.recv_exit_status()
            out_str = stdout.read().decode().strip()
            err_str = stderr.read().decode().strip()
            
            if exit_status == 0:
                logger.info(f"Submission successful: {out_str}")
            else:
                logger.error(f"Submission failed (exit {exit_status}): {err_str}")
                
        except Exception as e:
            logger.error(f"Failed to submit to {cluster}: {e}")
        finally:
            client.close()
            if jump_client:
                jump_client.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="meta/configs/ours.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    manager = SpinUpManager(args.config)
    manager.run(dry_run=args.dry_run)
