import yaml
import os
import math
import logging
import paramiko
import argparse
from jinja2 import Environment, FileSystemLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CLUSTER_MAPPING = {
    "NVDA:A100_80G:SXM": "bristen",
    "NVDA:GH200": "clariden",
}

GPUS_PER_NODE = {
    "NVDA:A100_80G:SXM": 4,
    "NVDA:GH200": 4,
}

class WorkloadSpinUpManager:
    def __init__(self, workload_path: str):
        self.workload_path = workload_path
        with open(workload_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        template_dir = os.path.join(os.path.dirname(__file__), "templates")
        self.env = Environment(loader=FileSystemLoader(template_dir))

    def run(self, dry_run: bool = False):
        placements = self.config.get("placement", [])
        
        for i, p in enumerate(placements):
            model = p["model"]
            tp = p["tensor-parallel-size"]
            gpu = p["gpus"]
            count = p["count"]
            
            gpu_memory_utilization = p.get("gpu-memory-utilization")
            
            cluster = CLUSTER_MAPPING.get(gpu)
            if not cluster:
                logger.warning(f"Unknown GPU {gpu}, skipping.")
                continue
                
            gpus_per_node = GPUS_PER_NODE.get(gpu, 4)
            # User requested to NOT pack multiple instances on a single node
            tasks_per_node = 1
            # tasks_per_node = gpus_per_node // tp
            if tasks_per_node < 1:
                 logger.error(f"TP size {tp} > GPUs per node {gpus_per_node} for {gpu}")
                 continue
                 
            total_nodes = math.ceil(count / tasks_per_node)
            
            logger.info(f"Processing {model} on {cluster}: {count} replicas, TP={tp}. Needs {total_nodes} nodes (Tasks per node: {tasks_per_node}).")
            
            script_content = self.generate_slurm_script(
                cluster=cluster,
                model=model,
                tp=tp,
                count=count,
                total_nodes=total_nodes,
                tasks_per_node=tasks_per_node,
                job_index=i,
                gpu_memory_utilization=gpu_memory_utilization
            )
            
            if dry_run:
                print(f"\n--- Script for {model} on {cluster} (Job {i}) ---")
                print(script_content)
                print("----------------------------------\n")
            else:
                self.submit_job(cluster, script_content, i)

    def generate_slurm_script(self, cluster, model, tp, count, total_nodes, tasks_per_node, job_index, gpu_memory_utilization=None):
        template = self.env.get_template("slurm_job_consolidated.j2")
        
        if "clariden" in cluster:
            ocf_binary = "/ocfbin/ocf-arm"
            env_file = "/capstor/store/cscs/swissai/infra02/xyao/envs/vllm-clariden.toml"
        else:
            ocf_binary = "/ocfbin/ocf-amd64"
            env_file = "/capstor/store/cscs/swissai/infra02/xyao/envs/vllm-bristen.toml"
        # Change to actual bootstrap address if needed. This is just a placeholder.
        bootstrap_addr = "/ip4/148.187.108.173/tcp/43905/p2p/QmSxh8s3UqmSXBa9SLLREKGAQ6DYCmaeCHeBDdzpJDgn45"
        
        return template.render(
            job_name=f"llm-sim-{cluster}-{job_index}",
            total_nodes=total_nodes,
            time="12:00:00",
            model=model,
            tp_size=tp,
            count=count,
            tasks_per_node=tasks_per_node,
            ocf_binary=ocf_binary,
            bootstrap_addr=bootstrap_addr,
            env_file=env_file,
            gpu_memory_utilization=gpu_memory_utilization
        )

    def submit_job(self, cluster: str, script_content: str, job_index: int):
        # Write to local temp file first
        local_dir = ".local"
        os.makedirs(local_dir, exist_ok=True)
        
        local_filename = f"spinup_workload_{cluster}_{job_index}.sh"
        local_path = os.path.join(local_dir, local_filename)
        local_path = os.path.abspath(local_path)
        
        with open(local_path, "w") as f:
            f.write(script_content)
        logger.info(f"Generated local script: {local_path}")

        remote_filename = f"spinup_workload_{cluster}_{job_index}.sh"
        # Use relative path which defaults to home
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    manager = WorkloadSpinUpManager(args.workload)
    manager.run(dry_run=args.dry_run)
