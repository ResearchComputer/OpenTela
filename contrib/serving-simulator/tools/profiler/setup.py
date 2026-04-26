#!/users/ibadanin/miniconda3/bin/python3
"""
Setup vLLM nodes.

Usage:
    setup.py <num> <model> [--gpu=h100] [--time=03:00:00]    # Launch N vLLM servers
    setup.py -d               # Discover nodes -> config.yaml
    setup.py -s               # Stop all vLLM jobs
    setup.py -l               # List jobs
    setup.py -p               # Ping models to check availability
"""
import subprocess
import re
import yaml
import sys
import os
import json
import requests
from openai import OpenAI

VLLM_PORT = 8080
ROUTER_PORT = 8000
DNT_TABLE_URL = "http://148.187.108.173:8092/v1/dnt/table"

def cmd(c):
    """Run command, return output."""
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True, check=True).stdout.strip()
    except:
        return None

def get_jobs():
    """Get vLLM jobs from squeue."""
    out = cmd("squeue --me --format='%i|%N|%j' --noheader")
    if not out: return []

    jobs = []
    for line in out.split('\n'):
        if not line.strip(): continue
        parts = line.split('|')
        if len(parts) == 3 and 'vllm' in parts[2].lower():
            jobs.append({'jobid': parts[0].strip(), 'nodelist': parts[1].strip(), 'name': parts[2].strip()})
    return jobs

def parse_nodes(nodelist):
    """Extract nidXXXXXX from nodelist."""
    nodes = []
    if '[' in nodelist:
        m = re.match(r'nid\[(.*?)\]', nodelist)
        if m:
            for item in m.group(1).split(','):
                if '-' in item:
                    s, e = item.split('-')
                    for n in range(int(s), int(e)+1):
                        nodes.append(f"nid{n:06d}")
                else:
                    nodes.append(f"nid{item}")
    elif nodelist.startswith('nid'):
        nodes.append(nodelist)
    return nodes

def fetch_dnt_table():
    """Fetch node hardware data from DNT table API."""
    try:
        response = requests.get(DNT_TABLE_URL, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error fetching DNT table: {e}")
        return {}

def identify_gpu_type(gpu_name):
    """Identify GPU type from name."""
    gpu_lower = gpu_name.lower()
    if "a100" in gpu_lower:
        return "a100"
    elif "gh200" in gpu_lower:
        return "gh200"
    return "unknown"

def get_node_models(node_id):
    """Fetch models from a specific node via p2p endpoint."""
    url = f"http://148.187.108.173:8092/v1/p2p/{node_id}/v1/_service/llm/v1/models"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        # Extract model IDs from the response
        if 'data' in data:
            return [model['id'] for model in data['data']]
        return []
    except Exception as e:
        print(f"    Error fetching models from {node_id[:16]}...: {e}")
        return []

def parse_dnt_nodes(fetch_models=True):
    """Parse DNT table to find nodes with 4 GPUs.

    Args:
        fetch_models: If True, fetch model info from each node's endpoint
    """
    dnt_data = fetch_dnt_table()
    nodes = []

    for node_key, node_data in dnt_data.items():
        # Extract node ID (remove leading /)
        node_id = node_data.get('id', node_key.lstrip('/'))

        # Get hardware info
        hardware = node_data.get('hardware', {})
        gpus = hardware.get('gpus', [])

        # Filter: only nodes with non-empty GPU lists and exactly 4 GPUs
        if not gpus or len(gpus) != 4:
            continue

        # Identify GPU type
        gpu_type = identify_gpu_type(gpus[0]['name'])

        # Fetch models from node endpoint if requested
        models = []
        if fetch_models and node_data.get('connected', False):
            models = get_node_models(node_id)

        nodes.append({
            'id': node_id,
            'gpu_type': gpu_type,
            'gpu_name': gpus[0]['name'],
            'gpu_count': len(gpus),
            'total_memory': gpus[0]['total_memory'],
            'models': models,
            'connected': node_data.get('connected', False)
        })

    return nodes

def discover():
    """Discover nodes from DNT table and update config.yaml."""
    print("Fetching node data from DNT table...")
    nodes = parse_dnt_nodes(fetch_models=True)

    if not nodes:
        print("No nodes with 4 GPUs found")
        return

    print(f"\nFound {len(nodes)} nodes with 4 GPUs:")
    for node in nodes:
        print(f"  {node['id'][:16]}... ({node['gpu_type'].upper()}): {node['gpu_count']}x {node['gpu_name']}")
        if node['models']:
            print(f"    Models: {', '.join(node['models'])}")

    # Load/create config
    cfg = yaml.safe_load(open('config.yaml')) if os.path.exists('config.yaml') else {
        'host': '0.0.0.0', 'port': ROUTER_PORT, 'scheduler_type': 'round_robin',
        'backend_servers': [], 'request_timeout': 300,
        'server_discovery': {'enabled': True, 'servers': [], 'port': VLLM_PORT},
        'health_check': {'enabled': True, 'interval_seconds': 30, 'timeout_seconds': 5, 'endpoint': '/health'},
        'nodes': []
    }

    # Build nodes list with hardware info
    cfg['nodes'] = []
    for node in nodes:
        if not node['connected']:
            continue

        node_config = {
            'id': node['id'],
            'models': node['models'] if node['models'] else ['unknown'],
            'hardware': {
                'gpu_type': node['gpu_type'],
                'gpu_name': node['gpu_name'],
                'gpu_count': node['gpu_count'],
                'total_memory_mb': node['total_memory']
            }
        }
        cfg['nodes'].append(node_config)

    print(f"\nAdded {len(cfg['nodes'])} connected nodes to config")
    yaml.dump(cfg, open('config.yaml', 'w'), default_flow_style=False, sort_keys=False)
    print("Config saved to config.yaml")

def stop():
    """Stop all vLLM jobs."""
    jobs = get_jobs()
    if not jobs:
        print("No vLLM jobs found")
        return

    for j in jobs:
        print(f"Stopping {j['jobid']} ({j['name']})")
        subprocess.run(f"scancel {j['jobid']}", shell=True)

def list_jobs():
    """List vLLM jobs."""
    jobs = get_jobs()
    if not jobs:
        print("No vLLM jobs")
        return

    for j in jobs:
        nodes = ', '.join(parse_nodes(j['nodelist']))
        print(f"{j['jobid']:<10} {j['name']:<30} {nodes}")

def get_models(nid, port=VLLM_PORT):
    """Get list of models from a node."""
    client = OpenAI(base_url=f"http://{nid}:{port}/v1", api_key="EMPTY")
    return [m.id for m in client.models.list().data]

def ping_node(node_id):
    """Test node with prompt and get response via p2p endpoint.

    Warning: Nodes may crash after receiving requests due to vLLM issues.
    """
    base_url = f"http://148.187.108.173:8092/v1/p2p/{node_id}/v1/_service/llm"
    client = OpenAI(base_url=base_url, api_key="EMPTY")

    # Get models
    models = [m.id for m in client.models.list().data]

    if not models:
        return [], "No models available"

    # Test with a simple prompt
    response = client.completions.create(
        model=models[0],
        prompt="Hello, how are you?",
        max_tokens=50,
        temperature=0.7
    )
    return models, response.choices[0].text

def ping():
    """Ping all nodes with prompt-response test via p2p endpoints.

    Warning: Nodes may crash after receiving requests due to vLLM issues.
    """
    if not os.path.exists('config.yaml'):
        print("config.yaml not found. Run: setup.py -d")
        return

    cfg = yaml.safe_load(open('config.yaml'))

    # Get nodes from the 'nodes' section (new format)
    nodes = cfg.get('nodes', [])

    if not nodes:
        print("No nodes in config.yaml. Run: setup.py -d")
        return

    print(f"Pinging {len(nodes)} nodes via p2p endpoints...")
    print("Warning: Nodes may crash after receiving requests\n")

    for node in nodes:
        node_id = node['id']
        print(f"Node: {node_id[:16]}... ({node['hardware']['gpu_type'].upper()})")
        print(f"  Hardware: {node['hardware']['gpu_count']}x {node['hardware']['gpu_name']}")

        try:
            models, response = ping_node(node_id)
            print(f"  Models: {models}")
            print(f"  Prompt: Hello, how are you?")
            print(f"  Response: {response}")
        except Exception as e:
            print(f"  Error: {e}")
        print()

def launch(n, model, gpu_type='h100', time='03:00:00'):
    """Launch N vLLM servers.

    Args:
        n: Number of servers to launch
        model: Model name to serve
        gpu_type: GPU type ('h100' for clariden/arm, 'a100' for bristen/amd64)
        time: Time limit for the job (default: '03:00:00')
    """
    print(f"Launching {n}x {model} on port 8080")

    # Determine binary and toml based on GPU type
    if gpu_type.lower() == 'a100':
        ocf_binary = '/ocfbin/ocf-amd64'
        cluster = 'bristen'
        toml_file = '/capstor/store/cscs/swissai/a09/xyao/llm_service/clariden/vllm-amd64.toml'
    else:  # h100 or default to clariden
        ocf_binary = '/ocfbin/ocf-arm'
        cluster = 'clariden'
        toml_file = '/capstor/store/cscs/swissai/a09/xyao/llm_service/clariden/vllm.toml'

    print(f"Using {cluster} cluster with {ocf_binary}, time limit: {time}")

    for i in range(n):
        bash_cmd = f'{ocf_binary} start --bootstrap.addr /ip4/148.187.108.173/tcp/43905/p2p/QmZkXtkXXSuBapkT7jKk631RJYGdSzHRTfiSuszZM5ZzUr --subprocess "vllm serve {model} --no-enable-chunked-prefill --no-enable-prefix-caching --disable-cascade-attn --async-scheduling --host 0.0.0.0 --port 8080 --tensor-parallel-size 4" --service.name llm --service.port 8080'

        full_cmd = [
            "srun", "--interactive", "--account=a-infra02", f"--time={time}",
            f"--environment={toml_file}", "--pty",
            "bash", "-c", bash_cmd
        ]

        print(f"\nExecuting command {i+1}/{n}:")
        print(" ".join(full_cmd))
        print()

        subprocess.Popen(full_cmd)

    print(f"Launched {n} servers. Run: setup.py -d")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    arg = sys.argv[1]

    if arg == '-d': discover()
    elif arg == '-s': stop()
    elif arg == '-l': list_jobs()
    elif arg == '-p': ping()
    elif len(sys.argv) >= 3 and arg.isdigit():
        # Parse launch command: <num> <model> [--gpu=h100] [--time=03:00:00]
        n = int(arg)
        model = sys.argv[2]
        gpu_type = 'h100'
        time = '03:00:00'

        # Parse optional arguments
        for arg in sys.argv[3:]:
            if arg.startswith('--gpu='):
                gpu_type = arg.split('=', 1)[1]
            elif arg.startswith('--time='):
                time = arg.split('=', 1)[1]

        launch(n, model, gpu_type, time)
    else: print(__doc__)
