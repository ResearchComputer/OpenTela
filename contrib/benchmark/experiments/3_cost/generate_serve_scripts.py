#!/usr/bin/env python3
"""Generate serve scripts for all workload/tp-size combinations."""
import os
import subprocess

# Detect GPU type
def get_gpu_type():
    try:
        result = subprocess.run(
            ["python3", "-c", "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_name = result.stdout.strip()

        if "A100" in gpu_name:
            return "a100"
        elif "GH200" in gpu_name:
            return "gh200"
        else:
            return "unknown"
    except Exception as e:
        print(f"Error detecting GPU: {e}")
        return "unknown"

# Configuration
WORKLOADS = ["ar.1", "ar.2", "ar.3", "ar.4", "ar.6.5", "ar.13", "ar.26"]
TP_CONFIGS = [
    {"tp_size": 1, "cuda_devices": "0"},
    {"tp_size": 2, "cuda_devices": "0,1"},
    {"tp_size": 4, "cuda_devices": "0,1,2,3"},
]
MODEL = "meta-llama/Llama-2-13b-hf"
PORT = 8080

# Detect GPU
gpu_type = get_gpu_type()
print(f"Detected GPU type: {gpu_type}")

# Create variations directory
variations_dir = "meta/experiments/3_cost/variations"
os.makedirs(variations_dir, exist_ok=True)

# Generate scripts
for workload in WORKLOADS:
    for config in TP_CONFIGS:
        tp_size = config["tp_size"]
        cuda_devices = config["cuda_devices"]

        # Create script filename
        script_name = f"{workload}_tp{tp_size}_{gpu_type}.sh"
        script_path = os.path.join(variations_dir, script_name)

        # Generate script content
        output_file = f".local/output/{workload}_13b_{tp_size}_{gpu_type}.jsonl"
        config_file = f"meta/experiments/3_cost/{workload}.yaml"

        script_content = f"""CUDA_VISIBLE_DEVICES={cuda_devices} vllm serve {MODEL} \\
  --tensor-parallel-size {tp_size} \\
  --port {PORT} \\
  --no-enable-chunked-prefill \\
  --no-enable-prefix-caching \\
  --disable-cascade-attn \\
  --async-scheduling &

echo "Waiting for vLLM server to be ready..."
for i in {{1..60}}; do
  if curl -s http://localhost:{PORT}/health > /dev/null 2>&1; then
    echo "vLLM server is ready!"
    break
  fi
  echo "Waiting... ($i/60)"
  sleep 5
done

# Run workload
python3 simulator/real/run_workloads.py \\
  --config {config_file} \\
  --output-file {output_file} \\
  --base-url http://localhost:{PORT}
"""

        # Write script
        with open(script_path, 'w') as f:
            f.write(script_content)

        # Make executable
        os.chmod(script_path, 0o755)

        print(f"Generated: {script_path}")

print(f"\nGenerated {len(WORKLOADS) * len(TP_CONFIGS)} scripts in {variations_dir}/")
print(f"Run with: bash {variations_dir}/<script_name>")
