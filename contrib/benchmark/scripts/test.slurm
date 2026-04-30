#!/bin/bash
#SBATCH --job-name=serving-exp
#SBATCH --output=logs/model-logs-%j.out
#SBATCH --container-writable
#SBATCH --time=6:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --account=a-infra02
#SBATCH --environment=/capstor/store/cscs/swissai/a09/xyao/llm_service/clariden/osdi_exp/vllm.toml

export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export NO_PROXY=0.0.0.0,127.0.0.1,148.187.108.172

# Get Head Node Info
export HEAD_NODE=$(hostname)
export LB_PORT=8001
export VLLM_SERVER_PORT=8080

echo "Head Node: $HEAD_NODE"
echo "LB Port: $LB_PORT"
echo "vLLM Server Port: $VLLM_SERVER_PORT"

# Start Load Balancer on the first node (Head Node)
if [ "$SLURM_PROCID" -eq 0 ]; then
    echo "Starting Load Balancer... on $HEAD_NODE"
    # Assuming config.yaml is in the current directory or passed correctly
    # We need to make sure the python environment for LB is set up.
    # If the container environment has the dependencies, great.
    cd /capstor/store/cscs/swissai/a09/xyao/llm_service/clariden/osdi_exp/code
    export PYTHONPATH=/capstor/store/cscs/swissai/a09/xyao/llm_service/clariden/osdi_exp/code/cluster_experiments
    python3 -m cluster_experiments.exp1_scaling.run --config cluster_experiments/exp1_scaling/config.yaml &
    LB_PID=$!
    echo "Load Balancer started with PID $LB_PID"
    
    # Wait for LB to be ready
    sleep 10
fi

# Create a helper script to run vLLM and register
cat << EOF > run_vllm_node.sh
#!/bin/bash
# Start vLLM
HEAD_NODE_ARG=\$1
LB_PORT_ARG=\$2

echo "Starting vLLM on \$(hostname)..."
vllm serve Qwen/Qwen3-0.6B --host 0.0.0.0 --port \$VLLM_SERVER_PORT --tensor-parallel-size 4 &
VLLM_PID=\$!

# Wait for vLLM to be ready
echo "Waiting for vLLM..."
timeout 300 bash -c "while ! curl -s localhost:\$VLLM_SERVER_PORT/health > /dev/null; do sleep 5; done"

# Register with Load Balancer (running on Head Node)
MY_HOSTNAME=\$(hostname)
echo "Registering \$MY_HOSTNAME with Load Balancer at \$HEAD_NODE_ARG:\$LB_PORT_ARG"
curl -X POST "http://\$HEAD_NODE_ARG:\$LB_PORT_ARG/register" \
     -H "Content-Type: application/json" \
     -d "{\"url\": \"http://\$MY_HOSTNAME:\$VLLM_SERVER_PORT\", \"name\": \"\$MY_HOSTNAME\", \"weight\": 1}"

# Wait for vLLM process
wait \$VLLM_PID
EOF

chmod +x run_vllm_node.sh

# Run vLLM on all nodes
echo "Launching vLLM on all nodes..."

srun --ntasks-per-node=1 --output=logs/vllm-%j.out --environment=/capstor/store/cscs/swissai/a09/xyao/llm_service/clariden/osdi_exp/vllm.toml /capstor/store/cscs/swissai/a09/xyao/llm_service/clariden/osdi_exp/code/run_vllm_node.sh $HEAD_NODE $LB_PORT

# Cleanup LB when srun finishes
if [ "$SLURM_PROCID" -eq 0 ]; then
    kill $LB_PID
fi