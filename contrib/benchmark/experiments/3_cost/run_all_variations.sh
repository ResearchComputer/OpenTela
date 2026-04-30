#!/bin/bash

# Run all GH200 variation experiments in parallel
# Create logs directory
mkdir -p meta/experiments/3_cost/logs


srun --interactive --account=a-infra02 --time=08:00:00 --environment=/capstor/store/cscs/swissai/a09/xyao/llm_service/clariden/vllm.toml --pty bash -c 'cd /scratch/cscs/ibadanin/LLM-simulator/ ; export PYTHONPATH="/users/ibadanin/LLM-simulator:$PYTHONPATH"; pip install -r requirements.txt  ; ./meta/experiments/3_cost/variations/ar.4_tp1_gh200.sh' > meta/experiments/3_cost/logs/ar.4_tp1_gh200.log 2>&1 & disown
sleep 10

srun --interactive --account=a-infra02 --time=08:00:00 --environment=/capstor/store/cscs/swissai/a09/xyao/llm_service/clariden/vllm.toml --pty bash -c 'cd /scratch/cscs/ibadanin/LLM-simulator/ ; export PYTHONPATH="/users/ibadanin/LLM-simulator:$PYTHONPATH"; pip install -r requirements.txt  ; ./meta/experiments/3_cost/variations/ar.26_tp1_gh200.sh' > meta/experiments/3_cost/logs/ar.26_tp1_gh200.log 2>&1 & disown
sleep 10



echo "All GH200 experiments launched! Logs available in meta/experiments/3_cost/logs/"
echo "You can now safely close this terminal - jobs will continue running."
