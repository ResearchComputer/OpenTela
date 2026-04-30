CUDA_VISIBLE_DEVICES=0,1 vllm serve meta-llama/Llama-2-13b-hf  --tensor-parallel-size 2 --port 8080 --no-enable-chunked-prefill --no-enable-prefix-caching --disable-cascade-attn --async-scheduling

python simulator/real/run_workloads.py --config meta/experiments/3_cost/13b_2x3090.yaml --output-file .local/output/ar13_13b_2x3090_0.jsonl --base-url http://localhost:8080
