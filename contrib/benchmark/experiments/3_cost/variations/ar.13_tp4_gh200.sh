CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve meta-llama/Llama-2-13b-hf \
  --tensor-parallel-size 4 \
  --port 8080 \
  --no-enable-chunked-prefill \
  --no-enable-prefix-caching \
  --disable-cascade-attn \
  --async-scheduling &

echo "Waiting for vLLM server to be ready..."
for i in {1..60}; do
  if curl -s http://localhost:8080/health > /dev/null 2>&1; then
    echo "vLLM server is ready!"
    break
  fi
  echo "Waiting... ($i/60)"
  sleep 5
done

# Run workload
python3 simulator/real/run_workloads.py \
  --config meta/experiments/3_cost/ar.13.yaml \
  --output-file .local/output/ar.13_13b_4_gh200.jsonl \
  --base-url http://localhost:8080
