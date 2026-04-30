# python3 -m cluster_experiments.exp1_scaling.run --config cluster_experiments/exp1_scaling/config.yaml
curl -X POST "http://localhost:8001/register" \
     -H "Content-Type: application/json" \
     -d "{\"url\": \"http://localhost:8000\", \"name\": \"localhost\", \"weight\": 1}"
