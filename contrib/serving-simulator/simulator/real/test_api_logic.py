import asyncio
import logging
from simulator.real.api_server import load_config, parse_backends, select_backend, state

# Mock Data
MOCK_DNT_DATA = {
    "node1": {
        "public_address": "1.2.3.4",
        "service": [
            {
                "name": "llm",
                "status": "connected",
                "host": "localhost",
                "port": "8001",
                "identity_group": ["model=meta-llama/Llama-3.3-70B-Instruct"]
            }
        ],
        "hardware": {
            "gpus": [{"name": "NVIDIA A100-SXM4-80GB"}]
        }
    },
    "node2": {
        "public_address": "5.6.7.8",
        "service": [
            {
                "name": "llm",
                "status": "connected",
                "host": "localhost",
                "port": "8002",
                "identity_group": ["model=meta-llama/Llama-3.3-70B-Instruct"]
            }
        ],
        "hardware": {
            "gpus": [{"name": "NVIDIA GH200 120GB"}]
        }
    }
}

def test_logic():
    print("Loading config...")
    load_config()
    print(f"Weights: {state.weights}")
    
    print("Parsing mock backends...")
    parse_backends(MOCK_DNT_DATA)
    print(f"Backends: {state.backends}")
    
    # Test Selection
    model = "meta-llama/Llama-3.3-70B-Instruct"
    print(f"Testing selection for {model} (1000 iterations)...")
    
    counts = {}
    for _ in range(1000):
        b = select_backend(model)
        key = b["hardware"]
        counts[key] = counts.get(key, 0) + 1
        
    print(f"Selection counts: {counts}")
    
    # Expected: GH200 (weight 2.0) should be ~2x A100 (weight 1.0)
    a100_count = counts.get("NVDA:A100_80G:SXM", 0)
    gh200_count = counts.get("NVDA:GH200", 0)
    
    ratio = gh200_count / a100_count if a100_count > 0 else 0
    print(f"Ratio GH200/A100: {ratio:.2f} (Expected ~2.0)")
    
    if 1.8 <= ratio <= 2.2:
        print("PASS: Ratio is within acceptable range.")
    else:
        print("FAIL: Ratio is off.")

    # Verify URL construction
    print("\nVerifying URL construction...")
    b = select_backend(model)
    node_id = b["node_id"]
    expected_url = f"http://148.187.108.173:8092/v1/p2p/{node_id}/v1/_service/llm/v1/chat/completions"
    
    # We can't easily check the internal logic of api_server from here without mocking more,
    # but we can check if the backend object has what we need.
    print(f"Selected backend node_id: {node_id}")
    print(f"Expected URL pattern: {expected_url}")


if __name__ == "__main__":
    test_logic()
