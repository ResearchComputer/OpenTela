import json
import os
import glob
import argparse
import statistics
from typing import List, Dict

BASE_DIR = "meta/experiments/output/exp_2/2_1"

def calculate_percentiles(data: List[float], percentiles: List[int]) -> Dict[str, float]:
    if not data:
        return {f"P{p}": 0.0 for p in percentiles}
    
    sorted_data = sorted(data)
    n = len(sorted_data)
    results = {}
    for p in percentiles:
        k = (n - 1) * (p / 100.0)
        f = int(k)
        c = int(k) + 1
        if c >= n:
            res = sorted_data[n-1]
        elif f == c:
            res = sorted_data[f]
        else:
            d0 = sorted_data[f]
            d1 = sorted_data[c]
            res = d0 + (d1 - d0) * (k - f)
        results[f"P{p}"] = res
    return results

def analyze_performance(directory: str):
    directory = os.path.join(BASE_DIR, directory)
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' does not exist.")
        return

    jsonl_files = glob.glob(os.path.join(directory, "*.jsonl"))
    if not jsonl_files:
        print(f"No .jsonl files found in '{directory}'.")
        return

    print(f"Found {len(jsonl_files)} JSONL files in '{directory}'.")

    all_requests = []
    
    # Track min start / max end for the "true" global span if we weren't normalizing
    # But here we normalize each file to 0.

    for file_path in jsonl_files:
        model_name = os.path.basename(file_path).replace(".jsonl", "")
        file_requests = []
        
        try:
            with open(file_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        # We need start_time and end_time to normalize
                        if "start_time" in record and "end_time" in record:
                            file_requests.append(record)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

        if not file_requests:
            print(f"Warning: No valid requests in {model_name}")
            continue

        # Sort by start_time to ensure we find the true first request
        file_requests.sort(key=lambda x: x["start_time"])
        
        start_offset = file_requests[0]["start_time"]
        
        print(f"Loaded {len(file_requests)} requests from {model_name}. Offset: {start_offset:.4f}")

        # Normalize and collect
        for req in file_requests:
            norm_start = req["start_time"] - start_offset
            norm_end = req["end_time"] - start_offset
            
            # Keep necessary metrics
            all_requests.append({
                "model": model_name,
                "start_time": norm_start,
                "end_time": norm_end,
                "e2e_latency": req.get("e2e_latency", 0.0),
                "ttft": req.get("ttft", 0.0),
                "generated_tokens": req.get("generated_tokens", 0)
            })

    if not all_requests:
        print("No valid requests found across all files.")
        return

    # --- Aggregation Analysis ---
    print("\n" + "="*50)
    print("AGGREGATED PERFORMANCE ANALYSIS (Simulating Concurrent Start)")
    print("="*50)

    total_requests = len(all_requests)
    
    # Calculate global duration based on normalized times
    # All start at ~0. Duration is max(end_time) - min(start_time).
    # Since we aligned all to 0, min(start_time) is 0.
    global_start = min(r["start_time"] for r in all_requests)
    global_end = max(r["end_time"] for r in all_requests)
    duration = global_end - global_start
    
    rps = total_requests / duration if duration > 0 else 0.0

    print(f"Total Requests Processed: {total_requests}")
    print(f"Effective Test Duration:  {duration:.2f} s")
    print(f"Aggregated Throughput:    {rps:.2f} rps")
    
    # Metrics
    latencies = [r["e2e_latency"] for r in all_requests]
    ttfts = [r["ttft"] for r in all_requests]
    
    lat_stats = calculate_percentiles(latencies, [50, 90, 99])
    ttft_stats = calculate_percentiles(ttfts, [50, 90, 99])
    
    print("-" * 30)
    print("E2E Latency (s):")
    print(f"  Mean: {statistics.mean(latencies):.2f}")
    print(f"  P50:  {lat_stats['P50']:.2f}")
    print(f"  P90:  {lat_stats['P90']:.2f}")
    print(f"  P99:  {lat_stats['P99']:.2f}")
    print(f"  Max:  {max(latencies):.2f}")
    
    print("-" * 30)
    print("Time To First Token (TTFT) (s):")
    print(f"  Mean: {statistics.mean(ttfts):.2f}")
    print(f"  P50:  {ttft_stats['P50']:.2f}")
    print(f"  P90:  {ttft_stats['P90']:.2f}")
    print(f"  P99:  {ttft_stats['P99']:.2f}")
    print(f"  Max:  {max(ttfts):.2f}")
    print("-" * 30)

    # --- Per-Model Analysis ---
    print("\n" + "="*50)
    print("PER-MODEL STATISTICS")
    print("="*50)
    
    # Header
    print(f"{'Model':<40} | {'RPS':>8} | {'Avg Lat':>8} | {'P50 Lat':>8} | {'P99 Lat':>8} | {'Avg TTFT':>8} | {'P99 TTFT':>8}")
    print("-" * 110)

    # Group requests by model
    model_groups = {}
    for req in all_requests:
        m = req["model"]
        if m not in model_groups:
            model_groups[m] = []
        model_groups[m].append(req)

    for model_name, reqs in model_groups.items():
        if not reqs:
            continue
            
        # Calculate model duration in this concurrent view
        # Start times are normalized to near 0, so duration is max(end) - min(start)
        m_start = min(r["start_time"] for r in reqs)
        m_end = max(r["end_time"] for r in reqs)
        m_dur = m_end - m_start
        m_rps = len(reqs) / m_dur if m_dur > 0 else 0.0
        
        m_lats = [r["e2e_latency"] for r in reqs]
        m_ttfts = [r["ttft"] for r in reqs]
        
        m_lat_stats = calculate_percentiles(m_lats, [50, 99])
        m_ttft_stats = calculate_percentiles(m_ttfts, [50, 99])
        
        print(f"{model_name:<40} | {m_rps:>8.2f} | {statistics.mean(m_lats):>8.2f} | {m_lat_stats['P50']:>8.2f} | {m_lat_stats['P99']:>8.2f} | {statistics.mean(m_ttfts):>8.2f} | {m_ttft_stats['P99']:>8.2f}")
    print("-" * 110)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aggregate performance metrics from multiple JSONL files.")
    parser.add_argument("directory", nargs='?', default="meta/experiments/output/exp_2/2_1/ours", 
                        help="Path to directory containing .jsonl files")
    args = parser.parse_args()
    
    analyze_performance(args.directory)