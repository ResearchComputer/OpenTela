import argparse
import csv
import glob
import json
import os
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description="Compile experiment results and configuration into a CSV.")
    parser.add_argument("--results-dir", default="meta/experiments/output/exp_2/2_1", help="Base directory containing result subdirectories (strategies)")
    parser.add_argument("--config-dir", default="meta/experiments/2_1_placement", help="Directory containing config YAML files")
    parser.add_argument("--output", default="compiled_results_with_config.csv", help="Output CSV file path")
    return parser.parse_args()

def load_config(config_path):
    """
    Loads the YAML configuration file and returns a mapping of model names to their placement details.
    """
    if not os.path.exists(config_path):
        print(f"Warning: Config file not found: {config_path}")
        return {}
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # placement is a list of dicts
    # We want to map model_name -> list of placements
    placement_map = {}
    if data and 'placement' in data and data['placement']:
        for item in data['placement']:
            model = item.get('model')
            if not model:
                continue
            if model not in placement_map:
                placement_map[model] = []
            placement_map[model].append(item)
    return placement_map

def format_placement_details(placements, total_requests):
    """
    Formats the placement details into a readable string.
    Example: "NVDA:A100_80G:SXM x1 (Weight: 0.11, Reqs: 7332); NVDA:GH200 x8 (Weight: 0.89, Reqs: 59322)"
    """
    details = []
    # Sort placements by weight descending to be consistent
    sorted_placements = sorted(placements, key=lambda x: x.get('weight', 0), reverse=True)
    
    for p in sorted_placements:
        gpu = p.get('gpus', 'Unknown')
        count = p.get('count', 0)
        weight = p.get('weight', 0)
        tp_size = p.get('tensor-parallel-size', 1)
        total_gpus_for_placement = tp_size * count
        
        reqs = int(total_requests * weight) if total_requests else 0
        details.append(f"{gpu} x{count} (TP={tp_size}, Total GPUs={total_gpus_for_placement}, Weight: {weight:.2f}, Reqs: {reqs})")
    
    return "; ".join(details)

def process_experiments(results_dir, config_dir, output_file):
    # Find strategy directories (subdirectories in results_dir)
    if not os.path.exists(results_dir):
        print(f"Error: Results directory does not exist: {results_dir}")
        return

    strategies = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
    
    csv_rows = []
    
    for strategy in strategies:
        strategy_dir = os.path.join(results_dir, strategy)
        config_path = os.path.join(config_dir, f"{strategy}.yaml")
        
        print(f"Processing strategy: {strategy}")
        placement_map = load_config(config_path)
        
        # Find jsonl files
        jsonl_files = glob.glob(os.path.join(strategy_dir, "*.jsonl"))
        
        for jsonl_file in jsonl_files:
            try:
                latencies = []
                
                with open(jsonl_file, 'r') as f:
                    lines = f.readlines()
                
                if not lines:
                    print(f"Skipping empty file: {jsonl_file}")
                    continue
                
                first_line = lines[0]
                
                try:
                    stats = json.loads(first_line)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON header in {jsonl_file}")
                    continue

                model_name = stats.get('model')
                
                if not model_name:
                    print(f"Skipping file with no model name in stats: {jsonl_file}")
                    continue
                
                latencies = []
                for line in lines[1:]:
                    try:
                        req = json.loads(line)
                        if 'e2e_latency' in req:
                            latencies.append(req['e2e_latency'])
                    except json.JSONDecodeError:
                        continue

                row = {
                    "strategy": strategy,
                    "model": model_name,
                    "filename": os.path.basename(jsonl_file),
                    "total_requests": stats.get("total_requests", 0),
                    "successful_requests": stats.get("successful_requests", 0),
                    "throughput_rps": stats.get("overall_throughput_rps", 0),
                    "avg_e2e_latency": stats.get("average_e2e_latency", 0),
                    "avg_ttft": stats.get("average_ttft", 0),
                    "avg_tbt": stats.get("average_time_between_tokens", 0)
                }


                
                if latencies:
                    latencies.sort()
                    n = len(latencies)
                    for i in range(1, 100):
                        idx = int(n * i / 100)
                        idx = min(idx, n - 1)
                        row[f'p{i}_e2e_latency'] = latencies[idx]
                else:
                    for i in range(1, 100):
                        row[f'p{i}_e2e_latency'] = 0
                
                # Get placement details
                a100_count = 0
                gh200_count = 0
                
                if model_name in placement_map:
                    placements = placement_map[model_name]
                    row["placement_details"] = format_placement_details(placements, row["total_requests"])
                    
                    # Calculate per-type GPU counts
                    for p in placements:
                        gpu = p.get('gpus', '')
                        tp_size = p.get('tensor-parallel-size', 1)
                        count = p.get('count', 0)
                        total_gpus = tp_size * count
                        
                        if "A100" in gpu:
                            a100_count += total_gpus
                        elif "GH200" in gpu:
                            gh200_count += total_gpus
                else:
                    # If config not found (e.g. commented out in yaml), report it
                    row["placement_details"] = "No config found in YAML"
                
                row["A100_Count"] = a100_count
                row["GH200_Count"] = gh200_count
                
                csv_rows.append(row)
                
            except Exception as e:
                print(f"Error processing {jsonl_file}: {e}")
    
    # Sort by strategy and model for better readability
    csv_rows.sort(key=lambda x: (x['strategy'], x['model']))

    # Write CSV
    if csv_rows:
        fieldnames = [
            "strategy", "model", "filename", 
            "total_requests", "successful_requests", 
            "throughput_rps", "avg_e2e_latency", "avg_ttft", "avg_tbt"
        ]
        # Add P1-P99 fields
        percentile_fields = [f'p{i}_e2e_latency' for i in range(1, 100)]
        fieldnames.extend(percentile_fields)
        
        fieldnames.extend([
            "A100_Count", "GH200_Count",
            "placement_details"
        ])
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"Successfully wrote {len(csv_rows)} rows to {output_file}")
    else:
        print("No data found.")

if __name__ == "__main__":
    args = parse_args()
    process_experiments(args.results_dir, args.config_dir, args.output)
