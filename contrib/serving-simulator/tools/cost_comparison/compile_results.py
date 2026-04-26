import argparse
import json
import csv
import os
import re
import glob

# Define cost mapping for different hardware setups (Hourly cost in USD)
# You can update these values based on your actual costs
HARDWARE_COSTS = {
    "1x3090": 0.0585,
    "2x3090": 0.0585 * 2,
    "4x3090": 0.0585 * 4,
    "1xa100": 0.145,
    "2xa100": 0.145 * 2,
    "4xa100": 0.145 * 4,
    "1xh100": 0.288,
    "2xh100": 0.288 * 2,
    "4xh100": 0.288 * 4,
    "1xgh200": 1.49,
    "2xgh200": 1.49 * 2,
    "4xgh200": 1.49 * 4,
}

def parse_filename(filename):
    """
    Parses the filename to extract experiment parameters.
    Expected format: ar.{rate}_{model}_{hardware}_{run}.jsonl or ar.{rate}_{model}_{hardware}.jsonl
    Example: ar.13_13b_1xh100_0.jsonl or ar.13_13b_1xa100.jsonl
    """
    basename = os.path.basename(filename)
    # Regex to match the pattern
    # Handles both with and without run_id
    match = re.match(r"ar\.([\d\.]+)_([^_]+)_([^_]+)(?:_(\d+))?\.jsonl", basename)
    if match:
        return {
            "arrival_rate": float(match.group(1)),
            "model_size": match.group(2),
            "hardware": match.group(3),
            "run_id": int(match.group(4)) if match.group(4) else 0
        }
    return None

def get_hardware_cost(hardware_setup):
    """
    Returns the cost for the given hardware setup.
    """
    if hardware_setup not in HARDWARE_COSTS:
        raise ValueError(f"Unknown hardware setup: {hardware_setup}")
    return HARDWARE_COSTS[hardware_setup]

def process_files(input_dir, output_file):
    results = []
    
    # Find all jsonl files in the directory
    files = glob.glob(os.path.join(input_dir, "*.jsonl"))
    
    print(f"Found {len(files)} files in {input_dir}")

    for file_path in files:
        try:
            # Parse filename for metadata
            metadata = parse_filename(file_path)
            if not metadata:
                print(f"Skipping file with unexpected name format: {file_path}")
                continue

            # Read the first line for summary statistics
            with open(file_path, 'r') as f:
                first_line = f.readline()
                if not first_line:
                    print(f"Skipping empty file: {file_path}")
                    continue
                
                try:
                    summary = json.loads(first_line)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {file_path}")
                    continue

            # Calculate costs
            hardware = metadata['hardware']
            hourly_cost = get_hardware_cost(hardware)
            throughput_rps = summary.get('overall_throughput_rps', 0)
            
            # Calculate Cost per 1k Requests
            # Cost per second = Hourly Cost / 3600
            # Cost per request = Cost per second / RPS
            if throughput_rps > 0:
                cost_per_1k_req = (hourly_cost / 3600) / throughput_rps * 1000
            else:
                cost_per_1k_req = float('inf')

            # Combine all data
            row = {
                "filename": os.path.basename(file_path),
                "arrival_rate": metadata['arrival_rate'],
                "model_size": metadata['model_size'],
                "hardware": metadata['hardware'],
                "run_id": metadata['run_id'],
                "throughput_rps": throughput_rps,
                "avg_e2e_latency": summary.get('average_e2e_latency', 0),
                "avg_ttft": summary.get('average_ttft', 0),
                "avg_tbt": summary.get('average_time_between_tokens', 0),
                "total_requests": summary.get('total_requests', 0),
                "successful_requests": summary.get('successful_requests', 0),
                "hourly_cost": hourly_cost,
                "cost_per_1k_req": cost_per_1k_req
            }
            results.append(row)

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")

    # Sort results by hardware and arrival rate
    results.sort(key=lambda x: (x['hardware'], x['arrival_rate']))

    # Write to CSV
    if results:
        fieldnames = [
            "filename", "hardware", "model_size", "arrival_rate", "run_id",
            "throughput_rps", "avg_e2e_latency", "avg_ttft", "avg_tbt",
            "total_requests", "successful_requests",
            "hourly_cost", "cost_per_1k_req"
        ]
        
        with open(output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Successfully wrote results to {output_file}")
    else:
        print("No results found to write.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile experiment results into a CSV file.")
    parser.add_argument("input_dir", nargs='?', default="meta/experiments/output/exp_3", help="Directory containing experiment output JSONL files")
    parser.add_argument("--output", default="compiled_results.csv", help="Output CSV file path")
    
    args = parser.parse_args()
    
    process_files(args.input_dir, args.output)
