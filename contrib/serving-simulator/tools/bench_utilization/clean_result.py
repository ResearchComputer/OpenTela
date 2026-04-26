import json
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Clean benchmark result file by filtering specific metrics.")
    parser.add_argument("--input", required=True, help="Path to the input .jsonl file")
    parser.add_argument("--output", required=True, help="Path to the output .jsonl file")
    args = parser.parse_args()

    # The list of metric substrings to look for.
    # We will use these strings as keys in the output 'metrics' dictionary.
    targets = [
        "kv_cache_usage_perc",
        "prompt_tokens_total", 
        "generation_tokens_total"
    ]

    try:
        with open(args.input, 'r') as f_in, open(args.output, 'w') as f_out:
            for line in f_in:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    # Skip invalid lines
                    continue
                
                # Prepare output record with timestamp info from the original record
                out = {
                    "timestamp": data.get("timestamp"),
                    "iso_time": data.get("iso_time"),
                    "metrics": {}
                }
                
                # Filter and flatten metrics
                if "metrics" in data:
                    for key, value in data["metrics"].items():
                        for target in targets:
                            # Check if the target simple name is in the full prometheus key
                            if target in key:
                                # Use the simple name (target) as the key in the output
                                out["metrics"][target] = value
                
                f_out.write(json.dumps(out) + "\n")
                
    except FileNotFoundError:
        print(f"Error: File {args.input} not found.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
