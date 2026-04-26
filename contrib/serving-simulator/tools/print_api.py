import requests
import json
from tabulate import tabulate
from collections import defaultdict

ENDPOINT = "http://148.187.108.173:8092/v1/dnt/table"

def get_model_from_identity(identity_group):
    for identity in identity_group:
        if identity.startswith("model="):
            return identity.split("=", 1)[1]
    return "Unknown Model"

def main():
    try:
        response = requests.get(ENDPOINT)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"Error communicating with API: {e}")
        return

    # Dictionary to store counts: {(model, gpu_type): count}
    stats = defaultdict(int)

    for node_id, node_data in data.items():
        # Determine GPU Type
        gpus = node_data.get("hardware", {}).get("gpus", [])
        if gpus:
            # Assuming all GPUs on a node are the same type
            gpu_type = gpus[0].get("name", "Unknown GPU")
        else:
            gpu_type = "No GPU"

        # Check services
        services = node_data.get("service", [])
        if not services:
            # If no services, we might still want to count the node as idle? 
            # The user asked for "instances of each model", so we only care about running services.
            continue

        for service in services:
            if service.get("name") == "llm":
                identity_group = service.get("identity_group", [])
                model = get_model_from_identity(identity_group)
                stats[(model, gpu_type)] += 1

    # Prepare data for tabulate
    table_data = []
    for (model, gpu_type), count in sorted(stats.items()):
        table_data.append([model, gpu_type, count])

    headers = ["Model", "GPU Type", "Count"]
    print(tabulate(table_data, headers=headers, tablefmt="pretty"))

if __name__ == "__main__":
    main()
