import time
import math

# --- INPUT DATA AND CONSTANTS ---

# GPU Capacities
A100_CAPACITY = 24
GH200_CAPACITY = 32
TOTAL_CAPACITY = A100_CAPACITY + GH200_CAPACITY

# GPU Requirements (TP Sizes)
# 13B: TP1, 34B: TP2, 70B: TP4
GPU_REQS = {
    '13B': 1,
    '34B': 2,
    '70B': 4
}

# Workload Parameters (from hexgen.yaml / ours_workload.yaml)
ARRIVAL_RATES = {
    '13B': 110,
    '34B': 185.5,
    '70B': 221
}

MODEL_SIZES = {
    '13B': 13,
    '34B': 34,
    '70B': 70
}

def calculate_target_distribution():
    """
    Calculates the target GPU allocation based on Memory Demand.
    Memory Demand = Arrival Rate * Model Size
    """
    memory_demands = {}
    total_demand = 0.0
    
    for model in ARRIVAL_RATES:
        demand = ARRIVAL_RATES[model] * MODEL_SIZES[model]
        memory_demands[model] = demand
        total_demand += demand
        
    targets = {}
    print("\n--- Target Distribution Calculation ---")
    print(f"{'Model':<10} | {'Arrival':<10} | {'Size':<5} | {'Mem Demand':<12} | {'Share':<8} | {'Target GPUs':<10}")
    print("-" * 75)
    
    for model in ['13B', '34B', '70B']:
        share = memory_demands[model] / total_demand
        target_gpus = share * TOTAL_CAPACITY
        targets[model] = target_gpus
        print(f"{model:<10} | {ARRIVAL_RATES[model]:<10} | {MODEL_SIZES[model]:<5} | {memory_demands[model]:<12.1f} | {share*100:<7.2f}% | {target_gpus:<10.2f}")
    print("-" * 75)
    print(f"Total GPUs Available: {TOTAL_CAPACITY}")
    print("-" * 75)
        
    return targets

def find_best_allocation():
    """
    Performs an exhaustive search to find the allocation that minimizes
    the distance to the target GPU distribution based on memory demand.
    """
    start_time = time.time()
    targets = calculate_target_distribution()
    
    best_error = float('inf')
    best_allocation = {}
    best_metrics = {}
    
    # Pre-calculate max instances
    # We use the same loop structure as baseline to ensure high utilization
    # A100 Loops
    MAX_70_A = A100_CAPACITY // GPU_REQS['70B']
    
    # GH200 Loops
    MAX_70_G = GH200_CAPACITY // GPU_REQS['70B']
    
    total_iterations = 0
    
    # --- Start Search ---
    # Iterate A100
    for N_70A in range(MAX_70_A + 1):
        rem_A1 = A100_CAPACITY - (N_70A * GPU_REQS['70B'])
        for N_34A in range(rem_A1 // GPU_REQS['34B'] + 1):
            rem_A2 = rem_A1 - (N_34A * GPU_REQS['34B'])
            N_13A = rem_A2 // GPU_REQS['13B']
            
            # Iterate GH200
            for N_70G in range(MAX_70_G + 1):
                rem_G1 = GH200_CAPACITY - (N_70G * GPU_REQS['70B'])
                for N_34G in range(rem_G1 // GPU_REQS['34B'] + 1):
                    rem_G2 = rem_G1 - (N_34G * GPU_REQS['34B'])
                    N_13G = rem_G2 // GPU_REQS['13B']
                    
                    total_iterations += 1
                    
                    # Calculate Total GPUs Allocated per Model
                    # Note: We sum A100 and GH200 usage for the model
                    gpus_13 = (N_13A + N_13G) * GPU_REQS['13B']
                    gpus_34 = (N_34A + N_34G) * GPU_REQS['34B']
                    gpus_70 = (N_70A + N_70G) * GPU_REQS['70B']
                    
                    # Calculate Error (L1 Distance from Target)
                    error = abs(gpus_13 - targets['13B']) + \
                            abs(gpus_34 - targets['34B']) + \
                            abs(gpus_70 - targets['70B'])
                            
                    if error < best_error:
                        best_error = error
                        best_allocation = {
                            '13B': {'A100': N_13A, 'GH200': N_13G},
                            '34B': {'A100': N_34A, 'GH200': N_34G},
                            '70B': {'A100': N_70A, 'GH200': N_70G}
                        }
                        best_metrics = {
                            'Allocated_13B': gpus_13,
                            'Allocated_34B': gpus_34,
                            'Allocated_70B': gpus_70,
                            'Target_13B': targets['13B'],
                            'Target_34B': targets['34B'],
                            'Target_70B': targets['70B'],
                            'Error': error
                        }

    end_time = time.time()
    
    return {
        "best_allocation": best_allocation,
        "metrics": best_metrics,
        "search_stats": {
            "total_iterations": total_iterations,
            "runtime_seconds": end_time - start_time
        }
    }

if __name__ == '__main__':
    print("--- GPU PLACEMENT: MEMORY DEMAND STRATEGY ---")
    results = find_best_allocation()
    
    alloc = results['best_allocation']
    metrics = results['metrics']
    
    print("\n==================================================================")
    print("      Optimal Allocation (Minimizing Distance to Target)")
    print("==================================================================")
    
    print(f"\nAllocation Details:")
    print(f"  13B Model:")
    print(f"    - A100 Instances: {alloc['13B']['A100']}")
    print(f"    - GH200 Instances: {alloc['13B']['GH200']}")
    print(f"    - Total GPUs: {metrics['Allocated_13B']} (Target: {metrics['Target_13B']:.2f})")
    
    print(f"  34B Model:")
    print(f"    - A100 Instances: {alloc['34B']['A100']}")
    print(f"    - GH200 Instances: {alloc['34B']['GH200']}")
    print(f"    - Total GPUs: {metrics['Allocated_34B']} (Target: {metrics['Target_34B']:.2f})")
    
    print(f"  70B Model:")
    print(f"    - A100 Instances: {alloc['70B']['A100']}")
    print(f"    - GH200 Instances: {alloc['70B']['GH200']}")
    print(f"    - Total GPUs: {metrics['Allocated_70B']} (Target: {metrics['Target_70B']:.2f})")
    
    print(f"\nTotal Error (L1 Distance): {metrics['Error']:.4f}")
    print(f"Total Iterations: {results['search_stats']['total_iterations']}")
    print(f"Runtime: {results['search_stats']['runtime_seconds']:.4f}s")
