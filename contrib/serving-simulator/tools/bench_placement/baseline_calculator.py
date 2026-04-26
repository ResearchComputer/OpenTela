import time
import math

# --- INPUT DATA AND CONSTANTS ---

# GPU Capacities
A100_CAPACITY = 24
GH200_CAPACITY = 32
TOTAL_CAPACITY = A100_CAPACITY + GH200_CAPACITY

# - 70B model now requires 4 A100s AND 4 GH200s.
GPU_REQS = {
    'A100': {
        '13B': 1,
        '34B': 2,
        '70B': 4   
    },
    'GH200': {
        '13B': 1,
        '34B': 2,
        '70B': 4,
    }
}

# Base Processing Times for 1 instance (seconds)
# The GH200 times are significantly lower, indicating higher throughput.
BASE_TIMES = {
    'A100': {
        '13B': 6538.165924,
        '34B': 37231.53667,
        '70B': 14897.00265
    },
    'GH200': {
        '13B': 2617.779327,
        '34B': 11626.39596,
        '70B': 5939.552729
    }
}


def calculate_throughput(model_name: str, N_A: int, N_G: int) -> float:
    """
    Calculates the total throughput for a given model.
    Throughput = Sum of (Instances / Base_Time) for each GPU type.
    This value is proportional to the total requests/second that can be handled.
    """
    T_A = BASE_TIMES['A100'][model_name]
    T_G = BASE_TIMES['GH200'][model_name]

    # Throughput contribution for A100: N_A / T_A
    th_A = N_A / T_A if N_A > 0 else 0.0
    
    # Throughput contribution for GH200: N_G / T_G
    th_G = N_G / T_G if N_G > 0 else 0.0
    
    return th_A + th_G

def find_best_allocation():
    """
    Performs an exhaustive search over all valid integer allocations of 
    instances across A100 and GH200 GPUs to maximize the minimum throughput,
    using the distinct GPU requirements for each type.
    """
    start_time = time.time()
    best_objective_value = -1.0
    best_allocation = {}
    
    REQ_A = GPU_REQS['A100']
    REQ_G = GPU_REQS['GH200']
    
    # Pre-calculate max possible instances for loop boundaries using the corrected requirements
    MAX_70_A = A100_CAPACITY // REQ_A['70B']  # Max 64 / 4 = 16
    MAX_34_A = A100_CAPACITY // REQ_A['34B']  # Max 64 / 2 = 32
    # CRITICAL CHANGE: MAX_70_G is now 64 / 4 = 16
    MAX_70_G = GH200_CAPACITY // REQ_G['70B']  
    MAX_34_G = GH200_CAPACITY // REQ_G['34B']  # Max 64 / 2 = 32

    total_iterations = 0

    # --- Start A100 Allocation Search (N_iA) ---
    for N_70A in range(MAX_70_A + 1):
        remaining_A100_1 = A100_CAPACITY - (N_70A * REQ_A['70B'])
        
        for N_34A in range(remaining_A100_1 // REQ_A['34B'] + 1):
            remaining_A100_2 = remaining_A100_1 - (N_34A * REQ_A['34B'])
            
            # The remaining A100 GPUs are automatically allocated to the 13B model
            N_13A = remaining_A100_2 // REQ_A['13B']
            
            # --- Start GH200 Allocation Search (N_iG) ---
            for N_70G in range(MAX_70_G + 1):
                remaining_GH200_1 = GH200_CAPACITY - (N_70G * REQ_G['70B'])
                
                for N_34G in range(remaining_GH200_1 // REQ_G['34B'] + 1):
                    remaining_GH200_2 = remaining_GH200_1 - (N_34G * REQ_G['34B'])
                    
                    # The remaining GH200 GPUs are automatically allocated to the 13B model
                    N_13G = remaining_GH200_2 // REQ_G['13B']
                    
                    total_iterations += 1
                    
                    # --- Calculate Objective Function (J) ---
                    
                    # 1. Store the full allocation for this iteration
                    current_allocation = {
                        '13B': {'A100': N_13A, 'GH200': N_13G},
                        '34B': {'A100': N_34A, 'GH200': N_34G},
                        '70B': {'A100': N_70A, 'GH200': N_70G}
                    }
                    
                    # 2. Calculate throughputs for all three models
                    Th_13 = calculate_throughput('13B', N_13A, N_13G)
                    Th_34 = calculate_throughput('34B', N_34A, N_34G)
                    Th_70 = calculate_throughput('70B', N_70A, N_70G)

                    # 3. Objective function: Maximize the minimum throughput
                    current_objective_value = min(Th_13, Th_34, Th_70)
                    
                    # 4. Update the best allocation if the objective is improved
                    if current_objective_value > best_objective_value:
                        best_objective_value = current_objective_value
                        best_allocation = current_allocation
                        
    # --- Final Result Processing ---
    
    if not best_allocation:
        return {"error": "No valid allocation found (should not happen with available resources)."}

    performance = {}
    models = ['13B', '34B', '70B']
    
    # 1. Calculate individual and total throughputs/metrics for the optimal allocation
    for model_name in models:
        N_A = best_allocation[model_name]['A100']
        N_G = best_allocation[model_name]['GH200']
        T_A = BASE_TIMES['A100'][model_name]
        T_G = BASE_TIMES['GH200'][model_name]

        # Individual Throughput (reqs/sec)
        Th_A = N_A / T_A if N_A > 0 else 0.0
        Th_G = N_G / T_G if N_G > 0 else 0.0
        Th_Total = Th_A + Th_G

        # Percentage Contribution to Total Throughput
        Perc_A = (Th_A / Th_Total) * 100 if Th_Total > 0 else 0.0
        Perc_G = (Th_G / Th_Total) * 100 if Th_Total > 0 else 0.0
        
        performance[model_name] = {
            'Th_Total': Th_Total,
            'Th_A100': Th_A,
            'Th_GH200': Th_G,
            'Time_Factor': 1.0 / Th_Total if Th_Total > 0 else float('inf'),
            'Perc_A100': Perc_A,
            'Perc_GH200': Perc_G
        }
        
    Th_13 = performance['13B']['Th_Total']
    Th_34 = performance['34B']['Th_Total']
    Th_70 = performance['70B']['Th_Total']
    
    Time_13 = performance['13B']['Time_Factor']
    Time_34 = performance['34B']['Time_Factor']
    Time_70 = performance['70B']['Time_Factor']
    
    max_time_factor = max(Time_13, Time_34, Time_70)

    # Total GPUs used (using the correct REQ tables)
    REQ_A = GPU_REQS['A100']
    REQ_G = GPU_REQS['GH200']
    
    A100_used = (best_allocation['13B']['A100'] * REQ_A['13B']) + \
                (best_allocation['34B']['A100'] * REQ_A['34B']) + \
                (best_allocation['70B']['A100'] * REQ_A['70B'])
    
    GH200_used = (best_allocation['13B']['GH200'] * REQ_G['13B']) + \
                 (best_allocation['34B']['GH200'] * REQ_G['34B']) + \
                 (best_allocation['70B']['GH200'] * REQ_G['70B'])
    
    end_time = time.time()

    results = {
        "best_allocation": best_allocation,
        "metrics": {
            "A100_used": A100_used,
            "GH200_used": GH200_used,
            "Total_GPUs_used": A100_used + GH200_used,
            "Total_Throughput_13B": Th_13,
            "Total_Throughput_34B": Th_34,
            "Total_Throughput_70B": Th_70,
            "Time_Factor_13B": Time_13,
            "Time_Factor_34B": Time_34,
            "Time_Factor_70B": Time_70,
            # NEW METRICS: Throughput Share Percentage
            "Perc_13B_A100": performance['13B']['Perc_A100'],
            "Perc_13B_GH200": performance['13B']['Perc_GH200'],
            "Perc_34B_A100": performance['34B']['Perc_A100'],
            "Perc_34B_GH200": performance['34B']['Perc_GH200'],
            "Perc_70B_A100": performance['70B']['Perc_A100'],
            "Perc_70B_GH200": performance['70B']['Perc_GH200'],
            "Max_Time_Factor_Bottleneck": max_time_factor,
            "Objective_Max_Min_Throughput": best_objective_value
        },
        "search_stats": {
            "total_iterations": total_iterations,
            "runtime_seconds": end_time - start_time
        }
    }
    
    return results

# --- EXECUTION ---

if __name__ == '__main__':
    results = find_best_allocation()
    
    print("--- GPU RESOURCE ALLOCATION OPTIMIZER (EXHAUSTIVE SEARCH) ---")
    print("GPU Requirements Corrected: 70B model requires 4 GPUs of either type.")
    print(f"Goal: Maximize the Minimum Throughput (i.e., Balance Processing Time)")
    print(f"Total Iterations: {results['search_stats']['total_iterations']:,}")
    print(f"Runtime: {results['search_stats']['runtime_seconds']:.4f} seconds\n")

    print("==================================================================")
    print("      Optimal Allocation for Balanced Processing Time")
    print("==================================================================")
    
    allocation = results['best_allocation']
    metrics = results['metrics']
    
    print(f"\nResource Utilization:")
    print(f"  A100s Used: {metrics['A100_used']} / {A100_CAPACITY}")
    print(f"  GH200s Used: {metrics['GH200_used']} / {GH200_CAPACITY}")
    print(f"  Total GPUs Used: {metrics['Total_GPUs_used']} / {TOTAL_CAPACITY}")
    
    print("\nInstance Allocation:")
    
    total_13B = allocation['13B']['A100'] + allocation['13B']['GH200']
    total_34B = allocation['34B']['A100'] + allocation['34B']['GH200']
    total_70B = allocation['70B']['A100'] + allocation['70B']['GH200']
    
    print(f"  13B Model (Total {total_13B} instances):")
    print(f"    - A100 Instances: {allocation['13B']['A100']}")
    print(f"    - GH200 Instances: {allocation['13B']['GH200']}")
    
    print(f"  34B Model (Total {total_34B} instances):")
    print(f"    - A100 Instances: {allocation['34B']['A100']}")
    print(f"    - GH200 Instances: {allocation['34B']['GH200']}")
    
    print(f"  70B Model (Total {total_70B} instances):")
    print(f"    - A100 Instances: {allocation['70B']['A100']}")
    print(f"    - GH200 Instances: {allocation['70B']['GH200']}")
    
    print("\nThroughput Breakdown by GPU Type (Effective Request Share):")
    print(f"  Model | A100 Share | GH200 Share")
    print("  --------------------------------------")
    print(f"  13B   | {metrics['Perc_13B_A100']:.2f}%     | {metrics['Perc_13B_GH200']:.2f}%")
    print(f"  34B   | {metrics['Perc_34B_A100']:.2f}%     | {metrics['Perc_34B_GH200']:.2f}%")
    print(f"  70B   | {metrics['Perc_70B_A100']:.2f}%     | {metrics['Perc_70B_GH200']:.2f}%")
    
    print("\nPerformance Metrics (Throughput is proportional to requests/second):")
    
    # We use a Time Factor (1/Throughput) as a normalized measure of processing time.
    print(f"  Model | Throughput (1/s) | Normalized Time Factor (s)")
    print("  --------------------------------------------------------")
    print(f"  13B   | {metrics['Total_Throughput_13B']:.5f}      | {metrics['Time_Factor_13B']:.2f}")
    print(f"  34B   | {metrics['Total_Throughput_34B']:.5f}      | {metrics['Time_Factor_34B']:.2f}")
    print(f"  70B   | {metrics['Total_Throughput_70B']:.5f}      | {metrics['Time_Factor_70B']:.2f}")
    print("  --------------------------------------------------------")
    print(f"  Max Min Throughput (Objective Value): {metrics['Objective_Max_Min_Throughput']:.5f}")
    print(f"  Bottleneck Time Factor (Max Time): {metrics['Max_Time_Factor_Bottleneck']:.2f}")
    print("==================================================================")
