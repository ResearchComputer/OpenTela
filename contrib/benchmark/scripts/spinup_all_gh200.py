import os

configs = [x for x in os.listdir("meta/experiments/2_placement") if x.endswith(".yaml")]

for config in configs:
    if "gh200" in config:
        os.system(f"python simulator/real/spinup.py --config meta/experiments/2_placement/{config}")