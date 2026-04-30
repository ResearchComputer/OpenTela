# for i in {1..2}
# do
#     # python cli/spinup.py --config meta/experiments/2_placement/70b_gh200.yaml

#     # python cli/run_workloads.py --config meta/experiments/2_placement/70b_gh200.yaml --output-file .local/output/70b_gh200_${i}.jsonl

#     # python cli/spinup.py --config meta/experiments/2_placement/34b_gh200.yaml

#     # python cli/run_workloads.py --config meta/experiments/2_placement/34b_gh200.yaml --output-file .local/output/34b_gh200_${i}.jsonl

#     # python cli/spinup.py --config meta/experiments/2_placement/13b_gh200.yaml

#     # python cli/run_workloads.py --config meta/experiments/2_placement/13b_gh200.yaml --output-file .local/output/13b_gh200_${i}.jsonl

#     # python cli/spinup.py --config meta/experiments/2_placement/8b_gh200.yaml

#     # python cli/run_workloads.py --config meta/experiments/2_placement/8b_gh200.yaml --output-file .local/output/8b_gh200_${i}.jsonl

#     # python cli/spinup.py --config meta/experiments/2_placement/7b_gh200.yaml

#     # python cli/run_workloads.py --config meta/experiments/2_placement/7b_gh200.yaml --output-file .local/output/7b_gh200_${i}.jsonl

#     # python cli/cancel_all.py
# done

for i in {1..3}
do
    # python cli/spinup.py --config meta/experiments/2_placement/70b_a100.yaml

    # python cli/run_workloads.py --config meta/experiments/2_placement/70b_a100.yaml --output-file .local/output/70b_a100_${i}.jsonl  

    python cli/spinup.py --config meta/experiments/2_placement/34b_a100.yaml

    python cli/run_workloads.py --config meta/experiments/2_placement/34b_a100.yaml --output-file .local/output/34b_a100_${i}.jsonl

    # python cli/spinup.py --config meta/experiments/2_placement/13b_a100.yaml

    # python cli/run_workloads.py --config meta/experiments/2_placement/13b_a100.yaml --output-file .local/output/13b_a100_${i}.jsonl

    # python cli/spinup.py --config meta/experiments/2_placement/8b_a100.yaml

    # python cli/run_workloads.py --config meta/experiments/2_placement/8b_a100.yaml --output-file .local/output/8b_a100_${i}.jsonl

    python cli/spinup.py --config meta/experiments/2_placement/7b_a100.yaml

    python cli/run_workloads.py --config meta/experiments/2_placement/7b_a100.yaml --output-file .local/output/7b_a100_${i}.jsonl

    python cli/cancel_all.py
done
