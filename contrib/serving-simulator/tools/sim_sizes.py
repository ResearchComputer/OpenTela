import os
import pandas as pd
from simulator.core.model_analyzer import ModelAnalyzer
from simulator.configs.models import llama

models = [
    "meta-llama/Llama-2-13b-hf",
]
input_seq_lens = [1]
output_lens = [1024]
batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
tp_sizes = [1]
hardwares = ["NVDA:H100:PCIe"]

def simulate(args):
    all_results = []
    for hw in hardwares:
        for model in models:
            analyzer = ModelAnalyzer(
                model_id=model,
                config=llama,
                hardware=hw,
            )
            for tp_size in tp_sizes:
                for input_len in input_seq_lens:
                    for output_len in output_lens:
                        for bsz in batch_sizes:
                            res = analyzer.analyze_generate_task(
                                prompt_len=input_len,
                                gen_len=output_len,
                                batchsize=bsz,
                                w_bit=16,
                                a_bit=16,
                                kv_bit=16,
                                tp_size=tp_size,
                            )
                            res['batch_size'] = bsz
                            res["model_id"] = model
                            res["input_length"] = input_len
                            res["output_length"] = output_len
                            res["hardware"] = hw
                            res["tp_size"] = tp_size
                            all_results.append(res)
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(args.output_dir, "simulation_results_2.csv"), index=False)
    print(f"All results saved to: {args.output_dir}/simulation_results_2.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run LLM serving benchmark with statistics"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results.json",
        help="Directory to save detailed results",
    )
    args = parser.parse_args()
    simulate(args)
