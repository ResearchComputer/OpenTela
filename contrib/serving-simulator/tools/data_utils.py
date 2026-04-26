import os
import json
import pandas as pd
from dataclasses import dataclass

@dataclass
class Record:
    source: str # ['simulation', 'real']
    model_id: str
    input_length: int
    output_length: int
    batch_size: int
    prefill_time_s: float
    decode_time_per_token_s: float
    hw_name: str
    tp_size: int

def load_real_bench_json(data_file: str):
    with open(data_file, 'r') as f:
        data = json.load(f)
    all_results = []
    for datum in data:
        for time in datum['prefill']['all_times']:
            res = {
                "model_id": datum["model_id"],
                "input_length": datum["input_prompt_len"],
                "output_length": datum["output_len"],
                "type": "prefill",
                "time_s": time,
            }
            all_results.append(res)
        for time in datum['decode']['all_times']:
            res = {
                "model_id": datum["model_id"],
                "input_length": datum["input_prompt_len"],
                "output_length": datum["output_len"],
                "type": "decode",
                "time_s": time,
            }
            all_results.append(res)
    return all_results

def parse_all_to_csv(result_dir=".local/bench_results/"):
    all_results = []
    files = os.listdir(result_dir)
    sim_res = [f for f in files if f.endswith(".csv")]
    for res in sim_res:
        df = pd.read_csv(os.path.join(result_dir, res))
        df_dicts = df.to_dict(orient='records')
        records = [Record(
            source='simulation',
            model_id=rec['model_id'],
            input_length=rec['input_length'],
            output_length=rec['output_length'],
            batch_size=rec['batch_size'] if 'batch_size' in rec else 1,
            prefill_time_s=rec['prefill_time'],
            decode_time_per_token_s=rec['tpot'],
            hw_name=rec['hardware'],
            tp_size=rec['tp_size']
        ) for rec in df_dicts]
        all_results.extend(records)
    
    bsz_1_benchmark_results = [f for f in files if f.endswith(".json") if "bsz" not in f]
    other_benchmark_results = [f for f in files if f.endswith(".json") if "bsz" in f]
    for res in bsz_1_benchmark_results:
        with open(os.path.join(result_dir, res), 'r') as f:
            data = json.load(f)
            if 'sgs-gpu02' in res.lower():
                data = data['benchmarks']
                
        for datum in data:
            if "h100" in res.lower():
                hardware = "NVDA:H100:PCIe"
                default_tp_size: int = 1
                for i in range(len(datum['prefill']['all_times'])):
                    record = Record(
                        source='real',
                        model_id=datum['model_id'],
                        input_length=datum['input_prompt_len'],
                        output_length=datum['output_len'],
                        batch_size=1,
                        prefill_time_s=datum['prefill']['all_times'][i],
                        decode_time_per_token_s=datum['decode']['all_times'][i],
                        hw_name=hardware,
                        tp_size=default_tp_size
                    )
                    all_results.append(record)
            elif "sgs-gpu02" in res.lower():
                hardware = "NVDA:RTX3090:PCIe"
                default_tp_size: int = datum['tp_size']
                for i in range(len(datum['prefill']['all_times'])):
                    record = Record(
                        source='real',
                        model_id=datum['model_id'],
                        input_length=datum['input_prompt_len'],
                        output_length=datum['output_len'],
                        batch_size=1,
                        prefill_time_s=datum['prefill']['all_times'][i],
                        decode_time_per_token_s=datum['decode']['all_times'][i],
                        hw_name=hardware,
                        tp_size=default_tp_size
                    )
                    all_results.append(record)
    for res in other_benchmark_results:
        if "sgs-gpu05" in res.lower():
            hardware = "NVDA:H100:PCIe"
            default_tp_size: int = 1
        elif "sgs-gpu02" in res.lower():
            default_tp_size = 1
            hardware = "NVDA:A100:PCIe"
        with open(os.path.join(result_dir, res), 'r') as f:
            data = json.load(f)
        res_benchmark = data['benchmarks']
        for datum in res_benchmark:
            record = Record(
                source='real',
                model_id=datum['config']['model_id'],
                input_length=datum['config']['input_prompt_len'],
                output_length=datum['config']['output_len'],
                batch_size=datum['config']['batch_size'],
                prefill_time_s=datum['streaming_metrics']['time_to_first_token']['average_seconds'],
                decode_time_per_token_s=datum['streaming_metrics']['time_between_tokens']['average_seconds'],
                hw_name=hardware,
                tp_size=default_tp_size
            )
            all_results.append(record)
    results = [r.__dict__ for r in all_results]
    results = pd.DataFrame(results)
    return results

if __name__ == "__main__":
    print("Parsing benchmark results to csv...")
    results =  parse_all_to_csv()
    results.to_csv("data/consolidated.csv", index=False)