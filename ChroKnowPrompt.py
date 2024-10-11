import torch
import numpy as np
import random
import argparse

from tqdm import tqdm
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams

from sources.template import *
from sources.utils import *


### Random Seed ###
SEED = 42
def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
seed_everything(SEED)
set_seed(SEED)
        
        
### Main Processing Part ###
def run_chroknowprompt(model_name, dtype, device_num, gpu_util, multi_gpu, max_tokens, domain, temperature, prev_span=3, next_span=3, token=None, cache_dir=None, save_results=True):
    
    if "gpt" not in model_name.lower():
        ### Load model with setting stop tokens ###
        model_path = get_model(model_name, token=token, cache_dir=cache_dir)
        llm = LLM(model=model_path, gpu_memory_utilization=gpu_util, dtype=dtype, device=device_num, tensor_parallel_size=multi_gpu)
        tokenizer = llm.get_tokenizer()
        stop = []
        if "llama3" in model_name.lower():
            stop = list(set(stop + ["Ċ", "ĊĊ"])) # In Llama \n is <0x0A>; In OPT \n is Ċ
        elif "llama2" in model_name.lower():
            stop = list(set(stop + ["Ċ", "ĊĊ", "<0x0A>"]))
        elif "phi" in model_name.lower():
            stop = list(set(stop + ["<0x0A>"]))
        elif "mistral" in model_name.lower():
            stop = list(set(stop + ["<0x0A>"]))
        elif "solar" in model_name.lower():
            stop = list(set(stop + ["<0x0A>"]))
        elif "gemma" in model_name.lower():
            stop = list(set(stop + ["\n\n"]))
        elif "flan" in model_name.lower():
            stop = list(set(stop + ["▁Q"]))
        else:
            stop = list(set(stop + ["Ċ", "ĊĊ"]))
        stop_token_ids = list(set([tokenizer.convert_tokens_to_ids(stop_token) for stop_token in stop] + [tokenizer.eos_token_id]))
        sampling_params = SamplingParams(temperature=temperature, top_p=1.0, seed=SEED, stop_token_ids=stop_token_ids, max_tokens=max_tokens, skip_special_tokens=True)
    else:
        llm = None 
        tokenizer = None
        sampling_params = None
    
    
    ### Load data with time stamp ###
    bench_data_c, bench_data_u, timestamp_c, timestamp_u = load_data_with_timestamp(model_name, domain)
    
    
    print("Chrono Gap Check for Dynamic")
    # Get indices with partial known data
    partial_known_indices = [i for i, entry in enumerate(timestamp_c["Partial_known"]) if entry]

    # Subset of bench_data_c with corresponding indices
    subset_bench_data_c = [bench_data_c[i] for i in partial_known_indices]

    # Use tqdm to track the progress of the outer loop
    for index, triplet in tqdm(zip(partial_known_indices, subset_bench_data_c), total=len(partial_known_indices), desc="Processing chrono gap check"):
        # print(timestamp_c["Partial_known"][index][0])
        partial_known = timestamp_c["Partial_known"][index][1]
        # Optional: track progress of inner loop
        for category in tqdm(["incorrect", "parital_correct2", "partial_correct1"], desc="Categories", leave=False):
            for year, objects_year in partial_known.items():
                if objects_year["category"] == category:
                    chrono_ans = generate_chrono_ans(model_name, partial_known, year, triplet, llm, tokenizer, sampling_params, temperature, max_tokens, domain, prev_year_span=prev_span, next_year_span=next_span)
                    # print(f"chrono_ans: {chrono_ans}")
                    if chrono_ans is not None:  # chrono_ans가 None이 아니면 처리
                        update_timestamp(timestamp_c, index, year, chrono_ans)
    
    if save_results:
        save_updated_timestamp(timestamp_c, f'./ChronoGap/{model_name}/Updated_Timestamp_{domain}_Dynamic.json')
        
        
    print("Chrono Gap Check for Static")
    # Get indices with partial known data
    partial_known_indices = [i for i, entry in enumerate(timestamp_u["Partial_known"]) if entry]

    # Subset of bench_data_c with corresponding indices
    subset_bench_data_u = [bench_data_u[i] for i in partial_known_indices]

    # Use tqdm to track the progress of the outer loop
    for index, triplet in tqdm(zip(partial_known_indices, subset_bench_data_u), total=len(partial_known_indices), desc="Processing chrono gap check"):
        # print(timestamp_u["Partial_known"][index][0])
        partial_known = timestamp_u["Partial_known"][index][1]
        # Optional: track progress of inner loop
        for category in tqdm(["incorrect", "partial_correct2", "partial_correct1"], desc="Categories", leave=False):
            for year, objects_year in partial_known.items():
                if objects_year["category"] == category:
                    chrono_ans = generate_chrono_ans(model_name, partial_known, year, triplet, llm, tokenizer, sampling_params, temperature, max_tokens, domain, prev_year_span=prev_span, next_year_span=next_span)
                    # print(f"chrono_ans: {chrono_ans}")
                    if chrono_ans is not None:  # chrono_ans가 None이 아니면 처리
                        update_timestamp(timestamp_u, index, year, chrono_ans)
    
    if save_results:
        save_updated_timestamp(timestamp_u, f'./ChronoGap/{model_name}/Updated_Timestamp_{domain}_Static.json')

# Main function to handle arguments
def main():
    parser = argparse.ArgumentParser(description="Run LLM experiments for specified year and temperature.")
    parser.add_argument('--model_name', type=str, required=True, default="Llama3.1_8B", help="Select model to CHeck")
    parser.add_argument('--dtype', type=str, required=True, default="bfloat16", help='torch data type like float16, bfloat16, FP16, FP32')
    parser.add_argument('--device_num', type=str, required=True, default="auto", help='Possible choices: auto, cuda, neuron, cpu, openvino, tpu, xpu')
    parser.add_argument('--gpu_util', type=float, required=True, default=0.90, help="Percentage of GPU memory utilization.")
    parser.add_argument('--multi_gpu', type=int, required=True, default=1, help="Number of multi GPUs.")
    parser.add_argument('--max_tokens', type=int, required=True, default=50, help='Max tokens for only generate one objects')
    parser.add_argument('--domain', type=str, required=True, default="General", help='Domain of CHROKNOWLEDGE')
    parser.add_argument('--temperature', type=float, required=True, default=0.0, help='Temperature for the experiments.')
    parser.add_argument('--prev_span', type=int, required=True, default=3, help='Previous year span')
    parser.add_argument('--next_span', type=int, required=True, default=3, help='Next year span')
    parser.add_argument('--token', type=str, default=None, help="Token for Huggingface model load.")
    parser.add_argument('--cache_dir', type=str, default=None, help="Use cache_dir if model already exists.")
    parser.add_argument('--save_results', type=bool, required=True, default=True, help="Save the results into json file.")
    args = parser.parse_args()
    
    run_chroknowprompt(model_name=args.model_name, dtype=args.dtype, device_num=args.device_num, gpu_util=args.gpu_util, multi_gpu=args.multi_gpu, max_tokens=args.max_tokens, domain=args.domain, temperature=args.temperature, prev_span=args.prev_span, next_span=args.next_span, token=args.token, cache_dir=args.cache_dir, save_results=args.save_results)  

if __name__ == "__main__":
    main()
