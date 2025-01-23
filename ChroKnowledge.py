import os
import json
import random
import argparse
import torch
import numpy as np

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

            
### Model Test with Time ###
def knowledge_check_with_time(model_name, dtype, device_num, gpu_util, multi_gpu, max_tokens, domain, template, temperature, token=None, cache_dir=None, save_results=True):
    
    ### Load model with setting stop tokens ###
    if "gpt" not in model_name.lower() or "gemini" not in model_name.lower():
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
        elif "nemotron" in model_name.lower():
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
    
    
    ### Inference in Biomedical Domain ###
    if domain == "Biomedical":
        bench_data_c, bench_data_u, bench_data_fs = load_data(domain)
        all_outputs_ct = []
        all_outputs_ut = []
        print("Benchmark Dynamic with Time.")
        for i, triplet in tqdm(enumerate(bench_data_c), total=len(bench_data_c)):
            matching_few_shots = [example for example in bench_data_fs if example['relation'] == triplet['relation']]
            triplet_outputs = {}
            years = get_years_from_triplet(triplet)
            
            prompts = []
            for year in years:
                for _ in range(5):
                    few_shots = random.sample(matching_few_shots, 4)
                    if template == "generation":
                        prompt_c = prompt_KT_t.render(
                            t1=str(few_shots[0][f'release_date_{year}'])[:4], sub1=few_shots[0]['subject'], rel1=few_shots[0]['relation'], obj1=few_shots[0][f'objects_{year}'][0],
                            t2=str(few_shots[1][f'release_date_{year}'])[:4], sub2=few_shots[1]['subject'], rel2=few_shots[1]['relation'], obj2=few_shots[1][f'objects_{year}'][0],
                            t3=str(few_shots[2][f'release_date_{year}'])[:4], sub3=few_shots[2]['subject'], rel3=few_shots[2]['relation'], obj3=few_shots[2][f'objects_{year}'][0],
                            t4=str(few_shots[3][f'release_date_{year}'])[:4], sub4=few_shots[3]['subject'], rel4=few_shots[3]['relation'], obj4=few_shots[3][f'objects_{year}'][0],
                            t=str(triplet[f'release_date_{year}'])[:4], sub=triplet["subject"], rel=triplet["relation"]
                        )
                    elif template == "QA":
                        prompt_template = eval(f'prompt_QA_{domain}_t')
                        fs1 = prompt_template[triplet['relation']].render(
                            t=str(few_shots[0][f'release_date_{year}'])[:4], sub=few_shots[0]['subject'], Ans1=few_shots[0]['distractor'][0], Ans2=few_shots[0][f'objects_{year}'][0], Ans3=few_shots[0]['distractor'][1], Ans4=few_shots[0]['distractor'][2]
                        ) + '\n' + '(b) ' + few_shots[0][f'objects_{year}'][0] + '\n'
                        
                        fs2 = prompt_template[triplet['relation']].render(
                            t=str(few_shots[1][f'release_date_{year}'])[:4], sub=few_shots[1]['subject'], Ans1=few_shots[1]['distractor'][0], Ans2=few_shots[1]['distractor'][1], Ans3=few_shots[1][f'objects_{year}'][0], Ans4=few_shots[1]['distractor'][2]
                        ) + '\n' + '(c) ' + few_shots[1][f'objects_{year}'][0] + '\n'
                        
                        fs3 = prompt_template[triplet['relation']].render(
                            t=str(few_shots[2][f'release_date_{year}'])[:4], sub=few_shots[2]['subject'], Ans1=few_shots[2]['distractor'][2], Ans2=few_shots[2]['distractor'][1], Ans3=few_shots[2]['distractor'][0], Ans4=few_shots[2][f'objects_{year}'][0]
                        ) + '\n' + '(d) ' + few_shots[2][f'objects_{year}'][0] + '\n'
                        
                        fs4 = prompt_template[triplet['relation']].render(
                            t=str(few_shots[3][f'release_date_{year}'])[:4], sub=few_shots[3]['subject'], Ans1=few_shots[3]['distractor'][1], Ans2=few_shots[3]['distractor'][0], Ans3=few_shots[3]['distractor'][2], Ans4=few_shots[3][f'objects_{year}'][0]
                        ) + '\n' + '(d) ' + few_shots[3][f'objects_{year}'][0] + '\n'
                        
                        prompt_c = fs1 + '\n' + fs2 + '\n' + fs3 + '\n' + fs4 + '\n' + prompt_template[triplet['relation']].render(
                            t=str(triplet[f'release_date_{year}'])[:4], sub=triplet['subject'], Ans1=triplet['distractor'][0], Ans2=triplet['distractor'][2], Ans3=triplet['distractor'][1], Ans4=triplet[f'objects_{year}'][0]
                        ) + '\n'
                    elif template == "TF":
                        prompt_template = eval(f'prompt_TF_{domain}_t')
                        fs1 = prompt_template[triplet['relation']].render(
                            t=str(few_shots[0][f'release_date_{year}'])[:4], sub=few_shots[0]['subject'], obj=few_shots[0][f'objects_{year}'][0]
                        ) + 'A. true' + '\n'
                        
                        fs2 = prompt_template[triplet['relation']].render(
                            t=str(few_shots[1][f'release_date_{year}'])[:4], sub=few_shots[1]['subject'], obj=few_shots[1]['distractor'][0]
                        ) + 'A. false' + '\n'
                        
                        fs3 = prompt_template[triplet['relation']].render(
                            t=str(few_shots[2][f'release_date_{year}'])[:4], sub=few_shots[2]['subject'], obj=few_shots[2]['distractor'][2]
                        ) + 'A. false' + '\n'
                        
                        fs4 = prompt_template[triplet['relation']].render(
                            t=str(few_shots[3][f'release_date_{year}'])[:4], sub=few_shots[3]['subject'], obj=few_shots[3][f'objects_{year}'][0]
                        ) + 'A. true' + '\n'
                        
                        prompt_c = fs1 + '\n' + fs2 + '\n' + fs3 + '\n' + fs4 + '\n' + prompt_template[triplet['relation']].render(
                            t=str(triplet[f'release_date_{year}'])[:4], sub=triplet['subject'], obj=triplet[f'objects_{year}'][0]
                        )
                    
                    prompts.append(prompt_c)
                
            if i == 0:
                print(prompts[0])
                
            if "gpt" in model_name.lower():
                outputs = gpt_batch_generation(model_name, prompts, temperature, max_tokens)
            elif "gemini" in model_name.lower():
                outputs = gemini_batch_generation(model_name, prompts, temperature, max_tokens)
            else:
                outputs = llm.generate(prompts, sampling_params)
    
            # Process the outputs
            output_index = 0
            for year in years:
                year_outputs = []
                for _ in range(5):
                    if "gpt" not in model_name.lower() or "gemini" not in model_name.lower():
                        generated_text = outputs[output_index].outputs[0].text
                    else:
                        generated_text = outputs[output_index]
                    year_outputs.append(generated_text)
                    output_index += 1
                triplet_outputs[f"objects_{year}"] = year_outputs
            
            if i == 0:
                print(triplet_outputs[f"objects_{years[0]}"][0])
            
            all_outputs_ct.append(triplet_outputs)
        
        if save_results:
            output_dir = f'./Results/{model_name}'
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f'Result_TimeVariant_{domain}_Dynamic_temp{temperature}.json'
            output_file = os.path.join(output_dir, output_filename)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_outputs_ct, f)
            print(f"Knowledge Check for {output_filename} Complete")
            
            
        print("Benchmark Static with Time.")
        for i, triplet in tqdm(enumerate(bench_data_u), total=len(bench_data_u)):
            matching_few_shots = [example for example in bench_data_fs if example['relation'] == triplet['relation']]
            triplet_outputs = {}
            years = get_years_from_triplet(triplet)
            
            prompts = []
            for year in years:
                for _ in range(5):
                    few_shots = random.sample(matching_few_shots, 4)
                    if template == "generation":
                        prompt_u = prompt_KT_t.render(
                            t1=str(few_shots[0][f'release_date_{year}'])[:4], sub1=few_shots[0]['subject'], rel1=few_shots[0]['relation'], obj1=few_shots[0][f'objects_{year}'][0],
                            t2=str(few_shots[1][f'release_date_{year}'])[:4], sub2=few_shots[1]['subject'], rel2=few_shots[1]['relation'], obj2=few_shots[1][f'objects_{year}'][0],
                            t3=str(few_shots[2][f'release_date_{year}'])[:4], sub3=few_shots[2]['subject'], rel3=few_shots[2]['relation'], obj3=few_shots[2][f'objects_{year}'][0],
                            t4=str(few_shots[3][f'release_date_{year}'])[:4], sub4=few_shots[3]['subject'], rel4=few_shots[3]['relation'], obj4=few_shots[3][f'objects_{year}'][0],
                            t=str(triplet[f'release_date_{year}'])[:4], sub=triplet["subject"], rel=triplet["relation"]
                        )
                    elif template == "QA":
                        prompt_template = eval(f'prompt_QA_{domain}_t')
                        fs1 = prompt_template[triplet['relation']].render(
                            t=str(few_shots[0][f'release_date_{year}'])[:4], sub=few_shots[0]['subject'], Ans1=few_shots[0]['distractor'][0], Ans2=few_shots[0][f'objects_{year}'][0], Ans3=few_shots[0]['distractor'][1], Ans4=few_shots[0]['distractor'][2]
                        ) + '\n' + '(b) ' + few_shots[0][f'objects_{year}'][0] + '\n'
                        
                        fs2 = prompt_template[triplet['relation']].render(
                            t=str(few_shots[1][f'release_date_{year}'])[:4], sub=few_shots[1]['subject'], Ans1=few_shots[1]['distractor'][0], Ans2=few_shots[1]['distractor'][1], Ans3=few_shots[1][f'objects_{year}'][0], Ans4=few_shots[1]['distractor'][2]
                        ) + '\n' + '(c) ' + few_shots[1][f'objects_{year}'][0] + '\n'
                        
                        fs3 = prompt_template[triplet['relation']].render(
                            t=str(few_shots[2][f'release_date_{year}'])[:4], sub=few_shots[2]['subject'], Ans1=few_shots[2]['distractor'][2], Ans2=few_shots[2]['distractor'][1], Ans3=few_shots[2]['distractor'][0], Ans4=few_shots[2][f'objects_{year}'][0]
                        ) + '\n' + '(d) ' + few_shots[2][f'objects_{year}'][0] + '\n'
                        
                        fs4 = prompt_template[triplet['relation']].render(
                            t=str(few_shots[3][f'release_date_{year}'])[:4], sub=few_shots[3]['subject'], Ans1=few_shots[3]['distractor'][1], Ans2=few_shots[3]['distractor'][0], Ans3=few_shots[3]['distractor'][2], Ans4=few_shots[3][f'objects_{year}'][0]
                        ) + '\n' + '(d) ' + few_shots[3][f'objects_{year}'][0] + '\n'
                        
                        prompt_u = fs1 + '\n' + fs2 + '\n' + fs3 + '\n' + fs4 + '\n' + prompt_template[triplet['relation']].render(
                            t=str(triplet[f'release_date_{year}'])[:4], sub=triplet['subject'], Ans1=triplet['distractor'][0], Ans2=triplet['distractor'][2], Ans3=triplet['distractor'][1], Ans4=triplet[f'objects_{year}'][0]
                        ) + '\n'
                    elif template == "TF":
                        prompt_template = eval(f'prompt_TF_{domain}_t')
                        fs1 = prompt_template[triplet['relation']].render(
                            t=str(few_shots[0][f'release_date_{year}'])[:4], sub=few_shots[0]['subject'], obj=few_shots[0][f'objects_{year}'][0]
                        ) + 'A. true' + '\n'
                        
                        fs2 = prompt_template[triplet['relation']].render(
                            t=str(few_shots[1][f'release_date_{year}'])[:4], sub=few_shots[1]['subject'], obj=few_shots[1]['distractor'][0]
                        ) + 'A. false' + '\n'
                        
                        fs3 = prompt_template[triplet['relation']].render(
                            t=str(few_shots[2][f'release_date_{year}'])[:4], sub=few_shots[2]['subject'], obj=few_shots[2]['distractor'][2]
                        ) + 'A. false' + '\n'
                        
                        fs4 = prompt_template[triplet['relation']].render(
                            t=str(few_shots[3][f'release_date_{year}'])[:4], sub=few_shots[3]['subject'], obj=few_shots[3][f'objects_{year}'][0]
                        ) + 'A. true' + '\n'
                        
                        prompt_u = fs1 + '\n' + fs2 + '\n' + fs3 + '\n' + fs4 + '\n' + prompt_template[triplet['relation']].render(
                            t=str(triplet[f'release_date_{year}'])[:4], sub=triplet['subject'], obj=triplet[f'objects_{year}'][0]
                        )
                    prompts.append(prompt_u)
                
            if i == 0:
                print(prompts[0])
                
            if "gpt" in model_name.lower():
                outputs = gpt_batch_generation(model_name, prompts, temperature, max_tokens)
            elif "gemini" in model_name.lower():
                outputs = gemini_batch_generation(model_name, prompts, temperature, max_tokens)
            else:
                outputs = llm.generate(prompts, sampling_params)
    
            # Process the outputs
            output_index = 0
            for year in years:
                year_outputs = []
                for _ in range(5):
                    if "gpt" not in model_name.lower() or "gemini" not in model_name.lower():
                        generated_text = outputs[output_index].outputs[0].text
                    else:
                        generated_text = outputs[output_index]
                    year_outputs.append(generated_text)
                    output_index += 1
                triplet_outputs[f"objects_{year}"] = year_outputs
            
            if i == 0:
                print(triplet_outputs[f"objects_{years[0]}"][0])
            
            all_outputs_ut.append(triplet_outputs)
        
        if save_results:
            output_dir = f'./Results/{model_name}'
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f'Result_TimeVariant_{domain}_Static_temp{temperature}.json'
            output_file = os.path.join(output_dir, output_filename)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_outputs_ut, f)
            print(f"Knowledge Check for {output_filename} Complete")
    
    ### Inference in General Domain ###    
    elif domain == "General":
        bench_data_c, bench_data_u, bench_data_fs = load_data(domain)
        all_outputs_ct = []
        all_outputs_ut = []

        print("Benchmark Dynamic with Time.")
        for i, triplet in tqdm(enumerate(bench_data_c), total=len(bench_data_c)):
            matching_few_shots = [example for example in bench_data_fs if example['relation'] == triplet['relation']]
            triplet_outputs = {}
            years = get_years_from_triplet(triplet)
            
            prompts = []
            
            for year in years:
                for _ in range(5):
                    few_shots = []
                    for shot in matching_few_shots:
                        if f'objects_{year}' in shot:
                            few_shots.append(shot)
                    
                    selected_few_shots = random.sample(few_shots, 4)
                    
                    if template == "generation":
                        prompt_c = prompt_KT_t.render(
                            # Few-shot examples for current year
                            t1=str(selected_few_shots[0][f'release_date_{year}'])[:4], sub1=selected_few_shots[0]['subject'], rel1=selected_few_shots[0]['relation'], obj1=selected_few_shots[0][f'objects_{year}'][0],
                            t2=str(selected_few_shots[1][f'release_date_{year}'])[:4], sub2=selected_few_shots[1]['subject'], rel2=selected_few_shots[1]['relation'], obj2=selected_few_shots[1][f'objects_{year}'][0],
                            t3=str(selected_few_shots[2][f'release_date_{year}'])[:4], sub3=selected_few_shots[2]['subject'], rel3=selected_few_shots[2]['relation'], obj3=selected_few_shots[2][f'objects_{year}'][0],
                            t4=str(selected_few_shots[3][f'release_date_{year}'])[:4], sub4=selected_few_shots[3]['subject'], rel4=selected_few_shots[3]['relation'], obj4=selected_few_shots[3][f'objects_{year}'][0],
                            # Current triplet data
                            t=str(triplet[f'release_date_{year}'])[:4], sub=triplet["subject"], rel=triplet["relation"]
                        )
                    elif template == "QA":
                        prompt_template = eval(f'prompt_QA_{domain}_t')
                        fs1 = prompt_template[triplet['relation']].render(
                            t=str(selected_few_shots[0][f'release_date_{year}'])[:4], sub=selected_few_shots[0]['subject'], Ans1=selected_few_shots[0][f'objects_{year}'][0], Ans2=selected_few_shots[0]['distractor'][2], Ans3=selected_few_shots[0]['distractor'][1], Ans4=selected_few_shots[0]['distractor'][0]
                        ) + '\n' + '(a) ' + selected_few_shots[0][f'objects_{year}'][0] + '\n'
                        
                        fs2 = prompt_template[triplet['relation']].render(
                            t=str(selected_few_shots[1][f'release_date_{year}'])[:4], sub=selected_few_shots[1]['subject'], Ans1=selected_few_shots[1]['distractor'][0], Ans2=selected_few_shots[1]['distractor'][1], Ans3=selected_few_shots[1][f'objects_{year}'][0], Ans4=selected_few_shots[1]['distractor'][2]
                        ) + '\n' + '(c) ' + selected_few_shots[1][f'objects_{year}'][0] + '\n'
                        
                        fs3 = prompt_template[triplet['relation']].render(
                            t=str(selected_few_shots[2][f'release_date_{year}'])[:4], sub=selected_few_shots[2]['subject'], Ans1=selected_few_shots[2]['distractor'][2], Ans2=selected_few_shots[2]['distractor'][1], Ans3=selected_few_shots[2][f'objects_{year}'][0], Ans4=selected_few_shots[2]['distractor'][0]
                        ) + '\n' + '(c) ' + selected_few_shots[2][f'objects_{year}'][0] + '\n'
                        
                        fs4 = prompt_template[triplet['relation']].render(
                            t=str(selected_few_shots[3][f'release_date_{year}'])[:4], sub=selected_few_shots[3]['subject'], Ans1=selected_few_shots[3]['distractor'][1], Ans2=selected_few_shots[3]['distractor'][0], Ans3=selected_few_shots[3]['distractor'][2], Ans4=selected_few_shots[3][f'objects_{year}'][0]
                        ) + '\n' + '(d) ' + selected_few_shots[3][f'objects_{year}'][0] + '\n'
                        
                        prompt_c = fs1 + '\n' + fs2 + '\n' + fs3 + '\n' + fs4 + '\n' + prompt_template[triplet['relation']].render(
                            t=str(triplet[f'release_date_{year}'])[:4], sub=triplet['subject'], Ans1=triplet['distractor'][0], Ans2=triplet['distractor'][2], Ans3=triplet['distractor'][1], Ans4=triplet[f'objects_{year}'][0]
                        ) + '\n'
                    elif template == "TF":
                        prompt_template = eval(f'prompt_TF_{domain}_t')
                        fs1 = prompt_template[triplet['relation']].render(
                            t=str(selected_few_shots[0][f'release_date_{year}'])[:4], sub=selected_few_shots[0]['subject'], obj=selected_few_shots[0]['distractor'][2]
                        ) + 'A. false' + '\n'
                        
                        fs2 = prompt_template[triplet['relation']].render(
                            t=str(selected_few_shots[1][f'release_date_{year}'])[:4], sub=selected_few_shots[1]['subject'], obj=selected_few_shots[1]['distractor'][0]
                        ) + 'A. false' + '\n'
                        
                        fs3 = prompt_template[triplet['relation']].render(
                            t=str(selected_few_shots[2][f'release_date_{year}'])[:4], sub=selected_few_shots[2]['subject'], obj=selected_few_shots[2]['distractor'][2]
                        ) + 'A. false' + '\n'
                        
                        fs4 = prompt_template[triplet['relation']].render(
                            t=str(selected_few_shots[3][f'release_date_{year}'])[:4], sub=selected_few_shots[3]['subject'], obj=selected_few_shots[3][f'objects_{year}'][0]
                        ) + 'A. true' + '\n'
                        
                        prompt_c = fs1 + '\n' + fs2 + '\n' + fs3 + '\n' + fs4 + '\n' + prompt_template[triplet['relation']].render(
                            t=str(triplet[f'release_date_{year}'])[:4], sub=triplet['subject'], obj=triplet[f'objects_{year}'][0]
                        )
                    prompts.append(prompt_c)
            
            # Print the first prompt for debugging
            if i == 0:
                print(prompts[0])
            
            if "gpt" in model_name.lower():
                outputs = gpt_batch_generation(model_name, prompts, temperature, max_tokens)
            elif "gemini" in model_name.lower():
                outputs = gemini_batch_generation(model_name, prompts, temperature, max_tokens)
            else:
                outputs = llm.generate(prompts, sampling_params)
            
            # Process and store the outputs for each year
            output_index = 0
            for year in years:
                year_outputs = []
                for _ in range(5):
                    if "gpt" not in model_name.lower() or "gemini" not in model_name.lower():
                        generated_text = outputs[output_index].outputs[0].text
                    else:
                        generated_text = outputs[output_index]
                    year_outputs.append(generated_text)
                    output_index += 1
                triplet_outputs[f"objects_{year}"] = year_outputs
            
            # Print the first output for the first year for debugging
            if i == 0:
                print(triplet_outputs[f"objects_{years[0]}"][0])
            
            # Append the processed triplet outputs to the final list
            all_outputs_ct.append(triplet_outputs)
        
        if save_results:
            output_dir = f'./Results/{model_name}'
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f'Result_TimeVariant_{domain}_Dynamic_temp{temperature}.json'
            output_file = os.path.join(output_dir, output_filename)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_outputs_ct, f)
            print(f"Knowledge Check for {output_filename} Complete")
            
            
        print("Benchmark Static with Time.")
        for i, triplet in tqdm(enumerate(bench_data_u), total=len(bench_data_u)):
            matching_few_shots = [example for example in bench_data_fs if example['relation'] == triplet['relation']]
            triplet_outputs = {}
            years = get_years_from_triplet(triplet)
            
            prompts = []
            
            for year in years:
                for _ in range(5):
                    few_shots = []
                    for shot in matching_few_shots:
                        if f'objects_{year}' in shot:
                            few_shots.append(shot)
                    
                    selected_few_shots = random.sample(few_shots, 4)
                    
                    if template == "generation":
                        prompt_u = prompt_KT_t.render(
                            # Few-shot examples for current year
                            t1=str(selected_few_shots[0][f'release_date_{year}'])[:4], sub1=selected_few_shots[0]['subject'], rel1=selected_few_shots[0]['relation'], obj1=selected_few_shots[0][f'objects_{year}'][0],
                            t2=str(selected_few_shots[1][f'release_date_{year}'])[:4], sub2=selected_few_shots[1]['subject'], rel2=selected_few_shots[1]['relation'], obj2=selected_few_shots[1][f'objects_{year}'][0],
                            t3=str(selected_few_shots[2][f'release_date_{year}'])[:4], sub3=selected_few_shots[2]['subject'], rel3=selected_few_shots[2]['relation'], obj3=selected_few_shots[2][f'objects_{year}'][0],
                            t4=str(selected_few_shots[3][f'release_date_{year}'])[:4], sub4=selected_few_shots[3]['subject'], rel4=selected_few_shots[3]['relation'], obj4=selected_few_shots[3][f'objects_{year}'][0],
                            # Current triplet data
                            t=str(triplet[f'release_date_{year}'])[:4], sub=triplet["subject"], rel=triplet["relation"]
                        )
                    elif template == "QA":
                        prompt_template = eval(f'prompt_QA_{domain}_t')
                        fs1 = prompt_template[triplet['relation']].render(
                            t=str(selected_few_shots[0][f'release_date_{year}'])[:4], sub=selected_few_shots[0]['subject'], Ans1=selected_few_shots[0][f'objects_{year}'][0], Ans2=selected_few_shots[0]['distractor'][2], Ans3=selected_few_shots[0]['distractor'][1], Ans4=selected_few_shots[0]['distractor'][0]
                        ) + '\n' + '(a) ' + selected_few_shots[0][f'objects_{year}'][0] + '\n'
                        
                        fs2 = prompt_template[triplet['relation']].render(
                            t=str(selected_few_shots[1][f'release_date_{year}'])[:4], sub=selected_few_shots[1]['subject'], Ans1=selected_few_shots[1]['distractor'][0], Ans2=selected_few_shots[1]['distractor'][1], Ans3=selected_few_shots[1][f'objects_{year}'][0], Ans4=selected_few_shots[1]['distractor'][2]
                        ) + '\n' + '(c) ' + selected_few_shots[1][f'objects_{year}'][0] + '\n'
                        
                        fs3 = prompt_template[triplet['relation']].render(
                            t=str(selected_few_shots[2][f'release_date_{year}'])[:4], sub=selected_few_shots[2]['subject'], Ans1=selected_few_shots[2]['distractor'][2], Ans2=selected_few_shots[2]['distractor'][1], Ans3=selected_few_shots[2][f'objects_{year}'][0], Ans4=selected_few_shots[2]['distractor'][0]
                        ) + '\n' + '(c) ' + selected_few_shots[2][f'objects_{year}'][0] + '\n'
                        
                        fs4 = prompt_template[triplet['relation']].render(
                            t=str(selected_few_shots[3][f'release_date_{year}'])[:4], sub=selected_few_shots[3]['subject'], Ans1=selected_few_shots[3]['distractor'][1], Ans2=selected_few_shots[3]['distractor'][0], Ans3=selected_few_shots[3]['distractor'][2], Ans4=selected_few_shots[3][f'objects_{year}'][0]
                        ) + '\n' + '(d) ' + selected_few_shots[3][f'objects_{year}'][0] + '\n'
                        
                        prompt_u = fs1 + '\n' + fs2 + '\n' + fs3 + '\n' + fs4 + '\n' + prompt_template[triplet['relation']].render(
                            t=str(triplet[f'release_date_{year}'])[:4], sub=triplet['subject'], Ans1=triplet['distractor'][0], Ans2=triplet['distractor'][2], Ans3=triplet['distractor'][1], Ans4=triplet[f'objects_{year}'][0]
                        ) + '\n'
                    elif template == "TF":
                        prompt_template = eval(f'prompt_TF_{domain}_t')
                        fs1 = prompt_template[triplet['relation']].render(
                            t=str(selected_few_shots[0][f'release_date_{year}'])[:4], sub=selected_few_shots[0]['subject'], obj=selected_few_shots[0]['distractor'][2]
                        ) + 'A. false' + '\n'
                        
                        fs2 = prompt_template[triplet['relation']].render(
                            t=str(selected_few_shots[1][f'release_date_{year}'])[:4], sub=selected_few_shots[1]['subject'], obj=selected_few_shots[1]['distractor'][0]
                        ) + 'A. false' + '\n'
                        
                        fs3 = prompt_template[triplet['relation']].render(
                            t=str(selected_few_shots[2][f'release_date_{year}'])[:4], sub=selected_few_shots[2]['subject'], obj=selected_few_shots[2]['distractor'][2]
                        ) + 'A. false' + '\n'
                        
                        fs4 = prompt_template[triplet['relation']].render(
                            t=str(selected_few_shots[3][f'release_date_{year}'])[:4], sub=selected_few_shots[3]['subject'], obj=selected_few_shots[3][f'objects_{year}'][0]
                        ) + 'A. true' + '\n'
                        
                        prompt_u = fs1 + '\n' + fs2 + '\n' + fs3 + '\n' + fs4 + '\n' + prompt_template[triplet['relation']].render(
                            t=str(triplet[f'release_date_{year}'])[:4], sub=triplet['subject'], obj=triplet[f'objects_{year}'][0]
                        )
                    prompts.append(prompt_u)
                
            if i == 0:
                print(prompts[0])
                
            if "gpt" in model_name.lower():
                outputs = gpt_batch_generation(model_name, prompts, temperature, max_tokens)
            elif "gemini" in model_name.lower():
                outputs = gemini_batch_generation(model_name, prompts, temperature, max_tokens)
            else:
                outputs = llm.generate(prompts, sampling_params)
    
            # Process the outputs
            output_index = 0
            for year in years:
                year_outputs = []
                for _ in range(5):
                    if "gpt" not in model_name.lower() or "gemini" not in model_name.lower():
                        generated_text = outputs[output_index].outputs[0].text
                    else:
                        generated_text = outputs[output_index]
                    year_outputs.append(generated_text)
                    output_index += 1
                triplet_outputs[f"objects_{year}"] = year_outputs
            
            if i == 0:
                print(triplet_outputs[f"objects_{years[0]}"][0])
            
            all_outputs_ut.append(triplet_outputs)
        
        if save_results:
            output_dir = f'./Results/{model_name}'
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f'Result_TimeVariant_{domain}_Static_temp{temperature}.json'
            output_file = os.path.join(output_dir, output_filename)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_outputs_ut, f)
            print(f"Knowledge Check for {output_filename} Complete")
    
    ### Inference in Legal Domain ###
    elif domain == "Legal":
        bench_data_c, bench_data_u, bench_data_fs = load_data(domain)
        all_outputs_ct = []
        all_outputs_ut = []
        print("Benchmark Dynamic with Time.")
        for i, question in tqdm(enumerate(bench_data_c), total=len(bench_data_c)):
            matching_few_shots = [example for example in bench_data_fs if example['question_type'] == question['question_type']]
            question_outputs = {}
            years = get_years_from_triplet(question)
            
            prompts = []
            for year in years:
                for _ in range(5):
                    few_shots = random.sample(matching_few_shots, 4)
                    if template == "generation":
                        prompt_c = prompt_GEN_Legal_t.render(
                            t1=str(few_shots[0][f'release_date_{year}'])[:4], q1=few_shots[0]['question'], a1=few_shots[0][f'answers_{year}'][0],
                            t2=str(few_shots[1][f'release_date_{year}'])[:4], q2=few_shots[1]['question'], a2=few_shots[1][f'answers_{year}'][0],
                            t3=str(few_shots[2][f'release_date_{year}'])[:4], q3=few_shots[2]['question'], a3=few_shots[2][f'answers_{year}'][0],
                            t4=str(few_shots[3][f'release_date_{year}'])[:4], q4=few_shots[3]['question'], a4=few_shots[3][f'answers_{year}'][0],
                            t=str(question[f'release_date_{year}'])[:4], q=question["question"],
                        )
                    elif template == "QA":
                        prompt_template = eval(f'prompt_QA_{domain}_t')
                        fs1 = prompt_template.render(
                            t=str(few_shots[0][f'release_date_{year}'])[:4], q=few_shots[0]['question'], Ans1=few_shots[0]['distractor'][0], Ans2=few_shots[0][f'answers_{year}'][0], Ans3=few_shots[0]['distractor'][1], Ans4=few_shots[0]['distractor'][2]
                        ) + '\n' + '(b) ' + few_shots[0][f'answers_{year}'][0] + '\n'
                        
                        fs2 = prompt_template.render(
                            t=str(few_shots[1][f'release_date_{year}'])[:4], q=few_shots[1]['question'], Ans1=few_shots[1][f'answers_{year}'][0], Ans2=few_shots[1]['distractor'][1], Ans3=few_shots[1]['distractor'][2], Ans4=few_shots[1]['distractor'][0]
                        ) + '\n' + '(a) ' + few_shots[1][f'answers_{year}'][0] + '\n'
                        
                        fs3 = prompt_template.render(
                            t=str(few_shots[2][f'release_date_{year}'])[:4], q=few_shots[2]['question'], Ans1=few_shots[2][f'answers_{year}'][0], Ans2=few_shots[2]['distractor'][1], Ans3=few_shots[2]['distractor'][0], Ans4=few_shots[2]['distractor'][2]
                        ) + '\n' + '(a) ' + few_shots[2][f'answers_{year}'][0] + '\n'
                        
                        fs4 = prompt_template.render(
                            t=str(few_shots[3][f'release_date_{year}'])[:4], q=few_shots[3]['question'], Ans1=few_shots[3]['distractor'][1], Ans2=few_shots[3]['distractor'][0], Ans3=few_shots[3][f'answers_{year}'][0], Ans4=few_shots[3]['distractor'][2]
                        ) + '\n' + '(c) ' + few_shots[3][f'answers_{year}'][0] + '\n'
                        
                        prompt_c = fs1 + '\n' + fs2 + '\n' + fs3 + '\n' + fs4 + '\n' + prompt_template.render(
                            t=str(question[f'release_date_{year}'])[:4], q=question["question"], Ans1=question['distractor'][0], Ans2=question['distractor'][2], Ans3=question['distractor'][1], Ans4=question[f'answers_{year}'][0]
                        ) + '\n'
                    elif template == "TF":
                        prompt_template = eval(f'prompt_TF_{domain}_t')
                        fs1 = prompt_template.render(
                            t=str(few_shots[0][f'release_date_{year}'])[:4], q=few_shots[0]['question'].replace(' ____ ', few_shots[0]['distractor'][0])
                        ) + 'A. false' + '\n'
                        
                        fs2 = prompt_template.render(
                            t=str(few_shots[1][f'release_date_{year}'])[:4], q=few_shots[1]['question'].replace(' ____ ', few_shots[1][f'answers_{year}'][0])
                        ) + 'A. true' + '\n'
                        
                        fs3 = prompt_template.render(
                            t=str(few_shots[2][f'release_date_{year}'])[:4], q=few_shots[2]['question'].replace(' ____ ', few_shots[2][f'answers_{year}'][0])
                        ) + 'A. true' + '\n'
                        
                        fs4 = prompt_template.render(
                            t=str(few_shots[3][f'release_date_{year}'])[:4], q=few_shots[3]['question'].replace(' ____ ', few_shots[3]['distractor'][1])
                        ) + 'A. false' + '\n'
                        
                        prompt_c = fs1 + '\n' + fs2 + '\n' + fs3 + '\n' + fs4 + '\n' + prompt_template.render(
                            t=str(question[f'release_date_{year}'])[:4], q=question["question"].replace(' ____ ', question[f'answers_{year}'][0])
                        )    
                    prompts.append(prompt_c)
                
            if i == 0:
                print(prompts[0])
                
            if "gpt" in model_name.lower():
                outputs = gpt_batch_generation(model_name, prompts, temperature, max_tokens)
            elif "gemini" in model_name.lower():
                outputs = gemini_batch_generation(model_name, prompts, temperature, max_tokens)
            else:
                outputs = llm.generate(prompts, sampling_params)
    
            # Process the outputs
            output_index = 0
            for year in years:
                year_outputs = []
                for _ in range(5):
                    if "gpt" not in model_name.lower() or "gemini" not in model_name.lower():
                        generated_text = outputs[output_index].outputs[0].text
                    else:
                        generated_text = outputs[output_index]
                    year_outputs.append(generated_text)
                    output_index += 1
                question_outputs[f"objects_{year}"] = year_outputs
            
            if i == 0:
                print(question_outputs[f"objects_{years[0]}"][0])
            
            all_outputs_ct.append(question_outputs)
        
        if save_results:
            output_dir = f'./Results/{model_name}'
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f'Result_TimeVariant_{domain}_Dynamic_temp{temperature}.json'
            output_file = os.path.join(output_dir, output_filename)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_outputs_ct, f)
            print(f"Knowledge Check for {output_filename} Complete")
            
            
        print("Benchmark Static with Time.")
        for i, question in tqdm(enumerate(bench_data_u), total=len(bench_data_u)):
            matching_few_shots = [example for example in bench_data_fs if example['question_type'] == question['question_type']]
            question_outputs = {}
            years = get_years_from_triplet(question)
            
            prompts = []
            for year in years:
                for _ in range(5):
                    few_shots = random.sample(matching_few_shots, 4)
                    if template == "generation":
                        prompt_u = prompt_GEN_Legal_t.render(
                            t1=str(few_shots[0][f'release_date_{year}'])[:4], q1=few_shots[0]['question'], a1=few_shots[0][f'answers_{year}'][0],
                            t2=str(few_shots[1][f'release_date_{year}'])[:4], q2=few_shots[1]['question'], a2=few_shots[1][f'answers_{year}'][0],
                            t3=str(few_shots[2][f'release_date_{year}'])[:4], q3=few_shots[2]['question'], a3=few_shots[2][f'answers_{year}'][0],
                            t4=str(few_shots[3][f'release_date_{year}'])[:4], q4=few_shots[3]['question'], a4=few_shots[3][f'answers_{year}'][0],
                            t=str(question[f'release_date_{year}'])[:4], q=question["question"],
                        )
                    elif template == "QA":
                        prompt_template = eval(f'prompt_QA_{domain}_t')
                        fs1 = prompt_template.render(
                            t=str(few_shots[0][f'release_date_{year}'])[:4], q=few_shots[0]['question'], Ans1=few_shots[0]['distractor'][0], Ans2=few_shots[0][f'answers_{year}'][0], Ans3=few_shots[0]['distractor'][1], Ans4=few_shots[0]['distractor'][2]
                        ) + '\n' + '(b) ' + few_shots[0][f'answers_{year}'][0] + '\n'
                        
                        fs2 = prompt_template.render(
                            t=str(few_shots[1][f'release_date_{year}'])[:4], q=few_shots[1]['question'], Ans1=few_shots[1][f'answers_{year}'][0], Ans2=few_shots[1]['distractor'][1], Ans3=few_shots[1]['distractor'][2], Ans4=few_shots[1]['distractor'][0]
                        ) + '\n' + '(a) ' + few_shots[1][f'answers_{year}'][0] + '\n'
                        
                        fs3 = prompt_template.render(
                            t=str(few_shots[2][f'release_date_{year}'])[:4], q=few_shots[2]['question'], Ans1=few_shots[2][f'answers_{year}'][0], Ans2=few_shots[2]['distractor'][1], Ans3=few_shots[2]['distractor'][0], Ans4=few_shots[2]['distractor'][2]
                        ) + '\n' + '(a) ' + few_shots[2][f'answers_{year}'][0] + '\n'
                        
                        fs4 = prompt_template.render(
                            t=str(few_shots[3][f'release_date_{year}'])[:4], q=few_shots[3]['question'], Ans1=few_shots[3]['distractor'][1], Ans2=few_shots[3]['distractor'][0], Ans3=few_shots[3][f'answers_{year}'][0], Ans4=few_shots[3]['distractor'][2]
                        ) + '\n' + '(c) ' + few_shots[3][f'answers_{year}'][0] + '\n'
                        
                        prompt_u = fs1 + '\n' + fs2 + '\n' + fs3 + '\n' + fs4 + '\n' + prompt_template.render(
                            t=str(question[f'release_date_{year}'])[:4], q=question["question"], Ans1=question['distractor'][0], Ans2=question['distractor'][2], Ans3=question['distractor'][1], Ans4=question[f'answers_{year}'][0]
                        ) + '\n'
                    elif template == "TF":
                        prompt_template = eval(f'prompt_TF_{domain}_t')
                        fs1 = prompt_template.render(
                            t=str(few_shots[0][f'release_date_{year}'])[:4], q=few_shots[0]['question'].replace(' ____ ', few_shots[0]['distractor'][0])
                        ) + 'A. false' + '\n'
                        
                        fs2 = prompt_template.render(
                            t=str(few_shots[1][f'release_date_{year}'])[:4], q=few_shots[1]['question'].replace(' ____ ', few_shots[1][f'answers_{year}'][0])
                        ) + 'A. true' + '\n'
                        
                        fs3 = prompt_template.render(
                            t=str(few_shots[2][f'release_date_{year}'])[:4], q=few_shots[2]['question'].replace(' ____ ', few_shots[2][f'answers_{year}'][0])
                        ) + 'A. true' + '\n'
                        
                        fs4 = prompt_template.render(
                            t=str(few_shots[3][f'release_date_{year}'])[:4], q=few_shots[3]['question'].replace(' ____ ', few_shots[3]['distractor'][1])
                        ) + 'A. false' + '\n'
                        
                        prompt_u = fs1 + '\n' + fs2 + '\n' + fs3 + '\n' + fs4 + '\n' + prompt_template.render(
                            t=str(question[f'release_date_{year}'])[:4], q=question["question"].replace(' ____ ', question[f'answers_{year}'][0])
                        )
                    prompts.append(prompt_u)
                
            if i == 0:
                print(prompts[0])
                
            if "gpt" in model_name.lower():
                outputs = gpt_batch_generation(model_name, prompts, temperature, max_tokens)
            elif "gemini" in model_name.lower():
                outputs = gemini_batch_generation(model_name, prompts, temperature, max_tokens)
            else:
                outputs = llm.generate(prompts, sampling_params)
    
            # Process the outputs
            output_index = 0
            for year in years:
                year_outputs = []
                for _ in range(5):
                    if "gpt" not in model_name.lower() or "gemini" not in model_name.lower():
                        generated_text = outputs[output_index].outputs[0].text
                    else:
                        generated_text = outputs[output_index]
                    year_outputs.append(generated_text)
                    output_index += 1
                question_outputs[f"objects_{year}"] = year_outputs
            
            if i == 0:
                print(question_outputs[f"objects_{years[0]}"][0])
            
            all_outputs_ut.append(question_outputs)
        
        if save_results:
            output_dir = f'./Results/{model_name}'
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f'Result_TimeVariant_{domain}_Static_temp{temperature}.json'
            output_file = os.path.join(output_dir, output_filename)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_outputs_ut, f)
            print(f"Knowledge Check for {output_filename} Complete")
    
    ### Inference in Time invariant Domain ###
    else:
        bench_data, bench_data_fs = load_data(domain)
        all_outputs_t = []
        print("Benchmark for Time Invariant with time")
        for i, triplet in tqdm(enumerate(bench_data), total=len(bench_data)):
            matching_few_shots = [example for example in bench_data_fs if example['relation'] == triplet['relation']]
            triplet_outputs = {}
            years = ["2020", "2021", "2022", "2023", "2024"]
                
            prompts = []
            for year in years:
                for _ in range(5):
                    few_shots = random.sample(matching_few_shots, 4)
                    if template == "generation":
                        prompt = prompt_KT_t.render(
                            t1=year, sub1=few_shots[0]['subject'], rel1=few_shots[0]['relation'], obj1=few_shots[0]['objects'][0],
                            t2=year, sub2=few_shots[1]['subject'], rel2=few_shots[1]['relation'], obj2=few_shots[1]['objects'][0],
                            t3=year, sub3=few_shots[2]['subject'], rel3=few_shots[2]['relation'], obj3=few_shots[2]['objects'][0],
                            t4=year, sub4=few_shots[3]['subject'], rel4=few_shots[3]['relation'], obj4=few_shots[3]['objects'][0],
                            t=year, sub=triplet["subject"], rel=triplet["relation"]
                        )
                        
                    elif template == "QA" and domain == "Math":
                        prompt_template = eval(f'prompt_QA_{domain}_t')
                        fs1 = prompt_template[triplet['relation']].render(
                            t=year, sub=few_shots[0]['subject'], Ans1=few_shots[0]['distractor'][0], Ans2=few_shots[0]['distractor'][2], Ans3=few_shots[0]['distractor'][1], Ans4=few_shots[0]['objects'][0]
                        ) + '\n' + '(d) ' + few_shots[0]['objects'][0] + '\n'
                        
                        fs2 = prompt_template[triplet['relation']].render(
                            t=year, sub=few_shots[1]['subject'], Ans1=few_shots[1]['distractor'][0], Ans2=few_shots[1]['objects'][0], Ans3=few_shots[1]['distractor'][2], Ans4=few_shots[1]['distractor'][1]
                        ) + '\n' + '(b) ' + few_shots[1]['objects'][0] + '\n'
                        
                        fs3 = prompt_template[triplet['relation']].render(
                            t=year, sub=few_shots[2]['subject'], Ans1=few_shots[2]['distractor'][2], Ans2=few_shots[2]['distractor'][1], Ans3=few_shots[2]['distractor'][0], Ans4=few_shots[2]['objects'][0]
                        ) + '\n' + '(d) ' + few_shots[2]['objects'][0] + '\n'
                        
                        fs4 = prompt_template[triplet['relation']].render(
                            t=year, sub=few_shots[3]['subject'], Ans1=few_shots[3]['objects'][0], Ans2=few_shots[3]['distractor'][0], Ans3=few_shots[3]['distractor'][2], Ans4=few_shots[3]['distractor'][1]
                        ) + '\n' + '(a) ' + few_shots[3]['objects'][0] + '\n'
                        
                        prompt = fs1 + '\n' + fs2 + '\n' + fs3 + '\n' + fs4 + '\n' + prompt_template[triplet['relation']].render(
                            t=year, sub=triplet['subject'], Ans1=triplet['distractor'][0], Ans2=triplet['distractor'][2], Ans3=triplet['distractor'][1], Ans4=triplet['objects'][0]
                        ) + '\n'
                        
                    elif template == "QA" and domain == "CommonSense":
                        prompt_template = eval(f'prompt_QA_{domain}_t')
                        fs1 = prompt_template[triplet['relation']].render(
                            t=year, sub=few_shots[0]['subject'], Ans1=few_shots[0]['distractor']['distractor'][0], Ans2=few_shots[0]['distractor']['distractor'][2], Ans3=few_shots[0]['distractor']['distractor'][1], Ans4=few_shots[0]['objects'][0]
                        ) + '\n' + '(d) ' + few_shots[0]['objects'][0] + '\n'
                        
                        fs2 = prompt_template[triplet['relation']].render(
                            t=year, sub=few_shots[1]['subject'], Ans1=few_shots[1]['distractor']['distractor'][0], Ans2=few_shots[1]['objects'][0], Ans3=few_shots[1]['distractor']['distractor'][2], Ans4=few_shots[1]['distractor']['distractor'][1]
                        ) + '\n' + '(b) ' + few_shots[1]['objects'][0] + '\n'
                        
                        fs3 = prompt_template[triplet['relation']].render(
                            t=year, sub=few_shots[2]['subject'], Ans1=few_shots[2]['distractor']['distractor'][2], Ans2=few_shots[2]['distractor']['distractor'][1], Ans3=few_shots[2]['distractor']['distractor'][0], Ans4=few_shots[2]['objects'][0]
                        ) + '\n' + '(d) ' + few_shots[2]['objects'][0] + '\n'
                        
                        fs4 = prompt_template[triplet['relation']].render(
                            t=year, sub=few_shots[3]['subject'], Ans1=few_shots[3]['objects'][0], Ans2=few_shots[3]['distractor']['distractor'][0], Ans3=few_shots[3]['distractor']['distractor'][2], Ans4=few_shots[3]['distractor']['distractor'][1]
                        ) + '\n' + '(a) ' + few_shots[3]['objects'][0] + '\n'
                        
                        prompt = fs1 + '\n' + fs2 + '\n' + fs3 + '\n' + fs4 + '\n' + prompt_template[triplet['relation']].render(
                            t=year, sub=triplet['subject'], Ans1=triplet['distractor'][0], Ans2=triplet['distractor'][2], Ans3=triplet['distractor'][1], Ans4=triplet['objects'][0]
                        ) + '\n'

                    elif template == "TF" and domain == "Math":
                        prompt_template = eval(f'prompt_TF_{domain}_t')
                        fs1 = prompt_template[triplet['relation']].render(
                            t=year, sub=few_shots[0]['subject'], obj=few_shots[0]['objects'][0]
                        ) + 'A. true' + '\n'
                        
                        fs2 = prompt_template[triplet['relation']].render(
                            t=year, sub=few_shots[1]['subject'], obj=few_shots[1]['distractor'][0]
                        ) + 'A. false' + '\n'
                        
                        fs3 = prompt_template[triplet['relation']].render(
                            t=year, sub=few_shots[2]['subject'], obj=few_shots[2]['objects'][0]
                        ) + 'A. true' + '\n'
                        
                        fs4 = prompt_template[triplet['relation']].render(
                            t=year, sub=few_shots[3]['subject'], obj=few_shots[3]['distractor'][0]
                        ) + 'A. false' + '\n'
                        
                        prompt = fs1 + '\n' + fs2 + '\n' + fs3 + '\n' + fs4 + '\n' + prompt_template[triplet['relation']].render(
                            t=year, sub=triplet['subject'], obj=triplet['objects'][0]
                        ) 
                        
                    elif template == "TF" and domain == "CommonSense":
                        prompt_template = eval(f'prompt_TF_{domain}_t')
                        fs1 = prompt_template[triplet['relation']].render(
                            t=year, sub=few_shots[0]['subject'], obj=few_shots[0]['objects'][0]
                        ) + 'A. true' + '\n'
                        
                        fs2 = prompt_template[triplet['relation']].render(
                            t=year, sub=few_shots[1]['subject'], obj=few_shots[1]['objects'][0]
                        ) + 'A. true' + '\n'
                        
                        fs3 = prompt_template[triplet['relation']].render(
                            t=year, sub=few_shots[2]['subject'], obj=few_shots[2]['objects'][0]
                        ) + 'A. true' + '\n'
                        
                        fs4 = prompt_template[triplet['relation']].render(
                            t=year, sub=few_shots[3]['subject'], obj=few_shots[3]['distractor'][0]
                        ) + 'A. false' + '\n'
                        
                        prompt = fs1 + '\n' + fs2 + '\n' + fs3 + '\n' + fs4 + '\n' + prompt_template[triplet['relation']].render(
                            t=year, sub=triplet['subject'], obj=triplet['objects'][0]
                        )
                    prompts.append(prompt)
                    
            if i == 0:
                print(prompts[0])
                    
            if "gpt" in model_name.lower():
                outputs = gpt_batch_generation(model_name, prompts, temperature, max_tokens)
            elif "gemini" in model_name.lower():
                outputs = gemini_batch_generation(model_name, prompts, temperature, max_tokens)
            else:
                outputs = llm.generate(prompts, sampling_params)
                
            # Process the outputs
            output_index = 0
            for year in years:
                year_outputs = []
                for _ in range(5):
                    if "gpt" not in model_name.lower() or "gemini" not in model_name.lower():
                        generated_text = outputs[output_index].outputs[0].text
                    else:
                        generated_text = outputs[output_index]
                    year_outputs.append(generated_text)
                    output_index += 1
                triplet_outputs[f"objects_{year}"] = year_outputs
                
            if i == 0:
                print(triplet_outputs[f"objects_{years[0]}"][0])
            
            all_outputs_t.append(triplet_outputs)
        
        if save_results:
            output_dir = f'./Results/{model_name}'
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f'Result_TimeInvariant_{domain}_temp{temperature}.json'
            output_file = os.path.join(output_dir, output_filename)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_outputs_t, f)
            print(f"Knowledge Check for {output_filename} Complete")
            

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
    parser.add_argument('--template', type=str, required=True, default="generation", help="Template for knowledge check")
    parser.add_argument('--temperature', type=float, required=True, default=0.0, help='Temperature for the experiments.')
    parser.add_argument('--token', type=str, default=None, help="Token for Huggingface model load.")
    parser.add_argument('--cache_dir', type=str, default=None, help="Use cache_dir if model already exists.")
    parser.add_argument('--save_results', type=bool, required=True, default=True, help="Save the results into json file.")
    args = parser.parse_args()
    
    knowledge_check_with_time(model_name=args.model_name, dtype=args.dtype, device_num=args.device_num, gpu_util=args.gpu_util, multi_gpu=args.multi_gpu, max_tokens=args.max_tokens, domain=args.domain, template=args.template, temperature=args.temperature, token=args.token, cache_dir=args.cache_dir, save_results=args.save_results)
        
        
if __name__ == "__main__":
    main()
