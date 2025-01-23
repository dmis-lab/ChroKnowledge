import re
import os
import json
import glob

from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

from template import *

from openai import OpenAI
api_key_openai = "YOUR API KEY"
client = OpenAI(api_key=api_key_openai)

import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
api_key_google = "YOUR API KEY"
genai.configure(api_key=api_key_google)


model_id_dict = {
    "Phi3.5_Mini": "microsoft/Phi-3.5-mini-instruct",
    "Llama3.1_8B": "meta-llama/Llama-3.1-8B-Instruct",
    "Gemma2_9B": "google/gemma-2-9b-it",
    "Mistral7B": "mistralai/Mistral-7B-Instruct-v0.3",
    "Llama3_8B": "meta-llama/Meta-Llama-3-8B-Instruct",
    "Gemma_7B": "google/gemma-7b-it",
    "SOLAR_10.7B": "upstage/SOLAR-10.7B-Instruct-v1.0",
    "Llama2_7B": "meta-llama/Llama-2-7b-chat-hf",
    "Llama3.1_70B": "meta-llama/Llama-3.1-70B-Instruct",
    "SOLAR_22B": "upstage/solar-pro-preview-instruct",
    "Qwen2.5_72B": "Qwen/Qwen2.5-72B-Instruct",
    "mpt_7B": "mosaicml/mpt-7b-chat",
    "Pythia_7B": "togethercomputer/Pythia-Chat-Base-7B",
    "Nemotron3": "mgoin/nemotron-3-8b-chat-4k-sft-hf",
    "Mistral7B_Chat": "Norquinal/Mistral-7B-claude-chat"
}


### Read data ###
def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def read_jsonl_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data
    
def load_data(domain):
    if domain == "CommonSense" or domain == "Math":
        bench_dir = f"./ChroKnowBench/TimeInvariant_{domain}.jsonl"
        bench_fs = f"./ChroKnowBench/Fewshots/Fewshot_{domain}.jsonl"
        bench_data = read_jsonl_file(bench_dir)
        bench_data_fs = read_jsonl_file(bench_fs)
        return bench_data, bench_data_fs
    
    else:
        bench_dir_c = f"./ChroKnowBench/TimeVariant_{domain}_Dynamic.jsonl"
        bench_dir_u = f"./ChroKnowBench/TimeVariant_{domain}_Static.jsonl"
        bench_fs = f"./ChroKnowBench/Fewshots/Fewshot_{domain}.jsonl"
        bench_data_c = read_jsonl_file(bench_dir_c)
        bench_data_u = read_jsonl_file(bench_dir_u)
        bench_data_fs = read_jsonl_file(bench_fs)
        return bench_data_c, bench_data_u, bench_data_fs
    
def load_data_with_timestamp(model_name, domain):
    bench_dir_c = f"./ChroKnowBench/TimeVariant_{domain}_Dynamic.jsonl"
    bench_dir_u = f"./ChroKnowBench/TimeVariant_{domain}_Static.jsonl"
    timestamp_dir_c = f'./ChronoGap/{model_name}/Timestamp_{domain}_Dynamic.json'
    timestamp_dir_u = f'./ChronoGap/{model_name}/Timestamp_{domain}_Static.json'
    bench_data_c = read_jsonl_file(bench_dir_c)
    bench_data_u = read_jsonl_file(bench_dir_u)
    timestamp_c = read_json_file(timestamp_dir_c)
    timestamp_u = read_json_file(timestamp_dir_u)
    return bench_data_c, bench_data_u, timestamp_c, timestamp_u
    
def get_years_from_triplet(triplet):
    years = []
    for key in triplet.keys():
        if key.startswith('release_date_'):
            year = str(triplet[key])[:4]
            years.append(year)
    return sorted(set(years))

def get_object_for_year(triplet):
    years = ['2010', '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023', '2024']
    for year in years:
        if f'objects_{year}' in triplet:
            return f'objects_{year}'
    
### Model Path ###
def get_model(model_name, token=None, cache_dir=None):
    
    model_id = model_id_dict[model_name]
    
    if cache_dir != None:
        model_dir_pattern = cache_dir
    else:
        base_dir = "./models"
        os.makedirs(base_dir, exist_ok=True)
        if "phi" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=base_dir + "/" + model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=base_dir + "/" + model_name, trust_remote_code=True)
        elif "solar" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=base_dir + "/" + model_name)
            model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=base_dir + "/" + model_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=base_dir + "/" + model_name, token=token)
            model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=base_dir + "/" + model_name, token=token)
        model_dir_pattern = os.path.join(base_dir, f"{model_name}*")
    
    model_dirs = glob.glob(model_dir_pattern)
    if not model_dirs:
        raise ValueError(f"No matching directory found for model: {model_name}")
    model_dir = model_dirs[0]
    hf_model_pattern = os.path.join(model_dir, "models--*")
    hf_model_dirs = glob.glob(hf_model_pattern)
    if not hf_model_dirs:
        raise ValueError(f"No matching huggingface directory found for model: {model_name}")
    hf_model_dir = hf_model_dirs[0]
    snapshots_dir = os.path.join(hf_model_dir, "snapshots")
    cache_dirs = glob.glob(os.path.join(snapshots_dir, "*"))
    if not cache_dirs:
        raise ValueError(f"No cache directory found for model: {model_name}")
    cache_dir = cache_dirs[0]
    
    return cache_dir

### GPT Generation ###
def gpt_generation(model_name, input, temperature, max_tokens):
  
    response = client.chat.completions.create(
      model=model_name,
      messages=[
        {
          "role": "system",
          "content": "You are a helpful assistant."
        },
        {
          "role": "user",
          "content": f"{input}"
        }
      ],
      temperature=temperature,
      max_tokens=max_tokens,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      seed=42
    )
  
    result = response.choices[0].message.content

    return result

### GPT Batch Generation ###
def gpt_batch_generation(model_name, inputs, temperature, max_tokens):
    results = []
    for input in inputs:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": f"{input}"
                }
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            seed=42
        )
        result = response.choices[0].message.content
        results.append(result)
    return results

### Gemini Generation ###
def gemini_generation(model_name, input, temperature, max_tokens):
    
    model = genai.GenerativeModel(model_name)
    response = model.generate_content(
                input,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    top_p=1
                ),
                safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
    result = response.candidates[0].content.parts[0].text
    
    return result

### Gemini Batch Generation ###
def gemini_batch_generation(model_name, inputs, temperature, max_tokens):
    model = genai.GenerativeModel(model_name)
    results = []
    for input in inputs:
        response = model.generate_content(
                input,
                generation_config=genai.types.GenerationConfig(
                    candidate_count=1,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    top_p=1
                ),
                safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        result = response.candidates[0].content.parts[0].text
        results.append(result)
    return results


### ChroKnow Prompt Generation Algorithm ###
def generate_chrono_ans(model_name, partial_known, target_year, triplet, llm, tokenizer, sampling_params, temperature, max_tokens, domain, prev_year_span=3, next_year_span=3):
    def extract_year_from_key(key):
        """Extracts the year from keys like 'objects_2021'."""
        return int(key.split('_')[1])

    def find_reference_data(partial_known, target_year, direction='previous'):
        """Finds the closest reference data in the specified direction (previous or next)."""
        target_year_int = extract_year_from_key(target_year)
        years = sorted([extract_year_from_key(key) for key in partial_known.keys()])
        target_index = years.index(target_year_int)

        if direction == 'previous':
            for year in reversed(years[:target_index]):
                year_key = f'objects_{year}'
                if (partial_known[year_key]["category"] in ["correct", "partial_correct1", "partial_correct2"] 
                        or "chrono_ans" in partial_known[year_key]):
                    return year, partial_known[year_key]
        else:  # next
            for year in years[target_index + 1:]:
                year_key = f'objects_{year}'
                if (partial_known[year_key]["category"] in ["correct", "partial_correct1", "partial_correct2"] 
                        or "chrono_ans" in partial_known[year_key]):
                    return year, partial_known[year_key]

        return None, None

    def create_example(reference_data):
        """Creates an example answer based on the reference data."""
        if "chrono_ans" in reference_data:
            # If chrono_ans is a list, get the most common element
            chrono_ans = reference_data["chrono_ans"]
            if isinstance(chrono_ans, list):
                return most_common_entity(chrono_ans)
            else:
                return chrono_ans
        elif reference_data["category"] == "correct":
            return most_common_entity(reference_data["temp0_ans"])
        else:
            return most_common_entity(reference_data["temp7_ans"])

    def most_common_entity(ans_list):
        """Finds the most common entity in the answer list."""
        if not ans_list:  # 리스트가 비어 있으면 None 또는 기본값 반환
            return "None"
        return Counter(map(str.lower, ans_list)).most_common(1)[0][0]

    def extract_answer(response_text):
        """Extracts the answer from the model's response using various patterns."""
        answer_patterns = [
            r"A\. (.*)",                        # "A. [answer]"
            r"Candidate A\. (.*)",              # "Candidate A. [answer]"
            r"Answer\. (.*)",                   # "Answer. [answer]"
            r"Final Answer\. (.*)",             # "Final Answer. [answer]"
            r"Correct Answer\. (.*)",           # "Correct Answer. [answer]"
            r"Candidate Answer\. (.*)",         # "Candidate Answer. [answer]"
            r".*the answer is (.*)",            # "the answer is [answer]" (sentence end)
            r".*the correct answer is (.*)",    # "the correct answer is [answer]" (sentence end)
            r".*Answer:\s*(.*)",                # "Answer: [answer]" (colon separator)
            r".*A\.\s*(.*)",                    # For cases where A. appears multiple times, match the last occurrence
            r"Answer to question is (.*)",      # "Answer to question is [answer]"
            r"The correct answer is (.*)",      # "The correct answer is [answer]"
            r"\d{4}, [^,]+, [^,]+, (.*)"        # Matches the pattern "year, subject, relation, [answer]"
        ]

        # Iterate through patterns and find the first match
        for pattern in answer_patterns:
            match = re.search(pattern, response_text)
            if match:
                return match.group(1).strip()

        # If no match, return a default value (e.g., empty string or None)
        return None  # or return "" if preferred

    def create_prompt(model_name, domain, reference_year, target_year, triplet, accumulated_prompt, answer=None, tentative_ans=None, direction='previous'):
        """
        Generates a prompt for a specific year and adds it to the accumulated prompt in correct order.
        Ensures target year content is only added once, and updates the Tentative_Answer each time.
        """
        if domain == "Biomedical":
            # Build question and answer for the reference year
            shot = prompt_KT_zs_t.render(
                t=str(reference_year), sub=triplet["subject"], rel=triplet["relation"]
            )

            if answer:
                shot += f"A. {answer}\n"

            # Check if target_year's question already exists in the accumulated prompt
            target_question = prompt_KT_zs_t.render(
                t=target_year.split('_')[1], sub=triplet["subject"], rel=triplet["relation"]
            )
            
        elif domain == "Legal":
            # Build question and answer for the reference year
            shot = prompt_GEN_zs_t.render(
                t=str(reference_year), sub=triplet["question"]
            )

            if answer:
                shot += f"A. {answer}\n"

            # Check if target_year's question already exists in the accumulated_prompt
            target_question = prompt_GEN_zs_t.render(
                t=target_year.split('_')[1], sub=triplet["question"]
            )
            
        elif domain == "General":
            # Build question and answer for the reference year
            shot = prompt_KT_zs_t.render(
                t=str(reference_year), sub=triplet["subject"], rel=triplet["relation"]
            )

            if answer:
                shot += f"A. {answer}\n"

            # Check if target_year's question already exists in the accumulated prompt
            target_question = prompt_KT_zs_t.render(
                t=target_year.split('_')[1], sub=triplet["subject"], rel=triplet["relation"]
            )
        
        # Use the most common element from tentative_ans (if tentative_ans is a list, extract the most common element)
        if isinstance(tentative_ans, list):
            tentative_ans = most_common_entity(tentative_ans)

        # Update the target year with Tentative_Answer, ensure it's added only once
        if target_question in accumulated_prompt:
            accumulated_prompt = re.sub(
                rf"{re.escape(target_question)}Candidate A\..*?\n",
                target_question + (f"Candidate A. {tentative_ans}\n" if tentative_ans else "Candidate A. [Object]\n"),
                accumulated_prompt
            )
        else:
            accumulated_prompt = target_question + (f"Candidate A. {tentative_ans}\n" if tentative_ans else "Candidate A. [Object]\n")

        # Ensure that previous years are inserted before the target year
        if direction == 'previous':
            # Insert reference year (previous year) before the existing accumulated_prompt
            accumulated_prompt_content = shot + '\n' + accumulated_prompt
        elif direction == 'next':
            # Insert the next year after the target year
            accumulated_prompt_content = accumulated_prompt + '\n' + shot
        else:
            raise ValueError("Invalid direction. Must be 'previous' or 'next'.")

        # Select the correct system prompt based on the presence of tentative_ans
        if tentative_ans:
            system_prompt = chrono_cand_system  # Use chrono_cand_system if tentative_ans is present
        else:
            system_prompt = chrono_none_system  # Use chrono_none_system if tentative_ans is None

        if "gemma" in model_name.lower():
            return [
                {"role": "user", "content": system_prompt + '\n\n' + accumulated_prompt_content} # Gemma model doesn't have system prompt
            ]
        else:
            return [
            {"role": "system", "content": system_prompt},  # Use the selected system prompt
            {"role": "user", "content": accumulated_prompt_content}
            ]

    # Step 1: Start by checking for previous or next year data
    prev_year, prev_data = find_reference_data(partial_known, target_year, 'previous')
    next_year, next_data = find_reference_data(partial_known, target_year, 'next')

    if not prev_data and not next_data:
        print(f"No reference data found for year {target_year}. Skipping to the next index.")
        return None  # 또는 빈 값을 반환하여 이후 작업 계속 진행

    # Initialize prompt accumulation and tentative answer
    accumulated_prompt = ""
    tentative_ans = None
    tentative_ans_list = []

    # Step 2: Process previous years first if prev_data exists
    if prev_data:
        # Sort previous years in chronological order (oldest to latest before target)
        prev_years = sorted(
            [extract_year_from_key(year) for year in partial_known.keys() if extract_year_from_key(year) < extract_year_from_key(target_year)]
        )

        # Apply prev_year_span to limit the number of years processed
        prev_years = list(reversed(prev_years[-prev_year_span:]))
        # print("Processing previous years:", prev_years)

        for idx, year in enumerate(prev_years):
            year_key = f'objects_{year}'
            reference_data = partial_known[year_key]

            if idx == 0 and tentative_ans is None:
                # First iteration, use initial prev_data
                example = create_example(reference_data)
                # print("Example for first previous year:", example)
                prompt_messages = create_prompt(
                    domain=domain,
                    reference_year=year,
                    target_year=target_year,
                    triplet=triplet,
                    accumulated_prompt=accumulated_prompt,
                    answer=example,
                    tentative_ans=None,
                    direction='previous'
                )
            else:
                example = create_example(reference_data)
                prompt_messages = create_prompt(
                    domain=domain,
                    reference_year=year,
                    target_year=target_year,
                    triplet=triplet,
                    accumulated_prompt=accumulated_prompt,
                    answer=example,
                    tentative_ans=tentative_ans,
                    direction='previous'
                )

            # print("Prompt Messages (Previous):", prompt_messages)
            if "gpt" in model_name.lower():
                new_ans = gpt_result(model_name, prompt_messages, temperature, max_tokens)
            elif "gemini" in model_name.lower():
                new_ans = gemini_result(model_name, prompt_messages, temperature, max_tokens)
            else:
                new_ans = generate_llm_response(llm, prompt_messages, tokenizer, sampling_params)
            # print("New Answer (Previous):", new_ans)

            # Extract only the answer part and compare/update tentative_ans
            extracted_ans = extract_answer(new_ans)
            
            if extracted_ans is not None:
                if tentative_ans is None or extracted_ans.lower() != tentative_ans.lower():
                    tentative_ans = extracted_ans  # Update the tentative answer if it has changed
            else:
                extracted_ans = new_ans
                if tentative_ans is None or extracted_ans.lower() != tentative_ans.lower():
                    tentative_ans = extracted_ans  # Update the tentative answer if it has changed
                    
            # Accumulate each tentative answer in the list
            tentative_ans_list.append(tentative_ans)

            # Accumulate the current Q&A pair into the overall prompt
            if "gemma" in model_name.lower():
                system_prompt_patterns = [re.escape(chrono_cand_system), re.escape(chrono_none_system)]

                # Join the patterns to create a single pattern that matches either system prompt
                system_prompt_regex = r"|".join(system_prompt_patterns)

                # Remove the system prompt from prompt_messages[0]["content"]
                cleaned_prompt_content = re.sub(system_prompt_regex, "", prompt_messages[0]["content"]).strip()

                # Update accumulated_prompt with the cleaned content
                accumulated_prompt = cleaned_prompt_content
            else:
                accumulated_prompt = prompt_messages[1]["content"]

    # Step 3: Process next years if next_data exists
    if next_data:
        # Sort next years in chronological order (earliest to latest after target)
        next_years = sorted(
            [extract_year_from_key(year) for year in partial_known.keys() if extract_year_from_key(year) > extract_year_from_key(target_year)]
        )

        # Apply next_year_span to limit the number of years processed
        next_years = next_years[:next_year_span]
        # print("Processing next years:", next_years)

        for idx, year in enumerate(next_years):
            year_key = f'objects_{year}'
            reference_data = partial_known[year_key]

            if idx == 0 and tentative_ans is None:
                # No previous data processed, start with next data
                example = create_example(reference_data)
                # print("Example for first next year:", example)
                prompt_messages = create_prompt(
                    domain=domain,
                    reference_year=year,
                    target_year=target_year,
                    triplet=triplet,
                    accumulated_prompt=accumulated_prompt,
                    answer=example,
                    tentative_ans=None,
                    direction='next'
                )
            else:
                example = create_example(reference_data)
                prompt_messages = create_prompt(
                    domain=domain,
                    reference_year=year,
                    target_year=target_year,
                    triplet=triplet,
                    accumulated_prompt=accumulated_prompt,
                    answer=example,
                    tentative_ans=tentative_ans,
                    direction='next'
                )

            # print("Prompt Messages (Next):", prompt_messages)
            if "gpt" in model_name.lower():
                new_ans = gpt_result(model_name, prompt_messages, temperature, max_tokens)
            elif "gemini" in model_name.lower():
                new_ans = gemini_result(model_name, prompt_messages, temperature, max_tokens)
            else:
                new_ans = generate_llm_response(llm, prompt_messages, tokenizer, sampling_params)
            # print("New Answer (Next):", new_ans)

            # Extract only the answer part and compare/update tentative_ans
            extracted_ans = extract_answer(new_ans)

            # Check if extracted_ans is valid and compare/update tentative_ans
            if extracted_ans is not None:
                if tentative_ans is None or extracted_ans.lower() != tentative_ans.lower():
                    tentative_ans = extracted_ans  # Update the tentative answer if it has changed
            else:
                extracted_ans = new_ans
                if tentative_ans is None or extracted_ans.lower() != tentative_ans.lower():
                    tentative_ans = extracted_ans  # Update the tentative answer if it has changed
                    
            # Accumulate each tentative answer in the list
            tentative_ans_list.append(tentative_ans)

            # Accumulate the prompt content for the next iteration
            if "gemma" in model_name.lower():
                system_prompt_patterns = [re.escape(chrono_cand_system), re.escape(chrono_none_system)]

                # Join the patterns to create a single pattern that matches either system prompt
                system_prompt_regex = r"|".join(system_prompt_patterns)

                # Remove the system prompt from prompt_messages[0]["content"]
                cleaned_prompt_content = re.sub(system_prompt_regex, "", prompt_messages[0]["content"]).strip()

                # Update accumulated_prompt with the cleaned content
                accumulated_prompt = cleaned_prompt_content
            else:
                accumulated_prompt = prompt_messages[1]["content"]

    # Save the final tentative_ans as chrono_ans for the target year
    # return tentative_ans
    return tentative_ans_list

def gpt_result(model_name, prompt, temperature, max_tokens):
  
    response = client.chat.completions.create(
      model=model_name,
      messages=prompt,
      temperature=temperature,
      max_tokens=max_tokens,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0,
      seed=42
    )
  
    result = response.choices[0].message.content

    return result

def gemini_result(model_name, prompt, temperature, max_tokens):
    
    system_prompt = next(
        (msg["content"] for msg in prompt if msg["role"] == "system"), ""
    )
    user_prompt = next(
        (msg["content"] for msg in prompt if msg["role"] == "user"), ""
    )

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_prompt
    )
    response = model.generate_content(
        user_prompt,
        generation_config=genai.types.GenerationConfig(
            candidate_count=1,
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=1
        ),
        safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    )
    result = response.candidates[0].content.parts[0].text
    
    return result

def generate_llm_response(llm, prompt, tokenizer, sampling_params):
    formatted_prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    outputs = llm.generate(formatted_prompt, sampling_params)
    return outputs[0].outputs[0].text

def update_timestamp(timestamp, index, year, chrono_ans):
    timestamp["Partial_known"][index][1][year]["chrono_ans"] = chrono_ans

def save_updated_timestamp(timestamp, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(timestamp, f, ensure_ascii=False, indent=4)
        
