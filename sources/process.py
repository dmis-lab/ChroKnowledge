import json
import re
import os
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib import cm
from collections import defaultdict
from rapidfuzz import fuzz

from tqdm import tqdm
from scipy.interpolate import make_interp_spline

from sources.utils import *


### Parsing Code ###
def parse_and_structure_time(json_data):
    structured_answers = []
    for obj in json_data:
        parsed_obj = {}
        for key, value_list in obj.items():
            if key.startswith('objects_'):
                parsed_values = []
                for string in value_list:
                    if string.startswith("A. "):
                        first_answer = string.split('\n')[0][3:].strip()
                        parsed_values.append(first_answer)
                parsed_obj[key] = parsed_values
        structured_answers.append(parsed_obj)
    return structured_answers

### Load Results of Knowledge Check ###
def load_result(model_name, domain, temp_state, mode):

    if domain != "CommonSense" and domain != "Math":
        bench_dir_dynamic = f"./ChroKnowBench/TimeVariant_{domain}_Dynamic.jsonl"
        bench_dir_static = f"./ChroKnowBench/TimeVarinat_{domain}_Static.jsonl"
        
        temp0_dir_dynamic = f"./Results/{model_name}/Result_TimeVariant_{domain}_Dynamic_temp0.0.json"
        temp7_dir_dynamic = f"./Results/{model_name}/Result_TimeVariant_{domain}_Dynamic_temp0.7.json"
        temp0_dir_static = f"./Results/{model_name}/Result_TimeVariant_{domain}_Static_temp0.0.json"
        temp7_dir_static = f"./Results/{model_name}/Result_TimeVariant_{domain}_Static_temp0.7.json"
        
        bench_dynamic = read_jsonl_file(bench_dir_dynamic)
        bench_static = read_jsonl_file(bench_dir_static)
        with open(temp0_dir_dynamic, 'r') as file:
            temp0_dynamic_data = json.load(file)
        with open(temp7_dir_dynamic, 'r') as file:
            temp7_dynamic_data = json.load(file)
        with open(temp0_dir_static, 'r') as file:
            temp0_static_data = json.load(file)
        with open(temp7_dir_static, 'r') as file:
            temp7_static_data = json.load(file)
        
    else:
        bench_dir = f"./ChroKnowBench/TimeInvariant_{domain}.jsonl"
        
        temp0_dir = f"./Results/{model_name}/Result_TimeInvariant_{domain}_temp0.0.json"
        temp7_dir = f"./Results/{model_name}/Result_TimeInvariant_{domain}_temp0.7.json"
        
        bench = read_jsonl_file(bench_dir)
        with open(temp0_dir, 'r') as file:
            temp0_data = json.load(file)
        with open(temp7_dir, 'r') as file:
            temp7_data = json.load(file)
            

    if domain != "CommonSense" and domain != "Math":    
        temp0_dynamic_parsed = parse_and_structure_time(temp0_dynamic_data)
        temp7_dynamic_parsed = parse_and_structure_time(temp7_dynamic_data)
        temp0_static_parsed = parse_and_structure_time(temp0_static_data)
        temp7_static_parsed = parse_and_structure_time(temp7_static_data)
    else:
        temp0_parsed = parse_and_structure_time(temp0_data)
        temp7_parsed = parse_and_structure_time(temp7_data)
        
    if mode == "generation":
        if temp_state == "Dynamic":
            bench = bench_dynamic
            temp0_parsed_time = temp0_dynamic_parsed
            temp7_parsed_time = temp7_dynamic_parsed
        elif temp_state == "Static":
            bench = bench_static
            temp0_parsed_time = temp0_static_parsed
            temp7_parsed_time = temp7_static_parsed
        else:
            bench = bench
            temp0_parsed_time = temp0_parsed
            temp7_parsed_time = temp7_parsed


    elif mode == "QA":
        if temp_state == "Dynamic":
            bench = bench_dynamic
            temp0_parsed_time = temp0_dynamic_parsed
            temp7_parsed_time = temp7_dynamic_parsed
        elif temp_state == "Static":
            bench = bench_static
            temp0_parsed_time = temp0_static_parsed
            temp7_parsed_time = temp7_static_parsed
        else:
            bench = bench
            temp0_parsed_time = temp0_parsed
            temp7_parsed_time = temp7_parsed

    else:
        raise AssertionError("unspecified mode")
        
    
    return bench, temp0_parsed_time, temp7_parsed_time


### Classification Rule ###
#### Exact Match ####
def classify_knowledge(benchmark, temp0_ans, temp7_ans):
    # Check for empty temp0_ans first
    if not temp0_ans:
        if not temp7_ans:
            return 'incorrect'  # Both lists are empty
        elif any(ans in benchmark for ans in temp7_ans):
            return 'partial_correct2'  # Only temp7_ans has matching elements
        else:
            return 'incorrect'

    # If temp0_ans is not empty, continue with the original logic
    if all(ans in benchmark for ans in temp0_ans):
        return 'correct'
    elif any(ans in benchmark for ans in temp0_ans):
        return 'partial_correct1'
    elif not any(ans in benchmark for ans in temp0_ans):
        if any(ans in benchmark for ans in temp7_ans):
            return 'partial_correct2'
        else:
            return 'incorrect'

#### Fuzzy match ####
def classify_knowledge_fuzz(benchmark, temp0_ans, temp7_ans, threshold=70):
    def is_fuzz_match(ans, benchmark_set):
        return any(fuzz.token_set_ratio(str(ans).lower(), str(benchmark_item).lower()) >= threshold for benchmark_item in benchmark_set)

    # Check for empty temp0_ans first
    if not temp0_ans:
        if not temp7_ans:
            return 'incorrect'  # Both lists are empty
        elif any(is_fuzz_match(ans, benchmark) for ans in temp7_ans):
            return 'partial_correct2'  # Only temp7_ans has matching elements
        else:
            return 'incorrect'

    # If temp0_ans is not empty, continue with the original logic
    if all(is_fuzz_match(ans, benchmark) for ans in temp0_ans):
        return 'correct'
    elif any(is_fuzz_match(ans, benchmark) for ans in temp0_ans):
        return 'partial_correct1'
    elif not any(is_fuzz_match(ans, benchmark) for ans in temp0_ans):
        if any(is_fuzz_match(ans, benchmark) for ans in temp7_ans):
            return 'partial_correct2'
        else:
            return 'incorrect'
        
#### QA Match ####
def classify_knowledge_qa(benchmark, temp0_ans, temp7_ans):
    def is_direct_match(ans):
        return ans.startswith("(d)")  # New logic: Check if the answer starts with "(d)"

    # Check for empty temp0_ans first
    if not temp0_ans:
        if not temp7_ans:
            return 'incorrect'  # Both lists are empty
        elif any(is_direct_match(ans) for ans in temp7_ans):
            return 'partial_correct2'  # Only temp7_ans has matching elements
        else:
            return 'incorrect'

    # If temp0_ans is not empty, continue with the direct matching logic
    if all(is_direct_match(ans) for ans in temp0_ans):
        return 'correct'
    elif any(is_direct_match(ans) for ans in temp0_ans):
        return 'partial_correct1'
    elif not any(is_direct_match(ans) for ans in temp0_ans):
        if any(is_direct_match(ans) for ans in temp7_ans):
            return 'partial_correct2'
        else:
            return 'incorrect'


### Answer Classification ###
def classify_results_time(data_entries, temp0_parsed_time, temp7_parsed_time, model_name, bench_name):
    def get_benchmark(entry):
        if bench_name == "Legal":
            if any(f'answers_{year}' in entry for year in range(2010, 2025)):
                return {f'objects_{year}': set(entry[f'answers_{year}']) for year in range(2010, 2025) if f'answers_{year}' in entry}
        elif bench_name == "Biomedical":
            if any(f'objects_{year}' in entry for year in range(2020, 2025)):
                return {f'objects_{year}': set(entry[f'objects_{year}']) for year in range(2020, 2025) if f'objects_{year}' in entry}
        elif bench_name == "General":
            if any(f'objects_{year}' in entry for year in range(1000, 3000)):
                return {
                    f'objects_{year}': set(obj for obj_list in entry[f'objects_{year}'] for obj in obj_list)
                    for year in range(1000, 3000) if f'objects_{year}' in entry
                }
        elif 'objects' in entry:
            return {'all_years': set(entry['objects'])}
        else:
            raise ValueError("Unsupported data format")

    results = defaultdict(lambda: defaultdict(int))
    object_classifications = defaultdict(lambda: defaultdict(dict))

    for idx, (entry, temp0_ans, temp7_ans) in enumerate(zip(data_entries, temp0_parsed_time, temp7_parsed_time)):
        benchmark = get_benchmark(entry)
        years = set(temp0_ans.keys()).union(set(temp7_ans.keys()))

        for year in years:
            if year in benchmark:
                benchmark_objects = benchmark[year]
                category = classify_knowledge_fuzz(benchmark_objects, temp0_ans[year], temp7_ans[year])
                results[year][category] += 1
                
                for obj in benchmark_objects:
                    object_classifications[year][obj] = category
            else:
                benchmark_objects = benchmark['all_years']
                category = classify_knowledge_fuzz(benchmark_objects, temp0_ans[year], temp7_ans[year])
                results[year][category] += 1
                
                for obj in benchmark_objects:
                    object_classifications[year][obj] = category

    return results, object_classifications

def classify_results_time_fine_graining(data_entries, temp0_parsed_time, temp7_parsed_time, model_name, bench_name):
    def get_benchmark(entry):
        if bench_name == "Legal":
            if any(f'answers_{year}' in entry for year in range(2010, 2025)):
                return {f'objects_{year}': set(entry[f'answers_{year}']) for year in range(2010, 2025) if f'answers_{year}' in entry}
        elif bench_name == "Biomedical":
            if any(f'objects_{year}' in entry for year in range(2020, 2025)):
                return {f'objects_{year}': set(entry[f'objects_{year}']) for year in range(2020, 2025) if f'objects_{year}' in entry}
        elif bench_name == "General":
            if any(f'objects_{year}' in entry for year in range(1000, 3000)):
                return {
                    f'objects_{year}': set(obj for obj_list in entry[f'objects_{year}'] for obj in obj_list)
                    for year in range(1000, 3000) if f'objects_{year}' in entry
                }
        elif 'objects' in entry:
            return {'all_years': set(entry['objects'])}
        else:
            raise ValueError("Unsupported data format")

    element_classifications = defaultdict(lambda: defaultdict(dict))
    classification_indices = defaultdict(list)
    fine_grained_results = defaultdict(int)

    for idx, (entry, temp0_ans, temp7_ans) in enumerate(zip(data_entries, temp0_parsed_time, temp7_parsed_time)):
        benchmark = get_benchmark(entry)
        years = sorted(set(temp0_ans.keys()).union(set(temp7_ans.keys())))

        year_classifications = {}  # This will hold the classifications for each year for this element

        for year in years:
            if year in benchmark:
                benchmark_objects = benchmark[year]
                category = classify_knowledge_fuzz(benchmark_objects, temp0_ans.get(year, []), temp7_ans.get(year, []))
            else:
                benchmark_objects = benchmark['all_years']
                category = classify_knowledge_fuzz(benchmark_objects, temp0_ans.get(year, []), temp7_ans.get(year, []))

            # Update the classification to include temp0_ans and temp7_ans
            year_classifications[f'{year}'] = {
                'category': category,
                'temp0_ans': temp0_ans.get(year, []),
                'temp7_ans': temp7_ans.get(year, [])
            }

        # Step 2: Fine-grained classification based on yearly classifications
        categories = [details['category'] for details in year_classifications.values()]
        
        # Detecting transitions
        first_known_index = next((i for i, cat in enumerate(categories) if cat in ['correct', 'partial_correct1', 'partial_correct2']), None)
        first_unknown_index = next((i for i, cat in enumerate(categories) if cat == 'incorrect'), None)

        if all(cat == 'correct' for cat in categories):
            fine_grained_results['Known'] += 1
            classification_indices['Known'].append((idx, year_classifications))
        elif all(cat == 'incorrect' for cat in categories):
            fine_grained_results['Unknown'] += 1
            classification_indices['Unknown'].append((idx, year_classifications))
        elif first_known_index is not None and first_unknown_index is not None:
            # If the first known state comes before the first unknown state and no known states after the first unknown state
            if first_known_index < first_unknown_index and all(cat == 'incorrect' for cat in categories[first_unknown_index:]):
                fine_grained_results['Cut-off'] += 1
                classification_indices['Cut-off'].append((idx, year_classifications))
            # If the first unknown state comes before the first known state and no unknown states after the first known state
            elif first_unknown_index < first_known_index and all(cat in ['correct', 'partial_correct1', 'partial_correct2'] for cat in categories[first_known_index:]):
                fine_grained_results['Cut-off'] += 1
                classification_indices['Cut-off'].append((idx, year_classifications))
            else:
                fine_grained_results['Partial_known'] += 1
                classification_indices['Partial_known'].append((idx, year_classifications))
        else:
            fine_grained_results['Partial_known'] += 1
            classification_indices['Partial_known'].append((idx, year_classifications))

    return fine_grained_results, classification_indices


### Down Sampling for ChroKnowPrompt ###
def sampling_results(fine_grained_results, classification_indices, model_name, domain, temp_state):
    sampling_percentage = 0.1
    sampling_counts = {key: max(1, int(count * sampling_percentage)) for key, count in fine_grained_results.items()}

    # 원본 및 샘플링된 개수 출력
    print(f"\n[model name: {model_name}, temporal state: {temp_state}]")
    print("Original counts:", dict(fine_grained_results))
    print("Downsampled counts (10%):", sampling_counts)

    downsampled_indices = defaultdict(list)
    for key, sample_count in sampling_counts.items():
        if key in classification_indices:
            downsampled_indices[key] = random.sample(classification_indices[key], sample_count)

    for key in fine_grained_results.keys():
        original_count = fine_grained_results[key]
        downsampled_count = len(downsampled_indices[key])
        print(f"Category: {key}, Original count: {original_count}, Downsampled count (10%): {downsampled_count}")

    # 샘플링된 classification_indices 저장
    output_dir = f'./ChronoGap/{model_name}'
    os.makedirs(output_dir, exist_ok=True)

    timestamp_dir = f"{output_dir}/Timestamp_{domain}_{temp_state}.json'"

    with open(timestamp_dir, 'w') as file:
        json.dump(downsampled_indices, file, indent=4)


### Evaluation Basis of ChroKnowPrompt ###

def get_benchmark(entry, bench_name):
    """
    Extract the benchmark data from an entry based on the benchmark name.
    """
    if bench_name == "Legal":
        return {f'objects_{year}': set(entry[f'answers_{year}']) for year in range(2010, 2025) if f'answers_{year}' in entry}
    elif bench_name == "Biomedical":
        return {f'objects_{year}': set(entry[f'objects_{year}']) for year in range(2020, 2025) if f'objects_{year}' in entry}
    elif bench_name == "General":
        return {f'objects_{year}': set(obj for obj_list in entry[f'objects_{year}'] for obj in obj_list)
                for year in range(1000, 3000) if f'objects_{year}' in entry}
    elif 'objects' in entry:
        return {'all_years': set(entry['objects'])}
    else:
        raise ValueError("Unsupported data format")

def is_fuzz_match(ans, benchmark_set, threshold=70):
    """
    Perform fuzzy matching for an answer against a benchmark set.
    """
    return any(fuzz.token_set_ratio(str(ans).lower(), str(benchmark_item).lower()) >= threshold for benchmark_item in benchmark_set)

def clean_chrono_ans_list(chrono_ans_list):
    """
    Clean the chrono_ans list by removing entries that are not valid entities,
    such as responses that start with non-informative text (e.g., "Sure", "Understood").
    """
    cleaned_list = []
    for ans in chrono_ans_list:
        # Define the patterns for invalid entries (e.g., answers that start with non-entity phrases)
        if re.match(r'^\s*(Sure|Understood|Here|Thank you|Okay|OK|Yes|No|I am ready|I can help)', ans, re.I):
            continue  # Skip invalid entries
        # Add only valid entities
        cleaned_list.append(ans.strip())  # Remove leading/trailing whitespace if needed
    return cleaned_list