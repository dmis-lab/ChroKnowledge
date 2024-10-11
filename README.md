# ChroKnowledge: Unveiling Chronological Knowledge of Language Models

### Overview
ChroKnowledge is a research framework designed to evaluate and update the chronological knowledge of large language models (LLMs). It builds on the ChroKnowBench dataset, which enables testing LLMs' ability to handle chronologically accumulated knowledge across multiple domains, including general, biomedical, legal, commonsense, and mathematical facts. This repository also features ChroKnowPrompt, a technique for in-depth prompting to enhance temporal reasoning and improve the accuracy of LLMs over a timeline.

[Link to Research Paper](#) | [Benchmark on HuggingFace](#)

![Benchmark Generation Overview](#)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ChroKnowledge.git
   cd ChroKnowledge
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset from HuggingFace:
   ```bash
   # Command to download the dataset
   ```

### Task Definition
The primary focus of ChroKnowBench is time-variant and time-invariant knowledge across different domains, as defined in Table 1 of the research paper. Knowledge is categorized into:
- **Correct**: All objects generated are included in the answer set.
- **Partial Correct**: At least one generated object is in the answer set.
- **Incorrect**: None of the generated objects are included in the answer set.

| Category        | Definition                                                                                           | Description                                                                                                 |
|-----------------|-------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| Correct         | \( \{ \hat{o_i} \mid M(D_i,s,r,t) = \hat{o_i}; M, \tau=0 \}_{i=1}^{n} \subseteq A \)                  | All objects generated with greedy decoding are entirely included within the answer set.                    |
| Partial Correct | \( \bigcup_{\tau \in \mathcal{T}} \{ \hat{o_i} \mid M(D_i, s, r, t) = \hat{o_i}; M, \tau \}_{i=1}^{n} \cap A \neq \emptyset \) | At least one generated object from greedy decoding or temperature sampling is in the answer set.           |
| Incorrect       | \( \bigcup_{\tau \in \mathcal{T}} \{ \hat{o_i} \mid M(D_i, s, r, t) = \hat{o_i}; M, \tau \}_{i=1}^{n} \cap A = \emptyset \)  | None of the generated objects, either from greedy decoding or temperature sampling, are included in the answer set. |

### Datasets
ChroKnowBench consists of datasets from multiple domains, with characteristics as shown in Table 2 of the research paper:
- **Time-variant Knowledge**: General, biomedical, and legal datasets, containing facts that change over time.
- **Time-invariant Knowledge**: Commonsense and mathematics datasets, containing facts that remain constant.

| Time Dependency | Domain (Time Frame) | # of Relations | Structured | Format                   | Temporal State | # of Examples | Source   |
|-----------------|---------------------|----------------|------------|-------------------------|----------------|---------------|----------|
| Time Variant    | General (2010-2023) | 8              | Yes        | (s, r, o, t)            | Dynamic        | 8,330         | Wikidata |
|                 |                     |                |            |                         | Static         | 8,302         | Wikidata |
|                 | Biomedical (2020-2024) | 14           | Yes        | (s, r, o, t)            | Dynamic        | 7,345         | UMLS     |
|                 |                     |                |            |                         | Static         | 7,345         | UMLS     |
|                 | Legal (2010-2023)   | 6*             | No         | QA                      | Dynamic        | 3,142         | CFR      |
|                 |                     |                |            |                         | Static         | 3,142         | CFR      |
| Time Invariant  | Commonsense         | 8              | Yes        | (s, r, o)               | Invariant      | 24,788        | CSKG     |
|                 | Math                | 12             | Yes        | (s, r, o)               | Invariant      | 2,585         | Math-KG  |

### Step 1
#### Usage
To evaluate the initial temporal knowledge of the model, you can run the `run_knowledge_check.sh` script. This script uses a variety of language models, including both open-source and proprietary LLMs, to assess their ability to recall time-sensitive information from different domains of ChroKnowBench. Below is an explanation of the key components:

1. **Set CUDA Devices**: The environment variable `CUDA_VISIBLE_DEVICES` allows you to specify which GPUs to use for running the script.
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```
   Adjust this variable according to the GPU resources available.

2. **Model Selection**: The script includes multiple language models such as `Llama3.1_8B` and `gpt-3.5-turbo-0125`. You can change the `model_name` variable to choose the model you wish to use.
   ```bash
   model_name="Llama3.1_8B"
   ```

3. **Domain and Template**: You can specify the domain (e.g., `General`, `Biomedical`, `Legal`) and the type of task (`generation`, `QA`) by setting the `domain` and `template` variables, respectively.
   ```bash
   domain="General"
   template="generation"
   ```

4. **Temperature and Decoding**: The script runs the model with different temperature settings (`0.0`, `0.7`) to capture variations in predictions. You can change the temperature variable for different behaviors.
   ```bash
   temperature=0.0
   ```

5. **Running the Script**: After setting up, you can execute the following command to run the model evaluation:
   ```bash
   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ChroKnowledge.py \
       --model_name $model_name \
       --dtype bfloat16 \
       --device_num auto \
       --gpu_util 0.90 \
       --multi_gpu $multi_gpu \
       --max_tokens 50 \
       --domain $domain \
       --template $template \
       --temperature $temperature \
       --save_results True
   ```

This script helps to evaluate how well the language models can recall time-sensitive knowledge across different domains.

#### Evaluation
Use the `run_classification.ipynb` notebook to classify and analyze the model's temporal knowledge.

### Step 2
#### Usage
To enhance the model's ability to recall and reason about chronological knowledge, you can use the ChroKnowPrompt prompting strategy. This is implemented using the `run_chroknow_prompt.sh` script. Below is a detailed breakdown of the process:

1. **Set CUDA Devices**: Specify the GPU to use by setting `CUDA_VISIBLE_DEVICES`.
   ```bash
   export CUDA_VISIBLE_DEVICES=0
   ```

2. **Model Selection**: Choose the language model for this task by modifying the `model_name` variable.
   ```bash
   model_name="Llama3.1_8B"
   ```

3. **Year Span for Temporal Reasoning**: You can configure the `prev_span` and `next_span` variables to determine how far back or forward in time the model should reason during the evaluation.
   ```bash
   prev_span=3
   next_span=3
   ```

4. **Running the Script**: To run the script and evaluate the prompting strategy, use the following command:
   ```bash
   CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python ChroKnowPrompt.py \
       --model_name $model_name \
       --dtype bfloat16 \
       --device_num auto \
       --gpu_util 0.90 \
       --multi_gpu $multi_gpu \
       --max_tokens 50 \
       --domain General \
       --temperature 0.0 \
       --prev_span $prev_span \
       --next_span $next_span \
       --save_results True
   ```

This script helps the model better navigate chronological changes in the knowledge base, allowing for a more accurate understanding of time-sensitive information.

#### Evaluation
Use the `run_evaluation.ipynb` notebook to evaluate the effectiveness of ChroKnowPrompt in improving temporal knowledge recall.

### Citation
If you use ChroKnowledge in your research, please cite:
```
@inproceedings{ChroKnowledge2025,
  title={ChroKnowledge: Unveiling Chronological Knowledge of Language Models in Multiple Domains},
  author={Anonymous},
  booktitle={ICLR},
  year={2025}
}
```

### Contact
For any questions or issues, feel free to reach out to [your email].