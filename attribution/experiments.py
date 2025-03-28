from attribution.dataset_utils import GSMDataset, PaddingCollator, extract_answer_gsm, is_correct_gsm
from attribution.constants import DatasetNames, ModelNames
from attribution.model_utils import Model
from torch.utils.data import DataLoader
import os
import json

import pandas as pd

def get_fewshot_prompt(dataset_name, tokenizer, config='main'):
    """Get a few-shot prompt for the given dataset"""
    dataset = GSMDataset(dataset_name, tokenizer, split='train', config=config, n_instruction_samples=8)
    return dataset.instructions

def evaluate_generations(generations, dataset):
    if dataset.name == DatasetNames.GSM8k:
        gt_answers = [extract_answer_gsm(sample) for sample in dataset.dataset['answer']]
    elif dataset.name == DatasetNames.MGSM:
        gt_answers = [sample for sample in dataset.dataset['answer_number']]
    else:
        raise ValueError(f"Dataset {dataset.name} not supported")
    
    num_correct = 0
    for gen, gt in zip(generations, gt_answers):
        if extract_answer_gsm(gen) == gt:
            num_correct += 1
            
    return num_correct / len(generations)

def conduct_experiment(model_name=ModelNames.QwenInstruct, dataset_name=DatasetNames.MGSM, config='en', batch_size=32, constrained_decoding=True, use_COT=True):
    
    model_file_name = model_name.split('/')[-1]
    dataset_file_name = dataset_name.split('/')[-1]
    constrained_decoding_str = 'constrained' if constrained_decoding else 'unconstrained'
    response_type = 'COT' if use_COT else 'regular'
    
    experiment_name = f'{dataset_file_name}_{config}_{response_type}_{constrained_decoding_str}_{model_file_name}'
    
    # If the results file already exists, skip the experiment
    if os.path.exists(f"results/generations/{experiment_name}_results.csv"):
        print(f"Experiment {experiment_name} already exists. Skipping...")
        return
    
    model = Model(model_name)
    instructions = get_fewshot_prompt(dataset_name, model.tokenizer, config) if use_COT else ''
    
    test_set = GSMDataset(dataset_name, model.tokenizer, instructions, split='test', config=config)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=PaddingCollator(model.tokenizer))
    
    generations = model.generate_responses(test_loader, constrained_decoding=constrained_decoding, config=config, use_COT=use_COT)
    
    # Save the generations
    os.makedirs('results/generations', exist_ok=True)
    results_df = pd.DataFrame(generations, columns=['response'])
    
    results_df.to_csv(f"results/generations/{experiment_name}_results.csv", index=False)
    
    # Evaluate the generations
    score = evaluate_generations(generations, test_set)
    print(f"Score for {model_name} on {dataset_name} with config {config}: {score}")
    
    # Create a file in results/metrics and record the score
    os.makedirs('results/metrics', exist_ok=True)
    
    # Save the score to a json file called generations.json, overwriting the same key in the file
    metrics = {}
    if os.path.exists('results/metrics/generations.json'):
        with open('results/metrics/generations.json', 'r') as f:
            metrics = json.load(f)
    metrics[f"{experiment_name}"] = score
    with open('results/metrics/generations.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Score saved to results/metrics/generations.json")
    
    
    