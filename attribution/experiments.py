from attribution.dataset_utils import GSMDataset, PaddingCollator, extract_answer_gsm, is_correct_gsm
from attribution.constants import DatasetNames, ModelNames
from attribution.model_utils import Model
from torch.utils.data import DataLoader
import os

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

def conduct_experiment(model_name=ModelNames.QwenInstruct, dataset_name=DatasetNames.MGSM, config='en', batch_size=32):
    
    model = Model(model_name)
    instructions = get_fewshot_prompt(dataset_name, model.tokenizer, config) # or '' for no prompt
    
    test_set = GSMDataset(dataset_name, model.tokenizer, instructions, split='test', config=config)
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=PaddingCollator(model.tokenizer))
    
    generations = model.generate_responses(test_loader)
    
    # Save the generations
    os.makedirs('results', exist_ok=True)
    results_df = pd.DataFrame(generations, columns=['response'])
    results_df.to_csv(f"results/{dataset_name.split('/')[-1]}_{config}_{model_name.split('/')[-1]}_results.csv", index=False)
    
    # Evaluate the generations
    score = evaluate_generations(generations, test_set)
    print(f"Score for {model_name} on {dataset_name} with config {config}: {score}")
    
    