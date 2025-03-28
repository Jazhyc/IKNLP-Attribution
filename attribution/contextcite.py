import pandas as pd
import json
import os
import torch

import numpy as np
from tqdm import tqdm
from context_cite import ContextCiter
from attribution.model_utils import Model
from attribution.constants import ModelNames

import warnings
import torch

# Filter specific warning categories
warnings.filterwarnings("ignore", category=UserWarning)  # For general user warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # For deprecation warnings

# Enable mixed precision

# Custom JSON encoder to handle NumPy types
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

def get_attributions(model_responses_path, output_path, model_name=ModelNames.QwenInstruct):
    """
    Get attributions for model responses using ContextCiter.
    
    Args:
        model_responses_path: Path to the CSV file containing model responses
        output_path: Path to save the JSON file with attributions
        model_name: Name of the model to use for attributions
    """
    context_model = Model(model_name)
    
    # Unlike RAG, the context follows the query
    prompt_template = '{query}\n{context}'
    
    model_responses = pd.read_csv(model_responses_path)
    
    cite_df = pd.DataFrame()
    
    # Get length of model_responses
    len_responses = len(model_responses)
    
    # initialize a progress bar
    pbar = tqdm(total=len_responses, desc="Getting attributions")
    error_counter = 0
    
    # Iterate over the rows of the DataFrame
    for index, row in model_responses.iterrows():
        pbar.update(1)
        context = row['model_gen_steps']
        query = row['question']
        answer_string = row['model_answer_str']
        
        # Abstain from pre-train because it creates a new model each time
        # Constructor is needed due to processing during initialization
        cc = ContextCiter(context_model.model, context_model.tokenizer, context, query, prompt_template=prompt_template, num_ablations=32)
        
        # We want to use precomputed answers
        # See https://github.com/MadryLab/context-cite/issues/4
        _, prompt = cc._get_prompt_ids(return_prompt=True)
        cc._cache["output"] = prompt + answer_string
        
        # This returns an importance for each line in the context
        with torch.inference_mode():
            line_importance = cc.get_attributions(as_dataframe=False, verbose=False)
        
        # Get each line and importance and add to df
        lines = context.split('\n')
        
        # If number of lines and importance values do not match, raise an error
        if len(lines) != len(line_importance):
            print(f"Number of lines ({len(lines)}) and importance values ({len(line_importance)}) do not match in example {index} Skipping...")
            error_counter += 1
            continue
        
        # Create a temporary DataFrame with sample_index to identify which example each line belongs to
        temp_df = pd.DataFrame({
            'sample_index': index,  # Use the DataFrame index as sample index
            'line': lines,
            'importance': line_importance
        })
        
        cite_df = pd.concat([cite_df, temp_df], ignore_index=True)
        
    pbar.close()
    print(f"Number of errors: {error_counter}")
    
    # Record error information to a JSON file
    os.makedirs('results/metrics', exist_ok=True)
    
    # Extract experiment name from the output path
    # Expected format: 'results/contextcite/contextcite_{config}_{model_name_part}_{response_type}.json'
    experiment_name = os.path.basename(output_path).replace('.json', '')
    
    # Save the error count to a json file
    error_metrics = {}
    error_metrics_path = 'results/metrics/contextcite_errors.json'
    if os.path.exists(error_metrics_path):
        with open(error_metrics_path, 'r') as f:
            error_metrics = json.load(f)
    
    error_metrics[experiment_name] = {
        'error_count': error_counter,
        'total_samples': len_responses,
        'error_rate': error_counter / len_responses if len_responses > 0 else 0
    }
    
    with open(error_metrics_path, 'w') as f:
        json.dump(error_metrics, f, indent=4)
    
    print(f"Error metrics saved to {error_metrics_path}")
    
    # Store results as JSON
    # Create a list to store one dictionary per question
    result_list = []
    
    for sample_index, group in cite_df.groupby('sample_index'):
        original_row = model_responses.iloc[sample_index]
        
        # Create a dictionary for this sample
        sample_dict = {
            'sample_index': sample_index,
            'question': original_row['question'],
            'actual_answer': original_row['actual_answer'],
            'model_gen_answer': original_row['model_gen_answer'],
            'model_answer_str': original_row['model_answer_str'],
            'lines_and_importance': [
                {'text': row['line'], 'importance': row['importance']} 
                for _, row in group.iterrows()
            ]
        }
        
        # Add this dictionary to our results list
        result_list.append(sample_dict)
    
    # Save as JSON file with proper formatting and custom encoder
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result_list, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        
    print(f"Attributions saved to {output_path}")
    
    # Clean torch cache
    torch.cuda.empty_cache()
    
    return output_path
