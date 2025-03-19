import datasets
import re
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding

import locale
locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

class BaseDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, instructions, split='train', config='main'):
        self.name = dataset_name
        self.tokenizer = tokenizer
        self.instructions = self._load_system_prompt(dataset_name) if instructions is None else instructions
        self.dataset = self._load_dataset(dataset_name, split, config)
        self.system_prompt = {'role': 'system', 'content': self.instructions} if self.instructions else None
        
    def _load_dataset(self, dataset_name, split, config='main'):
        """Load the dataset based on its name"""
        dataset = datasets.load_dataset(dataset_name, name=config, split=split)
        return dataset
    
    def _load_system_prompt(self, dataset_name):
        prompt = ''
        # if dataset_name == DatasetNames.GSM8k:
        #     prompt = open('attribution/prompts/GSM8k.txt').read()
        return prompt
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get a tokenized example at the given index"""
        raise NotImplementedError("Subclasses must implement this method")
        
    def create_instructions(self, n_samples=5):
        """Create instructions based on the dataset"""
        raise NotImplementedError("Subclasses must implement this method")


class GSMDataset(BaseDataset):
    def __init__(self, dataset_name, tokenizer, instructions=None, split='train', config='main', n_instruction_samples=5):
        super().__init__(dataset_name, tokenizer, instructions, split, config)
        
        if split == 'test' and instructions is None:
            raise ValueError("Instructions must be provided for the test split")
        
        # Create instructions if not provided
        if instructions is None:
            self.instructions = self.instructions + self.create_examples(n_instruction_samples)
            
        self.system_prompt = {'role': 'system', 'content': self.instructions}
    
    def create_examples(self, n_samples=5):
        # Take the first n_samples from the dataset
        samples = self.dataset.select(range(n_samples))
        
        # Create a system prompt
        system_prompt = []
        for sample in samples:
            system_prompt.append(f"Question: {sample['question']}\n{sample['answer']}")
            
        # Combine all the system prompts
        return '\n\n'.join(system_prompt)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        question = sample['question']
        answer = sample['answer']
        
        user_prompt = {'role': 'user', 'content': question}
        messages = [self.system_prompt, user_prompt]
        
        # Format the messages into a chat template string but don't tokenize
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        return {
            'formatted_prompt': formatted_prompt,
            'question': question,
            'answer': answer
        }


# GSM8k evaluation functions
INVALID_ANS = "[invalid]"

# Current regex:
LAST_NUMBER_RE = r'\b\d{1,3}(?:,?\d{3})*(?:\.\d+)?(?!\d)'

def extract_answer_gsm(completion):
    
    # Find the last occurrence of a number
    match = re.findall(LAST_NUMBER_RE, completion)
    
    if match:
        return locale.atof(match[-1])
    
    # No valid format found
    return INVALID_ANS

def is_correct_gsm(model_completion, gt_example):
    gt_answer = extract_answer_gsm(gt_example)
    if gt_answer == INVALID_ANS:
        print(model_completion)
        raise ValueError("Invalid answer for ground truth")
    return extract_answer_gsm(model_completion) == gt_answer

class PaddingCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        
    def __call__(self, batch):
        """
        Collate function that passes through the formatted prompts without tokenization
        """
        questions = [item['question'] for item in batch]
        answers = [item['answer'] for item in batch]
        formatted_prompts = [item['formatted_prompt'] for item in batch]
        
        # Return raw questions for the pipeline to process
        return {
            'questions': formatted_prompts,
            'raw_questions': questions,
            'answers': answers
        }