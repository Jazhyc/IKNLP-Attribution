from constants import DatasetNames
import datasets
import re
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding

class BaseDataset(Dataset):
    def __init__(self, dataset_name, tokenizer, instructions, split='train'):
        self.name = dataset_name
        self.tokenizer = tokenizer
        self.instructions = self._load_system_prompt(dataset_name) if instructions is None else instructions
        self.dataset = self._load_dataset(dataset_name, split)
        self.system_prompt = {'role': 'system', 'content': self.instructions} if self.instructions else None
        
    def _load_dataset(self, dataset_name, split):
        """Load the dataset based on its name"""
        if dataset_name == DatasetNames.GSM8k:
            config = 'main'
        else:
            raise ValueError(f"Dataset {dataset_name} not supported yet")
            
        dataset = datasets.load_dataset(dataset_name, name=config, split=split)
        return dataset
    
    def _load_system_prompt(self, dataset_name):
        prompt = ''
        if dataset_name == DatasetNames.GSM8k:
            prompt = open('attribution/prompts/GSM8k.txt').read()
        return prompt
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        """Get a tokenized example at the given index"""
        raise NotImplementedError("Subclasses must implement this method")
        
    def create_instructions(self, n_samples=5):
        """Create instructions based on the dataset"""
        raise NotImplementedError("Subclasses must implement this method")


class GSM8kDataset(BaseDataset):
    def __init__(self, tokenizer, instructions=None, split='train', n_instruction_samples=5):
        super().__init__(DatasetNames.GSM8k, tokenizer, instructions, split)
        
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
            system_prompt.append(f"Question: {sample['question']}\nAnswer: {sample['answer']}")
            
        # Combine all the system prompts
        return '\n\n'.join(system_prompt)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        question = sample['question']
        answer = sample['answer']
        
        user_prompt = {'role': 'user', 'content': question}
        messages = [self.system_prompt, user_prompt]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True
        )
        
        return {
            'input_ids': torch.tensor(input_ids),
            'question': question,
            'answer': answer
        }


# GSM8k evaluation functions

# Standard format
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
# Any LaTeX command that wraps an answer
LATEX_RE = re.compile(r"\\[a-zA-Z]+\{(\-?[0-9\.\,]+)\}")
# Inline LaTeX expressions
INLINE_LATEX_RE = re.compile(r"\$(\-?[0-9\.\,]+)\$")
INVALID_ANS = "[invalid]"

def extract_answer_gsm8k(completion):
    # First try to match the standard format
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    
    # Then try to match any LaTeX command wrapper
    match = LATEX_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    
    # Try to match inline LaTeX expressions
    match = INLINE_LATEX_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    
    # No valid format found
    return INVALID_ANS


def is_correct_gsm8k(model_completion, gt_example):
    gt_answer = extract_answer_gsm8k(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer_gsm8k(model_completion) == gt_answer


def is_correct_gsm8k(model_completion, gt_example):
    gt_answer = extract_answer_gsm8k(gt_example["answer"])
    assert gt_answer != INVALID_ANS
    return extract_answer_gsm8k(model_completion) == gt_answer

# Helper function to get the appropriate dataset instance
def get_dataset_instance(dataset_name, tokenizer, instructions=None, split='train'):
    if dataset_name == DatasetNames.GSM8k:
        return GSM8kDataset(tokenizer, instructions, split)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported yet")

class PaddingCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.data_collator = DataCollatorWithPadding(tokenizer)
        
    def __call__(self, batch):
        """
        Collate function for DataLoader that pads sequences using the tokenizer's built-in padding.
        """
        questions = [item['question'] for item in batch]
        answers = [item['answer'] for item in batch]
        
        # Extract the features that need padding
        features = [{'input_ids': item['input_ids']} for item in batch]
        
        # Use the HuggingFace collator to pad
        padded_batch = self.data_collator(features)
        
        # Add back the questions and answers
        padded_batch['questions'] = questions
        padded_batch['answers'] = answers
        
        return padded_batch