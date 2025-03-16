from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

class Model:
    
    def __init__(self, model_name: str):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype='auto',
            device_map='auto',
        )
        
        self.model = torch.compile(self.model)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.device = self.model.device
    
    def generate_responses(self, dataloader, max_new_tokens=1024):
        """
        Generate responses for inputs from a dataloader
        
        Args:
            dataloader: PyTorch DataLoader providing batches
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            List of generated responses as text
        """
        generations = []
        
        for batch in tqdm(dataloader, desc="Generating batches"):
            batch_input_ids = batch['input_ids'].to(self.device)
            batch_attention_mask = batch['attention_mask'].to(self.device) if 'attention_mask' in batch else None
            
            with torch.inference_mode():
                # Generate all outputs for this batch
                batch_output_ids = self.model.generate(
                    batch_input_ids,
                    attention_mask=batch_attention_mask,
                    max_new_tokens=max_new_tokens,
                )
 
            
            # Batch decode all generated parts at once
            batch_generated_text = self.tokenizer.batch_decode(batch_output_ids[:, batch_input_ids.shape[1]:], skip_special_tokens=True)
            generations.extend(batch_generated_text)
        
        return generations