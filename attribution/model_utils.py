from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, LogitsProcessorList
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import outlines

class Model:
    
    def __init__(self, model_name: str):
        # Create a text generation pipeline instead of initializing model separately
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        
        # Keep the original model for regex generation
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype='auto',
            device_map='auto',
        )
        
        # Create text generation pipeline with the same model
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        self.device = self.model.device
        
        # Ignore for now
        # Define language-specific patterns
        prefix_patterns = {
            'english': r'Step-by-Step Answer:',
            'french': r'Réponse étape par étape :',
            'german': r'Schritt-für-Schritt-Antwort:',
            'bangla': r'ধাপে ধাপে উত্তর:'
        }

        answer_phrases = {
            'english': r'The answer is',
            'french': r'La réponse est',
            'german': r'Die Antwort lautet',
            'bangla': r'উত্তর হল'
        }

        # Combine into regex pattern
        prefixes = '|'.join(prefix_patterns.values())
        phrases = '|'.join(answer_phrases.values())

        
        output_pattern = f'(?P<prefix>{prefixes})\\s*(?P<calculation>.*?)\\s+(?:{phrases})\\s+(?P<answer>\\d+)(\\.|\。|\\।)?'
        
        outlines_tokenizer = outlines.models.TransformerTokenizer(self.tokenizer)
        
        self.outlines_processor = outlines.processors.RegexLogitsProcessor(
            output_pattern,
            outlines_tokenizer
        )
    
    def generate_responses(self, dataloader, max_new_tokens=1024):
        """
        Generate responses for inputs from a dataloader using the pipeline
        
        Args:
            dataloader: PyTorch DataLoader providing batches
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            List of generated responses as text
        """
        generations = []
        
        for batch in tqdm(dataloader, desc="Generating batches"):
            # Get raw questions from the batch
            questions = batch['questions']
            
            # Use the pipeline for batch generation
            outputs = self.pipeline(
                questions,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                batch_size=len(questions),
                return_full_text=False,  # Only return the newly generated text
                logits_processor=LogitsProcessorList([self.outlines_processor]),
            )
            
            # Extract generated text from pipeline output
            batch_generated_text = [output[0]['generated_text'] for output in outputs]
            
            generations.extend(batch_generated_text)
        
        return generations