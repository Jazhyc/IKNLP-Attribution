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
            # 'french': r'Réponse étape par étape:',
            # 'german': r'Schritt-für-Schritt-Antwort:',
            # 'bangla': r'ধাপে ধাপে উত্তর:'
        }

        answer_phrases = {
            'english': r'The answer is',
            # 'french': r'La réponse est',
            # 'german': r'Die Antwort lautet',
            # 'bangla': r'উত্তর হল'
        }

        # Combine into regex pattern
        prefixes = '|'.join(prefix_patterns.values())
        phrases = '|'.join(answer_phrases.values())

        # Regex pattern that enforces the "Step by Step Answer" format with numbered steps
        self.output_pattern = f'(?i)(?P<prefix>{prefixes})\n(?P<calculation>(?:Step \d{{1,2}}\).*\n)+)(?:{phrases})\s+(?P<answer>.+)'

        # Detailed explanation:
        # (?i) - Case insensitive flag for the entire pattern
        # (?P<prefix>{prefixes}) - Named capture group "prefix" matching one of the prefix patterns (e.g., "Step-by-Step Answer:")
        # \n - Matches a newline character after the prefix
        # (?P<calculation> - Start of named capture group "calculation" to capture all reasoning steps
        #   (?:Step \d{{1,2}}\).*\n)+ - Non-capturing group that matches:
        #     - The word "Step" followed by a space
        #     - \d{{1,2}} - One or two digits (step number, from 1 to 99)
        #     - \) - A closing parenthesis
        #     - .* - Any characters after that (the step's content)
        #     - \n - A newline at the end of each step
        #     - The + makes this match one or more steps
        # ) - End of calculation capture group
        # (?:{phrases}) - Non-capturing group matching one of the answer phrases (e.g., "The answer is")
        # \s+ - One or more whitespace characters
        # (?P<answer>.+) - Named capture group "answer" capturing everything after the answer phrase

        self.outlines_tokenizer = outlines.models.TransformerTokenizer(self.tokenizer)
    
    def generate_responses(self, dataloader, max_new_tokens=512, batch_size=32):
        """
        Generate responses for inputs from a dataloader using the pipeline
        
        Args:
            dataloader: PyTorch DataLoader providing batches
            max_new_tokens: Maximum number of tokens to generate (We might need more for reasoning models)
            
        Returns:
            List of generated responses as text
        """
        generations = []
        
        for batch in tqdm(dataloader, desc="Generating batches"):
            # Get raw questions from the batch
            questions = batch['questions']
            
            # Needs to be defined every batch
            outlines_processor = outlines.processors.RegexLogitsProcessor(
                self.output_pattern,
                self.outlines_tokenizer
            )
            
            # Use the pipeline for batch generation
            outputs = self.pipeline(
                questions,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                batch_size=batch_size,
                return_full_text=False,  # Only return the newly generated text
                logits_processor=LogitsProcessorList([outlines_processor]),
            )
            
            # Extract generated text from pipeline output
            batch_generated_text = [output[0]['generated_text'] for output in outputs]
            
            generations.extend(batch_generated_text)
        
        return generations