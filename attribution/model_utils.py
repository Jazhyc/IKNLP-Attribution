from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, LogitsProcessorList, BitsAndBytesConfig
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
            # Uncomment on resource constrained machines, leads to slower inference
            # quantization_config=BitsAndBytesConfig(load_in_8bit=True)
        )
        
        # Create text generation pipeline with the same model
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
        )
        
        self.device = self.model.device

        # Detailed explanation (May be outdated):
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
        
    def format_regex(self, config, use_COT=True):
        
        prefix_patterns = {
            'en': r'Step-by-Step Answer:',
            'fr': r'Réponse étape par étape :', # There is a deliberate space here due to French's nature
            'de': r'Schritt-für-Schritt-Antwort:',
            'bn': r'ধাপে ধাপে উত্তর:',
            'zh': r'问题：'
        }

        answer_phrases = {
            'en': r'The answer is',
            'fr': r'La réponse est',
            'de': r'Die Antwort lautet',
            'bn': r'উত্তর হল',
            'zh': r'答案是'
        }
        
        # Regex pattern that enforces the "Step by Step Answer" format with numbered steps
        output_pattern_COT = f'(?P<prefix>{prefix_patterns[config]})\n(?P<calculation>(-[^\n]+[\.।。]\n){{1,8}})(?:{answer_phrases[config]})\s+(?P<answer>\d+)[\.।]<|endoftext|>'
        
        # Only get the answer
        output_pattern_regular = f'(?:{answer_phrases[config]})\s+(?P<answer>\d+)[\.।。]<|endoftext|>'
        
        return output_pattern_COT if use_COT else output_pattern_regular
    
    def generate_responses(self, dataloader, config='en', max_new_tokens=256, batch_size=32, constrained_decoding=True, use_COT=True):
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
            outlines_processor = LogitsProcessorList([outlines.processors.RegexLogitsProcessor(
                self.format_regex(config, use_COT),
                self.outlines_tokenizer
            )]) if constrained_decoding else None
            
            # Use the pipeline for batch generation
            outputs = self.pipeline(
                questions,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                batch_size=batch_size,
                return_full_text=False,  # Only return the newly generated text
                logits_processor=outlines_processor,
            )
            
            # Extract generated text from pipeline output
            batch_generated_text = [output[0]['generated_text'] for output in outputs]
            
            generations.extend(batch_generated_text)
        
        return generations