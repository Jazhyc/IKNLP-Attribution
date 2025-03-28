from attribution.dataset_utils import GSMDataset, extract_answer_gsm
from attribution.model_utils import Model
from transformers import AutoTokenizer
import pandas as pd
import os

class ResponseProcessor():
    def __init__(self, model_name, dataset_name, config='en', is_cot=True):
        self.df_column_names = ["question", "actual_answer", "model_gen_steps", "model_gen_answer", 'model_answer_str']
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        self.dataset_name = dataset_name
        self.config = config
        self.is_cot = is_cot
    
    def convert_dashes_incremental_steps_list(self, steps):
        furnished_steps = []

        i = 1
        for _, step in enumerate(steps[1:]):
            if step:  # Skip empty parts (if any)
                
                # I removed the full stop because contextcite treats the step number itself as a new sentence
                furnished_steps.append(str(i) + " " + step)  # Replace with number (1, 2, 3...)
                i += 1
        
        return furnished_steps

    def convert_dashes_incremental_steps(self, step):

        '''
        Returns str
        '''

        furnished_steps = self.convert_dashes_incremental_steps_list(step)

        final_str = "Step-by-Step Answer:\n"

        # Added a \n to better separate the steps
        final_str += "\n".join(furnished_steps)

        return final_str


    def process_model_responses_for_analysis(self, data_path):
        
        # Load train for instructions
        mgsm_train = GSMDataset(self.dataset_name, self.tokenizer, split='train', config=self.config)
        
        # Test set for questions
        mgsm_test = GSMDataset(self.dataset_name, self.tokenizer, instructions='', split='test', config=self.config)
        
        mgsm_generation_df = pd.read_csv(data_path)
        mgsm_generations = mgsm_generation_df['response'].tolist()
        
        all_steps = []
        all_gen_final_ans = []
        all_answer_strings = []  # For storing the last line
        
        for response in mgsm_generations:
            # Split response by newlines
            lines = response.strip().split('\n')
            
            # Extract the last line as the answer string
            answer_string = lines[-1]
            all_answer_strings.append(answer_string)
            
            # Use all lines except the last for steps
            remaining_response = '\n'.join(lines[:-1])
            steps = remaining_response.split("\n-")
                
            steps_str = self.convert_dashes_incremental_steps(steps)
            all_steps.append(steps_str)
            
            # Extract numerical answer
            gen_final_ans = extract_answer_gsm(response)
            all_gen_final_ans.append(gen_final_ans)
        
        # Combine each question with mgsm_train.instructions
        instructions = mgsm_train.instructions + '\n\n' if self.is_cot else ''
        
        # Get questions as a list
        questions = mgsm_test.dataset['question']
        
        # Create a list of questions with instructions prepended to each
        question_list = [instructions + q for q in questions]
        
        actual_answer = mgsm_test.dataset['answer_number']
        
        # Create DataFrame with all columns
        percentile_list = pd.DataFrame(
            data=zip(question_list, actual_answer, all_steps, all_gen_final_ans, all_answer_strings), 
            columns=self.df_column_names
        )
        
        # Separate model_path by / and get the last part
        experiment_name = data_path.split('/')[-1]
        
        os.makedirs('results/processed', exist_ok=True)
        percentile_list.to_csv(f'results/processed/{experiment_name}', index=False)

