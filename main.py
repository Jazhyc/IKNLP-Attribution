import itertools
from attribution.experiments import conduct_experiment, clean_generations, run_context_cite
from attribution.constants import LANGUAGE_MAPPING

if __name__ == '__main__':
    # Define configuration options
    languages = list(LANGUAGE_MAPPING.keys())
    cot_options = [False, True]
    
    # Generate all combinations using itertools.product
    all_configs = itertools.product(languages, cot_options)
    
    # Run experiments for each combination
    for language, use_cot in all_configs:
        print(f"Running experiment: language={language}, COT={use_cot}")
        conduct_experiment(config=language, use_COT=use_cot)
    
    print("Cleaning up generations...")
    clean_generations()
    
    # Run context cite on all processed files
    print("Running context cite on processed files...")
    run_context_cite()