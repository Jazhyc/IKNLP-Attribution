# What Really Counts? Examining Step and Token Level Attribution in Multilingual Chain-of-Thought Reasoning

This repository contains the code and experiments for the "What Really Counts? Examining Step and Token Level Attribution in Multilingual Chain-of-Thought Reasoning" project conducted as part of the Advanced Topics in NLP (IK-NLP) course 2025 at the University of Groningen. The project explores how multilingual models attribute importance to different reasoning steps when solving mathematical problems across various languages.
This repository contains the code and experiments for the "Attribution in Large Language Models" project conducted as part of the Advanced Topics in NLP (IK-NLP) course 2025 at the University of Groningen. The project explores how multilingual models attribute importance to different reasoning steps when solving mathematical problems across various languages in the [MGSM Dataset](https://huggingface.co/datasets/juletxara/mgsm).

## Overview

The project includes two main experimental frameworks:

- **ContextCite**: Analyzes which reasoning steps are most important for the model's predictions
- **Inseq**: Provides token-level attribution analysis for multilingual model outputs

## Context Cite Experiments

ContextCite evaluates the importance of each reasoning step in a chain-of-thought solution by systematically altering or removing steps and measuring the impact on the model's predictions.

### Installation

```bash
pip install -r requirements.txt
```

Note: You may want to install pytorch with GPU support separately.

### Running the Experiments

To run Context Cite experiments:

1. Modify `main.py` to specify your desired configuration. Options include using structured generation, few shot prompting, different model types, and languages to test.

2. Run the main script which will execute the whole pipeline:

```bash
python main.py
```

By default, the script will skip experiments that have already been executed. Manual deletion of files in `results/` is currently necessary to rerun the desired experiment.

### Analyzing Results

The Context Cite analysis results are stored in `results/contextcite/` and can be visualized using:

- **Notebook:** [`notebooks/analysis.ipynb`](/notebooks/analysis.ipynb)

## Inseq Experiments

The Inseq experiments provide token-level attribution analysis to identify which input tokens contribute most to specific output tokens in the model's prediction.

### Getting Detailed Inseq Attributions  

To reproduce the detailed [inseq](https://github.com/inseq-team/inseq) attributions discussed in the paper, run:  

- **Notebook:** [`notebooks/inseq_attributions_multilingual.ipynb`](/notebooks/inseq_experiments.ipynb)

### Data Containing Filtered Attribution Scores  

For convenience, we provide filtered attribution scores (manually selected from the full outputs):  

- **File:** [`results/inseq_heatmap_data.csv`](/results/inseq_heatmap_data.csv)

### Generating Heat Maps  

To regenerate the heat maps from the paper using the filtered data:  

- **Notebook:** [`notebooks/inseq_heatmaps.ipynb`](/notebooks/inseq_heatmaps.ipynb)
