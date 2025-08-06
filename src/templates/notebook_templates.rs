// Simplified notebook templates that avoid complex JSON escaping
pub const EXPLORATION_NOTEBOOK: &str = r#"# %%
"""
# Data Exploration for {{ project_name }}
This notebook explores the {{ dataset }} dataset for {{ task_type }} task.
"""

# %%
import sys
import os
sys.path.append('..')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('ggplot')
sns.set_palette('husl')

# %%
# Load dataset
dataset_name = '{{ dataset }}'

if dataset_name.startswith('/') or dataset_name.startswith('./'):
    dataset = load_dataset('csv', data_files=dataset_name)
else:
    dataset = load_dataset(dataset_name)

print(f"Dataset: {dataset}")
print(f"\nColumns: {dataset['train'].column_names if 'train' in dataset else 'N/A'}")
print(f"\nNumber of examples:")
for split in dataset:
    print(f"  {split}: {len(dataset[split])}")

# %%
# Convert to pandas for analysis
if 'train' in dataset:
    df_train = dataset['train'].to_pandas()
    print("Dataset shape:", df_train.shape)
    print("\nFirst 5 examples:")
    display(df_train.head())

# %%
# Basic statistics
print("Dataset Info:")
df_train.info()
print("\nMissing values:")
print(df_train.isnull().sum())

# %%
# Text analysis
text_col = 'text' if 'text' in df_train.columns else df_train.columns[0]

df_train['text_length'] = df_train[text_col].str.len()
df_train['word_count'] = df_train[text_col].str.split().str.len()

print(f"Text length statistics:")
print(df_train['text_length'].describe())
print(f"\nWord count statistics:")
print(df_train['word_count'].describe())

# %%
# Visualize text lengths
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].hist(df_train['text_length'], bins=50, edgecolor='black')
axes[0].set_xlabel('Text Length (characters)')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Text Lengths')

axes[1].hist(df_train['word_count'], bins=50, edgecolor='black')
axes[1].set_xlabel('Word Count')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Word Counts')

plt.tight_layout()
plt.show()

# %%
# Label distribution (for classification tasks)
if 'label' in df_train.columns:
    label_counts = df_train['label'].value_counts()
    
    plt.figure(figsize=(10, 6))
    label_counts.plot(kind='bar')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.title('Label Distribution')
    plt.xticks(rotation=45)
    plt.show()
    
    print("Label distribution:")
    print(label_counts)
    print(f"\nClass balance: {label_counts.min() / label_counts.max():.2%}")

# %%
# Tokenization analysis
tokenizer = AutoTokenizer.from_pretrained('{{ model_name }}')

sample_texts = df_train[text_col].sample(min(1000, len(df_train))).tolist()
token_lengths = []

for text in sample_texts:
    tokens = tokenizer(text, truncation=False)['input_ids']
    token_lengths.append(len(tokens))

token_lengths = np.array(token_lengths)

# %%
# Token length statistics
print(f"Token length statistics (sample of {len(sample_texts)}:")
print(f"  Mean: {token_lengths.mean():.1f}")
print(f"  Median: {np.median(token_lengths):.1f}")
print(f"  Max: {token_lengths.max()}")
print(f"  95th percentile: {np.percentile(token_lengths, 95):.1f}")
print(f"  99th percentile: {np.percentile(token_lengths, 99):.1f}")

plt.figure(figsize=(10, 6))
plt.hist(token_lengths, bins=50, edgecolor='black')
plt.axvline({{ max_length }}, color='r', linestyle='--', label=f'Max length: {{ max_length }}')
plt.xlabel('Token Count')
plt.ylabel('Frequency')
plt.title('Distribution of Token Counts')
plt.legend()
plt.show()

truncation_rate = (token_lengths > {{ max_length }}).mean()
print(f"\nTruncation rate at max_length={{ max_length }}: {truncation_rate:.1%}")
"#;

pub const RESULTS_NOTEBOOK: &str = r#"# %%
"""
# Training Results Analysis for {{ project_name }}
This notebook analyzes the training results and model performance.
"""

# %%
import sys
import os
sys.path.append('..')

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
{% if use_wandb %}import wandb
from wandb import Api{% endif %}

plt.style.use('ggplot')
sns.set_palette('husl')

# %%
# Load evaluation results
eval_results_path = Path('../models/evaluation_results.json')

if eval_results_path.exists():
    with open(eval_results_path, 'r') as f:
        eval_results = json.load(f)
    
    print("Evaluation Results:")
    for metric, value in eval_results.items():
        print(f"  {metric}: {value:.4f}")
    
    # Create a bar plot
    plt.figure(figsize=(10, 6))
    metrics = list(eval_results.keys())
    values = list(eval_results.values())
    
    bars = plt.bar(metrics, values)
    plt.ylabel('Score')
    plt.title('Model Evaluation Metrics')
    plt.ylim(0, 1)
    
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.show()
else:
    print("No evaluation results found. Run: python scripts/evaluate.py")

# %%
# Load and display confusion matrix
from IPython.display import Image, display

cm_path = Path('../models/confusion_matrix.png')
if cm_path.exists():
    display(Image(cm_path))
else:
    print("No confusion matrix found. Run evaluation first.")

# %%
# Load the fine-tuned model
model_path = Path('../models/final')

if model_path.exists():
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {{ model_name }}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage trainable: {trainable_params/total_params*100:.2f}%")
    
    {% if use_lora %}
    print("\nLoRA is enabled - only adapter weights were trained")
    {% endif %}
else:
    print("Model not found. Please complete training first.")

# %%
# Test the model with example inputs
if 'model' in locals():
    from transformers import pipeline
    
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    
    test_texts = [
        "This is a positive example.",
        "This is a negative example.",
        "This is a neutral statement."
    ]
    
    print("Model Predictions:")
    for text in test_texts:
        result = classifier(text)
        print(f"\nText: {text}")
        print(f"Prediction: {result}")

# %%
"""
## Summary and Next Steps

### Summary
- Model: {{ model_name }}
- Task: {{ task_type }}
- Dataset: {{ dataset }}
- Training epochs: {{ num_epochs }}
- Batch size: {{ batch_size }}
- Learning rate: {{ learning_rate }}

### Next Steps
1. **Hyperparameter tuning**: Try different learning rates, batch sizes
2. **Data augmentation**: Increase training data diversity
3. **Model architecture**: Try different base models
4. **Deployment**: Export model for production use
"""
"#;