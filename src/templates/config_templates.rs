pub const BASE_CONFIG: &str = r#"# Base configuration for {{ project_name }}
project:
  name: {{ project_name }}
  version: 0.1.0
  description: Fine-tuning project for {{ task_type }}

task:
  type: {{ task_type | lower | replace(from=" ", to="_") }}
  num_labels: 2  # Update based on your dataset

paths:
  data_dir: data/
  model_dir: models/
  config_dir: configs/
  scripts_dir: scripts/
  notebooks_dir: notebooks/
"#;

pub const DATASET_CONFIG: &str = r#"# Dataset configuration
dataset:
  name: {{ dataset }}
  max_length: {{ max_length }}
  eval_split: 0.2
  test_split: 0.1
  
  # Preprocessing options
  lowercase: false
  remove_punctuation: false
  remove_stopwords: false
  
  # Data augmentation
  augmentation:
    enabled: false
    techniques:
      - synonym_replacement
      - random_insertion
    augmentation_factor: 0.1
"#;

pub const MODEL_CONFIG: &str = r#"# Model configuration
model:
  name: {{ model_name }}
  num_labels: 2  # Update based on your task
  trust_remote_code: false
  
  # Model architecture overrides
  hidden_dropout_prob: 0.1
  attention_probs_dropout_prob: 0.1
  
{% if use_lora %}
# LoRA configuration
lora:
  r: 8
  alpha: 32
  dropout: 0.1
  target_modules:
    - q_proj
    - v_proj
    - k_proj
    - o_proj
{% endif %}

{% if use_qlora %}
# QLoRA configuration
qlora:
  load_in_4bit: true
  bnb_4bit_compute_dtype: float16
  bnb_4bit_use_double_quant: true
  bnb_4bit_quant_type: nf4
{% endif %}
"#;

pub const TRAINING_CONFIG: &str = r#"# Training configuration
training:
  num_epochs: {{ num_epochs }}
  batch_size: {{ batch_size }}
  eval_batch_size: {{ batch_size }}
  learning_rate: {{ learning_rate }}
  warmup_steps: {{ warmup_steps }}
  weight_decay: 0.01
  
  # Scheduler
  lr_scheduler: cosine
  
  # Optimization
  gradient_accumulation_steps: 1
  gradient_checkpointing: false
  fp16: true
  tf32: true
  
  # Evaluation
  evaluation_strategy: steps
  eval_steps: 500
  logging_steps: 100
  save_steps: 500
  save_total_limit: 3
  
  # Early stopping
  early_stopping_patience: 3
  load_best_model_at_end: true
  metric_for_best_model: eval_loss
  greater_is_better: false
  
  # Seed
  seed: 42
  
# Compute configuration
compute:
  provider: {{ "modal" if compute_backend == "Modal" else "local" }}
  {% if gpu_ids %}
  gpus: [{{ gpu_ids }}]
  {% else %}
  gpus: []
  {% endif %}
  use_ddp: {{ use_ddp | lower }}
  mixed_precision: fp16
  
  {% if modal_gpu_type %}
  # Modal specific
  gpu_type: {{ modal_gpu_type }}
  num_gpus: {{ modal_num_gpus }}
  volume: autosetup-data
  image: autosetup/image:latest
  {% endif %}
"#;

pub const WANDB_CONFIG: &str = r#"# Weights & Biases configuration
wandb:
  project: {{ project_name }}
  entity: null  # Your W&B entity/username
  run_name: {{ project_name }}-run
  tags:
    - {{ task_type | lower | replace(from=" ", to="-") }}
    - {{ model_name | replace(from="/", to="-") }}
  notes: Fine-tuning {{ model_name }} on {{ dataset }}
  
  # Logging options
  log_model: true
  log_metrics: true
  log_gradients: false
  log_predictions: true
  
  # Watch model
  watch: gradients
"#;

pub const REQUIREMENTS: &str = r#"# Core dependencies
torch>=2.0.0
transformers>=4.35.0
datasets>=2.14.0
accelerate>=0.24.0
evaluate>=0.4.0
scikit-learn>=1.3.0
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.65.0

# Configuration
pyyaml>=6.0
python-dotenv>=1.0.0

{% if use_wandb %}
# Experiment tracking
wandb>=0.15.0
{% else %}
# Experiment tracking
tensorboard>=2.13.0
{% endif %}

{% if use_lora or use_qlora %}
# Parameter-efficient fine-tuning
peft>=0.6.0
{% endif %}

{% if use_qlora %}
# Quantization
bitsandbytes>=0.41.0
{% endif %}

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Notebook support
jupyter>=1.0.0
ipywidgets>=8.0.0

# Development tools
black>=23.0.0
isort>=5.12.0
flake8>=6.0.0
pytest>=7.3.0
pre-commit>=3.3.0

{% if compute_backend == "Modal" %}
# Modal cloud compute
modal>=0.56.0
{% endif %}

# Optional: Model optimization
onnx>=1.14.0
onnxruntime>=1.15.0
"#;

pub const README_TEMPLATE: &str = r#"# {{ project_name }}

Fine-tuning project for {{ task_type }} using {{ model_name }} on {{ dataset }} dataset.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- CUDA 11.8+ (for GPU training)
- 8GB+ GPU memory recommended

### Installation

1. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   uv pip install -r requirements.txt
   ```

### Training

1. **Prepare your dataset:**
   ```bash
   python scripts/prepare_dataset.py
   ```

2. **Start training:**
   ```bash
   python scripts/train.py
   ```

3. **Evaluate the model:**
   ```bash
   python scripts/evaluate.py
   ```

### Inference

Run inference on new samples:
```bash
python scripts/inference.py
```

## 📊 Configuration

All configurations are stored in the `configs/` directory:

- `base.yaml`: Project metadata and paths
- `dataset.yaml`: Dataset configuration
- `model.yaml`: Model architecture settings
- `training.yaml`: Training hyperparameters
{% if use_wandb %}- `wandb.yaml`: Weights & Biases logging{% endif %}

## 🏗️ Project Structure

```
{{ project_name }}/
├── configs/           # Configuration files
├── data/             # Dataset storage
├── models/           # Model checkpoints and logs
├── notebooks/        # Jupyter notebooks
├── scripts/          # Training and evaluation scripts
├── src/              # Source code modules
└── requirements.txt  # Python dependencies
```

## ⚙️ Advanced Features

{% if use_lora %}
### LoRA Fine-tuning

This project uses LoRA (Low-Rank Adaptation) for efficient fine-tuning. Configure LoRA parameters in `configs/model.yaml`.
{% endif %}

{% if use_qlora %}
### QLoRA (4-bit Quantization)

QLoRA is enabled for memory-efficient training. The model is loaded in 4-bit precision with double quantization.
{% endif %}

{% if compute_backend == "Modal" %}
### Modal Cloud Training

To run training on Modal:
```bash
modal run scripts/train_modal.py
```
{% endif %}

{% if use_sweep %}
### Hyperparameter Sweeps

Run hyperparameter optimization:
```bash
python scripts/sweep.py
```
{% endif %}

## 📈 Monitoring

{% if use_wandb %}
Track experiments with Weights & Biases:
1. Set your API key: `export WANDB_API_KEY=your_key`
2. View runs at: https://wandb.ai/{{ wandb_entity | default("your-entity") }}/{{ project_name }}
{% else %}
Monitor training with TensorBoard:
```bash
tensorboard --logdir models/logs
```
{% endif %}

## 🤝 Contributing

1. Create a feature branch
2. Make your changes
3. Run tests: `pytest tests/`
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

---

*Generated with [autosetup](https://github.com/Pranav-Karra-3301/autosetup) 🚀*
"#;

pub const INSTRUCTIONS_TEMPLATE: &str = r#"# Detailed Instructions for {{ project_name }}

## 🎯 Project Overview

This project fine-tunes **{{ model_name }}** for **{{ task_type }}** using the **{{ dataset }}** dataset.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **CUDA**: 11.8+ (for GPU training)
- **RAM**: 16GB minimum
- **GPU Memory**: 8GB+ recommended
- **Disk Space**: 20GB+ for models and data

### Software Dependencies
All required packages are listed in `requirements.txt`. Key dependencies include:
- PyTorch 2.0+
- Transformers 4.35+
- Datasets 2.14+
{% if use_wandb %}- Weights & Biases{% endif %}
{% if use_lora %}- PEFT (Parameter-Efficient Fine-Tuning){% endif %}
{% if use_qlora %}- BitsAndBytes (4-bit quantization){% endif %}

## 🚀 Getting Started

### 1. Environment Setup

#### Activate Virtual Environment
```bash
# Linux/Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

#### Verify Installation
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 2. Dataset Preparation

#### Using HuggingFace Datasets
If using a HuggingFace dataset, the preparation script will automatically download and process it:
```bash
python scripts/prepare_dataset.py
```

#### Using Custom Data
Place your data in `data/raw/` with the following structure:
- **CSV format**: `train.csv`, `eval.csv`, `test.csv`
- **JSON format**: `train.json`, `eval.json`, `test.json`

Required columns:
{% if task_type == "Text Classification" %}
- `text`: Input text
- `label`: Class label (integer)
{% elif task_type == "Question Answering" %}
- `question`: Question text
- `context`: Context paragraph
- `answers`: Answer span with start position
{% elif task_type == "Language Modeling" %}
- `text`: Text for language modeling
{% else %}
- Depends on your specific task
{% endif %}

### 3. Configuration

#### Essential Settings

Edit `configs/training.yaml`:
```yaml
training:
  num_epochs: {{ num_epochs }}      # Number of training epochs
  batch_size: {{ batch_size }}       # Batch size per device
  learning_rate: {{ learning_rate }} # Learning rate
```

Edit `configs/model.yaml`:
```yaml
model:
  name: {{ model_name }}
  num_labels: 2  # Update based on your classification task
```

{% if use_lora %}
#### LoRA Configuration

Edit `configs/model.yaml`:
```yaml
lora:
  r: 8              # Rank
  alpha: 32         # Scaling parameter
  dropout: 0.1      # Dropout probability
  target_modules:   # Modules to apply LoRA
    - q_proj
    - v_proj
```
{% endif %}

### 4. Training

#### Basic Training
```bash
python scripts/train.py
```

#### Multi-GPU Training
```bash
torchrun --nproc_per_node=2 scripts/train.py
```

#### Resume from Checkpoint
```bash
python scripts/train.py --resume_from_checkpoint models/checkpoints/checkpoint-500
```

{% if compute_backend == "Modal" %}
#### Cloud Training with Modal
```bash
# First-time setup
modal token new

# Run training
modal run scripts/train_modal.py
```
{% endif %}

### 5. Monitoring

{% if use_wandb %}
#### Weights & Biases

1. Set your API key:
   ```bash
   export WANDB_API_KEY=your_key_here
   ```

2. View runs at: https://wandb.ai/your-entity/{{ project_name }}

3. Key metrics tracked:
   - Training/evaluation loss
   - Learning rate schedule
   - Gradient norms
   - Model predictions
{% else %}
#### TensorBoard

1. Launch TensorBoard:
   ```bash
   tensorboard --logdir models/logs
   ```

2. Open browser: http://localhost:6006

3. Available views:
   - Scalars: Loss, metrics
   - Graphs: Model architecture
   - Histograms: Weight distributions
{% endif %}

### 6. Evaluation

#### Run Evaluation
```bash
python scripts/evaluate.py
```

This will:
- Load the best checkpoint
- Evaluate on test set
- Generate confusion matrix
- Save metrics to `models/evaluation_results.json`

#### Metrics Computed
- Accuracy
- Precision (weighted)
- Recall (weighted)
- F1 Score (weighted)

### 7. Inference

#### Interactive Mode
```bash
python scripts/inference.py
```

#### Batch Inference
```python
from transformers import pipeline

# Load your fine-tuned model
classifier = pipeline(
    "text-classification",
    model="models/final",
    device=0  # GPU device
)

# Predict
texts = ["Example text 1", "Example text 2"]
predictions = classifier(texts)
```

### 8. Model Export

#### Export to ONNX
```bash
python scripts/export_model.py
```

#### Upload to HuggingFace Hub
```bash
huggingface-cli login
python scripts/export_model.py --push_to_hub --hub_model_id your-username/model-name
```

## 🔧 Troubleshooting

### Common Issues

#### CUDA Out of Memory
- Reduce `batch_size` in `configs/training.yaml`
- Enable gradient checkpointing
- Use mixed precision training (fp16)
{% if use_qlora %}- QLoRA is already enabled for memory efficiency{% endif %}

#### Slow Training
- Check GPU utilization: `nvidia-smi`
- Increase batch size if memory allows
- Enable mixed precision: `fp16: true`
- Use multiple GPUs with DDP

#### Poor Performance
- Adjust learning rate (try 1e-5 to 5e-5)
- Increase training epochs
- Check dataset quality and balance
- Try different model architectures

## 📊 Hyperparameter Tuning

{% if use_sweep %}
### Using W&B Sweeps

1. Configure sweep in `configs/sweep.yaml`
2. Run sweep:
   ```bash
   python scripts/sweep.py
   ```
{% else %}
### Manual Tuning

Key hyperparameters to tune:
1. Learning rate: [1e-5, 5e-5]
2. Batch size: [8, 16, 32]
3. Warmup steps: [0, 500, 1000]
4. Weight decay: [0.0, 0.01, 0.1]
{% endif %}

## 🧪 Testing

Run tests:
```bash
pytest tests/ -v
```

Code quality:
```bash
black scripts/ src/
isort scripts/ src/
flake8 scripts/ src/
```

## 📚 Additional Resources

- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [PyTorch Documentation](https://pytorch.org/docs/)
{% if use_wandb %}- [Weights & Biases Guide](https://docs.wandb.ai/){% endif %}
{% if use_lora %}- [PEFT Documentation](https://huggingface.co/docs/peft){% endif %}
{% if compute_backend == "Modal" %}- [Modal Documentation](https://modal.com/docs){% endif %}

## 💡 Tips & Best Practices

1. **Start Simple**: Begin with default hyperparameters
2. **Monitor Metrics**: Watch for overfitting (eval loss increasing)
3. **Save Checkpoints**: Keep best 3 checkpoints
4. **Version Control**: Track config changes in git
5. **Document Results**: Log experiments and findings
6. **Reproducibility**: Always set random seeds

## 🆘 Getting Help

1. Check error messages carefully
2. Review logs in `training.log`
3. Search similar issues on GitHub
4. Consult model documentation
5. Ask in community forums

---

*Generated with [autosetup](https://github.com/Pranav-Karra-3301/autosetup) 🚀*
"#;

pub const ENV_TEMPLATE: &str = r#"# Environment variables for {{ project_name }}

# HuggingFace
{% if hf_token %}HF_TOKEN={{ hf_token }}{% else %}# HF_TOKEN=your_token_here{% endif %}
HF_HOME=~/.cache/huggingface

{% if use_wandb %}
# Weights & Biases
{% if wandb_key %}WANDB_API_KEY={{ wandb_key }}{% else %}# WANDB_API_KEY=your_key_here{% endif %}
WANDB_PROJECT={{ project_name }}
{% endif %}

# CUDA
CUDA_VISIBLE_DEVICES={{ gpu_ids }}

# Python
PYTHONPATH=.
TOKENIZERS_PARALLELISM=false

# Logging
LOG_LEVEL=INFO
"#;

pub const RUN_SCRIPT: &str = r#"#!/bin/bash
# Training script for {{ project_name }}

set -e  # Exit on error

echo "🚀 Starting {{ project_name }} training..."

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "Virtual environment not found. Please run: uv venv && uv pip install -r requirements.txt"
    exit 1
fi

# Load environment variables
if [ -f ".env" ]; then
    export $(cat .env | grep -v '^#' | xargs)
fi

# Prepare dataset
echo "📊 Preparing dataset..."
python scripts/prepare_dataset.py

# Start training
echo "🏋️ Starting training..."
{% if use_ddp and gpu_ids | length > 1 %}
torchrun --nproc_per_node={{ gpu_ids | length }} scripts/train.py
{% else %}
python scripts/train.py
{% endif %}

# Evaluate model
echo "📈 Evaluating model..."
python scripts/evaluate.py

echo "✅ Training completed successfully!"
echo "📁 Model saved to: models/final/"
"#;