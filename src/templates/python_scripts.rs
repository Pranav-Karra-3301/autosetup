pub const TRAIN_SCRIPT: &str = r#"#!/usr/bin/env python3
"""
Training script for {{ project_name }}
Task: {{ task_type }}
Model: {{ model_name }}
"""

import os
import sys
import yaml
import torch
import logging
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from datasets import load_dataset, load_from_disk
{% if use_lora %}from peft import LoraConfig, TaskType, get_peft_model{% endif %}
{% if use_qlora %}from transformers import BitsAndBytesConfig{% endif %}
{% if use_wandb %}import wandb{% endif %}

sys.path.append(str(Path(__file__).parent.parent))
from src.dataset.dataset import load_and_prepare_dataset
from src.models.model import load_model_and_tokenizer
from src.training.trainer import CustomTrainer
from src.utils.utils import setup_logging, load_config, set_seed

def main():
    config = load_config()
    setup_logging()
    set_seed(config['training']['seed'])
    
    {% if use_wandb %}
    wandb.init(
        project=config['wandb']['project'],
        entity=config['wandb']['entity'],
        config=config,
        name=config['wandb']['run_name'],
    )
    {% endif %}
    
    logging.info(f"Loading model: {config['model']['name']}")
    model, tokenizer = load_model_and_tokenizer(config)
    
    {% if use_lora %}
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=config['lora']['r'],
        lora_alpha=config['lora']['alpha'],
        lora_dropout=config['lora']['dropout'],
        target_modules=config['lora']['target_modules'],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    {% endif %}
    
    logging.info("Loading dataset...")
    train_dataset, eval_dataset = load_and_prepare_dataset(config, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=config['paths']['output_dir'],
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['eval_batch_size'],
        warmup_steps=config['training']['warmup_steps'],
        weight_decay=config['training']['weight_decay'],
        logging_dir=config['paths']['logging_dir'],
        logging_steps=config['training']['logging_steps'],
        evaluation_strategy="steps",
        eval_steps=config['training']['eval_steps'],
        save_strategy="steps",
        save_steps=config['training']['save_steps'],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        {% if use_wandb %}report_to="wandb",{% else %}report_to="tensorboard",{% endif %}
        {% if use_ddp %}ddp_find_unused_parameters=False,{% endif %}
        fp16=config['training'].get('fp16', True),
        gradient_checkpointing=config['training'].get('gradient_checkpointing', False),
        gradient_accumulation_steps=config['training'].get('gradient_accumulation_steps', 1),
        learning_rate=config['training']['learning_rate'],
        lr_scheduler_type=config['training']['lr_scheduler'],
        seed=config['training']['seed'],
    )
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=config['training'].get('early_stopping_patience', 3)
        )
    ]
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    logging.info("Starting training...")
    trainer.train()
    
    logging.info("Saving final model...")
    trainer.save_model(config['paths']['final_model_dir'])
    tokenizer.save_pretrained(config['paths']['final_model_dir'])
    
    {% if use_wandb %}wandb.finish(){% endif %}
    
    logging.info("Training completed!")

if __name__ == "__main__":
    main()
"#;

pub const EVALUATE_SCRIPT: &str = r#"#!/usr/bin/env python3
"""
Evaluation script for {{ project_name }}
"""

import sys
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.utils import setup_logging, load_config
from src.eval.evaluator import Evaluator

def main():
    config = load_config()
    setup_logging()
    
    logging.info("Loading model for evaluation...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config['paths']['final_model_dir']
    )
    tokenizer = AutoTokenizer.from_pretrained(config['paths']['final_model_dir'])
    
    logging.info("Loading test dataset...")
    test_dataset = load_from_disk(config['paths']['test_dataset'])
    
    evaluator = Evaluator(model, tokenizer, config)
    results = evaluator.evaluate(test_dataset)
    
    logging.info("Evaluation Results:")
    for metric, value in results.items():
        logging.info(f"  {metric}: {value:.4f}")
    
    evaluator.save_results(results, config['paths']['evaluation_results'])
    evaluator.plot_confusion_matrix(test_dataset)
    
    logging.info("Evaluation completed!")

if __name__ == "__main__":
    main()
"#;

pub const PREPARE_DATASET_SCRIPT: &str = r#"#!/usr/bin/env python3
"""
Dataset preparation script for {{ project_name }}
"""

import sys
import logging
from pathlib import Path
from datasets import load_dataset, DatasetDict

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.utils import setup_logging, load_config
from src.dataset.dataset import DatasetProcessor

def main():
    config = load_config()
    setup_logging()
    
    logging.info(f"Loading dataset: {config['dataset']['name']}")
    
    if config['dataset']['name'].startswith('/') or config['dataset']['name'].startswith('./'):
        dataset = load_dataset('csv', data_files=config['dataset']['name'])
    else:
        dataset = load_dataset(config['dataset']['name'])
    
    processor = DatasetProcessor(config)
    
    logging.info("Processing dataset...")
    processed_dataset = processor.process(dataset)
    
    logging.info("Splitting dataset...")
    if isinstance(processed_dataset, DatasetDict):
        train_dataset = processed_dataset['train']
        eval_dataset = processed_dataset.get('validation', processed_dataset.get('test'))
    else:
        split_dataset = processed_dataset.train_test_split(
            test_size=config['dataset']['eval_split'],
            seed=config['training']['seed']
        )
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']
    
    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Eval samples: {len(eval_dataset)}")
    
    logging.info("Saving processed datasets...")
    train_dataset.save_to_disk(config['paths']['train_dataset'])
    eval_dataset.save_to_disk(config['paths']['eval_dataset'])
    
    logging.info("Dataset preparation completed!")

if __name__ == "__main__":
    main()
"#;

pub const INFERENCE_SCRIPT: &str = r#"#!/usr/bin/env python3
"""
Inference script for {{ project_name }}
"""

import sys
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.utils import setup_logging, load_config

def predict(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=-1)
    
    return predicted_class.item(), predictions[0].tolist()

def main():
    config = load_config()
    setup_logging()
    
    logging.info("Loading model for inference...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config['paths']['final_model_dir']
    )
    tokenizer = AutoTokenizer.from_pretrained(config['paths']['final_model_dir'])
    
    {% if task_type == "Text Classification" %}
    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    {% elif task_type == "Question Answering" %}
    qa_pipeline = pipeline(
        "question-answering",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )
    {% endif %}
    
    while True:
        text = input("\nEnter text (or 'quit' to exit): ")
        if text.lower() == 'quit':
            break
        
        {% if task_type == "Text Classification" %}
        result = classifier(text)
        print(f"Prediction: {result}")
        {% elif task_type == "Question Answering" %}
        context = input("Enter context: ")
        result = qa_pipeline(question=text, context=context)
        print(f"Answer: {result}")
        {% else %}
        predicted_class, probabilities = predict(text, model, tokenizer)
        print(f"Predicted class: {predicted_class}")
        print(f"Probabilities: {probabilities}")
        {% endif %}

if __name__ == "__main__":
    main()
"#;

pub const EXPORT_MODEL_SCRIPT: &str = r#"#!/usr/bin/env python3
"""
Model export script for {{ project_name }}
"""

import sys
import logging
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.append(str(Path(__file__).parent.parent))
from src.utils.utils import setup_logging, load_config

def export_to_onnx(model, tokenizer, output_path):
    dummy_input = tokenizer(
        "This is a dummy input for export",
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    torch.onnx.export(
        model,
        tuple(dummy_input.values()),
        output_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence'},
            'attention_mask': {0: 'batch_size', 1: 'sequence'},
            'logits': {0: 'batch_size'}
        },
        opset_version=11
    )

def main():
    config = load_config()
    setup_logging()
    
    logging.info("Loading model for export...")
    model = AutoModelForSequenceClassification.from_pretrained(
        config['paths']['final_model_dir']
    )
    tokenizer = AutoTokenizer.from_pretrained(config['paths']['final_model_dir'])
    
    export_dir = Path(config['paths']['export_dir'])
    export_dir.mkdir(parents=True, exist_ok=True)
    
    logging.info("Exporting to ONNX...")
    onnx_path = export_dir / "model.onnx"
    export_to_onnx(model, tokenizer, str(onnx_path))
    
    logging.info("Saving model for HuggingFace Hub...")
    hub_path = export_dir / "hub"
    model.save_pretrained(hub_path)
    tokenizer.save_pretrained(hub_path)
    
    logging.info(f"Model exported to {export_dir}")

if __name__ == "__main__":
    main()
"#;

pub const SWEEP_SCRIPT: &str = r#"#!/usr/bin/env python3
"""
Hyperparameter sweep script for {{ project_name }}
"""

import sys
import yaml
import wandb
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

def train_sweep():
    wandb.init()
    config = wandb.config
    
    import subprocess
    subprocess.run([
        "python", "scripts/train.py",
        "--learning_rate", str(config.learning_rate),
        "--batch_size", str(config.batch_size),
        "--warmup_steps", str(config.warmup_steps),
        "--weight_decay", str(config.weight_decay),
    ])

def main():
    with open("configs/sweep.yaml", "r") as f:
        sweep_config = yaml.safe_load(f)
    
    sweep_id = wandb.sweep(sweep_config, project="{{ project_name }}-sweep")
    wandb.agent(sweep_id, train_sweep, count=10)

if __name__ == "__main__":
    main()
"#;

pub const DATASET_MODULE: &str = r#""""
Dataset module for {{ project_name }}
"""

import logging
from typing import Dict, Any, Optional
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizer

class DatasetProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.max_length = config['dataset']['max_length']
        self.task_type = config['task']['type']
    
    def process(self, dataset: Dataset) -> Dataset:
        """Process raw dataset based on task type"""
        if self.task_type == "text_classification":
            return self._process_classification(dataset)
        elif self.task_type == "language_modeling":
            return self._process_lm(dataset)
        elif self.task_type == "question_answering":
            return self._process_qa(dataset)
        else:
            return dataset
    
    def _process_classification(self, dataset: Dataset) -> Dataset:
        """Process dataset for text classification"""
        def preprocess(examples):
            return {
                'text': examples.get('text', examples.get('sentence', examples.get('review', ''))),
                'label': examples.get('label', examples.get('sentiment', 0))
            }
        
        return dataset.map(preprocess, batched=True)
    
    def _process_lm(self, dataset: Dataset) -> Dataset:
        """Process dataset for language modeling"""
        def preprocess(examples):
            return {
                'text': examples.get('text', examples.get('content', ''))
            }
        
        return dataset.map(preprocess, batched=True, remove_columns=dataset.column_names)
    
    def _process_qa(self, dataset: Dataset) -> Dataset:
        """Process dataset for question answering"""
        def preprocess(examples):
            return {
                'question': examples.get('question', ''),
                'context': examples.get('context', ''),
                'answers': examples.get('answers', {'text': [''], 'answer_start': [0]})
            }
        
        return dataset.map(preprocess, batched=True)

def load_and_prepare_dataset(config: Dict[str, Any], tokenizer: PreTrainedTokenizer):
    """Load and tokenize dataset"""
    from datasets import load_from_disk
    
    train_dataset = load_from_disk(config['paths']['train_dataset'])
    eval_dataset = load_from_disk(config['paths']['eval_dataset'])
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=config['dataset']['max_length']
        )
    
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = eval_dataset.map(tokenize_function, batched=True)
    
    return train_dataset, eval_dataset
"#;

pub const MODEL_MODULE: &str = r#""""
Model module for {{ project_name }}
"""

import logging
from typing import Dict, Any, Tuple
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    PreTrainedModel,
    PreTrainedTokenizer
)
{% if use_qlora %}from transformers import BitsAndBytesConfig{% endif %}

def load_model_and_tokenizer(config: Dict[str, Any]) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """Load model and tokenizer based on task type"""
    
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        use_fast=True,
        trust_remote_code=config['model'].get('trust_remote_code', False)
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model_kwargs = {
        'pretrained_model_name_or_path': config['model']['name'],
        'trust_remote_code': config['model'].get('trust_remote_code', False),
    }
    
    {% if use_qlora %}
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    model_kwargs['quantization_config'] = quantization_config
    model_kwargs['device_map'] = 'auto'
    {% endif %}
    
    task_type = config['task']['type']
    
    if task_type == "text_classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            **model_kwargs,
            num_labels=config['model']['num_labels']
        )
    elif task_type == "language_modeling":
        model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    elif task_type == "question_answering":
        model = AutoModelForQuestionAnswering.from_pretrained(**model_kwargs)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            **model_kwargs,
            num_labels=2
        )
    
    return model, tokenizer
"#;

pub const TRAINER_MODULE: &str = r#""""
Custom trainer module for {{ project_name }}
"""

import logging
from typing import Dict, Any, Optional
import torch
from transformers import Trainer
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class CustomTrainer(Trainer):
    """Custom trainer with additional metrics and callbacks"""
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def log(self, logs: Dict[str, float]) -> None:
        """Custom logging"""
        super().log(logs)
        
        if self.state.global_step % 100 == 0:
            logging.info(f"Step {self.state.global_step}: {logs}")
"#;

pub const UTILS_MODULE: &str = r#""""
Utility functions for {{ project_name }}
"""

import os
import yaml
import json
import random
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch

def setup_logging(level: str = "INFO") -> None:
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('training.log'),
            logging.StreamHandler()
        ]
    )

def load_config() -> Dict[str, Any]:
    """Load configuration from YAML files"""
    config = {}
    config_dir = Path("configs")
    
    for config_file in config_dir.glob("*.yaml"):
        with open(config_file, 'r') as f:
            section = config_file.stem
            config[section] = yaml.safe_load(f)
    
    config['paths'] = {
        'train_dataset': 'data/processed/train',
        'eval_dataset': 'data/processed/eval',
        'test_dataset': 'data/processed/test',
        'output_dir': 'models/checkpoints',
        'final_model_dir': 'models/final',
        'logging_dir': 'models/logs',
        'export_dir': 'models/export',
        'evaluation_results': 'models/evaluation_results.json'
    }
    
    return config

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_json(data: Dict[str, Any], filepath: str) -> None:
    """Save dictionary to JSON file"""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath: str) -> Dict[str, Any]:
    """Load dictionary from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)
"#;

pub const EVALUATOR_MODULE: &str = r#""""
Evaluator module for {{ project_name }}
"""

import logging
from typing import Dict, Any, List
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluator:
    """Model evaluator with comprehensive metrics"""
    
    def __init__(self, model, tokenizer, config: Dict[str, Any]):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(self, dataset) -> Dict[str, float]:
        """Evaluate model on dataset"""
        all_predictions = []
        all_labels = []
        
        for batch in self._create_dataloader(dataset):
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        return self._compute_metrics(all_predictions, all_labels)
    
    def _create_dataloader(self, dataset):
        """Create dataloader for evaluation"""
        from torch.utils.data import DataLoader
        
        dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        return DataLoader(
            dataset,
            batch_size=self.config['training']['eval_batch_size'],
            shuffle=False
        )
    
    def _compute_metrics(self, predictions: List, labels: List) -> Dict[str, float]:
        """Compute evaluation metrics"""
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_results(self, results: Dict[str, float], filepath: str) -> None:
        """Save evaluation results"""
        import json
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
    
    def plot_confusion_matrix(self, dataset) -> None:
        """Plot and save confusion matrix"""
        predictions = []
        labels = []
        
        for batch in self._create_dataloader(dataset):
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
            batch_labels = batch['labels'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_predictions = torch.argmax(outputs.logits, dim=-1)
            
            predictions.extend(batch_predictions.cpu().numpy())
            labels.extend(batch_labels.cpu().numpy())
        
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('models/confusion_matrix.png')
        plt.close()
"#;