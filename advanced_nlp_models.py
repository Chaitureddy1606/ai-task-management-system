#!/usr/bin/env python3
"""
Advanced NLP Models with Transformers
- BERT/RoBERTa for task analysis
- Fine-tuning on labeled task data
- Embedding extraction for other models
- Multi-task learning for task classification, urgency detection, etc.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertModel, BertForSequenceClassification,
    RobertaTokenizer, RobertaModel, RobertaForSequenceClassification,
    AutoTokenizer, AutoModel, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import os
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TaskAnalysisConfig:
    """Configuration for task analysis models"""
    model_name: str = "bert-base-uncased"  # or "roberta-base"
    max_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_steps: int = 500
    weight_decay: float = 0.01
    save_steps: int = 1000
    eval_steps: int = 500
    logging_steps: int = 100
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "accuracy"
    greater_is_better: bool = True

class TaskDataset(Dataset):
    """Custom dataset for task analysis"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class MultiTaskTaskAnalysisModel(nn.Module):
    """Multi-task model for task analysis"""
    
    def __init__(self, model_name: str, num_categories: int, num_priorities: int, num_urgency_levels: int):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        
        # Task-specific heads
        self.category_classifier = nn.Linear(self.transformer.config.hidden_size, num_categories)
        self.priority_classifier = nn.Linear(self.transformer.config.hidden_size, num_priorities)
        self.urgency_classifier = nn.Linear(self.transformer.config.hidden_size, num_urgency_levels)
        
        # Shared layers
        self.shared_layer = nn.Linear(self.transformer.config.hidden_size, 256)
        self.activation = nn.ReLU()
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Shared representation
        shared_features = self.activation(self.shared_layer(pooled_output))
        
        # Task-specific predictions
        category_logits = self.category_classifier(shared_features)
        priority_logits = self.priority_classifier(shared_features)
        urgency_logits = self.urgency_classifier(shared_features)
        
        outputs = {
            'category_logits': category_logits,
            'priority_logits': priority_logits,
            'urgency_logits': urgency_logits,
            'shared_features': shared_features
        }
        
        if labels is not None:
            # Calculate losses for each task
            category_loss = nn.CrossEntropyLoss()(category_logits, labels['category'])
            priority_loss = nn.CrossEntropyLoss()(priority_logits, labels['priority'])
            urgency_loss = nn.CrossEntropyLoss()(urgency_logits, labels['urgency'])
            
            # Combined loss
            total_loss = category_loss + priority_loss + urgency_loss
            outputs['loss'] = total_loss
        
        return outputs

class AdvancedTaskAnalyzer:
    """Advanced task analyzer using transformers"""
    
    def __init__(self, config: TaskAnalysisConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = None
        self.model = None
        self.category_mapping = {}
        self.priority_mapping = {}
        self.urgency_mapping = {}
        
        # Initialize tokenizer and model
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the transformer model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.config.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info(f"Model loaded successfully on device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def prepare_training_data(self, data_path: str) -> Tuple[List[str], Dict[str, List[int]]]:
        """Prepare training data from CSV file"""
        try:
            df = pd.read_csv(data_path)
            
            # Combine title and description
            texts = []
            for _, row in df.iterrows():
                title = str(row.get('title', ''))
                description = str(row.get('description', ''))
                combined_text = f"{title}. {description}".strip()
                texts.append(combined_text)
            
            # Create label mappings
            categories = df['category'].unique()
            priorities = df['priority'].unique()
            urgency_levels = df['urgency_score'].unique()
            
            self.category_mapping = {cat: idx for idx, cat in enumerate(categories)}
            self.priority_mapping = {pri: idx for idx, pri in enumerate(priorities)}
            self.urgency_mapping = {urg: idx for idx, urg in enumerate(sorted(urgency_levels))}
            
            # Convert labels to indices
            category_labels = [self.category_mapping[cat] for cat in df['category']]
            priority_labels = [self.priority_mapping[pri] for pri in df['priority']]
            urgency_labels = [self.urgency_mapping[urg] for urg in df['urgency_score']]
            
            labels = {
                'category': category_labels,
                'priority': priority_labels,
                'urgency': urgency_labels
            }
            
            logger.info(f"Prepared {len(texts)} training samples")
            logger.info(f"Categories: {len(categories)}, Priorities: {len(priorities)}, Urgency levels: {len(urgency_levels)}")
            
            return texts, labels
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
    
    def create_datasets(self, texts: List[str], labels: Dict[str, List[int]]) -> Tuple[TaskDataset, TaskDataset]:
        """Create train and validation datasets"""
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = TaskDataset(
            train_texts, 
            train_labels, 
            self.tokenizer, 
            self.config.max_length
        )
        val_dataset = TaskDataset(
            val_texts, 
            val_labels, 
            self.tokenizer, 
            self.config.max_length
        )
        
        return train_dataset, val_dataset
    
    def train_model(self, train_dataset: TaskDataset, val_dataset: TaskDataset):
        """Train the transformer model"""
        try:
            # Initialize model
            num_categories = len(self.category_mapping)
            num_priorities = len(self.priority_mapping)
            num_urgency_levels = len(self.urgency_mapping)
            
            self.model = MultiTaskTaskAnalysisModel(
                self.config.model_name,
                num_categories,
                num_priorities,
                num_urgency_levels
            ).to(self.device)
            
            # Training arguments
            training_args = TrainingArguments(
                output_dir="./models/advanced_nlp",
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                warmup_steps=self.config.warmup_steps,
                weight_decay=self.config.weight_decay,
                save_steps=self.config.save_steps,
                eval_steps=self.config.eval_steps,
                logging_steps=self.config.logging_steps,
                save_total_limit=self.config.save_total_limit,
                load_best_model_at_end=self.config.load_best_model_at_end,
                metric_for_best_model=self.config.metric_for_best_model,
                greater_is_better=self.config.greater_is_better,
                evaluation_strategy="steps",
                save_strategy="steps",
                logging_dir="./logs/advanced_nlp",
                report_to=None,  # Disable wandb
                dataloader_pin_memory=False,
            )
            
            # Custom trainer
            trainer = AdvancedNLPTrainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
            )
            
            # Train model
            logger.info("Starting model training...")
            trainer.train()
            
            # Save model and tokenizer
            trainer.save_model("./models/advanced_nlp_final")
            self.tokenizer.save_pretrained("./models/advanced_nlp_final")
            
            # Save mappings
            mappings = {
                'category_mapping': self.category_mapping,
                'priority_mapping': self.priority_mapping,
                'urgency_mapping': self.urgency_mapping
            }
            
            with open("./models/advanced_nlp_final/mappings.json", 'w') as f:
                json.dump(mappings, f, indent=2)
            
            logger.info("Model training completed successfully!")
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def load_trained_model(self, model_path: str = "./models/advanced_nlp_final"):
        """Load a trained model"""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Load mappings
            with open(f"{model_path}/mappings.json", 'r') as f:
                mappings = json.load(f)
            
            self.category_mapping = mappings['category_mapping']
            self.priority_mapping = mappings['priority_mapping']
            self.urgency_mapping = mappings['urgency_mapping']
            
            # Initialize model
            num_categories = len(self.category_mapping)
            num_priorities = len(self.priority_mapping)
            num_urgency_levels = len(self.urgency_mapping)
            
            self.model = MultiTaskTaskAnalysisModel(
                self.config.model_name,
                num_categories,
                num_priorities,
                num_urgency_levels
            ).to(self.device)
            
            # Load trained weights
            self.model.load_state_dict(torch.load(f"{model_path}/pytorch_model.bin", map_location=self.device))
            self.model.eval()
            
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def analyze_task(self, title: str, description: str) -> Dict:
        """Analyze a task using the trained model"""
        try:
            # Prepare input
            text = f"{title}. {description}".strip()
            
            # Tokenize
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            
            # Move to device
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Get predictions
                category_probs = torch.softmax(outputs['category_logits'], dim=1)
                priority_probs = torch.softmax(outputs['priority_logits'], dim=1)
                urgency_probs = torch.softmax(outputs['urgency_logits'], dim=1)
                
                # Get predicted classes
                category_pred = torch.argmax(category_probs, dim=1).item()
                priority_pred = torch.argmax(priority_probs, dim=1).item()
                urgency_pred = torch.argmax(urgency_probs, dim=1).item()
                
                # Get confidence scores
                category_conf = category_probs[0][category_pred].item()
                priority_conf = priority_probs[0][priority_pred].item()
                urgency_conf = urgency_probs[0][urgency_pred].item()
            
            # Convert back to original labels
            category_label = list(self.category_mapping.keys())[list(self.category_mapping.values()).index(category_pred)]
            priority_label = list(self.priority_mapping.keys())[list(self.priority_mapping.values()).index(priority_pred)]
            urgency_label = list(self.urgency_mapping.keys())[list(self.urgency_mapping.values()).index(urgency_pred)]
            
            # Extract embeddings for other models
            embeddings = outputs['shared_features'].cpu().numpy()
            
            return {
                'category': {
                    'prediction': category_label,
                    'confidence': category_conf,
                    'probabilities': category_probs[0].cpu().numpy().tolist()
                },
                'priority': {
                    'prediction': priority_label,
                    'confidence': priority_conf,
                    'probabilities': priority_probs[0].cpu().numpy().tolist()
                },
                'urgency': {
                    'prediction': urgency_label,
                    'confidence': urgency_conf,
                    'probabilities': urgency_probs[0].cpu().numpy().tolist()
                },
                'embeddings': embeddings[0].tolist(),
                'analysis_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing task: {e}")
            raise
    
    def extract_embeddings(self, title: str, description: str) -> np.ndarray:
        """Extract embeddings for use in other models"""
        try:
            text = f"{title}. {description}".strip()
            
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.config.max_length,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings = outputs['shared_features'].cpu().numpy()
            
            return embeddings[0]
            
        except Exception as e:
            logger.error(f"Error extracting embeddings: {e}")
            raise

class AdvancedNLPTrainer(Trainer):
    """Custom trainer for multi-task learning"""
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation for multi-task learning"""
        outputs = model(**inputs)
        loss = outputs['loss']
        return (loss, outputs) if return_outputs else loss

def main():
    """Main function to demonstrate advanced NLP models"""
    print("🚀 Advanced NLP Models with Transformers")
    print("=" * 50)
    print("✅ BERT/RoBERTa for task analysis")
    print("✅ Fine-tuning on labeled data")
    print("✅ Embedding extraction for other models")
    print("✅ Multi-task learning")
    print("=" * 50)
    
    # Configuration
    config = TaskAnalysisConfig(
        model_name="bert-base-uncased",
        max_length=256,
        batch_size=8,
        num_epochs=2,
        learning_rate=3e-5
    )
    
    # Initialize analyzer
    analyzer = AdvancedTaskAnalyzer(config)
    
    # Check if trained model exists
    model_path = "./models/advanced_nlp_final"
    if os.path.exists(model_path):
        print("📦 Loading pre-trained model...")
        analyzer.load_trained_model(model_path)
    else:
        print("🏋️  Training new model...")
        
        # Prepare training data
        data_path = "data/combined_training_tasks.csv"
        if os.path.exists(data_path):
            texts, labels = analyzer.prepare_training_data(data_path)
            train_dataset, val_dataset = analyzer.create_datasets(texts, labels)
            
            # Train model
            analyzer.train_model(train_dataset, val_dataset)
        else:
            print(f"❌ Training data not found: {data_path}")
            print("💡 Please ensure the training data is available")
            return
    
    # Test task analysis
    test_tasks = [
        {
            "title": "Fix critical security vulnerability in login system",
            "description": "Urgent fix needed for SQL injection vulnerability in user authentication. This affects all users and poses high security risk."
        },
        {
            "title": "Implement new user dashboard with analytics",
            "description": "Create a comprehensive dashboard for users to view their analytics, reports, and account information. This is a new feature request from marketing team."
        },
        {
            "title": "Add comprehensive unit tests for payment module",
            "description": "Create comprehensive unit tests for the payment processing module to ensure reliability and catch potential bugs before production."
        }
    ]
    
    print(f"\n🔍 Testing Task Analysis:")
    for i, task in enumerate(test_tasks, 1):
        print(f"\n📋 Task {i}: {task['title'][:50]}...")
        
        try:
            analysis = analyzer.analyze_task(task['title'], task['description'])
            
            print(f"   🏷️  Category: {analysis['category']['prediction']} (Confidence: {analysis['category']['confidence']:.3f})")
            print(f"   ⚡ Priority: {analysis['priority']['prediction']} (Confidence: {analysis['priority']['confidence']:.3f})")
            print(f"   🚨 Urgency: {analysis['urgency']['prediction']} (Confidence: {analysis['urgency']['confidence']:.3f})")
            print(f"   📊 Embeddings: {len(analysis['embeddings'])} dimensions")
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
    
    # Test embedding extraction
    print(f"\n🔍 Testing Embedding Extraction:")
    test_task = test_tasks[0]
    
    try:
        embeddings = analyzer.extract_embeddings(test_task['title'], test_task['description'])
        print(f"   📊 Embedding shape: {embeddings.shape}")
        print(f"   📊 Embedding sample: {embeddings[:5]}")
        print(f"   ✅ Embeddings extracted successfully")
    except Exception as e:
        print(f"   ❌ Embedding extraction failed: {e}")
    
    print(f"\n🎉 Advanced NLP Models Ready!")
    print("=" * 50)
    print("✅ Transformer-based task analysis")
    print("✅ Multi-task learning implemented")
    print("✅ Embedding extraction available")
    print("✅ Ready for integration with auto-assignment")

if __name__ == "__main__":
    main() 