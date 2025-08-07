"""Fine-tuning utilities for transformer models.

Based on OpenAI Cookbook: https://cookbook.openai.com/articles/gpt-oss/fine-tune-transformers
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback,
    TrainerCallback,
)
from datasets import Dataset, load_dataset
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
import wandb

logger = logging.getLogger(__name__)


@dataclass
class FineTuneConfig:
    """Configuration for fine-tuning."""
    
    # Model settings
    model_id: str
    output_dir: str = "./fine_tuned_model"
    
    # Training hyperparameters
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    warmup_steps: int = 500
    weight_decay: float = 0.01
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    
    # Optimization settings
    fp16: bool = True
    gradient_checkpointing: bool = True
    optim: str = "adamw_torch"
    max_grad_norm: float = 0.3
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    
    # Data settings
    max_seq_length: int = 2048
    train_data_path: Optional[str] = None
    eval_data_path: Optional[str] = None
    dataset_name: Optional[str] = None
    
    # Tracking
    use_wandb: bool = False
    wandb_project: str = "fine-tuning"
    wandb_run_name: Optional[str] = None
    
    # Advanced settings
    use_flash_attention: bool = False
    use_8bit: bool = False
    use_4bit: bool = False
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"


class FineTuner:
    """Fine-tuning manager for transformer models."""
    
    def __init__(self, config: FineTuneConfig):
        """Initialize fine-tuner.
        
        Args:
            config: Fine-tuning configuration
        """
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # Setup tracking
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=config.__dict__
            )
    
    def load_model_and_tokenizer(self):
        """Load model and tokenizer with appropriate settings."""
        logger.info(f"Loading model: {self.config.model_id}")
        
        # Quantization config
        quantization_config = None
        if self.config.use_4bit or self.config.use_8bit:
            from transformers import BitsAndBytesConfig
            
            if self.config.use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                    bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                    bnb_4bit_use_double_quant=True,
                )
            else:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load model
        model_kwargs = {
            "pretrained_model_name_or_path": self.config.model_id,
            "quantization_config": quantization_config,
            "torch_dtype": torch.float16 if self.config.fp16 else torch.float32,
            "device_map": "auto",
        }
        
        if self.config.use_flash_attention:
            model_kwargs["use_flash_attention_2"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
        
        # Prepare for training
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        if quantization_config:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Apply LoRA if configured
        if self.config.use_lora:
            self._apply_lora()
        
        logger.info("Model and tokenizer loaded successfully")
    
    def _apply_lora(self):
        """Apply LoRA configuration to the model."""
        logger.info("Applying LoRA configuration")
        
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def load_dataset(self) -> tuple[Dataset, Optional[Dataset]]:
        """Load training and evaluation datasets.
        
        Returns:
            Tuple of (train_dataset, eval_dataset)
        """
        train_dataset = None
        eval_dataset = None
        
        if self.config.dataset_name:
            # Load from HuggingFace datasets
            logger.info(f"Loading dataset: {self.config.dataset_name}")
            dataset = load_dataset(self.config.dataset_name)
            train_dataset = dataset["train"]
            eval_dataset = dataset.get("validation") or dataset.get("test")
        
        elif self.config.train_data_path:
            # Load from local files
            logger.info(f"Loading training data from: {self.config.train_data_path}")
            train_dataset = self._load_local_dataset(self.config.train_data_path)
            
            if self.config.eval_data_path:
                logger.info(f"Loading evaluation data from: {self.config.eval_data_path}")
                eval_dataset = self._load_local_dataset(self.config.eval_data_path)
        
        else:
            raise ValueError("No dataset specified. Set either dataset_name or train_data_path")
        
        # Preprocess datasets
        train_dataset = self._preprocess_dataset(train_dataset)
        if eval_dataset:
            eval_dataset = self._preprocess_dataset(eval_dataset)
        
        return train_dataset, eval_dataset
    
    def _load_local_dataset(self, path: str) -> Dataset:
        """Load dataset from local file.
        
        Args:
            path: Path to dataset file
            
        Returns:
            Loaded dataset
        """
        file_path = Path(path)
        
        if file_path.suffix == ".jsonl":
            return Dataset.from_json(str(file_path))
        elif file_path.suffix == ".csv":
            return Dataset.from_csv(str(file_path))
        elif file_path.suffix == ".txt":
            with open(file_path, "r") as f:
                texts = f.readlines()
            return Dataset.from_dict({"text": texts})
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
    
    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Preprocess dataset for training.
        
        Args:
            dataset: Raw dataset
            
        Returns:
            Preprocessed dataset
        """
        def tokenize_function(examples):
            # Handle different input formats
            if "text" in examples:
                texts = examples["text"]
            elif "prompt" in examples and "completion" in examples:
                texts = [
                    f"{prompt}{completion}"
                    for prompt, completion in zip(examples["prompt"], examples["completion"])
                ]
            elif "instruction" in examples and "output" in examples:
                texts = []
                for i in range(len(examples["instruction"])):
                    instruction = examples["instruction"][i]
                    output = examples["output"][i]
                    input_text = examples.get("input", [""] * len(examples["instruction"]))[i]
                    
                    if input_text:
                        text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
                    else:
                        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
                    texts.append(text)
            else:
                raise ValueError("Dataset must contain 'text', 'prompt'/'completion', or 'instruction'/'output' columns")
            
            return self.tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=self.config.max_seq_length,
            )
        
        # Tokenize dataset
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
        )
        
        return tokenized_dataset
    
    def create_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None
    ) -> Trainer:
        """Create trainer instance.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Optional evaluation dataset
            
        Returns:
            Configured trainer
        """
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            warmup_steps=self.config.warmup_steps,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps,
            save_total_limit=self.config.save_total_limit,
            load_best_model_at_end=self.config.load_best_model_at_end,
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            optim=self.config.optim,
            learning_rate=self.config.learning_rate,
            max_grad_norm=self.config.max_grad_norm,
            evaluation_strategy="steps" if eval_dataset else "no",
            report_to=["wandb"] if self.config.use_wandb else [],
            push_to_hub=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Callbacks
        callbacks = []
        if eval_dataset and self.config.load_best_model_at_end:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=3))
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            callbacks=callbacks,
        )
        
        return self.trainer
    
    def train(self):
        """Run fine-tuning."""
        if not self.model or not self.tokenizer:
            self.load_model_and_tokenizer()
        
        # Load datasets
        train_dataset, eval_dataset = self.load_dataset()
        logger.info(f"Train dataset size: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Eval dataset size: {len(eval_dataset)}")
        
        # Create trainer
        self.create_trainer(train_dataset, eval_dataset)
        
        # Train
        logger.info("Starting fine-tuning...")
        train_result = self.trainer.train()
        
        # Save model
        logger.info(f"Saving model to {self.config.output_dir}")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.config.output_dir)
        
        # Save training results
        with open(Path(self.config.output_dir) / "train_results.txt", "w") as f:
            f.write(str(train_result))
        
        logger.info("Fine-tuning completed!")
        
        return train_result
    
    def evaluate(self, dataset: Optional[Dataset] = None) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            dataset: Optional dataset to evaluate on
            
        Returns:
            Evaluation metrics
        """
        if not self.trainer:
            raise ValueError("Trainer not initialized. Run train() first.")
        
        if dataset is None:
            _, dataset = self.load_dataset()
        
        if dataset is None:
            raise ValueError("No evaluation dataset available")
        
        logger.info("Running evaluation...")
        metrics = self.trainer.evaluate(eval_dataset=dataset)
        
        return metrics
    
    def push_to_hub(
        self,
        repo_name: str,
        private: bool = False,
        use_auth_token: Optional[str] = None
    ):
        """Push fine-tuned model to HuggingFace Hub.
        
        Args:
            repo_name: Repository name on HuggingFace Hub
            private: Whether to make repository private
            use_auth_token: HuggingFace API token
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded or trained")
        
        logger.info(f"Pushing model to HuggingFace Hub: {repo_name}")
        
        self.model.push_to_hub(
            repo_name,
            private=private,
            use_auth_token=use_auth_token
        )
        
        self.tokenizer.push_to_hub(
            repo_name,
            private=private,
            use_auth_token=use_auth_token
        )
        
        logger.info("Model pushed successfully!")
