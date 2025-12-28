#!/usr/bin/env python3
"""
Fine-tune Qwen3-8B model with extended vocabulary for semantic IDs using transformers and deepspeed.
Stage 1: Embedding initialization - trains only new token embeddings.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)


@dataclass
class FineTuneConfig:
    """Configuration for Stage 1: Embedding initialization."""

    # Model settings
    model_name: str = "Qwen/Qwen3-8B"
    max_seq_length: int = 2048
    dtype: torch.dtype = torch.bfloat16  # Fixed to bfloat16
    random_state: int = 1368
    num_proc: int = 32
    enable_thinking: bool = False

    # Semantic ID vocabulary extension
    extend_vocabulary: bool = True
    codebook_levels: int = 4
    codebook_size: int = 256
    num_semantic_tokens: int = 1024  # <|sid_0|> to <|sid_1023|>

    # Data settings
    category: str = "Video_Games"
    data_dir: Path = Path("data")
    max_training_samples: int = 32000  # Sample size for embedding init

    # Training params
    learning_rate: float = 1e-3
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000
    num_train_epochs: int = 1
    warmup_steps: int = 100
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

    # Memory optimization
    gradient_checkpointing: bool = False

    # Optimizer settings
    optim: str = "adamw_torch"

    # Output settings
    output_dir: Path = Path("models/qwen3_8b_vocab_extended")
    logging_steps: int = 100
    eval_steps: int = 250
    save_steps: int = 5000

    # DeepSpeed settings
    deepspeed: Optional[str] = None

    # Computed paths (set in __post_init__)
    train_path: Optional[Path] = None
    val_path: Optional[Path] = None

    def __post_init__(self):
        """Post-initialization setup and validation."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.train_path = self.data_dir / "output" / f"{self.category}_conversations_train.parquet"
        self.val_path = self.data_dir / "output" / f"{self.category}_conversations_val.parquet"

        if not self.train_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {self.train_path}. "
                f"Please ensure you have run the data preparation scripts for category '{self.category}'."
            )


def extend_tokenizer(model, tokenizer, config: FineTuneConfig):
    """Add semantic ID tokens to the tokenizer. Returns the number of new tokens added."""
    original_vocab_size = len(tokenizer)
    original_embedding_size = model.get_input_embeddings().weight.shape[0]

    if original_embedding_size > original_vocab_size:
        model.resize_token_embeddings(original_vocab_size)

    new_tokens = ["<|rec|>", "<|sid_start|>", "<|sid_end|>"]
    for i in range(config.num_semantic_tokens):
        new_tokens.append(f"<|sid_{i}|>")

    num_added = tokenizer.add_tokens(new_tokens, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    # Initialize new embeddings with mean of existing embeddings
    embedding_layer = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    original_vocab_size = len(tokenizer) - num_added

    with torch.no_grad():
        existing_embeddings = embedding_layer.weight[:original_vocab_size]
        mean_embedding = existing_embeddings.mean(dim=0)
        new_embedding_size = embedding_layer.weight.shape[0] - original_vocab_size
        embedding_layer.weight[original_vocab_size:].data = mean_embedding.unsqueeze(0).repeat(
            new_embedding_size, 1
        )

        if output_embeddings is not None and output_embeddings.weight.shape[0] == embedding_layer.weight.shape[0]:
            existing_output = output_embeddings.weight[:original_vocab_size]
            mean_output = existing_output.mean(dim=0)
            output_embeddings.weight[original_vocab_size:].data = mean_output.unsqueeze(0).repeat(
                new_embedding_size, 1
            )

    # Verify consistency
    new_vocab_size = len(tokenizer)
    new_embedding_size = model.get_input_embeddings().weight.shape[0]
    new_lm_head_size = model.get_output_embeddings().weight.shape[0]

    if new_vocab_size != new_embedding_size or new_vocab_size != new_lm_head_size:
        model.resize_token_embeddings(new_vocab_size)

    assert new_vocab_size == new_embedding_size == new_lm_head_size, "Model dimension mismatch!"
    return num_added


def prepare_model(model, tokenizer, config: FineTuneConfig, num_new_tokens: int):
    """Prepare model for embedding-only training by freezing all parameters except new embeddings."""
    current_vocab_size = len(tokenizer)
    current_embedding_size = model.get_input_embeddings().weight.shape[0]
    assert current_embedding_size == current_vocab_size, "Embedding size mismatch!"

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze input and output embeddings
    embedding_layer = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()
    embedding_layer.weight.requires_grad = True

    if output_embeddings is not None:
        output_embeddings.weight.requires_grad = True

    # Set gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    else:
        model.gradient_checkpointing_disable()

    model.config.use_cache = not config.gradient_checkpointing
    return model


def load_sid_dataset(config: FineTuneConfig, tokenizer, split="train"):
    """Load and prepare the conversation dataset with semantic IDs."""
    data_path = config.train_path if split == "train" else config.val_path

    if split == "val" and not data_path.exists():
        return None

    dataset = load_dataset("parquet", data_files=str(data_path), split="train")

    num_samples = min(len(dataset), 500 if split == "val" else config.max_training_samples)
    dataset = dataset.shuffle(seed=config.random_state).select(range(num_samples))

    def apply_chat_template(example):
        text = tokenizer.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=config.enable_thinking,
        )
        return {"text": text}

    dataset = dataset.map(apply_chat_template, remove_columns=dataset.column_names, num_proc=config.num_proc)
    return dataset


def tokenize_dataset(examples, tokenizer, max_length):
    """Tokenize text data for language modeling."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_overflowing_tokens=False,
    )


def train_embeddings(model, tokenizer, config: FineTuneConfig, num_new_tokens: int):
    """
    Train only the new token embeddings using Trainer with deepspeed support.
    
    Args:
        model: Model with extended vocabulary (only embeddings are trainable)
        tokenizer: Extended tokenizer
        config: Training configuration
        num_new_tokens: Number of newly added tokens
    """
    train_dataset = load_sid_dataset(config, tokenizer, split="train")
    val_dataset = load_sid_dataset(config, tokenizer, split="val")

    # Tokenize datasets
    train_dataset = train_dataset.map(
        lambda x: tokenize_dataset(x, tokenizer, config.max_seq_length),
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=config.num_proc,
    )

    if val_dataset:
        val_dataset = val_dataset.map(
            lambda x: tokenize_dataset(x, tokenizer, config.max_seq_length),
            batched=True,
            remove_columns=val_dataset.column_names,
            num_proc=config.num_proc,
        )

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Setup distributed training parameters
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        max_steps=config.max_steps if config.max_steps > 0 else -1,
        num_train_epochs=config.num_train_epochs if config.max_steps <= 0 else 1,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        logging_steps=config.logging_steps,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=config.eval_steps if val_dataset else None,
        save_steps=config.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,
        fp16=False,  # Disable fp16
        bf16=True,  # Force bf16
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        seed=config.random_state,
        report_to="tensorboard",  # Use TensorBoard for logging
        deepspeed=config.deepspeed,
        ddp_find_unused_parameters=False if ddp else None,
        eval_delay=1 if val_dataset else 0,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    return trainer.state


def save_model_and_tokenizer(model, tokenizer, config: FineTuneConfig):
    """Save the model with initialized embeddings and extended tokenizer."""
    # Verify dimensions before saving
    input_size = model.get_input_embeddings().weight.shape[0]
    output_size = model.get_output_embeddings().weight.shape[0]
    vocab_size = len(tokenizer)

    if input_size != vocab_size or output_size != vocab_size:
        model.resize_token_embeddings(vocab_size)
        input_size = model.get_input_embeddings().weight.shape[0]
        output_size = model.get_output_embeddings().weight.shape[0]

    # Final verification
    assert input_size == vocab_size, f"Input embeddings size mismatch: {input_size} != {vocab_size}"
    assert output_size == vocab_size, f"Output embeddings size mismatch: {output_size} != {vocab_size}"

    # Save to final directory
    final_save_path = config.output_dir / "final"
    final_save_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(final_save_path))
    tokenizer.save_pretrained(str(final_save_path))

    config_dict = {
        "stage": "vocab_extension",
        "model_name": config.model_name,
        "num_semantic_tokens": config.num_semantic_tokens,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "category": config.category,
        "vocabulary_size": len(tokenizer),
    }

    with open(final_save_path / "training_config.json", "w") as f:
        json.dump(config_dict, f, indent=2)


if __name__ == "__main__":
    config = FineTuneConfig()

    # Handle command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-8B with vocabulary extension using DeepSpeed")
    parser.add_argument("--deepspeed", type=str, default=None, help="Path to deepspeed config file")
    parser.add_argument("--model_name", type=str, default=None, help="Model name")
    parser.add_argument("--category", type=str, default=None, help="Data category")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--max_steps", type=int, default=None, help="Max training steps")

    args = parser.parse_args()

    # Override config with command line arguments
    if args.deepspeed:
        config.deepspeed = args.deepspeed
    if args.model_name:
        config.model_name = args.model_name
    if args.category:
        config.category = args.category
    if args.output_dir:
        config.output_dir = Path(args.output_dir)
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.max_steps:
        config.max_steps = args.max_steps

    # Ensure bfloat16
    config.dtype = torch.bfloat16

    # Setup device_map for model loading
    # Note: When using DeepSpeed, set device_map=None as DeepSpeed manages device placement automatically
    if config.deepspeed:
        device_map = None
    else:
        # For standard training, use device_map for efficient multi-GPU/CPU loading
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        device_map = {"": local_rank} if world_size > 1 else "auto"

    # Load model and tokenizer
    model_kwargs = {
        "torch_dtype": config.dtype,
        "trust_remote_code": True,
    }
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    model = AutoModelForCausalLM.from_pretrained(config.model_name, **model_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        model_max_length=config.max_seq_length,
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Extend vocabulary
    num_new_tokens = 0
    if config.extend_vocabulary:
        num_new_tokens = extend_tokenizer(model, tokenizer, config)
        model = prepare_model(model, tokenizer, config, num_new_tokens)

    # Train embeddings
    train_embeddings(model, tokenizer, config, num_new_tokens)

    # Save model and tokenizer
    save_model_and_tokenizer(model, tokenizer, config)
