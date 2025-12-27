#!/usr/bin/env python3
"""
Fine-tune Qwen3-8B model with extended vocabulary for semantic IDs using transformers and deepspeed.
Stage 1: Embedding initialization - trains only new token embeddings.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    DataCollatorForLanguageModeling,
)
from transformers.trainer_utils import is_bfloat16_supported

import wandb
from src.device_manager import DeviceManager
from src.logger import setup_logger
from src.test_prompts import REC_TEST_PROMPTS, SYSTEM_PROMPT

logger = setup_logger("finetune-qwen3-vocab-deepspeed", log_to_file=True)


@dataclass
class FineTuneConfig:
    """Configuration for Stage 1: Embedding initialization."""

    # Model settings
    model_name: str = "Qwen/Qwen3-8B"  # Use HuggingFace model name directly
    max_seq_length: int = 2048
    dtype: Optional[torch.dtype] = None  # None for auto detection
    random_state: int = 1368
    num_proc: int = 32
    enable_thinking: bool = False

    # Semantic ID vocabulary extension
    extend_vocabulary: bool = True
    codebook_levels: int = 4  # Number of hierarchical levels
    codebook_size: int = 256  # Number of codes per codebook
    num_semantic_tokens: int = 1024  # <|sid_0|> to <|sid_1023|>
    system_prompt: str = SYSTEM_PROMPT

    # Data settings
    category: str = "Video_Games"
    data_dir: Path = Path("data")
    max_training_samples: int = 32000  # Sample size for embedding init

    # Training params
    learning_rate: float = 1e-3
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_steps: int = 1000
    num_train_epochs: int = 1  # Train for 1 epoch
    warmup_steps: int = 100
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

    # Memory optimization
    gradient_checkpointing: bool = False  # Disable to use more memory but slightly faster

    # Optimizer settings
    optim: str = "adamw_torch"  # Use standard adamw for deepspeed compatibility

    # Output settings
    output_dir: Path = Path("models/qwen3_8b_vocab_extended")
    steps_per_train_log: int = 100  # Log training progress every N steps
    steps_per_val_log: int = 250  # Validate and checkpoint every N steps
    save_steps: int = 5000  # Save checkpoints during training

    # DeepSpeed settings
    deepspeed: Optional[str] = None  # Path to deepspeed config file

    # Computed paths (set in __post_init__)
    train_path: Optional[Path] = None
    val_path: Optional[Path] = None

    def __post_init__(self):
        """Post-initialization setup and validation."""
        if self.dtype is None:
            self.dtype = torch.float16 if not is_bfloat16_supported() else torch.bfloat16

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set and validate data paths
        self.train_path = self.data_dir / "output" / f"{self.category}_conversations_train.parquet"
        self.val_path = self.data_dir / "output" / f"{self.category}_conversations_val.parquet"

        # Validate that training data exists
        if not self.train_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {self.train_path}. "
                f"Please ensure you have run the data preparation scripts for category '{self.category}'."
            )

        # Validation data is optional, just log if missing
        if not self.val_path.exists():
            logger.warning(f"Validation data not found at {self.val_path}. Training without validation set.")

    def log_config(self):
        """Log all configuration parameters."""
        logger.info("=== Qwen3-8B Vocabulary Extension Configuration (DeepSpeed) ===")
        logger.info("Stage 1: Embedding Initialization")

        # Model settings
        logger.info("Model Settings:")
        logger.info(f"  model_name: {self.model_name}")
        logger.info(f"  max_seq_length: {self.max_seq_length}")
        logger.info(f"  dtype: {self.dtype}")
        logger.info(f"  gradient_checkpointing: {self.gradient_checkpointing}")
        logger.info(f"  random_state: {self.random_state}")

        # Vocabulary extension
        logger.info("Vocabulary Extension:")
        logger.info(f"  extend_vocabulary: {self.extend_vocabulary}")
        logger.info(f"  codebook_levels: {self.codebook_levels}")
        logger.info(f"  codebook_size: {self.codebook_size}")
        logger.info(f"  num_semantic_tokens: {self.num_semantic_tokens}")
        logger.info(f"  Total new tokens: {self.num_semantic_tokens + 2} (including <|sid_start|> and <|sid_end|>)")

        # Data settings
        logger.info("Data Settings:")
        logger.info(f"  category: {self.category}")
        logger.info(f"  data_dir: {self.data_dir}")
        logger.info(f"  train_path: {self.train_path}")
        logger.info(f"  val_path: {self.val_path}")
        logger.info(f"  max_training_samples: {self.max_training_samples}")

        # Training parameters
        logger.info("Training Parameters (Stage 1):")
        logger.info(f"  learning_rate: {self.learning_rate} (high for embedding init)")
        logger.info(f"  batch_size: {self.batch_size}")
        logger.info(f"  gradient_accumulation_steps: {self.gradient_accumulation_steps}")
        logger.info(f"  effective_batch_size: {self.batch_size * self.gradient_accumulation_steps}")
        logger.info(f"  max_steps: {self.max_steps}")
        logger.info(f"  num_train_epochs: {self.num_train_epochs}")
        logger.info(f"  warmup_steps: {self.warmup_steps}")
        logger.info(f"  weight_decay: {self.weight_decay}")
        logger.info(f"  lr_scheduler_type: {self.lr_scheduler_type}")
        logger.info(f"  optim: {self.optim}")

        # DeepSpeed settings
        logger.info("DeepSpeed Settings:")
        logger.info(f"  deepspeed: {self.deepspeed}")

        # Output settings
        logger.info("Output Settings:")
        logger.info(f"  output_dir: {self.output_dir}")
        logger.info(f"  steps_per_train_log: {self.steps_per_train_log}")
        logger.info(f"  steps_per_val_log: {self.steps_per_val_log}")
        logger.info(f"  save_steps: {self.save_steps}")
        logger.info("============================================")


def extend_tokenizer(model, tokenizer, config: FineTuneConfig):
    """
    Add semantic ID tokens to the tokenizer.

    Returns the number of new tokens added.
    """
    logger.info("=== Extending tokenizer with semantic ID tokens ===")

    # Log initial state
    original_vocab_size = len(tokenizer)
    original_embedding_size = model.get_input_embeddings().weight.shape[0]
    original_lm_head_size = model.get_output_embeddings().weight.shape[0]

    logger.info(
        f"Before - Vocab size: {original_vocab_size:,}, Embedding matrix: {original_embedding_size:,}, LM head matrix: {original_lm_head_size:,}"
    )

    # Fix size mismatch if model has more embeddings than tokenizer vocab
    if original_embedding_size > original_vocab_size:
        logger.warning(
            f"⚠ Model has {original_embedding_size - original_vocab_size} more embeddings than tokenizer tokens."
        )
        logger.info("Resizing model to match tokenizer before adding new tokens")
        model.resize_token_embeddings(original_vocab_size)

        # Update sizes after resize
        original_embedding_size = model.get_input_embeddings().weight.shape[0]
        original_lm_head_size = model.get_output_embeddings().weight.shape[0]
        logger.info(
            f"After resize - Embedding matrix: {original_embedding_size:,}, LM head matrix: {original_lm_head_size:,}"
        )

    # Add special tokens
    new_tokens = ["<|rec|>", "<|sid_start|>", "<|sid_end|>"]

    # Add tokens for semantic IDs: <|sid_0|> through <|sid_1023|>
    # Token mapping: <|sid_X|> where X = level * 256 + value
    # Level 0: <|sid_0|> through <|sid_255|>
    # Level 1: <|sid_256|> through <|sid_511|>
    # Level 2: <|sid_512|> through <|sid_767|>
    # Level 3: <|sid_768|> through <|sid_1023|>
    for i in range(config.num_semantic_tokens):
        new_tokens.append(f"<|sid_{i}|>")

    logger.info(f"Adding {len(new_tokens)} new tokens")
    logger.info("  Special tokens: <|rec|>, <|sid_start|>, <|sid_end|>")
    logger.info(f"  Semantic ID tokens: <|sid_0|> through <|sid_{config.num_semantic_tokens - 1}|>")

    # Add new tokens to tokenizer
    num_added = tokenizer.add_tokens(new_tokens, special_tokens=True)
    logger.info(f"Tokenizer added {num_added} new tokens")

    # Resize model embeddings to match new vocab size
    model.resize_token_embeddings(len(tokenizer))

    # Initialize new embeddings with mean of existing embeddings
    embedding_layer = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    original_vocab_size = len(tokenizer) - num_added
    with torch.no_grad():
        # Get mean of existing embeddings
        existing_embeddings = embedding_layer.weight[:original_vocab_size]
        mean_embedding = existing_embeddings.mean(dim=0)

        # Initialize new embeddings with mean
        new_embedding_size = embedding_layer.weight.shape[0] - original_vocab_size
        embedding_layer.weight[original_vocab_size:].data = mean_embedding.unsqueeze(0).repeat(
            new_embedding_size, 1
        )

        # Same for output embeddings
        if output_embeddings is not None and output_embeddings.weight.shape[0] == embedding_layer.weight.shape[0]:
            existing_output = output_embeddings.weight[:original_vocab_size]
            mean_output = existing_output.mean(dim=0)
            output_embeddings.weight[original_vocab_size:].data = mean_output.unsqueeze(0).repeat(
                new_embedding_size, 1
            )

    logger.info("✅ Initialized new embeddings with mean of existing embeddings")

    # Log final state
    new_vocab_size = len(tokenizer)
    new_embedding_size = model.get_input_embeddings().weight.shape[0]
    new_lm_head_size = model.get_output_embeddings().weight.shape[0]

    logger.info(
        f"After - Vocab size: {new_vocab_size:,}, Embedding matrix: {new_embedding_size:,}, LM head matrix: {new_lm_head_size:,}"
    )

    # Verify consistency - CRITICAL CHECK
    if new_vocab_size != new_embedding_size:
        logger.error(f"❌ CRITICAL: Tokenizer size ({new_vocab_size}) != Embedding size ({new_embedding_size})")
        logger.info("Attempting to force resize model embeddings")
        model.resize_token_embeddings(new_vocab_size)
        new_embedding_size = model.get_input_embeddings().weight.shape[0]
        new_lm_head_size = model.get_output_embeddings().weight.shape[0]
        logger.info(f"After forced resize - Embedding: {new_embedding_size}, LM head: {new_lm_head_size}")

    if new_vocab_size != new_lm_head_size:
        logger.error(f"❌ CRITICAL: Tokenizer size ({new_vocab_size}) != LM head size ({new_lm_head_size})")
        logger.error("Model will NOT be able to generate new tokens!")
        # Try to fix by resizing
        model.resize_token_embeddings(new_vocab_size)
        new_lm_head_size = model.get_output_embeddings().weight.shape[0]
        logger.info(f"After forced resize - LM head: {new_lm_head_size}")

    # Final verification
    if new_vocab_size == new_embedding_size == new_lm_head_size:
        logger.info("✅ Model dimensions verified: All layers properly sized")
    else:
        logger.error("❌ Model dimension mismatch persists - this will cause generation issues!")

    logger.info(f"\n✓ Successfully added {num_added} new tokens")
    logger.info("=" * 50)

    return num_added


def prepare_model(model, tokenizer, config: FineTuneConfig, num_new_tokens: int):
    """
    Prepare model for embedding-only training by freezing all parameters except new embeddings.
    """
    logger.info("=== Preparing model for embedding-only training ===")

    # Get current vocab size
    current_vocab_size = len(tokenizer)
    current_embedding_size = model.get_input_embeddings().weight.shape[0]

    logger.info(
        f"Current - Vocab size: {current_vocab_size:,}, Embedding matrix: {current_embedding_size:,}, New tokens: {num_new_tokens:}"
    )

    # Verify embeddings are properly sized
    assert current_embedding_size == current_vocab_size, "Embedding size mismatch!"

    # Get the original vocab size (before adding new tokens)
    original_vocab_size = current_vocab_size - num_new_tokens

    # Freeze all parameters first
    for _, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze BOTH input and output embeddings
    embedding_layer = model.get_input_embeddings()
    output_embeddings = model.get_output_embeddings()

    # Unfreeze entire embedding layers for simplicity
    embedding_layer.weight.requires_grad = True

    if output_embeddings is not None:
        output_embeddings.weight.requires_grad = True
        logger.info("✅ Unfroze both input and output embedding layers for training")
    else:
        logger.error("❌ Could not access output embeddings - only input embeddings will be trained!")
        logger.error("This will cause generation issues!")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())

    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    logger.info(f"Percentage trainable: {100 * trainable_params / total_params:.4f}%")

    # Check initialization of new embeddings
    with torch.no_grad():
        new_embeddings = embedding_layer.weight[original_vocab_size:]
        logger.info("New embeddings statistics:")
        logger.info(f"  Shape: {new_embeddings.shape}")
        logger.info(f"  Mean: {new_embeddings.mean().item():.6f}")
        logger.info(f"  Std: {new_embeddings.std().item():.6f}")
        logger.info(f"  Min: {new_embeddings.min().item():.6f}")
        logger.info(f"  Max: {new_embeddings.max().item():.6f}")

    # Set gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled for memory efficiency")
    else:
        model.gradient_checkpointing_disable()
        logger.info("Gradient checkpointing disabled")

    # Set cache based on gradient checkpointing
    model.config.use_cache = not config.gradient_checkpointing

    logger.info("=== Model preparation complete ===")

    return model


def load_sid_dataset(config: FineTuneConfig, tokenizer, split="train"):
    """
    Load and prepare the conversation dataset with semantic IDs.

    Args:
        config: Configuration object
        tokenizer: Tokenizer to apply chat template
        split: "train" or "val" to load respective dataset
    """
    logger.info(f"Loading semantic ID conversation dataset ({split})")

    # Select the appropriate path based on split
    if split == "train":
        data_path = config.train_path
    elif split == "val":
        data_path = config.val_path
    logger.info(f"Loading from: {data_path}")

    dataset = load_dataset("parquet", data_files=str(data_path), split="train")
    logger.info(f"Loaded {len(dataset)} conversations")

    # For validation, use fewer samples
    if split == "val":
        num_samples = min(len(dataset), 500)  # Max 500 for validation
    else:
        num_samples = min(len(dataset), config.max_training_samples)

    logger.info(f"Sampling {num_samples} examples for {split}")
    dataset = dataset.shuffle(seed=config.random_state).select(range(num_samples))

    # Apply chat template using map
    def apply_chat_template(example):
        text = tokenizer.apply_chat_template(
            example["conversations"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    logger.info("Applying chat template to conversations")
    dataset = dataset.map(apply_chat_template, remove_columns=dataset.column_names, num_proc=config.num_proc)

    logger.info(f"Created dataset with {len(dataset)} examples")

    # Verify semantic IDs are present (only for train)
    if split == "train" and len(dataset) > 0:
        sample_text = dataset[0]["text"]
        if "<|sid_start|>" in sample_text and "<|sid_end|>" in sample_text:
            logger.info("✓ Verified: Semantic ID tokens found in dataset")
            sid_count = sample_text.count("<|sid_start|>")
            logger.info(f"  Sample contains {sid_count} semantic ID(s)")

            # Log a sample of the chat template output
            logger.info("=" * 60)
            logger.info(f"Sample chat template output ({split}): {sample_text[:500]}...")
            logger.info("=" * 60)

        else:
            logger.warning("⚠ Warning: No semantic ID tokens found in sample")

    return dataset


class DataInspectionCallback(TrainerCallback):
    """Inspect training data and tokenization at each logging step."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.trainer = None  # Will be set later
        self.first_batch_inspected = False

    def set_trainer(self, trainer):
        """Set the trainer after it's been created."""
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        """Inspect first batch at training start."""
        if not self.first_batch_inspected:
            self.first_batch_inspected = True
            logger.info("\n" + "=" * 60)
            logger.info("=== Initial Training Data Inspection ===")
            logger.info("=" * 60)

            try:
                if self.trainer is None:
                    logger.info("Trainer not yet set, skipping inspection")
                    return

                train_dataloader = self.trainer.get_train_dataloader()

                # Get first batch
                for batch in train_dataloader:
                    logger.info(f"Batch keys: {batch.keys()}")
                    logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
                    logger.info(f"Attention mask shape: {batch['attention_mask'].shape}")

                    # Inspect first example in batch
                    first_example = batch["input_ids"][0]
                    decoded = self.tokenizer.decode(first_example, skip_special_tokens=False)

                    # Show full token list
                    logger.info(f"Tokens (first example): {first_example.tolist()[:50]}...")
                    logger.info(f"Decoded: {decoded[:500]}...")

                    break  # Just check first batch

            except Exception as e:
                logger.info(f"Could not inspect first batch: {e}")

            logger.info("=" * 60 + "\n")


class EmbeddingMonitorCallback(TrainerCallback):
    """Monitor embedding statistics and log to W&B."""

    def __init__(self, tokenizer, num_new_tokens, codebook_size=256, monitor_interval=100):
        self.tokenizer = tokenizer
        self.num_new_tokens = num_new_tokens
        self.codebook_size = codebook_size
        self.monitor_interval = monitor_interval
        self.original_vocab_size = len(tokenizer) - num_new_tokens
        self.initial_embeddings = None
        self.prev_embeddings = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        """Capture initial embedding state."""
        embeddings = model.get_input_embeddings().weight
        self.initial_embeddings = embeddings[self.original_vocab_size :].clone().detach()
        self.prev_embeddings = self.initial_embeddings.clone()

        # Log initial statistics
        mean = self.initial_embeddings.mean().item()
        std = self.initial_embeddings.std().item()

        wandb.log(
            {
                "embeddings/initial_mean": mean,
                "embeddings/initial_std": std,
                "embeddings/initial_norm": self.initial_embeddings.norm(dim=-1).mean().item(),
            }
        )

        logger.info(f"Initial embeddings - Mean: {mean:.4f}, Std: {std:.4f}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Monitor embedding changes and log."""
        if state.global_step % self.monitor_interval == 0 and state.global_step > 0:
            embeddings = model.get_input_embeddings().weight
            new_embeddings = embeddings[self.original_vocab_size :]

            # Calculate changes
            change_from_init = (new_embeddings - self.initial_embeddings).abs().mean().item()
            change_from_prev = (new_embeddings - self.prev_embeddings).abs().mean().item()

            # Calculate statistics
            mean = new_embeddings.mean().item()
            std = new_embeddings.std().item()
            norm = new_embeddings.norm(dim=-1).mean().item()

            # Calculate per-level statistics (for semantic ID levels)
            level_stats = {}
            tokens_per_level = self.codebook_size
            for level in range(4):
                start_idx = level * tokens_per_level
                end_idx = min((level + 1) * tokens_per_level, self.num_new_tokens - 2)  # -2 for <sid>, </sid>
                if start_idx < self.num_new_tokens - 2:
                    level_embeddings = new_embeddings[start_idx:end_idx]
                    level_stats[f"embeddings/level_{level}_mean"] = level_embeddings.mean().item()
                    level_stats[f"embeddings/level_{level}_std"] = level_embeddings.std().item()
                    level_stats[f"embeddings/level_{level}_norm"] = level_embeddings.norm(dim=-1).mean().item()

            wandb_log = {
                "embeddings/change_from_init": change_from_init,
                "embeddings/change_from_prev": change_from_prev,
                "embeddings/mean": mean,
                "embeddings/std": std,
                "embeddings/norm": norm,
                "step": state.global_step,
                **level_stats,
            }

            wandb.log(wandb_log)

            # Console logging
            logger.info(
                f"Step {state.global_step} - Embeddings: "
                f"Change(init)={change_from_init:.4f}, Change(prev)={change_from_prev:.6f}, "
                f"Mean={mean:.4f}, Std={std:.4f}, Norm={norm:.4f}"
            )

            self.prev_embeddings = new_embeddings.clone().detach()


class SemanticIDGenerationCallback(TrainerCallback):
    """Test semantic ID generation and log to W&B."""

    def __init__(self, tokenizer, test_interval=200):
        self.tokenizer = tokenizer
        self.test_interval = test_interval
        self.test_messages = REC_TEST_PROMPTS

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Test generation at specified intervals."""
        if state.global_step % self.test_interval == 0 and state.global_step > 0:
            self.test_generation(model, state.global_step)

    def test_generation(self, model, step):
        """Test if model can generate with semantic IDs and log to W&B."""
        logger.info("=" * 60)
        logger.info(f"Testing semantic ID generation at step {step}")
        logger.info("=" * 60)

        training_mode = model.training
        model.eval()

        successful_generations = 0
        generation_results = []

        # Get device from model
        device = next(model.parameters()).device

        for i, messages in enumerate(self.test_messages, 1):
            # Apply chat template to format the messages properly
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(device)

            try:
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, max_new_tokens=100, temperature=0.7, top_p=0.8, top_k=20
                    )

                generated_full = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
                generated_new = generated_full[len(prompt) :]

                # Analysis
                has_sid_tags = "<|sid_start|>" in generated_new or "<|sid_end|>" in generated_new
                sid_tokens = [t for t in generated_new.split() if t.startswith("<|sid_") and t.endswith("|>")]
                uses_semantic_ids = has_sid_tags or len(sid_tokens) > 0

                if uses_semantic_ids:
                    successful_generations += 1

                # Get user message for cleaner logging
                user_message = messages[-1]["content"]

                # Store for table
                generation_results.append([step, user_message, generated_new, uses_semantic_ids, len(sid_tokens)])

                # Log results
                logger.info(f"\nTest {i}: {user_message}")
                logger.info(f"  Generated: {generated_new}")
                logger.info(f"  ✓ Uses SIDs: {uses_semantic_ids} (tags={has_sid_tags}, tokens={len(sid_tokens)})")

            except Exception as e:
                user_message = messages[-1]["content"]
                logger.warning(f"Generation failed for prompt {i}: {e}")
                generation_results.append([step, user_message[:50], f"[Error: {e}]", False, 0])

        success_rate = successful_generations / len(self.test_messages)

        wandb.log(
            {
                "generation/success_rate": success_rate,
                "generation/successful_count": successful_generations,
                "generation/total_prompts": len(self.test_messages),
                "generation/examples": wandb.Table(
                    columns=["Step", "User_Message", "Generated", "Uses_SID", "Num_Tokens"], data=generation_results
                ),
                "step": step,
            }
        )

        logger.info(
            f"\nSummary: {successful_generations}/{len(self.test_messages)} "
            f"({success_rate:.0%}) prompts generated semantic IDs"
        )

        model.train(training_mode)
        logger.info("=" * 60 + "\n")


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
        model: The model with extended vocabulary
        tokenizer: The extended tokenizer
        config: Training configuration
        num_new_tokens: Number of new tokens added to vocabulary
    """
    logger.info("Starting Stage 1: Embedding initialization training")

    train_dataset = load_sid_dataset(config, tokenizer, split="train")
    val_dataset = load_sid_dataset(config, tokenizer, split="val")

    # Tokenize datasets
    logger.info("Tokenizing datasets")
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

    wandb.log(
        {
            "dataset/train_size": len(train_dataset) if train_dataset else 0,
            "dataset/val_size": len(val_dataset) if val_dataset else 0,
            "dataset/vocabulary_size": len(tokenizer),
            "dataset/new_tokens": num_new_tokens,
        }
    )

    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False  # Causal LM, not masked LM
    )

    # Setup device map for distributed training
    device_map = "auto"
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if ddp:
        device_map = {"": local_rank}
        logger.info(f"Using DDP with local_rank={local_rank}")

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
        logging_steps=config.steps_per_train_log,
        eval_strategy="steps" if val_dataset else "no",
        eval_steps=config.steps_per_val_log if val_dataset else None,
        save_steps=config.save_steps,
        save_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model="eval_loss" if val_dataset else None,
        greater_is_better=False,
        fp16=config.dtype == torch.float16,
        bf16=config.dtype == torch.bfloat16,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        seed=config.random_state,
        report_to="wandb",
        deepspeed=config.deepspeed,
        ddp_find_unused_parameters=False if ddp else None,
        eval_delay=1 if val_dataset else 0,
    )

    # Create data inspection callback
    data_inspection_callback = DataInspectionCallback(tokenizer)

    # Create other callbacks
    callbacks = [
        data_inspection_callback,
        EmbeddingMonitorCallback(
            tokenizer, num_new_tokens, codebook_size=config.codebook_size, monitor_interval=config.steps_per_val_log
        ),
        SemanticIDGenerationCallback(tokenizer, test_interval=config.steps_per_val_log),
    ]

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # Set trainer in the data inspection
    data_inspection_callback.set_trainer(trainer)

    # Show current memory stats (if CUDA available)
    if torch.cuda.is_available():
        gpu_stats = torch.cuda.get_device_properties(local_rank if ddp else 0)
        start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        logger.info(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
        logger.info(f"{start_gpu_memory} GB of memory reserved.")

    trainer.train()

    # Get training stats
    train_stats = trainer.state

    # Show final memory and time stats (if CUDA available)
    if torch.cuda.is_available():
        used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
        used_memory_for_training = round(used_memory - start_gpu_memory, 3)
        used_percentage = round(used_memory / max_memory * 100, 3)
        training_percentage = round(used_memory_for_training / max_memory * 100, 3)
        logger.info(f"Peak reserved memory = {used_memory} GB.")
        logger.info(f"Peak reserved memory for training = {used_memory_for_training} GB.")
        logger.info(f"Peak reserved memory % of max memory = {used_percentage} %.")
        logger.info(f"Peak reserved memory for training % of max memory = {training_percentage} %.")

    # Log final summary
    final_loss = train_stats.log_history[-1].get("train_loss") if train_stats.log_history else None
    total_steps = train_stats.global_step if hasattr(train_stats, "global_step") else config.max_steps

    wandb.summary["final_loss"] = final_loss
    wandb.summary["total_steps"] = total_steps

    logger.info("Stage 1 embedding initialization completed!")
    return train_stats


def save_model_and_tokenizer(model, tokenizer, config: FineTuneConfig):
    """
    Save the model with initialized embeddings and extended tokenizer.
    """
    logger.info("Saving model and tokenizer")

    # CRITICAL: Verify dimensions before saving
    input_size = model.get_input_embeddings().weight.shape[0]
    output_size = model.get_output_embeddings().weight.shape[0]
    vocab_size = len(tokenizer)

    logger.info("=== Pre-save verification ===")
    logger.info(f"Model input embedding size: {input_size}")
    logger.info(f"Model output embedding size: {output_size}")
    logger.info(f"Tokenizer vocabulary size: {vocab_size}")

    if input_size != vocab_size or output_size != vocab_size:
        logger.error("❌ CRITICAL: Size mismatch detected before save!")
        logger.error(f"  Input: {input_size}, Output: {output_size}, Tokenizer: {vocab_size}")
        logger.error("This will cause generation issues in Stage 2!")
        # Try to fix one more time
        model.resize_token_embeddings(vocab_size)
        input_size = model.get_input_embeddings().weight.shape[0]
        output_size = model.get_output_embeddings().weight.shape[0]
        logger.info(f"After final resize - Input: {input_size}, Output: {output_size}")

    # Final verification
    assert input_size == vocab_size, f"Input embeddings size mismatch: {input_size} != {vocab_size}"
    assert output_size == vocab_size, f"Output embeddings size mismatch: {output_size} != {vocab_size}"
    logger.info("✅ All dimensions verified - safe to save")

    # Save to final directory
    final_save_path = config.output_dir / "final"
    final_save_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving to: {final_save_path}")

    # Save model
    model.save_pretrained(str(final_save_path))

    # Save tokenizer
    tokenizer.save_pretrained(str(final_save_path))

    logger.info("Model and tokenizer saved successfully!")
    logger.info(f"Checkpoint location: {final_save_path}")

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

    logger.info("Training configuration saved")


if __name__ == "__main__":
    config = FineTuneConfig()

    # Handle command line arguments for deepspeed config
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

    device_manager = DeviceManager(logger)
    device = device_manager.device

    run_name = f"qwen3-8b-embed-{config.category}-lr{config.learning_rate}-deepspeed"
    run = wandb.init(project="semantic-id-vocab-extension", name=run_name, config=config.__dict__)
    config.log_config()

    logger.info("Loading base model")
    
    # When using DeepSpeed, don't use device_map as DeepSpeed handles device placement
    # Setup device_map for distributed training
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Device map setup for distributed training
    # When using DeepSpeed, device_map should match the distributed setup
    if ddp:
        device_map = {"": local_rank}
        logger.info(f"Using device_map with local_rank={local_rank}")
    else:
        device_map = "auto"
        logger.info("Using device_map='auto'")
    
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        torch_dtype=config.dtype,
        device_map=device_map,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        model_max_length=config.max_seq_length,
    )

    # Set pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    num_new_tokens = 0
    if config.extend_vocabulary:
        num_new_tokens = extend_tokenizer(model, tokenizer, config)
        model = prepare_model(model, tokenizer, config, num_new_tokens)

    train_stats = train_embeddings(model, tokenizer, config, num_new_tokens)

    logger.info("Saving embeddings as W&B artifact")
    embeddings = model.get_input_embeddings().weight
    new_embeddings = embeddings[len(tokenizer) - num_new_tokens :].detach().cpu()

    artifact = wandb.Artifact(
        f"semantic_embeddings_{config.category}",
        type="embeddings",
        description=f"Trained semantic ID embeddings for {config.category}",
        metadata={
            "num_tokens": num_new_tokens,
            "model": config.model_name,
            "steps": train_stats.global_step if hasattr(train_stats, "global_step") else config.max_steps,
        },
    )

    embeddings_path = config.output_dir / "semantic_embeddings.npy"
    np.save(embeddings_path, new_embeddings.float().numpy())
    artifact.add_file(str(embeddings_path))
    wandb.log_artifact(artifact)

    save_model_and_tokenizer(model, tokenizer, config)

    wandb.finish()

    logger.info("=" * 50)
    logger.info("Stage 1: Embedding initialization complete!")
    logger.info(f"Initialized {config.num_semantic_tokens + 2} new semantic ID tokens")
    logger.info(f"Model saved to: {config.output_dir / 'final'}")
    logger.info("=" * 50)

