#!/usr/bin/env python3
"""
Generate embeddings for product items using Qwen3-Embedding model on multiple NPU devices.
Reads pre-tokenized data and generates embeddings in parallel across 8 NPU cards.

First run src/tokenize_items.py to pre-process data on CPU.
"""

import os
import time
from pathlib import Path
from multiprocessing import Process, Queue, Manager
from typing import Tuple, Dict, List

# Disable tokenizers parallelism to avoid forking warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel

from src.logger import setup_logger

logger = setup_logger("embed-items-multi-npu", log_to_file=True)

# Data settings
CATEGORY = "Video_Games"  # Product category to process
NUM_ROWS = None  # Number of rows to process (None = all)
DATA_DIR = Path("data")  # Data directory path

# Model settings
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"  # HuggingFace model name
BATCH_SIZE = 64  # Batch size for processing per NPU
TARGET_DIM = 1024  # Target embedding dimension
NUM_NPU = 8  # Number of NPU devices to use

# Other settings
LOG_FREQ = 1000  # Log progress every N items


class TokenizedDataset(Dataset):
    """Dataset for pre-tokenized data."""

    def __init__(self, input_ids: np.ndarray, attention_mask: np.ndarray, start_idx: int = 0, end_idx: int = None):
        """Initialize with numpy arrays for a specific slice."""
        self.input_ids = input_ids[start_idx:end_idx] if end_idx else input_ids[start_idx:]
        self.attention_mask = attention_mask[start_idx:end_idx] if end_idx else attention_mask[start_idx:]
        self.length = len(self.input_ids)
        self.start_idx = start_idx

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return {
            "input_ids": torch.from_numpy(self.input_ids[idx]),
            "attention_mask": torch.from_numpy(self.attention_mask[idx]),
            "original_idx": self.start_idx + idx,  # Keep track of original index for merging
        }


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Extract embeddings using last token pooling."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def generate_embeddings(
    model: AutoModel,
    device: str,
    pretokenized_batch: dict,
    target_dim: int = 1024,
) -> np.ndarray:
    """
    Generate embeddings for a batch of pre-tokenized inputs using last token pooling.
    Returns L2-normalized embeddings, optionally truncated to target dimension.
    """
    # Move to device
    encoded = {k: v.to(device) for k, v in pretokenized_batch.items() if k != "original_idx"}

    # Generate embeddings
    with torch.no_grad():
        outputs = model(**encoded)

        # Use last token pooling
        embeddings = last_token_pool(outputs.last_hidden_state, encoded["attention_mask"])

        # Truncate to target dimension if specified
        if target_dim and target_dim < embeddings.shape[1]:
            embeddings = embeddings[:, :target_dim]

        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()


def collate_fn(batch):
    """Custom collate function to handle original_idx."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    original_indices = [item["original_idx"] for item in batch]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "original_idx": original_indices,
    }


def worker_process(
    rank: int,
    num_npu: int,
    model_name: str,
    batch_size: int,
    target_dim: int,
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    start_idx: int,
    end_idx: int,
    result_queue: Queue,
    progress_queue: Queue,
):
    """
    Worker process that runs on a specific NPU device.

    Args:
        rank: NPU device rank (0-7)
        num_npu: Total number of NPU devices
        model_name: HuggingFace model name
        batch_size: Batch size per device
        target_dim: Target embedding dimension
        input_ids: Full input_ids array
        attention_mask: Full attention_mask array
        start_idx: Start index for this worker's data slice
        end_idx: End index for this worker's data slice
        result_queue: Queue to send results back to main process
        progress_queue: Queue to send progress updates
    """
    try:
        # Set device for this process
        device = f"npu:{rank}"
        torch.npu.set_device(rank)

        worker_logger = setup_logger(f"embed-items-npu-{rank}", log_to_file=True)
        worker_logger.info(f"Worker {rank} started on {device}, processing indices {start_idx} to {end_idx}")

        # Load model on this NPU
        worker_logger.info(f"Loading model {model_name} on {device}")
        model = AutoModel.from_pretrained(model_name)
        model = model.to(device)
        model.eval()
        worker_logger.info(f"Model loaded on {device}")

        # Create dataset for this worker's slice
        dataset = TokenizedDataset(input_ids, attention_mask, start_idx, end_idx)
        num_items = len(dataset)
        worker_logger.info(f"Worker {rank} processing {num_items:,} items")

        # Create DataLoader (no multiprocessing within worker to avoid conflicts)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # No sub-workers to avoid conflicts
            pin_memory=False,
            collate_fn=collate_fn,
        )

        # Pre-allocate output array for this worker's slice
        embeddings = np.zeros((num_items, target_dim), dtype=np.float32)
        original_indices = []

        # Process batches
        start_time = time.time()
        current_local_idx = 0

        for batch_idx, batch in enumerate(dataloader):
            try:
                batch_size_actual = batch["input_ids"].size(0)
                batch_indices = batch["original_idx"]

                # Generate embeddings
                batch_embeddings = generate_embeddings(model, device, batch, target_dim)

                # Store embeddings and indices
                embeddings[current_local_idx : current_local_idx + batch_size_actual] = batch_embeddings
                original_indices.extend(batch_indices)
                current_local_idx += batch_size_actual

                # Send progress update
                if (batch_idx + 1) % 10 == 0 or current_local_idx == num_items:
                    elapsed = time.time() - start_time
                    items_per_sec = current_local_idx / elapsed if elapsed > 0 else 0
                    progress_queue.put((rank, current_local_idx, num_items, items_per_sec))

            except Exception as e:
                worker_logger.error(f"Error processing batch {batch_idx} on worker {rank}: {e}")
                raise

        # Final timing
        total_time = time.time() - start_time
        worker_logger.info(f"Worker {rank} complete! Processed {num_items:,} items in {total_time:.1f}s ({num_items/total_time:.1f} items/sec)")

        # Send results back to main process
        result_queue.put((rank, embeddings, original_indices))

        worker_logger.info(f"Worker {rank} sent results to main process")

    except Exception as e:
        worker_logger.error(f"Worker {rank} failed with error: {e}", exc_info=True)
        result_queue.put((rank, None, None))  # Signal failure
        raise


def embed_items_multi_npu():
    """Main function to generate embeddings using multiple NPU devices."""
    # Check NPU availability
    if not torch.npu.is_available():
        raise RuntimeError("NPU is not available. This script requires NPU devices.")

    num_npu_available = torch.npu.device_count()
    if num_npu_available < NUM_NPU:
        logger.warning(f"Only {num_npu_available} NPU devices available, but {NUM_NPU} requested. Using {num_npu_available} devices.")
        num_npu = num_npu_available
    else:
        num_npu = NUM_NPU

    logger.info(f"Using {num_npu} NPU devices for parallel embedding generation")
    logger.info(f"Model: {MODEL_NAME}, Batch size per NPU: {BATCH_SIZE}, Target dim: {TARGET_DIM}")

    # Setup paths
    input_path = DATA_DIR / "output" / f"{CATEGORY}_items_updated.parquet"
    output_path = DATA_DIR / "output" / f"{CATEGORY}_items_with_embeddings.parquet"

    # Tokenized data path
    suffix = f"_{NUM_ROWS}" if NUM_ROWS else ""
    tokenized_path = DATA_DIR / "output" / f"{CATEGORY}_tokenized{suffix}.npz"

    # Load data
    logger.info(f"Loading data from {input_path}")
    item_df = pl.read_parquet(input_path)
    logger.info(f"Loaded {len(item_df):,} items from {input_path}")

    # Apply row limit if specified
    if NUM_ROWS is not None:
        logger.info(f"Limiting to {NUM_ROWS} rows for testing")
        item_df = item_df.head(NUM_ROWS)

    total_items = len(item_df)
    logger.info(f"Processing {total_items:,} items across {num_npu} NPU devices")

    # Load pre-tokenized data
    if not tokenized_path.exists():
        raise FileNotFoundError(
            f"Pre-tokenized data not found at {tokenized_path}. Please run src/tokenize_items.py first."
        )

    logger.info(f"Loading pre-tokenized data from {tokenized_path}")
    pretokenized_data = np.load(tokenized_path)

    # Verify data matches
    if pretokenized_data["n_items"] != total_items:
        raise ValueError(
            f"Pre-tokenized data has {pretokenized_data['n_items']} items, but current data has {total_items}"
        )

    input_ids = pretokenized_data["input_ids"]
    attention_mask = pretokenized_data["attention_mask"]
    logger.info(f"Loaded pre-tokenized data: shape {input_ids.shape}")

    # Split data across NPU devices
    items_per_npu = total_items // num_npu
    remainder = total_items % num_npu

    splits = []
    start_idx = 0
    for rank in range(num_npu):
        # Distribute remainder items across first few devices
        end_idx = start_idx + items_per_npu + (1 if rank < remainder else 0)
        splits.append((start_idx, end_idx))
        logger.info(f"NPU {rank}: indices {start_idx} to {end_idx} ({end_idx - start_idx} items)")
        start_idx = end_idx

    # Create queues for communication
    manager = Manager()
    result_queue = manager.Queue()
    progress_queue = manager.Queue()

    # Start worker processes
    processes = []
    for rank in range(num_npu):
        start_idx, end_idx = splits[rank]
        p = Process(
            target=worker_process,
            args=(
                rank,
                num_npu,
                MODEL_NAME,
                BATCH_SIZE,
                TARGET_DIM,
                input_ids,
                attention_mask,
                start_idx,
                end_idx,
                result_queue,
                progress_queue,
            ),
        )
        p.start()
        processes.append(p)
        logger.info(f"Started worker process for NPU {rank}")

    # Monitor progress and collect results
    logger.info("Waiting for workers to complete...")
    all_results = {}
    completed_workers = 0
    start_time = time.time()
    progress_tracker = {i: 0 for i in range(num_npu)}  # Track progress per worker

    # Progress bar
    with tqdm(total=total_items, desc="Generating embeddings") as pbar:
        while completed_workers < num_npu:
            # Check for progress updates
            try:
                while not progress_queue.empty():
                    rank, current, total, items_per_sec = progress_queue.get_nowait()
                    progress_tracker[rank] = current
                    total_progress = sum(progress_tracker.values())
                    pbar.n = total_progress
                    pbar.set_postfix({"items/sec": f"{items_per_sec:.1f}"})
                    pbar.refresh()
            except:
                pass

            # Check for completed workers
            try:
                rank, embeddings, original_indices = result_queue.get(timeout=1.0)
                if embeddings is not None:
                    all_results[rank] = (embeddings, original_indices)
                    completed_workers += 1
                    logger.info(f"Received results from worker {rank} ({completed_workers}/{num_npu})")
                    # Update progress bar with final count
                    progress_tracker[rank] = len(embeddings)
                    total_progress = sum(progress_tracker.values())
                    pbar.n = total_progress
                    pbar.refresh()
                else:
                    logger.error(f"Worker {rank} failed")
                    completed_workers += 1
            except:
                # Timeout, continue waiting
                pass

            # Check if any process died
            for i, p in enumerate(processes):
                if not p.is_alive() and i not in all_results:
                    logger.error(f"Worker process {i} died unexpectedly")
                    completed_workers += 1

    # Wait for all processes to finish
    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            logger.warning(f"Process {p.pid} did not terminate, killing it")
            p.terminate()
            p.join()

    logger.info("All workers completed")

    # Merge results in correct order
    logger.info("Merging results from all NPU devices...")
    all_embeddings = np.zeros((total_items, TARGET_DIM), dtype=np.float32)

    for rank in range(num_npu):
        if rank not in all_results:
            raise RuntimeError(f"Missing results from worker {rank}")
        embeddings, original_indices = all_results[rank]
        # Place embeddings at their original indices
        # Note: embeddings and original_indices should be in the same order
        for i, orig_idx in enumerate(original_indices):
            if orig_idx < 0 or orig_idx >= total_items:
                raise ValueError(f"Invalid original index {orig_idx} from worker {rank}")
            all_embeddings[orig_idx] = embeddings[i]
        logger.info(f"Merged results from NPU {rank}: {len(embeddings)} embeddings")

    # Final timing
    total_time = time.time() - start_time
    logger.info("Embedding generation complete!")
    logger.info(f"Total time: {total_time:.1f} seconds ({total_time / 60:.1f} minutes)")
    logger.info(f"Average time per item: {total_time / total_items * 1000:.1f} ms")
    logger.info(f"Throughput: {total_items / total_time:.1f} items/sec")

    # Verify embeddings are normalized
    norms = np.linalg.norm(all_embeddings, axis=1)
    logger.info(f"Embedding L2 norms - Mean: {norms.mean():.6f}, Std: {norms.std():.6f}")

    # Add embeddings to dataframe
    embeddings_list = all_embeddings.tolist()
    item_df_with_embeddings = item_df.with_columns(pl.Series("embedding", embeddings_list, dtype=pl.List(pl.Float32)))

    # Save results
    logger.info(f"Saving embeddings to: {output_path}")
    item_df_with_embeddings.write_parquet(output_path)

    # Final statistics
    logger.info("Final statistics:")
    logger.info(f"- Total items with embeddings: {len(item_df_with_embeddings):,}")
    logger.info(f"- Embedding dimension: {all_embeddings.shape[1]}")
    logger.info(f"- Output file size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    logger.info(f"- Processing rate: {total_items / total_time:.1f} items/sec")

    # NPU memory stats
    for rank in range(num_npu):
        try:
            torch.npu.set_device(rank)
            allocated = torch.npu.memory_allocated(rank) / 1024**3
            reserved = torch.npu.memory_reserved(rank) / 1024**3
            logger.info(f"- NPU {rank} memory: {allocated:.1f} GB allocated, {reserved:.1f} GB reserved")
        except:
            pass


if __name__ == "__main__":
    embed_items_multi_npu()

