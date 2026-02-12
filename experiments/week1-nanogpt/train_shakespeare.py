"""
Train GPT on Shakespeare Character-Level Dataset

Week 1, Days 3-5 Activity

This script trains the GPT model on the Shakespeare character-level dataset
using MPS (Apple Silicon GPU acceleration).

Dataset: Tiny Shakespeare corpus
Task: Character-level language modeling
Device: MPS (Apple Silicon)
Compile: Disabled (MPS compatibility)

Key command:
    python train.py config/train_shakespeare_char.py --device=mps --compile=False
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import pickle


# ============================================================================
# Configuration and Hyperparameters
# ============================================================================
# TODO: Define hyperparameters
# - batch_size: Number of sequences per batch
# - block_size: Maximum context length
# - max_iters: Total training iterations
# - eval_interval: How often to evaluate
# - eval_iters: Number of batches for evaluation
# - learning_rate: AdamW learning rate
# - device: 'mps' for Apple Silicon
# - n_embd: Embedding dimension
# - n_head: Number of attention heads
# - n_layer: Number of transformer blocks
# - dropout: Dropout rate


# ============================================================================
# Data Loading
# ============================================================================
# TODO: Load Shakespeare dataset
# - Download if not present
# - Read text file
# - Create character-level tokenizer (char -> int mapping)
# - Encode entire dataset
# - Split into train/val sets (90/10 split)
# - Create data loading functions (get_batch)


# ============================================================================
# Model Instantiation
# ============================================================================
# TODO: Import GPT model from my_gpt.py
# TODO: Initialize model with config
# TODO: Move model to MPS device
# TODO: Count and print number of parameters


# ============================================================================
# Training Loop
# ============================================================================
# TODO: Set up AdamW optimizer
# TODO: Implement training loop:
#   - Sample batch from training data
#   - Forward pass
#   - Compute loss
#   - Backward pass
#   - Optimizer step
#   - Log training metrics
#
# TODO: Implement evaluation loop:
#   - Disable gradient computation
#   - Evaluate on train and val sets
#   - Compute average loss
#   - Log validation metrics


# ============================================================================
# Checkpoint Saving
# ============================================================================
# TODO: Save model checkpoints periodically
# - Save model state_dict
# - Save optimizer state
# - Save training iteration
# - Save best validation loss


# ============================================================================
# Sample Generation
# ============================================================================
# TODO: Generate samples after training
# - Create context (e.g., single newline character)
# - Generate 500 characters
# - Decode and print
# - Compare quality to initial random samples


if __name__ == "__main__":
    # TODO: Parse command-line arguments if needed
    # TODO: Run training
    # TODO: Save final checkpoint
    # TODO: Generate and display samples
    pass
