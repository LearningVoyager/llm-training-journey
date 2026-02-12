"""
Train GPT on Custom Dataset

Week 1, Day 6 Activity

This script trains the GPT model on a custom dataset of my choosing:
- Poems or literature
- Code (Python, JavaScript, etc.)
- My own writing
- Any other text corpus I'm interested in

Goal: Compare performance with Shakespeare model and observe how the model
adapts to different domains and writing styles.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import pickle


# ============================================================================
# Dataset Path Configuration
# ============================================================================
# TODO: Define path to custom dataset
# - Where is the raw text file?
# - What encoding should be used?
# - What's the dataset size?

# Example datasets to consider:
# - Poetry: Edgar Allan Poe, Emily Dickinson, Robert Frost
# - Code: Python libraries, my own code, GitHub repos
# - Personal: My journal entries, blog posts, notes
# - Other: Philosophy texts, song lyrics, Reddit comments


# ============================================================================
# Data Preparation
# ============================================================================
# TODO: Load custom dataset
# - Read text file
# - Examine dataset statistics (length, vocabulary size)
# - Create tokenizer (character-level or BPE)
# - Encode text to integers
# - Create train.bin and val.bin files

# TODO: Compare with Shakespeare dataset:
# - Vocabulary size
# - Text length
# - Character distribution
# - Average word length


# ============================================================================
# Model Configuration
# ============================================================================
# TODO: Define hyperparameters
# - Use same architecture as Shakespeare model for fair comparison
# - Or experiment with different sizes based on dataset

# Considerations:
# - Smaller dataset? Use smaller model or more regularization
# - Larger vocabulary? Increase embedding dimension
# - Different domain? Adjust training iterations


# ============================================================================
# Training Loop
# ============================================================================
# TODO: Set up training (similar to train_shakespeare.py)
# - Initialize model
# - Create optimizer
# - Training loop with logging
# - Save checkpoints


# ============================================================================
# Sample Generation
# ============================================================================
# TODO: Generate samples after training
# - Create diverse prompts relevant to your dataset
# - Generate completions
# - Evaluate quality subjectively

# TODO: Compare with Shakespeare model:
# - Does it capture the style of my dataset?
# - What unique patterns does it learn?
# - How does generation quality compare?


# ============================================================================
# Comparison Notes
# ============================================================================
# TODO: Document observations:
# - Training dynamics (loss curves, convergence speed)
# - Generated sample quality
# - Memorization vs generalization
# - Interesting failure modes or surprising behaviors
# - What does this tell me about the model and the domain?


if __name__ == "__main__":
    # TODO: Run data preparation
    # TODO: Run training
    # TODO: Generate and analyze samples
    # TODO: Compare with Shakespeare baseline
    pass
