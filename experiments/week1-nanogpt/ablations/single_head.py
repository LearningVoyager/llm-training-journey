"""
Ablation Experiment: Single Attention Head

Week 1, Day 7 Activity

Ablation: Use single attention head instead of multi-head to observe
capacity reduction.

Hypothesis:
A single attention head limits the model's ability to attend to different
representation subspaces simultaneously, reducing model capacity and
performance.

Why multi-head attention matters:
- Different heads can specialize in different patterns
- One head might focus on local syntax, another on long-range dependencies
- Increases model capacity without adding depth
- Allows parallel processing of different attention patterns

Expected observations:
- Model can still train (unlike previous ablations)
- Lower final performance compared to multi-head baseline
- Generated text may lack diversity or complexity
- Loss plateau higher than baseline
- Model may struggle with complex patterns that need multiple types of attention

Single head vs Multi-head:
- Single head: One set of Q, K, V projections
- Multi-head: Multiple parallel heads, concatenated and projected
- Multi-head typically uses 4-8 heads

This is a "softer" ablation - the model should still work, just less effectively.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os

# Add parent directory to path to import from my_gpt.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Modified Model (Single Head Attention)
# ============================================================================
# TODO: Copy MultiHeadAttention class from my_gpt.py
# TODO: Modify to use only 1 head instead of n_head
# TODO: Keep all other components the same (LayerNorm, residuals, FFN)

# Alternative approach:
# TODO: Replace MultiHeadAttention with single Head class
# TODO: Adjust dimensions to match multi-head output size
# TODO: Use a projection layer to maintain correct dimensions


# ============================================================================
# Training for 200 Iterations
# ============================================================================
# TODO: Set up training loop
# - Load Shakespeare data
# - Initialize modified model (n_head = 1)
# - Train for 200 iterations
# - Log loss every 10 iterations

# TODO: Compare with multi-head baseline:
# - Loss convergence speed
# - Final loss value
# - Training stability (should be stable)


# ============================================================================
# Logging and Results
# ============================================================================
# TODO: Save results
# - Loss curve (should be smooth but higher than baseline)
# - Model checkpoint
# - Generated samples

# TODO: Qualitative analysis of generated text:
# - Does it capture language patterns?
# - How does quality compare to multi-head?
# - What kinds of errors appear?
# - Does it struggle with specific patterns?

# TODO: Quantitative comparison:
# - Final train/val loss vs baseline
# - Perplexity comparison
# - Generation diversity metrics


if __name__ == "__main__":
    print("=" * 60)
    print("Ablation: Single Attention Head")
    print("=" * 60)
    print("\nHypothesis: Single head reduces model capacity")
    print("Running 200 iterations...\n")

    # TODO: Run experiment
    # TODO: Compare with multi-head baseline
    # TODO: Analyze generated samples
    # TODO: Report findings
    pass
