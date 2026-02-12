"""
Ablation Experiment: Remove LayerNorm

Week 1, Day 7 Activity

Ablation: Remove layer normalization to observe training instability.

Hypothesis:
Without LayerNorm, training should become unstable or fail to converge.
LayerNorm stabilizes the distribution of activations, preventing them from
exploding or vanishing as they flow through deep networks. Removing it should
lead to:
- Higher variance in activations
- Gradient instability
- Slower convergence or divergence
- Higher sensitivity to learning rate

Expected observations:
- Loss may spike or oscillate wildly
- Training may diverge (NaN losses)
- If training succeeds, it will be much slower
- Final performance will be worse

Why this matters:
Understanding the role of normalization techniques is crucial for training
deep networks. This experiment demonstrates why LayerNorm (or BatchNorm) is
a critical component of modern architectures.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os

# Add parent directory to path to import from my_gpt.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Modified Model (No LayerNorm)
# ============================================================================
# TODO: Copy Block class from my_gpt.py
# TODO: Remove all LayerNorm layers
# TODO: Keep only:
#   - x = x + attention(x)  # No LayerNorm before attention
#   - x = x + ffn(x)        # No LayerNorm before FFN

# TODO: Copy GPT class from my_gpt.py
# TODO: Remove final LayerNorm before output projection


# ============================================================================
# Training for 200 Iterations
# ============================================================================
# TODO: Set up minimal training loop
# - Load Shakespeare data
# - Initialize modified model
# - Train for only 200 iterations (quick experiment)
# - Log loss every 10 iterations

# TODO: Monitor for instability:
# - Check for NaN or inf in loss
# - Track gradient norms
# - Compare loss curve to baseline


# ============================================================================
# Logging and Results
# ============================================================================
# TODO: Save results to file
# - Loss values over iterations
# - Final model state (if training succeeds)
# - Observations about training dynamics

# TODO: Generate samples (if model trains at all)
# - Compare quality to baseline
# - Note any degradation

# TODO: Create visualization
# - Plot loss curve vs baseline
# - Highlight instabilities


if __name__ == "__main__":
    print("=" * 60)
    print("Ablation: No LayerNorm")
    print("=" * 60)
    print("\nHypothesis: Training will be unstable without LayerNorm")
    print("Running 200 iterations...\n")

    # TODO: Run experiment
    # TODO: Report findings
    pass
