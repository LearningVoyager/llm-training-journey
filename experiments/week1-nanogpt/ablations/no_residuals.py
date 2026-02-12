"""
Ablation Experiment: Remove Residual Connections

Week 1, Day 7 Activity

Ablation: Remove residual connections to observe gradient flow issues.

Hypothesis:
Without residual connections (skip connections), gradients won't flow
effectively through deep networks, causing training to degrade.

Why residuals matter:
- In deep networks, gradients can vanish as they backpropagate through layers
- Residual connections provide "shortcuts" for gradients to flow
- They allow training of much deeper networks (ResNet insight)
- Each layer only needs to learn the residual (delta) instead of full transformation

Expected observations:
- Gradient vanishing in deeper layers
- Training loss may plateau early
- Much slower convergence
- Final performance significantly worse
- Deeper models will suffer more than shallow ones

Without residuals:
- x = attention(x)  # Instead of: x = x + attention(x)
- x = ffn(x)        # Instead of: x = x + ffn(x)

This forces each layer to learn the complete transformation, making
optimization much harder.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import sys
import os

# Add parent directory to path to import from my_gpt.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================================
# Modified Model (No Residual Connections)
# ============================================================================
# TODO: Copy Block class from my_gpt.py
# TODO: Remove residual connections:
#   - Change: x = x + attention(LayerNorm(x))
#   - To: x = attention(LayerNorm(x))
#   - Change: x = x + ffn(LayerNorm(x))
#   - To: x = ffn(LayerNorm(x))

# TODO: Keep LayerNorm to isolate the effect of residuals
# (We're only testing residuals, not normalization)


# ============================================================================
# Training for 200 Iterations
# ============================================================================
# TODO: Set up minimal training loop
# - Load Shakespeare data
# - Initialize modified model
# - Train for 200 iterations
# - Log loss every 10 iterations

# TODO: Monitor gradient flow:
# - Track gradient norms in each layer
# - Check if early layers receive gradients
# - Compare to baseline gradient flow


# ============================================================================
# Logging and Results
# ============================================================================
# TODO: Save results
# - Loss curve
# - Gradient norms per layer
# - Training time comparison

# TODO: Analyze gradient flow
# - Do gradients reach early layers?
# - How does depth affect learning?
# - Compare final layer vs first layer gradient magnitudes

# TODO: Generate samples
# - Likely poor quality
# - Document degradation


if __name__ == "__main__":
    print("=" * 60)
    print("Ablation: No Residual Connections")
    print("=" * 60)
    print("\nHypothesis: Gradients won't flow through deep network")
    print("Running 200 iterations...\n")

    # TODO: Run experiment
    # TODO: Analyze gradient flow
    # TODO: Report findings
    pass
