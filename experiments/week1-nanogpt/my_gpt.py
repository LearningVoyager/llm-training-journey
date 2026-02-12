"""
My GPT Implementation from Scratch

Week 1, Days 3-5 Activity

This is a clean, standalone implementation of GPT based on Andrej Karpathy's
"Let's build GPT" video. This code is extracted and refined from the messy
exploratory work in build_gpt_codealong.ipynb.

Architecture:
- Character-level tokenization
- Multi-head self-attention
- Feed-forward networks
- Layer normalization
- Residual connections
- Positional embeddings

Reference: https://www.youtube.com/watch?v=kCc8FmEb1nY
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


# ============================================================================
# Hyperparameters and Configuration
# ============================================================================
# Define model size, training params, etc.
# batch_size, block_size, n_embd, n_head, n_layer, dropout, learning_rate, etc.


# ============================================================================
# Single Attention Head
# ============================================================================
class Head(nn.Module):
    """
    Single self-attention head.

    Computes attention scores between tokens in the sequence, allowing each
    token to gather information from previous tokens (causal masking).

    Key concepts:
    - Query, Key, Value projections
    - Scaled dot-product attention
    - Causal masking (lower triangular) for autoregressive generation
    """

    def __init__(self, head_size):
        super().__init__()
        # TODO: Initialize query, key, value projections
        # TODO: Register causal mask buffer
        pass

    def forward(self, x):
        # TODO: Compute Q, K, V
        # TODO: Compute attention scores (Q @ K.T) / sqrt(head_size)
        # TODO: Apply causal mask
        # TODO: Softmax to get attention weights
        # TODO: Apply weights to values
        pass


# ============================================================================
# Multi-Head Attention
# ============================================================================
class MultiHeadAttention(nn.Module):
    """
    Multiple attention heads running in parallel.

    Why multiple heads?
    - Each head can learn to attend to different aspects of the input
    - Heads might specialize (e.g., one for syntax, one for semantics)
    - Increases model capacity without depth

    Architecture:
    - Run multiple heads in parallel
    - Concatenate their outputs
    - Project back to embedding dimension
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # TODO: Create list of attention heads
        # TODO: Create output projection
        # TODO: Add dropout
        pass

    def forward(self, x):
        # TODO: Run all heads in parallel
        # TODO: Concatenate outputs
        # TODO: Project and dropout
        pass


# ============================================================================
# Feed-Forward Network
# ============================================================================
class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Why needed?
    - Attention is just weighted averaging (linear operation)
    - FFN adds non-linearity and per-position computation
    - Gives model capacity to process what attention gathered

    Architecture:
    - Linear layer that expands dimension (typically 4x)
    - Non-linearity (ReLU/GELU)
    - Linear layer that projects back
    - Dropout
    """

    def __init__(self, n_embd):
        super().__init__()
        # TODO: Build FFN layers
        # TODO: Add dropout
        pass

    def forward(self, x):
        # TODO: Forward pass through FFN
        pass


# ============================================================================
# Transformer Block
# ============================================================================
class Block(nn.Module):
    """
    Single transformer block.

    Combines self-attention and feed-forward with residual connections
    and layer normalization.

    Why residual connections?
    - Allow gradients to flow through deep networks
    - Enable training of very deep models
    - Each block only needs to learn the "residual" (delta)

    Why layer normalization?
    - Stabilizes training
    - Normalizes activations across features
    - Applied before attention and FFN (pre-norm variant)

    Architecture:
    - x = x + attention(LayerNorm(x))
    - x = x + ffn(LayerNorm(x))
    """

    def __init__(self, n_embd, n_head):
        super().__init__()
        # TODO: Multi-head attention
        # TODO: Feed-forward network
        # TODO: Layer norms
        pass

    def forward(self, x):
        # TODO: Attention block with residual
        # TODO: FFN block with residual
        pass


# ============================================================================
# Full GPT Model
# ============================================================================
class GPT(nn.Module):
    """
    Full GPT language model.

    Architecture:
    1. Token embeddings: convert token indices to vectors
    2. Position embeddings: add positional information
    3. Stack of transformer blocks
    4. Final layer norm
    5. Linear head: project to vocabulary logits

    Forward pass:
    - Embed tokens and positions
    - Pass through transformer blocks
    - Normalize and project to logits

    Generation:
    - Autoregressive: generate one token at a time
    - Sample from output distribution
    - Append to context and repeat
    """

    def __init__(self, vocab_size):
        super().__init__()
        # TODO: Token embedding table
        # TODO: Position embedding table
        # TODO: Stack of transformer blocks
        # TODO: Final layer norm
        # TODO: Language model head
        pass

    def forward(self, idx, targets=None):
        # TODO: Get token and position embeddings
        # TODO: Pass through transformer blocks
        # TODO: Apply final layer norm
        # TODO: Get logits
        # TODO: Compute loss if targets provided
        pass

    def generate(self, idx, max_new_tokens):
        """
        Generate new tokens autoregressively.

        idx: (B, T) array of indices in current context
        max_new_tokens: number of tokens to generate

        Returns: (B, T+max_new_tokens) array of indices
        """
        # TODO: Implement generation loop
        # TODO: For each new token:
        #   - Get predictions (forward pass)
        #   - Sample from distribution
        #   - Append to sequence
        pass


# ============================================================================
# Training Loop
# ============================================================================
def train():
    """
    Main training loop.

    Steps:
    1. Load and prepare data
    2. Initialize model
    3. Set up optimizer
    4. Training loop:
       - Sample batch
       - Forward pass
       - Compute loss
       - Backward pass
       - Update weights
       - Log metrics
    5. Generate samples periodically
    """
    # TODO: Load data
    # TODO: Initialize model
    # TODO: Create optimizer
    # TODO: Training loop
    # TODO: Evaluation and generation
    pass


if __name__ == "__main__":
    # TODO: Run training
    # TODO: Generate samples from trained model
    pass
