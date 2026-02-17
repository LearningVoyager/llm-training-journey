"""
═══════════════════════════════════════════════════════════════════════════════
GOAL: SCALE UP the model and add DROPOUT for better generalization
═══════════════════════════════════════════════════════════════════════════════

What we're building:
    A MUCH BIGGER transformer with dropout regularization to prevent overfitting.
    This is our first step toward a production-scale model!

The Two Big Changes in v8:
    
    1. SCALING UP - Making the model bigger and more powerful:
       - Context window: 8 → 256 characters (32x larger!)
       - Embedding size: 32 → 384 dimensions (12x larger!)
       - Num layers: 3 → 6 blocks (2x deeper!)
       - Num heads: 4 → 6 heads
       - Batch size: 32 → 64 (more examples per update)
       
       Why scale up?
           Bigger models can learn more complex patterns and generate
           better text. We're moving from a toy model to something that
           can actually write convincing Shakespeare!
    
    2. DROPOUT - Randomly "turning off" neurons during training:
       
       What is dropout?
           During training, randomly set some activations to zero.
           Rate of 0.2 means 20% of values become 0 at each step.
           
       Why does this help?
           Prevents overfitting (memorization) by forcing the network
           to learn robust features that work even when some neurons
           are missing.
           
       Real-world analogy:
           Training a sports team where random players are benched
           each practice. This forces:
           - Every player to be useful (no "dead weight")
           - The team to work together (no relying on one star)
           - Robust strategies that work with any lineup
           
           When the full team plays (inference), they perform even better!

Where We Add Dropout:
    
    1. After attention weights (in Head)
       → Prevents over-reliance on specific attention patterns
    
    2. After multi-head projection (in MultiHeadAttention)
       → Regularizes the attention output before residual connection
    
    3. After feedforward (in FeedForward)
       → Regularizes the computation before residual connection
    
    Key principle: Add dropout RIGHT BEFORE residual connections
    This prevents the network from just "copying" inputs through
    the residual pathway without learning useful transformations.

Hyperparameter Scaling Logic:
    
    n_embd = 384    (embedding dimension)
    n_head = 6      (number of attention heads)
    head_size = 384 / 6 = 64 dimensions per head
    
    Why 64 dimensions per head?
        Each head gets 64 dimensions to work with (query, key, value all 64-dim).
        6 heads × 64 dims = 384 total, matching n_embd perfectly.
        
    Why 6 heads instead of 4?
        More heads = more diverse perspectives on the data.
        With a bigger model (384 dims vs 32), we can afford more heads
        to capture different types of patterns simultaneously.

Architecture Changes:
    
    v7: Hard-coded 3 blocks
        self.blocks = nn.Sequential(
            Block(...),
            Block(...),
            Block(...),
        )
    
    v8: Configurable n_layer blocks (6 in this case)
        self.blocks = nn.Sequential(
            *[Block(...) for _ in range(n_layer)]
        )
        
        The * unpacks the list into individual arguments.
        This makes it easy to experiment with different depths!

═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS - Scaled up for better performance!
# ══════════════════════════════════════════════════════════════════════════════

batch_size    = 64     # Process 64 sequences at once (2x more than v7)
block_size    = 256    # Each sequence is 256 characters (32x longer context!)
max_iters     = 5000   # Train for 5,000 steps
eval_interval = 500    # Check progress every 500 steps
learning_rate = 3e-4   # Lower learning rate for stable training of bigger model
eval_iters    = 200    # Average over 200 batches for loss estimates
n_embd        = 384    # 384-dimensional embeddings (12x larger than v7!)
n_head        = 6      # 6 attention heads (each head is 384/6 = 64 dims)
n_layer       = 6      # 6 transformer blocks (2x deeper than v7!)
dropout       = 0.2    # Drop 20% of activations during training (prevents overfitting)

# Why these specific values?
#   - block_size=256: Can see much more context (full sentences, not just words)
#   - n_embd=384: Large enough to learn rich representations
#   - n_head=6: Divides evenly into 384 (64 dims per head)
#   - n_layer=6: Deep enough for sophisticated patterns, not too deep to train
#   - dropout=0.2: Standard value that works well in practice

# ══════════════════════════════════════════════════════════════════════════════
# DEVICE SETUP - Use the fastest hardware we have available
# ══════════════════════════════════════════════════════════════════════════════
# Why? Training on GPU is 10-100x faster than CPU
# We check in order: NVIDIA GPU → Apple GPU → CPU fallback

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device} \n")

torch.manual_seed(1337)  # Same random numbers every run = reproducible experiments

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING - Get the Shakespeare text we'll learn from
# ══════════════════════════════════════════════════════════════════════════════
# Goal: Load raw text into memory so we can process it
# Dataset: ~1MB of Shakespeare plays (40,000 lines, 1 million characters)

current_dir  = Path.cwd()
project_root = current_dir.parent.parent.parent
data_path    = project_root / 'data/shakespeare/input.txt'

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset size: {len(text):,} characters")
print(text[:1000])  # Preview the first 1000 characters

# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZATION - Convert text into numbers
# ══════════════════════════════════════════════════════════════════════════════
# Why? Neural networks can only process numbers, not letters.
# Strategy: Give each unique character its own ID number (0-64)
#   Example: 'a'→0, 'b'→1, 'c'→2, ..., 'z'→25, etc.

chars      = sorted(list(set(text)))  # Find all unique characters, alphabetically
vocab_size = len(chars)               # How many different characters exist? (65 total)

print(f"Vocabulary ({vocab_size} characters): {''.join(chars)}")

# Create lookup dictionaries for converting back and forth
stoi = {ch: i for i, ch in enumerate(chars)}   # string to integer: 'a' → 0
itos = {i: ch for i, ch in enumerate(chars)}   # integer to string: 0 → 'a'

# Helper functions for encoding/decoding
encode = lambda s: [stoi[c] for c in s]          # "hi" → [20, 21]
decode = lambda l: ''.join([itos[i] for i in l]) # [20, 21] → "hi"

print(f"encode('hii there') = {encode('hii there')}")
print(f"decode back         = {decode(encode('hii there'))}")

# ══════════════════════════════════════════════════════════════════════════════
# CONVERT ENTIRE DATASET TO NUMBERS
# ══════════════════════════════════════════════════════════════════════════════
# Why? We need the whole text as a big tensor of integers for training
# Result: 1 million characters → 1 million integers in a PyTorch tensor

data = torch.tensor(encode(text), dtype=torch.long)

print(f"Data shape: {data.shape}  |  dtype: {data.dtype}")
print(f"First 100 tokens: {data[:100]}")

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN/VALIDATION SPLIT - Set aside some data for testing
# ══════════════════════════════════════════════════════════════════════════════
# Why split? We need to know if the model actually *learned* or just *memorized*
# 
# Analogy: It's like studying for an exam
#   - Train data = practice problems you study from
#   - Validation data = the actual exam (never seen during study)
#   - If you do well on the exam, you truly learned. If you only do well on
#     practice problems, you just memorized answers.
#
# Split: 90% training, 10% validation

n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")

# ══════════════════════════════════════════════════════════════════════════════
# DATA BATCHING - Grab random chunks of text for training
# ══════════════════════════════════════════════════════════════════════════════

def get_batch(split):
    """
    Goal: Get a batch of (input, target) pairs for training
    
    What's returned:
        x: input sequences  - shape (64, 256) = 64 sequences of 256 characters each
        y: target sequences - shape (64, 256) = what should come next for each character
    
    Changes from v7:
        - Bigger batches (64 instead of 32)
        - Much longer sequences (256 instead of 8)
        → More data per training step, more context per prediction
    """
    source = train_data if split == "train" else val_data

    # Pick 64 random starting positions in the text
    ix = torch.randint(len(source) - block_size, (batch_size,))

    # Extract sequences and their targets (shifted by 1)
    x = torch.stack([source[i     : i + block_size    ] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])

    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """Get stable loss estimates by averaging over 200 batches"""
    out = {}
    model.eval()  # Put model in evaluation mode (disables dropout!)

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y      = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()  # Switch back to training mode (re-enables dropout!)
    return out

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE ATTENTION HEAD - One specialized pattern detector
# ══════════════════════════════════════════════════════════════════════════════

class Head(nn.Module):
    """
    Goal: Let each character "look at" and "gather information from" previous characters
    
    Now with dropout for regularization!
    """

    def __init__(self, head_size):
        super().__init__()
        
        # Communication channels for attention
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        # Causal mask (can only see past, not future)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        # NEW IN V8: Dropout for attention weights
        # Why here? Prevents over-reliance on specific attention patterns
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: (B, T, C) where B=batch, T=time/position, C=channels/embedding_dim
        
        Goal: For each position, gather relevant info from all previous positions
        """
        B, T, C = x.shape

        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Compute attention scores and apply causal mask
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)  # (B, T, T)

        # NEW IN V8: Apply dropout to attention weights
        # During training: randomly zero out 20% of attention weights
        # During inference: keep all weights (dropout is automatically disabled)
        # Why? Forces the model to not rely too heavily on any single token
        wei = self.dropout(wei)
        
        # Weighted aggregation: gather values based on attention weights
        v = self.value(x)  # (B, T, head_size)
        out = wei @ v      # (B, T, head_size)
        
        return out

# ══════════════════════════════════════════════════════════════════════════════
# MULTI-HEAD ATTENTION - Multiple specialized pattern detectors in parallel
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Goal: Let the model examine context from MULTIPLE perspectives simultaneously
    
    Changes in v8:
        - 6 heads instead of 4 (more diverse perspectives)
        - Each head is 64-dimensional (384 / 6 = 64)
        - Added dropout after projection
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        
        # Create 6 independent attention heads (each 64-dimensional)
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        # Projection layer back to n_embd dimensions
        self.proj = nn.Linear(n_embd, n_embd)

        # NEW IN V8: Dropout after projection
        # Why here? Regularizes the multi-head attention output before
        # it's added back to the residual pathway
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Process input through 6 attention heads, then project back
        
        Flow:
            Input (B, T, 384)
               ↓
            6 heads each output (B, T, 64)
               ↓
            Concatenate → (B, T, 384)
               ↓
            Project → (B, T, 384)
               ↓
            Dropout (randomly zero 20%)
               ↓
            Output (B, T, 384)
        """
        # Run all heads in parallel and concatenate
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, 384)
        
        # Project and apply dropout before residual connection
        out = self.dropout(self.proj(out))  # (B, T, 384)
        
        return out

# ══════════════════════════════════════════════════════════════════════════════
# FEEDFORWARD NETWORK - Where the "thinking" happens
# ══════════════════════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """
    Goal: Let each token PROCESS the information it gathered from attention
    
    Architecture with 4x expansion:
        Input (384) → Expand (1536) → ReLU → Project back (384) → Dropout → Output
        
    Changes in v8:
        - Much larger expansion (384 → 1536 instead of 32 → 128)
        - Added dropout at the end
    """

    def __init__(self, n_embd):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand: 384 → 1536 dimensions
            nn.ReLU(),                       # Non-linearity
            nn.Linear(4 * n_embd, n_embd),   # Project back: 1536 → 384
            nn.Dropout(dropout),             # NEW IN V8: Dropout before residual connection
        )
        # Why dropout here?
        #   Applied RIGHT BEFORE adding to residual pathway.
        #   Prevents the network from just copying inputs through without learning.

    def forward(self, x):
        """Process tokens: (B, T, 384) → (B, T, 1536) → (B, T, 384)"""
        return self.net(x)

# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK - Communication + Computation with all optimizations
# ══════════════════════════════════════════════════════════════════════════════

class Block(nn.Module):
    """
    Goal: One complete transformer block with all optimizations
    
    What's inside:
        ✅ Multi-head attention (6 heads of 64 dims each)
        ✅ Feedforward network (4x expansion)
        ✅ Residual connections (gradient superhighway)
        ✅ Layer normalization (stable activations)
        ✅ Dropout (prevents overfitting)
    
    This is the production-ready transformer block!
    """

    def __init__(self, n_embd, n_head):
        """
        Args:
            n_embd: Embedding dimension (384)
            n_head: Number of attention heads (6)
        """
        super().__init__()
        
        head_size = n_embd // n_head  # 384 // 6 = 64 dims per head
        
        # Communication: 6-head attention
        self.sa = MultiHeadAttention(n_head, head_size)
        
        # Computation: feedforward with 4x expansion
        self.ffwd = FeedForward(n_embd)

        # Layer normalization for stable training
        self.ln1 = nn.LayerNorm(n_embd)  # Before attention
        self.ln2 = nn.LayerNorm(n_embd)  # Before feedforward

    def forward(self, x):
        """
        Pre-norm residual connections with dropout
        
        Flow:
            x → LayerNorm → Attention (with dropout) → (+x)
            ↓
            x → LayerNorm → Feedforward (with dropout) → (+x)
            ↓
            output
        """
        # Communication with pre-norm and dropout
        x = x + self.sa(self.ln1(x))  # (B, T, 384)
        
        # Computation with pre-norm and dropout
        x = x + self.ffwd(self.ln2(x))  # (B, T, 384)
        
        return x

# ══════════════════════════════════════════════════════════════════════════════
# THE LANGUAGE MODEL - Scaled up and ready for production!
# ══════════════════════════════════════════════════════════════════════════════

class BigramLanguageModel(nn.Module):
    """
    Goal: Production-scale transformer for character-level text generation
    
    What changed in v8:
        - 6 layers instead of 3 (deeper network)
        - 384-dim embeddings instead of 32 (richer representations)
        - 256-char context instead of 8 (much more context!)
        - Dropout everywhere (better generalization)
        - Configurable depth via n_layer parameter
    
    Model size:
        v7: ~10K parameters (tiny toy model)
        v8: ~10M parameters (small but respectable!)
        
        For comparison:
        - GPT-2 Small: 117M parameters
        - GPT-3: 175B parameters
        
        We're getting closer to real models!
    """

    def __init__(self):
        super().__init__()
        
        # Token embeddings: 65 vocab → 384 dims
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # Position embeddings: 256 positions → 384 dims
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # NEW IN V8: Configurable number of transformer blocks
        # Instead of hard-coding 3 blocks, we use n_layer parameter (6 blocks)
        # The * unpacks the list: nn.Sequential(*[Block1, Block2, ...])
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        
        # Final layer normalization
        self.ln_f = nn.LayerNorm(n_embd)
        
        # Final prediction layer: 384 dims → 65 vocab
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        Process input sequences and make predictions
        
        Args:
            idx: (B, T) - batch of input sequences (now T=256, not 8!)
            targets: (B, T) - correct next characters
        
        Returns:
            logits: (B, T, vocab_size) - prediction scores
            loss: scalar - how wrong we were
        """
        B, T = idx.shape
        
        # Get embeddings
        tok_emb = self.token_embedding_table(idx)  # (B, T, 384)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, 384)
        x = tok_emb + pos_emb  # (B, T, 384)
        
        # Process through 6 transformer blocks
        x = self.blocks(x)  # (B, T, 384)
        
        # Final layer norm
        x = self.ln_f(x)  # (B, T, 384)
        
        # Predictions
        logits = self.lm_head(x)  # (B, T, 65)

        # Calculate loss
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generate new text autoregressively
        
        Changes from v7:
            - Can now look back 256 characters (instead of 8)
            - Much better at maintaining long-range coherence
        """
        for _ in range(max_new_tokens):
            # Crop to block_size (256 in this case)
            idx_cond = idx[:, -block_size:]
            
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

# ══════════════════════════════════════════════════════════════════════════════
# MODEL INITIALIZATION AND TRAINING SETUP
# ══════════════════════════════════════════════════════════════════════════════

model = BigramLanguageModel()
m = model.to(device)

# Lower learning rate for bigger model (3e-4 instead of 1e-3)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════

for iter in range(max_iters):
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter:4d}: train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# ══════════════════════════════════════════════════════════════════════════════
# GENERATION
# ══════════════════════════════════════════════════════════════════════════════

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

"""
═══════════════════════════════════════════════════════════════════════════════
KEY TAKEAWAYS FROM V8 - Scaled Up Production Model
═══════════════════════════════════════════════════════════════════════════════

What We Added:

1. DROPOUT - The regularization technique
   
   What: Randomly set 20% of activations to zero during training
   Where: After attention weights, after projections, after feedforward
   Why: Prevents overfitting by forcing robust feature learning
   
   Training mode: Dropout active (20% randomly zeroed)
   Eval mode: Dropout off (all activations used)
   
   Analogy: Practice with random teammates benched → team is robust

2. SCALING UP - Making the model more powerful
   
   Context: 8 → 256 characters (can see much more!)
   Embeddings: 32 → 384 dims (richer representations)
   Layers: 3 → 6 blocks (deeper reasoning)
   Heads: 4 → 6 (more perspectives)
   
   Each head: 384 / 6 = 64 dimensions
   
   Why these numbers?
       - 256 context: Balance between memory and capability
       - 384 dims: Divisible by 6 (clean head sizes)
       - 6 layers: Deep enough for sophistication, shallow enough to train

3. CONFIGURABLE DEPTH - Using n_layer parameter
   
   Instead of: Hard-coded Block(), Block(), Block()
   We use: *[Block(...) for _ in range(n_layer)]
   
   Benefit: Easy to experiment with different depths!

The Complete Architecture:

Token Embeddings (65 → 384) + Position Embeddings (256 → 384)
    ↓
Block 1: LayerNorm → 6-Head Attention (dropout) → (+) → LayerNorm → FFN (dropout) → (+)
    ↓
Block 2: LayerNorm → 6-Head Attention (dropout) → (+) → LayerNorm → FFN (dropout) → (+)
    ↓
Block 3: LayerNorm → 6-Head Attention (dropout) → (+) → LayerNorm → FFN (dropout) → (+)
    ↓
Block 4: LayerNorm → 6-Head Attention (dropout) → (+) → LayerNorm → FFN (dropout) → (+)
    ↓
Block 5: LayerNorm → 6-Head Attention (dropout) → (+) → LayerNorm → FFN (dropout) → (+)
    ↓
Block 6: LayerNorm → 6-Head Attention (dropout) → (+) → LayerNorm → FFN (dropout) → (+)
    ↓
Final LayerNorm → Linear (384 → 65) → Predictions

This is a REAL transformer that can generate coherent text!

From Toy to Production:
    v1: Bigram (no context) - gibberish
    v2-v4: Basic attention - learning to form words
    v5-v7: Deep transformer - learning grammar
    v8: Scaled transformer - writing convincing Shakespeare! 🎭

What's Left for GPT-Scale?
    - More layers (96 instead of 6)
    - Bigger embeddings (12,288 instead of 384)
    - Better tokenization (BPE instead of characters)
    - Massive data (billions of tokens)
    - Distributed training (multiple GPUs)
    
    But the architecture? WE BUILT IT! 🚀

═══════════════════════════════════════════════════════════════════════════════
"""