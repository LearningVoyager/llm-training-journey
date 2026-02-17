"""
═══════════════════════════════════════════════════════════════════════════════
GOAL: Add LAYER NORMALIZATION for stable, efficient training
═══════════════════════════════════════════════════════════════════════════════

What we're building:
    A transformer with both residual connections AND layer normalization,
    giving us a production-ready architecture that trains smoothly at depth.

The Problem We're Solving (from v6):
    Even with residual connections, deep networks can be unstable. Different
    neurons can have wildly different activation scales, making optimization
    difficult.
    
    Think of it like a team where some people shout and others whisper - it's
    hard to have a productive conversation!

The Solution: LAYER NORMALIZATION
    
    The Big Idea:
        For each token, normalize its features to have mean=0 and std=1.
        This ensures all neurons operate at similar scales.
        
        Before LayerNorm: Features might be [100, -50, 0.1, 200, -5, ...]
        After LayerNorm:  Features become  [0.5, -1.2, 0.0, 1.8, -0.4, ...]
        
        All values are now in a similar range, making learning more stable!
    
    How it works:
        For each token's 32-dimensional vector:
        1. Compute mean across the 32 dimensions
        2. Compute std across the 32 dimensions  
        3. Normalize: (x - mean) / std
        4. Scale and shift with learned parameters (γ and β)
    
    Why This Helps:
        ✅ Stabilizes training (reduces "covariate shift")
        ✅ Allows higher learning rates (faster training)
        ✅ Acts as regularization (reduces overfitting)
        ✅ Makes the network less sensitive to initialization

Layer Norm vs Batch Norm:
    
    Batch Normalization:
        Normalizes ACROSS the batch dimension (across different examples)
        - Computes mean/std over all 32 examples in the batch
        - Problem: Depends on batch size, doesn't work well for small batches
        - Used in: CNNs (computer vision)
    
    Layer Normalization:
        Normalizes ACROSS the feature dimension (within each example)
        - Computes mean/std over the 32 features for EACH token independently
        - Benefit: Works regardless of batch size
        - Used in: Transformers (NLP)
    
    Visual comparison (one token):
        Batch Norm: Look at this feature across all examples → normalize
        Layer Norm: Look at all features for this example → normalize

Pre-Norm vs Post-Norm:
    
    Original Transformer Paper (Post-Norm):
        x → [attention] → (+x) → LayerNorm → [ffwd] → (+x) → LayerNorm
        Normalize AFTER the residual connection
    
    Modern Practice (Pre-Norm) ← What we're implementing:
        x → LayerNorm → [attention] → (+x) → LayerNorm → [ffwd] → (+x)
        Normalize BEFORE the transformation
    
    Why Pre-Norm is Better:
        ✅ More stable training (gradients flow better)
        ✅ Easier to train very deep networks
        ✅ Less sensitive to learning rate
        ✅ No need for learning rate warmup
        
        Post-norm was used in the original paper, but later research showed
        pre-norm works better in practice!

Changes in v7:
    1. ✅ Added LayerNorm before attention (self.ln1)
    2. ✅ Added LayerNorm before feedforward (self.ln2)
    3. ✅ Added final LayerNorm before prediction head
    4. ✅ Using pre-norm formulation (normalize before transformations)

Architecture Pattern:
    x → LayerNorm → [attention] → (+x) → LayerNorm → [feedforward] → (+x)
    
    Each transformation now receives normalized inputs, making learning stable!

Real-World Analogy:
    Without LayerNorm: A classroom where students speak at wildly different volumes
    - Teacher struggles to hear quiet students, gets overwhelmed by loud ones
    
    With LayerNorm: Everyone speaks at a normalized volume
    - Teacher can understand everyone equally, classroom runs smoothly

═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS - The knobs we can turn to change how the model learns
# ══════════════════════════════════════════════════════════════════════════════

batch_size    = 32    # Process 32 different text sequences at once (faster training)
block_size    = 8     # Each sequence is 8 characters long (the "context window")
max_iters     = 10000 # Train for 10,000 steps total
eval_interval = 500   # Every 500 steps, check how well we're doing
learning_rate = 1e-3  # Small steps = stable learning (attention is sensitive!)
eval_iters    = 200   # Average 200 batches to get reliable loss estimates
n_embd        = 32    # Each character becomes a 32-dimensional vector
                      # (larger = more expressive, but slower and needs more data)

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
    
    Why batches? Training on 1 example at a time is slow. Processing 32 examples
    at once (batch_size=32) is much faster on GPUs.
    
    What's returned:
        x: input sequences  - shape (32, 8) = 32 sequences of 8 characters each
        y: target sequences - shape (32, 8) = what should come next for each character
    
    Example for ONE sequence:
        If text is "hello world"
        x = "hello wo"  (8 characters)
        y = "ello wor"  (same thing, shifted right by 1)
        
        The model learns: given 'h', predict 'e'
                         given 'e', predict 'l'
                         given 'l', predict 'l'
                         ... and so on
    """
    source = train_data if split == "train" else val_data

    # Pick 32 random starting positions in the text
    ix = torch.randint(len(source) - block_size, (batch_size,))

    # Extract sequences and their targets (shifted by 1)
    x = torch.stack([source[i     : i + block_size    ] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])

    # Move to GPU/MPS/CPU (wherever the model is)
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()  # Don't track gradients here - saves memory, speeds things up
def estimate_loss():
    """
    Goal: Get an accurate measurement of how well the model is doing
    
    Why? Loss on a single batch is noisy - could be an easy or hard batch by luck.
    Averaging over 200 batches gives us a stable, trustworthy number.
    
    Returns: {'train': 2.34, 'val': 2.41}
        - Lower is better (means the model is more confident in its predictions)
        - If val loss >> train loss → model is memorizing (overfitting)
        - If val loss ≈ train loss → model is genuinely learning
    """
    out = {}
    model.eval()  # Put model in evaluation mode

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y      = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()  # Switch back to training mode
    return out

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE ATTENTION HEAD - One specialized pattern detector
# ══════════════════════════════════════════════════════════════════════════════

class Head(nn.Module):
    """
    Goal: Let each character "look at" and "gather information from" previous characters
    
    The big idea (self-attention):
        Instead of treating each character in isolation, we let characters 
        communicate with each other. Each position asks: "Which previous 
        characters should I pay attention to?"
        
    Example: In "Harry Potter ate"
        - When processing "ate", the model can look back at "Harry Potter"
        - It learns: "ate" often follows nouns (subjects like "Harry Potter")
        - This context makes predictions much smarter
    
    How it works (3 steps):
        1. Query: "What am I looking for?"
        2. Key: "What do I contain?"
        3. Value: "What information do I offer?"
        
        Each character compares its Query with all previous Keys to decide
        which Values to gather.
    """

    def __init__(self, head_size):
        super().__init__()
        
        # These are the "communication channels" for attention
        # They transform the n_embd-dimensional input into head_size dimensions
        self.key   = nn.Linear(n_embd, head_size, bias=False)  # What do I contain?
        self.query = nn.Linear(n_embd, head_size, bias=False)  # What am I looking for?
        self.value = nn.Linear(n_embd, head_size, bias=False)  # What do I offer?
        
        # Create a lower-triangular mask for "causal" attention
        # Why? We can only look at PAST characters, never future ones
        # (otherwise the model would "cheat" during training)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """
        x: (B, T, C) where B=batch, T=time/position, C=channels/embedding_dim
        
        Goal: For each position, gather relevant info from all previous positions
        """
        B, T, C = x.shape

        k = self.key(x)    # (B, T, head_size) "Here's what I contain"
        q = self.query(x)  # (B, T, head_size) "Here's what I'm looking for"

        # Compute attention scores ("affinities") and apply causal mask
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)  # (B, T, T), scaled for stability
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Hide future
        wei = F.softmax(wei, dim=-1)  # (B, T, T) Convert to probabilities
        
        # Weighted aggregation: gather values based on attention weights
        v = self.value(x)  # (B, T, head_size) "Here's my information"
        out = wei @ v      # (B, T, head_size) "Weighted mix of relevant info"
        
        return out

# ══════════════════════════════════════════════════════════════════════════════
# MULTI-HEAD ATTENTION - Multiple specialized pattern detectors in parallel
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Goal: Let the model examine context from MULTIPLE perspectives simultaneously
    
    Why multiple heads?
        Language is rich and multifaceted. A single head can only learn ONE type
        of pattern, but we need to capture many different aspects:
        - Grammatical structure (subject-verb agreement)
        - Semantic relationships (synonyms, antonyms)
        - Positional patterns (what typically comes first/last)
        - Contextual nuances (tone, formality)
        
    How it works:
        1. Split embedding into 4 heads: 32 dims → 4 heads × 8 dims each
        2. Each head runs self-attention independently (in PARALLEL on GPU)
        3. Concatenate outputs: 4 × 8 dims → 32 dims total
        4. Project back to n_embd dimensions
        
    Analogy:
        Instead of one detective examining evidence with 32 tools,
        we have 4 detectives with 8 specialized tools each.
        Each finds different clues, then they share all findings.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        
        # Create multiple independent attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        # Projection layer - transforms concatenated heads back to residual pathway
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        """
        Process input through multiple attention heads, then project back
        
        Args:
            x: (B, T, n_embd) - input representations
            
        Returns:
            (B, T, n_embd) - attention output ready to add to residual pathway
        """
        # Run all heads in parallel and concatenate outputs
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B, T, 32)
        
        # Project back before adding to residual pathway
        out = self.proj(out)  # (B, T, n_embd)
        
        return out

# ══════════════════════════════════════════════════════════════════════════════
# FEEDFORWARD NETWORK - Where the "thinking" happens (with 4x expansion)
# ══════════════════════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """
    Goal: Let each token PROCESS the information it gathered from attention
    
    The Big Picture:
        Attention = COMMUNICATION (tokens gather info from each other)
        Feedforward = COMPUTATION (tokens think about what they learned)
        
    Architecture (with 4x expansion):
        Input (32) → Expand (128) → ReLU → Project back (32) → Output
        
        This "bottleneck" structure gives the network more capacity to learn
        complex transformations, then compresses back to fit residual pathway.
    """

    def __init__(self, n_embd):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # Expand: 32 → 128 dimensions
            nn.ReLU(),                      # Non-linearity
            nn.Linear(4 * n_embd, n_embd),  # Project back: 128 → 32
        )

    def forward(self, x):
        """Process tokens: (B, T, 32) → (B, T, 128) → (B, T, 32)"""
        return self.net(x)

# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK - Now with LAYER NORMALIZATION! ← NEW IN V7
# ══════════════════════════════════════════════════════════════════════════════

class Block(nn.Module):
    """
    Goal: Package communication + computation with residual connections AND layer norm
    
    What changed in v7:
        v6: x = x + self.sa(x)           ← unstable activations
        v7: x = x + self.sa(self.ln1(x)) ← normalized inputs to attention!
        
        We now normalize BEFORE each transformation (pre-norm formulation)
    
    Why Two Layer Norms?
        
        We normalize before BOTH transformations in the block:
        1. self.ln1: Normalizes before attention (communication)
        2. self.ln2: Normalizes before feedforward (computation)
        
        Why separate?
            Each transformation gets its own normalized input. This ensures:
            - Attention receives features with consistent scale
            - Feedforward receives features with consistent scale
            - Both can learn effectively without interference
        
        Think of it like: Before each meeting (attention) and each work session
        (feedforward), everyone resets to speak at normal volume (normalization)
    
    Pre-Norm Pattern (what we're using):
        
        x → LayerNorm → [attention] → (+x) ← residual pathway
        ↓
        x → LayerNorm → [feedforward] → (+x) ← residual pathway
        ↓
        output
        
        Benefits:
            ✅ More stable gradient flow
            ✅ Easier to train deep networks
            ✅ No need for learning rate warmup
            ✅ Less sensitive to hyperparameters
    
    How Layer Norm Works (for each token):
        
        Input: [2.5, 100, -50, 0.1, ...]  (32 numbers with wild scales)
        ↓
        Step 1: Compute mean across 32 dimensions
            mean = sum(all 32 values) / 32
        ↓
        Step 2: Compute std across 32 dimensions  
            std = sqrt(variance of all 32 values)
        ↓
        Step 3: Normalize
            normalized = (x - mean) / std
        ↓
        Step 4: Scale and shift (learnable γ and β)
            output = γ * normalized + β
        ↓
        Output: [0.5, 1.2, -0.8, 0.0, ...]  (32 numbers with similar scale)
        
        Now all features are at similar magnitudes, making learning stable!
    """

    def __init__(self, n_embd, n_head):
        """
        Args:
            n_embd: Embedding dimension (32 in our case)
            n_head: Number of attention heads (4 in our case)
        """
        super().__init__()
        
        head_size = n_embd // n_head  # Split 32 dims → 4 heads of 8 dims each
        
        # Communication: multi-head attention with projection
        self.sa = MultiHeadAttention(n_head, head_size)
        
        # Computation: feedforward with 4x expansion
        self.ffwd = FeedForward(n_embd)

        # NEW IN V7: Layer normalization for stable training
        # Each LayerNorm normalizes across the n_embd (32) dimension
        # This means for each token, we normalize its 32 feature values to mean=0, std=1
        self.ln1 = nn.LayerNorm(n_embd)  # Normalize before attention
        self.ln2 = nn.LayerNorm(n_embd)  # Normalize before feedforward

    def forward(self, x):
        """
        Process input with PRE-NORM residual connections
        
        Args:
            x: (B, T, n_embd) - input token representations
            
        Returns:
            (B, T, n_embd) - refined representations
        
        The pre-norm flow:
            x → LayerNorm → [attention] → (+x) → LayerNorm → [feedforward] → (+x)
        
        KEY CHANGE FROM V6 (Pre-norm vs no norm):
            v6: x = x + self.sa(x)           ← attention receives raw x
            v7: x = x + self.sa(self.ln1(x)) ← attention receives normalized x
        """
        
        # STEP 1: Communication with pre-norm
        # Normalize, apply attention, add back to residual pathway
        x = x + self.sa(self.ln1(x))  # (B, T, n_embd)
        # What happens: ln1 normalizes x → sa processes normalized input → result added to original x
        
        # STEP 2: Computation with pre-norm
        # Normalize, apply feedforward, add back to residual pathway  
        x = x + self.ffwd(self.ln2(x))  # (B, T, n_embd)
        # What happens: ln2 normalizes x → ffwd processes normalized input → result added to original x
        
        # Result: x has been enriched by normalized transformations
        # while preserving the residual pathway
        return x

# ══════════════════════════════════════════════════════════════════════════════
# THE LANGUAGE MODEL - Now with full layer normalization!
# ══════════════════════════════════════════════════════════════════════════════

class BigramLanguageModel(nn.Module):
    """
    Goal: Predict the next character using a DEEP, STABLE transformer
    
    What changed in v7:
        v6: Had residual connections, but activations could still be unstable
        v7: Added layer normalization everywhere for maximum stability
        
    Why this matters:
        v6 solved gradient flow (residual connections)
        v7 solves activation stability (layer normalization)
        
        Together, these create a transformer that trains smoothly even at
        great depth. This is the production-ready architecture!
    
    The Complete Transformer Stack:
        
        Embeddings (token + position)
           ↓
        Block 1: LayerNorm → Attention → (+) → LayerNorm → Feedforward → (+)
           ↓
        Block 2: LayerNorm → Attention → (+) → LayerNorm → Feedforward → (+)
           ↓  
        Block 3: LayerNorm → Attention → (+) → LayerNorm → Feedforward → (+)
           ↓
        Final LayerNorm ← NEW: Normalize before final predictions
           ↓
        Linear → Predictions
        
        Every transformation receives normalized inputs!
    """

    def __init__(self):
        super().__init__()
        
        # Token embeddings: convert character IDs → rich vector representations
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # Position embeddings: give each position its own learned pattern
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Stack 3 transformer blocks with residual connections + layer norm
        # Then add a final layer norm before predictions
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),  # Block 1: basic patterns
            Block(n_embd, n_head=4),  # Block 2: higher patterns
            Block(n_embd, n_head=4),  # Block 3: abstract patterns
            nn.LayerNorm(n_embd),     # NEW IN V7: Final normalization
        )
        # Why final LayerNorm?
        #   After all blocks, we normalize once more before making predictions.
        #   This ensures the prediction head receives well-scaled inputs,
        #   making training more stable and predictions more reliable.
        
        # Final layer: convert normalized representations → vocabulary predictions
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        Process input sequences and make predictions
        
        Args:
            idx: (B, T) - batch of input character sequences
            targets: (B, T) - correct next characters (only during training)
        
        Returns:
            logits: (B, T, vocab_size) - prediction scores for each position
            loss: scalar - how wrong our predictions were (lower = better)
        """
        B, T = idx.shape
        
        # Get token and position embeddings, combine them
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd) "what + where"
        
        # Process through 3 transformer blocks + final layer norm
        # Each block: LayerNorm → Attention → (+) → LayerNorm → Feedforward → (+)
        # Final: LayerNorm once more before predictions
        x = self.blocks(x)  # (B, T, n_embd)
        
        # Convert normalized representations to vocabulary predictions
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Calculate loss if training
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits  = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generate new text autoregressively (one character at a time)"""
        for _ in range(max_new_tokens):
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

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP - Where the learning happens!
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
# GENERATION - Let's see what our model learned!
# ══════════════════════════════════════════════════════════════════════════════

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

"""
═══════════════════════════════════════════════════════════════════════════════
KEY TAKEAWAYS FROM V7 - Production-Ready Transformer Architecture!
═══════════════════════════════════════════════════════════════════════════════

What We Built:
    ✅ Complete decoder-only transformer (like GPT)
    ✅ Multi-head self-attention (multiple perspectives)
    ✅ Feedforward networks (4x expansion for capacity)
    ✅ Residual connections (gradient superhighway)
    ✅ Layer normalization (stable activations)
    ✅ Pre-norm formulation (modern best practice)

The Two Critical Optimizations:
    
    1. Residual Connections (v6):
       - Solve: Vanishing/exploding gradients
       - How: Create direct gradient paths via addition
       - Impact: Makes deep networks trainable
    
    2. Layer Normalization (v7):
       - Solve: Unstable activation scales
       - How: Normalize features to mean=0, std=1
       - Impact: Makes training smooth and efficient

Pre-Norm vs Post-Norm:
    
    Post-Norm (Original Transformer Paper):
        x → [transform] → (+x) → LayerNorm
        Problem: Less stable for very deep networks
    
    Pre-Norm (Modern Practice - What We Use):
        x → LayerNorm → [transform] → (+x)
        Benefits: More stable, easier to train, no warmup needed

Why This Is a Complete Architecture:
    
    This v7 code contains ALL the essential components of a modern transformer:
    - Token + position embeddings
    - Multi-head attention (communication)
    - Feedforward networks (computation)
    - Residual connections (optimization)
    - Layer normalization (stability)
    
    The only things missing for a production model like GPT:
    - More layers (GPT-3 has 96 blocks instead of 3)
    - Bigger dimensions (GPT-3 uses 12,288 instead of 32)
    - Better tokenization (BPE instead of character-level)
    - More data (billions of tokens instead of 1 million)
    
    But the ARCHITECTURE is identical!

This Is a Decoder-Only Transformer:
    
    We built a DECODER (generates text autoregressively).
    This is the same architecture as GPT, not BERT or the original Transformer.
    
    Key feature: Causal masking (can only see past, not future)
    Use case: Text generation, completion, chatbots

What We Learned:
    
    The journey from bigram → full transformer:
    v1: Bigram (no context)
    v2: Single-head attention (basic communication)
    v3: Multi-head attention (multiple perspectives)
    v4: + Feedforward (computation)
    v5: + Stacked blocks (depth)
    v6: + Residual connections (trainability)
    v7: + Layer normalization (stability)
    
    Each addition built on the previous, teaching us WHY each component matters!

═══════════════════════════════════════════════════════════════════════════════
"""