"""
═══════════════════════════════════════════════════════════════════════════════
GOAL: Build a DEEP transformer by STACKING transformer blocks
═══════════════════════════════════════════════════════════════════════════════

What we're building:
    A character-level language model with MULTIPLE transformer blocks stacked
    on top of each other, enabling deeper, more sophisticated reasoning.

The Evolution:
    v1: Bigram model - no context, just lookup tables
    v2: Single-head attention - tokens can communicate
    v3: Multi-head attention - tokens communicate from multiple perspectives
    v4: Added feedforward - tokens can THINK about what they learned
    v5: STACK multiple blocks - enable DEEP, hierarchical reasoning ← YOU ARE HERE
    
What's "Interspersing" Communication and Computation?
    
    Interspersing means ALTERNATING these two operations REPEATEDLY:
    - Communication (attention): tokens gather info from each other
    - Computation (feedforward): tokens process that info individually
    
    Instead of doing it once:
        communicate → compute → predict
    
    We do it MULTIPLE times:
        communicate → compute → communicate → compute → communicate → compute → predict
    
    Think of it like a team project:
        Meeting 1: Everyone shares ideas (communicate)
        Work alone: Think about ideas (compute)
        Meeting 2: Share refined thoughts (communicate)
        Work alone: Develop further (compute)
        Meeting 3: Final synthesis (communicate)
        Work alone: Polish your contribution (compute)
        → Present final project

Why Stack Blocks?
    Each block builds increasingly abstract understanding:
    - Block 1: Basic patterns ("'t' often follows 'a'")
    - Block 2: Word-level patterns ("'the' often starts sentences")
    - Block 3: Phrase-level patterns ("'once upon a time' signals a story")
    
    Deeper = more sophisticated reasoning, like building a tall building
    where each floor is built on top of the previous one.

New Architecture:
    Token + Position Embeddings
           ↓
    Block 1: attention → feedforward  ← basic patterns
           ↓
    Block 2: attention → feedforward  ← higher patterns
           ↓
    Block 3: attention → feedforward  ← abstract patterns
           ↓
    Predictions
    
This is the core pattern of ALL transformer models (GPT, BERT, etc.)!

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
        
    Analogy:
        Instead of one detective examining evidence with 32 tools,
        we have 4 detectives with 8 specialized tools each.
        Each finds different clues, then they share all findings.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        # Create multiple independent attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        # Run all heads in parallel, concatenate their outputs along channel dimension
        # Input: (B, T, 32) → 4 heads each output (B, T, 8) → Concat: (B, T, 32)
        return torch.cat([h(x) for h in self.heads], dim=-1)

# ══════════════════════════════════════════════════════════════════════════════
# FEEDFORWARD NETWORK - Where the "thinking" happens
# ══════════════════════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """
    Goal: Let each token PROCESS the information it gathered from attention
    
    The Big Picture:
        Attention = COMMUNICATION (tokens gather info from each other)
        Feedforward = COMPUTATION (tokens think about what they learned)
        
    Why We Need This:
        Attention is just weighted averaging - good at MIXING information,
        but not TRANSFORMING it. The feedforward network adds:
        
        1. Non-linearity (ReLU) - enables complex pattern recognition
        2. Feature extraction - transforms raw context into useful representations
        3. Independent processing - each token "thinks" on its own
        
    Example: Processing "The cat sat"
        After attention: "sat" gathered context about "cat"
        Feedforward thinks: "Hmm, 'cat' is an animal, 'sat' is a motion verb,
                            animals sitting is common → this makes sense"
        
    Key Property: POSITION-WISE
        Each token processes independently. Unlike attention where tokens interact,
        here tokens "think alone" about what they learned.
        
        Analogy: After a group discussion (attention), each person goes home
        and reflects individually on what they heard (feedforward).
    """

    def __init__(self, n_embd):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),  # Transform: mix the 32 dimensions
            nn.ReLU(),                  # Non-linearity: enables complex patterns
        )

    def forward(self, x):
        # Process each token independently: (B, T, n_embd) → (B, T, n_embd)
        # Each token at x[b, t, :] is transformed without looking at other tokens
        return self.net(x)

# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK - The fundamental unit we'll stack ← NEW IN V5!
# ══════════════════════════════════════════════════════════════════════════════

class Block(nn.Module):
    """
    Goal: Package one cycle of communication + computation into a reusable unit
    
    What is a Transformer Block?
        It's a single "round" of the communicate → compute pattern.
        By packaging these together, we can easily stack multiple rounds
        to build deeper networks.
        
    Why is this called "interspersing"?
        Interspersing = alternating/repeating communication and computation
        
        Instead of:     communicate → compute → predict
        We do:          communicate → compute → communicate → compute → ... → predict
        
        Each round refines the understanding further.
    
    Real-world analogy - Writing a research paper:
        Block 1: Read sources (communicate) → Take notes (compute)
        Block 2: Compare notes (communicate) → Form arguments (compute)
        Block 3: Discuss with advisor (communicate) → Revise (compute)
        → Write final draft
        
        Each block builds on previous understanding to create deeper insights.
    
    What happens in each block:
        Input → Self-Attention (gather context) → Feedforward (process) → Output
        
    Architecture:
        x → MultiHeadAttention → feedforward → output (same shape as input)
    """

    def __init__(self, n_embd, n_head):
        """
        Args:
            n_embd: Embedding dimension (32 in our case)
            n_head: Number of attention heads (4 in our case)
        """
        super().__init__()
        
        head_size = n_embd // n_head  # Split 32 dims → 4 heads of 8 dims each
        
        # STEP 1: Communication - tokens gather info from each other
        self.sa = MultiHeadAttention(n_head, head_size)
        
        # STEP 2: Computation - tokens process gathered info independently
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        """
        Process input through one complete transformer block
        
        Args:
            x: (B, T, n_embd) - input token representations
            
        Returns:
            (B, T, n_embd) - refined representations after one round of
                             communication + computation
        
        The flow:
            Input x → Self-Attention (communicate) → Feedforward (compute) → Output
        """
        # STEP 1: COMMUNICATE - tokens gather information from each other
        # Each token looks at all previous tokens and decides what to pay attention to
        x = self.sa(x)  # (B, T, n_embd)
        
        # STEP 2: COMPUTE - tokens independently process what they gathered
        # Each token "thinks" about the information it collected, without interacting
        x = self.ffwd(x)  # (B, T, n_embd)
        
        return x

# ══════════════════════════════════════════════════════════════════════════════
# THE LANGUAGE MODEL - Now with STACKED transformer blocks! ← KEY CHANGE IN V5
# ══════════════════════════════════════════════════════════════════════════════

class BigramLanguageModel(nn.Module):
    """
    Goal: Predict the next character using a DEEP transformer with stacked blocks
    
    What changed in v5:
        v4: Single block (one round of communicate → compute)
        v5: THREE stacked blocks (three rounds of communicate → compute)
        
    Why stack blocks?
        Each block learns increasingly sophisticated patterns:
        - Block 1: Basic character sequences ("qu" often appears together)
        - Block 2: Word-level patterns ("the" often starts sentences)
        - Block 3: Phrase-level patterns ("once upon a time" signals stories)
        
        Stacking enables HIERARCHICAL learning, like building a pyramid:
        Each layer builds on and refines the understanding from below.
    
    New Architecture:
        Token + Position Embeddings
           ↓
        Block 1: communicate → compute  ← learns basic patterns
           ↓
        Block 2: communicate → compute  ← learns higher-level patterns
           ↓
        Block 3: communicate → compute  ← learns abstract patterns
           ↓
        Predictions
        
    This is the CORE pattern used in GPT, BERT, and all modern transformers!
    Real models use 12-96 blocks, but the principle is exactly the same.
    """

    def __init__(self):
        super().__init__()
        
        # Token embeddings: convert character IDs → rich vector representations
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # Position embeddings: give each position its own learned pattern
        # Why? "cat" vs "act" have same characters but different meanings due to position
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # NEW IN V5: Stack 3 transformer blocks for deep processing
        # This replaces the single self.sa_head and self.ffwd from v4
        # Each block does: attention (communicate) → feedforward (compute)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),  # Block 1: basic character patterns
            Block(n_embd, n_head=4),  # Block 2: word-level patterns
            Block(n_embd, n_head=4),  # Block 3: phrase/sentence patterns
        )
        
        # Final layer: convert rich representations → vocabulary predictions
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
        
        # NEW IN V5: Process through 3 stacked transformer blocks
        # Data flows: Block1 → Block2 → Block3
        # Each block refines representations through communicate → compute
        x = self.blocks(x)  # (B, T, n_embd)
        
        # Convert final representations to vocabulary predictions
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # Calculate loss if training
        if targets is None:
            loss = None  # No targets during generation
        else:
            # Reshape for cross_entropy: expects (N, C) not (B, T, C)
            B, T, C = logits.shape
            logits  = logits.view(B * T, C)     # Flatten batch and time
            targets = targets.view(B * T)       # Flatten targets
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Generate new text autoregressively (one character at a time)
        
        Process:
            1. Feed current sequence through all 3 transformer blocks
            2. Look at prediction for next character
            3. Sample one character from the probability distribution
            4. Append it to sequence and repeat
        
        Args:
            idx: (B, T) - starting sequence (e.g., just a newline)
            max_new_tokens: how many characters to generate
        
        Returns:
            idx: (B, T+max_new_tokens) - original + generated text
        """
        for _ in range(max_new_tokens):
            # Crop to block_size (our position embeddings only go up to 8)
            idx_cond = idx[:, -block_size:]
            
            # Get predictions by running through all 3 blocks
            logits, loss = self(idx_cond)  # (B, T, vocab_size)
            
            # Focus on last position (that's our "next character" prediction)
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Convert scores to probabilities and sample
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append sampled character to sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx

# ══════════════════════════════════════════════════════════════════════════════
# MODEL INITIALIZATION AND TRAINING SETUP
# ══════════════════════════════════════════════════════════════════════════════

model = BigramLanguageModel()
m = model.to(device)  # Move to GPU/MPS/CPU

# AdamW optimizer: smart algorithm that adjusts weights to reduce loss
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP - Where the learning happens!
# ══════════════════════════════════════════════════════════════════════════════
# Goal: Repeatedly show the model examples and adjust weights to improve
#
# The process:
#   1. Get a batch of training examples
#   2. Make predictions
#   3. Calculate how wrong we were (loss)
#   4. Compute gradients (how to adjust weights)
#   5. Update weights in the direction that reduces loss
#   6. Repeat 10,000 times!

for iter in range(max_iters):

    # Every 500 steps, check how we're doing on both train and validation data
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter:4d}: train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    # Training step
    xb, yb = get_batch('train')              # 1. Get batch of data
    logits, loss = model(xb, yb)             # 2. Forward pass: compute predictions and loss
    optimizer.zero_grad(set_to_none=True)    # 3. Clear old gradients
    loss.backward()                          # 4. Backward pass: compute new gradients
    optimizer.step()                         # 5. Update weights

# ══════════════════════════════════════════════════════════════════════════════
# GENERATION - Let's see what our model learned!
# ══════════════════════════════════════════════════════════════════════════════
# Start with a single newline character and generate 500 new characters

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

"""
═══════════════════════════════════════════════════════════════════════════════
IMPORTANT NOTE: Deep Networks Have Optimization Problems!
═══════════════════════════════════════════════════════════════════════════════

Why This Code Doesn't Train Well Yet:

We've entered DEEP NETWORK territory (3 blocks = 6+ layers). Deep networks
suffer from the VANISHING/EXPLODING GRADIENT problem:

The Problem:
    When gradients flow backward through many layers during training:
    - They can VANISH (become too small → early layers don't learn)
    - They can EXPLODE (become too large → training becomes unstable)
    
    It's like playing telephone with 10 people - the message gets garbled!

Why This Happens:
    Gradients are computed by multiplying values across layers.
    - If values < 1: multiply many times → vanishes to ~0
    - If values > 1: multiply many times → explodes to infinity
    
    With 6+ layers, small errors compound catastrophically.

The Solution (Coming in v6):
    RESIDUAL CONNECTIONS (also called "skip connections")
    
    These create "shortcut highways" that allow gradients to flow directly
    through the network, bypassing the problematic multiplications.
    
    This breakthrough from ResNet (2015) is what made deep learning DEEP!
    Without it, we couldn't train networks with 100+ layers (like GPT-3).

The Next Step:
    In v6, we'll add two critical components from the Transformer paper:
    1. Residual connections - solve gradient flow problems
    2. Layer normalization - stabilize training
    
    These will transform our struggling 3-block network into a properly
    trainable deep transformer!

Key Takeaway:
    Stacking blocks = good for learning complex patterns (depth)
    But depth alone isn't enough - we need residual connections to actually
    train effectively.

═══════════════════════════════════════════════════════════════════════════════
"""