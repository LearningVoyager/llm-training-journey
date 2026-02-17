"""
═══════════════════════════════════════════════════════════════════════════════
GOAL: Add RESIDUAL CONNECTIONS to make deep networks trainable
═══════════════════════════════════════════════════════════════════════════════

What we're building:
    A deep transformer that can ACTUALLY TRAIN effectively by adding "gradient
    superhighways" that allow learning signals to flow freely through all layers.

The Problem We're Solving (from v5):
    Deep networks suffer from vanishing/exploding gradients. When you stack
    many layers, gradients either disappear to zero or explode to infinity
    during backpropagation, making the network impossible to train.
    
    Think of it like a game of telephone with 20 people - by the end, the
    message is completely garbled!

The Solution: RESIDUAL CONNECTIONS (Skip Connections)
    
    Invented in the ResNet paper (2015), this simple idea revolutionized deep learning.
    
    The Big Idea:
        Instead of:  x → [transform] → output
        We do:       x → [transform] → output + x
        
        We ADD the input back to the output. The "+" creates a direct path for
        gradients to flow backward unchanged.
    
    Why This Works - The Gradient Superhighway:
        
        During forward pass:
            Input → Block1(+) → Block2(+) → Block3(+) → Output
                     ↓           ↓           ↓
                  transform   transform   transform
        
        During backward pass (gradient flow):
            Input ←←←←←←←←←←←←←←←←←←←←←←←←←←←← Output
                 ↖         ↖         ↖
               also flows  also flows also flows
               through     through    through
               transforms  transforms transforms
        
        The "+" distributes gradients EQUALLY to both branches:
        - Main path: gradients flow directly backward (gradient superhighway!)
        - Side path: gradients also flow through the transformations
        
        This means gradients can reach ALL layers without vanishing!
    
    Real-World Analogy:
        Without residual connections: A mountain road with only one winding path
        - Trucks (gradients) get slower and smaller as they climb
        
        With residual connections: A mountain road with a straight highway PLUS
        scenic routes that connect back
        - Trucks can zoom up the highway, ensuring all towns (layers) get supplies
        - Scenic routes (transformations) add extra value along the way

Changes in v6:
    1. ✅ Added residual connections (x + transformation) in Block
    2. ✅ Added projection layers after attention and in feedforward
    3. ✅ Expanded feedforward to 4x channels (32 → 128 → 32)
    
    Coming in v7:
    - Layer Normalization (stabilizes training further)

New Architecture Pattern:
    x → [attention] → (+) ← x  (residual connection)
         ↓
    x → [feedforward] → (+) ← x  (residual connection)
    
    The input x is ADDED back at each step, creating a "residual pathway"

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
        4. NEW IN V6: Project back to n_embd dimensions
        
    Analogy:
        Instead of one detective examining evidence with 32 tools,
        we have 4 detectives with 8 specialized tools each.
        Each finds different clues, then they share all findings.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        
        # Create multiple independent attention heads
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        # NEW IN V6: Projection layer - transforms concatenated heads back to residual pathway
        # Why? This is the "gate" that controls how much the attention output
        # contributes to the residual pathway. Initialized near zero, it allows
        # the network to gradually learn to use attention during training.
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
        # 4 heads each output (B, T, 8) → concatenate to (B, T, 32)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        
        # Project back to n_embd dimensions before adding to residual pathway
        # This linear transformation lets the network control how much to use
        # the attention output vs the original input
        out = self.proj(out)  # (B, T, n_embd)
        
        return out

# ══════════════════════════════════════════════════════════════════════════════
# FEEDFORWARD NETWORK - Where the "thinking" happens (now with expansion!)
# ══════════════════════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """
    Goal: Let each token PROCESS the information it gathered from attention
    
    The Big Picture:
        Attention = COMMUNICATION (tokens gather info from each other)
        Feedforward = COMPUTATION (tokens think about what they learned)
        
    What changed in v6:
        We now EXPAND to 4x channels (32 → 128 → 32) following the Transformer paper.
        
        Why 4x expansion?
            - Gives the network MORE CAPACITY to learn complex transformations
            - The middle layer (128 dims) can learn richer feature combinations
            - Then we PROJECT BACK to 32 dims to fit the residual pathway
            
        Think of it like brainstorming:
            - Start with your notes (32 ideas)
            - Expand to many possibilities (128 ideas)
            - Distill back to key insights (32 refined ideas)
    
    Architecture:
        Input (32) → Expand (128) → ReLU → Project back (32) → Output
        
        This is called a "bottleneck" structure - expand, process, compress.
    """

    def __init__(self, n_embd):
        super().__init__()
        
        self.net = nn.Sequential(
            # EXPAND: 32 → 128 dimensions (gives more capacity to learn)
            nn.Linear(n_embd, 4 * n_embd),
            
            # Non-linearity: enables complex pattern recognition
            nn.ReLU(),
            
            # NEW IN V6: PROJECT BACK to original dimension (128 → 32)
            # Why? We need to return to n_embd size to add to the residual pathway
            # This projection also acts as a "gate" controlling contribution
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        """
        Process tokens through expansion, transformation, and projection
        
        Args:
            x: (B, T, n_embd) - input representations
            
        Returns:
            (B, T, n_embd) - transformed output ready for residual connection
            
        Flow:
            (B, T, 32) → expand → (B, T, 128) → ReLU → (B, T, 128) 
                      → project → (B, T, 32)
        """
        return self.net(x)

# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK - Now with RESIDUAL CONNECTIONS! ← KEY CHANGE IN V6
# ══════════════════════════════════════════════════════════════════════════════

class Block(nn.Module):
    """
    Goal: Package communication + computation with RESIDUAL CONNECTIONS
    
    What are residual connections (skip connections)?
        
        Instead of:  x → [transform] → output
        We do:       x → [transform] → output + x
        
        We ADD the original input back to the transformed output!
    
    Why is this revolutionary?
        
        The Gradient Superhighway:
            Without residual connections:
                Gradients must flow THROUGH all transformations
                → They can vanish (become too small) or explode
                → Deep networks can't train
            
            With residual connections:
                Gradients can flow DIRECTLY backward via the "+" operation
                → They bypass transformations and flow unimpeded
                → Deep networks train effectively!
        
        The "+" distributes gradients equally to both branches:
            gradient → 50% flows directly backward (superhighway)
                    → 50% also flows through transformation (learning still happens)
    
    Real-World Analogy:
        Building a company:
            Without residuals: Every decision must go through every department
            → Slow, information gets lost, nothing gets done
            
            With residuals: Direct communication channels PLUS departmental processing
            → Fast decision-making AND specialized work both happen
    
    The Residual Pathway:
        
        Input x
          ↓
          ├──→ [Attention] ──→ (+) ← x feeds directly here
          ↓                      ↓
          ├──→ [Feedforward] ──→ (+) ← x feeds directly here  
          ↓                      ↓
        Output
        
        The input "forks off" to be transformed, then "merges back" via addition.
        This creates an unobstructed path from input to output.
    
    Initialization:
        At the start of training, the transformations are initialized to contribute
        very little (near zero). This means initially:
            output ≈ x + 0 = x
        
        Over time, the transformations "come online" and gradually contribute more.
        This gentle start makes training stable!
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

    def forward(self, x):
        """
        Process input with residual connections
        
        Args:
            x: (B, T, n_embd) - input token representations
            
        Returns:
            (B, T, n_embd) - refined representations
        
        The flow with residual connections:
            x → [attention] → (+ x) → [feedforward] → (+ x) → output
            
        KEY CHANGE FROM V5:
            v5: x = self.sa(x)        ← x is replaced
            v6: x = x + self.sa(x)    ← x is preserved and added to
        """
        
        # RESIDUAL CONNECTION 1: Communication
        # Fork off: compute attention, then merge back via addition
        # x is preserved on the "main highway" while self.sa(x) adds insights
        x = x + self.sa(x)  # (B, T, n_embd)
        
        # RESIDUAL CONNECTION 2: Computation  
        # Fork off: compute feedforward, then merge back via addition
        # x continues on the "main highway" while self.ffwd(x) adds refinements
        x = x + self.ffwd(x)  # (B, T, n_embd)
        
        # Result: x has been enriched by attention and feedforward,
        # but also retains a direct path from the original input
        return x

# ══════════════════════════════════════════════════════════════════════════════
# THE LANGUAGE MODEL - Now trainable at depth with residual connections!
# ══════════════════════════════════════════════════════════════════════════════

class BigramLanguageModel(nn.Module):
    """
    Goal: Predict the next character using a DEEP, TRAINABLE transformer
    
    What changed in v6:
        v5: Stacked blocks, but training was unstable (vanishing gradients)
        v6: Same structure, but with residual connections for stable training
        
    Why this matters:
        v5 had the right architecture but couldn't train effectively.
        v6 adds the critical ingredient (residual connections) that makes
        deep learning actually DEEP.
        
        Now our 3-block network can train as effectively as a shallow one,
        while learning much more sophisticated patterns!
    
    The Residual Pathway Through the Whole Model:
        
        Embeddings (x)
           ↓
        Block 1: x + [attention + feedforward]
           ↓
        Block 2: x + [attention + feedforward]
           ↓  
        Block 3: x + [attention + feedforward]
           ↓
        Predictions
        
        Gradients can flow directly from predictions → Block 3 → Block 2 → 
        Block 1 → Embeddings without getting stuck!
    """

    def __init__(self):
        super().__init__()
        
        # Token embeddings: convert character IDs → rich vector representations
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # Position embeddings: give each position its own learned pattern
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Stack 3 transformer blocks (each with residual connections)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),  # Block 1: basic patterns
            Block(n_embd, n_head=4),  # Block 2: higher patterns
            Block(n_embd, n_head=4),  # Block 3: abstract patterns
        )
        
        # Final layer: convert representations → vocabulary predictions
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
        
        # Process through 3 transformer blocks with residual connections
        # Now training is stable because gradients can flow freely!
        x = self.blocks(x)  # (B, T, n_embd)
        
        # Convert final representations to vocabulary predictions
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
KEY TAKEAWAYS FROM V6
═══════════════════════════════════════════════════════════════════════════════

What We Achieved:
    ✅ Deep networks can now TRAIN effectively
    ✅ Gradients flow freely through all layers (gradient superhighway)
    ✅ Network starts gentle (near identity) and gradually learns
    ✅ Stable optimization even with 3+ blocks

How Residual Connections Work:
    1. Addition creates direct gradient paths (superhighway)
    2. Transformations "fork off" and "merge back" 
    3. Network learns gradually (transformations start at ~zero)
    4. Both direct path AND transformations contribute to learning

The Three Key Changes:
    1. x = x + self.sa(x)     ← residual connection in attention
    2. x = x + self.ffwd(x)   ← residual connection in feedforward
    3. Projection layers       ← control contribution to residual pathway

Coming in v7:
    Layer Normalization - further stabilizes training by normalizing
    activations within each layer. This complements residual connections
    to create the final, production-ready transformer architecture!

Why This Matters:
    This breakthrough (ResNet 2015) is WHY we can train GPT-3 (96 layers),
    GPT-4 (hundreds of layers), and other massive models. Without residual
    connections, we'd be stuck with shallow networks that can't learn
    complex patterns.

═══════════════════════════════════════════════════════════════════════════════
"""