"""
═══════════════════════════════════════════════════════════════════════════════
GOAL OF THIS FILE: Add COMPUTATION to our transformer (Feedforward Networks)

Intutive Idea: Without feedforward, attention is like a student who collected notes from classmates but never actually studied them!
═══════════════════════════════════════════════════════════════════════════════

The Evolution So Far:
    v1: Bigram model - no context, just lookup tables
    v2: Added single-head attention - tokens can communicate
    v3: Added multi-head attention - tokens communicate from multiple perspectives
    v4: Added feedforward networks - tokens can THINK about what they learned
    
The Problem We're Solving:
    In v3, tokens could gather information from each other (communication),
    but they went straight to making predictions without processing that info.
    
    Analogy: It's like collecting research for an essay but never actually
    analyzing it - you just dump the raw notes onto the page!
    
The Solution: Two-Stage Processing
    1. COMMUNICATION (Multi-Head Attention): "What info should I gather?"
    2. COMPUTATION (Feedforward): "Now let me think about what I gathered"
    
Why This Works:
    - Attention is great at AGGREGATING information (weighted averaging)
    - Feedforward is great at TRANSFORMING information (complex processing)
    - Together, they give tokens both awareness AND intelligence

Architecture Pattern (this is the CORE of transformers):
    Token Embeddings + Position Embeddings
           ↓
    Multi-Head Attention  ← tokens communicate, gather context
           ↓
    Feedforward Network   ← tokens think independently about what they learned
           ↓
    Predictions

** The Big Idea: Communication vs Computation
Your Insight is Perfect:

"Tokens looked at each other but didn't have time to think on what they found"

This is exactly right! Here's what's happening:
Before (v3 - only attention):
Token: "I just gathered info from my neighbors... but what do I DO with it?"
Model: "Sorry, time to make a prediction now!"
Token: "But I didn't process anything! 😰"
After (v4 - attention + feedforward):
Token: "I gathered info from neighbors (attention)"
Token: "Now let me THINK about what I learned (feedforward)"
Token: "Okay, NOW I'm ready to predict!"

📊 Two-Stage Process
Stage 1: Communication (Multi-Head Attention)

Tokens talk to each other
They share information
Result: Each token gathers context from the past
Happens across all tokens (they interact)

Stage 2: Computation (Feedforward Network)

Each token thinks independently
They process what they learned
Result: Rich, transformed representations
Happens per-token (no interaction between tokens)


** Why Did This Happen? **
Why didn't tokens have time to think before?
Attention is a weighted average operation. It's like:

Taking 20% of Token A's info
Plus 30% of Token B's info
Plus 50% of Token C's info
= Mixed information

But it's just mixing, not transforming! There's no non-linear processing, no complex computation. The feedforward network adds:

Non-linearity (ReLU) - enables complex patterns
Transformation - lets the model extract features
"Thinking time" - processes the gathered information



    


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

        # Step 1: Each position creates its query and key
        k = self.key(x)    # (B, T, head_size) "Here's what I contain"
        q = self.query(x)  # (B, T, head_size) "Here's what I'm looking for"

        # Step 2: Compute attention scores (affinities)
        # We multiply queries with keys to find "who is relevant to whom"
        wei = q @ k.transpose(-2, -1)  # (B, T, T)
        
        # Scale by sqrt(head_size) to keep values stable
        # Why? Without scaling, softmax becomes too sharp (almost one-hot)
        wei = wei * (C ** -0.5)
        
        # Step 3: Apply causal masking (hide future tokens)
        # Set attention to -inf for future positions, so softmax makes them 0
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        
        # Step 4: Convert scores to probabilities (sum to 1 for each row)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        
        # Step 5: Use attention weights to gather values
        v = self.value(x)  # (B, T, head_size) "Here's my information"
        out = wei @ v      # (B, T, head_size) "Here's the weighted mix"
        
        # What is 'out' conceptually?
        # For each position, it's a summary of all relevant information from the past,
        # weighted by how much attention we paid to each previous position.
        
        return out

# ════════════════════════════════════════════════════════════════════════════════════════════════
# MULTI-HEAD ATTENTION - Multiple specialized pattern detectors working in parallel
# ════════════════════════════════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Goal: Let the model examine the context from MULTIPLE perspectives simultaneously
    
    Why multiple heads?
        A single head can only learn ONE type of pattern. But language is rich:
        - Grammatical structure (subject-verb agreement)
        - Semantic relationships (synonyms, antonyms)
        - Positional patterns (what typically comes first/last)
        - Contextual nuances (tone, formality)
        
        Multiple heads = multiple "specialists" examining the same text
    
    How it works:
        1. Split the embedding dimension among multiple heads
           Example: 32-dim split into 4 heads of 8-dim each
        2. Each head independently performs self-attention (in PARALLEL)
        3. Concatenate all head outputs back together
        4. Total dimension stays the same: 4 × 8 = 32
    
    Analogy:
        Instead of one detective with 32 tools examining evidence,
        we have 4 detectives with 8 specialized tools each.
        Each detective finds different clues, then they pool their findings.
    """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        # Run all heads in parallel and concatenate their outputs
        # Input: (B, T, 32) → Each head outputs (B, T, 8) → Concat: (B, T, 32)
        return torch.cat([h(x) for h in self.heads], dim=-1)

# ════════════════════════════════════════════════════════════════════════════════════════════════
# FEEDFORWARD NETWORK - Where the "thinking" happens
# ════════════════════════════════════════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """
    Goal: Let each token PROCESS the information it gathered from attention
    
    The Big Picture:
        Attention is COMMUNICATION - tokens gather info from each other
        Feedforward is COMPUTATION - tokens think about what they learned
        
    Why We Need This:
        Attention is just weighted averaging (linear operation). It's good at
        MIXING information but not at TRANSFORMING it into complex features.
        
        The feedforward network adds:
        1. Non-linearity (ReLU) - enables learning complex patterns
        2. Depth - allows multi-step reasoning
        3. Feature extraction - transforms raw info into useful representations
        
    Example: Processing "The cat sat"
        After attention: Token "sat" gathered context about "cat"
        Feedforward thinks: "Hmm, 'cat' is an animal, 'sat' is a motion verb,
                            animals sitting is a common pattern, this makes sense"
        
    How It Works:
        Input: (B, T, 32)
          ↓
        Linear: Projects to same dimension (32 → 32)
          ↓  
        ReLU: Adds non-linearity (sets negative values to 0)
          ↓
        Output: (B, T, 32)
        
    Key Property: POSITION-WISE
        This operates on each token INDEPENDENTLY. Unlike attention where tokens
        interact, here each token processes its own data separately.
        
        Think of it like: After a group discussion (attention), each person
        goes home and reflects on what they heard (feedforward).
    """

    def __init__(self, n_embd):
        """
        Args:
            n_embd: The embedding dimension (32 in our case)
        """
        super().__init__()
        
        # nn.Sequential: Chains operations together (output of one feeds into next)
        # This creates a simple 2-layer network:
        #   Layer 1: Linear transformation (n_embd → n_embd)
        #   Layer 2: ReLU activation (adds non-linearity)
        
        self.net = nn.Sequential(
            # Linear layer: Learns a transformation of the input
            # Input shape: (B, T, 32) → Output shape: (B, T, 32)
            # This is a LEARNABLE transformation with weights and biases
            # Each of the 32 output dimensions is a weighted combination of all 32 inputs
            nn.Linear(n_embd, n_embd),
            
            # ReLU (Rectified Linear Unit): max(0, x)
            # Why? Adds non-linearity so the network can learn complex patterns
            # Without this, stacking linear layers would just be equivalent to
            # one big linear layer (useless!)
            # ReLU keeps positive values, zeros out negative values
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Process each token's representation independently
        
        Args:
            x: (B, T, n_embd) - token representations after attention
            
        Returns:
            (B, T, n_embd) - transformed representations
            
        What happens:
            For EACH token in EACH sequence in the batch:
            1. Apply linear transformation: mix the 32 dimensions in learned ways
            2. Apply ReLU: introduce non-linearity
            
        Key insight: This operates PER-TOKEN
            x[0, 0, :] is processed independently of x[0, 1, :]
            Unlike attention where tokens interact, here each token "thinks alone"
        """
        return self.net(x)
        
        # Conceptual example for one token:
        # Input:  [0.1, -0.5, 0.3, ..., 0.7]  (32 numbers)
        #    ↓ Linear transformation
        # After:  [0.8, -0.2, 0.1, ..., -0.3] (32 numbers, mixed together)
        #    ↓ ReLU (zero out negatives)
        # Output: [0.8, 0.0, 0.1, ..., 0.0]   (32 numbers, non-linear!)

# ══════════════════════════════════════════════════════════════════════════════
# THE LANGUAGE MODEL - Putting it all together
# ══════════════════════════════════════════════════════════════════════════════

class BigramLanguageModel(nn.Module):
    """
    Goal: Predict the next character given previous characters
    
    Architecture (the journey of a character through the model):
        Input (character IDs)
            ↓
        Token Embeddings (what is this character?)
            +
        Position Embeddings (where is this character in the sequence?)
            ↓
        Multi-Head Self-Attention (COMMUNICATE: gather context from multiple perspectives)
            ↓
        Feedforward Network (COMPUTE: think about what was gathered)
            ↓
        Linear Layer (convert back to vocabulary predictions)
            ↓
        Output (scores for each possible next character)
    
    The Two-Stage Pattern (this is KEY to transformers):
        Stage 1 - Communication: Tokens exchange information (attention)
        Stage 2 - Computation: Tokens process information individually (feedforward)
        
        This pattern can be repeated many times (stacked transformer blocks),
        but we're just doing it once for now.
    """

    def __init__(self):
        super().__init__()
        
        # STEP 1: Embed the token identities
        # Goal: Convert each character ID into a rich vector representation
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # STEP 2: Embed the positions
        # Goal: Give the model a sense of "where" each character is
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # STEP 3a: Multi-Head Self-Attention (COMMUNICATION)
        # Goal: Let characters communicate and gather context from multiple perspectives
        # 4 heads × 8 dimensions each = 32 total dimensions
        self.sa_head = MultiHeadAttention(4, n_embd//4)
        
        # STEP 3b: Feedforward Network (COMPUTATION) ← NEW IN V4!
        # Goal: Let each token THINK about the information it gathered
        # 
        # Why is this needed?
        #   - Attention gathered information (communication)
        #   - But it's just weighted averaging (linear operation)
        #   - Feedforward adds non-linear processing (actual computation)
        #   - This transforms raw context into useful features
        # 
        # Analogy: 
        #   Attention = listening to everyone speak in a meeting
        #   Feedforward = going back to your desk and analyzing what you heard
        # 
        # Architecture: 32-dim → Linear → ReLU → 32-dim
        self.ffwd = FeedForward(n_embd)
        
        # STEP 4: Final prediction layer
        # Goal: Convert our n_embd representation back to vocabulary-sized predictions
        # Maps: 32-dimensional vectors → 65 scores (one for each possible next character)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        Goal: Process input sequences and compute predictions (and loss if training)
        
        Args:
            idx: (B, T) - batch of input sequences (character IDs)
            targets: (B, T) - correct next characters (only provided during training)
        
        Returns:
            logits: (B, T, vocab_size) - prediction scores for next character
            loss: scalar - how wrong our predictions were (lower = better)
        """
        B, T = idx.shape
        
        # STEP 1 & 2: Get embeddings for both tokens and positions
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        x = tok_emb + pos_emb  # (B, T, n_embd)
        
        # STEP 3a: COMMUNICATION - Multi-head self-attention
        # Each token gathers information from previous tokens
        # Multiple heads look for different types of patterns
        x = self.sa_head(x)  # (B, T, n_embd)
        # After this: x contains "gathered context from multiple perspectives"
        
        # STEP 3b: COMPUTATION - Feedforward network ← NEW IN V4!
        # Each token now PROCESSES the information it gathered
        # 
        # What happens here:
        #   - Each token independently transforms its representation
        #   - Linear layer learns complex feature combinations
        #   - ReLU adds non-linearity (enables complex reasoning)
        #   - Output has same shape but richer, more processed information
        # 
        # Before v4: We went straight from attention → predictions
        #   Problem: Just mixing info, no real "thinking"
        # After v4: attention → feedforward → predictions  
        #   Better: Mix info (attention) then process it (feedforward)
        x = self.ffwd(x)  # (B, T, n_embd)
        # After this: x contains "processed, transformed representations"
        
        # STEP 4: Convert to vocabulary predictions
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # CALCULATE LOSS (if we're training)
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
        Goal: Generate new text autoregressively (one character at a time)
        
        Process:
            1. Feed current sequence into model
            2. Look at prediction for next character
            3. Sample one character from those predictions
            4. Add it to the sequence
            5. Repeat!
        
        Args:
            idx: (B, T) - starting sequence (e.g., just a newline character)
            max_new_tokens: how many new characters to generate
        
        Returns:
            idx: (B, T+max_new_tokens) - original sequence + generated text
        """
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
** Key Takeaways **
The Communication-Computation Pattern

Attention (Communication): "What should I gather from my neighbors?"
Feedforward (Computation): "Now let me think about what I gathered"

Why Position-Wise?
Feedforward operates on each token independently because:

Attention already handled the interaction between tokens
Now each token needs to process what it learned individually
This separation of concerns is more efficient and effective

The Role of Non-Linearity

Without ReLU, stacking layers doesn't help (linear + linear = linear)
ReLU enables the network to learn complex, non-linear patterns
This is what gives the model true "reasoning" ability

"""