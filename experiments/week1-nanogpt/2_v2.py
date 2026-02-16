"""
═══════════════════════════════════════════════════════════════════════════════
GOAL OF THIS FILE: Build a simple transformer language model from scratch by building on our previous simple bigram model.
 Here we are implementing one self-attention head for our model to learn from previous characters in the sequence. 
 This allows the model to understand context and generate much more coherent text.
═══════════════════════════════════════════════════════════════════════════════

What we're building:
    A character-level language model that can:
    1. Learn patterns in Shakespeare's writing
    2. Generate new text that looks like Shakespeare wrote it

How it works (the big picture):
    1. Convert text into numbers (tokenization)
    2. Feed sequences of numbers into a neural network
    3. The network learns: "given these characters, what comes next?"
    4. Use those predictions to generate new text

New in v2 (we're adding attention!):
    - Token embeddings: represent WHAT each character is
    - Position embeddings: represent WHERE each character sits
    - Self-attention: let each character "look at" previous characters to understand context
    - This makes the model much smarter than the basic bigram model

═══════════════════════════════════════════════════════════════════════════════
"""

# Import necessary libraries
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
# ATTENTION HEAD - The core mechanism that makes transformers work
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
        Self-Attention (gather context from previous characters)
            ↓
        Linear Layer (convert back to vocabulary predictions)
            ↓
        Output (scores for each possible next character)
    
    Note: We still call it "BigramLanguageModel" but it's actually much smarter
    now because of the attention mechanism!
    """

    def __init__(self):
        super().__init__()
        
        # STEP 1: Embed the token identities
        # Goal: Convert each character ID into a rich vector representation
        # Why n_embd instead of vocab_size? We want a compact, learnable representation
        # that captures character relationships (e.g., vowels vs consonants)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        
        # STEP 2: Embed the positions
        # Goal: Give the model a sense of "where" each character is
        # Why? "cat" and "act" have the same letters but different meanings!
        # Position matters. Each position (0 to 7) gets its own learned pattern.
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # STEP 3: Self-attention
        # Goal: Let characters communicate and gather context
        # This is where the magic happens - characters can now "see" each other
        self.sa_head = Head(n_embd)
        
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
        # Why separate? We need to know WHAT the character is AND WHERE it sits
        
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)
        # tok_emb now contains: "This is the character 'h'" (as a 32-dim vector)
        
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, n_embd)
        # pos_emb now contains: "This is position 0", "This is position 1", etc.
        
        # Combine them through addition (broadcasting happens automatically)
        # (B, T, n_embd) + (T, n_embd) → (B, T, n_embd)
        # Now x contains: "Character 'h' at position 3"
        x = tok_emb + pos_emb
        
        # STEP 3: Apply self-attention
        # Let each character gather context from previous characters
        x = self.sa_head(x)  # (B, T, n_embd)
        # Now x contains: "Character 'h' at position 3, informed by context 'The '"
        
        # STEP 4: Convert to vocabulary predictions
        # Transform our rich n_embd representation into scores for each possible next char
        logits = self.lm_head(x)  # (B, T, vocab_size)
        # logits[b, t, :] = scores for what character comes after position t in sequence b

        # CALCULATE LOSS (if we're training)
        if targets is None:
            loss = None  # During generation, we don't have targets
        else:
            # Reshape for cross_entropy (it expects 2D inputs)
            B, T, C = logits.shape
            logits  = logits.view(B * T, C)     # Flatten: (B*T, vocab_size)
            targets = targets.view(B * T)       # Flatten: (B*T,)
            
            # Cross-entropy loss: "How wrong were our predictions?"
            # It's low when we assign high probability to the correct character
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
            # IMPORTANT: Crop to block_size
            # Why? Our position embedding table only goes up to block_size (8).
            # If our sequence grows to 100 characters, we can't embed position 99!
            # Solution: Only feed the last 8 characters into the model
            idx_cond = idx[:, -block_size:]
            
            # Get predictions
            logits, loss = self(idx_cond)  # (B, T, vocab_size)
            
            # Focus only on the last position's prediction
            # Why? That's our prediction for "what comes next"
            logits = logits[:, -1, :]  # (B, vocab_size)
            
            # Convert scores to probabilities
            probs = F.softmax(logits, dim=-1)  # (B, vocab_size)
            
            # Sample one character from the probability distribution
            # Why sample instead of taking max? Makes generation more creative/diverse
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            
            # Append to sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx

# ══════════════════════════════════════════════════════════════════════════════
# MODEL INITIALIZATION AND TRAINING SETUP
# ══════════════════════════════════════════════════════════════════════════════

model = BigramLanguageModel()
m = model.to(device)  # Move to GPU/MPS/CPU

# AdamW optimizer: Adjusts model weights to reduce loss
# lr=1e-3: Small learning rate because attention is sensitive to large updates
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING LOOP - Where the learning happens!
# ══════════════════════════════════════════════════════════════════════════════
# Goal: Repeatedly show the model examples and adjust its weights to improve
#
# The loop:
#   1. Get a batch of examples
#   2. Make predictions
#   3. Calculate how wrong we were (loss)
#   4. Figure out how to adjust weights to reduce loss (backward)
#   5. Update weights (optimizer.step)
#   6. Repeat 10,000 times!

for iter in range(max_iters):

    # Every 500 steps, check our progress
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter:4d}: train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    # Get a batch of training data
    xb, yb = get_batch('train')

    # Forward pass: make predictions and calculate loss
    logits, loss = model(xb, yb)

    # Backward pass: calculate gradients
    optimizer.zero_grad(set_to_none=True)  # Clear old gradients
    loss.backward()                         # Compute new gradients

    # Update weights in the direction that reduces loss
    optimizer.step()

# ══════════════════════════════════════════════════════════════════════════════
# GENERATION - Let's see what our model learned!
# ══════════════════════════════════════════════════════════════════════════════
# Start with a newline character and let the model generate 500 new characters

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))