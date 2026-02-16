"""
═══════════════════════════════════════════════════════════════════════════════
GOAL: Build a DEEP transformer by STACKING transformer blocks
═══════════════════════════════════════════════════════════════════════════════

The Evolution:
    v1-v3: Single layer of attention
    v4: Added feedforward (one cycle of communicate → compute)
    v5: STACK multiple blocks (repeated cycles of communicate → compute)
    
What's "Interspersing"?
    It means ALTERNATING between communication and computation, repeatedly.
    Like a conversation where people talk (communicate), think (compute),
    talk again, think again, multiple times before reaching a conclusion.
    
Why Stack Blocks?
    One cycle of attention+feedforward is good, but limited. By stacking
    multiple blocks, we enable DEEPER reasoning:
    - Block 1: Basic patterns ("this word follows that word")
    - Block 2: Higher-level patterns ("these phrases form sentences")
    - Block 3: Abstract patterns ("sentences convey meaning/tone")
    
    Each layer builds on the previous layer's understanding.

New Architecture:
    Embeddings
       ↓
    Block 1: Attention → Feedforward
       ↓
    Block 2: Attention → Feedforward  
       ↓
    Block 3: Attention → Feedforward
       ↓
    Predictions
    
This is the core transformer pattern! Real models like GPT use 12-96 blocks.

═══════════════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

batch_size    = 32
block_size    = 8
max_iters     = 10000
eval_interval = 500
learning_rate = 1e-3
eval_iters    = 200
n_embd        = 32

# ══════════════════════════════════════════════════════════════════════════════
# DEVICE SETUP
# ══════════════════════════════════════════════════════════════════════════════

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device} \n")
torch.manual_seed(1337)

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

current_dir  = Path.cwd()
project_root = current_dir.parent.parent.parent
data_path    = project_root / 'data/shakespeare/input.txt'

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset size: {len(text):,} characters")
print(text[:1000])

# ══════════════════════════════════════════════════════════════════════════════
# TOKENIZATION
# ══════════════════════════════════════════════════════════════════════════════

chars      = sorted(list(set(text)))
vocab_size = len(chars)

print(f"Vocabulary ({vocab_size} characters): {''.join(chars)}")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"encode('hii there') = {encode('hii there')}")
print(f"decode back         = {decode(encode('hii there'))}")

# ══════════════════════════════════════════════════════════════════════════════
# CONVERT DATASET TO NUMBERS
# ══════════════════════════════════════════════════════════════════════════════

data = torch.tensor(encode(text), dtype=torch.long)
print(f"Data shape: {data.shape}  |  dtype: {data.dtype}")
print(f"First 100 tokens: {data[:100]}")

# ══════════════════════════════════════════════════════════════════════════════
# TRAIN/VALIDATION SPLIT
# ══════════════════════════════════════════════════════════════════════════════

n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")

# ══════════════════════════════════════════════════════════════════════════════
# DATA BATCHING
# ══════════════════════════════════════════════════════════════════════════════

def get_batch(split):
    """Grab random batch of (input, target) pairs for training"""
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - block_size, (batch_size,))
    x = torch.stack([source[i     : i + block_size    ] for i in ix])
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    """Get stable loss estimate by averaging over many batches"""
    out = {}
    model.eval()

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y      = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out

# ══════════════════════════════════════════════════════════════════════════════
# SINGLE ATTENTION HEAD
# ══════════════════════════════════════════════════════════════════════════════

class Head(nn.Module):
    """One head of self-attention - lets tokens gather info from the past"""

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        
        # Compute attention scores and apply causal masking
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        
        # Weighted aggregation of values
        v = self.value(x)
        out = wei @ v
        return out

# ══════════════════════════════════════════════════════════════════════════════
# MULTI-HEAD ATTENTION
# ══════════════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """Multiple attention heads running in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        # Run all heads and concatenate results: 4 heads × 8 dims = 32 total
        return torch.cat([h(x) for h in self.heads], dim=-1)

# ══════════════════════════════════════════════════════════════════════════════
# FEEDFORWARD NETWORK
# ══════════════════════════════════════════════════════════════════════════════

class FeedForward(nn.Module):
    """Position-wise feedforward network - tokens process info independently"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)

# ══════════════════════════════════════════════════════════════════════════════
# TRANSFORMER BLOCK - The fundamental building block
# ══════════════════════════════════════════════════════════════════════════════

class Block(nn.Module):
    """
    Transformer Block: One cycle of communication (attention) + computation (feedforward)
    
    This is the KEY pattern in transformers. We package attention and feedforward
    together so we can easily stack multiple blocks to make the network deeper.
    
    What does "interspersing" mean?
        It means alternating/repeating communication and computation:
        Block 1: communicate → compute
        Block 2: communicate → compute  
        Block 3: communicate → compute
        
        Instead of doing attention once and feedforward once, we do it
        multiple times. Each block refines the representations further.
    
    Why is this powerful?
        - Block 1 might learn basic patterns ("the" often precedes nouns)
        - Block 2 builds on that (sentence structure)
        - Block 3 builds even higher (paragraph-level coherence)
        
        Stacking blocks = stacking layers of abstraction
    """

    def __init__(self, n_embd, n_head):
        """
        Args:
            n_embd: Embedding dimension (32)
            n_head: Number of attention heads (4)
        """
        super().__init__()
        
        head_size = n_embd // n_head  # Split 32 dims into 4 heads of 8 dims each
        
        # Communication: tokens talk to each other
        self.sa = MultiHeadAttention(n_head, head_size)
        
        # Computation: tokens think independently
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        """
        Process input through one transformer block
        
        Args:
            x: (B, T, n_embd) - token representations
            
        Returns:
            (B, T, n_embd) - refined representations after communication + computation
        """
        # Step 1: COMMUNICATE - gather info from other tokens
        x = self.sa(x)
        
        # Step 2: COMPUTE - process the gathered info independently  
        x = self.ffwd(x)
        
        return x

# ══════════════════════════════════════════════════════════════════════════════
# LANGUAGE MODEL - Now with stacked transformer blocks!
# ══════════════════════════════════════════════════════════════════════════════

class BigramLanguageModel(nn.Module):
    """
    Predict next character using a DEEP transformer
    
    New in v5: Instead of one attention + one feedforward, we now have
    3 complete transformer blocks stacked on top of each other.
    
    Architecture:
        Embeddings (token + position)
           ↓
        Block 1 (communicate → compute)
           ↓
        Block 2 (communicate → compute)
           ↓
        Block 3 (communicate → compute)
           ↓
        Final prediction layer
    """

    def __init__(self):
        super().__init__()
        
        # Token and position embeddings (same as before)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # NEW: Stack 3 transformer blocks for deeper processing
        # Each block does: attention (communicate) → feedforward (compute)
        # This replaces the single self.sa_head and self.ffwd from v4
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),  # Block 1: basic patterns
            Block(n_embd, n_head=4),  # Block 2: higher-level patterns
            Block(n_embd, n_head=4),  # Block 3: abstract patterns
        )
        
        # Final prediction layer (same as before)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """Process input and make predictions"""
        B, T = idx.shape
        
        # Get token and position embeddings, combine them
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb  # (B, T, n_embd)
        
        # NEW: Process through 3 transformer blocks sequentially
        # Each block refines the representation through communicate→compute
        # x goes through: Block1 → Block2 → Block3
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
        """Generate new text one character at a time"""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ══════════════════════════════════════════════════════════════════════════════
# MODEL INITIALIZATION AND TRAINING
# ══════════════════════════════════════════════════════════════════════════════

model = BigramLanguageModel()
m = model.to(device)
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
IMPORTANT NOTE: Why This Doesn't Train Well Yet
═══════════════════════════════════════════════════════════════════════════════

We've now entered DEEP NETWORK territory (3 stacked blocks = 6+ layers total).
Deep networks have OPTIMIZATION PROBLEMS:

The Problem:
    When gradients flow backward through many layers, they can:
    - Vanish (become too small → no learning in early layers)
    - Explode (become too large → unstable training)
    
    This is the "vanishing/exploding gradient problem" that plagued deep
    learning before 2015.

The Solution (coming in v6):
    RESIDUAL CONNECTIONS (skip connections) from the ResNet paper.
    These allow gradients to flow directly through the network via
    "shortcut paths", solving the optimization issues.
    
    This is why the Transformer paper is called "Attention Is All You Need"
    but really should be "Attention + Residual Connections Are All You Need"!

Key Takeaway:
    Stacking blocks gives us depth (good for learning complex patterns),
    but we need residual connections to actually train deep networks effectively.

═══════════════════════════════════════════════════════════════════════════════
"""