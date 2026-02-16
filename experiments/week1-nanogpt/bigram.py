"""
Bigram Language Model — built from scratch following Karpathy's nanoGPT video.

A bigram model is the simplest possible language model:
    - It looks at ONE character at a time
    - And predicts what character comes NEXT
    - It has no memory of anything before the current character

Think of it like this: if you see the letter 'q', you can guess 'u' is coming next.
That single-character-to-next-character relationship is exactly what this model learns.

Key concepts introduced here:
    1. Character-level tokenization  — turning text into integers and back
    2. Batching                      — processing many sequences in parallel for speed
    3. The Embedding table           — a lookup table that stores what comes after each character
    4. Cross-entropy loss            — how we measure how wrong the model's predictions are
    5. The training loop             — how we nudge the model to get better over time
    6. GPU / MPS / CPU support       — running on the best available hardware automatically
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from pathlib import Path

# ── Hyperparameters ────────────────────────────────────────────────────────────
batch_size    = 32    # How many independent sequences we train on at the same time
block_size    = 8     # Max length of input sequence fed to the model (context window)
max_iters     = 10000  # Total number of training steps
eval_interval = 300   # How often we pause training to check our loss numbers
learning_rate = 1e-2  # How big a step the optimizer takes when updating weights
eval_iters    = 200   # How many batches we average over when estimating loss
# ──────────────────────────────────────────────────────────────────────────────

# ── Device Setup ───────────────────────────────────────────────────────────────
# We want to use the fastest hardware available:
#   - CUDA  → NVIDIA GPU (Linux/Windows)
#   - MPS   → Apple Silicon GPU (M1/M2/M3 Mac)
#   - CPU   → fallback if neither is available
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device} \n")
# ──────────────────────────────────────────────────────────────────────────────

torch.manual_seed(1337)  # Fix the random seed so results are reproducible

# ── Load Data ─────────────────────────────────────────────────────────────────
# To download the dataset run:
#   wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

current_dir  = Path.cwd()
project_root = current_dir.parent.parent.parent       # Adjust this to match your folder structure
data_path    = project_root / 'data/shakespeare/input.txt'

with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

print(f"Dataset size: {len(text):,} characters")
print(text[:1000])  # Sneak a peek at the raw text
# ──────────────────────────────────────────────────────────────────────────────

# ── Tokenization ──────────────────────────────────────────────────────────────
# We work at the CHARACTER level — each unique character gets its own integer ID.
# This is the simplest possible tokenizer.

chars      = sorted(list(set(text)))  # All unique characters, sorted alphabetically
vocab_size = len(chars)               # How many unique tokens we have

print(f"Vocabulary ({vocab_size} characters): {''.join(chars)}")

# Mappings: character ↔ integer
stoi = {ch: i for i, ch in enumerate(chars)}   # char  → int  (e.g. 'a' → 1)
itos = {i: ch for i, ch in enumerate(chars)}   # int   → char (e.g. 1  → 'a')

encode = lambda s: [stoi[c] for c in s]          # String  → list of ints
decode = lambda l: ''.join([itos[i] for i in l]) # List of ints → string

# Quick sanity check
print(f"encode('hii there') = {encode('hii there')}")
print(f"decode back         = {decode(encode('hii there'))}")
# ──────────────────────────────────────────────────────────────────────────────

# ── Encode the Entire Dataset ─────────────────────────────────────────────────
# We turn the whole text into a 1D tensor of integers.
# dtype=torch.long because embedding tables expect 64-bit integer indices.
data = torch.tensor(encode(text), dtype=torch.long)

print(f"Data shape: {data.shape}  |  dtype: {data.dtype}")
print(f"First 100 tokens: {data[:100]}")
# ──────────────────────────────────────────────────────────────────────────────

# ── Train / Validation Split ──────────────────────────────────────────────────
# 90% of data for training, 10% held out for validation.
# Validation data is NEVER trained on — it's used to check if the model
# is memorising the training data (overfitting) or genuinely learning.
n          = int(0.9 * len(data))
train_data = data[:n]
val_data   = data[n:]

print(f"Train tokens: {len(train_data):,}  |  Val tokens: {len(val_data):,}")
# ──────────────────────────────────────────────────────────────────────────────

# ── Data Loading ──────────────────────────────────────────────────────────────
def get_batch(split):
    """
    Randomly grab a batch of (input, target) pairs from the dataset.

    For every sequence in the batch:
        x = characters at positions [i   → i+block_size]    ← the input
        y = characters at positions [i+1 → i+block_size+1]  ← the target (shifted by 1)

    Why shift y by 1?
        Because the model's job is: "given character at position t, predict position t+1."
        So y is simply x moved one step into the future.
    """
    source = train_data if split == "train" else val_data

    # Pick batch_size random starting positions
    ix = torch.randint(len(source) - block_size, (batch_size,))

    x = torch.stack([source[i     : i + block_size    ] for i in ix])  # (B, T)
    y = torch.stack([source[i + 1 : i + block_size + 1] for i in ix])  # (B, T) shifted right by 1

    # Move tensors to the correct device (GPU/MPS/CPU)
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()  # Tell PyTorch: "don't track gradients here" — saves memory and speeds things up
def estimate_loss():
    """
    Get a reliable estimate of the current loss on both train and val splits.

    Why not just use the loss from a single training batch?
        Because one batch is noisy — it might be an easy batch or a hard batch by chance.
        Averaging over eval_iters (200) batches gives a much more stable, trustworthy number.

    Returns:
        out: a dict for example {'train': 2.34, 'val': 2.41}
    """
    out = {}

    model.eval()  # Switch to eval mode — important for layers like Dropout/BatchNorm
                  # (not used in this simple model, but it's the correct habit)

    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y       = get_batch(split)
            logits, loss    = model.forward(X, Y)        # We only need the loss here, not the logits
            losses[k]  = loss.item()        # .item() pulls the scalar value out of the tensor
        out[split] = losses.mean()          # Average all eval_iters losses into one number

    model.train()  # Switch back to training mode
    return out
# ──────────────────────────────────────────────────────────────────────────────

# ── The Model ─────────────────────────────────────────────────────────────────
class BigramLanguageModel(nn.Module):
    """
    The simplest possible language model.

    It uses a single Embedding table of shape (vocab_size, vocab_size).
    Think of it as a direct lookup:
        - Row i answers the question: "When I see character i, what are the scores
          for each possible next character?"

    There is NO attention, NO memory of previous characters beyond the current one.
    That's what makes it a BIGRAM model — it only uses the current (1) character
    to predict the next one.
    """

    def __init__(self, vocab_size):
        super().__init__() # Call the parent class's constructor to set up the module first before we start adding our own stuff.
        # The embedding table doubles as our prediction table:
        # - Input:  a character index (e.g. 40 for 'H')
        # - Output: a row of vocab_size scores (logits) for what comes next
        # Weights are initialised randomly by PyTorch — learning will adjust them.
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        idx     : (B, T) — batch of input token sequences
        targets : (B, T) — the correct next tokens (used during training)

        Returns:
            logits : raw prediction scores before softmax, shape (B, T, C)
            loss   : cross-entropy loss if targets provided, else None
        """
        # Look up each token in the embedding table → raw prediction scores
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            # Inference mode — no loss needed
            loss = None
        else:
            # PyTorch's cross_entropy expects shape (N, C), not (B, T, C)
            # So we flatten B and T into one dimension
            B, T, C = logits.shape
            logits  = logits.view(B * T, C)   # (B*T, C)
            targets = targets.view(B * T)     # (B*T,)
            loss    = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        Autoregressively generate max_new_tokens new characters.

        At each step:
            1. Run forward pass to get scores (logits) for all positions
            2. Look only at the LAST position's scores (that's the "next token" prediction)
            3. Convert scores to probabilities via softmax
            4. Sample one token from those probabilities
            5. Append it to the sequence and repeat
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)              # Forward pass  → (B, T, C)

            logits    = logits[:, -1, :]       # Only care about the last time step → (B, C)
            probs     = F.softmax(logits, dim=-1)              # Scores → probabilities → (B, C)
            idx_next  = torch.multinomial(probs, num_samples=1) # Sample one token     → (B, 1)
            idx       = torch.cat((idx, idx_next), dim=1)      # Append to sequence   → (B, T+1)

        return idx
# ──────────────────────────────────────────────────────────────────────────────

# ── Initialise Model & Optimizer ──────────────────────────────────────────────
model     = BigramLanguageModel(vocab_size)
m         = model.to(device)  # Move all model weights to GPU/MPS/CPU

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
# ──────────────────────────────────────────────────────────────────────────────

# ── Training Loop ─────────────────────────────────────────────────────────────
for iter in range(max_iters):

    # Periodically check how we're doing on both train and val data
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter:4d}: train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

    # 1. Get a random batch of training data
    xb, yb = get_batch('train')

    # 2. Forward pass — compute predictions and loss
    logits, loss = model(xb, yb)

    # 3. Zero out gradients from the previous step (they accumulate otherwise)
    optimizer.zero_grad(set_to_none=True)

    # 4. Backward pass — compute how much each weight contributed to the loss
    loss.backward()

    # 5. Update weights — take one step in the direction that reduces loss
    optimizer.step()
# ──────────────────────────────────────────────────────────────────────────────

# ── Generate Text ─────────────────────────────────────────────────────────────
# Start from a single "newline" token (index 0) and let the model run freely.
# The [0] at the end picks the first (and only) sequence from the batch.
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# ──────────────────────────────────────────────────────────────────────────────