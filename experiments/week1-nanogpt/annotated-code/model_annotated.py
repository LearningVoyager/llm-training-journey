"""
The Entire file is answering the following question:
"Given a sequence of tokens, what's the next token most likely to be?"

Everything: Everything — attention, MLP, residuals, LayerNorm — is just 
            machinery in service of that one goal. Keep coming back to it.
"""


# Import the necessary class libraries
from dataclasses import dataclass


@dataclass  # Python shortcut that auto-generates __init__ # Without this, you'd have to manually write self.block_size = block_size for every single field
class GPTConfig:
    
    block_size: int = 1024      # Size of the context window, or the sequence that will be inputed to the model for training
                                # "How many tokens the model can see at once". Think of it as "Model's Memory Window"
                                # It can look back upto 1024 otkens when predicting the next one
    
    vocab_size: int = 50304     # Total number of unique tokens the model knows about.
                                # GPT-2's real vocab is 50257, but padded to 50304 
                                # (nearest multiple of 64) because GPUs run matrix math 
                                # faster when dimensions are multiples of 64. 
                                # The extra 47 fake tokens are just ignored during training.

    n_layer: int = 12           # How many Blocks get stacked on top of each other.
                                # This is the "depth" of the network — like floors in a building.
                                # Each floor refines the understanding built by the floor below.
                                # NOT batch size. Batch size lives in train.py, not here.

    n_head: int = 12            # Number of attention heads inside each Block.
                                # Each head independently learns a different attention pattern.
                                # 12 heads = 12 specialists looking at the sequence differently,
                                # then combining their findings.

    n_embd: int = 768       # How many numbers are used to describe a single token.
                            # This is the "richness" of each token's representation.
                            # NOTE: n_embd and block_size measure completely different things.
                            # block_size = how many tokens (rows in a spreadsheet)
                            # n_embd     = how many features per token (columns in a spreadsheet)
                            # They are independent. Comparing them is like comparing 
                            # the number of people in a room to their average height.


    dropout: float = 0.0    # Fraction of neurons randomly switched off during training.
                            # Acts as a regularizer — prevents the model from memorizing.
                            # 0.0 means dropout is fully OFF here (no neurons dropped).


    bias: bool = True       # Whether to include a bias term in Linear layers and LayerNorms.
                            # A Linear layer computes: output = (input × weight) + bias
                            # Setting bias=False removes that +bias term.
                            # False = slightly fewer parameters, slightly faster, often just as good.
                            # True here = matching GPT-2's original architecture.
                            
def __init__(self, config):
    """How is the model built?"""
   
    
    super().__init__()      # Run PyTorch's own internal setup before we do anything.
                            # nn.Module is the parent class of all neural networks in PyTorch.
                            # It handles weight tracking, GPU movement, saving/loading.
                            # Think of it like calling the landlord's setup checklist before
                            # you move into an apartment and customize it yourself.
                            # Always required. Without this, nothing in PyTorch works.
    

    assert config.vocab_size is not None    # Hard safety checks. The model physically cannot be built without
    assert config.block_size is not None    # knowing vocab_size (can't make the token table) or block_size
                                            # (can't make the position table). Crash loudly now rather than
                                            # failing silently later in a confusing place.
    
    

    self.config = config    # Save the GPT config onto the model so every other method can access
                            # these settings later. Without this line, config disappears after
                            # __init__ finishes and forward() would have no settings to read.
    


    # ================================================================
    # BUILDING THE TRANSFORMER ENGINE (self.transformer)
    # ================================================================

    self.transformer = nn.ModuleDict(dict(          # nn.ModuleDict is a dictionary of layers that PyTorch can track.
                                                    # A regular Python dict would make PyTorch blind to these layers —
                                                    # it couldn't train them, move them to GPU, or save/load them.
                                                    # nn.ModuleDict says: "hey PyTorch, watch all of these."
                                                    #
                                                    # self.transformer is NOT the full model by itself.
                                                    # Full model = self.transformer (engine) + self.lm_head (output gauge)
                                                    # self.transformer does all the thinking.
                                                    # self.lm_head converts that thinking into a prediction.
    


        # ── TOKEN EMBEDDING TABLE (wte) ──────────────────────────────
        # wte = weight token embedding
        # "weight" just means learnable parameter — starts random,
        # gets updated during training like every other weight.
        #
        # This is a lookup table. Shape: (50304, 768)
        # 50304 rows = one row per vocabulary token
        # 768 columns = 768 numbers describing that token
        #
        # How it works: give it token ID 42, it grabs row 42 and
        # hands back 768 numbers. No multiplication — pure lookup.
        # The 768 numbers ARE the embedding. wte IS the embedding table.
        #
        # What do the 768 numbers mean?
        # They describe WHAT the token is — its meaning, identity, tone.
        # They do NOT describe relationships to other tokens yet.
        # That job belongs to the attention mechanism inside the blocks.
        # At this stage: "what is this token?"
        # After attention: "what is this token given everything around it?"
        #
        # These start as random noise and slowly organize into meaningful
        # geometry through training. After training, similar meanings
        # end up pointing in similar directions in 768-dimensional space.
        wte = nn.Embedding(config.vocab_size, config.n_embd),


        # ── POSITION EMBEDDING TABLE (wpe) ───────────────────────────
        # wpe = weight position embedding
        # Same structure as wte — a lookup table. Shape: (1024, 768)
        # 1024 rows = one row per position in the sequence
        # 768 columns = 768 numbers describing that position
        #
        # IMPORTANT: rows here represent POSITIONS, not tokens.
        # Row 0 = learned vector for "being in position 0"
        # Row 5 = learned vector for "being in position 5"
        # It doesn't matter WHAT token is sitting there —
        # purely about WHERE in the sequence that slot is.
        #
        # Why do we need this?
        # Without position embeddings, "dog bites man" and
        # "man bites dog" look identical to the model — same tokens,
        # just reordered. The model would be completely blind to word order.
        #
        # These are ABSOLUTE positions, not relative.
        # Think of a cinema: seat 5 always gets the same position embedding
        # regardless of who is sitting in it.
        #
        # Token embedding answers:  WHAT is this token? (768 numbers)
        # Position embedding answers: WHERE does it sit? (768 numbers)
        # These two get ADDED together before entering the blocks,
        # giving the model both pieces of information simultaneously.
        wpe = nn.Embedding(config.block_size, config.n_embd),


        # ── DROPOUT ──────────────────────────────────────────────────
        # Randomly switches off a fraction of neurons during training.
        # Prevents the model from memorizing training data (overfitting).
        # 0.0 here means dropout is fully OFF — no neurons dropped.
        # Note: lowercase d — config.dropout, not config.Dropout.
        drop = nn.Dropout(config.dropout),


        # ── THE TRANSFORMER BLOCKS / HIDDEN LAYERS (h) ───────────────
        # h = hidden layers. These ARE the transformer blocks AND
        # the hidden layers — same thing, two names.
        #
        # In any neural network:
        # Input layer  → raw data comes in (wte + wpe here)
        # Hidden layers → where all the thinking happens (these blocks)
        # Output layer  → prediction comes out (lm_head below)
        #
        # This single line builds the entire deep part of the network.
        # Breaking it apart:
        #
        # [Block(config) for _ in range(config.n_layer)]
        # → Python list comprehension. Creates 12 Block objects.
        #   Each Block = one transformer block (Attention + MLP).
        #   Each has its own independent weights but same structure.
        #
        # nn.ModuleList([...])
        # → Wraps the list so PyTorch can track all 12 blocks.
        #   Regular Python list = PyTorch can't see them.
        #   nn.ModuleList = PyTorch watches, trains, saves all 12.
        #
        # All 12 start with identical structure but different random weights.
        # Through training they naturally specialize:
        # Early blocks  → simple patterns, grammar, word pairings
        # Middle blocks → sentence structure, topic tracking
        # Late blocks   → abstract reasoning, long-range context
        h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),


        # ── FINAL LAYER NORM (ln_f) ───────────────────────────────────
        # ln_f = Layer Norm final.
        # Applied ONCE after all 12 blocks are done.
        #
        # Intuition: imagine 12 cooks in a kitchen assembly line.
        # Each cook adds seasoning and passes the dish forward.
        # Without quality control, by cook 12 the dish is overwhelmingly
        # salty — each cook's additions compounded unpredictably.
        #
        # LayerNorm is the quality control station between each cook.
        # For each token's 768 numbers it:
        # 1. Computes the average of those 768 numbers
        # 2. Computes how spread out they are (standard deviation)
        # 3. Rescales so average = 0, spread = 1
        # 4. Applies two small learnable parameters to fine-tune the scale
        #
        # This keeps numbers stable and predictable as they flow through
        # 12 blocks. Without it, training becomes chaotic — gradients
        # explode, the optimizer loses direction, loss jumps wildly.
        #
        # ln_f is the FINAL quality control check — after all 12 blocks,
        # right before lm_head makes its prediction.
        ln_f = LayerNorm(config.n_embd, bias=config.bias),
    ))


    # ================================================================
    # OUTPUT LAYER (self.lm_head)
    # ================================================================

    # lm = language model, head = final prediction layer.
    # This lives OUTSIDE self.transformer.
    # Full model = self.transformer + self.lm_head. Both are needed.
    #
    # After 12 blocks, each token is still a 768-number vector.
    # That's rich and meaningful but unreadable — you can't extract
    # "next token" directly from 768 numbers.
    #
    # lm_head is a Linear layer: shape (768, 50304)
    # It converts 768 numbers → 50304 scores (one per vocabulary token)
    # These scores are called LOGITS.
    # The highest logit = model's best guess for the next token.
    #
    # The full output pipeline:
    # 768 numbers (rich representation after 12 blocks)
    #     ↓
    # lm_head (768 → 50304)
    #     ↓
    # 50304 logits (one score per vocabulary token)
    #     ↓
    # softmax → probabilities
    #     ↓
    # highest probability = predicted next token
    self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)


    # ================================================================
    # WEIGHT TYING
    # ================================================================

    # This single line makes wte and lm_head share the exact same matrix.
    #
    # Intuition: imagine learning a foreign language.
    # A bad student builds two separate mental dictionaries:
    #   Reading dictionary  → when I see a word, what does it mean?
    #   Writing dictionary  → when I want to express an idea, which word?
    # These two drift apart. They recognize "melancholy" but never use it.
    #
    # A good student uses ONE unified dictionary for both reading and writing.
    # True understanding of a word = you can both recognize it AND use it.
    #
    # wte    = reading dictionary (token ID → 768 vector, at the START)
    # lm_head = writing dictionary (768 vector → token scores, at the END)
    #
    # They do opposite jobs but are fundamentally about the same thing:
    # understanding tokens. So we force them to share one matrix.
    #
    # Benefits:
    # 1. Saves ~40 million parameters (50304 × 768)
    # 2. Forces consistency — how the model reads tokens at input
    #    must match how it predicts tokens at output
    #
    # When training updates wte, lm_head automatically updates too.
    # They are literally the same numbers in the same memory location.
    self.transformer.wte.weight = self.lm_head.weight


    # ================================================================
    # WEIGHT INITIALIZATION
    # ================================================================

    # self.apply() is a PyTorch method that walks through EVERY layer
    # in the model and calls the given function on each one.
    # So this says: "run _init_weights on every Linear and Embedding layer."
    #
    # _init_weights sets all weights to small random numbers:
    # Linear weights  → random, mean=0, std=0.02
    # Linear biases   → all zeros
    # Embedding weights → random, mean=0, std=0.02
    #
    # Why these specific numbers? Small random weights (std=0.02) are
    # known to work well for transformer training. Not too large
    # (causes exploding gradients) and not too small (causes vanishing
    # gradients). This is the GPT-2 paper's recommended starting point.
    self.apply(self._init_weights)  # single underscore, not double


    # ================================================================
    # SPECIAL SMALLER INIT FOR RESIDUAL PROJECTIONS
    # ================================================================

    # Residual connections: every block does x = x + attention(x)
    # and x = x + mlp(x). With 12 blocks, you're adding 24 times total
    # (2 additions per block × 12 blocks).
    #
    # Intuition: imagine 12 people passing a bucket down a line.
    # Each person adds a splash of water before passing it on.
    # Normal splash × 24 people = bucket overflows every time.
    #
    # The fix: tell everyone upfront — we have 24 splashes happening,
    # so each person's splash should be divided by √24 so the total
    # stays reasonable.
    #
    # This loop finds every c_proj layer (the output projection at the
    # end of each block — the layer responsible for what gets added
    # back in the residual) and initializes its weights smaller:
    # std = 0.02 / √(2 × n_layer)
    # The 2× is because each block has 2 residual additions (attn + mlp)
    # More layers = smaller denominator = smaller initial weights
    # = accumulation stays controlled throughout training
    for pn, p in self.named_parameters():
        if pn.endswith('c_proj.weight'):
            torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))


    # ================================================================
    # SANITY CHECK
    # ================================================================

    # Count every single learnable number in the model.
    # Divide by 1 million to express in millions.
    # For GPT-2 this should print ~124.44M.
    # If you see a wildly different number, something went wrong
    # in your architecture — catch it here before training for hours.
    print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))


class LayerNorm(nn.Module):
    """ LayerNorm with optional bias. PyTorch doesn't support simply bias = False"""



class CausalSelfAttention(nn.Module):

class MLP(nn.Module):

class Block(nn.Module):


