"""
The Entire file is answering the following question:
"Given a sequence of tokens, what's the next token most likely to be?"

Everything: Everything — attention, MLP, residuals, LayerNorm — is just 
            machinery in service of that one goal. Keep coming back to it.
"""


# Import the necessary class libraries
import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


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


    def forward(self, idx, targets=None):
        """
        How data flows through the model from raw tokens to prediction.

        Inputs:
            idx     : token IDs of shape (b, t)
                    b = batch size (how many sequences at once)
                    t = sequence length (how many tokens per sequence)
                    These are your input features — the x in y=f(x)

            targets : correct next-token IDs of shape (b, t), or None
                    When provided (training): model calculates loss
                    When None (inference/generation): no correct answer
                    exists yet — just return the prediction, skip loss

        Outputs:
            logits  : raw unnormalized scores, shape (b, t, vocab_size)
                    one score per vocabulary token per position
                    highest score = model's best guess for next token
                    NOT yet probabilities — softmax converts them later

            loss    : single number measuring how wrong the predictions were
                    None during inference (no targets to compare against)
        """

    device = idx.device  # Grab which device idx lives on (CPU/GPU/MPS)
                         # Everything created in this function must live
                         # on the same device or PyTorch crashes

    b, t = idx.size()    # Unpack the two dimensions of idx
                         # b = batch size (e.g. 32 sequences)
                         # t = sequence length (e.g. 512 tokens)

    # Safety check: sequence can't be longer than position table has rows
    # wpe only has block_size rows — can't look up position 1025 if max is 1024
    assert t <= self.config.block_size, \
        f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

    # Create position indices [0, 1, 2, ..., t-1]
    # These are the row indices we'll use to look up position embeddings
    # dtype=torch.long because embedding tables require integer indices
    # device=device so this tensor lives on the same device as everything else
    pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t,)


    # ----------------------------------------------------------------
    # STEP 1: Build input representations
    # Combine WHAT each token is with WHERE it sits
    # ----------------------------------------------------------------

    # TOKEN EMBEDDINGS — WHAT is each token?
    # Pass token IDs into wte lookup table
    # Each token ID → grabs its row (768 numbers) from wte
    # Shape: (b, t, n_embd) — every token in every sequence gets 768 numbers
    tok_emb = self.transformer.wte(idx)

    # POSITION EMBEDDINGS — WHERE does each token sit?
    # Pass position indices [0,1,2,...t-1] into wpe lookup table
    # Each position index → grabs its row (768 numbers) from wpe
    # NOTE: pos goes in here, NOT idx. idx = token IDs, pos = position numbers
    # Shape: (t, n_embd) — one position vector per slot, same for all batches
    # PyTorch broadcasts this to (b, t, n_embd) during the addition below
    pos_emb = self.transformer.wpe(pos)

    # ADD token + position embeddings, then apply dropout
    # tok_emb + pos_emb: (b,t,768) + (t,768) → broadcasts to (b,t,768)
    # Each token now carries both WHAT it is and WHERE it sits in one vector
    # dropout: randomly zeros out some values during training to prevent
    # memorization. 0.0 here = no-op, nothing gets zeroed
    # x is what enters the transformer blocks. Shape: (b, t, n_embd)
    x = self.transformer.drop(tok_emb + pos_emb)


    # ----------------------------------------------------------------
    # STEP 2: Pass through all 12 transformer blocks
    # This is the entire deep thinking part of the network
    # ----------------------------------------------------------------

    # Walk through Block 1, Block 2, ... Block 12 in order
    # Each block receives the OUTPUT of the previous block — not the original x
    # x shape never changes: always (b, t, n_embd)
    # Each block refines and deepens the understanding built by the one before
    #
    # Block 1:  x = block1(x)  → slightly smarter representation
    # Block 2:  x = block2(x)  → even smarter
    # ...
    # Block 12: x = block12(x) → deeply understood representation
    for block in self.transformer.h:
        x = block(x)


    # ----------------------------------------------------------------
    # STEP 3: Final LayerNorm
    # ----------------------------------------------------------------

    # One last recalibration after all 12 blocks
    # Stabilizes the numbers before lm_head makes its prediction
    # Same kitchen quality control as before — final check before serving
    x = self.transformer.ln_f(x)


    # ----------------------------------------------------------------
    # STEP 4: Predict next token (and calculate loss if training)
    # ----------------------------------------------------------------

    if targets is not None:
        # TRAINING MODE — targets are provided
        # Run lm_head on ALL t positions — need predictions everywhere for loss
        # Shape: (b, t, n_embd) → (b, t, vocab_size)
        logits = self.lm_head(x)

        # Calculate how wrong the predictions were
        # logits.view(-1, logits.size(-1)):
        #   (b, t, vocab_size) → (b×t, vocab_size)
        #   collapse batch+time into one dimension
        #   every token in every sequence becomes its own row
        #   e.g. (32, 512, 50304) → (16384, 50304)
        #
        # targets.view(-1):
        #   (b, t) → (b×t,)
        #   one correct answer per row
        #   e.g. (32, 512) → (16384,)
        #
        # ignore_index=-1:
        #   some positions have -1 as target meaning "ignore this position"
        #   used for padding — don't calculate loss for padded slots
        #
        # cross_entropy compares each of the 16384 predictions to its
        # correct answer and returns one single average loss number
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
            ignore_index=-1
        )

    else:
        # INFERENCE/GENERATION MODE — no targets provided
        # Optimization: only run lm_head on the LAST position
        # We only care about what comes AFTER the final token
        # Running lm_head on all t positions would waste compute
        #
        # x[:, [-1], :] means:
        #   :     → all batches
        #   [-1]  → only the last time step (square brackets preserve the dimension)
        #   :     → all embedding dimensions
        # Shape: (b, t, n_embd) → (b, 1, n_embd)
        #
        # lm_head on (b, 1, n_embd) is t times cheaper than on (b, t, n_embd)
        # For t=1024 that's 1024x less work just for this one layer
        logits = self.lm_head(x[:, [-1], :])
        loss = None

    return logits, loss
    

    @torch.no_grad()
    # Turns off gradient tracking for this entire function.
    # During generation we are NOT training — no backprop, no weight updates.
    # Gradient recording uses memory and compute we don't need here.
    # Result: faster and lighter generation.
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        HIGH LEVEL: Generates new tokens one at a time by feeding
        each prediction back into the model as the next input.

        INPUTS:
            idx            : the prompt — starting seed tokens of shape (b, t)
                            NOT the training dataset. This is what you give
                            the model to continue writing from.
                            Grows by 1 token each iteration of the loop.

            max_new_tokens : how many tokens to generate total.
                            The loop runs exactly this many times.

            temperature    : controls randomness of predictions.
                            < 1.0 → more confident, repetitive, safe output
                            > 1.0 → more random, creative, surprising output
                            = 1.0 → model's natural behavior (default)
                            Works by dividing logits before softmax.

            top_k          : only allow the top k most likely tokens to survive.
                            e.g. top_k=10 → out of 50304 possible tokens,
                            only the 10 highest-scoring ones get any probability.
                            Everything else is set to -infinity → 0% chance.
        """

        for _ in range(max_new_tokens):
            # Run once per token to generate. _ = loop counter unused.

            # SAFETY: crop if sequence has grown longer than position table allows.
            # idx.size(1) = current sequence length (grows by 1 each loop).
            # If within limit: use whole sequence.
            # If over limit: keep only the most recent block_size tokens.
            # Sliding window — oldest tokens fall off, newest stay.
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                        else idx[:, -self.config.block_size:]

            # Run the full forward pass to get logits.
            # self(idx_cond) automatically calls forward().
            # In inference mode forward() returns (logits, None).
            # _ discards the None loss — not needed during generation.
            logits, _ = self(idx_cond)

            # GRAB ONLY THE LAST POSITION'S LOGITS AND APPLY TEMPERATURE.
            # [:, -1, :] = all batches, last token position, all vocab scores.
            # Last position = what comes after the final token we have.
            # All other positions predict after earlier tokens — useless here.
            # Dividing by temperature controls confidence vs randomness.
            # shape after: (b, vocab_size)
            logits = logits[:, -1, :] / temperature

            # TOP-K FILTERING (optional).
            if top_k is not None:
                # Get the top k scores. min() ensures we never ask for
                # more tokens than the vocabulary contains.
                # v = top k values, _ = their indices (discarded).
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))

                # v[:, [-1]] = the smallest score among the top k = threshold.
                # Everything below the threshold → -infinity → 0% after softmax.
                # Only the top k tokens survive with any probability.
                logits[logits < v[:, [-1]]] = -float('Inf')

            # CONVERT LOGITS TO PROBABILITIES.
            # softmax turns raw scores into values that sum to 1.0.
            # dim=-1 = apply across vocabulary dimension.
            probs = F.softmax(logits, dim=-1)

            # SAMPLE ONE TOKEN FROM THE PROBABILITY DISTRIBUTION.
            # Weighted random draw — not always the highest score.
            # Sampling introduces controlled randomness so output
            # feels natural and varied rather than robotic and repetitive.
            # Temperature + top_k together shape how this sampling behaves.
            idx_next = torch.multinomial(probs, num_samples=1)  # shape (b, 1)

            # APPEND NEW TOKEN AND LOOP AGAIN.
            # Join along sequence dimension: (b, t) + (b, 1) → (b, t+1).
            # This growing sequence feeds back into the top of the loop.
            # The model uses its own output as the next input —
            # this is called AUTOREGRESSIVE generation.
            idx = torch.cat((idx, idx_next), dim=1)

        # Return original prompt + all generated tokens.
        # Shape: (b, original_t + max_new_tokens)
        return idx

    # Initializes the parameter weights
    def _init_weights(self, module):
        """
        High Level Description:Intialize the 

        Inputs:
            - self:
            - module:
        
        Outputs:
            - Intialized weights
        """

        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02) # Assign normalized weights if module is linear. What does linear mean here?
            if module.bias is not None: # if Model has bias then
                torch.nn.init.zeros_(module.bias) # Set bias terms to zero
        elif isinstance(module, nn.Embedding): # Assign normalized weight for embeddings
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)


    # Get the number of parameters for the entire model
    def get_num_parameters(self, non_embedding = True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameters sharing these
        params are actually used as weights in the final layer, so we include them.

        # What does the above comments mean for non-embedding count? And the token embedding subtraction with whom?

        Input:
            - self:
            - non_embedding: # Are tokens the non-embedding?

        Output:
            - number of parameters used in the model
        """

        n_params = sum(p.numel() for p in self.parameters()) # What is this code doing ? numel()?

        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel() # Why are we doing this? What's the impact

        return n_params

    # Configuring the optimizer
    def configure_optimizer(self, weight_decay, learning_rate, betas, device_type):
        """
        HIGH LEVEL Description:

        Input:
            - self
            - weight decay: I think the weights of the model decay with every run if we have this. I don't know why this would be done
            - learning_rate: Learning rate of the model
            - betas: I know alpha is the learning rate. I don't know what is betas
            - device_type: if we have GPU then this would put data on it for computation
        
        Output:
            - Returns the optimizer with the config that we provided during input
        """
        # start with all of the candidate parameters
        param_dict = {pn : p for pn, p in self.named_parameters()} # What does the code represent here and what is happening? # What are named_parameters?

        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad} # I think we are taking only those parameters that we need to update during training

        
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no. # Why? Don't understand this
        # i.e all weight tensors in matmuls + embedding decay, all biases and layernorms don't. # Why? Dont' understand this
        decay_params = [ p for n, p in param_dict.items() if p.dim() >= 2] # What's happening here? # What are decay parameters? What's the significance of them?
        nodecay_params = [ p for n, p in param_dict.items() if p.dim() < 2] # What's happening here ? What are nodecay parameters? What's the significance of them?

        optim_groups = [
            {'params' : decay_params, 'weight_decay' : weight_decay},
            {'params' : nodecay_params, 'weight_decay' : 0.0}
        ]

        num_decay_params = sum(p.numel() for p in decay_params) # What is happening here? # Are we counting all the decay params?
        num_nodecay_params = sum(p.numel() for p in nodecay_params) # Are we counting nodecay params here?

        print(f"num decayed parameters tensors: {len(decay_params)}, with {num_decay_params : ,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params : ,} parameters")

        # Create AdamW Optimizer and use the fused version if it is available #What is the fused version here? What's the significance and impact?
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters # What's happening here? I am guessing it's checking for fused thing?
        use_fused = fused_available and device_type == 'cuda' # Need to add apple silicon here as well?
        extra_args = dict(fused = True) if use_fused else dict() # What is this line doing?
        
        # Assign the AdamW optimizer
        optimizer = torch.optimz.AdamW(optim_groups, lr = learning_rate, betas = betas, **extra_args) #What is betas here? Why **extra_agrs?

        print(f"using fused AdamW: {use_fused}")

        return optimizer
        

    # What's the significance of crop_block_size()
    def crop_block_size(self, block_size):
        """
        High Level Description: How do we strink the model? 

        Input:
            - block_size: Length of the input sequence
        
        Output:
            - 
        """

        # model surgery to decrease the block sizes if necessary
        # e.g we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model

        assert block_size <= self.config.block_size # If the block size of the model or pretrained model is less than or equal to the block_size of our GPT config file that we provide # if it then we don't need to use this I guess

        self.config.block_size = block_size # Assign the block_size that we provide this function as input and replace the config that we provided originally
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[: block_size]) # What are we doing here? # Are we changing the dimensions of weights based on the new block size?

        # What's happening in the loop below
        for block in self.transformer.h: # For each of our Transformer Blocks
            if hasattr(block.attn, 'bias'): # What's happening here 
                block.attn.bias = block.attn.bias[ :, :, : block_size, : block_size] # What's happening here?

        

    @classmethod # What is this line doing?
    def from_pretrained(cls, model_type, override_args = None): # What's the significance of this?
        """
        HIGH LEVEL DESCRIPTION: How do we load GPT 2 weights

        INPUT:
            - model_type : 
            - override_args : # What is this

        OUTPUT:
            - model
        """

        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'} # Assert if there is any other model apart from the ones mentioned here

        override_args = override_args or {} # default to empty dict # What does this do? What's the significance?
        
        # only dropout can be overridden see more notes below
        assert all(k == 'dropout' for k in override_args) # Why?

        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt : %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2' : dict( n_layer = 12, n_head = 12, n_embd = 768 ), # 124M params
            'gpt2-medium' : dict( n_layer = 12, n_head = 16, n_embd = 1024 ), # 350M params
            'gpt2-large' : dict( n_layer = 36, n_head = 20, n_embd = 1280), # 774M params
            'gpt2-xl' : dict(n_layer = 48, n_head = 25, n_embd = 1600) # 1558 params
        }[model_type] # What is this model type doing here?

        print("forcing vocab_size = 50257, block_size = 1024, bias = True") # Why forcing ? Aren't we changing the parameters to chance based on the model we choose

        config_arg['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_arg['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_arg['bias'] = True # always True for GPT model checkpoints

        # we can override the dropout rate, if desired
        if 'dropout' in override_args: 
            print(f"overriding droput rate to {override_args['dropout']}")
            config_args['dropout'] = override_args['dropout']

        # Create a from-srcatch initialized minGPT model
        config = GPTConfig(**config_args) #What is this doing?
        model = GPT(config) #What is this doing?
        sd = model.state_dict() #What is this doing?
        sd_keys = sd.keys() #What is this doing?
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface / transformers model 
        model_hf = GPT2LMHeadModel.from_pretrained(model_type) # What's going on here?
        sd_hf = model_hf.state_dict() # What's going on here?

        # What is each line doing in the following code?
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight'] # What's the significance of this?

        # basically the open ai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear #What's the significance of this ?
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len{sd_keys_hf}} != {len(sd_keys)}" #What is this doing

        # What's each line doing of the following loop? Why is it doing it. What's the significance?
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
        
        return model




    # Get's the efficency of GPU utilization
    def estimated_mfu(self, fwdbwd_per_iter, dt):
        """
        Estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS

        INPUT:
            - fwdbwd_per_iter: forward and backward passes per iterations 

        OUTPUT:
            -
        """

        # First estimate the number of flops we do per iteration.
        # See PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311

        N = self.get_num_params() # Store the number of parameters
        cfg = self.config # What is this doing? How does cfg look like?
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size # What are H, Q, L, T? What is this line doing? I am guessing this assignign layers (tranformer blocks), nubmer of heads, number of embeddings per number of heads, and block size

        flops_per_token = 6*N + 12*L*H*Q*T #WHY?
        flops_per_fwdbwd = flops_per_token * T # Why?
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter # Why?



        # Express out flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS

        mfu = flops_achieved / flops_promised # Where do we get the flops promised from if this Apple Silicon? # Will this work for Apple Silicon?

        return mfu


class Block(nn.Module):
    """ 
    HIGH LEVEL DESCRIPTION: ONE REPEATABLE UNIT OF TRANSFORMER BLOCK

    INPUT:
        - nn.Module: Inherting things from nn.Module class I guess?
    OUTPUT: 
        - 
    """

    # Intialize the class object
    def __init__(self, config):
        super().__init__() # Letting nn.Module do it's setup before we do our setup

        self.ln_1 = LayerNorm(config.n_embd, bias = config.bias) # I think this is setting up Layer Norm
        
        self.attn = CausalSelfAttention(config) # providing the Self Attention block with config

        self.ln_2 = LayerNorm(config.n_embd, bias = config.bias) # A Second LayerNorm but why?

        self.mlp = MLP(config) # Providing the config to Multi-Layer Perceptrons

    # Defining the Forward function # Question, How is the forward function different from the forward function in GPT class & causualSelfAttention Class?
    def forward(self, x):
        """
        HIGH LEVEL DESCRIPTION: Forward pass block of the transformer I think

        INPUT:
            - x: Input/Features/Sequences of tokens
        OUTPUT: 
            - 
        """

        #
        x = x + self.attn(self.ln_1(x)) # I am guessing this is matrix addition, i.e x and self.attn(self.ln_1(x)) are both matrices and we are adding them # But why are we doing that? And we are passing the layer norm x to the self.attn, why are we doing that too?

        # 
        x = x + self.mlp(self.ln_2(x)) # Why are we doing this now?  Why ln_2 goes in mlp and not ln1? 

        # Can't we do this in one step by x = x + self.attn(self.ln_1(x)) + self.mlp(self.ln_2(x + self.attn(self.ln_1(x))))

        return x
    

# This is the core engine of the Transformer Architecture that helps the model learn context about the tokens in terms of their description and position I think
class CausalSelfAttention(nn.Module):
    """
        HIGH LEVEL DESCRIPTION: The block that helps the transformer learn about context

        INPUT:
            - nn.Module: Inherting things from nn.Module class I guess?

        OUTPUT: 
            - 
    """

    # Initialize the class object
    def __init__(self, config):
        
        super().__init__() # Let the nn.Module do its setup first before we do our setup

        assert config.n_embd % config.n_head == 0 # Why we not want any other remainders except 0 for this?

        # key, query, value projections for all heads, but in batch. # NOT described by me

        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias = config.bias)   # nn.Linear(in_features, out_features) applies the Linear Transformation y = xW' + b
                                                                                        # Holds a weight matrix of Shape(out_features, in_features) 
                                                                                        #TO DO LATER: GO Deep into the code and understand what is happening on a matrix level

        
        # output projection # NOT Desribed by me
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias = config.bias) # What is this doing? Where is c_proj comming from? where is c_attn cominng from?

        # Regularization # Not Described by me
        # Are we assigning all the configs from different classes below?
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0 # Not Described by me
        # What is flash attention? Why does it make GPU go brrrr? What is brrr
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')

        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)) # How is each line of code working?
                                        .view(1, 1, config.block_size, config.block_size))

        # How is this forward method() different than forward method in Block() and GPT() classes?
        def forward(self, x):
            """
                HIGH LEVEL DESCRIPTION: Forward pass block of the Self Attention Block I think

                INPUT:
                    - x: Input/Features/Sequences of tokens
                OUTPUT: 
                    - 
            """

            B, T, C = x.size() # batch size, token/context_window/block_size of the sequence, C = channels or embedding (n_embd) This is what represent or descrive the tokens
            
            # Calculate  query, key, values for all heads in batch and move head forward to be the batch dim # What does moving head forward mean?
            q, k, v = self.c_attn(x).split(self.n_embd, dim = 2) # what is happening here?, what is c_attn and split() doing here?

            # I guessing we are reshaping the Tensor below. But I don't know why or how?
            k = k.view( B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) # What is happening in this code? Go deeper and understand ? What is n_head? and what is head_dim?
            q = q.view( B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) # Same comment as above? What is happening here? Go deeper and understand
            v = v.view( B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs) # Same comment as above? What is happening here? Go deeper and understand
            
            # Causal self-attention; Self attend: (B, nh, T, hs) x (B, nh, hs, T) --> (B, nh, T, T) # What is nh, hs here?
            if self.flash:
                # efficient attention using Flash Attention CUDA Kernels # What is this? Learn more about it?
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask = None, dropout_p = self.dropout if self.training else 0, is_causal = True)

            else:
                # Manual implementation of attention
                att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # Why Sqrt here? Is it to normalize the values in a row? # Multiply Query & Key to get What affinities of similarities that will provide info to the tokens about what they are looking for and whether it matches with other tokens keys
                att = att.masked_fill(self.bias[ :, :, :T, :T] == 0, float('-inf') ) # We are masking the future values here I think so that the tokens don't know what comes next in the sequence
                att = F.softmax( att, dim = -1) # Masking complete here # Why softmax? What's the significance? I am guessing this is converting things into percentage scores for next tokens
                att = self.attn_dropout(att) # Apply drop out to n number of parameters based on the dropout percentage. This is random while training

                y = att @ v # (B, nh, T, T) # This is the final attention scores. Intuitively this tells each tokens 
            
            y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assembe all head outputs side by side # What does this mean? # What is contiguous() and view() do?
            # Why do we need to re-assemble all head outputs side by side

            # output projection
            y = self.resid_dropout(self.c_proj(y)) # What does this mean? Are we decoding the tokens back here to the character values?

            return y 

# This might be the basic building block of multiple neurons in the Deep Learning Network
class MLP(nn.Module):
    
    # Initialized the class object
    def __init__(self, config):
        super().__init() # Let nn.Module do it's setup first

        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias = config.bias) # What does this code do? Is this the first layer of the Neural Network? Why are we doing 4 & config.n_embd?

        self.gelu = nn.GELU() # I think we pass the output of middles layers through this

        self.c_proj = nn.Linear( 4 * config.n_embd, config.n_embd, bias = config.bias) # What does this mean? What is this c_proj? What does this intuitively describe?

        self.dropout = nn.Dropout(config.dropout) # provide the config for dropout

    # How is this Forward different from the forward in SelfAttenion(), Block(), and GPT() classes
    def forward(self, x):
        
        # Pass x through first layer
        x = self.c_fc(x)

        # Pass the new x through Gelu
        x = self.gelu(x)

        x = self.c_proj(x) # Why this?

        x = self.dropout(x) # apply random dropout

        return x
    

class LayerNorm(nn.Module):
    """ LayerNorm with optional bias. PyTorch doesn't support simply bias = False"""

    # initialize the class object
    def __init__(self, ndim, bias): # What is ndim here?
        super().__init__() # Let nn.Module do it's setup first

        self.weight = nn.Parameter(torch.ones(ndim)) # What is this?
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None # Assign Bias to all the parameters I guess

    
    # How is the following forward different from all the other forwards or is it related to them?
    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5) # What does this code do? Go deeper and understand each line?










