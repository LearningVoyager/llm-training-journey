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
                            
class GPT(nn.Module):


class LayerNorm(nn.Module):
    """ LayerNorm with optional bias. PyTorch doesn't support simply bias = False"""



class CausalSelfAttention(nn.Module):

class MLP(nn.Module):

class Block(nn.Module):


