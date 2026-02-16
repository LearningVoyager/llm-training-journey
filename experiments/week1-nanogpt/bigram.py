"""
SIMPLIFIED WORK OF THE BIGRAM MODEL FROM THE JUPITYER NOTEBOOK
"""

### 
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
from pathlib import Path

# hyperparameters
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # What is the max context length for predictions?
max_iters = 3000
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu" # What will be equivalent of running it on mac silicon
eval_iters = 200
n_embd = 32
# --------------------


torch.manual_seed(1337)

# Get current directory and go up to nanochat-learning
current_dir = Path.cwd()
project_root = current_dir.parent.parent.parent  # Goes up to nanochat-learning

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespear/input.txt
# Build path to data
data_path = project_root / 'data/shakespeare/input.txt'


with open(data_path, 'r', encoding='utf-8') as f:
    text = f.read()

# Print the length of characters in dataset
print("length of dataset in characters: ", len(text))

# Look at the first 1000 characters
print(text[:1000])

# Check out all the unique characters in the text
# Extract all unique characters from text and sort them alphabetically  
chars = sorted(list(set(text))) # print each function output in this line if needed
vocab_size = len(chars) # Possible elements in our sequences

print("The following are all the characters in the vocabulary of the input: ", ''.join(chars))
print("\n")
print("The number of unique characters we have in our vocabulary: ",vocab_size)


# Create a mapping from characters that occur in this text
stoi = { ch : i for i, ch in enumerate(chars)}
itos = { i : ch for i, ch in enumerate(chars)}
encode = lambda s: [ stoi[c] for c in s] # Encoder: take string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # Decoder: Take a list of integers, output a string



# Sample Examples of decoding and encoding:
print("The char i is encoded as integer: ",encode("i"))
print("The char i is decoded back from integer value",encode("i")," to ",decode(encode("i")))
print("\n\n")
print(encode("hii there"))
print(decode(encode("hii there")))



# Tokenize the entire text input dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype = torch.long) # Why the long datatype?

print("The shape of the data is ", data.shape, "\n") #check the data shape
print("The data type of each values in the data object is " , data.dtype, "\n" ) # check the datatype

print("The 1st 1000 character encoding in the data object looks like ", data[:1000]) # check the encoding of the first 1000 characters of the input

#---------------------------------------------

# Split into train and validation Dataset

n = int(0.9 * len(data)) # First 90% train dataset, the rest will be validation dataset

train_data = data[:n] # Creating Training Dataset # Dataset the model is trained on
val_data = data[n:] # Creating Validation Dataset # Dataset that helps us test how much we are overfitting

print("The length of training data is ", len(train_data))
print("The length of validation data is ", len(val_data))