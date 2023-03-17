import torch
import torch.nn as nn
import tiktoken
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
from model import smallGPT

# -------------------------------------------------------------------------
# dataset
files_no = 100 # number of input text files to read
tokenization = 'bigram'
train_test_split = 0.85
block_size = 64 # number of tokens in one block. Maximum context sequence.
batch_size = 8 # size of the batch with blocks

# training
epochs = 2500 # no. of epochs to train the model
eval_epochs = 100 # no. of epochs to calculate mean test loss
info_interval = 100 # test loss information frequency
learning_rate = 1e-3 

# model
n_channels = 128 # embedded channels
n_head = 4 # number of heads in multihead attention mechanism
n_layer = 4 # number of transformer blocks
dropout = 0.0

device = "cuda" if torch.cuda.is_available() else "cpu" # Set device to GPU
# -------------------------------------------------------------------------

# Read and merge all data files
data_dir = Path("data/")
text_paths = list(data_dir.glob("*"))

text = ''

print("Loading data files:")
for file in tqdm(text_paths[:files_no]):
    with open(file, 'r', encoding='utf-8') as f:
        inp = f.read()
        text += inp

chars = sorted(list(set(text)))
vocab_size = len(chars)

# tokenizing input text
if tokenization == 'bigram':
    to_int = { c:i for i,c in enumerate(chars) }
    to_str = { i:c for i,c in enumerate(chars) }
    encode = lambda s: [to_int[c] for c in s] 
    decode = lambda l: ''.join([to_str[i] for i in l])

    input_data = torch.tensor(encode(text), dtype=torch.long)

elif tokenization == 'gpt2':
    enc = tiktoken.get_encoding("gpt2")
    input_data = torch.tensor(enc.encode(text), dtype=torch.long)

# Train/test data split
sep = int(train_test_split*len(input_data))
train_data = input_data[:sep]
test_data = input_data[sep:]

def random_batch(mode):
    data = train_data if mode == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# initialize the model
m = smallGPT(vocab_size=vocab_size,
             n_channels=n_channels,
             block_size=block_size,
             n_head=n_head,
             n_layer=n_layer,
             dropout=dropout)

model = m.to(device)

@torch.no_grad()
def avg_loss():
    out = {}
    model.eval()
    for mode in ['train', 'test']:
        losses = torch.zeros(eval_epochs)
        for k in range(eval_epochs):
            X, Y = random_batch(mode)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[mode] = losses.mean()
    model.train()
    return out

def train(model, optimizer, epochs):

    results = {"train_loss": []}

    print("Training the model:")
    for epoch in range(epochs):

        if epoch % info_interval == 0 or epoch == epochs - 1:
            losses = avg_loss()
            print(f"Step {epoch}: train loss {losses['train']:.4f}, test loss {losses['test']:.4f}")

        X, Y = random_batch('train')

        logits, loss = model(X, Y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        results["train_loss"].append(loss.item())

    print(f"Final train loss: {loss.item()}")

    return results

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

model_results = train(model=model, 
                      optimizer=optimizer, 
                      epochs=epochs)

# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(model.generate(context, new_tokens_no=2000, block_size)[0].tolist()))