import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
# Should use:
#device = 'cuda' if torch.cuda.is_available() else 'cpu
device = 'cpu'

# Hyperparameters
    #######
batch_size = 32
block_size = 8
eval_iters = 200
learning_rate = 0.01
max_iters = 3000
eval_interval = 300
    #######

# Open the text file "input.txt" which contains all of the shakespear's work
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# create a tokenizer, encoding and decoding
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for i, s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Prepare the data
data = torch.tensor(encode(text), dtype=torch.long)

# Prepare splits
n = int(len(data)*0.9)
train_data = data[:n]
test_data = data[n:]

# Our Bigram model class
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)    # ouput of the embedding table is (*, C), where * is the input shape and C is the embedding_dim(here, vocab_size)
        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            # cross entropy expects our inputs to be of shape (minibatch, C) for batched inputs and (C) for unbatched, so we have to reshape our logits and targets
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, __ = self(idx)  # feeds our entire current sequence into the bigram model
            # At each of the T positions, the model spits out C scores (one for each possible next token).
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# Initalize the model and optimizer
model=BigramLanguageModel(vocab_size)
m = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Create methods for calculating loss and getting batched inputs
def get_batch(split):
    data = train_data if split == 'train' else test_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training loop
for iter in range(max_iters):
    
    xb, yb = get_batch('train')

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")

context = torch.tensor(encode('I am Pawan '), dtype=torch.long, device=device)
context = context.view(-1, 1)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))