import torch
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
# Should use:
#device = 'cuda' if torch.cuda.is_available() else 'cpu
device = 'cpu'

# Hyperparameters
    #######
batch_size = 64
block_size = 256
eval_iters = 200
learning_rate = 0.0005
max_iters = 10001
eval_interval = 500
n_embd = 384
n_heads = 6
n_layer = 6
n_dropout = 0.2
    #######

# Open the text file "input.txt" which contains all of the shakespear's work
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
print(len(text))

# create a tokenizer, encoding and decoding
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {s:i for i, s in enumerate(chars)}
itos = {i:s for i, s in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Prepare the data
data = torch.tensor(encode(text ),dtype=torch.long)

# Prepare splits
n = int(len(data)*0.9)
train_data = data[:n]
test_data = data[n:]

class Head(nn.Module):
    """
    Head for multi-head attention.
    """
    def __init__(self, head_size):
        super().__init__()
        self.head_size = head_size
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # This is not a model parameter, but a buffer that is registered to the module. It won't be changed but will be stored in the model's state_dict(If persisted)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))    # mask
        self.dropout = nn.Dropout(n_dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)
        v = self.value(x)   # (B, T, head_size)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * self.head_size ** (-0.5)    # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)    # (B, T, T), apply softmax to the last dimension
        wei = self.dropout(wei)
        out = wei @ v    # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    """
    Multiple heads of self attention running in parallel.
    """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(n_dropout)
    def forward(self, x):
        return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))    # concatenate along the channel dimension

class FeedForward(nn.Module):
    """
    A simple feed forward network with ReLU
    """
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(n_dropout),
        )

    def forward(self, x):
        out = self.net(x)
        return out

class Block(nn.Module):
    """
    Transformer Block: communication followed bu computation
    """
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, self.head_size)
        self.ffn = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        token_emb = self.token_embedding_table(idx)    # ouput of the embedding table is (*, C), where * is the input shape and C is the embedding_dim(here, vocab_size)
        pos_emb = self.positional_embedding_table(torch.arange(T, device=device))
        x = token_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)

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
            # 1) Crop *after* concatenation, so you feed at most `block_size` tokens to the model:
            idx_cond = idx[:, -block_size:]           # shape: (B, ≤block_size)

            # 2) Get logits on that context
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]                 # take last time‑step → (B, vocab_size)
            probs  = F.softmax(logits, dim=-1)        # (B, vocab_size)

            # 3) Sample one new token
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # 4) Append to the full history
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


# Initalize the model and optimizer
model=GPT()
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

context = torch.tensor(encode('MockingBird'), dtype=torch.long, device=device)
context = context.unsqueeze(0)   # shape: (1, 13)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))