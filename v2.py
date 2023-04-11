import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?

# using 256 previous characters to predict the 257th
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4

if torch.cuda.is_available():
    dev = 'cuda:0'
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    dev = 'mps'
else:
    dev = 'cpu'
print(dev)
device = dev
eval_iters = 200
n_embd = 384 # number of embedding dimensions
n_head = 6 # 384/6 = 64 ; every head is a 64-dimensional as a standard
n_layer = 6 # 6 layers of that
dropout = 0.2 # every forward/backward pass 20% of intermediate calculations are dropped to zero
# ------------
print(torch.cuda.is_available())

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# 1 head of self attention
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x) #B,T,C
        q = self.query(x) #B, T, C
        #computer attention scores "Affinities"
        wei = q @ k.transpose(-2, -1) * C**-0.5 #[B, T, C] @ [B, C, T] -> [B, T, T] (matrices)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # {B, T, T}
        wei = F.softmax(wei, dim = -1) # [B, T, T]
        wei = self.dropout(wei) # randomly prevent some nodes from communicating
        #perform the weighted aggregration
        v = self.value(x) # [B, T, T] @ [B, T, C] -> [B, T, C] (matrices)
        out = wei @ v
        return out
        
        
class MultiHeadAttention(nn.Module):
    #multiple heads of self attention running in parallel
    def __init__(self, num_heads, head_size):
        super().__init__()
        # creating multiple heads & run in parallel
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.projection = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #concatinating over channel dimension (dim)
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.projection(out)
        return out

class FeedForward(nn.Module):
    # simple linear layer follow by a non-linearity
    
    def __init__(self, n_embd):
        super().__init__()
        # applying on a per token level, all tokens do this independently, think independently
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    # Transformer block: communication followed by computation
    
    def __init__(self, n_embd, n_head):
        #n_embd: embedding dimension, h_head: the num of heads we like
        super().__init__()
        head_size = n_embd // n_head
        self.selfAttentionHead = MultiHeadAttention(n_head, head_size)
        self.feedForward = FeedForward(n_embd)
        self.layerNorm1 = nn.LayerNorm(n_embd)
        self.layerNorm2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        # fork off -> computation -> come back
        x = x + self.selfAttentionHead(self.layerNorm1(x))
        # fork off -> computation -> come back
        x = x + self.feedForward(self.layerNorm2(x))
        # residual connections
        return x
        

# super simple bigram model
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        # self.selfAttentionHead = MultiHeadAttention(4, n_embd//4) 
        # i.e. 4 heads of 8-dimentional self attention
        # self.feedForward = FeedForward(n_embd)
        #n_embd = 32
        # self.blocks = nn.Sequential(
        #     Block(n_embd, n_head = 4),
        #     Block(n_embd, n_head = 4),
        #     Block(n_embd, n_head = 4),
        # )
        self.blocks = nn.Sequential(*[Block(n_embd, n_head= n_head) for _ in range(n_layer)])
        self.layerNorm_final = nn.LayerNorm(n_embd)
        self.langModelHead = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers
        tokenEmbeddings = self.token_embedding_table(idx) # (B,T,C)
        posEmbedding = self.position_embedding_table(torch.arange(T, device=device)) #  (T, C)
        x = tokenEmbeddings + posEmbedding
        # x = self.selfAttentionHead(x) 
        # apply one head of self attention [B, T, C]
        # x = self.feedForward(x)
        x = self.blocks(x)
        x = self.layerNorm_final(x)
        logits = self.langModelHead(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        # if idx > block size, position embedding table is out of scope
        for _ in range(max_new_tokens):
            #crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

# Attention is a communication mechanism. Can be seen as nodes in a directed graph looking at each other and aggregating information with a weighted sum from all nodes that point to them, with data-dependent weights.
# There is no notion of space. Attention simply acts over a set of vectors. This is why we need to positionally encode tokens.
# Each example across batch dimension is of course processed completely independently and never "talk" to each other
# In an "encoder" attention block just delete the single line that does masking with tril, allowing all tokens to communicate. This block here is called a "decoder" attention block because it has triangular masking, and is usually used in autoregressive settings, like language modeling.
# "self-attention" just means that the keys and values are produced from the same source as queries. In "cross-attention", the queries still get produced from x, but the keys and values come from some other, external source (e.g. an encoder module)


# k = torch.randn(B,T,head_size)
# q = torch.randn(B,T,head_size)
# wei = q @ k.transpose(-2, -1) * head_size**-0.5


# Regularization technique: dropout
# prevents overfitting
# Dropout is a concept that every forward & backward pass 
# shuts down some subset of neurons
# randomly drops them to 0, and trains without them
# ends up training an ensemble of subnetworks
# at this time everything is enabled
# all subnetworks are merged into one