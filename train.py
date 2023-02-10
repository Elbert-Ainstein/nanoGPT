import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# all unique characters in the input
chars = sorted(list(set(text)))
vocab_size = len(chars)

# map chars to ints
stringToInt = { ch:i for i,ch in enumerate(chars) }
intToString = { i:ch for i,ch in enumerate(chars) }

# encoder: take a string, output a list of integers
encode = lambda s: [stringToInt[c] for c in s] 
# decoder: take a list of integers, output a string
decode = lambda l: ''.join([intToString[i] for i in l])

# encode the entire text dataset and store it into a torch.Tensor
data = torch.tensor(encode(text), dtype=torch.long)
# split up data into train & validation sets
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

block_size = 8
train_data[:block_size+1]

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]

batch_size = 4 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')
# print('inputs:')
# print(xb.shape)
# print(xb)
# print('targets:')
# print(yb.shape)
# print(yb)

# print('----')

# for b in range(batch_size): # batch dimension
#     for t in range(block_size): # time dimension
#         context = xb[b, :t+1]
#         target = yb[b,t]
#         print(f"when input is {context.tolist()} the target: {target}")

class BigramLangModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for next token from lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None) :
        #idx & targets are both (B, T) tensor of integers
        #B = Batch
        #T = Time
        #C = Channel
        #logits = scores of the next character in the sequence
        #predicting the next character based on previous character
        logits = self.token_embedding_table(idx) #B, T, C
        
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            #the quality of prediction
            #the logits should be high, other dimensions should be low
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities            
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idxNext = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idxNext), dim=1) # (B, T+1)
        return idx
    
m = BigramLangModel(vocab_size)
logits, loss = m(xb, yb)

# print(logits.shape)
# print(loss)

#creating a batch: just one, time = 1, holding a 0, datatype = integer
#0 = new line character

# idx = torch.zeros((1, 1), dtype=torch.long)

#ask for 100 max tokens, it will generate
#set the list to [0] to unplug the batch dimension
#one dimension indices

# print(decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

batch_size = 32
for steps in range(20000): # increase number of steps for good results... 
    
    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
print(loss.item())
print(decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
