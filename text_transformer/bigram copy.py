import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 300
learning_rate = (
    3e-4  # self attention can't handle low learning rates # bigger network => reduce lr
)
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 384
n_layer = 6  # number of layers
dropout = 0.2
n_head = 6  # number of attention heads
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()  # telling pytorch to not call .backward as it doesn't have to store intermediate vars
def estimate_loss():
    out = {}
    model.eval()  # switching modes, simple model here but be aware of when switching modes
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )  # not a parameter

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)  # (B,T,head_size)
        q = self.query(x)  # (B,T,head_size)
        v = self.value(x)  # (B,T,head_size)

        wei = (
            q @ k.transpose(-2, -1) * (C**-0.5)
        )  # (B,T,T) normalized by the sqrt(head_size) so that peaks are subdued
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)  # (B,T,T)

        out = wei @ v  # (B,T,head_size)
        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, n_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(n_heads)])
        self.proj = nn.Linear(n_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B,T,n_heads*head_size)
        out = self.dropout(self.proj(out))  # (B,T,n_embd)
        return out


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # expand to 4x
            nn.ReLU(),
            # nn.Linear(n_embd, n_embd),  # then back to n_embd
            # nn.ReLU(),
            # nn.Linear(n_embd, n_embd),
            # nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # back into residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)  # (B,T,n_embd)


class Block(nn.Module):
    """tranformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(4, head_size)
        self.ffwd = FeedForward(n_embd)

        self.ln1 = nn.LayerNorm(n_embd)  # layer norm for self-attention
        self.ln2 = nn.LayerNorm(n_embd)  # layer norm for feed

    def forward(self, x):
        x = x + self.sa(self.ln1(x))  # (B,T,n_embd)
        x = x + self.ffwd(self.ln2(x))  # (B,T,n_embd)
        return x


# super simple bigram model
class LanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd
        )  # n_embed is the number of embedding dimensions
        self.position_embedding_table = nn.Embedding(
            block_size, n_embd
        )  # position embeddings. each position from block size to n - 1 will get embedding vector

        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head) for _ in range(n_layer)],
            nn.LayerNorm(n_embd),
        )

        self.lm_head = nn.Linear(
            n_embd, vocab_size
        )  # linear layer that casts n_embd up to vocab_size

    def forward(self, idx, targets=None):

        # idx and targets are both (B,T) tensor of integers
        token_emb = self.token_embedding_table(idx)  # (B,T,C)
        B, T, C = token_emb.shape
        # running it through that linear layer
        pos_emb = self.position_embedding_table(
            torch.arange(T, device=device)
        )  # (T, C)

        x = token_emb + pos_emb
        x = self.blocks(x)  # (B,T,n_embd)

        logits = self.lm_head(x)  # (B,T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):

            idx_cond = idx[
                :, -block_size:
            ]  # (B, T) but only the last block_size tokens

            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


model = LanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

with open("out.txt", "w", encoding="utf-8") as f:
    f.write(decode(m.generate(context, max_new_tokens=5000)[0].tolist()))
