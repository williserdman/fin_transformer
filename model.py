import torch
import torch.nn as nn
from torch.nn import functional as F

batch_size = 64  # how many independent sequences will we process in parallel?
block_size = 256  # what is the maximum context length for predictions?
max_iters = 2000
eval_interval = 300
learning_rate = (
    4e-4  # self attention can't handle low learning rates # bigger network => reduce lr
)
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 128
n_layer = 6  # number of layers
dropout = 0.2
n_head = 12  # number of attention heads

vocab = [
    "R1_Q1",
    "R1_Q2",
    "R1_Q3",
    "R1_Q4",
    "R1_Q5",
    "R1_Q6",
    "R1_Q7",
    "R1_Q8",
    "R2_Q1",
    "R2_Q2",
    "R2_Q3",
    "R2_Q4",
    "R2_Q5",
    "R2_Q6",
    "R2_Q7",
    "R2_Q8",
    "R3_Q1",
    "R3_Q2",
    "R3_Q3",
    "R3_Q4",
    "R3_Q5",
    "R3_Q6",
    "R3_Q7",
    "R3_Q8",
    "R4_Q1",
    "R4_Q2",
    "R4_Q3",
    "R4_Q4",
    "R4_Q5",
    "R4_Q6",
    "R4_Q7",
    "R4_Q8",
    "R5_Q1",
    "R5_Q2",
    "R5_Q3",
    "R5_Q4",
    "R5_Q5",
    "R5_Q6",
    "R5_Q7",
    "R5_Q8",
    "R6_Q1",
    "R6_Q2",
    "R6_Q3",
    "R6_Q4",
    "R6_Q5",
    "R6_Q6",
    "R6_Q7",
    "R6_Q8",
    "R7_Q1",
    "R7_Q2",
    "R7_Q3",
    "R7_Q4",
    "R7_Q5",
    "R7_Q6",
    "R7_Q7",
    "R7_Q8",
    "R8_Q1",
    "R8_Q2",
    "R8_Q3",
    "R8_Q4",
    "R8_Q5",
    "R8_Q6",
    "R8_Q7",
    "R8_Q8",
]
vocab_size = len(vocab)
with open("to_words.txt", "r", encoding="utf-8") as f:
    text = f.read()

stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}


def encode(text):
    """simple encoder"""
    return [stoi[x] for x in text]


def decode(code):
    """decoder for simple encoder"""
    return [itos[x] for x in code]


# print(encode(text)[:100])
data = torch.tensor(encode(text.strip().split("\n")), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    """returns inputs and outputs of batch size"""

    data = train_data if split == "train" else val_data
    ix = torch.randint(
        len(data) - block_size, (batch_size,)
    )  # creating (batchsize, ) random starting places
    x = torch.stack(
        [data[i : i + block_size] for i in ix]
    )  # grabbing the according vectors
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])  # ^
    x, y = x.to(device), y.to(device)  # move to GPU if applicable
    return x, y


# loss estimation
@torch.no_grad
def estimate_loss():
    """switches model to eval mode and estimates losses"""
    out = {}
    model.eval()  # switch to eval mode
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)  # how many to check
        for k in range(eval_iters):
            X, Y = get_batch(split)  # grab new random starting point
            logits, loss = model(X, Y)  # run through model
            losses[k] = loss.item()  # grab the loss value
        out[split] = losses.mean()
    model.train()  # switch back to train mode
    return out


# creating an attention head
class Head(nn.Module):
    def __init__(self, head_size):
        """single attention head"""
        super().__init__()

        self.key = nn.Linear(
            n_embd, head_size, bias=False
        )  # typical not to use bias here, why? # key matrix
        self.query = nn.Linear(n_embd, head_size, bias=False)  # query matrix
        self.value = nn.Linear(n_embd, head_size, bias=False)  # values matrix

        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size))
        )  # don't want as parameter to model # tril of (block_size, block_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """forward pass in our attention head"""
        B, T, C = x.shape
        k = self.key(x)  # get the keys by applying linear layer to x # (B,T,head_size)
        q = self.query(x)  # queries, ^
        v = self.value(x)  # values, ^

        weights = (
            q @ k.transpose(-2, -1) * (C**-5)
        )  # matrix mult w/ transpose to make sizing work --> (B, T, T) normalized by sqrt(head_size) to subdue peaks

        weights = weights.masked_fill(
            self.tril[:T, :T] == 0, float("-inf")
        )  # applying that matrix before softmax
        weights = F.softmax(weights, dim=-1)  # softmax across last dimension
        weights = self.dropout(weights)

        out = weights @ v

        return out


## **
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
class TransformerModel(nn.Module):

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


if __name__ == "__main__":
    model = TransformerModel()
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
        f.write("\n".join(decode(m.generate(context, max_new_tokens=5000)[0].tolist())))

    # Save the trained model
    torch.save(model.state_dict(), "fin_transformer_model.pth")
