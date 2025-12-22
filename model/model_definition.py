import torch
import torch.nn as nn
from torch.nn import MultiheadAttention


class Block(nn.Module):
    """tranformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head, sequence_length, dropout=0):
        super().__init__()
        self.sa = MultiheadAttention(n_embd, n_head, dropout, batch_first=True)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),  # expand to 4x
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),  # back into residual pathway
            nn.Dropout(dropout),
        )

        self.ln1 = nn.LayerNorm(n_embd)  # layer norm for self-attention
        self.ln2 = nn.LayerNorm(n_embd)  # layer norm for feed

        self.seq_len = sequence_length

    def forward(self, x):
        x = self.ln1(x)
        n, _ = self.sa(
            x,
            x,
            x,
            need_weights=False,
            is_causal=True,
            attn_mask=torch.triu(
                torch.ones(
                    (self.seq_len, self.seq_len), device=x.device, dtype=torch.bool
                ),
                1,
            ),
        )  # (B,T,n_embd)
        x = x + n
        x = x + self.ffwd(self.ln2(x))  # (B,T,n_embd)
        return x


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        embed_table_sizes,
        sequence_len,
        output_classes,
        symbol_count,
        output_class_freq=None,
        hidden_dim=32,
        heads=4,
        attention_layers=2,
        dropout=0.5,
    ):
        super().__init__()
        self.h = hidden_dim
        ### LATENT REPRESENTATION OF TOKENS/POSITIONS ###
        self.c_embed_table = nn.Embedding(embed_table_sizes, hidden_dim)
        self.e_embed_table = nn.Embedding(embed_table_sizes, hidden_dim)
        self.rv_embed_table = nn.Embedding(embed_table_sizes, hidden_dim)

        self.position_embed_table = nn.Embedding(sequence_len, hidden_dim)
        self.symbol_embed_table = nn.Embedding(
            symbol_count, hidden_dim
        )  # N = symbol_count
        self.seq_len = sequence_len

        ### ATTENTION BLOCKS ###
        self.blocks = nn.Sequential(
            *[
                Block(hidden_dim, heads, sequence_len, dropout=dropout)
                for _ in range(attention_layers)
            ],
            nn.LayerNorm(hidden_dim),
        )

        self.lm = nn.Linear(hidden_dim, output_classes)
        self.c = output_classes
        self.smax = nn.Softmax(dim=-1)

        if output_class_freq is not None:
            w = torch.reciprocal(torch.tensor(output_class_freq))
            self.register_buffer("class_weight", w)
            self.loss_func = nn.CrossEntropyLoss(weight=w)  # TODO: cast to device?
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x_data: torch.tensor, y_data=None):
        assert (
            x_data.dim() == 4
        ), f"{x_data.shape}"  # set up to iterate over batch outside of here

        B, T, N, F = x_data.shape
        H = self.h

        # x = x_data.squeeze(-1)  # (B, T, N, F)
        x = x_data.permute(0, 2, 1, 3).contiguous()  # (B, N, T, F)
        x_flat = x.view(B * N, T, F)  # (B*N, T, F)

        symbol = torch.arange(0, N, device=x.device)  # (N)
        symbol_encoding = self.symbol_embed_table(symbol)  # (N, H)
        symbol_encoding = symbol_encoding.unsqueeze(1).unsqueeze(0)  # (N, 1, H)

        # print(x_flat[:, :, 0].shape) # (N, T)
        c_embed = self.c_embed_table(x_flat[:, :, 0]).view(B, N, T, H)  # (B, N, T, H)
        e_embed = self.c_embed_table(x_flat[:, :, 1]).view(B, N, T, H)  # (B, N, T, H)
        rv_embed = self.c_embed_table(x_flat[:, :, 2]).view(B, N, T, H)  # (B, N, T, H)
        position = torch.arange(0, self.seq_len, device=x_data.device)  # (T)
        position_encoding = (
            self.position_embed_table(position).unsqueeze(0).unsqueeze(0)
        )  # (1, 1, T, H)

        # sum token + position + symbol embeddings -> (B, N, T, H)
        # print(position_encoding.shape, symbol_encoding.shape, c_embed.shape) # torch.Size([1, 1, T, H]) torch.Size([N, 1, H]) torch.Size([B, N, T, H])
        data = position_encoding + symbol_encoding + c_embed + e_embed + rv_embed

        data = data.view(B * N, T, H)

        data = self.blocks(data)
        # logits -> reshape back to (B, T, N, C)
        logits = self.lm(data)  # (B*N, T, C)
        logits = (
            logits.view(B, N, T, self.c).permute(0, 2, 1, 3).contiguous()
        )  # (B, T, N, C)

        temperature = 0.5
        logits = logits / temperature

        ### LOSS CALCULATIONS
        loss = None
        if y_data is not None:
            y = y_data.squeeze(-1)  # (B, T, N)
            out_flat = logits.view(B * T * N, self.c)
            y_flat = y.view(B * T * N).long()
            loss = self.loss_func(out_flat, y_flat)  # already mean over all elements

        probs = self.smax(logits)  # (B, T, N, C)
        return probs, loss
