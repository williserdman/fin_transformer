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
                torch.ones((self.seq_len, self.seq_len), device=x.device), True
            ),
        )  # (B,T,n_embd)
        x += n
        x = x + self.ffwd(self.ln2(x))  # (B,T,n_embd)
        return x


class SimpleTransformer(nn.Module):
    def __init__(
        self,
        vocab: list[str],
        sequence_len,
        output_classes,
        output_class_freq=None,
        hidden_dim=32,
        heads=4,
        attention_layers=2,
        dropout=0.5,
    ):
        super().__init__()

        ### LATENT REPRESENTATION OF TOKENS/POSITIONS ###
        self.token_embed_table = nn.Embedding(len(vocab), hidden_dim)
        self.position_embed_table = nn.Embedding(sequence_len, hidden_dim)
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
            self.loss_func = nn.CrossEntropyLoss(weight=w)  # TODO: cast to device?
        else:
            self.loss_func = nn.CrossEntropyLoss()

    def forward(self, x_data: torch.tensor, y_data=None):
        assert (
            x_data.dim() == 4
        ), f"{x_data.shape}"  # set up to iterate over batch outside of here

        B, T, N, F = x_data.shape

        def _process_sequence(x_seq, y_seq):
            # F just one here as tokens need to be cast to their learned latents

            # print("x_seq:", x_seq.shape) # (T, N, 1)
            x_seq = x_seq.squeeze(-1)  # (T, N)
            data = self.token_embed_table(x_seq)  # (T, N, H)
            # print(data.shape)

            position = torch.arange(0, self.seq_len, device=x_seq.device)
            position_encoding = self.position_embed_table(position)

            # print(data.shape, position_encoding.shape) -> (T, N, H) (T, H)
            position_encoding = position_encoding.unsqueeze(1).to(data.device)
            data = data + position_encoding

            # treat each N as a separate batch and make time the sequence dim: (N, T, H)data = self.blocks(data)  # (T, N, H)
            data = data.permute(1, 0, 2).contiguous()  # (N, T, H)
            data = self.blocks(data)
            logits = self.lm(data)  # (T, N, output_classes)
            logits = logits.permute(1, 0, 2).contiguous()

            temperature = 0.5
            logits /= temperature

            ### LOSS CALCULATIONS
            loss = None
            if y_data is not None:
                # print(out.shape, y_seq.shape)
                out_flat = out.view(T * N, self.c)
                y_flat = y_seq.view(T * N).long()
                # print(out_flat.shape, y_flat.shape)
                loss = self.loss_func(out_flat, y_flat)

            return self.smax(out), loss  # (T, N, 2), loss

        results = []
        total_loss = 0
        for b in range(B):
            x_seq = x_data[b]
            y_seq = None
            if y_data is not None:
                y_seq = y_data[b]

            res_b, loss_b = _process_sequence(x_seq, y_seq)
            if y_data is not None:
                total_loss += loss_b

            results.append(res_b)

        if y_data is None:
            total_loss = None

        return torch.stack(results, dim=0), total_loss
