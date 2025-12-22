import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
import numpy as np
from pathlib import Path
import math
import torch.nn.functional as F


class SimpleGraphGRU(nn.Module):
    """Simplified Graph-GRU without external dependencies.

    Combines graph convolution with GRU for temporal modeling.
    No need for torch-scatter or torch-sparse.
    """

    def __init__(self, in_channels, hidden_channels, num_layers=1):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers

        # GRU gates
        self.gru = nn.GRUCell(in_channels, hidden_channels)

        # Graph convolution weights
        self.weight_msg = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.weight_self = nn.Linear(hidden_channels, hidden_channels, bias=False)

    def forward(self, x, edge_index, edge_weight=None, h=None):
        """
        Args:
            x: (N, in_channels) node features
            edge_index: (2, E) edge indices
            edge_weight: (E,) edge weights (correlations)
            h: (N, hidden_channels) hidden state (optional)
        Returns:
            h_new: (N, hidden_channels) updated hidden state
        """
        N = x.shape[0]

        # Initialize hidden state if needed
        if h is None:
            h = torch.zeros(N, self.hidden_channels, device=x.device, dtype=x.dtype)

        # Graph convolution: aggregate neighbor messages
        if edge_index is not None and edge_index.numel() > 0:
            # Extract source and target nodes
            src, dst = edge_index[0], edge_index[1]

            # Message from neighbors
            msg = self.weight_msg(h[src])  # (E, hidden)

            # Weight by edge correlation
            if edge_weight is not None:
                msg = msg * edge_weight.unsqueeze(1)  # (E, hidden)

            # Aggregate messages by destination node (simple sum)
            aggregated = torch.zeros(
                N, self.hidden_channels, device=x.device, dtype=x.dtype
            )
            aggregated.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)

            # Normalize by degree (approximate)
            degree = torch.zeros(N, device=x.device, dtype=x.dtype)
            degree.scatter_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
            degree = degree.clamp(min=1).unsqueeze(1)
            aggregated = aggregated / degree
        else:
            aggregated = torch.zeros(
                N, self.hidden_channels, device=x.device, dtype=x.dtype
            )

        # Self-loop contribution
        self_msg = self.weight_self(h)

        # Combine graph convolution with input
        graph_out = F.relu(aggregated + self_msg)

        # GRU update
        h_new = self.gru(x, graph_out)

        return h_new


class ComboModel(nn.Module):
    def __init__(
        self,
        num_features,
        seq_len,
        hidden_dim=32,
        heads=4,
        attn_layers=2,
        dropout_rate=0.3,
    ):
        super().__init__()

        self.dropout_rate = dropout_rate

        # encode
        self.encoder = nn.Linear(num_features, hidden_dim)

        self.inner_encoder = nn.Linear(hidden_dim * 3, hidden_dim)
        self.lin_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # self.seq_len = nn.Parameter(torch.tensor(seq_len), requires_grad=False)
        # normalize in forward pass

        # positional embedding
        self.position_embedding_table = nn.Embedding(
            seq_len, hidden_dim
        )  # position embeddings. each position from block size to n - 1 will get embedding vector

        self.symbol_embedding_table = nn.Embedding(
            seq_len, hidden_dim
        )  # position embeddings. each position from block size to n - 1 will get embedding vector

        # attention step
        self.atts = nn.ModuleList()
        self.queries = nn.ModuleList()
        self.keys = nn.ModuleList()
        self.values = nn.ModuleList()
        self.lms = nn.ModuleList()
        for _ in range(attn_layers):
            self.atts.append(nn.MultiheadAttention(hidden_dim, heads))
            self.queries.append(nn.Linear(hidden_dim, hidden_dim))
            self.keys.append(nn.Linear(hidden_dim, hidden_dim))
            self.values.append(nn.Linear(hidden_dim, hidden_dim))
            self.lms.append(nn.Linear(hidden_dim, hidden_dim))

        # transfer

        # graph step - using our custom implementation
        self.graph = SimpleGraphGRU(hidden_dim, hidden_dim, num_layers=1)

        # decode
        self.decoder = nn.Linear(hidden_dim, 1)

    def forward(self, x_data, y_data=None):
        # x_data expected shape: (T, N, F) or (B, T, N, F)
        batched = x_data.dim() == 4

        def _process_sequence(x_seq, edge_info):
            edge_info = None  # disables graph
            # x_seq: (T, N, F)
            T, N, F_dim = x_seq.shape
            # encode
            h = self.encoder(x_seq)  # (T, N, H)

            T, N, H = h.shape

            """ position = torch.arange(0, T, device=att_in.device).unsqueeze(
                1
            )  # (T,1) # T is sequence length
            div_term = torch.exp(
                torch.arange(0, H, 2, device=att_in.device, dtype=att_in.dtype)
                * (-(math.log(10000.0) / H))
            )  # (H/2,)

            pe = torch.zeros(T, H, device=att_in.device, dtype=att_in.dtype)
            pe[:, 0::2] = torch.sin(position * div_term)
            if H > 1:
                pe[:, 1::2] = torch.cos(position * div_term)

            # add positional encoding (broadcast over nodes)
            pos = pe[:T].unsqueeze(1)  # (T, 1, H) """

            pos = self.position_embedding_table(torch.arange(0, T)).unsqueeze(1)
            # add symbol (node) embeddings: one embedding per node id 0..N-1, broadcast over time
            node_ids = torch.arange(0, N, device=h.device)
            sym = self.symbol_embedding_table(node_ids).unsqueeze(0)  # (1, N, H)
            if sym.dtype != h.dtype:
                sym = sym.to(h.dtype)

            # print(h.shape, pos.shape, sym.shape)
            # expand positional and symbol embeddings to match (T, N, H)
            pos_exp = pos.expand(T, N, H) if pos.size(1) == 1 else pos
            sym_exp = sym.expand(T, N, H) if sym.size(0) == 1 else sym
            att_in = torch.cat([h, pos_exp, sym_exp], dim=-1)
            att_in = self.inner_encoder(att_in)  # T, N, H

            att_in = self.lin_block(att_in)

            h = F.dropout(h, self.dropout_rate, self.training)

            for i in range(len(self.atts)):
                att_in = F.layer_norm(att_in, att_in.shape)
                att_layer = self.atts[i]
                lin = self.lms[i]
                T = att_in.size(0)
                # causal mask: True where positions should be masked
                attn_mask = torch.triu(
                    torch.ones(T, T, device=att_in.device, dtype=torch.bool),
                    diagonal=1,
                )

                q = self.queries[i](att_in)
                k = self.keys[i](att_in)
                v = self.values[i](att_in)

                att_out, _ = att_layer(
                    q, k, v, attn_mask=attn_mask, need_weights=False
                )  # (T, N, H)

                att_in = att_in + att_out
                att_in = F.layer_norm(att_in, att_in.shape)
                att_in = F.dropout(att_in, self.dropout_rate, self.training)

                ff = lin(att_in)  # (T, N, H)
                att_in = F.selu(ff + att_in)

            # pass each time-step through the graph recurrently
            edges = None
            if edge_info is not None:
                if isinstance(edge_info, (list, tuple)):
                    # assume one per time step or static if length == 2 (edge_index, edge_weight) or length == T
                    if len(edge_info) == T:
                        edges = list(edge_info)
                    else:
                        edges = [edge_info] * T
                else:
                    edges = [edge_info] * T

            outs = []
            hidden = None
            for t in range(T):
                x_t = att_in[t]  # (N, H)

                # Extract edge info for this timestep
                if edges is not None and t < len(edges):
                    edge_tuple = edges[t]
                    if isinstance(edge_tuple, (list, tuple)) and len(edge_tuple) == 2:
                        edge_index, edge_weight = edge_tuple
                    else:
                        edge_index, edge_weight = edge_tuple, None
                else:
                    edge_index, edge_weight = None, None

                # Call graph module with proper signature
                hidden = self.graph(x_t, edge_index, edge_weight, hidden)

                out_t = self.decoder(hidden)  # (N, 1) or (N, out_dim)
                outs.append(out_t)

            return torch.stack(outs, dim=0)  # (T, N, 1)

        # handle batching
        if batched:
            B, T, N, F_dim = x_data.shape
            results = []
            for b in range(B):
                seq = x_data[b]  # (T, N, F)
                # pass the corresponding per-batch edge_info (if provided)
                edge_info_b = None
                if y_data is not None:
                    try:
                        edge_info_b = y_data[b]
                    except Exception:
                        # fallback: if y_data isn't indexable by batch, pass it through
                        edge_info_b = y_data
                res_b = _process_sequence(seq, edge_info_b)
                results.append(res_b)
            return torch.stack(results, dim=0)  # (B, T, N, 1)
        else:
            return _process_sequence(x_data, y_data)


class TemporalGraphDataset(torch.utils.data.Dataset):
    """Dataset handler for temporal graph sequences with dynamic edges."""

    def __init__(self, x_tensor, y_tensor, edge_info, sequence_length=20, stride=1):
        super().__init__()
        """
        Args:
            x_tensor: (T, N, F) features tensor
            y_tensor: (T, N, 1) targets tensor
            edge_info: list of (edge_index, edge_weight) tuples, length T
            sequence_length: number of timesteps per sequence
            stride: step size for sliding window
        """
        self.x = x_tensor
        self.y = y_tensor
        self.edge_info = edge_info
        self.seq_len = sequence_length
        self.stride = stride

        # compute valid starting indices
        T = x_tensor.shape[0]
        self.indices = list(range(0, T - sequence_length + 1, stride))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.seq_len

        x_seq = self.x[start:end]  # (seq_len, N, F)
        y_seq = self.y[start:end]  # (seq_len, N, 1)
        edges_seq = self.edge_info[start:end]  # list of (edge_index, edge_weight)

        return x_seq, y_seq, edges_seq


def create_dataloader(
    x_tensor,
    y_tensor,
    edge_info,
    sequence_length=20,
    batch_size=32,
    stride=1,
    shuffle=True,
):
    """Create a DataLoader for temporal graph sequences."""
    dataset = TemporalGraphDataset(
        x_tensor, y_tensor, edge_info, sequence_length, stride
    )

    def collate_fn(batch):
        x_batch = torch.stack([item[0] for item in batch])  # (B, T, N, F)
        y_batch = torch.stack([item[1] for item in batch])  # (B, T, N, 1)
        edges_batch = [
            item[2] for item in batch
        ]  # list of length B, each containing list of (edge_index, edge_weight)
        return x_batch, y_batch, edges_batch

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn
    )
    return loader


def compute_metrics(predictions, targets, mask=None):
    """Compute regression and directional accuracy metrics.

    Args:
        predictions: (B, T, N, 1) or (T, N, 1)
        targets: same shape as predictions
        mask: optional boolean mask for valid samples
    """
    if mask is not None:
        predictions = predictions[mask]
        targets = targets[mask]

    mse = torch.mean((predictions - targets) ** 2).item()
    mae = torch.mean(torch.abs(predictions - targets)).item()

    # directional accuracy: sign(pred) == sign(target)
    pred_dir = (predictions > 0).float()
    target_dir = (targets > 0).float()
    dir_acc = torch.mean((pred_dir == target_dir).float()).item()

    return {"mse": mse, "mae": mae, "rmse": np.sqrt(mse), "dir_acc": dir_acc}


def train_epoch(model, dataloader, optimizer, criterion, device, clip_grad=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_idx, (x_batch, y_batch, edges_batch) in enumerate(dataloader):
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        # move edge tensors to device
        edges_batch_device = []
        for edges_seq in edges_batch:
            edges_seq_device = [(ei.to(device), ew.to(device)) for ei, ew in edges_seq]
            edges_batch_device.append(edges_seq_device)

        optimizer.zero_grad()

        # forward pass with batched sequences
        predictions = model(x_batch, edges_batch_device)  # (B, T, N, 1)

        # compute loss (ignore last timestep target which is often zero-padded)
        loss = criterion(torch.exp(predictions[:, :-1]), torch.exp(y_batch[:, :-1]))

        # directional classification loss (predict sign correctness) and combine with regression loss
        # create binary targets for direction: positive vs non-positive
        target_dir = (y_batch[:, :-1] > 0).float()  # (B, T-1, N, 1)

        # use stable BCE-with-logits (apply directly to raw predictions)
        directional_loss_fn = nn.BCEWithLogitsLoss()
        directional_loss = directional_loss_fn(predictions[:, :-1], target_dir)

        # weight for directional loss (tunable)
        dir_loss_weight = 0.1
        loss = loss + dir_loss_weight * directional_loss

        loss.backward()

        # gradient clipping
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        total_loss += loss.item()
        all_preds.append(predictions[:, :-1].detach().cpu())
        all_targets.append(y_batch[:, :-1].detach().cpu())

    # compute epoch metrics
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(dataloader)

    return metrics


def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch."""
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch, edges_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # move edge tensors to device
            edges_batch_device = []
            for edges_seq in edges_batch:
                edges_seq_device = [
                    (ei.to(device), ew.to(device)) for ei, ew in edges_seq
                ]
                edges_batch_device.append(edges_seq_device)

            predictions = model(x_batch, edges_batch_device)
            loss = criterion(predictions[:, :-1], y_batch[:, :-1])

            total_loss += loss.item()
            all_preds.append(predictions[:, :-1].cpu())
            all_targets.append(y_batch[:, :-1].cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(dataloader)

    return metrics


def train_model(
    model,
    train_loader,
    val_loader,
    num_epochs=100,
    learning_rate=1e-3,
    weight_decay=1e-5,
    device=None,
    checkpoint_dir="checkpoints",
    patience=10,
    clip_grad=1.0,
    verbose=True,
):
    """
    Complete training loop with validation, checkpointing, and early stopping.

    Args:
        model: ComboModel instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        num_epochs: maximum number of epochs
        learning_rate: initial learning rate
        weight_decay: L2 regularization
        device: torch device (auto-detected if None)
        checkpoint_dir: directory to save model checkpoints
        patience: early stopping patience (epochs without improvement)
        clip_grad: gradient clipping threshold
        verbose: print progress

    Returns:
        Dictionary with training history and best model state
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    # optimizer and loss
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=patience // 2
    )
    criterion = nn.MSELoss()

    # setup checkpointing
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # tracking
    history = {
        "train_loss": [],
        "train_mse": [],
        "train_mae": [],
        "train_dir_acc": [],
        "val_loss": [],
        "val_mse": [],
        "val_mae": [],
        "val_dir_acc": [],
        "learning_rates": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    if verbose:
        print(f"Training on device: {device}")
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(
            f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}"
        )
        print("-" * 80)

    for epoch in range(num_epochs):
        # train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, clip_grad
        )

        # validate
        val_metrics = validate_epoch(model, val_loader, criterion, device)

        # update scheduler
        scheduler.step(val_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        # record history
        history["train_loss"].append(train_metrics["loss"])
        history["train_mse"].append(train_metrics["mse"])
        history["train_mae"].append(train_metrics["mae"])
        history["train_dir_acc"].append(train_metrics["dir_acc"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_mse"].append(val_metrics["mse"])
        history["val_mae"].append(val_metrics["mae"])
        history["val_dir_acc"].append(val_metrics["dir_acc"])
        history["learning_rates"].append(current_lr)

        # checkpointing
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0

            # save best model
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": best_val_loss,
                "val_metrics": val_metrics,
                "history": history,
            }
            torch.save(checkpoint, checkpoint_dir / "best_model.pt")

            if verbose:
                print(f"✓ New best model saved at epoch {epoch+1}")
        else:
            patience_counter += 1

        # logging
        if verbose and (epoch % 1 == 0):
            print(
                f"Epoch {epoch+1:3d}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.6f} | "
                f"Val Loss: {val_metrics['loss']:.6f} | "
                f"Val MAE: {val_metrics['mae']:.6f} | "
                f"Val DirAcc: {val_metrics['dir_acc']:.3f} | "
                f"LR: {current_lr:.2e}"
            )

        # early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                print(
                    f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}"
                )
            break

    # load best model
    best_checkpoint = torch.load(checkpoint_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    if verbose:
        print("-" * 80)
        print(f"Training completed. Best epoch: {best_epoch+1}")
        print(f"Best validation metrics:")
        for k, v in best_checkpoint["val_metrics"].items():
            print(f"  {k}: {v:.6f}")

    return {
        "model": model,
        "history": history,
        "best_checkpoint": best_checkpoint,
        "best_epoch": best_epoch,
    }
