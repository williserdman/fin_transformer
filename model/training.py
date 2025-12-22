import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import Iterable


def _epoch_metrics(preds, targets):
    return {}


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim,
    device: torch.device,
    clip_grad: float,
):
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    # lets us iterate through batches
    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        preds, loss = model(x_batch, y_batch)

        # preds: (T, N, 2)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(preds)
        all_targets.append(y_batch)

    # compute epoch metrics
    all_preds = torch.cat(all_preds, dim=0)  # (T*B, N, 2)
    all_targets = torch.cat(all_targets, dim=0)  # (T*B, N, 1)
    metrics = _epoch_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(dataloader)

    return metrics


def validate_epoch(model: nn.Module, dataloader: DataLoader, device: torch.device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            preds, loss = model(x_batch, y_batch)

            total_loss += loss.item()
            all_preds.append(preds)
            all_targets.append(y_batch)

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    metrics = _epoch_metrics(all_preds, all_targets)
    metrics["loss"] = total_loss / len(dataloader)

    return metrics


def train_loop(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device | None,
    checkpoint_dir: Path | str,
    patience: int,
    clip_grad=1.0,
    verbose=True,
) -> dict:

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )  # adam with weight decay decoupled (better for transformers)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=(patience // 2)
    )

    if not isinstance(checkpoint_dir, Path):
        checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history = {
        "train_loss": [],
        "val_loss": [],
        "learning_rates": [],
    }

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0

    if verbose:
        print(f"Training on device: {device}")
        # trainable params only
        print(f"Trainable parameters: {sum(p.numel() for p in model.parameters())}")
        print(
            f"Training samples: {len(train_loader.dataset)}, Validation samples: {len(val_loader.dataset)}"
        )
        print("-" * 80)

    for epoch in range(num_epochs):
        train_metrics = train_epoch(model, train_loader, optimizer, device, clip_grad)

        val_metrics = validate_epoch(model, val_loader, device)

        scheduler.step(train_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["learning_rates"].append(current_lr)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_counter = 0

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
                print(f"New best model saved at epoch {epoch + 1}")
        else:
            patience_counter += 1

        # logging
        print(
            f"Epoch: {epoch} | Train loss: {train_metrics["loss"]} | Val loss: {val_metrics["loss"]}"
        )

        # early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                print(
                    f"Best validation loss: {best_val_loss} at epoch {best_epoch + 1}"
                )
            break

    best_checkpoint = torch.load(checkpoint_dir / "best_model.pt", weights_only=False)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    if verbose:
        print("-" * 80)
        print(f"Training completed. Best epoch: {best_epoch + 1}")
        print("Best validation metrics:")
        for k, v in best_checkpoint["val_metrics"].items():
            print(f"    {k}: {v}")

    return {
        "model": model,
        "history": history,
        "best_checkpoint": best_checkpoint,
        "best_epoch": best_epoch,
    }
