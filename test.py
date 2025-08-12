import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA_DIR = "alpha_vantage_data"
SEQ_LEN = 52  # number of weeks in a year
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Step 1: Load & preprocess all CSV data ===


def load_and_preprocess(data_dir):
    # Load all CSVs
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    df_list = []
    for f in files:
        df = pd.read_csv(os.path.join(data_dir, f))
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["ticker"] = f.replace(".csv", "").upper()
        df_list.append(df)

    df = pd.concat(df_list, ignore_index=True)

    # Sort and calculate log returns per ticker
    df = df.sort_values(["ticker", "timestamp"])
    df["log_return"] = df.groupby("ticker")["close"].transform(
        lambda x: np.log(x / x.shift(1))
    )
    df.dropna(subset=["log_return"], inplace=True)

    # Filter trading hours (if intraday; adjust if daily)
    # If your data is daily, you can skip this step

    # Bin volume into quantiles (8 bins)
    volume_q = df["volume"].quantile([0.02, 0.98])
    df = df[(df["volume"] >= volume_q.iloc[0]) & (df["volume"] <= volume_q.iloc[1])]
    df["volume_category"] = pd.qcut(
        df["volume"], 8, labels=[f"Q{i+1}" for i in range(8)]
    )

    # Assign sectors — for simplicity, assign based on ticker mapping (you need to prepare this)
    # For demo, random assignment (replace with your sector mapping)
    np.random.seed(42)
    unique_tickers = df["ticker"].unique()
    sectors = ["Tech", "Finance", "Healthcare", "Energy", "Consumer"]
    ticker_to_sector = {t: np.random.choice(sectors) for t in unique_tickers}
    df["sector"] = df["ticker"].map(ticker_to_sector)

    # Create weekly bins
    df["week"] = df["timestamp"].dt.to_period("W").apply(lambda r: r.start_time)

    # Aggregate per week and ticker: mean log_return, mode volume_cat, mode sector
    def mode(series):
        return series.mode()[0]

    weekly = (
        df.groupby(["week", "ticker"])
        .agg(
            log_return=("log_return", "mean"),
            volume_category=("volume_category", mode),
            sector=("sector", mode),
        )
        .reset_index()
    )

    # Calculate next week's return per ticker for supervised learning
    weekly = weekly.sort_values(["ticker", "week"])
    weekly["next_week_return"] = weekly.groupby("ticker")["log_return"].shift(-1)

    weekly.dropna(subset=["next_week_return"], inplace=True)

    return weekly, ticker_to_sector


weekly_data, ticker_to_sector = load_and_preprocess(DATA_DIR)

print(
    f"Loaded weekly data for {len(weekly_data['ticker'].unique())} tickers over {len(weekly_data['week'].unique())} weeks"
)

# === Step 2: Split weeks 80/20 train/val ===
all_weeks = sorted(weekly_data["week"].unique())
split_idx = int(len(all_weeks) * TRAIN_RATIO)
train_weeks = all_weeks[:split_idx]
val_weeks = all_weeks[split_idx:]

train_data = weekly_data[weekly_data["week"].isin(train_weeks)]
val_data = weekly_data[weekly_data["week"].isin(val_weeks)]

print(f"Train weeks: {len(train_weeks)}, Val weeks: {len(val_weeks)}")

# Prepare feature mappings for categorical features
volume_categories = sorted(weekly_data["volume_category"].unique())
volume_cat_to_idx = {c: i for i, c in enumerate(volume_categories)}
sectors = sorted(weekly_data["sector"].unique())
sector_to_idx = {s: i for i, s in enumerate(sectors)}


num_heads = 4
raw_feature_dim = 1 + len(volume_categories) + len(sectors)


def next_multiple_of_n(n, multiple):
    return ((n + multiple - 1) // multiple) * multiple


feature_dim = next_multiple_of_n(raw_feature_dim, num_heads)
assert feature_dim % num_heads == 0, "Feature dim must be divisible by number of heads"


def features_from_df(df):
    global feature_dim
    log_return = torch.tensor(df["log_return"].values, dtype=torch.float32).unsqueeze(1)
    vol_idx = torch.tensor(
        [volume_cat_to_idx[v] for v in df["volume_category"]], dtype=torch.long
    )
    vol_onehot = F.one_hot(vol_idx, num_classes=len(volume_categories)).float()
    sec_idx = torch.tensor([sector_to_idx[s] for s in df["sector"]], dtype=torch.long)
    sec_onehot = F.one_hot(sec_idx, num_classes=len(sectors)).float()

    x = torch.cat([log_return, vol_onehot, sec_onehot], dim=1)

    # Padding if needed:
    if x.shape[1] < feature_dim:
        pad_size = feature_dim - x.shape[1]
        padding = torch.zeros((x.shape[0], pad_size))
        x = torch.cat([x, padding], dim=1)

    return x


def target_from_df(df):
    return torch.tensor(df["next_week_return"].values, dtype=torch.float32)


# Group weekly data per ticker into sequences of length SEQ_LEN (52 weeks)
def make_sequences(df):
    sequences = []
    targets = []
    tickers = df["ticker"].unique()
    for t in tickers:
        ticker_df = df[df["ticker"] == t].sort_values("week")
        total_weeks = len(ticker_df)
        # create sequences with sliding window of size SEQ_LEN
        for start in range(0, total_weeks - SEQ_LEN):
            seq_df = ticker_df.iloc[start : start + SEQ_LEN]
            tgt_df = ticker_df.iloc[
                start + 1 : start + SEQ_LEN + 1
            ]  # next week returns shifted by one
            if len(tgt_df) < SEQ_LEN:
                continue
            sequences.append(features_from_df(seq_df))
            targets.append(target_from_df(tgt_df))
    return sequences, targets


train_weeks = sorted(weekly_data["week"].unique())[
    : int(len(weekly_data["week"].unique()) * 0.8)
]
val_weeks = sorted(weekly_data["week"].unique())[
    int(len(weekly_data["week"].unique()) * 0.8) :
]

train_data = weekly_data[weekly_data["week"].isin(train_weeks)]
val_data = weekly_data[weekly_data["week"].isin(val_weeks)]

train_seqs, train_tgts = make_sequences(train_data)
val_seqs, val_tgts = make_sequences(val_data)

print(f"Train sequences: {len(train_seqs)}, Validation sequences: {len(val_seqs)}")

# Dataset & DataLoader
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


train_dataset = TimeSeriesDataset(train_seqs, train_tgts)
val_dataset = TimeSeriesDataset(val_seqs, val_tgts)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# Simple Transformer model for time series
class TimeSeriesTransformer(nn.Module):
    def __init__(self, feature_dim, nhead=4, num_layers=2, dim_feedforward=128):
        super().__init__()
        self.pos_encoder = nn.Parameter(torch.zeros(SEQ_LEN, feature_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=nhead, dim_feedforward=dim_feedforward
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.linear = nn.Linear(feature_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        x = x + self.pos_encoder.unsqueeze(0)  # add positional encoding
        x = x.permute(1, 0, 2)  # (seq_len, batch, feature_dim)
        out = self.transformer_encoder(x)  # (seq_len, batch, feature_dim)
        out = out.permute(1, 0, 2)  # back to (batch, seq_len, feature_dim)
        out = self.linear(out).squeeze(-1)  # predict next week return per timestep
        return out


print(f"Using feature_dim={feature_dim}, num_heads={num_heads}")

model = TimeSeriesTransformer(feature_dim=feature_dim, nhead=num_heads).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()


def train_epoch():
    model.train()
    total_loss = 0
    for x, y in train_loader:
        x = x.to(DEVICE)
        # x = torch.stack(x).to(DEVICE)
        y = y.to(DEVICE)
        # y = torch.stack(y).to(DEVICE)
        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def validate():
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            # x = torch.stack(x).to(DEVICE)
            # y = torch.stack(y).to(DEVICE)
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            total_loss += loss.item()
    return total_loss / len(val_loader)


for epoch in range(EPOCHS):
    train_loss = train_epoch()
    val_loss = validate()
    print(f"Epoch {epoch+1} - Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

torch.save(model.state_dict(), "timeseries_transformer.pth")
print("Done training.")
