# %%
import pandas as pd
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

p = Path("data/futures")
parquet_files = sorted(p.glob("*.parquet"))
parquet_files

# %%
dfs = []
for f in parquet_files:
    df = pd.read_parquet(f)

    # flatten multi index
    df.columns = [c[0] for c in df.columns]

    df["ticker"] = f.stem

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    dfs.append(df)

df = pd.concat(dfs, ignore_index=True, sort=False)
# df = df[df["ticker"] == "6E"]


# %%
def get_earliest_per_ticker(df):
    return (
        df.groupby("ticker")["timestamp"].min().reset_index().sort_values("timestamp")
    )


get_earliest_per_ticker(df)

# %%
# remove rows with timestamps before 2002-04-05
import datetime

cutoff = datetime.datetime(2002, 4, 5)
df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
df

# %%
import numpy as np
import pandas as pd


def get_causal_tokens(df, n_bins=10):
    # 1. Calculate ATR (Trailing only)
    # Using 'high' and 'low' directly for TR to avoid look-ahead
    prev_close = df["close"].shift(1)
    high_low = df["high"] - df["low"]
    high_pc = (df["high"] - prev_close).abs()
    low_pc = (df["low"] - prev_close).abs()

    tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # 2. Features

    # A. Candle Shape (Differential)
    # How much did we move today relative to vol?
    conviction = (df["close"] - df["open"]) / atr
    explosion = (df["high"] - df["low"]) / atr

    # B. Trend Context (Integral) - CRITICAL UPDATE
    # Where are we relative to the 20-day mean? (Proxy for visual trend)
    ma_20 = df["close"].rolling(20).mean()
    trend_score = (df["close"] - ma_20) / atr

    # C. Relative Volume
    # Add small epsilon to avoid div by zero
    vol_ma = df["volume"].rolling(50).mean() + 1e-12
    rel_volume = (df["volume"] + 1e-12) / vol_ma

    # 3. Discretization (Fixed Bins to avoid Look-Ahead)
    # We assume standardized features roughly follow N(0,1) or similar.
    # We explicitly define bins rather than learning them from the future.

    def digitize_feature(series, min_val, max_val, bins):
        # Clip outliers
        clipped = series.clip(min_val, max_val)
        # Linearly space bins (or you can use norm.ppf for gaussian spacing)
        # This creates integer codes 0 to bins-1
        return pd.cut(clipped, bins=bins, labels=False, include_lowest=True)

    # Conviction: usually within -3 to 3 ATR
    c_codes = digitize_feature(conviction, -3, 3, n_bins)

    # Trend: usually within -5 to 5 ATR
    t_codes = digitize_feature(trend_score, -5, 5, n_bins)

    # Volume: usually 0 to 5x average
    v_codes = digitize_feature(rel_volume, 0, 5, n_bins)

    # Explosion
    e_codes = digitize_feature(explosion, -3, 3, n_bins)

    # 4. Create Tokens
    # Format: "C{code}_T{code}_V{code}"
    # Using Int64 to handle NaNs gracefully
    df_codes = pd.DataFrame(
        {"c": c_codes, "t": t_codes, "v": v_codes, "e": e_codes}
    ).astype("Int64")

    # Drop early rows where rolling window features are NaN
    df_codes = df_codes.dropna()

    tokens = (
        df_codes["c"].astype(str)
        + "_"
        + df_codes["t"].astype(str)
        + "_"
        + df_codes["v"].astype(str)
        + "_"
        + df_codes["e"].astype(str)
    )

    # Align indices
    tokens = tokens.reindex(df.index)

    return tokens


number_of_classes = 3
feature_bins = 10
# compute tokens per ticker and store back into the main df
df["token"] = None
for _, g in df.groupby("ticker", sort=False):
    df.loc[g.index, "token"] = get_causal_tokens(g, n_bins=feature_bins)


# %%
def get_return(df, periods=1):
    # 1. Use Log Returns (Approximates % change, but additive and symmetric)
    # Formula: ln(Price_future / Price_current)
    # We shift 'close' backwards by 'periods' to compare Today vs. Future

    future_close = df["close"].shift(-periods)

    # Calculate return
    # If future > current, this is positive.
    df["log_ret"] = np.log(future_close / df["close"])

    # 2. (Optional but Recommended) Volatility Normalization
    # This matches the "Sharpe Ratio" logic of the paper.
    # We divide the return by the daily volatility (ATR or Std Dev).
    # This helps the model distinguish a "significant move" from noise.

    # Calculate 20-day volatility (standard deviation of log returns)
    daily_log_ret = np.log(df["close"] / df["close"].shift(1))
    volatility = daily_log_ret.rolling(20).std()

    # The Target: "How many standard deviations will price move?"
    df["target"] = df["log_ret"] / volatility

    return df


def get_hurdle_target(df, cost_threshold=0.002):  # e.g. 0.2% hurdle
    # ... calc returns ...

    # 0 = sell, 1 = neutral, 2 = buy
    df["target"] = 1

    # Only buy if return > cost hurdle
    df.loc[df["ret"] > cost_threshold, "target"] = 1

    # Only sell if return < -cost hurdle
    df.loc[df["ret"] < -cost_threshold, "target"] = 2

    return df


def get_target(df, periods=1, sigma_threshold=0.5):
    # --- 1. Calculate Future Return ---
    # Shift backwards to see what happens 'periods' days later
    future_close = df["close"].shift(-periods)

    # Log returns (Additive, symmetric)
    df["log_ret"] = np.log(future_close / df["close"])

    # --- 2. Normalize by Volatility (The "JKX" Paper Trick) ---
    # Calculate daily volatility (trailing 20 days) to avoid look-ahead
    daily_returns = np.log(df["close"] / df["close"].shift(1))
    volatility = daily_returns.rolling(20).std()

    # Calculate "Normalized Return" (Z-Score)
    # Value of 2.0 means price moved 2 standard deviations
    df["norm_ret"] = df["log_ret"] / volatility

    # --- 3. Create Classes (The Hurdle) ---
    # We use the Normalized Return for the hurdle
    # 0 = Sell (Significant Down Move)
    # 1 = Neutral (Noise / Small Move)
    # 2 = Buy (Significant Up Move)

    # Initialize all as Neutral (1)
    df["target"] = 1

    # Buy: If move is > +0.5 sigma
    df.loc[df["norm_ret"] > sigma_threshold, "target"] = 2

    # Sell: If move is < -0.5 sigma
    df.loc[df["norm_ret"] < -sigma_threshold, "target"] = 0

    # Cleanup: Drop NaN values generated by rolling window and shift
    return df.dropna()


df = get_target(df)
df["target"].value_counts(normalize=True)

# %%
bigger_df = pd.read_parquet("data/ibm_data.parquet").rename(
    columns={"symbol": "ticker"}
)
bigger_df["token"] = get_causal_tokens(bigger_df)
bigger_df = get_target(bigger_df)
bigger_df = bigger_df.dropna()
# df = bigger_df

massive_df = pd.read_parquet("data/small_group").rename(columns={"symbol": "ticker"})
massive_df["token"] = get_causal_tokens(massive_df)
massive_df = get_target(massive_df)
massive_df = massive_df.dropna()
# df = massive_df

# %%
# align all tickers to a common timestamp index and rebuild a compact (T_common, N, 1) integer array
# pivot tokens so each column is a ticker and index are timestamps
token_pivot = df.pivot(index="timestamp", columns="ticker", values="token").sort_index()
ret_pivot = df.pivot(index="timestamp", columns="ticker", values="target").sort_index()


# keep only timestamps present for every ticker
common_mask = token_pivot.notna().all(axis=1) & ret_pivot.notna().all(axis=1)
T_common = int(common_mask.sum())
print("T_common:", T_common)

# restrict pivot to the common timestamps (ensures every column has the same index)
token_pivot = token_pivot.loc[common_mask]
ret_pivot = ret_pivot.loc[common_mask]

# tickers in consistent order
tickers = token_pivot.columns.tolist()
N = len(tickers)
print("N (tickers):", N, "tickers:", tickers)

# split token strings into three component classes (conviction, explosion, rel_volume)
c_pivot = token_pivot.map(lambda s: int(s.split("_")[0]))
t_pivot = token_pivot.map(lambda s: int(s.split("_")[1]))
v_pivot = token_pivot.map(lambda s: int(s.split("_")[2]))
e_pivot = token_pivot.map(lambda s: int(s.split("_")[3]))

# as numpy arrays (T_common, N)
c_arr = c_pivot.values.astype(np.int32)
e_arr = e_pivot.values.astype(np.int32)
t_arr = t_pivot.values.astype(np.int32)
v_arr = v_pivot.values.astype(np.int32)


# combined array with last dim = 3 (conviction, explosion, rel_volume)
class_arr = np.stack([c_arr, e_arr, t_arr, v_arr], axis=-1)

# show result (replace with any further processing you need)
c_pivot.head()

# %%
ret_pivot

# %%
df = df.dropna()
vocab = df["token"].unique().tolist()
vocab_len = len(vocab)

T = len(token_pivot)
N = len(df["ticker"].unique())

T, N

# %%
# build integer mapping once (vocab is defined in the notebook)
# token_to_id = {tok: i for i, tok in enumerate(range(vocab)}

# allocate array (T,N,1) with integer dtype
all_sequences = np.zeros((T, N, 4), dtype=np.int32)
all_targets = np.zeros((T, N))

for idx, sym in enumerate(tickers):
    seq_ids = np.array(
        [
            c_pivot[sym].astype(np.int32).values,
            t_pivot[sym].astype(np.int32).values,
            v_pivot[sym].astype(np.int32).values,
            e_pivot[sym].astype(np.int32).values,
        ]
    )

    # print(seq_ids.T.shape)
    # assert seq_ids.shape[0] == T
    all_sequences[:, idx] = seq_ids.T
    all_targets[:, idx] = ret_pivot[sym].astype(np.int32)

all_sequences.shape, all_sequences.dtype

# %%
all_targets

# %%
# build dataset

# build dataloader

# instantiate model

# train model

# inspect

# %%
from model.my_dataset import MyDataset
from torch.utils.data import DataLoader
from model.model_definition import SimpleTransformer
from model.training import train_loop
import torch

# %%
np.unique(all_targets)

# %%
class_freqs = np.unique(all_targets, return_counts=True)[1]
class_freqs, number_of_classes

# %%
SEQ_LEN = 256
BATCH_SIZE = 32  # the batches chosen should be set up so that the components of each batch are sequential, then shuffle the order of batches/batches, custom dataloader

train_dataset = MyDataset(
    all_sequences[: int(T * 0.8)], all_targets[: int(T * 0.8)], SEQ_LEN
)
tmp = []
[tmp.append(train_dataset[i]) for i in range(30)]
val_dataset = MyDataset(
    all_sequences[int(T * 0.8) :], all_targets[int(T * 0.8) :], SEQ_LEN
)
tmp2 = []
[tmp2.append(val_dataset[i]) for i in range(30, 40)]


train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)  # pin_memory=True and num_workers=4+
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE, shuffle=True
)  # pin_memory=True and num_workers=4+


# pass frequencies to the model (as a list)
model = SimpleTransformer(
    feature_bins,
    SEQ_LEN,
    number_of_classes,
    symbol_count=N,
    output_class_freq=class_freqs,
    attention_layers=2,
    hidden_dim=32,
)

res = train_loop(
    model,
    train_dataloader,
    val_dataloader,
    20,
    4e-4,
    1e-5,
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    "script_checkpoint",
    8,
)

# %%
plt.plot(res["history"]["train_loss"])

# %%
plt.plot(res["history"]["val_loss"])


# %%
""" SEQ_LEN = 256
BATCH_SIZE = 32 """

m2 = SimpleTransformer(
    feature_bins,
    SEQ_LEN,
    number_of_classes,
    symbol_count=N,
    output_class_freq=class_freqs,
    attention_layers=2,
    hidden_dim=32,
)
info = torch.load("script_checkpoint/best_model.pt")
m2.load_state_dict(info["model_state_dict"])

train_dataset = MyDataset(
    all_sequences[: int(T * 0.8)], all_targets[: int(T * 0.8)], SEQ_LEN
)
val_dataset = MyDataset(
    all_sequences[int(T * 0.8) :], all_targets[int(T * 0.8) :], SEQ_LEN
)

train_dataloader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE
)  # pin_memory=True and num_workers=4+
val_dataloader = DataLoader(
    val_dataset, batch_size=BATCH_SIZE
)  # pin_memory=True and num_workers=4+


# %%
model = m2


model.eval()
all_preds = []
with torch.no_grad():
    for x_batch, y_batch in val_dataloader:
        preds, _ = model(x_batch)
        all_preds.append(preds)
        # print(preds)
all_preds

# %%
preds = torch.cat(all_preds, dim=0).detach().cpu().numpy().squeeze(1)
# preds = torch.stack(all_preds).squeeze(1).squeeze(1).detach().cpu().numpy()
print(preds.shape)  # (timesteps, N, output_classes)

# %%
import numpy as np


def build_decile_signals(preds, top_pct=0.10):
    """
    Build top-decile long / short boolean signals from model softprob preds.

    Args:
        preds : np.ndarray, shape (T, N, C) or (N, U, C)
            Softmax probabilities where class 0=negative, 1=neutral, 2=positive.
        top_pct : float
            Fraction for the decile (0.10 -> top 10% longs, top 10% negatives -> shorts).

    Returns:
        long_signal, short_signal : np.ndarray of dtype bool, shape (T, N)
            long_signal[t,i] == True means asset i at timestep t is in the long decile.
            short_signal[t,i] == True means asset i at timestep t is in the short decile.
    """
    arr = np.asarray(preds)
    if arr.ndim != 3:
        raise ValueError("preds must be a 3D array (T, N, C)")

    # ensure shape: (T, N, C)
    T, N, C = arr.shape
    pos = arr[:, :, 2]
    neg = arr[:, :, 0]

    # per-timestep thresholds (percentiles computed over assets)
    p_high = 100 * (1.0 - top_pct)
    pos_thresh = np.nanpercentile(pos, p_high, axis=1)  # shape (T,)
    neg_thresh = np.nanpercentile(neg, p_high, axis=1)  # shape (T,)

    # broadcast thresholds and pick top decile
    long_mask = pos >= pos_thresh[:, None]
    short_mask = neg >= neg_thresh[:, None]

    # resolve any overlaps (both True). use higher margin (pos-neg) to break ties.
    overlap = long_mask & short_mask
    if overlap.any():
        margin = pos - neg
        # where margin > 0 prefer long, else prefer short
        prefer_long = margin > 0
        # for overlapping cells, assign per preference
        long_mask[overlap] = prefer_long[overlap]
        short_mask[overlap] = (~prefer_long)[overlap]

    return long_mask.astype(bool), short_mask.astype(bool)


# Example usage in this notebook:
# assuming `preds` is a numpy array shape (timesteps, n_assets, 3)
# long_signal, short_signal = build_decile_signals(preds, top_pct=0.10)
# These arrays can be passed directly to vectorbt or similar backtest code.
long_signal, short_signal = build_decile_signals(preds)

# %%
preds

# %%
long_signal = preds[:, :, 2] > 0.4
short_signal = preds[:, :, 0] > 0.4
neutral = ~(long_signal | short_signal)  # elementwise NOT of the elementwise OR
neutral.shape

# %%
long_signal.sum(), short_signal.sum()

# %%
close_prices = []
for _, g in df.groupby("ticker"):
    close_price = g[-preds.shape[0] :].sort_index()["close"].values
    close_prices.append(close_price)

close_prices = np.stack(close_prices).T
close_prices.shape

# %%
import vectorbt as vbt

portfolio = vbt.Portfolio.from_signals(
    close=close_prices,
    entries=long_signal,
    exits=short_signal,
    short_entries=short_signal,
    short_exits=long_signal,
    fees=0.0002,
    freq="1d",
)

st = portfolio.stats()
print(st)
# plot equity curves using pandas / matplotlib (supports multiple columns)
pf_value = portfolio.value()  # DataFrame with index=dates and columns=tickers
pf_value.plot(figsize=(12, 6))
plt.title("Portfolio value per ticker")
plt.xlabel("Date")
plt.ylabel("Portfolio value")
plt.legend(loc="best", bbox_to_anchor=(1.02, 1))
plt.tight_layout()
plt.savefig("sharpegraph.png")
# plt.show()

# %%
plt.plot(close_prices - 20)

# %%
plt.hist(preds[:, 0, 0], bins=50)

# %%
