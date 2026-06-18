import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

"""
This code transform row data to 3-dimensional array:
[client1, client2, clien3, ...]
client_i = [product1, product2, produc3,...]
product_i = [feature_1,feature_2, ...]
"""

#export DATA_DIR=/Users/arina/Downloads

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
TRAIN_PATH = DATA_DIR / "train_data.parquet"
TEST_PATH = DATA_DIR / "test_data.parquet"
TARGET_PATH = DATA_DIR / "train_target.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
OUT_DIR = Path("artifacts/seq")
BATCH_SIZE = 900_000 # batch to read data
MAX_ID = 3_000_000 # max client id
MAX_LEN = 58 # max len of feature vector


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


FEATURE_COLUMNS = [c for c in pq.ParquetFile(TRAIN_PATH).schema.names if c != "id"]
N_FEATURES = len(FEATURE_COLUMNS)


def build_one(parquet_path: Path, ids: np.ndarray, label: str):
    # label can be train or test
    out_x = OUT_DIR / f"{label}_x_uint8.dat" # story of each client
    out_len = OUT_DIR / f"{label}_len_uint8.npy" # len of story of each client
    if out_x.exists() and out_len.exists():
        log(f"{label}: sequence files already exist, skipping")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    id_to_pos = np.full(MAX_ID + 1, -1, dtype=np.int32)
    # mapping between ids and sequential numbers
    id_to_pos[ids.astype(np.int64)] = np.arange(len(ids), dtype=np.int32)

    X = np.memmap(out_x, dtype=np.uint8, mode="w+", shape=(len(ids), MAX_LEN, N_FEATURES)) # data is on the disk, not in RAM
    X[:] = 0
    lengths = np.zeros(len(ids), dtype=np.uint8) # this array indicates real data to distinguish it from padding

    pf = pq.ParquetFile(parquet_path)
    seen_rows = 0
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE), start=1):
        df = batch.to_pandas()
        if not df["id"].is_monotonic_increasing: 
            df = df.sort_values(["id", "rn"], kind="mergesort")

        batch_ids = df["id"].to_numpy(np.int64, copy=False)
        data = df[FEATURE_COLUMNS].to_numpy(dtype=np.uint8, copy=True)
        unique_ids, first_idx, counts = np.unique(batch_ids, return_index=True, return_counts=True)

        for uid, first, cnt in zip(unique_ids, first_idx, counts):
            pos = id_to_pos[int(uid)]
            if pos < 0:
                continue
            start = int(lengths[pos])
            stop = min(start + int(cnt), MAX_LEN)
            take = stop - start
            if take > 0:
                X[pos, start:stop, :] = data[first : first + take]
                lengths[pos] = stop

        seen_rows += len(df)
        log(f"{label}: batch {batch_idx}, rows {seen_rows:,}/{pf.metadata.num_rows:,}")

    np.save(out_len, lengths)
    if np.any(lengths == 0):
        raise RuntimeError(f"{label}: empty histories for {(lengths == 0).sum()} ids")
    log(
        f"{label}: saved X={out_x}, len={out_len}, shape=({len(ids)}, {MAX_LEN}, {N_FEATURES}), "
        f"mean_len={lengths.mean():.3f}, max_len={lengths.max()}"
    )


def main():
    target = pd.read_csv(TARGET_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    build_one(TRAIN_PATH, target["id"].to_numpy(np.int64), "train")
    build_one(TEST_PATH, sample["id"].to_numpy(np.int64), "test")

    meta = {
        "max_len": MAX_LEN,
        "n_features": N_FEATURES,
        "feature_columns": FEATURE_COLUMNS,
    }
    pd.Series(meta, dtype=object).to_pickle(OUT_DIR / "meta.pkl")


if __name__ == "__main__":
    main()
