import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

"""
Here I construct features and implement gradient boosting using CatBoost 
"""

#export DATA_DIR=/Users/arina/Downloads
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
TRAIN_PATH = DATA_DIR / "train_data.parquet"
TEST_PATH = DATA_DIR / "test_data.parquet"
TARGET_PATH = DATA_DIR / "train_target.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
OUT_PATH = Path("submissions/submission_v2.csv")

BATCH_SIZE = 700_000
MAX_ID = 3_000_000
RANDOM_STATE = 77


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


ALL_COLUMNS = pq.ParquetFile(TRAIN_PATH).schema.names
BASE_COLUMNS = [c for c in ALL_COLUMNS if c not in ("id", "rn")]
PAYM_COLUMNS = [f"enc_paym_{i}" for i in range(25)]


DERIVED_COLUMNS = [
    "paym_mean_all", # mean value across all payments
    "paym_max_all", # max value across all payments
    "paym_bad_ge2_all", # fraction of payments >=2
    "paym_bad_ge3_all", # fraction of payments >=3
    "paym_mean_recent3", # mean enc_paym across last 3 periods
    "paym_max_recent3", # max enc_paym across last 3 periods
    "paym_bad_ge2_recent3", # fraction of payments >=2  across last 3 periods
    "paym_mean_recent6", # mean enc_paym across last 6 periods
    "paym_max_recent6", # max enc_paym across last 6 periods
    "paym_bad_ge2_recent6", # fraction of payments >=2  across last 6 periods
]
AGG_COLUMNS = BASE_COLUMNS + DERIVED_COLUMNS

# constructing features to calculate frequences of categorial features
HIST_SPECS = []
for col in PAYM_COLUMNS:
    for val in range(5):
        HIST_SPECS.append((col, val))
for col, values in [
    ("enc_loans_account_holder_type", range(7)),
    ("enc_loans_credit_status", range(7)),
    ("enc_loans_credit_type", range(8)),
    ("enc_loans_account_cur", range(4)),
]:
    for val in values:
        HIST_SPECS.append((col, val))


def add_payment_features(df: pd.DataFrame) -> np.ndarray:
    paym = df[PAYM_COLUMNS].to_numpy(dtype=np.int16, copy=False)
    recent3 = paym[:, :3]
    recent6 = paym[:, :6]
    pieces = [
        paym.mean(axis=1),
        paym.max(axis=1),
        (paym >= 2).mean(axis=1),
        (paym >= 3).mean(axis=1),
        recent3.mean(axis=1),
        recent3.max(axis=1),
        (recent3 >= 2).mean(axis=1),
        recent6.mean(axis=1),
        recent6.max(axis=1),
        (recent6 >= 2).mean(axis=1),
    ]
    return np.column_stack(pieces).astype(np.float32, copy=False)


def add_hist_features(df: pd.DataFrame) -> np.ndarray:
    hist = np.empty((len(df), len(HIST_SPECS)), dtype=np.uint8)
    for j, (col, val) in enumerate(HIST_SPECS):
        hist[:, j] = (df[col].to_numpy(copy=False) == val)
    return hist


def aggregate_to_memmap(path: Path, ids_needed: np.ndarray, label: str, mmap_path: Path):
    n_agg = len(AGG_COLUMNS)
    n_hist = len(HIST_SPECS)
    counts = np.zeros(MAX_ID + 1, dtype=np.uint16) # amount of client records
    sums = np.zeros((MAX_ID + 1, n_agg), dtype=np.float32) #sum of features
    sqs = np.zeros((MAX_ID + 1, n_agg), dtype=np.float32)
    maxs = np.full((MAX_ID + 1, n_agg), -1, dtype=np.float32)
    mins = np.full((MAX_ID + 1, n_agg), 999, dtype=np.float32)
    firsts = np.zeros((MAX_ID + 1, n_agg), dtype=np.float32)
    lasts = np.zeros((MAX_ID + 1, n_agg), dtype=np.float32)
    seen = np.zeros(MAX_ID + 1, dtype=bool) # it's not the first time to meet this client
    hist_sums = np.zeros((MAX_ID + 1, n_hist), dtype=np.uint16)

    pf = pq.ParquetFile(path)
    rows_seen = 0
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE), start=1):
        df = batch.to_pandas()
        ids = df["id"].to_numpy(np.int64, copy=False)
        if ids.size > 1 and np.any(ids[1:] < ids[:-1]):
            order = np.argsort(ids, kind="mergesort")
            df = df.iloc[order]
            ids = ids[order] 

        base = df[BASE_COLUMNS].to_numpy(dtype=np.float32, copy=False)
        derived = add_payment_features(df)
        data = np.concatenate([base, derived], axis=1)
        hist = add_hist_features(df)

        unique_ids, first_idx, group_counts = np.unique(
            ids, return_index=True, return_counts=True
        )

        # add.reduceat summarize specified ranges
        # for example: np.add.reduceat([1, 4, 5, 7, 8], [0,2], axis=0) = [5, 20]
        group_sums = np.add.reduceat(data, first_idx, axis=0) # sum of strings for each client, first_idx contains idx when data of new client starts
        group_sqs = np.add.reduceat(data * data, first_idx, axis=0)
        group_maxs = np.maximum.reduceat(data, first_idx, axis=0)
        group_mins = np.minimum.reduceat(data, first_idx, axis=0)
        group_firsts = data[first_idx]
        group_lasts = data[first_idx + group_counts - 1]
        group_hist = np.add.reduceat(hist.astype(np.uint16, copy=False), first_idx, axis=0)

        new_ids = unique_ids[~seen[unique_ids]]
        if len(new_ids):
            new_pos = np.searchsorted(unique_ids, new_ids)
            firsts[new_ids] = group_firsts[new_pos]
        seen[unique_ids] = True

        sums[unique_ids] += group_sums
        sqs[unique_ids] += group_sqs
        maxs[unique_ids] = np.maximum(maxs[unique_ids], group_maxs)
        mins[unique_ids] = np.minimum(mins[unique_ids], group_mins)
        lasts[unique_ids] = group_lasts
        hist_sums[unique_ids] += group_hist
        counts[unique_ids] += group_counts.astype(np.uint16)

        rows_seen += len(df)
        log(f"{label}: batch {batch_idx}, rows {rows_seen:,}/{pf.metadata.num_rows:,}")


    n_features = 1 + 6 * n_agg + n_hist
    mmap_path.parent.mkdir(parents=True, exist_ok=True)
    X = np.memmap(mmap_path, dtype=np.float32, mode="w+", shape=(len(ids_needed), n_features)) # matrix on the disk
    chunk = 100_000
    for start in range(0, len(ids_needed), chunk):
        stop = min(start + chunk, len(ids_needed))
        idx = ids_needed[start:stop].astype(np.int64)
        n = counts[idx].astype(np.float32)
        if np.any(n == 0):
            raise RuntimeError(f"{label}: missing {(n == 0).sum()} ids")
        mean = sums[idx] / n[:, None] # broadcasting
        var = np.maximum(sqs[idx] / n[:, None] - mean * mean, 0.0)
        std = np.sqrt(var)
        hist_prop = hist_sums[idx].astype(np.float32) / n[:, None]
        X[start:stop] = np.concatenate(
            [
                n[:, None],
                mean,
                maxs[idx],
                mins[idx],
                std,
                lasts[idx],
                firsts[idx],
                hist_prop,
            ],
            axis=1,
        )


    names = (
        ["records_count"]
        + [f"{c}__mean" for c in AGG_COLUMNS]
        + [f"{c}__max" for c in AGG_COLUMNS]
        + [f"{c}__min" for c in AGG_COLUMNS]
        + [f"{c}__std" for c in AGG_COLUMNS]
        + [f"{c}__last" for c in AGG_COLUMNS]
        + [f"{c}__first" for c in AGG_COLUMNS]
        + [f"{c}__prop_{v}" for c, v in HIST_SPECS]
    )
    log(f"{label}: saved memmap {mmap_path}, shape=({len(ids_needed)}, {n_features})")
    return len(ids_needed), n_features, names


def fit_model(X, y, train_idx=None, valid_idx=None, iterations=950):
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations,
        learning_rate=0.045,
        depth=6,
        l2_leaf_reg=9.0,
        random_seed=RANDOM_STATE,
        od_type="Iter" if valid_idx is not None else None, # to check oh validation dataset
        od_wait=90, #overfitting_detector
        bootstrap_type="Bernoulli",
        subsample=0.88,
        allow_writing_files=False,
        verbose=100,
    )
    if valid_idx is None:
        model.fit(Pool(X, y))
        return model
    model.fit(Pool(X[train_idx], y[train_idx]), eval_set=Pool(X[valid_idx], y[valid_idx]), use_best_model=True)
    return model


def main():
    target = pd.read_csv(TARGET_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    y = target["flag"].to_numpy(np.int8)
    train_ids = target["id"].to_numpy(np.int64)
    test_ids = sample["id"].to_numpy(np.int64)

    train_shape = aggregate_to_memmap(TRAIN_PATH, train_ids, "train-v2", Path("artifacts/train_v2.dat"))
    n_train, n_features, _ = train_shape
    X_train = np.memmap("artifacts/train_v2.dat", dtype=np.float32, mode="r", shape=(n_train, n_features))

    idx_train, idx_valid = train_test_split(
        np.arange(len(y)), test_size=0.16, random_state=RANDOM_STATE, stratify=y
    )
    log("fitting v2 validation model")
    val_model = fit_model(X_train, y, idx_train, idx_valid, iterations=950)
    val_pred = val_model.predict_proba(Pool(X_train[idx_valid]))[:, 1]
    auc = roc_auc_score(y[idx_valid], val_pred)
    best_iter = val_model.get_best_iteration()
    log(f"v2 validation AUC={auc:.6f}, best_iteration={best_iter}")

    final_iterations = int(best_iter + 1) if best_iter is not None and best_iter > 0 else 850
    log(f"fitting v2 final model iterations={final_iterations}")
    final_model = fit_model(X_train, y, iterations=final_iterations)

    test_shape = aggregate_to_memmap(TEST_PATH, test_ids, "test-v2", Path("artifacts/test_v2.dat"))
    n_test, n_features_test, _ = test_shape
    X_test = np.memmap("artifacts/test_v2.dat", dtype=np.float32, mode="r", shape=(n_test, n_features_test))
    pred = final_model.predict_proba(Pool(X_test))[:, 1]
    out = pd.DataFrame({"id": sample["id"].to_numpy(), "flag": np.clip(pred, 0, 1)})
    out["flag"] = out["flag"].round(6)
    out.to_csv(OUT_PATH, index=False, float_format="%.6f")
    log(f"saved {OUT_PATH}, rows={len(out):,}, mean={out['flag'].mean():.6f}")


if __name__ == "__main__":
    main()
