import gc
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
Here I implement another version of catboost prediction, it analizes only one feature 
and aggregates results from different features after prediction
"""

#export DATA_DIR=/Users/arina/Downloads
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
TRAIN_PATH = DATA_DIR / "train_data.parquet"
TEST_PATH = DATA_DIR / "test_data.parquet"
TARGET_PATH = DATA_DIR / "train_target.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
OUT_PATH = Path("submissions/submission_row_blend.csv")

BATCH_SIZE = 900_000
RANDOM_STATE = 2026
NEG_SAMPLE_VALID = 0.38
NEG_SAMPLE_FINAL = 0.48


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


FEATURE_COLUMNS = [c for c in pq.ParquetFile(TRAIN_PATH).schema.names if c != "id"]
CAT_FEATURES = []


def make_target_maps():
    target = pd.read_csv(TARGET_PATH)
    max_id = 3_000_000
    y_map = np.full(max_id + 1, -1, dtype=np.int8)
    y_map[target["id"].to_numpy(np.int64)] = target["flag"].to_numpy(np.int8)
    return target, y_map


def collect_row_sample(y_map, train_client_mask=None, neg_sample=0.4, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    xs = []
    ys = []
    pf = pq.ParquetFile(TRAIN_PATH)
    total = 0
    selected = 0
    positive = 0
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE), start=1):
        df = batch.to_pandas()
        ids = df["id"].to_numpy(np.int64, copy=False)
        labels = y_map[ids]
        mask = labels >= 0
        if train_client_mask is not None:
            mask &= train_client_mask[ids]
        pos = mask & (labels == 1)
        neg = mask & (labels == 0) & (rng.random(len(df)) < neg_sample) # There much more clients not in default, so we can drop part of them
        take = pos | neg

        if take.any():
            x = df.loc[take, FEATURE_COLUMNS].to_numpy(dtype=np.int16, copy=True)
            y = labels[take].astype(np.int8, copy=True)
            xs.append(x)
            ys.append(y)
            selected += len(y)
            positive += int(y.sum())
        total += len(df)
        log(
            f"sample batch {batch_idx}: scanned {total:,}/{pf.metadata.num_rows:,}, "
            f"selected {selected:,}, pos {positive:,}"
        )
        del df, ids, labels, mask, pos, neg, take
        gc.collect()
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    log(f"row sample shape={X.shape}, positive_rate={y.mean():.5f}")
    return X, y


def fit_row_model(X, y, iterations=650, seed=RANDOM_STATE):
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations,
        learning_rate=0.075,
        depth=6,
        l2_leaf_reg=10.0,
        random_seed=seed,
        bootstrap_type="Bernoulli",
        subsample=0.85,
        allow_writing_files=False,
        verbose=100,
    )
    pool = Pool(X.astype(np.float32, copy=False), y, feature_names=FEATURE_COLUMNS)
    model.fit(pool)
    return model


def aggregate_predictions_for_ids(model, parquet_path, ids_needed, label, y_map=None, valid_mask=None):
    max_id = 3_000_000
    counts = np.zeros(max_id + 1, dtype=np.uint16)
    sums = np.zeros(max_id + 1, dtype=np.float32)
    maxs = np.zeros(max_id + 1, dtype=np.float32)
    lasts = np.zeros(max_id + 1, dtype=np.float32)

    pf = pq.ParquetFile(parquet_path)
    total = 0
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=BATCH_SIZE), start=1):
        df = batch.to_pandas()
        ids = df["id"].to_numpy(np.int64, copy=False)
        take = np.ones(len(df), dtype=bool)
        if y_map is not None:
            take &= y_map[ids] >= 0
        if valid_mask is not None:
            take &= valid_mask[ids]
        if not take.any():
            total += len(df)
            continue

        ids_take = ids[take]
        X = df.loc[take, FEATURE_COLUMNS].to_numpy(dtype=np.int16, copy=True)
        pred = model.predict_proba(Pool(X.astype(np.float32, copy=False)))[:, 1].astype(np.float32)

        unique_ids, first_idx, group_counts = np.unique(
            ids_take, return_index=True, return_counts=True
        )
        group_sums = np.add.reduceat(pred, first_idx)
        group_maxs = np.maximum.reduceat(pred, first_idx)
        group_lasts = pred[first_idx + group_counts - 1]

        sums[unique_ids] += group_sums
        maxs[unique_ids] = np.maximum(maxs[unique_ids], group_maxs)
        lasts[unique_ids] = group_lasts
        counts[unique_ids] += group_counts.astype(np.uint16)

        total += len(df)
        log(f"{label} pred batch {batch_idx}: scanned {total:,}/{pf.metadata.num_rows:,}")
        del df, ids, take, ids_take, X, pred, unique_ids, first_idx, group_counts
        del group_sums, group_maxs, group_lasts
        gc.collect()

    idx = ids_needed.astype(np.int64)
    n = counts[idx].astype(np.float32)
    if np.any(n == 0):
        raise RuntimeError(f"{label}: missing predictions for {(n == 0).sum()} ids")
    out = {
        "mean": sums[idx] / n,
        "max": maxs[idx],
        "last": lasts[idx],
    }
    return out


def main():
    target, y_map = make_target_maps()
    ids = target["id"].to_numpy(np.int64)
    y = target["flag"].to_numpy(np.int8)

    train_ids, valid_ids = train_test_split(
        ids, test_size=0.16, random_state=RANDOM_STATE, stratify=y
    )
    train_mask = np.zeros(3_000_001, dtype=bool)
    valid_mask = np.zeros(3_000_001, dtype=bool)
    train_mask[train_ids] = True
    valid_mask[valid_ids] = True
    y_valid = y_map[valid_ids]

    log("collecting validation-training row sample")
    X_row, y_row = collect_row_sample(
        y_map, train_client_mask=train_mask, neg_sample=NEG_SAMPLE_VALID, seed=RANDOM_STATE
    )
    log("fitting validation row model")
    val_model = fit_row_model(X_row, y_row, iterations=650, seed=RANDOM_STATE)
    del X_row, y_row
    gc.collect()

    valid_preds = aggregate_predictions_for_ids(
        val_model,
        TRAIN_PATH,
        valid_ids,
        "valid",
        y_map=y_map,
        valid_mask=valid_mask,
    )
    candidates = {
        "mean": valid_preds["mean"],
        "max": valid_preds["max"],
        "last": valid_preds["last"],
        "0.55mean_0.30max_0.15last": (
            0.55 * valid_preds["mean"] + 0.30 * valid_preds["max"] + 0.15 * valid_preds["last"]
        ),
        "0.45mean_0.45max_0.10last": (
            0.45 * valid_preds["mean"] + 0.45 * valid_preds["max"] + 0.10 * valid_preds["last"]
        ),
    }
    scores = {name: roc_auc_score(y_valid, pred) for name, pred in candidates.items()}
    for name, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        log(f"valid row AUC {name}={score:.6f}")
    best_name = max(scores, key=scores.get)
    log(f"best row aggregator: {best_name}")

    del val_model, valid_preds, candidates
    gc.collect()

    log("collecting final row sample")
    X_final, y_final = collect_row_sample(
        y_map, train_client_mask=None, neg_sample=NEG_SAMPLE_FINAL, seed=RANDOM_STATE + 1
    )
    log("fitting final row model")
    final_model = fit_row_model(X_final, y_final, iterations=650, seed=RANDOM_STATE + 1)
    del X_final, y_final
    gc.collect()

    sample = pd.read_csv(SAMPLE_PATH)
    test_ids = sample["id"].to_numpy(np.int64)
    test_preds = aggregate_predictions_for_ids(final_model, TEST_PATH, test_ids, "test")
    if best_name == "mean":
        pred = test_preds["mean"]
    elif best_name == "max":
        pred = test_preds["max"]
    elif best_name == "last":
        pred = test_preds["last"]
    elif best_name == "0.55mean_0.30max_0.15last":
        pred = 0.55 * test_preds["mean"] + 0.30 * test_preds["max"] + 0.15 * test_preds["last"]
    else:
        pred = 0.45 * test_preds["mean"] + 0.45 * test_preds["max"] + 0.10 * test_preds["last"]

    out = pd.DataFrame({"id": sample["id"].to_numpy(), "flag": np.clip(pred, 0, 1)})
    out["flag"] = out["flag"].round(6)
    out.to_csv(OUT_PATH, index=False, float_format="%.6f")
    log(f"saved {OUT_PATH}, rows={len(out):,}, mean={out['flag'].mean():.6f}")


if __name__ == "__main__":
    main()
