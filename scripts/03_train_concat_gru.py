import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

"""
Here I implement bidirectional gru with embeggings' concatenation
"""
#export DATA_DIR=/Users/arina/Downloads
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
SEQ_DIR = Path("artifacts/seq")
TARGET_PATH = DATA_DIR / "train_target.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
OUT_PATH = Path("submissions/submission_nn_concat.csv")
CAL_OUT_PATH = Path("submissions/submission_nn_concat_calibrated.csv")
BEST_CKPT_PATH = SEQ_DIR / "sequence_concat_best.pt"

MAX_LEN = 58
N_FEATURES = 60
BATCH_SIZE = 3072
VALID_EPOCHS = 4
FINAL_EPOCHS_CAP = 6
RANDOM_STATE = 137




def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


class ConcatGRU(nn.Module):
    def __init__(self, max_values: np.ndarray, emb_dim=3, projection_dim=96, hidden=64, dropout=0.15):
        super().__init__()
        vocab_sizes = (max_values.astype(np.int64) + 2).tolist() # we need to add 2 because value = 1 is reserved for padding
        offsets = np.cumsum([0] + vocab_sizes[:-1]).astype(np.int64) # to separate features in embedding space
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.embedding = nn.Embedding(int(sum(vocab_sizes)), emb_dim)
        concat_dim = int(len(max_values) * emb_dim)
        self.embedding_dropout = nn.Dropout1d(dropout)
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, projection_dim),
            nn.LayerNorm(projection_dim),
            nn.GELU(), 
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=projection_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        out_dim = hidden * 2 # because of bidirectional gru
        self.attention = nn.Linear(out_dim, 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim * 4 + 1, 160),
            nn.LayerNorm(160),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(160, 48),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(48, 1),
        )

    def forward(self, x, lengths):
        z = self.embedding(x.long() + self.offsets.view(1, 1, -1))
        z = z.reshape(x.shape[0], x.shape[1], -1) # batch х 58 х 60 х 3 -> batch х 58 х 180
        z = self.embedding_dropout(z.transpose(1, 2)).transpose(1, 2)
        z = self.projection(z)

        lengths_cpu = lengths.detach().cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            z, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed) # returns output, final_hidden_state
        states, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.shape[1]
        )

        max_length = states.shape[1]
        mask = torch.arange(max_length, device=states.device)[None, :] < lengths[:, None] # we need mask to differ real state from padding
        float_mask = mask.unsqueeze(-1).float()
        mean_pool = (states * float_mask).sum(dim=1) / lengths.float().clamp_min(1).unsqueeze(-1) 
        max_pool = states.masked_fill(~mask.unsqueeze(-1), -1e9).amax(dim=1)
        last_idx = (lengths - 1).clamp_min(0)
        last_pool = states[torch.arange(states.shape[0], device=states.device), last_idx]
        attn_logits = self.attention(states).squeeze(-1).masked_fill(~mask, -1e9)
        attn = torch.softmax(attn_logits, dim=1)
        attn_pool = (states * attn.unsqueeze(-1)).sum(dim=1)
        length_feat = (lengths.float() / MAX_LEN).unsqueeze(1)
        return self.head(
            torch.cat([last_pool, mean_pool, max_pool, attn_pool, length_feat], dim=1)
        ).squeeze(1)


def load_data():
    target = pd.read_csv(TARGET_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    y = target["flag"].to_numpy(np.float32)
    train_x = np.memmap(
        SEQ_DIR / "train_x_uint8.dat", dtype=np.uint8, mode="r", shape=(len(y), MAX_LEN, N_FEATURES)
    )
    train_len = np.load(SEQ_DIR / "train_len_uint8.npy")
    test_x = np.memmap(
        SEQ_DIR / "test_x_uint8.dat", dtype=np.uint8, mode="r", shape=(len(sample), MAX_LEN, N_FEATURES)
    )
    test_len = np.load(SEQ_DIR / "test_len_uint8.npy")
    max_values = np.maximum(
        train_x.reshape(-1, N_FEATURES).max(axis=0),
        test_x.reshape(-1, N_FEATURES).max(axis=0),
    )
    return target, sample, y, train_x, train_len, test_x, test_len, max_values


def make_splits(target: pd.DataFrame, y: np.ndarray):
    ids = target["id"].to_numpy(np.int64)
    order = np.argsort(ids, kind="mergesort")
    valid_size = int(round(len(ids) * 0.16))
    time_valid = order[-valid_size:]
    time_train = order[:-valid_size]

    rng = np.random.default_rng(RANDOM_STATE)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    rng.shuffle(pos)
    rng.shuffle(neg)
    random_valid = np.concatenate(
        [pos[: int(len(pos) * 0.16)], neg[: int(len(neg) * 0.16)]]
    )
    random_train_mask = np.ones(len(y), dtype=bool)
    random_train_mask[random_valid] = False
    random_train = np.flatnonzero(random_train_mask)
    rng.shuffle(random_valid)
    rng.shuffle(random_train)
    return {
        "time": (time_train, time_valid),
        "random": (random_train, random_valid),
    }


def batch_tensors(x_mmap, len_arr, y_arr, indices, dev):
    xb = torch.from_numpy(np.asarray(x_mmap[indices], dtype=np.int64)).to(dev)
    lb = torch.from_numpy(np.asarray(len_arr[indices], dtype=np.int64)).to(dev)
    mask = torch.arange(MAX_LEN, device=dev)[None, :] < lb[:, None]
    xb = xb + 1
    xb = xb.masked_fill(~mask.unsqueeze(-1), 0)
    if y_arr is None:
        return xb, lb, None
    yb = torch.from_numpy(np.asarray(y_arr[indices], dtype=np.float32)).to(dev)
    return xb, lb, yb


@torch.no_grad()
def predict(model, x_mmap, len_arr, indices, dev, batch_size=BATCH_SIZE * 2):
    model.eval()
    pred = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        xb, lb, _ = batch_tensors(x_mmap, len_arr, None, indices[start:stop], dev)
        pred[start:stop] = torch.sigmoid(model(xb, lb)).detach().cpu().numpy()
    return pred


def train_epoch(model, optimizer, criterion, scheduler, x_mmap, len_arr, y, indices, dev, epoch):
    model.train()
    order = indices.copy()
    np.random.default_rng(RANDOM_STATE + epoch).shuffle(order)
    total_loss = 0.0
    seen = 0
    for step, start in enumerate(range(0, len(order), BATCH_SIZE), start=1):
        stop = min(start + BATCH_SIZE, len(order))
        xb, lb, yb = batch_tensors(x_mmap, len_arr, y, order[start:stop], dev)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(xb, lb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        bs = stop - start
        total_loss += float(loss.item()) * bs
        seen += bs
        if step % 150 == 0:
            log(f"epoch {epoch} step {step}, loss={total_loss/seen:.5f}, seen={seen:,}")
    return total_loss / max(seen, 1)


def run_validation_training(split_name, train_idx, valid_idx, y, train_x, train_len, max_values, dev):
    model = ConcatGRU(max_values).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    criterion = nn.BCEWithLogitsLoss()
    steps_per_epoch = math.ceil(len(train_idx) / BATCH_SIZE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, VALID_EPOCHS * steps_per_epoch), eta_min=1.0e-5
    )
    best_auc = -1.0
    best_epoch = 0
    patience = 2
    bad_epochs = 0
    for epoch in range(1, VALID_EPOCHS + 1):
        t0 = time.time()
        loss = train_epoch(
            model, optimizer, criterion, scheduler, train_x, train_len, y, train_idx, dev, epoch
        )
        pred = predict(model, train_x, train_len, valid_idx, dev)
        auc = roc_auc_score(y[valid_idx], pred)
        log(
            f"{split_name} epoch {epoch}: loss={loss:.5f}, auc={auc:.6f}, "
            f"minutes={(time.time()-t0)/60:.1f}"
        )
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            torch.save(
                {"model": model.state_dict(), "auc": auc, "epoch": epoch, "max_values": max_values},
                BEST_CKPT_PATH,
            )
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    return best_auc, best_epoch


def train_final(best_epochs, y, train_x, train_len, max_values, dev):
    model = ConcatGRU(max_values).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    criterion = nn.BCEWithLogitsLoss()
    all_idx = np.arange(len(y))
    epochs = max(1, min(int(best_epochs), FINAL_EPOCHS_CAP))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs * math.ceil(len(all_idx) / BATCH_SIZE)),
        eta_min=1.0e-5,
    )
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss = train_epoch(
            model, optimizer, criterion, scheduler, train_x, train_len, y, all_idx, dev, epoch
        )
        log(f"final epoch {epoch}/{epochs}: loss={loss:.5f}, minutes={(time.time()-t0)/60:.1f}")
    return model

# compare with catboost
def calibrate_to_reference(ids, raw_pred, reference_path, out_path):
    ref = pd.read_csv(reference_path)
    assert np.array_equal(ids, ref["id"].to_numpy())
    rank = pd.Series(raw_pred).rank(method="average", pct=True).to_numpy()
    order = np.argsort(rank, kind="mergesort")
    probs = np.sort(ref["flag"].to_numpy(float))
    calibrated = np.empty_like(probs)
    calibrated[order] = probs
    out = pd.DataFrame({"id": ids, "flag": np.round(calibrated, 6)})
    out.to_csv(out_path, index=False, float_format="%.6f")
    return calibrated


def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.set_num_threads(4)
    dev = device()
    log(f"device={dev}, batch_size={BATCH_SIZE}")
    target, sample, y, train_x, train_len, test_x, test_len, max_values = load_data()
    log(f"embedding dims total={len(max_values) * 3}")

    splits = make_splits(target, y)
    random_auc, random_epoch = run_validation_training(
        "random", *splits["random"], y, train_x, train_len, max_values, dev
    )
    time_auc, time_epoch = -1.0, 0
    log(
        f"validation summary: random_auc={random_auc:.6f} epoch={random_epoch}; "
        f"time_auc={time_auc:.6f} epoch={time_epoch}"
    )

    final_epochs = max(random_epoch, 3)
    log(f"training final model on all train for {final_epochs} epochs")
    final_model = train_final(final_epochs, y, train_x, train_len, max_values, dev)
    test_idx = np.arange(len(sample))
    raw = predict(final_model, test_x, test_len, test_idx, dev)
    raw_out = pd.DataFrame({"id": sample["id"].to_numpy(), "flag": np.round(np.clip(raw, 0, 1), 6)})
    raw_out.to_csv(OUT_PATH, index=False, float_format="%.6f")
    calibrate_to_reference(
        sample["id"].to_numpy(),
        raw,
        Path("submissions/submission_v2.csv"),
        CAL_OUT_PATH,
    )
    log(
        f"saved raw={OUT_PATH}, calibrated={CAL_OUT_PATH}, "
        f"raw_mean={raw.mean():.6f}, raw_min={raw.min():.6f}, raw_max={raw.max():.6f}"
    )


if __name__ == "__main__":
    main()
