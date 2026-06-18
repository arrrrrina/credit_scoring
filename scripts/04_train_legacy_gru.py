import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch import nn

"""
This is the first version of GRU and it has disadvantage - is summirize embeddings. Different enbeddings have different meaning, 
so it turned out it's not the best solution
"""
#export DATA_DIR=/Users/arina/Downloads
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
SEQ_DIR = Path("artifacts/seq")
TARGET_PATH = DATA_DIR / "train_target.csv"
SAMPLE_PATH = DATA_DIR / "sample_submission.csv"
OUT_PATH = Path("submissions/submission_nn.csv")
CAL_OUT_PATH = Path("submissions/submission_nn_calibrated.csv")
CKPT_PATH = SEQ_DIR / "sequence_nn.pt"

MAX_LEN = 58
N_FEATURES = 60
RANDOM_STATE = 2027
BATCH_SIZE = 2048
EPOCHS = 3


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class SeqNet(nn.Module):
    def __init__(self, max_values, emb_dim=8, proj_dim=48, hidden=48, dropout=0.18):
        super().__init__()
        vocab_sizes = [int(v) + 1 for v in max_values]
        offsets = np.cumsum([0] + vocab_sizes[:-1]).astype(np.int64)
        self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
        self.embedding = nn.Embedding(int(sum(vocab_sizes)), emb_dim)
        self.input = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        out_dim = hidden * 2
        self.attn = nn.Sequential(
            nn.Linear(out_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.head = nn.Sequential(
            nn.Linear(out_dim * 3 + 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 48),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(48, 1),
        )

    def forward(self, x, lengths):
        idx = x.long() + self.offsets.view(1, 1, -1)
        token = self.embedding(idx).sum(dim=2) / math.sqrt(x.shape[-1])
        token = self.input(token)

        lengths_cpu = lengths.detach().cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            token, lengths_cpu, batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(
            packed_out, batch_first=True, total_length=x.shape[1]
        )

        mask = torch.arange(x.shape[1], device=x.device).view(1, -1) < lengths.view(-1, 1)
        score = self.attn(out).squeeze(-1).masked_fill(~mask, -1e4)
        weight = torch.softmax(score, dim=1)
        attn_pool = torch.sum(out * weight.unsqueeze(-1), dim=1)
        mean_pool = torch.sum(out * mask.unsqueeze(-1), dim=1) / lengths.float().view(-1, 1)
        max_pool = out.masked_fill(~mask.unsqueeze(-1), -1e4).amax(dim=1)
        len_feat = torch.stack(
            [lengths.float() / MAX_LEN, torch.log1p(lengths.float()) / math.log1p(MAX_LEN)],
            dim=1,
        )
        return self.head(torch.cat([attn_pool, mean_pool, max_pool, len_feat], dim=1)).squeeze(1)


def load_memmaps():
    target = pd.read_csv(TARGET_PATH)
    sample = pd.read_csv(SAMPLE_PATH)
    y = target["flag"].to_numpy(np.float32)
    train_x = np.memmap(SEQ_DIR / "train_x_uint8.dat", dtype=np.uint8, mode="r", shape=(len(y), MAX_LEN, N_FEATURES))
    train_len = np.load(SEQ_DIR / "train_len_uint8.npy")
    test_x = np.memmap(SEQ_DIR / "test_x_uint8.dat", dtype=np.uint8, mode="r", shape=(len(sample), MAX_LEN, N_FEATURES))
    test_len = np.load(SEQ_DIR / "test_len_uint8.npy")
    max_values = np.maximum(train_x.reshape(-1, N_FEATURES).max(axis=0), test_x.reshape(-1, N_FEATURES).max(axis=0))
    return target, sample, y, train_x, train_len, test_x, test_len, max_values


def batch_arrays(x_mmap, len_arr, y_arr, indices, device):
    xb = torch.from_numpy(np.asarray(x_mmap[indices], dtype=np.int64)).to(device)
    lb = torch.from_numpy(np.asarray(len_arr[indices], dtype=np.int64)).to(device)
    if y_arr is None:
        return xb, lb, None
    yb = torch.from_numpy(np.asarray(y_arr[indices], dtype=np.float32)).to(device)
    return xb, lb, yb


@torch.no_grad()
def predict(model, x_mmap, len_arr, indices, device):
    model.eval()
    preds = np.empty(len(indices), dtype=np.float32)
    for start in range(0, len(indices), BATCH_SIZE * 2):
        stop = min(start + BATCH_SIZE * 2, len(indices))
        xb, lb, _ = batch_arrays(x_mmap, len_arr, None, indices[start:stop], device)
        logits = model(xb, lb)
        preds[start:stop] = torch.sigmoid(logits).detach().cpu().numpy()
    return preds


def calibrate_to_catboost(ids, raw_pred):
    reference = pd.read_csv("submissions/submission_v2.csv")
    if not np.array_equal(ids, reference["id"].to_numpy()):
        raise ValueError("CatBoost and NN id order differs")
    order = np.argsort(pd.Series(raw_pred).rank(method="average").to_numpy(), kind="mergesort")
    calibrated = np.empty(len(ids), dtype=np.float64)
    calibrated[order] = np.sort(reference["flag"].to_numpy(float))
    pd.DataFrame({"id": ids, "flag": np.round(calibrated, 6)}).to_csv(
        CAL_OUT_PATH, index=False, float_format="%.6f"
    )


def main():
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.set_num_threads(4)
    device = choose_device()
    log(f"device={device}, torch={torch.__version__}")

    target, sample, y, train_x, train_len, test_x, test_len, max_values = load_memmaps()
    train_idx, valid_idx = train_test_split(
        np.arange(len(y)), test_size=0.16, random_state=RANDOM_STATE, stratify=y
    )
    pos_weight = float((len(train_idx) - y[train_idx].sum()) / max(y[train_idx].sum(), 1))
    log(f"train={len(train_idx):,}, valid={len(valid_idx):,}, pos_weight={pos_weight:.2f}")

    model = SeqNet(max_values=max_values).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, EPOCHS * math.ceil(len(train_idx) / BATCH_SIZE))
    )

    best_auc = -1.0
    best_epoch = -1
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = train_idx.copy()
        np.random.default_rng(RANDOM_STATE + epoch).shuffle(order)
        total_loss = 0.0
        seen = 0
        t0 = time.time()
        for step, start in enumerate(range(0, len(order), BATCH_SIZE), start=1):
            stop = min(start + BATCH_SIZE, len(order))
            idx = order[start:stop]
            xb, lb, yb = batch_arrays(train_x, train_len, y, idx, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb, lb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()
            bs = stop - start
            total_loss += float(loss.item()) * bs
            seen += bs
            if step % 100 == 0:
                log(f"epoch {epoch} step {step}, loss={total_loss/seen:.5f}, seen={seen:,}")

        val_pred = predict(model, train_x, train_len, valid_idx, device)
        auc = roc_auc_score(y[valid_idx], val_pred)
        log(f"epoch {epoch} done in {(time.time()-t0)/60:.1f} min, loss={total_loss/seen:.5f}, valid_auc={auc:.6f}")
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "max_values": max_values, "auc": auc}, CKPT_PATH)
            log(f"saved best checkpoint: auc={auc:.6f}")

    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    log(f"best_epoch={best_epoch}, best_auc={best_auc:.6f}")
    test_indices = np.arange(len(sample))
    pred = predict(model, test_x, test_len, test_indices, device)
    out = pd.DataFrame({"id": sample["id"].to_numpy(), "flag": np.round(np.clip(pred, 0, 1), 6)})
    out.to_csv(OUT_PATH, index=False, float_format="%.6f")
    calibrate_to_catboost(sample["id"].to_numpy(), pred)
    log(f"saved {OUT_PATH} and {CAL_OUT_PATH}, rows={len(out):,}, mean={out['flag'].mean():.6f}")


if __name__ == "__main__":
    main()
