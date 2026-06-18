"""Build the exact rank blend used for the best submitted file."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


WEIGHTS = {
    "submission_nn_concat.csv": 0.70, # concat_gru
    "submission_v2.csv": 0.20, # concat_catboost
    "submission_nn_calibrated.csv": 0.07, # legacy GRU
    "submission_row_blend.csv": 0.03, #row_catboost
}


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("submissions"))
    parser.add_argument("--output", type=Path, default=Path("submissions/submission.csv"))
    args = parser.parse_args()

    frames = {name: pd.read_csv(args.input_dir / name) for name in WEIGHTS}
    reference = frames["submission_v2.csv"]
    ids = reference["id"].to_numpy()
    for name, frame in frames.items():
        if list(frame.columns) != ["id", "flag"] or not np.array_equal(ids, frame["id"]):
            raise ValueError(f"{name}: columns or id order do not match submission_v2.csv")

    score = sum(
        WEIGHTS[name] * percentile_rank(frame["flag"].to_numpy(float))
        for name, frame in frames.items()
    )
    order = np.argsort(score, kind="mergesort")
    calibrated = np.empty(len(ids), dtype=np.float64)
    calibrated[order] = np.sort(reference["flag"].to_numpy(float))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "flag": np.round(calibrated, 6)}).to_csv(
        args.output, index=False, float_format="%.6f"
    )
    print(f"saved {args.output} ({args.output.stat().st_size / 1_000_000:.2f} MB)")


if __name__ == "__main__":
    main()

