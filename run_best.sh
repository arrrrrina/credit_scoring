#!/usr/bin/env bash
set -euo pipefail

python scripts/02_train_catboost_v2.py
python scripts/01_prepare_sequences.py
python scripts/04_train_legacy_gru.py
python scripts/03_train_concat_gru.py
python scripts/05_train_row_catboost.py
python scripts/06_ensemble.py
python scripts/07_validate_submission.py submissions/submission.csv

