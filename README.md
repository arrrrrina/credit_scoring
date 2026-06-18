# Кредитный скоринг

Проект решает задачу бинарной классификации: по кредитной истории
клиента нужно оценить вероятность дефолта.

Целевая переменная `flag`:

- `0` — дефолта нет;
- `1` — дефолт произошёл.

У каждого клиента может быть до 58 записей истории. Каждая запись содержит
60 закодированных признаков, включая `rn` — порядковый номер записи.

Основная метрика качества — ROC-AUC. Лучшая отдельная модель в проекте —
`ConcatGRU` с локальным ROC-AUC `0.769945`. Это результат на случайной
стратифицированной validation-выборке.

## Структура проекта

```text
credit_scoring/
├── data/                         # исходные данные
├── artifacts/                    # промежуточные массивы и checkpoints
├── submissions/                  # прогнозы моделей
├── scripts/
│   ├── 01_prepare_sequences.py   # parquet -> последовательности для GRU
│   ├── 02_train_catboost_v2.py   # агрегирование клиента + CatBoost
│   ├── 03_train_concat_gru.py    # лучшая отдельная модель
│   ├── 04_train_legacy_gru.py    # первая версия GRU
│   ├── 05_train_row_catboost.py  # CatBoost по отдельным записям
│   └── 06_ensemble.py            # ранговый ансамбль
├── requirements.txt
└── run_best.sh                   # полный запуск
```

## Установка

```bash
git clone git@github.com:arrrrrina/credit_scoring.git

cd credit_scoring

python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

После активации окружения команда `python` должна быть доступна:

```bash
python --version
```

## Исходные данные

По умолчанию в папке `data/` должны лежать:

```text
data/
├── train_data.parquet
├── test_data.parquet
├── train_target.csv
└── sample_submission.csv
```

Если данные лежат в другой папке, укажите её через `DATA_DIR`:

```bash
export DATA_DIR="/path/to/data"
```

В `DATA_DIR` должны находиться все четыре указанных файла.

## Полный запуск

Сначала активируйте виртуальное окружение:

```bash
source .venv/bin/activate
```

Затем запустите:

```bash
./run_best.sh
```
`run_best.sh` последовательно запускает все модели, строит ансамбль и генерирует финальный файл submission.csv.

## Запуск по шагам

Полный pipeline можно запустить вручную:

```bash
# 1. Агрегирование признаков клиента и CatBoost v2
python scripts/02_train_catboost_v2.py

# 2. Подготовка последовательностей для GRU
python scripts/01_prepare_sequences.py

# 3. Первая GRU
python scripts/04_train_legacy_gru.py

# 4. Лучшая ConcatGRU
python scripts/03_train_concat_gru.py

# 5. CatBoost по отдельным записям
python scripts/05_train_row_catboost.py

# 6. Ансамбль
python scripts/06_ensemble.py

# 7. Проверка итогового файла
python scripts/07_validate_submission.py submissions/submission.csv
```

Порядок важен: обе GRU используют `submission_v2.csv` для ранговой калибровки, поэтому
`02_train_catboost_v2.py` запускается первым.

## Запуск только лучшей модели

Для обучения `ConcatGRU` нужны последовательности и опорный CatBoost-прогноз:

```bash
python scripts/02_train_catboost_v2.py
python scripts/01_prepare_sequences.py
python scripts/03_train_concat_gru.py
```

Результаты:

```text
submissions/submission_nn_concat.csv
submissions/submission_nn_concat_calibrated.csv
```

## Что создаётся в `artifacts/`

`artifacts/` хранит не исходные данные, а тяжёлые промежуточные файлы:

```text
artifacts/
├── train_v2.dat                  # 566 CatBoost-признаков train
├── test_v2.dat                   # 566 CatBoost-признаков test
└── seq/
    ├── train_x_uint8.dat         # train: клиенты × 58 × 60
    ├── test_x_uint8.dat          # test: клиенты × 58 × 60
    ├── train_len_uint8.npy       # длины train-историй
    ├── test_len_uint8.npy        # длины test-иторий
    ├── meta.pkl                   # метаданные
    ├── sequence_nn.pt            # checkpoint Legacy GRU
    └── sequence_concat_best.pt   # checkpoint ConcatGRU
```

Файлы `.dat` читаются через `numpy.memmap`: в оперативную память загружается только нужная
часть массива.

## Модели

### CatBoost v2

Скрипт `02_train_catboost_v2.py` превращает всю историю клиента в одну строку из 566 признаков:

- количество записей;
- среднее, минимум и максимум;
- стандартное отклонение;
- первое и последнее значения;
- доли категориальных значений;
- статистики платёжной истории.

После этого обучается CatBoost глубиной 6.

### Legacy GRU

`04_train_legacy_gru.py` получает всю историю клиента как последовательность. Embeddings 60
признаков складываются, затем передаются в двунаправленную GRU.

Итоговое описание клиента собирается через:

- attention pooling;
- mean pooling;
- max pooling;
- длину истории.

### ConcatGRU

`03_train_concat_gru.py` — лучшая отдельная модель. Её архитектура:

```text
история: 58 × 60
        ↓
embedding размера 3 для каждого признака
        ↓
конкатенация: 60 × 3 = 180
        ↓
Linear: 180 → 96, LayerNorm, GELU, Dropout
        ↓
Bidirectional GRU, hidden size 64
        ↓
128 признаков на каждой позиции истории
        ↓
last + mean + max + attention pooling
        ↓
512 признаков + длина истории
        ↓
MLP: 513 → 160 → 48 → 1
        ↓
вероятность дефолта
```

В отличие от Legacy GRU, embeddings разных полей не складываются, а ставятся рядом.
Благодаря этому модель лучше сохраняет информацию о том, какому столбцу принадлежит каждое
значение.

### Row-level CatBoost

`05_train_row_catboost.py` обучает CatBoost на отдельных записях истории. Каждая запись получает
`flag` своего клиента. Прогнозы записей затем агрегируются на уровне клиента через:

- `mean` — средний риск;
- `max` — максимальный риск;
- `last` — риск последней записи;
- взвешенные смеси этих трёх вариантов.

## Итоговый ансамбль

Скрипт `06_ensemble.py` сначала превращает прогнозы каждой модели в процентильные ранги,
а затем смешивает их:

| Модель | Вес |
|---|---:|
| ConcatGRU | 70% |
| CatBoost v2 | 20% |
| Legacy GRU | 7% |
| Row-level CatBoost | 3% |

После смешивания порядок клиентов переносится на отсортированное распределение вероятностей
CatBoost v2. Это сохраняет ранжирование ансамбля, но делает итоговое распределение вероятностей
похожим на CatBoost.

Итоговый файл:

```text
submissions/submission.csv
```


## Основные выходные файлы

| Файл | Содержимое |
|---|---|
| `submission_v2.csv` | CatBoost v2 |
| `submission_nn.csv` | Legacy GRU |
| `submission_nn_calibrated.csv` | Legacy GRU с ранговой калибровкой |
| `submission_nn_concat.csv` | ConcatGRU |
| `submission_nn_concat_calibrated.csv` | ConcatGRU с ранговой калибровкой |
| `submission_row_blend.csv` | Row-level CatBoost |
| `submission.csv` | итоговый ансамбль |

