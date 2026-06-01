# Credit Card Fraud Detection

Machine learning project for detecting fraudulent credit card transactions using the [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset.

The goal is to learn a complete, leakage-aware ML workflow — not just maximize a leaderboard score.

## Project structure

```
CreditCardFraudDetection/
├── src/                  # Reusable pipeline code
│   ├── config.py         # Constants and split ratios
│   ├── data_loader.py    # Load CSV, dedupe, feature columns
│   ├── splits.py         # Stratified and time-based splits
│   ├── evaluation.py     # Metrics and threshold tuning
│   ├── imbalance.py      # class_weight helpers, SMOTE
│   └── models.py         # Train/predict RF, XGBoost, LightGBM
├── tests/                # Unit tests (pytest)
├── train.py              # CLI training script
├── eda.ipynb             # Exploratory analysis + step-by-step workflow
├── requirements.txt
└── data/                 # creditcard.csv (not tracked in git)
```

## Setup

1. **Clone and enter the project**

```bash
cd CreditCardFraudDetection
```

2. **Create a virtual environment**

```bash
python3 -m venv .fraud
source .fraud/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download the dataset**

Place `creditcard.csv` in the `data/` folder (from Kaggle).

## ML workflow (notebook)

Open `eda.ipynb` and run cells top-to-bottom. The notebook walks through:

| Phase | Topic |
|---|---|
| 0 | Evaluation foundation — stratified split, metrics |
| 3 | Dummy classifier baseline |
| 4–6 | Random Forest, XGBoost, LightGBM |
| 7 | Threshold tuning on validation |
| 8 | One-time test evaluation |
| 9 | Time-based split comparison |
| 10 | SMOTE vs class weights |

**Primary metric:** PR-AUC (Average Precision)  
**Secondary metric:** ROC-AUC

Run cell 2 first — it loads the `src/` modules.

## Train from the command line

```bash
# Default: Random Forest, stratified split, class_weight
python train.py

# XGBoost with time-based split
python train.py --model xgb --split time

# Random Forest with SMOTE (train only)
python train.py --model rf --imbalance smote

# Save model + metrics to artifacts/
python train.py --model lgb --output-dir artifacts --quiet
```

### CLI options

| Flag | Choices | Default | Description |
|---|---|---|---|
| `--model` | `rf`, `xgb`, `lgb` | `rf` | Model type |
| `--split` | `stratified`, `time` | `stratified` | Split strategy |
| `--imbalance` | `class_weight`, `smote` | `class_weight` | Imbalance handling |
| `--recall-target` | float | `0.80` | Target recall for threshold |
| `--data-dir` | path | `./data` | Data directory |
| `--output-dir` | path | `artifacts` | Save model + metrics JSON |
| `--quiet` | flag | off | Reduce booster training logs |

Outputs in `artifacts/`:
- `metrics.json` — validation and test metrics
- `model.joblib` / `model.xgb` / `model.lgb` — trained model

## Run tests

```bash
pytest tests/ -v
```

Tests use a small synthetic dataset — they do **not** require `data/creditcard.csv`.

## Key design rules

1. **Drop duplicates** before splitting
2. **Stratified split** for model development; **time-based split** for realistic evaluation
3. **Tune threshold on validation** — evaluate test **once**
4. **SMOTE on train only** — never on validation or test
5. **Use probability scores** for ROC-AUC and PR-AUC, not hard 0/1 predictions

## Dependencies

See `requirements.txt`. Core packages: pandas, scikit-learn, xgboost, lightgbm, imbalanced-learn.
