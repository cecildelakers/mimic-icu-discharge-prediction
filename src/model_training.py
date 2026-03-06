"""
SPH6004 Individual Assignment
Author: Chen Wenhao
Date: 2026-02-06

model_training.py
Trains and tests 10 prediction schemes (feature-set × classifier).
Saves per-scheme prediction results to results/predictions/.
Errors are caught so that one failing scheme never blocks the rest.
"""

import os
import traceback
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# ── paths ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PRED_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'predictions')

TRAIN_PATH = os.path.join(DATA_DIR, 'mimic-train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'mimic-test.csv')
TARGET_COL = 'icu_death_flag'


# ── load feature lists from txt files ──────────────────────────────────
def _load_feature_list(filename: str) -> list:
    """Read one-feature-per-line text file."""
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


# ── define the 10 schemes ─────────────────────────────────────────────
def _build_schemes(all_features: list) -> list:
    """Return a list of dicts, each with name / features / model."""
    feat_lasso = _load_feature_list('features_lasso.txt')
    feat_rf = _load_feature_list('features_rf.txt')
    feat_anova = _load_feature_list('features_anova.txt')

    return [
        {"name": "1. All + LogReg",    "features": all_features, "model": LogisticRegression(max_iter=2000, random_state=42)},
        {"name": "2. All + RF",        "features": all_features, "model": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)},
        {"name": "3. Lasso + LogReg",  "features": feat_lasso,   "model": LogisticRegression(max_iter=2000, random_state=42)},
        {"name": "4. Lasso + SVM",     "features": feat_lasso,   "model": SVC(kernel='linear', probability=True, random_state=42)},
        {"name": "5. Lasso + RF",      "features": feat_lasso,   "model": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)},
        {"name": "6. RF_FI + RF",      "features": feat_rf,      "model": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)},
        {"name": "7. RF_FI + LogReg",  "features": feat_rf,      "model": LogisticRegression(max_iter=2000, random_state=42)},
        {"name": "8. RF_FI + SVM",     "features": feat_rf,      "model": SVC(kernel='linear', probability=True, random_state=42)},
        {"name": "9. ANOVA + DecTree", "features": feat_anova,   "model": DecisionTreeClassifier(max_depth=10, random_state=42)},
        {"name": "10. ANOVA + KNN",    "features": feat_anova,   "model": KNeighborsClassifier(n_neighbors=5, n_jobs=-1)},
    ]


# ── train a single scheme and return a result DataFrame ────────────────
def _run_scheme(scheme: dict,
                df_train: pd.DataFrame,
                df_test: pd.DataFrame,
                y_train: pd.Series,
                y_test: pd.Series) -> pd.DataFrame:
    """
    Train the model and produce predictions.
    Returns a DataFrame with columns: [y_true, y_pred, y_prob].
    """
    feats = scheme['features']
    model = scheme['model']

    model.fit(df_train[feats], y_train)

    y_pred = model.predict(df_test[feats])

    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(df_test[feats])[:, 1]
    else:
        y_prob = y_pred.astype(float)

    return pd.DataFrame({
        'y_true': y_test.values,
        'y_pred': y_pred,
        'y_prob': y_prob,
    })


# ── main orchestration ─────────────────────────────────────────────────
def main() -> None:
    os.makedirs(PRED_DIR, exist_ok=True)

    # load data
    print("[Load] Reading train/test data...")
    df_train = pd.read_csv(TRAIN_PATH)
    df_test = pd.read_csv(TEST_PATH)

    y_train = df_train[TARGET_COL]
    y_test = df_test[TARGET_COL]

    all_features = df_train.drop(columns=[TARGET_COL]).columns.tolist()
    schemes = _build_schemes(all_features)

    print(f"  Train samples: {len(y_train)}  |  Test samples: {len(y_test)}")
    print(f"  Total features: {len(all_features)}")
    print(f"  Schemes to run: {len(schemes)}\n")

    # run each scheme
    for i, scheme in enumerate(schemes, start=1):
        tag = f"scheme_{i:02d}"
        out_path = os.path.join(PRED_DIR, f'{tag}.csv')

        print(f"[{tag}] {scheme['name']} ... ", end='', flush=True)

        try:
            result_df = _run_scheme(scheme, df_train, df_test, y_train, y_test)
            result_df.insert(0, 'scheme', scheme['name'])
            result_df.to_csv(out_path, index=False)
            print("OK")

        except Exception:
            # save an error record so evaluation can flag this scheme
            err_msg = traceback.format_exc().strip().split('\n')[-1]
            err_df = pd.DataFrame({
                'scheme': [scheme['name']],
                'y_true': [np.nan],
                'y_pred': [np.nan],
                'y_prob': [np.nan],
                'error': [err_msg],
            })
            err_df.to_csv(out_path, index=False)
            print(f"FAILED  ({err_msg})")

    print(f"\n{'='*50}")
    print(f"All {len(schemes)} schemes processed.")
    print(f"Prediction files saved to results/predictions/")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
