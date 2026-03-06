"""
SPH6004 Individual Assignment
Author: Chen Wenhao
Date: 2026-02-06

feature_selection.py
Executes three feature-selection strategies (Lasso, Random Forest, ANOVA)
on the preprocessed training data, plots feature importance charts,
and saves the selected feature lists.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif


# ── paths ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')

TRAIN_PATH = os.path.join(DATA_DIR, 'mimic-train.csv')
TARGET_COL = 'icu_death_flag'

# RF and ANOVA select top-K features; Lasso decides automatically via L1
K_FEATURES = 30


# ── Strategy 1: Lasso (L1 regularisation) ─────────────────────────────
def select_lasso(X: pd.DataFrame, y: pd.Series) -> list:
    """Fit L1-penalised logistic regression; return features with non-zero coefficients."""
    model = LogisticRegression(
        penalty='l1', solver='saga', C=0.1,
        random_state=42, max_iter=2000, n_jobs=-1,
    )
    model.fit(X, y)

    coefs = np.abs(model.coef_[0])
    selected = X.columns[coefs > 0].tolist()
    print(f"  Lasso retained {len(selected)} features (auto).")

    # plot top-20 coefficients
    imp_df = (pd.DataFrame({'Feature': X.columns, 'Importance': coefs})
              .query('Importance > 0')
              .sort_values('Importance', ascending=True)
              .tail(20))

    plt.figure(figsize=(10, 8))
    plt.barh(imp_df['Feature'], imp_df['Importance'], color='steelblue')
    plt.title('Top 20 Features Selected by Lasso (L1 Penalty)')
    plt.xlabel('Absolute Coefficient')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig1_lasso_importance.pdf'), dpi=300)
    plt.close()

    return selected


# ── Strategy 2: Random Forest Gini importance ─────────────────────────
def select_rf(X: pd.DataFrame, y: pd.Series, k: int = K_FEATURES) -> list:
    """Fit a Random Forest and return the top-k features by Gini importance."""
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X, y)

    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:k]
    selected = X.columns[top_idx].tolist()
    print(f"  Random Forest selected top {k} features.")

    # plot top-20 importances
    plot_feats = X.columns[top_idx][:20]
    plot_imps = importances[top_idx][:20]
    imp_df = (pd.DataFrame({'Feature': plot_feats, 'Importance': plot_imps})
              .sort_values('Importance', ascending=True))

    plt.figure(figsize=(10, 8))
    plt.barh(imp_df['Feature'], imp_df['Importance'], color='darkorange')
    plt.title('Top 20 Features by Random Forest Gini Importance')
    plt.xlabel('Feature Importance Score')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'fig2_rf_importance.pdf'), dpi=300)
    plt.close()

    return selected


# ── Strategy 3: ANOVA F-test (SelectKBest) ────────────────────────────
def select_anova(X: pd.DataFrame, y: pd.Series, k: int = K_FEATURES) -> list:
    """Use one-way ANOVA F-value to select the top-k features."""
    selector = SelectKBest(score_func=f_classif, k=k)
    selector.fit(X, y)

    selected = X.columns[selector.get_support()].tolist()
    print(f"  ANOVA selected top {k} features.")

    return selected


# ── helper: save a feature list to a text file ─────────────────────────
def _save_feature_list(features: list, filename: str) -> None:
    """Write one feature name per line."""
    path = os.path.join(DATA_DIR, filename)
    with open(path, 'w') as f:
        f.write('\n'.join(features))


# ── main orchestration ─────────────────────────────────────────────────
def main() -> None:
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # load preprocessed training set
    print("[Load] Reading preprocessed training data...")
    df_train = pd.read_csv(TRAIN_PATH)
    X_train = df_train.drop(columns=[TARGET_COL])
    y_train = df_train[TARGET_COL]
    print(f"  Total features: {X_train.shape[1]}  |  samples: {X_train.shape[0]}")

    # Strategy 1 – Lasso
    print("\n[Strategy 1] Lasso (L1 Regularisation)")
    lasso_feats = select_lasso(X_train, y_train)
    _save_feature_list(lasso_feats, 'features_lasso.txt')

    # Strategy 2 – Random Forest
    print("\n[Strategy 2] Random Forest Importance")
    rf_feats = select_rf(X_train, y_train)
    _save_feature_list(rf_feats, 'features_rf.txt')

    # Strategy 3 – ANOVA
    print("\n[Strategy 3] ANOVA F-value")
    anova_feats = select_anova(X_train, y_train)
    _save_feature_list(anova_feats, 'features_anova.txt')

    # summary
    print(f"\n{'='*50}")
    print("Feature selection complete.")
    print(f"  Lasso  : {len(lasso_feats)} features  ->  features_lasso.txt")
    print(f"  RF     : {len(rf_feats)} features  ->  features_rf.txt")
    print(f"  ANOVA  : {len(anova_feats)} features  ->  features_anova.txt")
    print(f"  Figures saved to results/figures/")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
