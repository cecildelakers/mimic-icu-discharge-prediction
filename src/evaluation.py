"""
SPH6004 Individual Assignment
Author: Chen Wenhao
Date: 2026-02-06

evaluation.py
Reads per-scheme prediction CSVs from results/predictions/,
computes classification metrics, writes a summary to results/metrics.csv,
and plots a combined ROC curve figure.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


# ── paths ──────────────────────────────────────────────────────────────
PRED_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'predictions')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
METRICS_PATH = os.path.join(RESULTS_DIR, 'metrics.csv')


# ── metric computation ─────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_prob) -> dict:
    """Return a dict of classification metrics."""
    return {
        'Accuracy':  accuracy_score(y_true, y_pred),
        'ROC-AUC':   roc_auc_score(y_true, y_prob),
        'F1-Score':  f1_score(y_true, y_pred, zero_division=0),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall':    recall_score(y_true, y_pred, zero_division=0),
    }


# ── ROC curve plotting ─────────────────────────────────────────────────
def plot_roc_curves(pred_files: list) -> None:
    """Plot ROC curves for all successful schemes on a single figure."""
    plt.figure(figsize=(12, 10))

    for path in pred_files:
        df = pd.read_csv(path)
        if 'error' in df.columns:
            continue

        scheme_name = df['scheme'].iloc[0]
        auc = roc_auc_score(df['y_true'], df['y_prob'])
        fpr, tpr, _ = roc_curve(df['y_true'], df['y_prob'])
        plt.plot(fpr, tpr, lw=2, label=f"{scheme_name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    # plt.title('Receiver Operating Characteristic (ROC) Curves across 10 Schemes', fontsize=20)
    plt.legend(loc='lower right', fontsize=16)
    plt.tick_params(axis='both', labelsize=16)
    plt.tight_layout()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    save_path = os.path.join(FIGURES_DIR, 'fig3_roc_comparison.pdf')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"\n[Plot] ROC curve saved -> results/figures/fig3_roc_comparison.pdf")


# ── main orchestration ─────────────────────────────────────────────────
def main() -> None:
    # only read scheme_*.csv files inside predictions folder
    pattern = os.path.join(PRED_DIR, 'scheme_*.csv')
    pred_files = sorted(glob.glob(pattern))

    if not pred_files:
        print(f"[Error] No prediction files found in {PRED_DIR}")
        return

    print(f"[Load] Found {len(pred_files)} prediction file(s) in results/predictions/\n")

    rows = []

    for path in pred_files:
        df = pd.read_csv(path)
        scheme_name = df['scheme'].iloc[0]
        tag = os.path.basename(path)

        # check for error column (scheme failed during training)
        if 'error' in df.columns:
            print(f"  [{tag}] {scheme_name} — SKIPPED (training error)")
            rows.append({
                'Scheme': scheme_name,
                'Accuracy': float('nan'),
                'ROC-AUC': float('nan'),
                'F1-Score': float('nan'),
                'Precision': float('nan'),
                'Recall': float('nan'),
            })
            continue

        # compute metrics
        metrics = compute_metrics(df['y_true'], df['y_pred'], df['y_prob'])
        metrics['Scheme'] = scheme_name
        rows.append(metrics)

        print(f"  [{tag}] {scheme_name}  "
              f"AUC={metrics['ROC-AUC']:.4f}  F1={metrics['F1-Score']:.4f}")

    # assemble final table
    col_order = ['Scheme', 'Accuracy', 'ROC-AUC', 'F1-Score', 'Precision', 'Recall']
    metrics_df = pd.DataFrame(rows)[col_order]

    metrics_df.to_csv(METRICS_PATH, index=False)

    print(f"\n{'='*80}")
    print("Model Performance Summary")
    print('='*80)
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print('='*80)
    print(f"\nSaved to {os.path.basename(METRICS_PATH)}")

    # plot combined ROC curves
    plot_roc_curves(pred_files)


if __name__ == '__main__':
    main()
