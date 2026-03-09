# MIMIC ICU Mortality Prediction

> **SPH6004 Individual Assignment** — Author: Chen Wenhao

A complete machine-learning pipeline for predicting **ICU in-hospital mortality** using the [MIMIC-IV](https://physionet.org/content/mimiciv/) clinical database.  
The project covers every stage from raw data preprocessing through feature selection, model training (10 prediction schemes), and evaluation with comparative ROC analysis.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Pipeline Stages](#pipeline-stages)
  - [1. Data Preprocessing](#1-data-preprocessing-preprocesspy)
  - [2. Feature Selection](#2-feature-selection-feature_selectionpy)
  - [3. Model Training](#3-model-training-model_trainingpy)
  - [4. Evaluation](#4-evaluation-evaluationpy)
- [Results](#results)
- [Getting Started](#getting-started)
- [Requirements](#requirements)

---

## Overview

The goal of this project is to predict whether a patient will die during their ICU stay (`icu_death_flag`) using structured clinical data extracted from MIMIC-IV. The pipeline systematically:

1. **Preprocesses** the raw dataset — removing data-leakage columns, regrouping categorical features, dropping high-missing columns, and applying a leak-free sklearn pipeline (imputation + scaling + one-hot encoding).
2. **Selects features** using three independent strategies (Lasso, Random Forest importance, ANOVA F-test) and saves selected feature lists.
3. **Trains 10 prediction schemes** — combinations of feature subsets and classifiers (Logistic Regression, Random Forest, SVM, Decision Tree, KNN).
4. **Evaluates** all schemes with Accuracy, ROC-AUC, F1-Score, Precision, and Recall; generates a comparative ROC curve figure.

---

## Project Structure

```
mimic-icu-discharge-prediction/
├── data/                        # Raw & processed data (git-ignored)
│   ├── mimic-vanilla.csv        #   Raw MIMIC-IV extract
│   ├── mimic-prep.csv           #   After cleaning & high-missing-col removal
│   ├── mimic-train.csv          #   Preprocessed training set (80%)
│   ├── mimic-test.csv           #   Preprocessed test set (20%)
│   ├── features_lasso.txt       #   Feature list selected by Lasso
│   ├── features_rf.txt          #   Feature list selected by Random Forest
│   └── features_anova.txt       #   Feature list selected by ANOVA
│
├── src/                         # Source code
│   ├── preprocess.py            #   Data loading, cleaning, and preprocessing
│   ├── feature_selection.py     #   Three feature-selection strategies
│   ├── model_training.py        #   Train & predict with 10 schemes
│   └── evaluation.py            #   Metrics computation & ROC plotting
│
├── results/                     # Output artifacts
│   ├── metrics.csv              #   Summary table of all scheme metrics
│   ├── predictions/             #   Per-scheme prediction CSVs (git-ignored)
│   └── figures/                 #   Generated plots (git-ignored)
│       ├── fig1_lasso_importance.pdf/.png
│       ├── fig2_rf_importance.pdf/.png
│       └── fig3_roc_comparison.pdf/.png
│
├── .gitignore
└── README.md
```

> **Note:** The `data/` directory and large regenerable outputs (`results/predictions/`, `results/figures/`) are excluded from version control via `.gitignore`. Only `results/metrics.csv` is tracked.

---

## Pipeline Stages

### 1. Data Preprocessing (`preprocess.py`)

| Step | Description |
|------|-------------|
| **Step 1** | Load `mimic-vanilla.csv` and drop leakage / identifier columns (`subject_id`, `hadm_id`, `stay_id`, `intime`, `outtime`, `deathtime`, `last_careunit`, `hospital_expire_flag`, `los`). |
| **Step 2** | Regroup six categorical features (`first_careunit`, `insurance`, `language`, `race`, `marital_status`, `gender`) into clinically meaningful super-categories. |
| **Step 3** | Remove columns with >50% missing values; save intermediate file (`mimic-prep.csv`). |
| **Step 4** | Stratified 80/20 train-test split (preserving class distribution). |
| **Steps 5–7** | Build a leak-free sklearn `ColumnTransformer` — OneHotEncoder for categoricals, median imputation + StandardScaler for numerics. Fit on train, transform both sets. |
| **Step 8** | Reassemble into DataFrames with clean column names; save `mimic-train.csv` and `mimic-test.csv`. |

### 2. Feature Selection (`feature_selection.py`)

Three independent strategies are applied to the preprocessed training data:

| Strategy | Method | Selection Criterion |
|----------|--------|---------------------|
| **Lasso** | L1-penalised Logistic Regression (`C=0.1`, solver=`saga`) | Features with non-zero coefficients (automatic) |
| **Random Forest** | RF Gini importance (`n_estimators=100`) | Top-30 features by importance score |
| **ANOVA** | One-way ANOVA F-test (`SelectKBest`) | Top-30 features by F-value |

Each strategy saves its selected feature list to `data/features_*.txt` and generates a bar-chart figure in `results/figures/`.

### 3. Model Training (`model_training.py`)

10 prediction schemes are defined by crossing feature subsets with classifiers:

| # | Feature Set | Classifier |
|---|-------------|------------|
| 1 | All features | Logistic Regression |
| 2 | All features | Random Forest |
| 3 | Lasso-selected | Logistic Regression |
| 4 | Lasso-selected | SVM (linear kernel) |
| 5 | Lasso-selected | Random Forest |
| 6 | RF-selected | Random Forest |
| 7 | RF-selected | Logistic Regression |
| 8 | RF-selected | SVM (linear kernel) |
| 9 | ANOVA-selected | Decision Tree |
| 10 | ANOVA-selected | KNN (k=5) |

Each scheme saves a CSV to `results/predictions/scheme_XX.csv` with columns `[scheme, y_true, y_pred, y_prob]`. Robust error handling ensures that a single failing scheme does not block the rest.

### 4. Evaluation (`evaluation.py`)

- Reads all `scheme_*.csv` files from `results/predictions/`.
- Computes five classification metrics per scheme: **Accuracy**, **ROC-AUC**, **F1-Score**, **Precision**, **Recall**.
- Writes the summary table to `results/metrics.csv`.
- Plots a combined ROC curve comparison across all schemes and saves it to `results/figures/fig3_roc_comparison.pdf`.

---

## Results

The final metric summary (`results/metrics.csv`) for all 10 schemes:

| Scheme | Accuracy | ROC-AUC | F1-Score | Precision | Recall |
|--------|----------|---------|----------|-----------|--------|
| 1. All + LogReg | 0.9335 | 0.8955 | 0.4797 | 0.7454 | 0.3536 |
| 2. All + RF | 0.9344 | 0.9137 | 0.4597 | 0.8040 | 0.3219 |
| 3. Lasso + LogReg | 0.9335 | 0.8955 | 0.4803 | 0.7444 | 0.3545 |
| 4. Lasso + SVM | 0.9277 | 0.8853 | 0.3187 | 0.8735 | 0.1949 |
| 5. Lasso + RF | 0.9344 | 0.9131 | 0.4624 | 0.7987 | 0.3254 |
| 6. RF_FI + RF | 0.9334 | 0.9055 | 0.4573 | 0.7792 | 0.3236 |
| 7. RF_FI + LogReg | 0.9304 | 0.8816 | 0.4341 | 0.7363 | 0.3078 |
| 8. RF_FI + SVM | 0.9253 | 0.8669 | 0.2660 | 0.8985 | 0.1561 |
| 9. ANOVA + DecTree | 0.9250 | 0.7978 | 0.4111 | 0.6453 | 0.3016 |
| 10. ANOVA + KNN | 0.9280 | 0.7859 | 0.3972 | 0.7260 | 0.2734 |

**Key findings:**
- **Best ROC-AUC**: Scheme 2 (All + RF) achieves the highest discriminative ability (AUC = 0.914).
- **Best F1-Score**: Scheme 3 (Lasso + LogReg) yields the best balance of precision and recall (F1 = 0.480).
- **Highest Precision**: Scheme 8 (RF_FI + SVM) has the highest precision (0.899) but at the cost of very low recall.
- Lasso-based feature selection preserves model performance while substantially reducing dimensionality.

---

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ and the required packages installed (see [Requirements](#requirements)).

### Running the Pipeline

Execute each stage sequentially from the project root:

```bash
# Step 1: Preprocess raw data
python src/preprocess.py

# Step 2: Feature selection
python src/feature_selection.py

# Step 3: Train models & generate predictions
python src/model_training.py

# Step 4: Evaluate & generate metrics + ROC plot
python src/evaluation.py
```

> **Important:** The raw MIMIC-IV data file (`data/mimic-vanilla.csv`) must be placed in the `data/` directory before running the pipeline. Access to MIMIC-IV requires credentialed access through [PhysioNet](https://physionet.org/).

---

## Requirements

- Python ≥ 3.8
- NumPy
- Pandas
- scikit-learn
- Matplotlib

Install all dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib
```

---

## License

This project uses data from the [MIMIC-IV database](https://physionet.org/content/mimiciv/), which is subject to the PhysioNet Credentialed Health Data Use Agreement.
