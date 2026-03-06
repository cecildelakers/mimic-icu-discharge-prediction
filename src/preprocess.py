"""
SPH6004 Individual Assignment
Author: Chen Wenhao
Date: 2026-02-06

preprocess.py
Loads the raw MIMIC-IV dataset, removes leakage-prone columns,
regroups categorical features, filters high-missing columns,
splits into train/test, and applies a leak-free sklearn Pipeline
(imputation + scaling for numerics, one-hot for categoricals).
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ── paths ──────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_PATH = os.path.join(DATA_DIR, 'mimic-vanilla.csv')
PREP_PATH = os.path.join(DATA_DIR, 'mimic-prep.csv')
TRAIN_PATH = os.path.join(DATA_DIR, 'mimic-train.csv')
TEST_PATH = os.path.join(DATA_DIR, 'mimic-test.csv')

TARGET_COL = 'icu_death_flag'
MISSING_THRESHOLD = 0.50

# ── columns to drop (leakage / identifiers) ────────────────────────────
LEAK_COLS = [
    'subject_id', 'hadm_id', 'stay_id',             # Identity
    'intime', 'outtime', 'deathtime',               # Time information
    'last_careunit', 'hospital_expire_flag', 'los',  # Outcome-related
]

# ── categorical regrouping mappings ────────────────────────────────────
CAT_COLS = ['first_careunit', 'insurance', 'language', 'race',
            'marital_status', 'gender']

FIRST_CAREUNIT_MAP = {
    'Cardiac': [
        'Cardiac Vascular Intensive Care Unit (CVICU)',
        'Coronary Care Unit (CCU)',
    ],
    'Medical': [
        'Intensive Care Unit (ICU)', 'Med/Surg',
        'Medical Intensive Care Unit (MICU)',
        'Medical/Surgical Intensive Care Unit (MICU/SICU)',
        'Medicine',
    ],
    'Surgical': [
        'Surgery/Trauma', 'Surgery/Vascular/Intermediate',
        'Surgical Intensive Care Unit (SICU)',
        'Trauma SICU (TSICU)',
    ],
    'Neuro': [
        'Neuro Intermediate', 'Neuro Stepdown',
        'Neuro Surgical Intensive Care Unit (Neuro SICU)',
        'Neurology',
    ],
    'PACU': ['PACU'],
}

INSURANCE_MAP = {
    'Government': ['Medicaid', 'Medicare'],
    'Private':    ['Private'],
    'Self_pay':   ['No charge'],
    'Other':      ['Other'],
}

RACE_MAP = {
    'Asian': [
        'ASIAN', 'ASIAN - ASIAN INDIAN', 'ASIAN - CHINESE',
        'ASIAN - KOREAN', 'ASIAN - SOUTH EAST ASIAN',
    ],
    'Black': [
        'BLACK/AFRICAN', 'BLACK/AFRICAN AMERICAN',
        'BLACK/CAPE VERDEAN', 'BLACK/CARIBBEAN ISLAND',
    ],
    'Hispanic': [
        'HISPANIC OR LATINO',
        'HISPANIC/LATINO - CENTRAL AMERICAN',
        'HISPANIC/LATINO - COLUMBIAN',
        'HISPANIC/LATINO - CUBAN',
        'HISPANIC/LATINO - DOMINICAN',
        'HISPANIC/LATINO - GUATEMALAN',
        'HISPANIC/LATINO - HONDURAN',
        'HISPANIC/LATINO - MEXICAN',
        'HISPANIC/LATINO - PUERTO RICAN',
        'HISPANIC/LATINO - SALVADORAN',
        'SOUTH AMERICAN',
    ],
    'White': [
        'WHITE', 'WHITE - BRAZILAN', 'WHITE - EASTERN EUROPEAN',
        'WHITE - OTHER EUROPEAN', 'WHITE - RUSSIAN', 'PORTUGUESE',
    ],
    'Other': [
        'OTHER', 'MULTIPLE RACE/ETHNICITY',
        'AMERICAN INDIAN/ALASKA NATIVE',
        'NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER',
    ],
    'Unknown': [
        'PATIENT DECLINED TO ANSWER', 'UNABLE TO OBTAIN', 'UNKNOWN',
    ],
}

MARITAL_MAP = {
    'Married':            ['MARRIED'],
    'Single':             ['SINGLE'],
    'Prev_Married': ['WIDOWED', 'DIVORCED'],
}


# ── helper: build a reverse lookup from {group: [values]} ─────────────
def _invert_map(mapping: dict) -> dict:
    """Invert {group: [values]} -> {value: group}."""
    return {v: group for group, vals in mapping.items() for v in vals}


_CAREUNIT_LUT = _invert_map(FIRST_CAREUNIT_MAP)
_INSURANCE_LUT = _invert_map(INSURANCE_MAP)
_RACE_LUT = _invert_map(RACE_MAP)
_MARITAL_LUT = _invert_map(MARITAL_MAP)


def _remap(series: pd.Series, lut: dict, default: str) -> pd.Series:
    """Map a Series through a lookup table; fill missing/unmatched with *default*."""
    return series.fillna(default).map(lambda x: lut.get(x, default))


def regroup_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Apply categorical regrouping to the six categorical columns."""
    df = df.copy()

    if 'first_careunit' in df.columns:
        df['first_careunit'] = _remap(df['first_careunit'], _CAREUNIT_LUT, 'Unknown')

    if 'insurance' in df.columns:
        df['insurance'] = _remap(df['insurance'], _INSURANCE_LUT, 'Unknown')

    if 'language' in df.columns:
        df['language'] = df['language'].apply(
            lambda x: 'English' if x == 'English' else 'Non_English'
        )

    if 'race' in df.columns:
        df['race'] = _remap(df['race'], _RACE_LUT, 'Unknown')

    if 'marital_status' in df.columns:
        df['marital_status'] = _remap(df['marital_status'], _MARITAL_LUT, 'Unknown')

    # gender: keep 'M' / 'F' as-is (no remapping needed)

    return df


# ── Step 1: load & drop leakage columns ───────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    """Read csv and drop leakage / id columns."""
    df = pd.read_csv(path)
    cols_to_drop = [c for c in LEAK_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"[Step 1] Loaded {os.path.basename(path)}  |  shape: {df.shape}"
          f"  |  dropped {len(cols_to_drop)} leakage cols")
    return df


# ── Step 3: drop columns with > 50 % missing & save intermediate ──────
def drop_high_missing(df: pd.DataFrame,
                      threshold: float = MISSING_THRESHOLD) -> pd.DataFrame:
    """Drop columns whose missing rate exceeds *threshold*."""
    miss_rate = df.isnull().mean()
    high_miss = miss_rate[miss_rate > threshold].index.tolist()
    df = df.drop(columns=high_miss)
    print(f"[Step 3] Dropped {len(high_miss)} columns with >{threshold*100:.0f}% missing"
          f"  |  remaining shape: {df.shape}")
    if high_miss:
        print(f"         Dropped cols: {high_miss}")
    return df


# ── Step 5: build sklearn preprocessing pipeline ──────────────────────
def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer:
      - categorical cols -> OneHotEncoder
      - numeric cols     -> median imputation + standard scaling
    """
    cat_features = [c for c in CAT_COLS if c in X.columns]
    num_features = [c for c in X.columns if c not in cat_features]

    cat_pipe = Pipeline([
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False)),
    ])

    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler()),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', cat_pipe, cat_features),
            ('num', num_pipe, num_features),
        ],
        remainder='drop',
    )
    return preprocessor


# ── Step 8: retrieve feature names & reorder columns ──────────────────
def build_final_df(array: np.ndarray,
                   labels: pd.Series,
                   preprocessor: ColumnTransformer) -> pd.DataFrame:
    """
    Convert the transformed numpy array back into a DataFrame.
    Column order: [icu_death_flag, <categorical OHE cols>, <numeric cols>].
    """
    feature_names = preprocessor.get_feature_names_out()
    # strip transformer prefixes and convert to lowercase
    clean_names = [n.split('__', 1)[-1].lower() for n in feature_names]

    df_out = pd.DataFrame(array, columns=clean_names, index=labels.index)

    # separate categorical (OHE) and numeric column names
    cat_name_prefix = preprocessor.transformers_[0][2]  # original cat col list
    ohe: OneHotEncoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    ohe_names = [n.split('__', 1)[-1].lower() for n in ohe.get_feature_names_out(cat_name_prefix)]
    num_names = [c for c in clean_names if c not in ohe_names]

    # reorder: target first, then categorical OHE, then numeric
    ordered_cols = ohe_names + num_names
    df_out = df_out[ordered_cols]
    df_out.insert(0, TARGET_COL, labels.values)

    return df_out


# ── main orchestration ─────────────────────────────────────────────────
def main() -> None:
    # Step 1 – load & remove leakage columns
    df = load_and_clean(RAW_PATH)

    # Step 2 – regroup categorical features (on full data, before split)
    df = regroup_categoricals(df)
    print(f"[Step 2] Categorical regrouping done.")

    # Step 3 – drop columns with > 50% missing, save intermediate file
    df = drop_high_missing(df)
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(PREP_PATH, index=False)
    print(f"         Intermediate file saved -> {os.path.basename(PREP_PATH)}")

    # Step 4 – stratified train/test split (80/20)
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in the dataset.")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print(f"[Step 4] Train/Test split  |  train: {X_train.shape[0]}  test: {X_test.shape[0]}")

    # Step 5 – build preprocessing pipeline
    preprocessor = build_preprocessor(X_train)
    cat_used = [c for c in CAT_COLS if c in X_train.columns]
    num_used = [c for c in X_train.columns if c not in cat_used]
    print(f"[Step 5] Pipeline built  |  {len(cat_used)} categorical + {len(num_used)} numeric features")

    # Step 6 – fit on train, transform train
    X_train_arr = preprocessor.fit_transform(X_train)
    print(f"[Step 6] fit_transform on train  |  output shape: {X_train_arr.shape}")

    # Step 7 – transform test only (no re-fitting)
    X_test_arr = preprocessor.transform(X_test)
    print(f"[Step 7] transform on test       |  output shape: {X_test_arr.shape}")

    # Step 8 – reassemble DataFrames, reorder columns, save
    train_df = build_final_df(X_train_arr, y_train, preprocessor)
    test_df  = build_final_df(X_test_arr,  y_test,  preprocessor)

    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"\n{'='*50}")
    print(f"[Step 8] Final outputs saved")
    print(f"  Train : {train_df.shape}  ->  {os.path.basename(TRAIN_PATH)}")
    print(f"  Test  : {test_df.shape}  ->  {os.path.basename(TEST_PATH)}")
    print(f"  Train mortality : {y_train.mean()*100:.2f}%")
    print(f"  Test  mortality : {y_test.mean()*100:.2f}%")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
