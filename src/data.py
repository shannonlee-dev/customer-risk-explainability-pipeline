"""Strict input schema; targets are never predictors or clustering dimensions."""
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

FEATURES = ['age', 'annual_income', 'spending_score', 'debt_ratio',
            'credit_card_count', 'overdue_count_6m']
TARGETS = ['credit_score', 'is_overdue']
SEED = 42


def load_data(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Mission 23 CSV is required: {path}')
    frame = pd.read_csv(path)
    if set(frame.columns) != set(FEATURES + TARGETS) or len(frame.columns) != 8:
        raise ValueError(f'Expected exactly these original columns: {FEATURES + TARGETS}')
    frame = frame[FEATURES + TARGETS].apply(pd.to_numeric, errors='raise')
    if len(frame) < 40:
        raise ValueError('At least 40 rows are required for clustering and a stratified holdout.')
    if np.isinf(frame.to_numpy()).any():
        raise ValueError('Infinite values are not supported.')
    if frame[FEATURES].isna().all().any():
        raise ValueError('A feature has no observed values.')
    if frame.is_overdue.isna().any() or set(frame.is_overdue.unique()) != {0, 1}:
        raise ValueError('is_overdue must contain both 0 and 1, without missing labels.')
    if frame.is_overdue.value_counts().min() < 5:
        raise ValueError('Each target class requires at least five observations.')
    return frame


def prepare_features(frame):
    """Descriptive clustering uses all features, with median imputation and z-scores."""
    imputer = SimpleImputer(strategy='median')
    x = pd.DataFrame(imputer.fit_transform(frame[FEATURES]), columns=FEATURES, index=frame.index)
    scaler = StandardScaler()
    return x, scaler.fit_transform(x), scaler


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
