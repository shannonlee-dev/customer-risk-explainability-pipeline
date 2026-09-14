"""Holdout evaluation with imputation fitted on the training partition only."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from .data import FEATURES, SEED


def train_model(frame):
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame.is_overdue.astype(int), test_size=.25,
        random_state=SEED, stratify=frame.is_overdue)
    imputer = SimpleImputer(strategy='median')
    x_train = pd.DataFrame(imputer.fit_transform(x_train), columns=FEATURES, index=x_train.index)
    x_test = pd.DataFrame(imputer.transform(x_test), columns=FEATURES, index=x_test.index)
    model = RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                   random_state=SEED, n_jobs=2)
    model.fit(x_train, y_train)
    p = model.predict_proba(x_test)[:, list(model.classes_).index(1)]
    predicted = p >= .5
    metrics = {
        'train_rows': len(x_train), 'test_rows': len(x_test),
        'test_prevalence': float(y_test.mean()), 'threshold': .5,
        'roc_auc': float(roc_auc_score(y_test, p)),
        'average_precision': float(average_precision_score(y_test, p)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, predicted)),
        'precision': float(precision_score(y_test, predicted, zero_division=0)),
        'recall': float(recall_score(y_test, predicted, zero_division=0)),
        'brier_score': float(brier_score_loss(y_test, p)),
        'confusion_matrix': confusion_matrix(y_test, predicted, labels=[0, 1]).tolist(),
        'training_medians': dict(zip(FEATURES, imputer.statistics_.tolist())),
        'model_params': model.get_params(),
    }
    return model, x_train, x_test, y_test, metrics


def select_cases(model, x):
    """Pick the most confident case per predicted class; both are holdout rows."""
    p = model.predict_proba(x)[:, list(model.classes_).index(1)]
    if not (p < .5).any():
        raise ValueError('No approval case at threshold 0.5.')
    if not (p >= .5).any():
        raise ValueError('No rejection case at threshold 0.5; do not relabel a low-risk row.')
    return {'approval': int(x.index[np.argmin(p)]), 'rejection': int(x.index[np.argmax(p)])}
