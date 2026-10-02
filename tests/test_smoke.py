"""설치된 CLI와 임시 데이터의 스키마·분할·SHAP 계약을 확인한다."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from customer_risk.data import FEATURES, load_data, prepare_features
from customer_risk.modeling import train_model
from customer_risk.shap_analysis import explain_positive


@pytest.fixture
def finance_csv(tmp_path):
    rng = np.random.RandomState(7)
    frame = pd.DataFrame(rng.uniform(size=(120, 6)), columns=FEATURES)
    frame["credit_score"] = 500
    frame["is_overdue"] = (frame["debt_ratio"] > 0.65).astype(int)
    path = tmp_path / "finance.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.mark.smoke
def test_fixture_holdout_and_explanation(finance_csv):
    frame = load_data(finance_csv)
    inputs, scaled, _ = prepare_features(frame)
    assert list(inputs.columns) == FEATURES
    assert np.isfinite(scaled).all()
    model, train, holdout, _, metrics = train_model(frame)
    assert not set(train.index) & set(holdout.index)
    assert metrics["train_rows"] + metrics["test_rows"] == len(frame)
    explanation = explain_positive(model, holdout)
    np.testing.assert_allclose(
        explanation.base_values + explanation.values.sum(axis=1),
        model.predict_proba(holdout)[:, 1],
        atol=1e-6,
    )


@pytest.mark.smoke
@pytest.mark.parametrize("module", ["analysis_clustering", "analysis_shap", "generate_data"])
def test_cli_help_outside_checkout(module, tmp_path):
    completed = subprocess.run(
        [sys.executable, "-m", f"customer_risk.{module}", "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
