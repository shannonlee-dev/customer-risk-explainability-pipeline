import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import FEATURES, load_data, prepare_features
from src.modeling import train_model, select_cases
from src.shap_analysis import explain_positive


class PipelineTests(unittest.TestCase):
    def frame(self):
        rng = np.random.RandomState(7)
        x = pd.DataFrame(rng.uniform(size=(120, 6)), columns=FEATURES)
        x['credit_score'] = 500
        x['is_overdue'] = (x['debt_ratio'] > .65).astype(int)
        return x

    def test_targets_never_enter_inputs(self):
        frame = self.frame()
        frame.loc[0, 'annual_income'] = np.nan
        x, scaled, _ = prepare_features(frame)
        self.assertEqual(list(x.columns), FEATURES)
        self.assertTrue(np.isfinite(scaled).all())
        np.testing.assert_allclose(scaled.mean(axis=0), 0, atol=1e-12)

    def test_invalid_target_and_extra_columns_rejected(self):
        for kind in ['target', 'extra']:
            frame = self.frame()
            if kind == 'target':
                frame.loc[0, 'is_overdue'] = 2
            else:
                frame['customer_id'] = 1
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / 'data.csv'
                frame.to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    load_data(path)

    def test_held_out_predictions_and_shap_additivity(self):
        frame = self.frame()
        frame.loc[0, 'annual_income'] = np.nan
        model, x_train, x_test, y_test, metrics = train_model(frame)
        self.assertFalse(set(x_train.index) & set(x_test.index))
        self.assertEqual(list(x_test.columns), FEATURES)
        for feature in FEATURES:
            self.assertAlmostEqual(metrics['training_medians'][feature],
                                   frame.loc[x_train.index, feature].median())
        explanation = explain_positive(model, x_test)
        np.testing.assert_allclose(explanation.base_values + explanation.values.sum(axis=1),
                                   model.predict_proba(x_test)[:, 1], atol=1e-6)
        cases = select_cases(model, x_test)
        self.assertLess(model.predict_proba(x_test.loc[[cases['approval']]])[0, 1], .5)
        self.assertGreaterEqual(model.predict_proba(x_test.loc[[cases['rejection']]])[0, 1], .5)

    def test_missing_rejection_is_explicit(self):
        frame = self.frame()
        model, _, x_test, _, _ = train_model(frame)
        low = x_test.loc[model.predict_proba(x_test)[:, 1] < .5]
        with self.assertRaisesRegex(ValueError, 'rejection'):
            select_cases(model, low)


if __name__ == '__main__':
    unittest.main()
