"""Explain class 1 in probability units, with explicit shape and sum checks."""
import numpy as np
from .plotting import plt, save_plot
import shap


def explain_positive(model, x):
    explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent', model_output='raw')
    raw = explainer(x, check_additivity=True)
    class_index = list(model.classes_).index(1)
    if raw.values.ndim != 3 or raw.values.shape[2] != 2:
        raise ValueError(f'Expected binary multi-output Tree SHAP values, got {raw.values.shape}')
    result = shap.Explanation(values=raw.values[:, :, class_index],
                              base_values=raw.base_values[:, class_index],
                              data=x.to_numpy(), feature_names=list(x.columns))
    # sklearn RandomForest raw output is a probability, not a log-odds margin.
    np.testing.assert_allclose(result.base_values + result.values.sum(axis=1),
                               model.predict_proba(x)[:, class_index], atol=1e-6, rtol=1e-6)
    return result


def create_summary_plot(explanation, output_path):
    shap.plots.beeswarm(explanation, max_display=6, show=False)
    plt.xlabel('SHAP value for P(is_overdue = 1)')
    save_plot(output_path)


def create_waterfall_plot(explanation, output_path):
    shap.plots.waterfall(explanation, max_display=6, show=False)
    save_plot(output_path)


def create_dependence_plot(explanation, feature, output_path):
    shap.plots.scatter(explanation[:, feature], color=explanation, show=False)
    save_plot(output_path)
