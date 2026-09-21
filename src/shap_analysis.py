"""연체 클래스의 예측 확률을 SHAP 기여도로 설명한다."""

import numpy as np
from .plotting import plt, save_plot
import shap


def explain_positive(model, x):
    """연체 클래스의 SHAP 값을 계산하고 확률 합산을 검증한다."""
    explainer = shap.TreeExplainer(
        model,
        feature_perturbation='tree_path_dependent',
        model_output='raw',
    )
    raw = explainer(x, check_additivity=True)
    class_index = list(model.classes_).index(1)

    if raw.values.ndim != 3 or raw.values.shape[2] != 2:
        raise ValueError(
            f'Expected binary multi-output Tree SHAP values, got {raw.values.shape}'
        )

    result = shap.Explanation(
        values=raw.values[:, :, class_index],
        base_values=raw.base_values[:, class_index],
        data=x.to_numpy(),
        feature_names=list(x.columns),
    )

    # Random Forest의 raw 출력은 로그 오즈가 아닌 확률이다.
    np.testing.assert_allclose(
        result.base_values + result.values.sum(axis=1),
        model.predict_proba(x)[:, class_index],
        atol=1e-6,
        rtol=1e-6,
    )

    return result


def create_summary_plot(explanation, output_path):
    """전체 특성 기여도를 beeswarm 그래프로 저장한다."""
    shap.plots.beeswarm(explanation, max_display=6, show=False)
    plt.xlabel('SHAP value for P(is_overdue = 1)')
    save_plot(output_path)


def create_waterfall_plot(explanation, output_path):
    """개별 고객의 기여도를 waterfall 그래프로 저장한다."""
    shap.plots.waterfall(explanation, max_display=6, show=False)
    save_plot(output_path)


def create_dependence_plot(explanation, feature, output_path):
    """특성값과 SHAP 기여도의 관계를 그래프로 저장한다."""
    shap.plots.scatter(explanation[:, feature], color=explanation, show=False)
    save_plot(output_path)
