"""Random Forest를 평가하고 전체·고객별 연체 위험을 SHAP으로 설명한다."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import FEATURES, SEED, fingerprint, load_data
from src.modeling import select_cases, train_model
from src.shap_analysis import (
    create_dependence_plot,
    create_summary_plot,
    create_waterfall_plot,
    explain_positive,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/finance_data.csv'))
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--sample-size', type=int, default=1000)
    args = parser.parse_args()

    if args.sample_size < 2:
        parser.error('--sample-size must be at least 2')

    cluster_path = args.output / 'clustering.json'
    if not cluster_path.is_file():
        parser.error('Run analysis_clustering.py first.')

    clusters = json.loads(cluster_path.read_text(encoding='utf-8'))
    if fingerprint(args.data) != clusters['data_sha256']:
        parser.error('CSV differs from the clustering run; rerun clustering.')

    # 모델 학습과 전체 SHAP 분석에는 홀드아웃 고객만 사용한다.
    frame = load_data(args.data)
    model, x_train, x_test, y_test, metrics = train_model(frame)
    cases = select_cases(model, x_test)
    global_x = x_test.sample(
        n=min(args.sample_size, len(x_test)),
        random_state=SEED,
    )
    explanation = explain_positive(model, global_x)
    create_summary_plot(explanation, args.output / 'shap_summary.png')

    importance = pd.DataFrame(
        {
            'feature': FEATURES,
            'mean_abs_shap': np.abs(explanation.values).mean(axis=0),
            'impurity_importance': model.feature_importances_,
        }
    ).sort_values('mean_abs_shap', ascending=False)
    importance.to_csv(args.output / 'feature_importance.csv', index=False)

    dependencies = list(
        dict.fromkeys(importance.feature.head(2).tolist() + ['debt_ratio'])
    )
    dependence_stats = []
    directions = []

    for feature in FEATURES:
        values = global_x[feature].to_numpy()
        shap_values = explanation.values[:, FEATURES.index(feature)]
        q25, q75 = np.quantile(values, [.25, .75])
        stats = {
            'feature': feature,
            'q25': float(q25),
            'q75': float(q75),
            'low_mean_shap': float(shap_values[values <= q25].mean()),
            'high_mean_shap': float(shap_values[values >= q75].mean()),
        }
        directions.append(stats)

        if feature in dependencies:
            create_dependence_plot(
                explanation,
                feature,
                args.output / f'dependence_{feature}.png',
            )
            dependence_stats.append(stats)

    assignments = pd.read_csv(args.output / 'cluster_assignments.csv').set_index('row_id')

    # 각 군집 평균에 가장 가까운 홀드아웃 고객을 대표 사례로 선택한다.
    scale = x_train.std(ddof=0).replace(0, 1)
    for persona in clusters['personas']:
        member_mask = assignments.loc[x_test.index, 'cluster'].eq(persona['cluster'])
        members = x_test.loc[member_mask]
        if members.empty:
            raise ValueError(f'Cluster {persona["cluster"]} has no holdout representative.')

        distance = ((members - pd.Series(persona['means'])) / scale).pow(2).sum(axis=1)
        cases[f'cluster_{persona["cluster"]}'] = int(distance.idxmin())

    # 승인·거절 사례와 군집 대표 사례의 Local SHAP을 계산한다.
    local_x = x_test.loc[list(dict.fromkeys(cases.values()))]
    local_explanation = explain_positive(model, local_x)
    local_results = []

    for name, row_id in cases.items():
        position = local_x.index.get_loc(row_id)
        exp = local_explanation[position]
        probability = float(model.predict_proba(local_x.loc[[row_id]])[0, 1])
        create_waterfall_plot(exp, args.output / f'waterfall_{name}.png')
        local_results.append(
            {
                'case': name,
                'row_id': row_id,
                'cluster': int(assignments.loc[row_id, 'cluster']),
                'actual_is_overdue': int(y_test.loc[row_id]),
                'decision': 'rejection' if probability >= .5 else 'approval',
                'probability': probability,
                'base_value': float(exp.base_values),
                'features': local_x.loc[row_id].to_dict(),
                'contributions': dict(zip(FEATURES, exp.values.tolist())),
            }
        )

    holdout_predictions = pd.DataFrame(
        {
            'row_id': x_test.index,
            'actual_is_overdue': y_test,
            'probability': model.predict_proba(x_test)[:, 1],
            'cluster': assignments.loc[x_test.index, 'cluster'],
        }
    )
    holdout_predictions.to_csv(
        args.output / 'holdout_predictions.csv',
        index=False,
    )
    pd.DataFrame({'row_id': x_train.index}).to_csv(
        args.output / 'train_rows.csv',
        index=False,
    )
    pd.DataFrame(
        explanation.values,
        index=global_x.index,
        columns=FEATURES,
    ).rename_axis('row_id').to_csv(args.output / 'global_shap_values.csv')

    predicted_probability = model.predict_proba(global_x)[:, 1]
    additivity_error = np.abs(
        explanation.base_values
        + explanation.values.sum(axis=1)
        - predicted_probability
    )
    result = {
        'data_sha256': fingerprint(args.data),
        'model_source': (
            'New RandomForest trained on a fixed 75% split; not an imported '
            'Mission 23 model.'
        ),
        'metrics': metrics,
        'global_sample_size': len(global_x),
        'importance': importance.to_dict(orient='records'),
        'dependence': dependence_stats,
        'directions': directions,
        'local_cases': local_results,
        'max_additivity_error': float(np.max(additivity_error)),
    }
    (args.output / 'shap.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(
        f'SHAP complete: {len(global_x)} global rows, '
        f'{len(cases)} local cases; ROC-AUC={metrics["roc_auc"]:.4f}'
    )


if __name__ == '__main__':
    main()
