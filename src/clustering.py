"""표준화한 원본 특성으로 군집을 분석하고 PCA 시각화용 데이터를 만든다."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .data import FEATURES, SEED
from .plotting import plot_k_selection, plot_pca_clusters


def analyze_clusters(frame, x, scaled, out):
    # 실루엣 점수가 가장 높은 K를 최종 군집 수로 선택한다.
    rows = []
    models = {}

    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(scaled)
        score = silhouette_score(
            scaled,
            model.labels_,
            sample_size=min(2000, len(x)),
            random_state=SEED,
        )
        rows.append(
            {
                'k': k,
                'inertia': float(model.inertia_),
                'silhouette': float(score),
            }
        )
        models[k] = model

    scores = pd.DataFrame(rows)
    scores['inertia_drop_fraction'] = -scores.inertia.pct_change()
    scores.to_csv(out / 'k_scores.csv', index=False)

    selected = int(scores.loc[scores.silhouette.idxmax(), 'k'])
    model = models[selected]
    plot_k_selection(scores, selected, out / 'k_selection.png')

    # PCA는 군집 계산이 아닌 2차원 시각화에만 사용한다.
    pca = PCA(n_components=2).fit(scaled)
    coordinates = pca.transform(scaled)
    variance = pca.explained_variance_ratio_
    plot_pca_clusters(
        coordinates,
        model.labels_,
        variance,
        selected,
        out / 'pca_clusters.png',
    )
    pd.DataFrame(
        pca.components_.T,
        index=FEATURES,
        columns=['PC1', 'PC2'],
    ).to_csv(out / 'pca_loadings.csv')

    assignments = pd.DataFrame(
        {
            'row_id': x.index,
            'cluster': model.labels_,
            'PC1': coordinates[:, 0],
            'PC2': coordinates[:, 1],
        }
    )
    assignments.to_csv(out / 'cluster_assignments.csv', index=False)

    # 군집별 통계와 전체 평균 대비 차이를 계산한다.
    descriptive = x.assign(cluster=model.labels_)
    profiles = descriptive.groupby('cluster')[FEATURES].agg(['mean', 'median', 'std'])
    profiles.to_csv(out / 'cluster_statistics.csv')

    means = descriptive.groupby('cluster')[FEATURES].mean()
    baseline_mean, baseline_std = x.mean(), x.std(ddof=0).replace(0, 1)

    personas = []
    names = {
        'age': ('연령 높은', '연령 낮은'),
        'annual_income': ('소득 높은', '소득 낮은'),
        'spending_score': ('소비점수 높은', '소비점수 낮은'),
        'debt_ratio': ('부채비율 높은', '부채비율 낮은'),
        'credit_card_count': ('다카드', '소수카드'),
        'overdue_count_6m': ('연체이력 많은', '연체이력 적은'),
    }

    for cluster in range(selected):
        z = (means.loc[cluster] - baseline_mean) / baseline_std
        ranked = z.abs().sort_values(ascending=False)

        # 작은 차이는 이름에서 제외하되 전체 통계에는 그대로 남긴다.
        top = (
            ranked[ranked >= .25].head(2).index.tolist()
            or ranked.head(1).index.tolist()
        )

        mask = model.labels_ == cluster
        center_distances = np.linalg.norm(
            scaled[mask] - model.cluster_centers_[cluster],
            axis=1,
        )
        representative = int(x.index[mask][np.argmin(center_distances)])

        personas.append(
            {
                'cluster': cluster,
                'name': ' · '.join(names[feature][int(z[feature] < 0)] for feature in top),
                'count': int(mask.sum()),
                'share': float(mask.mean()),
                'means': means.loc[cluster].to_dict(),
                'z_scores': z.to_dict(),
                'distinctive_features': top,
                'representative_row_id': representative,
                'observed_overdue_rate': float(frame.loc[mask, 'is_overdue'].mean()),
            }
        )

    silhouette_pca_2d = silhouette_score(
        coordinates,
        model.labels_,
        sample_size=min(2000, len(x)),
        random_state=SEED,
    )

    return {
        'selected_k': selected,
        'scores': rows,
        'selection_reason': (
            'Maximum silhouette on a fixed 2,000-row sample; inertia curve is '
            'supporting evidence. No override for persona aesthetics.'
        ),
        'pca_variance_ratio': variance.tolist(),
        'silhouette_pca_2d': float(silhouette_pca_2d),
        'personas': personas,
        'missing_values': frame[FEATURES].isna().sum().to_dict(),
    }
