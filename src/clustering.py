"""K selection on standardized original features; PCA is visualization only."""
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from .data import FEATURES, SEED
from .plotting import plt, save_plot


def analyze_clusters(frame, x, scaled, out):
    rows = []
    models = {}
    for k in range(2, 9):
        model = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(scaled)
        score = silhouette_score(scaled, model.labels_, sample_size=min(2000, len(x)),
                                 random_state=SEED)
        rows.append({'k': k, 'inertia': float(model.inertia_), 'silhouette': float(score)})
        models[k] = model
    scores = pd.DataFrame(rows)
    scores['inertia_drop_fraction'] = -scores.inertia.pct_change()
    scores.to_csv(out / 'k_scores.csv', index=False)
    selected = int(scores.loc[scores.silhouette.idxmax(), 'k'])
    model = models[selected]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric in zip(axes, ['inertia', 'silhouette']):
        ax.plot(scores.k, scores[metric], 'o-', color='#247b8b')
        ax.axvline(selected, color='#d65f3a', linestyle='--', label=f'Selected K={selected}')
        ax.set(xlabel='Number of clusters K', ylabel=metric.title())
        ax.legend()
    fig.suptitle('K selection: silhouette primary, elbow supporting')
    save_plot(out / 'k_selection.png')
    pca = PCA(n_components=2).fit(scaled)
    coordinates = pca.transform(scaled)
    plt.figure(figsize=(9, 6))
    for cluster in range(selected):
        mask = model.labels_ == cluster
        plt.scatter(coordinates[mask, 0], coordinates[mask, 1], s=8, alpha=.3,
                    label=f'Cluster {cluster} (n={int(mask.sum())})')
    variance = pca.explained_variance_ratio_
    plt.xlabel(f'PC1 ({variance[0]:.1%} explained variance)')
    plt.ylabel(f'PC2 ({variance[1]:.1%} explained variance)')
    plt.title(f'Customer personas in PCA space | total variance {sum(variance):.1%}')
    plt.legend(markerscale=2)
    save_plot(out / 'pca_clusters.png')
    pd.DataFrame(pca.components_.T, index=FEATURES, columns=['PC1', 'PC2']).to_csv(out / 'pca_loadings.csv')
    assignments = pd.DataFrame({'row_id': x.index, 'cluster': model.labels_,
                                'PC1': coordinates[:, 0], 'PC2': coordinates[:, 1]})
    assignments.to_csv(out / 'cluster_assignments.csv', index=False)
    descriptive = x.assign(cluster=model.labels_)
    profiles = descriptive.groupby('cluster')[FEATURES].agg(['mean', 'median', 'std'])
    profiles.to_csv(out / 'cluster_statistics.csv')
    means = descriptive.groupby('cluster')[FEATURES].mean()
    baseline_mean, baseline_std = x.mean(), x.std(ddof=0).replace(0, 1)
    personas = []
    names = {'age': ('연령 높은', '연령 낮은'), 'annual_income': ('소득 높은', '소득 낮은'),
             'spending_score': ('소비점수 높은', '소비점수 낮은'), 'debt_ratio': ('부채비율 높은', '부채비율 낮은'),
             'credit_card_count': ('다카드', '소수카드'), 'overdue_count_6m': ('연체이력 많은', '연체이력 적은')}
    for cluster in range(selected):
        z = (means.loc[cluster] - baseline_mean) / baseline_std
        ranked = z.abs().sort_values(ascending=False)
        # Ignore tiny differences when naming personas; retain full statistics.
        top = ranked[ranked >= .25].head(2).index.tolist() or ranked.head(1).index.tolist()
        mask = model.labels_ == cluster
        representative = int(x.index[mask][np.argmin(np.linalg.norm(scaled[mask] - model.cluster_centers_[cluster], axis=1))])
        personas.append({'cluster': cluster, 'name': ' · '.join(names[f][int(z[f] < 0)] for f in top),
                         'count': int(mask.sum()), 'share': float(mask.mean()),
                         'means': means.loc[cluster].to_dict(), 'z_scores': z.to_dict(),
                         'distinctive_features': top, 'representative_row_id': representative,
                         'observed_overdue_rate': float(frame.loc[mask, 'is_overdue'].mean())})
    return {'selected_k': selected, 'scores': rows,
            'selection_reason': 'Maximum silhouette on a fixed 2,000-row sample; inertia curve is supporting evidence. No override for persona aesthetics.',
            'pca_variance_ratio': variance.tolist(),
            'silhouette_pca_2d': float(silhouette_score(coordinates, model.labels_, sample_size=min(2000, len(x)), random_state=SEED)),
            'personas': personas, 'missing_values': frame[FEATURES].isna().sum().to_dict()}
