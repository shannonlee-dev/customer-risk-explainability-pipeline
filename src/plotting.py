"""노트북 없이 일관된 스타일의 그래프를 생성하고 저장한다."""

import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1] / '.mplconfig'))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi': 120, 'savefig.dpi': 160, 'font.size': 10})


def save_plot(path):
    """현재 그래프를 저장하고 Figure 자원을 정리한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


def plot_k_selection(scores, selected, path):
    """K별 inertia와 silhouette를 나란히 그린다."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, metric in zip(axes, ['inertia', 'silhouette']):
        ax.plot(scores.k, scores[metric], 'o-', color='#247b8b')
        ax.axvline(
            selected,
            color='#d65f3a',
            linestyle='--',
            label=f'Selected K={selected}',
        )
        ax.set(xlabel='Number of clusters K', ylabel=metric.title())
        ax.legend()

    fig.suptitle('K selection: silhouette primary, elbow supporting')
    save_plot(path)


def plot_pca_clusters(coordinates, labels, variance, cluster_count, path):
    """PCA 좌표를 군집별 산점도로 그린다."""
    fig, ax = plt.subplots(figsize=(11, 6))

    for cluster in range(cluster_count):
        mask = labels == cluster
        ax.scatter(
            coordinates[mask, 0],
            coordinates[mask, 1],
            s=8,
            alpha=.3,
            label=f'Cluster {cluster} (n={int(mask.sum())})',
        )

    ax.set_xlabel(f'PC1 ({variance[0]:.1%} explained variance)')
    ax.set_ylabel(f'PC2 ({variance[1]:.1%} explained variance)')
    ax.set_title(
        f'Customer personas in PCA space | total variance {sum(variance):.1%}'
    )
    ax.legend(
        markerscale=2,
        bbox_to_anchor=(1.02, 1),
        loc='upper left',
        borderaxespad=0,
        fontsize=8,
    )
    save_plot(path)
