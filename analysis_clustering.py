"""전처리부터 군집 선택, PCA, 페르소나 통계 생성을 순서대로 실행한다."""

import argparse
import json
from pathlib import Path

from src.clustering import analyze_clusters
from src.data import fingerprint, load_data, prepare_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/finance_data.csv'))
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    args = parser.parse_args()

    frame = load_data(args.data)
    args.output.mkdir(parents=True, exist_ok=True)

    x, scaled, _ = prepare_features(frame)
    result = analyze_clusters(frame, x, scaled, args.output)
    result.update(
        {
            'data_path': str(args.data),
            'data_sha256': fingerprint(args.data),
            'rows': len(frame),
        }
    )

    (args.output / 'clustering.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(f'Clustering complete: K={result["selected_k"]}; output={args.output}')


if __name__ == '__main__':
    main()
