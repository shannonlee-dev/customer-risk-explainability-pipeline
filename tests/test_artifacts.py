"""임시 데이터로 실제 CLI와 생성 산출물의 일관성을 검증한다."""

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


class ArtifactTests(unittest.TestCase):
    def test_cli_pipeline_artifacts(self):
        rng = np.random.RandomState(11)
        # K=50까지 탐색해도 군집별 홀드아웃 대표를 확보할 수 있는 표본 규모.
        n = 2000
        frame = pd.DataFrame(
            {
                'age': rng.randint(20, 70, n),
                'annual_income': rng.uniform(1500, 9000, n),
                'spending_score': rng.randint(1, 100, n),
                'debt_ratio': rng.uniform(0, 1, n),
                'credit_card_count': rng.randint(1, 10, n),
                'overdue_count_6m': rng.poisson(.5, n),
                'credit_score': rng.randint(300, 900, n),
            }
        )
        frame['is_overdue'] = (frame.debt_ratio > .7).astype(int)

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data = tmp / 'fixture.csv'
            out = tmp / 'outputs'
            report = tmp / 'report.md'
            frame.to_csv(data, index=False)

            env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            for script in ['analysis_clustering.py', 'analysis_shap.py']:
                result = subprocess.run(
                    [
                        sys.executable,
                        str(ROOT / script),
                        '--data',
                        str(data),
                        '--output',
                        str(out),
                    ],
                    cwd=ROOT,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / 'build_report.py'),
                    '--output',
                    str(out),
                    '--destination',
                    str(report),
                    '--provenance',
                    '테스트 전용 데이터',
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            clustering = json.loads((out / 'clustering/clustering.json').read_text())
            shap = json.loads((out / 'shap/shap.json').read_text())
            self.assertEqual(clustering['data_sha256'], shap['data_sha256'])

            scores = clustering['scores']
            self.assertEqual([score['k'] for score in scores], list(range(2, 51)))
            csv_scores = pd.read_csv(out / 'clustering/k_scores.csv')
            self.assertEqual(csv_scores.k.tolist(), list(range(2, 51)))
            np.testing.assert_allclose(
                csv_scores.silhouette, [score['silhouette'] for score in scores]
            )
            best_k = max(scores, key=lambda score: score['silhouette'])['k']
            self.assertEqual(clustering['selected_k'], best_k)
            self.assertEqual(len(shap['local_cases']), clustering['selected_k'] + 2)

            required_plots = [
                'clustering/k_selection.png',
                'clustering/pca_clusters.png',
                'shap/shap_summary.png',
                'shap/waterfall/waterfall_approval.png',
                'shap/waterfall/waterfall_rejection.png',
            ]
            required_plots += [
                f'shap/dependence/dependence_{item["feature"]}.png'
                for item in shap['dependence']
            ]
            required_plots += [
                f'shap/waterfall/waterfall_cluster_{cluster}.png'
                for cluster in range(clustering['selected_k'])
            ]
            for filename in required_plots:
                self.assertTrue((out / filename).is_file(), filename)

            self.assertTrue(all(path.is_dir() for path in out.iterdir()))
            holdout = pd.read_csv(out / 'model/holdout_predictions.csv')
            train_ids = set(pd.read_csv(out / 'model/train_rows.csv').row_id)
            self.assertFalse(train_ids & set(holdout.row_id))

            for case in shap['local_cases']:
                self.assertIn(case['row_id'], set(holdout.row_id))
                contribution_sum = sum(case['contributions'].values())
                self.assertAlmostEqual(
                    case['base_value'] + contribution_sum,
                    case['probability'],
                    places=6,
                )

            for path in out.rglob('*.png'):
                with Image.open(path) as image:
                    image.verify()

            content = report.read_text()
            for link in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', content):
                self.assertTrue((report.parent / link).is_file(), link)
            self.assertIn('테스트 전용 데이터', content)
            self.assertIn('K=2~50 범위를 탐색', content)

            # 입력이 바뀌면 이전 군집 배정 결과를 재사용할 수 없다.
            frame.loc[0, 'age'] += 1
            frame.to_csv(data, index=False)
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / 'analysis_shap.py'),
                    '--data',
                    str(data),
                    '--output',
                    str(out),
                ],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('differs', result.stderr)


if __name__ == '__main__':
    unittest.main()
