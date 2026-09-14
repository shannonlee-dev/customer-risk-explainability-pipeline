"""Integration checks run the real CLIs on a temporary test fixture, not submission data."""
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
        n = 240
        frame = pd.DataFrame({
            'age': rng.randint(20, 70, n), 'annual_income': rng.uniform(1500, 9000, n),
            'spending_score': rng.randint(1, 100, n), 'debt_ratio': rng.uniform(0, 1, n),
            'credit_card_count': rng.randint(1, 10, n), 'overdue_count_6m': rng.poisson(.5, n),
            'credit_score': rng.randint(300, 900, n)})
        frame['is_overdue'] = (frame.debt_ratio > .7).astype(int)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            data, out, report = tmp / 'fixture.csv', tmp / 'outputs', tmp / 'report.md'
            frame.to_csv(data, index=False)
            env = dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
            for script in ['analysis_clustering.py', 'analysis_shap.py']:
                result = subprocess.run([sys.executable, str(ROOT / script), '--data', str(data), '--output', str(out)],
                                        cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            result = subprocess.run([sys.executable, str(ROOT / 'build_report.py'), '--output', str(out),
                                     '--destination', str(report), '--provenance', '테스트 전용 데이터'],
                                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            clustering = json.loads((out / 'clustering.json').read_text())
            shap = json.loads((out / 'shap.json').read_text())
            self.assertEqual(clustering['data_sha256'], shap['data_sha256'])
            scores = clustering['scores']
            self.assertEqual(clustering['selected_k'], max(scores, key=lambda s: s['silhouette'])['k'])
            self.assertEqual(len(shap['local_cases']), clustering['selected_k'] + 2)
            required_plots = ['k_selection.png', 'pca_clusters.png', 'shap_summary.png',
                              'waterfall_approval.png', 'waterfall_rejection.png']
            required_plots += [f'dependence_{d["feature"]}.png' for d in shap['dependence']]
            required_plots += [f'waterfall_cluster_{c}.png' for c in range(clustering['selected_k'])]
            for filename in required_plots:
                self.assertTrue((out / filename).is_file(), filename)
            holdout = pd.read_csv(out / 'holdout_predictions.csv')
            train_ids = set(pd.read_csv(out / 'train_rows.csv').row_id)
            self.assertFalse(train_ids & set(holdout.row_id))
            for case in shap['local_cases']:
                self.assertIn(case['row_id'], set(holdout.row_id))
                self.assertAlmostEqual(case['base_value'] + sum(case['contributions'].values()), case['probability'], places=6)
            for path in out.glob('*.png'):
                with Image.open(path) as image:
                    image.verify()
            content = report.read_text()
            for link in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', content):
                self.assertTrue((report.parent / link).is_file(), link)
            self.assertIn('테스트 전용 데이터', content)
            # A changed input must not reuse stale cluster assignments.
            frame.loc[0, 'age'] += 1
            frame.to_csv(data, index=False)
            result = subprocess.run([sys.executable, str(ROOT / 'analysis_shap.py'), '--data', str(data), '--output', str(out)],
                                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=30)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('differs', result.stderr)


if __name__ == '__main__':
    unittest.main()
