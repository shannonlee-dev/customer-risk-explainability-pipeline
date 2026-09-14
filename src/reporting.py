"""Render a Korean report from measured JSON/CSV artifacts, without invented results."""
import json
from pathlib import Path
import pandas as pd

LABELS = {'age': '나이', 'annual_income': '연 소득(만원)', 'spending_score': '소비 점수',
          'debt_ratio': '부채비율', 'credit_card_count': '카드 수', 'overdue_count_6m': '최근 6개월 연체 횟수'}


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(map(str, headers)) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def render_report(out, image_prefix='outputs', provenance='입력 CSV의 Mission 23 원본 여부는 확인되지 않았습니다.'):
    cluster = json.loads((out / 'clustering.json').read_text(encoding='utf-8'))
    shap = json.loads((out / 'shap.json').read_text(encoding='utf-8'))
    if cluster['data_sha256'] != shap['data_sha256']:
        raise ValueError('Clustering and SHAP input hashes do not match.')
    predictions = pd.read_csv(out / 'holdout_predictions.csv')
    m = shap['metrics']
    k = cluster['selected_k']
    best = next(s for s in cluster['scores'] if s['k'] == k)
    v1, v2 = cluster['pca_variance_ratio']
    top = shap['importance'][:3]
    def pic(file, title):
        return f'![{title}]({image_prefix}/{file})'
    parts = [
        '# 고객 위험 설명 파이프라인',
        '## 프로젝트 소개',
        f'고객 {cluster["rows"]:,}명의 여섯 입력 변수로 페르소나를 만들고, 연체 분류 모델의 판정 근거를 SHAP으로 설명합니다. '
        f'군집 수는 K={k}, 최대 silhouette는 {best["silhouette"]:.4f}이며, PCA 두 축은 분산의 {v1 + v2:.1%}를 보존합니다. '
        f'홀드아웃 ROC-AUC는 {m["roc_auc"]:.4f}, Average Precision은 {m["average_precision"]:.4f}입니다. '
        f'전체 예측에 크게 기여한 변수는 {", ".join(LABELS[r["feature"]] for r in top)}입니다. '
        '군집은 고객 특성의 요약이며, 개별 위험 판정은 분류 확률과 해당 고객의 기여도를 함께 확인해야 합니다.',
        '## 핵심 특징',
        '- 타겟 두 개를 제외한 여섯 원본 변수만 사용하며, 외부 데이터 병합·증강·입력 컬럼 추가를 하지 않습니다.\n'
        '- 표준화 → K-Means → PCA → 군집 통계 → 홀드아웃 분류 → Global/Local SHAP을 연결합니다.\n'
        '- 모든 그래프는 PNG, 수치 근거는 CSV/JSON으로 저장하며 README는 측정 결과에서 생성합니다.',
        '## 아키텍처',
        '```text\nfinance_data.csv\n  ├─ analysis_clustering.py → src/data.py + src/clustering.py → 군집·PCA·통계\n'
        '  └─ analysis_shap.py → src/modeling.py + src/shap_analysis.py → 평가·SHAP\n'
        '       └─ build_report.py → src/reporting.py → README.md\n```',
        '## 실행 방법',
        'Python 3.13에서 검증했습니다. 정확한 패키지 버전은 `requirements.txt`에 고정되어 있습니다.\n\n'
        '```bash\npython3 -m venv .venv\nsource .venv/bin/activate\npython -m pip install -r requirements.txt\n'
        'python analysis_clustering.py --data data/finance_data.csv\n'
        'python analysis_shap.py --data data/finance_data.csv\n'
        'python build_report.py\npython -m unittest discover -s tests\n```\n\n'
        '다른 원본을 사용할 때 두 분석 명령에 같은 `--data`를 지정합니다. 실행 순서가 바뀌거나 CSV 해시가 다르면 실패합니다. '
        '`--output`으로 출력 폴더를 바꿀 수 있고, 리포트는 `--destination`으로 별도 저장할 수 있습니다. '
        '추론 모델 파일은 불러오지 않으며 동일한 데이터·시드·패키지로 다시 학습합니다.',
        '## 데이터와 평가 설계',
        provenance + '\n\n'
        f'입력 SHA-256: `{cluster["data_sha256"]}`. 행 ID는 CSV의 0 기반 행 번호이며 개인 식별자가 아닙니다. '
        '`credit_score`와 `is_overdue`를 군집화와 분류 입력에서 모두 제외합니다. '
        '신용 점수는 타겟 생성에 직접 쓰였으므로 입력에 넣으면 타겟 생성 규칙을 우회적으로 노출합니다. '
        '결측치는 중앙값으로 대체하며 임의의 이상치 제거는 하지 않습니다. '
        f'이번 입력의 결측치 합계는 {sum(cluster["missing_values"].values())}개입니다. '
        '군집화는 전체 고객에 대한 기술적 분석이므로 전체 입력에 전처리를 적합합니다. '
        '분류는 stratified 75:25, seed=42로 분리하고 중앙값을 학습 행에서만 계산합니다. '
        '군집 ID와 PCA 좌표는 분류 입력에 추가하지 않습니다.\n\n'
        '기존 Mission 23 모델 파일이 없어 RandomForestClassifier(200 trees, max_depth=8, min_samples_leaf=5)를 새로 학습한 실행입니다. '
        '원본 모델 재사용 요구에 대한 대체 실행임을 구분해야 합니다. 임계값은 0.5로 사전에 고정하며 '
        '거절은 P(is_overdue=1) ≥ 0.5, 승인은 그 미만입니다. 과제의 “확률이5 이상”은 0.5의 오기로 해석했습니다. '
        '과제용 판정이며 실제 대출 승인 여부나 인과적 설명을 의미하지 않습니다.',
        table(['지표', '홀드아웃 결과'], [
            ['학습 / 평가 행 수', f'{m["train_rows"]} / {m["test_rows"]}'],
            ['실제 연체율 / AP 무작위 기준', f'{m["test_prevalence"]:.4f}'],
            ['ROC-AUC', f'{m["roc_auc"]:.4f}'], ['Average Precision', f'{m["average_precision"]:.4f}'],
            ['Precision / Recall @0.5', f'{m["precision"]:.4f} / {m["recall"]:.4f}'],
            ['Balanced accuracy', f'{m["balanced_accuracy"]:.4f}'], ['Brier score', f'{m["brier_score"]:.4f}'],
            ['혼동행렬 [[TN, FP], [FN, TP]]', str(m['confusion_matrix'])]]),
        '불균형 데이터에서는 정확도만으로 성능을 판단하지 않습니다. AP는 연체율 기준선과 비교하고, '
        '재현율은 놓치는 연체 고객을, 정밀도는 고위험 판정의 신뢰도를 보여줍니다. '
        '테스트 데이터를 이용한 임계값 최적화는 하지 않았습니다.',
        '## 군집 수와 분리도',
        pic('k_selection.png', 'Elbow와 Silhouette'),
        table(['K', 'Inertia', 'Silhouette'], [[s['k'], f'{s["inertia"]:.1f}', f'{s["silhouette"]:.4f}'] for s in cluster['scores']]),
        f'같은 seed=42의 최대 2,000개 행에서 silhouette를 비교해 최대인 K={k}를 선택했습니다. '
        'Inertia는 군집 내 중심까지의 제곱거리 합이며 K가 증가하면 줄어드는 것이 정상입니다. '
        'Elbow 곡선은 감소 폭이 완만해지는지 확인하는 보조 근거로만 사용했고, 명확한 꺾임을 전제하지 않았습니다. '
        '페르소나 이름은 전체 대비 표준화 평균 차이가 가장 큰 두 변수로 정했으며 이름을 보기 좋게 만들기 위해 K를 변경하지 않았습니다. '
        f'Silhouette {best["silhouette"]:.4f}는 ' + ('약한 분리와 군집 간 중첩을 시사합니다.' if best['silhouette'] < .25 else '일정한 거리 기반 분리의 근거입니다.') +
        ' 고객 집단이 자연적으로 존재한다고 단정할 수는 없습니다.',
        pic('pca_clusters.png', '군집별 PCA 산점도'),
        f'PC1={v1:.2%}, PC2={v2:.2%}, 합계={v1+v2:.2%}이며 2차원 투영에서 분산의 {1-v1-v2:.2%}를 버립니다. '
        f'투영 좌표에서 같은 라벨의 silhouette는 {cluster["silhouette_pca_2d"]:.4f}입니다. '
        '산점도에서 겹치는 점만으로 원래 6차원 군집 품질을 판단하면 안 됩니다. '
        'PC는 표준화된 여섯 원본 변수의 선형결합이지 소득이나 부채비율 자체가 아닙니다. '
        '`outputs/pca_loadings.csv`의 계수는 축의 구성을 보여주지만 위험 기여도는 아닙니다. '
        'PCA의 부호도 임의적이므로 PC1 증가를 위험 증가로 읽지 않습니다.\n\n'
        '스케일링 없이 소득 수천 단위와 0~1 부채비율의 유클리드 거리를 계산하면 소득이 거리를 지배합니다. '
        'StandardScaler로 평균 0·표준편차 1을 만들어 각 변수의 단위 차이를 제거했습니다. '
        '표준화는 이상치의 영향을 없애지는 않으므로 실제 데이터 적용 시 별도 점검이 필요합니다.',
        '## 군집 요약과 페르소나',
        table(['군집 / 페르소나', '인원', '소득 평균(만원)', '부채 평균', '연체횟수 평균', '관측 연체율'],
              [[f'C{p["cluster"]} {p["name"]}', p['count'], f'{p["means"]["annual_income"]:.0f}',
                f'{p["means"]["debt_ratio"]:.3f}', f'{p["means"]["overdue_count_6m"]:.2f}',
                f'{p["observed_overdue_rate"]:.2%}'] for p in cluster['personas']]),
        '관측 연체율은 군집을 만든 뒤 타겟으로 요약한 기술통계입니다. 군집 수 선정이나 거리 계산에 사용하지 않았습니다. '
        '전체 변수의 평균·중앙값·표준편차는 `outputs/cluster_statistics.csv`에 있습니다.'
    ]
    for p in cluster['personas']:
        characteristic = ', '.join(f'{LABELS[f]} {p["means"][f]:.2f} (전체 대비 {p["z_scores"][f]:+.2f} 표준편차)' for f in p['distinctive_features'])
        parts.append(f'### C{p["cluster"]}: {p["name"]}\n\n'
                     f'- 특징: {characteristic}.\n'
                     f'- 해석: 전체의 {p["share"]:.1%}를 차지하는 상대적 특성 집단이며, 군집 내 모든 고객이 같은 위험을 갖지는 않습니다.\n'
                     f'- 연결: 아래 `cluster_{p["cluster"]}` 사례는 이 군집 평균에 가까운 평가 고객입니다. 개별 기여도로 설명을 확인합니다.')
    parts.extend(['## Global SHAP 해석', pic('shap_summary.png', '전체 변수 중요도와 방향성'),
                  f'학습에 사용하지 않은 고객 중 seed=42로 {shap["global_sample_size"]:,}명을 추출했습니다. '
                  'Summary의 각 점은 고객 한 명이고 빨강은 큰 입력값, 파랑은 작은 입력값입니다. '
                  '양의 SHAP은 연체 확률을 기준값보다 올리고 음의 SHAP은 내립니다. '
                  'Global은 전체적으로 무엇이 모델을 움직이는지, Local은 특정 고객이 왜 그 확률을 받았는지 설명합니다.',
                  table(['변수', '평균 |SHAP|', '불순도 중요도'], [[LABELS[r['feature']], f'{r["mean_abs_shap"]:.5f}', f'{r["impurity_importance"]:.5f}'] for r in shap['importance']]),
                  table(['상위 3개 변수', '하위 사분위 평균 SHAP', '상위 사분위 평균 SHAP', '큰 값의 상대적 방향'],
                        [[LABELS[r['feature']], f'{d["low_mean_shap"]:+.4f}', f'{d["high_mean_shap"]:+.4f}',
                          '위험 증가' if d['high_mean_shap'] > d['low_mean_shap'] else '위험 감소']
                         for r in top for d in shap['directions'] if r['feature'] == d['feature']]),
                  'Random Forest의 불순도 중요도는 학습 트리에서 분할로 줄어든 불순도를 누적·정규화하며 방향성이 없습니다. '
                  'SHAP은 특성 조합에 대한 예측 차이의 가중 기여도를 구하고, 여기서는 TreeExplainer의 tree_path_dependent 방식으로 '
                  '학습 트리의 경로 분포를 사용합니다. 평균 절댓값은 영향의 크기이며 각 고객의 부호가 방향을 나타냅니다. '
                  '고유값이 많은 변수에 대한 불순도 중요도의 편향, 상관 변수 사이의 기여 배분, 평가 표본 차이 때문에 순위가 달라질 수 있습니다.',
                  '## Dependence 해석'])
    for d in shap['dependence']:
        parts.extend([pic(f'dependence_{d["feature"]}.png', LABELS[d['feature']] + ' dependence'),
                      f'{LABELS[d["feature"]]}는 ' + ('Summary 상위 변수' if d['feature'] in [r['feature'] for r in top[:2]] else '비즈니스상 중요한 부채 부담 변수') +
                      f'로 선택했습니다. 하위 사분위 구간(≤{d["q25"]:.2f})의 평균 기여는 {d["low_mean_shap"]:+.4f}, '
                      f'상위 사분위 구간(≥{d["q75"]:.2f})은 {d["high_mean_shap"]:+.4f}로, 큰 값의 구간이 상대적으로 연체 확률을 '
                      f'{"높이는" if d["high_mean_shap"] > d["low_mean_shap"] else "낮추는"} 방향입니다. '
                      '이는 다른 조건이 섞인 모델 내 연관성이며, 해당 변수만 바꾸면 이만큼 위험이 변한다는 인과 효과가 아닙니다. '
                      '같은 입력값에서도 점이 수직으로 퍼지는 것은 다른 입력과 상호작용이 있음을 보여주며, 색상은 자동 선택된 다른 입력값입니다. '
                      '사분위 경계가 같은 이산형 변수에서는 두 비교 집단이 겹칠 수 있습니다.'])
    parts.extend(['## Local 사례: 승인·거절과 군집 대표 고객',
                  '승인/거절 예시는 평가 고객 중 예측 확률이 가장 낮거나 높은 고객으로, 전형적 고객을 의미하지 않습니다. '
                  '군집별 예시는 군집의 전체 평균과 표준화 거리가 가장 가까운 평가 고객을 별도로 골랐습니다. '
                  '훈련 표준편차로 거리를 나누며 원본 단위를 유지한 피처값과 확률 기여를 아래에 함께 제시합니다. '
                  '결측치가 있었다면 표의 값은 학습 중앙값으로 대체한 모델 입력입니다.\n\n'
                  '`P(연체=1) = 기준 기대값 + 각 변수 SHAP의 합`입니다. '
                  '이 Random Forest의 raw 출력은 로그오즈가 아닌 확률이며, +0.10은 기준값에 10%p를 더한다는 뜻입니다. '
                  f'Global 표본의 최대 합산 오차는 {shap["max_additivity_error"]:.2e}이며 모든 Local 사례도 1e-6 허용오차로 검증했습니다.'])
    for c in shap['local_cases']:
        drivers = sorted(c['contributions'], key=lambda f: abs(c['contributions'][f]), reverse=True)[:3]
        parts.extend([f'### {c["case"]}: 고객 #{c["row_id"]} / C{c["cluster"]}',
                      pic(f'waterfall_{c["case"]}.png', c['case'] + ' waterfall'),
                      f'판정: **{"거절(고위험)" if c["decision"] == "rejection" else "승인(저위험)"}**, '
                      f'예측 연체 확률 **{c["probability"]:.2%}**, 관측 연체 라벨 {c["actual_is_overdue"]}. '
                      f'기준값 {c["base_value"]:.6f} + 기여도 합 {sum(c["contributions"].values()):+.6f} = {c["probability"]:.6f}.',
                      table(['변수', '모델 입력값', 'SHAP 기여(확률 %p)'],
                            [[LABELS[f], f'{c["features"][f]:.2f}', f'{v*100:+.3f}'] for f, v in c['contributions'].items()]),
                      '주요 이유: ' + '; '.join(f'{LABELS[f]} {c["features"][f]:.2f}가 기준 확률을 {abs(c["contributions"][f])*100:.2f}%p '
                                                f'{"높임" if c["contributions"][f] > 0 else "낮춤"}' for f in drivers) + '.',
                      '영업팀 설명: “비슷한 특성을 가진 고객군 안에서도 이 고객의 입력값이 모델 위험 추정에 기여한 정도는 다릅니다. '
                      '이 확률은 확정된 미래나 실제 승인 결과가 아니며, 상환 상황을 추가 확인한 뒤 안내하겠습니다.”'])
    parts.extend(['## 군집별 실행 제언과 효과 검증',
                  '아래 연체 집중도는 홀드아웃에서 측정한 군집 연체율 ÷ 전체 홀드아웃 연체율입니다. '
                  '마케팅 전환율이나 상담으로 줄어드는 연체율을 측정한 값은 아닙니다. '
                  '연체 집중도가 높으면 같은 100건의 상담에서 위험 고객을 더 많이 만날 수 있다는 가설만 세울 수 있습니다.'])
    business_rows = []
    for p in cluster['personas']:
        group = predictions.loc[predictions.cluster.eq(p['cluster'])]
        rate = group.actual_is_overdue.mean()
        lift = rate / m['test_prevalence']
        high = lift >= 1
        business_rows.append([f'C{p["cluster"]}', len(group), f'{rate:.1%}', f'{lift:.2f}배',
                              '상환 일정·자동이체 상담 / 연체 예방' if high else '낮은 부담의 혜택 안내 / 유지율 실험',
                              '상담 100건당 연체 고객 약 ' + f'{rate*100:.1f}명'])
    parts.append(table(['대상', '평가 n', '관측 연체율', '연체 집중도', '메시지 / 액션', '관측 기반 가설'], business_rows))
    parts.extend(['군집별 무작위 대조군을 두고 연락당 전환율·유지율·30/60일 연체율·수신거부율을 비교합니다. '
                  '예를 들어 기준 전환율 5%, 실험군 6%를 가정하면 상대 개선은 20%지만 이는 **계획용 가정**이며 본 분석 결과가 아닙니다. '
                  '상담 우선순위는 개별 위험 확률과 함께 정하고 군집 이름만으로 한도·가격을 결정하지 않습니다.',
                  '## 한계와 운영 모니터링 제안',
                  '- 합성 데이터 생성 규칙을 학습한 결과는 실제 고객에게 일반화된다는 증거가 아닙니다. '
                  '소비 점수는 지출액이 아니므로 소득 대비 과소비라고 해석하지 않습니다.\n'
                  '- 단일 분할 결과이며 시간 외 검증·교차검증·확률 보정·공정성 분석은 추가 실험입니다. '
                  '연령은 입력에 포함되어 있으므로 실제 적용 전 집단별 오류율과 적합성을 검토해야 합니다.\n'
                  '- 실제 운영 시 매주 입력 결측률·범위·예측 확률·군집 비중을 점검하고, 월별 PSI를 학습 분포와 비교합니다. '
                  'PSI 0.1 초과 주의, 0.2 초과 2회 연속이면 조사한다는 초기 운영 가정을 제안하며 도메인 검증이 필요합니다.\n'
                  '- 라벨 도착 이후 월별 AP·재현율·Brier score를 확인합니다. AP가 기준 대비 10% 이상 하락하거나 '
                  '목표 재현율 미달이면 입력 수집 장애와 모집단 변화를 먼저 조사합니다. 표본 크기와 불확실성도 함께 확인합니다.\n'
                  '- 데이터 오류는 원천 수정 후 재실행하고, 지속적 분포 변화는 최근 라벨 데이터로 재학습합니다. '
                  '시간 기준 검증 후 기존 모델과 비교해 승격하고 성능 악화 시 이전 버전으로 되돌립니다. '
                  '분기별 재검토를 제안하며 모델·전처리·임계값·데이터 해시·패키지 버전을 함께 관리합니다. '
                  '이 정책은 운영 제안이며 자동 모니터링 시스템을 구현한 것은 아닙니다.',
                  '## 산출물과 검증',
                  '`outputs/`에는 K 점수, PCA 좌표·계수, 군집 통계, 평가 예측, SHAP 기여도, JSON 요약 및 PNG가 있습니다. '
                  '단위 테스트는 입력 스키마·타겟 제외·결측치 처리·평가 분리·승인/거절 기준·SHAP 확률 합산을 확인합니다. '
                  '`tests/test_artifacts.py`는 전체 실행 후 산출물의 일관성과 이미지 파일을 검증합니다.'])
    return '\n\n'.join(parts) + '\n'
