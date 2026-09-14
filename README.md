# 고객 위험 설명 파이프라인

## 프로젝트 소개

고객 특성으로 군집과 페르소나를 만들고, Random Forest의 연체 예측을 SHAP으로 설명하는 분석 프로젝트입니다. `docs/private/mission.md`와 `rubric.md`에 맞춰 분석 코드와 결과 리포트 생성기를 구현했습니다.

**현재 상태:** Mission 23의 원본 `finance_data.csv`와 학습 모델이 제공되지 않아 최종 데이터 분석·PNG·실측 인사이트는 아직 생성하지 않았습니다. 테스트 전용 데이터로 전체 CLI 실행과 산출물 검증은 통과했습니다. 테스트 결과를 과제 분석 결과로 사용하지 않습니다.

## 핵심 특징

- `credit_score`, `is_overdue`를 제외한 원본 입력 여섯 개만 사용합니다.
- 중앙값 대체와 StandardScaler 후 K=2~8의 Silhouette/Elbow를 비교하고 PCA 설명분산·군집 통계를 저장합니다.
- 학습/평가를 층화 분리하고 학습 데이터만으로 분류용 결측치를 처리합니다.
- TreeExplainer로 연체 확률의 Global Summary, 주요 변수 Dependence, 승인·거절 및 군집 대표 고객 Waterfall을 저장합니다.
- SHAP 기준값과 기여도 합이 모델 확률과 일치하는지 검증합니다.
- 실제 산출물에서 한국어 리포트를 생성합니다. 군집별 전략, 효과 검증 가정, 비기술자 설명, 모니터링 제안도 포함합니다.

## 아키텍처

```text
finance_data.csv
  ├─ analysis_clustering.py → src/data.py + src/clustering.py → outputs/clustering.json · PCA · 통계
  └─ analysis_shap.py → src/modeling.py + src/shap_analysis.py → outputs/shap.json · 평가 · SHAP
       └─ build_report.py → src/reporting.py → README.md
```

## 실행

Python 3.13에서 검증했습니다. `requirements.txt`는 scikit-learn, SHAP 및 실행 의존성의 정확한 버전을 고정합니다.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Mission 23 원본을 data/finance_data.csv에 준비한 뒤 실행
python analysis_clustering.py --data data/finance_data.csv
python analysis_shap.py --data data/finance_data.csv
python build_report.py --provenance '실제로 사용한 데이터의 출처를 입력'

python -m unittest discover -s tests -v
```

`build_report.py`는 이 README를 실측 결과 리포트로 덮어씁니다. 별도 저장하려면 `--destination REPORT.md`를 사용합니다. 분석 출력 위치는 세 명령의 `--output`으로 맞춥니다. 입력 CSV가 달라지면 군집 분석부터 다시 실행해야 합니다.

원본이 없을 때 제공 생성 코드를 재현하는 스크립트는 `scripts/reproduce_mission_data.py`입니다. **현재는 실행하지 않았으며, 재현 데이터 사용 여부의 확인이 필요합니다.** 재현은 기존 Mission 23 파일과의 동일성을 증명하지 않습니다. 기존 데이터 파일은 덮어쓰지 않습니다.

현재 모델링 코드는 기존 모델을 불러오는 대신 고정 시드로 Random Forest를 새로 학습합니다. Mission 23 모델 자체를 재사용하려면 해당 파일과 전처리·학습 분할 정보가 필요합니다.

## 분석 기준

- 원본 CSV의 여덟 컬럼을 검증하며, 입력 컬럼 추가·외부 데이터 병합·증강을 수행하지 않습니다. 별도 결과 파일의 군집 ID와 행 번호는 입력에 추가하지 않습니다.
- 군집화는 전체 고객의 기술적 분석입니다. K 선택은 같은 최대 2,000개 샘플의 Silhouette를 우선하며 Elbow는 보조 근거입니다. PCA는 시각화에만 사용합니다.
- 분류는 stratified 75:25, seed=42입니다. 정확도 대신 ROC-AUC, Average Precision, Precision/Recall, balanced accuracy, Brier score, 혼동행렬을 보고합니다.
- 거절은 `P(is_overdue=1) >= 0.5`, 승인은 그 미만입니다. 문서의 “확률이5 이상”은 0.5의 오기로 해석합니다. 이는 과제용 분류이며 실제 승인 결과가 아닙니다.
- 승인·거절은 평가 고객의 최저·최고 확률 사례입니다. 군집별 대표는 군집 평균과 가까운 평가 고객을 별도로 선정합니다.
- SHAP은 모델 내 예측 기여이며 인과 효과가 아닙니다. 군집 연체 집중도와 가정한 마케팅 효과를 구분해 보고합니다.

## 검증

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python -m unittest discover -s tests -v
```

테스트는 임시 폴더의 240행 테스트 데이터로 전체 분석과 리포트 생성까지 실행합니다. 입력 스키마·타겟 제외·학습 전용 중앙값·평가 분리·확률 기준·SHAP 합산·필수 PNG·이미지 링크·입력 변경 시 오래된 군집 결과 거부를 검증합니다. 임시 산출물은 테스트 종료 후 제거합니다.
