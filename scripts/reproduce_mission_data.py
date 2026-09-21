"""요청할 때만 과제의 고정 시드 데이터 생성 과정을 재현한다.

사용할 수 없는 Mission 23 CSV와 바이트 단위로 동일함을 보장하지는 않는다.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('data/finance_data.csv'))
    args = parser.parse_args()
    destination = args.output

    if destination.exists():
        parser.error(f'Refusing to overwrite existing data: {destination}')

    destination.parent.mkdir(parents=True, exist_ok=True)

    # 원본 생성 코드와 같은 시점에 의존성을 불러온다.
    import numpy as np
    import pandas as pd

    sample_count = 10000
    random_state = 42
    np.random.seed(random_state)

    # 가상의 고객 입력 특성을 생성한다.
    data = {
        'age': np.random.randint(20, 70, sample_count),
        'annual_income': np.random.normal(5000, 2000, sample_count).round(0),
        'spending_score': np.random.randint(1, 100, sample_count),
        'debt_ratio': np.random.uniform(0, 1, sample_count).round(2),
        'credit_card_count': np.random.randint(1, 10, sample_count),
        'overdue_count_6m': np.random.poisson(0.5, sample_count),
    }
    df = pd.DataFrame(data)

    # 생성 노이즈로 낮아진 소득에 하한을 적용한다.
    df['annual_income'] = df['annual_income'].apply(lambda x: max(x, 1500))

    # 소득이 높고 연체가 적을수록 높아지는 신용 점수를 만든다.
    df['credit_score'] = (
        300
        + (df['annual_income'] / 100) * 3
        - (df['overdue_count_6m'] * 50)
        - (df['debt_ratio'] * 100)
        + np.random.normal(0, 30, sample_count)
    )
    df['credit_score'] = df['credit_score'].clip(0, 1000).round(0)

    # 하위 신용 점수와 높은 부채비율을 조합해 불균형 연체 타겟을 만든다.
    threshold = df['credit_score'].quantile(0.15)
    df['is_overdue'] = np.where(
        (df['credit_score'] < threshold)
        & (np.random.rand(sample_count) > 0.2),
        1,
        0,
    )

    # 행 순서를 섞은 뒤 결과를 저장한다.
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    df.to_csv(destination, index=False)

    print('데이터 생성 완료: finance_data.csv')
    print(f'전체 샘플 수: {len(df)}')
    print(f'연체(1) 비율: {df["is_overdue"].mean() * 100:.2f}% (불균형 데이터 확인)')


if __name__ == '__main__':
    main()
