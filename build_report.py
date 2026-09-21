"""현재 군집·SHAP 산출물로 분석 보고서를 생성한다."""

import argparse
import os
from pathlib import Path

from src.reporting import render_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--destination', type=Path, default=Path('README.md'))
    parser.add_argument(
        '--provenance',
        default=(
            '입력 CSV의 Mission 23 원본 여부는 확인되지 않았습니다. '
            '데이터 출처는 실행자가 확인해야 합니다.'
        ),
    )
    args = parser.parse_args()

    prefix = os.path.relpath(args.output.resolve(), args.destination.resolve().parent)
    content = render_report(args.output, prefix, args.provenance)
    args.destination.write_text(content, encoding='utf-8')

    print(f'Report written: {args.destination}')


if __name__ == '__main__':
    main()
