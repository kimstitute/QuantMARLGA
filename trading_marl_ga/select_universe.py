"""
종목 Universe 선정 스크립트

전체 실험 기간(2021-2024)에 대해 결측치 없는 상위 30개 종목을 선정하고 저장합니다.
이렇게 선정된 종목은 train.py와 test.py에서 공통으로 사용되어 일관성을 보장합니다.

사용법:
    python trading_marl_ga/select_universe.py

출력:
    data/selected_tickers.pkl - 선정된 종목 리스트 및 메타데이터
"""

import os
import sys
import pickle
from datetime import datetime

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from trading_marl_ga.data.market_data_manager import MarketDataManager
from trading_marl_ga import config


def select_universe(
    full_start_date="2021-01-01",
    full_end_date="2024-12-31",
    num_tickers=30,
    output_path="data/selected_tickers.pkl"
):
    """
    전체 기간에 대해 결측치 없는 상위 종목을 선정합니다.
    
    Args:
        full_start_date: 전체 기간 시작일
        full_end_date: 전체 기간 종료일
        num_tickers: 선정할 종목 수
        output_path: 저장 경로
    
    Returns:
        선정된 종목 리스트
    """
    print("=" * 80)
    print("종목 Universe 선정")
    print("=" * 80)
    print(f"전체 기간: {full_start_date} ~ {full_end_date}")
    print(f"선정 종목 수: {num_tickers}개")
    print(f"저장 경로: {output_path}")
    print("=" * 80)
    
    # MarketDataManager 초기화
    print("\n[1/3] 데이터 로드 중...")
    data_manager = MarketDataManager()
    data_manager.initialize(
        start_date=full_start_date,
        end_date=full_end_date,
        n_stocks=num_tickers * 2,  # 여유있게 2배 로드
        lookback_days=60  # 최소 60일 데이터 필요
    )
    
    # 선정된 종목 확인
    selected_tickers = data_manager.tickers
    print(f"\n[2/3] 종목 선정 완료: {len(selected_tickers)}개")
    
    # 상위 num_tickers개만 선택
    if len(selected_tickers) > num_tickers:
        selected_tickers = selected_tickers[:num_tickers]
        print(f"   → 상위 {num_tickers}개로 필터링")
    
    # 종목 정보 출력
    print("\n[선정된 종목 리스트]")
    print("-" * 80)
    for i, ticker in enumerate(selected_tickers, 1):
        # 종목 데이터 확인
        ticker_data = data_manager.prices[ticker]
        data_days = len(ticker_data)
        start_date = ticker_data.index[0].strftime("%Y-%m-%d")
        end_date = ticker_data.index[-1].strftime("%Y-%m-%d")
        
        print(f"[{i:2d}] {ticker:6s} | {data_days:4d}일 | {start_date} ~ {end_date}")
        
        if i % 10 == 0:
            print("-" * 80)
    
    # 메타데이터 생성
    metadata = {
        'tickers': selected_tickers,
        'num_tickers': len(selected_tickers),
        'full_start_date': full_start_date,
        'full_end_date': full_end_date,
        'selection_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'data_days': {ticker: len(data_manager.prices[ticker]) for ticker in selected_tickers},
        'config': {
            'min_data_days': 60,
            'selection_method': 'KOSPI 시가총액 상위 + 전체 기간 결측치 없음'
        }
    }
    
    # 저장
    print(f"\n[3/3] 종목 리스트 저장 중...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ 저장 완료: {output_path}")
    
    # 요약 출력
    print("\n" + "=" * 80)
    print("종목 Universe 선정 완료")
    print("=" * 80)
    print(f"선정 종목: {len(selected_tickers)}개")
    print(f"전체 기간: {full_start_date} ~ {full_end_date}")
    print(f"선정 방법: KOSPI 시가총액 상위 + 전체 기간 결측치 없음")
    print(f"선정 일시: {metadata['selection_date']}")
    print("=" * 80)
    print("\n다음 단계:")
    print("  1. python trading_marl_ga/train.py  # 학습 (2021-2023)")
    print("  2. python trading_marl_ga/test.py   # 테스트 (2024)")
    print("=" * 80)
    
    return selected_tickers


def load_universe(path="data/selected_tickers.pkl"):
    """
    저장된 종목 universe를 로드합니다.
    
    Args:
        path: 저장된 파일 경로
    
    Returns:
        dict: 메타데이터 (tickers 키에 종목 리스트)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"종목 universe 파일이 없습니다: {path}\n"
            f"먼저 'python trading_marl_ga/select_universe.py'를 실행하세요."
        )
    
    with open(path, 'rb') as f:
        metadata = pickle.load(f)
    
    return metadata


if __name__ == "__main__":
    # 전체 기간에 대해 종목 선정
    # 2021-2024 전체 기간에 데이터가 있는 종목만 선정
    selected_tickers = select_universe(
        full_start_date="2021-01-01",
        full_end_date="2024-12-31",
        num_tickers=config.N_STOCKS,  # N_STOCKS 사용
        output_path="data/selected_tickers.pkl"
    )

