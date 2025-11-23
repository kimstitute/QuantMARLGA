"""
데이터 파이프라인 테스트

주가 데이터, 펀더멘털 데이터, MarketDataManager 테스트
"""

import sys
import numpy as np
from pathlib import Path

# 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from trading_marl_ga.data.market_data_manager import MarketDataManager
from trading_marl_ga.config import config


def test_market_data_manager():
    """MarketDataManager 통합 테스트"""
    
    print("\n" + "="*60)
    print("데이터 파이프라인 테스트 시작")
    print("="*60)
    
    # ============================================
    # Test 1: MarketDataManager 초기화
    # ============================================
    print("\n[Test 1] MarketDataManager 초기화")
    
    manager = MarketDataManager(cache_dir='data/cache')
    
    # 이미 캐시에 있는 데이터 사용 (빠름)
    manager.initialize(
        start_date='2023-01-01',
        end_date='2023-12-31',
        n_stocks=10
    )
    
    print("[OK] 초기화 완료")
    
    # ============================================
    # Test 2: 종목 확인
    # ============================================
    print("\n[Test 2] 종목 확인")
    
    tickers = manager.get_ticker_names()
    print(f"[OK] 종목 수: {len(tickers)}")
    print(f"[OK] 종목 리스트: {tickers[:5]}... (처음 5개)")
    
    # ============================================
    # Test 3: 날짜 범위 확인
    # ============================================
    print("\n[Test 3] 날짜 범위 확인")
    
    start_date, end_date = manager.get_date_range()
    print(f"[OK] 시작일: {start_date}")
    print(f"[OK] 종료일: {end_date}")
    print(f"[OK] 총 거래일: {len(manager.common_dates)}일")
    
    # ============================================
    # Test 4: 특정 날짜 데이터 가져오기
    # ============================================
    print("\n[Test 4] 특정 날짜 시장 데이터")
    
    # 중간 날짜 선택
    test_date = manager.common_dates[len(manager.common_dates)//2]
    print(f"[OK] 테스트 날짜: {test_date}")
    
    market_data = manager.get_market_data_for_date(test_date, lookback=60)
    
    print(f"\n[OK] 시장 데이터 키: {list(market_data.keys())}")
    
    # ============================================
    # Test 5: 데이터 차원 확인
    # ============================================
    print("\n[Test 5] 데이터 차원 확인")
    
    expected_dim = len(tickers)
    
    data_checks = [
        ('prices', expected_dim),
        ('per', expected_dim),
        ('pbr', expected_dim),
        ('roe', expected_dim),
        ('debt_ratio', expected_dim),
        ('volatility', expected_dim),
        ('beta', expected_dim),
        ('sharpe', expected_dim),
        ('correlation', expected_dim),
        ('max_drawdown', expected_dim),
        ('var_95', expected_dim),
    ]
    
    all_passed = True
    
    for key, expected in data_checks:
        actual = market_data[key].shape[0] if hasattr(market_data[key], 'shape') else len(market_data[key])
        status = "[OK]" if actual == expected else "[ERROR]"
        if actual != expected:
            all_passed = False
        print(f"  {status} {key}: {actual} (expected: {expected})")
    
    # 스칼라 지표
    scalar_checks = [
        'market_per',
        'market_sentiment',
        'vix',
        'market_volatility',
        'market_return',
        'market_beta',
        'interest_rate',
    ]
    
    print("\n[OK] 스칼라 지표:")
    for key in scalar_checks:
        value = market_data[key]
        print(f"  - {key}: {value:.4f}")
    
    # ============================================
    # Test 6: 데이터 품질 확인
    # ============================================
    print("\n[Test 6] 데이터 품질 확인")
    
    # NaN 체크
    has_nan = False
    for key in ['prices', 'per', 'pbr', 'roe', 'debt_ratio', 'volatility']:
        arr = market_data[key]
        if np.isnan(arr).any():
            print(f"  [WARNING]  {key}에 NaN 존재!")
            has_nan = True
    
    if not has_nan:
        print("  [OK] 모든 데이터 정상 (NaN 없음)")
    
    # 이상치 체크
    print("\n  [OK] 주요 지표 범위:")
    print(f"    PER: {market_data['per'].min():.2f} ~ {market_data['per'].max():.2f}")
    print(f"    PBR: {market_data['pbr'].min():.2f} ~ {market_data['pbr'].max():.2f}")
    print(f"    ROE: {market_data['roe'].min():.2%} ~ {market_data['roe'].max():.2%}")
    print(f"    Debt: {market_data['debt_ratio'].min():.2%} ~ {market_data['debt_ratio'].max():.2%}")
    print(f"    Volatility: {market_data['volatility'].min():.2%} ~ {market_data['volatility'].max():.2%}")
    print(f"    Beta: {market_data['beta'].min():.2f} ~ {market_data['beta'].max():.2f}")
    
    # ============================================
    # Test 7: 여러 날짜 테스트 (샘플링)
    # ============================================
    print("\n[Test 7] 여러 날짜 샘플링 (5개 날짜)")
    
    sample_dates = [
        manager.common_dates[i] 
        for i in [0, 
                  len(manager.common_dates)//4, 
                  len(manager.common_dates)//2,
                  3*len(manager.common_dates)//4,
                  -1]
    ]
    
    for i, date in enumerate(sample_dates, 1):
        try:
            data = manager.get_market_data_for_date(date, lookback=60)
            print(f"  [{i}/5] {date}: [OK] (가격: {data['prices'][0]:.0f}원)")
        except Exception as e:
            print(f"  [{i}/5] {date}: [ERROR] {e}")
            all_passed = False
    
    # ============================================
    # 최종 결과
    # ============================================
    print("\n" + "="*60)
    if all_passed:
        print("[OK] 모든 테스트 통과!")
    else:
        print("[ERROR] 일부 테스트 실패!")
    print("="*60)
    
    return manager


if __name__ == "__main__":
    try:
        manager = test_market_data_manager()
        print("\n[OK] 데이터 파이프라인 테스트 완료")
        print(f"\n[OK] manager 객체가 생성되었습니다.")
        print(f"[OK] 사용 가능 메서드:")
        print(f"  - manager.get_market_data_for_date(date)")
        print(f"  - manager.get_ticker_names()")
        print(f"  - manager.get_date_range()")
        
    except Exception as e:
        print(f"\n[ERROR] 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

