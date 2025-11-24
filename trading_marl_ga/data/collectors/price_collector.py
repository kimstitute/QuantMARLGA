"""
주가 데이터 수집기

FinanceDataReader를 사용하여 한국 주식 시장 데이터 수집
"""

import FinanceDataReader as fdr
import pandas as pd
from pathlib import Path
import pickle
from datetime import datetime


class PriceDataCollector:
    """
    주가 데이터 수집 및 캐싱
    
    FinanceDataReader를 사용하여:
    - KOSPI 시가총액 상위 종목 선택
    - 주가 데이터 (OHLCV) 수집
    - 로컬 캐싱 (재사용 가능)
    """
    
    def __init__(self, cache_dir='data/cache'):
        """
        Args:
            cache_dir (str): 캐시 디렉토리 경로
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[OK] 캐시 디렉토리: {self.cache_dir}")
    
    def get_kospi_top_tickers(self, top_n=10):
        """
        KOSPI 시가총액 상위 N개 종목 선택
        
        Args:
            top_n (int): 선택할 종목 수
        
        Returns:
            list: 종목 코드 리스트 (예: ['005930', '000660', ...])
        """
        print(f"\n[종목 선택] KOSPI 시가총액 상위 {top_n}개")
        
        try:
            # KOSPI 전체 종목 리스트
            kospi = fdr.StockListing('KOSPI')
            
            # 시가총액 결측치 제거
            kospi = kospi.dropna(subset=['Marcap'])
            
            # 시가총액이 0인 종목 제외
            kospi = kospi[kospi['Marcap'] > 0]
            
            # 시가총액 기준 정렬 (단위: 억원)
            kospi = kospi.sort_values('Marcap', ascending=False)
            
            # 우선주 제외 (종목명에 '우' 포함)
            kospi = kospi[~kospi['Name'].str.contains('우', na=False)]
            
            # 상위 N개 * 2 선택 (데이터 부족 종목 필터링 위해 여유분 확보)
            top_stocks = kospi.head(top_n * 2)
            
            tickers = top_stocks['Code'].tolist()
            names = top_stocks['Name'].tolist()
            marcaps = top_stocks['Marcap'].tolist()
            
            print(f"[OK] 후보 종목 {len(tickers)}개 (시가총액 상위)")
            print(f"[OK] 상위 5개: {', '.join([f'{names[i]}({int(marcaps[i]):,}억)' for i in range(min(5, len(names)))])}")
            
            return tickers[:top_n]
        
        except Exception as e:
            print(f"[ERROR] 종목 선택 실패: {e}")
            print(f"[ERROR] 상세: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            
            # 폴백: 기본 종목 리스트 (2023년 시가총액 기준)
            default_tickers = [
                '005930',  # 삼성전자
                '000660',  # SK하이닉스
                '373220',  # LG에너지솔루션
                '207940',  # 삼성바이오로직스
                '005380',  # 현대차
                '051910',  # LG화학
                '006400',  # 삼성SDI
                '035420',  # NAVER
                '035720',  # 카카오
                '105560',  # KB금융
                '000270',  # 기아
                '055550',  # 신한지주
                '012330',  # 현대모비스
                '028260',  # 삼성물산
                '068270',  # 셀트리온
                '086790',  # 하나금융지주
                '005490',  # POSCO홀딩스
                '015760',  # 한국전력
                '032830',  # 삼성생명
                '017670',  # SK텔레콤
                '096770',  # SK이노베이션
                '009150',  # 삼성전기
                '011170',  # 롯데케미칼
                '010950',  # S-Oil
                '018260',  # 삼성에스디에스
                '047050',  # 포스코인터내셔널
                '009540',  # HD한국조선해양
                '011070',  # LG이노텍
                '034020',  # 두산에너빌리티
                '003550',  # LG
            ]
            print(f"[WARNING]  기본 종목 리스트 사용: {len(default_tickers)}개")
            return default_tickers[:top_n]
    
    def get_price_data(self, tickers, start_date, end_date, use_cache=True):
        """
        주가 데이터 수집 (OHLCV)
        
        Args:
            tickers (list): 종목 코드 리스트
            start_date (str): 시작일 (YYYY-MM-DD)
            end_date (str): 종료일 (YYYY-MM-DD)
            use_cache (bool): 캐시 사용 여부
        
        Returns:
            dict: {ticker: DataFrame} 형태
                DataFrame 컬럼: Open, High, Low, Close, Volume, Change
        """
        print(f"\n[데이터 수집] {start_date} ~ {end_date}")
        
        # 캐시 파일 경로
        cache_file = self.cache_dir / f"prices_{start_date}_{end_date}_{len(tickers)}.pkl"
        
        # 캐시 확인
        if use_cache and cache_file.exists():
            print(f"[OK] 캐시 파일 발견: {cache_file.name}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"[OK] 캐시에서 로드: {len(data)}개 종목")
            return data
        
        # 데이터 수집
        data = {}
        success_count = 0
        
        for i, ticker in enumerate(tickers, 1):
            try:
                print(f"  [{i}/{len(tickers)}] {ticker} 수집 중...", end=' ')
                
                # FinanceDataReader로 데이터 수집
                df = fdr.DataReader(ticker, start_date, end_date)
                
                if df is not None and len(df) > 0:
                    data[ticker] = df
                    success_count += 1
                    print(f"[OK] ({len(df)}일)")
                else:
                    print("[ERROR] 데이터 없음")
            
            except Exception as e:
                print(f"[ERROR] 오류: {e}")
        
        print(f"\n[OK] 수집 완료: {success_count}/{len(tickers)}개 성공")
        
        # 캐시 저장
        if use_cache and data:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"[OK] 캐시 저장: {cache_file.name}")
        
        return data
    
    def clear_cache(self):
        """캐시 디렉토리 삭제"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            print(f"[OK] 캐시 삭제 완료: {self.cache_dir}")

