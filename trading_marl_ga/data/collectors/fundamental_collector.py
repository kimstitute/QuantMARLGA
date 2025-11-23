"""
펀더멘털 데이터 수집기

pykrx를 사용하여 PER, PBR, DIV, EPS 등의 펀더멘털 지표 수집
"""

from pykrx import stock
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime, timedelta


class FundamentalDataCollector:
    """
    펀더멘털 데이터 수집 및 캐싱
    
    pykrx를 사용하여:
    - PER (주가수익비율)
    - PBR (주가순자산비율) 
    - DIV (배당수익률)
    - EPS (주당순이익)
    - BPS (주당순자산)
    등을 수집
    """
    
    def __init__(self, cache_dir='data/cache'):
        """
        Args:
            cache_dir (str): 캐시 디렉토리 경로
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"[OK] Fundamental Collector 캐시: {self.cache_dir}")
    
    def get_fundamental_data(self, tickers, start_date, end_date, use_cache=True):
        """
        펀더멘털 데이터 수집 (날짜별)
        
        Args:
            tickers (list): 종목 코드 리스트
            start_date (str): 시작일 (YYYY-MM-DD)
            end_date (str): 종료일 (YYYY-MM-DD)
            use_cache (bool): 캐시 사용 여부
        
        Returns:
            dict: {ticker: {date: {per, pbr, div, eps, bps}}}
        """
        print(f"\n[펀더멘털 데이터 수집] {start_date} ~ {end_date}")
        
        # 캐시 파일
        cache_file = self.cache_dir / f"fundamental_{start_date}_{end_date}_{len(tickers)}.pkl"
        
        # 캐시 확인
        if use_cache and cache_file.exists():
            print(f"[OK] 캐시 파일 발견: {cache_file.name}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"[OK] 캐시에서 로드: {len(data)}개 종목")
            return data
        
        # 데이터 수집
        fundamental_data = {}
        
        # 날짜 범위 생성 (월말만, pykrx는 일별 펀더멘털 제공 안 함)
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # 월말 날짜 생성 (대략 월 1회)
        dates = pd.date_range(start, end, freq='M')
        if end not in dates:
            dates = dates.append(pd.DatetimeIndex([end]))
        
        print(f"  수집 날짜: {len(dates)}개 (월말 기준)")
        
        for date in dates:
            date_str = date.strftime('%Y%m%d')
            
            try:
                # pykrx로 시장 전체 펀더멘털 데이터 수집
                df_fundamental = stock.get_market_fundamental(date_str, market="ALL")
                
                if df_fundamental is not None and len(df_fundamental) > 0:
                    # 우리 종목만 필터링
                    for ticker in tickers:
                        if ticker in df_fundamental.index:
                            if ticker not in fundamental_data:
                                fundamental_data[ticker] = {}
                            
                            row = df_fundamental.loc[ticker]
                            
                            # ROE 계산: EPS / BPS (간접 계산)
                            eps = row.get('EPS', np.nan)
                            bps = row.get('BPS', np.nan)
                            
                            if not np.isnan(eps) and not np.isnan(bps) and bps > 0:
                                roe = eps / bps  # ROE = EPS / BPS
                            else:
                                roe = np.nan
                            
                            fundamental_data[ticker][date] = {
                                'per': row.get('PER', np.nan),
                                'pbr': row.get('PBR', np.nan),
                                'div': row.get('DIV', np.nan),  # 배당수익률
                                'eps': eps,
                                'bps': bps,
                                'roe': roe,  # ROE 추가
                            }
                
                print(f"  [OK] {date.strftime('%Y-%m-%d')}: {len([t for t in tickers if t in df_fundamental.index])}개 종목")
            
            except Exception as e:
                print(f"  [WARNING]  {date.strftime('%Y-%m-%d')}: {e}")
        
        print(f"\n[OK] 펀더멘털 수집 완료: {len(fundamental_data)}개 종목")
        
        # 캐시 저장
        if use_cache and fundamental_data:
            with open(cache_file, 'wb') as f:
                pickle.dump(fundamental_data, f)
            print(f"[OK] 캐시 저장: {cache_file.name}")
        
        return fundamental_data
    
    def get_fundamental_for_date(self, ticker, date, fundamental_data):
        """
        특정 날짜의 펀더멘털 데이터 가져오기
        
        날짜가 정확히 없으면 가장 가까운 과거 날짜의 데이터 사용
        
        Args:
            ticker (str): 종목 코드
            date: 날짜 (pandas Timestamp)
            fundamental_data (dict): get_fundamental_data() 결과
        
        Returns:
            dict: {per, pbr, div, eps, bps} or None
        """
        if ticker not in fundamental_data:
            return None
        
        ticker_data = fundamental_data[ticker]
        available_dates = sorted(ticker_data.keys())
        
        if not available_dates:
            return None
        
        # 가장 가까운 과거 날짜 찾기
        past_dates = [d for d in available_dates if d <= date]
        
        if past_dates:
            closest_date = past_dates[-1]  # 가장 최근 과거
            return ticker_data[closest_date]
        
        # 과거 데이터가 없으면 가장 첫 데이터 사용
        return ticker_data[available_dates[0]]
    
    def get_batch_fundamental_for_date(self, tickers, date, fundamental_data):
        """
        여러 종목의 펀더멘털 데이터를 한 번에 가져오기
        
        Args:
            tickers (list): 종목 코드 리스트
            date: 날짜
            fundamental_data (dict): get_fundamental_data() 결과
        
        Returns:
            dict: {ticker: {per, pbr, div, eps, bps}}
        """
        result = {}
        
        for ticker in tickers:
            data = self.get_fundamental_for_date(ticker, date, fundamental_data)
            if data:
                result[ticker] = data
            else:
                # 기본값
                result[ticker] = {
                    'per': 15.0,
                    'pbr': 1.5,
                    'div': 2.0,
                    'eps': 1000.0,
                    'bps': 10000.0,
                    'roe': 0.10,  # 10% (기본값)
                }
        
        return result

