"""
OpenDart 재무제표 수집기

전자공시시스템(DART)에서 기업의 재무제표 데이터 수집
- 부채비율: 실제 계산 (부채총계 / 자본총계)
- ROE: 실제 계산 (당기순이익 / 자본총계)
- 자산총계, 매출액, 영업이익 등
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
import time

try:
    import OpenDartReader
    OPENDART_AVAILABLE = True
except ImportError as e:
    OPENDART_AVAILABLE = False
    OpenDartReader = None
    print(f"[WARNING]  OpenDartReader import 실패: {e}")
except Exception as e:
    OPENDART_AVAILABLE = False
    OpenDartReader = None
    print(f"[WARNING]  OpenDartReader 초기화 오류: {e}")

from config import config


class OpenDartCollector:
    """
    OpenDart 재무제표 데이터 수집 및 캐싱
    
    기능:
    1. 재무상태표 (자산, 부채, 자본)
    2. 손익계산서 (매출, 영업이익, 당기순이익)
    3. 핵심 지표 계산 (부채비율, ROE, 영업이익률)
    """
    
    # 종목 코드 -> DART 기업 코드 매핑 (KOSPI 대형주)
    CORP_CODE_MAP = {
        '005930': '00126380',  # 삼성전자
        '000660': '00164779',  # SK하이닉스
        '373220': '01445758',  # LG에너지솔루션
        '207940': '00413046',  # 삼성바이오로직스
        '005380': '00164742',  # 현대차
        '329180': '00990314',  # HD현대중공업
        '034020': '00150785',  # 두산에너빌리티
        '105560': '00138905',  # KB금융
        '000270': '00161842',  # 기아
        '012450': '00102527',  # 한화에어로스페이스
    }
    
    def __init__(self, api_key=None, cache_dir='data/cache'):
        """
        Args:
            api_key (str): OpenDart API 키
            cache_dir (str): 캐시 디렉토리
        """
        self.api_key = api_key or config.OPENDART_API_KEY
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        if not OPENDART_AVAILABLE:
            print("[ERROR] OpenDartReader를 사용할 수 없습니다.")
            self.dart = None
        elif not self.api_key:
            print("[ERROR] OpenDart API 키가 설정되지 않았습니다.")
            self.dart = None
        else:
            try:
                self.dart = OpenDartReader(self.api_key)
                print(f"[OK] OpenDart 초기화 완료 (캐시: {self.cache_dir})")
            except Exception as e:
                print(f"[ERROR] OpenDart 초기화 실패: {e}")
                self.dart = None
    
    def get_financial_statements(self, tickers, start_date, end_date, use_cache=True):
        """
        재무제표 데이터 수집
        
        Args:
            tickers (list): 종목 코드 리스트
            start_date (str): 시작일 (YYYY-MM-DD)
            end_date (str): 종료일 (YYYY-MM-DD)
            use_cache (bool): 캐시 사용 여부
        
        Returns:
            dict: {ticker: {date: {assets, liabilities, equity, revenue, ...}}}
        """
        if not self.dart:
            print("[ERROR] OpenDart를 사용할 수 없습니다. 추정값 사용.")
            return None
        
        print(f"\n[OpenDart 재무제표 수집] {start_date} ~ {end_date}")
        
        # 캐시 파일
        cache_file = self.cache_dir / f"opendart_{start_date}_{end_date}_{len(tickers)}.pkl"
        
        # 캐시 확인
        if use_cache and cache_file.exists():
            print(f"[OK] 캐시 파일 발견: {cache_file.name}")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            print(f"[OK] 캐시에서 로드: {len(data)}개 종목")
            return data
        
        # 재무제표 수집
        financial_data = {}
        
        # 연도 범위
        start_year = int(start_date[:4])
        end_year = int(end_date[:4])
        years = list(range(start_year, end_year + 1))
        
        print(f"  수집 연도: {years}")
        
        for ticker in tickers:
            if ticker not in self.CORP_CODE_MAP:
                print(f"  [WARNING]  {ticker}: 기업 코드 미등록")
                continue
            
            corp_code = self.CORP_CODE_MAP[ticker]
            financial_data[ticker] = {}
            
            print(f"  [{ticker}] 수집 중...", end=' ')
            
            try:
                # 연도별 재무제표 수집
                for year in years:
                    # API 호출 제한 (초당 1건)
                    time.sleep(0.3)
                    
                    # 재무상태표 (BS: Balance Sheet)
                    try:
                        bs = self.dart.finstate(
                            corp=corp_code,
                            bsns_year=year,
                            reprt_code='11011'  # 사업보고서
                        )
                        
                        if bs is not None and len(bs) > 0:
                            # 자산총계, 부채총계, 자본총계 추출
                            assets = self._extract_amount(bs, ['자산총계'])
                            liabilities = self._extract_amount(bs, ['부채총계'])
                            equity = self._extract_amount(bs, ['자본총계'])
                            
                            # 손익계산서 (IS: Income Statement)
                            is_data = self.dart.finstate(
                                corp=corp_code,
                                bsns_year=year,
                                reprt_code='11011'
                            )
                            
                            revenue = self._extract_amount(is_data, ['매출액', '수익(매출액)'])
                            operating_income = self._extract_amount(is_data, ['영업이익'])
                            net_income = self._extract_amount(is_data, ['당기순이익'])
                            
                            # 연말 날짜
                            date = pd.Timestamp(f"{year}-12-31")
                            
                            financial_data[ticker][date] = {
                                'assets': assets,
                                'liabilities': liabilities,
                                'equity': equity,
                                'revenue': revenue,
                                'operating_income': operating_income,
                                'net_income': net_income,
                            }
                    
                    except Exception as e:
                        print(f"[{year} 실패: {e}]", end=' ')
                        continue
                
                print(f"[OK] ({len(financial_data[ticker])}년)")
            
            except Exception as e:
                print(f"[ERROR] {e}")
        
        print(f"\n[OK] OpenDart 수집 완료: {len(financial_data)}개 종목")
        
        # 캐시 저장
        if use_cache and financial_data:
            with open(cache_file, 'wb') as f:
                pickle.dump(financial_data, f)
            print(f"[OK] 캐시 저장: {cache_file.name}")
        
        return financial_data
    
    def _extract_amount(self, df, keywords):
        """
        재무제표에서 특정 항목 금액 추출
        
        Args:
            df (DataFrame): 재무제표 DataFrame
            keywords (list): 검색 키워드 리스트
        
        Returns:
            float: 금액 (백만원 단위) 또는 NaN
        """
        if df is None or len(df) == 0:
            return np.nan
        
        for keyword in keywords:
            # 계정명으로 검색
            matched = df[df['account_nm'].str.contains(keyword, na=False)]
            if len(matched) > 0:
                # 금액 추출 (문자열 -> 숫자)
                amount_str = matched.iloc[0]['thstrm_amount']  # 당기금액
                try:
                    amount = float(str(amount_str).replace(',', ''))
                    return amount  # 이미 백만원 단위
                except:
                    return np.nan
        
        return np.nan
    
    def calculate_financial_ratios(self, financial_data):
        """
        재무 비율 계산
        
        Args:
            financial_data (dict): get_financial_statements() 결과
        
        Returns:
            dict: {ticker: {date: {debt_ratio, roe, operating_margin}}}
        """
        ratios = {}
        
        for ticker, data in financial_data.items():
            ratios[ticker] = {}
            
            for date, financials in data.items():
                assets = financials['assets']
                liabilities = financials['liabilities']
                equity = financials['equity']
                revenue = financials['revenue']
                operating_income = financials['operating_income']
                net_income = financials['net_income']
                
                # 부채비율 = 부채총계 / 자본총계
                if not np.isnan(liabilities) and not np.isnan(equity) and equity > 0:
                    debt_ratio = liabilities / equity
                else:
                    debt_ratio = np.nan
                
                # ROE = 당기순이익 / 자본총계
                if not np.isnan(net_income) and not np.isnan(equity) and equity > 0:
                    roe = net_income / equity
                else:
                    roe = np.nan
                
                # 영업이익률 = 영업이익 / 매출액
                if not np.isnan(operating_income) and not np.isnan(revenue) and revenue > 0:
                    operating_margin = operating_income / revenue
                else:
                    operating_margin = np.nan
                
                ratios[ticker][date] = {
                    'debt_ratio': debt_ratio,
                    'roe': roe,
                    'operating_margin': operating_margin,
                }
        
        return ratios
    
    def get_financial_ratio_for_date(self, ticker, date, financial_ratios):
        """
        특정 날짜의 재무 비율 가져오기
        
        가장 최근 연말 결산 데이터 사용 (날짜가 정확히 없으면 가장 가까운 과거)
        
        Args:
            ticker (str): 종목 코드
            date: 날짜 (pandas Timestamp)
            financial_ratios (dict): calculate_financial_ratios() 결과
        
        Returns:
            dict: {debt_ratio, roe, operating_margin} or None
        """
        if ticker not in financial_ratios:
            return None
        
        ticker_data = financial_ratios[ticker]
        available_dates = sorted(ticker_data.keys())
        
        if not available_dates:
            return None
        
        # 가장 가까운 과거 날짜 (연말 결산)
        past_dates = [d for d in available_dates if d <= date]
        
        if past_dates:
            closest_date = past_dates[-1]
            return ticker_data[closest_date]
        
        # 과거 데이터가 없으면 가장 첫 데이터
        return ticker_data[available_dates[0]]

