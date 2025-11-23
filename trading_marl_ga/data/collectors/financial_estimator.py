"""
재무 지표 추정기

pykrx에서 직접 제공하지 않는 지표들을 추정
"""

import numpy as np
import FinanceDataReader as fdr


class FinancialEstimator:
    """
    재무 지표 추정
    
    부채비율, 업종 정보 등을 추정하여 제공
    """
    
    # 업종별 평균 부채비율 (2023년 기준, 한국 상장사 평균)
    SECTOR_DEBT_RATIOS = {
        '반도체': 0.35,        # 삼성전자, SK하이닉스 등
        '자동차': 0.85,        # 현대차, 기아 등
        '조선': 1.20,          # HD현대중공업 등
        '제약': 0.45,          # 삼성바이오로직스 등
        '2차전지': 0.60,       # LG에너지솔루션 등
        '금융': 0.90,          # KB금융 등
        '방산': 0.70,          # 한화에어로스페이스 등
        '에너지': 1.00,        # 두산에너빌리티 등
        '기타': 0.80,          # 기본값
    }
    
    def __init__(self):
        """초기화"""
        self.sector_cache = {}  # 종목 코드 -> 업종 매핑
    
    def estimate_debt_ratio(self, ticker, market_cap=None):
        """
        부채비율 추정
        
        Args:
            ticker (str): 종목 코드
            market_cap (float, optional): 시가총액 (원)
        
        Returns:
            float: 추정 부채비율 (0~2.0 범위)
        """
        # 1. 종목별 휴리스틱 (KOSPI 대형주)
        debt_ratio_map = {
            '005930': 0.35,  # 삼성전자
            '000660': 0.40,  # SK하이닉스
            '373220': 0.55,  # LG에너지솔루션
            '207940': 0.30,  # 삼성바이오로직스
            '005380': 0.85,  # 현대차
            '329180': 1.20,  # HD현대중공업
            '034020': 1.00,  # 두산에너빌리티
            '105560': 0.90,  # KB금융
            '000270': 0.80,  # 기아
            '012450': 0.70,  # 한화에어로스페이스
        }
        
        if ticker in debt_ratio_map:
            return debt_ratio_map[ticker]
        
        # 2. 시가총액 기반 추정 (없으면 기본값)
        if market_cap is not None:
            # 시가총액이 클수록 부채비율 낮다고 가정
            if market_cap > 100_000_000_000_000:  # 100조 이상
                return np.random.uniform(0.3, 0.5)
            elif market_cap > 10_000_000_000_000:  # 10조 이상
                return np.random.uniform(0.5, 0.8)
            else:
                return np.random.uniform(0.7, 1.2)
        
        # 3. 기본값
        return 0.80
    
    def get_batch_debt_ratios(self, tickers):
        """
        여러 종목의 부채비율 일괄 추정
        
        Args:
            tickers (list): 종목 코드 리스트
        
        Returns:
            np.ndarray: 부채비율 배열
        """
        return np.array([self.estimate_debt_ratio(ticker) for ticker in tickers])


class InterestRateProvider:
    """
    무위험 이자율 제공
    
    한국은행 기준금리 (시기별 실제 값)
    """
    
    # 한국은행 기준금리 (연도별)
    BASE_RATES = {
        2020: 0.0050,  # 0.50%
        2021: 0.0100,  # 1.00%
        2022: 0.0325,  # 평균 3.25% (1.25% -> 3.25%)
        2023: 0.0350,  # 평균 3.50% (3.50%)
        2024: 0.0325,  # 평균 3.25% (3.50% -> 3.00%)
    }
    
    @staticmethod
    def get_interest_rate(date):
        """
        특정 날짜의 무위험 이자율 반환
        
        Args:
            date: pandas Timestamp
        
        Returns:
            float: 무위험 이자율 (연율)
        """
        year = date.year
        return InterestRateProvider.BASE_RATES.get(year, 0.03)


class MarketSentimentEstimator:
    """
    시장 심리 추정기
    
    변동성, 거래량 등을 기반으로 시장 심리 추정
    """
    
    @staticmethod
    def calculate_sentiment(volatility, returns):
        """
        시장 심리 계산
        
        Args:
            volatility (float): 시장 변동성 (0~1)
            returns (float): 시장 수익률
        
        Returns:
            float: 시장 심리 (0~1, 0.5가 중립)
        """
        # 변동성이 낮고 수익률이 높으면 긍정적
        # 변동성이 높고 수익률이 낮으면 부정적
        
        # 변동성 기여 (낮을수록 좋음)
        vol_factor = 1 - np.clip(volatility / 0.5, 0, 1)  # 0~1
        
        # 수익률 기여
        return_factor = np.tanh(returns * 20) * 0.5 + 0.5  # 0~1
        
        # 종합 (변동성 40%, 수익률 60%)
        sentiment = 0.4 * vol_factor + 0.6 * return_factor
        
        return np.clip(sentiment, 0, 1)

