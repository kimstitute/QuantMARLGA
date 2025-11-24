"""
시장 데이터 통합 관리자 (학습 종목 지정 가능 버전)

실제 주식 데이터를 수집하고 팩터를 계산하여 BacktestEnv2에 제공
tickers 파라미터로 학습 시 사용한 종목을 동일하게 테스트 가능
"""

import numpy as np
import pandas as pd
from pathlib import Path
from data.collectors.price_collector import PriceDataCollector
from data.collectors.fundamental_collector import FundamentalDataCollector
from data.collectors.opendart_collector import OpenDartCollector
from data.collectors.financial_estimator import (
    FinancialEstimator, 
    InterestRateProvider, 
    MarketSentimentEstimator
)
from config import config


class MarketDataManager2:
    """
    시장 데이터 통합 관리 (학습 종목 지정 가능)
    
    기능:
    1. 주가 데이터 수집 (PriceDataCollector 사용)
    2. 펀더멘털 데이터 수집 (FundamentalDataCollector 사용)
    3. 팩터 계산 (PER, PBR, ROE, Volatility 등)
    4. 특정 날짜의 시장 데이터 제공
    5. 학습 시 사용한 종목을 지정하여 테스트 가능 (tickers 파라미터)
    """
    
    def __init__(self, cache_dir='data/cache'):
        """
        Args:
            cache_dir (str): 캐시 디렉토리 경로
        """
        self.price_collector = PriceDataCollector(cache_dir)
        self.fundamental_collector = FundamentalDataCollector(cache_dir)
        self.opendart_collector = OpenDartCollector(cache_dir=cache_dir)
        self.financial_estimator = FinancialEstimator()
        self.tickers = None
        self.price_data = None
        self.fundamental_data = None
        self.opendart_data = None  # 재무제표 데이터
        self.financial_ratios = None  # 계산된 재무 비율
        self.common_dates = None
        self.all_dates = None  # 전체 날짜 (lookback 포함)
        self.backtest_dates = None  # 백테스트용 날짜 (슬라이싱)
        
        print(f"[OK] MarketDataManager2 초기화")
    
    def initialize(self, start_date, end_date, n_stocks=10, lookback_days=60, tickers=None):
        """
        데이터 초기화 및 수집
        
        Args:
            start_date (str): 백테스트 시작일 (YYYY-MM-DD)
            end_date (str): 종료일 (YYYY-MM-DD)
            n_stocks (int): 종목 수
            lookback_days (int): 팩터 계산용 과거 데이터 일수
            tickers (list, optional): 사용할 종목 리스트 (지정 시 시가총액 선택 스킵)
        """
        # 실제 데이터는 lookback_days 이전부터 로드
        # 거래일 기준이므로 달력일로는 약 1.5배 필요 (주말/공휴일 제외)
        from datetime import datetime, timedelta
        calendar_days = int(lookback_days * 1.5)  # 거래일 → 달력일 변환
        actual_start = (datetime.strptime(start_date, '%Y-%m-%d') - 
                       timedelta(days=calendar_days)).strftime('%Y-%m-%d')
        
        print(f"\n{'='*60}")
        print(f"MarketDataManager2 데이터 로드")
        print(f"{'='*60}")
        print(f"백테스트 기간: {start_date} ~ {end_date}")
        print(f"데이터 로드: {actual_start} ~ {end_date} (거래일 {lookback_days}일 확보)")
        print(f"종목 수: {n_stocks}")
        
        # 원래 start_date 저장 (나중에 필터링용)
        self.backtest_start_date = start_date
        
        # 1. 종목 선택
        if tickers is not None:
            # 명시적으로 지정된 종목 리스트 사용 (테스트 시)
            print(f"[INFO] 지정된 종목 리스트 사용: {len(tickers)}개")
            candidate_tickers = tickers
        else:
            # 시가총액 상위 종목 선택 (학습 시)
            print(f"[INFO] 시가총액 상위 종목 자동 선택")
            candidate_tickers = self.price_collector.get_kospi_top_tickers(n_stocks * 2)
        
        # 2. 주가 데이터 수집 (lookback 포함)
        candidate_price_data = self.price_collector.get_price_data(
            candidate_tickers, 
            actual_start,  # lookback 포함한 시작일
            end_date,
            use_cache=True
        )
        
        # 3. 데이터 필터링
        if tickers is not None:
            # 지정된 종목 리스트 사용 시: 필터링 없이 모두 유지, 결측값 처리
            print(f"\n[데이터 처리] 지정된 종목 리스트 사용 - 결측값 forward fill")
            valid_tickers = []
            valid_price_data = {}
            
            for ticker in candidate_tickers:
                if ticker in candidate_price_data:
                    df = candidate_price_data[ticker].copy()
                    # 결측값 처리: forward fill (이전 값으로 채우기)
                    df = df.fillna(method='ffill').fillna(method='bfill')
                    valid_tickers.append(ticker)
                    valid_price_data[ticker] = df
                else:
                    print(f"  [WARNING] {ticker}: 데이터 없음 (스킵)")
            
            if len(valid_tickers) < len(candidate_tickers):
                print(f"  [INFO] {len(candidate_tickers)}개 중 {len(valid_tickers)}개 데이터 확보")
        else:
            # 자동 선택 시: 데이터 기간이 충분한 종목만 선택
            min_data_days = lookback_days
            start_date_dt = pd.to_datetime(start_date)
            valid_tickers = []
            valid_price_data = {}
            
            print(f"\n[데이터 필터링] 최소 {min_data_days}일 이상 & {start_date} 이전부터 데이터 존재")
            for ticker in candidate_tickers:
                if ticker in candidate_price_data:
                    df = candidate_price_data[ticker]
                    data_len = len(df)
                    data_start = df.index[0]
                    
                    # 조건: 최소 일수 + start_date 이전부터 존재
                    if data_len >= min_data_days and data_start <= start_date_dt:
                        valid_tickers.append(ticker)
                        valid_price_data[ticker] = df
                        if len(valid_tickers) >= n_stocks:
                            break  # 필요한 개수만큼 확보하면 중단
                    else:
                        skip_reason = []
                        if data_len < min_data_days:
                            skip_reason.append(f"데이터 부족 ({data_len}일 < {min_data_days}일)")
                        if data_start > start_date_dt:
                            skip_reason.append(f"늦은 시작 ({data_start.date()} > {start_date})")
                        print(f"  [SKIP] {ticker}: {', '.join(skip_reason)}")
            
            if len(valid_tickers) < n_stocks:
                print(f"[WARNING]  요청 종목 수({n_stocks}개)보다 적음 ({len(valid_tickers)}개만 확보)")
        
        self.tickers = valid_tickers
        self.price_data = valid_price_data
        
        print(f"[OK] 유효 종목: {len(self.tickers)}개")
        if len(self.tickers) == 0:
            raise ValueError("유효한 종목이 없습니다. 날짜 범위를 조정하거나 종목 수를 늘려주세요.")
        
        # 4. 펀더멘털 데이터 수집 (PER, PBR 등) - lookback 포함
        try:
            self.fundamental_data = self.fundamental_collector.get_fundamental_data(
                self.tickers,
                actual_start,  # lookback 포함
                end_date,
                use_cache=True
            )
            print(f"[OK] 펀더멘털 데이터 로드 완료 (PER, PBR)")
        except Exception as e:
            print(f"[WARNING]  펀더멘털 데이터 수집 실패 (가상 데이터 사용): {e}")
            self.fundamental_data = None
        
        # 5. OpenDart 재무제표 데이터 수집 (부채비율, ROE) - lookback 포함
        try:
            self.opendart_data = self.opendart_collector.get_financial_statements(
                self.tickers,
                actual_start,  # lookback 포함
                end_date,
                use_cache=True
            )
            if self.opendart_data:
                self.financial_ratios = self.opendart_collector.calculate_financial_ratios(
                    self.opendart_data
                )
                print(f"[OK] OpenDart 재무제표 로드 완료 (부채비율, ROE)")
            else:
                print(f"[WARNING]  OpenDart 사용 불가 (추정값 사용)")
                self.financial_ratios = None
        except Exception as e:
            print(f"[WARNING]  OpenDart 데이터 수집 실패 (추정값 사용): {e}")
            self.opendart_data = None
            self.financial_ratios = None
        
        # 6. 공통 거래일 추출
        if self.price_data:
            all_dates = [set(df.index) for df in self.price_data.values()]
            all_common_dates = sorted(set.intersection(*all_dates))
            
            # 백테스트 시작일 이후만 필터링 (lookback 데이터는 팩터 계산용으로만 사용)
            backtest_start_dt = pd.to_datetime(self.backtest_start_date)
            self.common_dates = [d for d in all_common_dates if d >= backtest_start_dt]
            
            # 전체 데이터는 보존 (팩터 계산용)
            self.all_dates = all_common_dates
            
            print(f"\n{'='*60}")
            print(f"[OK] 초기화 완료")
            print(f"  - 종목: {len(self.tickers)}개")
            print(f"  - 전체 데이터: {len(self.all_dates)}일 ({self.all_dates[0]} ~ {self.all_dates[-1]})")
            print(f"  - 백테스트 기간: {len(self.common_dates)}일 ({self.common_dates[0]} ~ {self.common_dates[-1]})")
            print(f"  - PER/PBR: {'[OK] 실제' if self.fundamental_data else '[ERROR] 가상'}")
            print(f"  - ROE/부채비율: {'[OK] 실제 (OpenDart)' if self.financial_ratios else '[WARNING]  추정'}")
            print(f"{'='*60}\n")
        else:
            raise ValueError("[ERROR] 주가 데이터 수집 실패!")
    
    def get_market_data_for_date(self, date, lookback=60):
        """
        특정 날짜의 시장 데이터 및 팩터 제공
        
        Args:
            date: 날짜 (pandas Timestamp 또는 datetime)
            lookback (int): 팩터 계산용 과거 데이터 일수
        
        Returns:
            dict: 시장 데이터
                - 'prices': (n_stocks,) 현재 종가
                - 'per': (n_stocks,) PER (가상 데이터)
                - 'pbr': (n_stocks,) PBR (가상 데이터)
                - 'roe': (n_stocks,) ROE (가상 데이터)
                - 'debt_ratio': (n_stocks,) 부채비율 (가상 데이터)
                - 'volatility': (n_stocks,) 변동성 (실제 계산)
                - 'beta': (n_stocks,) 베타 (실제 계산)
                - 'sharpe': (n_stocks,) 샤프 (실제 계산)
                - 'correlation': (n_stocks,) 시장 상관계수 (실제 계산)
                - ... 기타 팩터
        """
        # 날짜 인덱스 (전체 데이터 기준)
        try:
            date_idx = self.all_dates.index(date)
        except ValueError:
            raise ValueError(f"[ERROR] 날짜 {date}가 거래일에 없습니다!")
        
        # ===========================================================
        # 1. 현재 가격 (종가)
        # ===========================================================
        prices = np.array([
            self.price_data[ticker].loc[date, 'Close']
            for ticker in self.tickers
        ])
        
        # ===========================================================
        # 2. 과거 데이터로 팩터 계산 (all_dates 기준)
        # ===========================================================
        start_idx = max(0, date_idx - lookback)
        
        # 데이터가 충분하지 않은 경우 (최소 10일 필요)
        # 정상적으로는 발생하지 않음 (데이터 로드시 lookback 포함)
        available_days = date_idx - start_idx
        if available_days < 10:
            # 실제 데이터 기반 기본값 반환
            n_stocks = len(self.tickers)
            
            # 펀더멘털: 실제 데이터 사용 (PER, PBR)
            if self.fundamental_data:
                fundamental_for_date = self.fundamental_collector.get_batch_fundamental_for_date(
                    self.tickers, date, self.fundamental_data
                )
                per_default = np.array([fundamental_for_date[t]['per'] for t in self.tickers])
                pbr_default = np.array([fundamental_for_date[t]['pbr'] for t in self.tickers])
                per_default = np.nan_to_num(per_default, nan=15.0)
                pbr_default = np.nan_to_num(pbr_default, nan=1.5)
            else:
                per_default = np.ones(n_stocks) * 15.0
                pbr_default = np.ones(n_stocks) * 1.5
            
            # ROE, 부채비율: OpenDart 사용
            if self.financial_ratios:
                roe_list = []
                debt_ratio_list = []
                for ticker in self.tickers:
                    ratios = self.opendart_collector.get_financial_ratio_for_date(
                        ticker, date, self.financial_ratios
                    )
                    if ratios:
                        roe_list.append(ratios['roe'])
                        debt_ratio_list.append(ratios['debt_ratio'])
                    else:
                        roe_list.append(0.10)
                        debt_ratio_list.append(0.80)
                roe_default = np.array(roe_list)
                debt_ratio_default = np.array(debt_ratio_list)
            else:
                roe_default = np.ones(n_stocks) * 0.10
                debt_ratio_default = self.financial_estimator.get_batch_debt_ratios(self.tickers)
            
            return {
                'prices': prices,
                'per': per_default,
                'pbr': pbr_default,
                'roe': roe_default,
                'debt_ratio': debt_ratio_default,
                'volatility': np.ones(n_stocks) * 0.2,
                'beta': np.ones(n_stocks),
                'sharpe': np.zeros(n_stocks),
                'correlation': np.ones(n_stocks) * 0.5,
                'max_drawdown': np.zeros(n_stocks),
                'var_95': np.ones(n_stocks) * 0.05,
                'market_per': per_default.mean(),
                'market_sentiment': 0.5,  # 중립 (데이터 부족)
                'vix': 20.0,
                'market_volatility': 0.2,
                'market_return': 0.0,
                'market_beta': 1.0,
                'interest_rate': InterestRateProvider.get_interest_rate(date),  # 실제 기준금리
            }
        
        # 2-1. 변동성 (Volatility)
        volatilities = []
        for ticker in self.tickers:
            returns = self.price_data[ticker]['Close'].pct_change()
            vol = returns.iloc[start_idx:date_idx].std() * np.sqrt(252)  # 연율화
            volatilities.append(vol if not np.isnan(vol) else 0.2)  # 기본값
        volatilities = np.array(volatilities)
        
        # 2-2. 시장 수익률 (Market Return)
        # 전체 종목 평균 수익률
        all_returns = []
        for ticker in self.tickers:
            ret = self.price_data[ticker]['Close'].pct_change()
            all_returns.append(ret.iloc[start_idx:date_idx])
        market_returns = pd.concat(all_returns, axis=1).mean(axis=1)
        
        # 2-3. 베타 (Beta)
        betas = []
        for ticker in self.tickers:
            stock_returns = self.price_data[ticker]['Close'].pct_change().iloc[start_idx:date_idx]
            
            # 공통 인덱스로 align
            stock_clean = stock_returns.dropna()
            market_clean = market_returns.loc[stock_clean.index]
            
            if len(stock_clean) > 1 and len(market_clean) > 1:
                # Covariance / Variance
                try:
                    cov = np.cov(stock_clean, market_clean)[0, 1]
                    var = np.var(market_clean)
                    beta = cov / var if var > 0 else 1.0
                    betas.append(beta if not np.isnan(beta) else 1.0)
                except:
                    betas.append(1.0)  # 기본값
            else:
                betas.append(1.0)  # 기본값
        betas = np.array(betas)
        
        # 2-4. 샤프 비율 (Sharpe Ratio)
        sharpes = []
        for ticker in self.tickers:
            stock_returns = self.price_data[ticker]['Close'].pct_change().iloc[start_idx:date_idx]
            mean_ret = stock_returns.mean()
            std_ret = stock_returns.std()
            sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
            sharpes.append(sharpe if not np.isnan(sharpe) else 0.0)
        sharpes = np.array(sharpes)
        
        # 2-5. 시장 상관계수 (Correlation with Market)
        correlations = []
        for ticker in self.tickers:
            stock_returns = self.price_data[ticker]['Close'].pct_change().iloc[start_idx:date_idx]
            corr = stock_returns.corr(market_returns)
            correlations.append(corr if not np.isnan(corr) else 0.0)
        correlations = np.array(correlations)
        
        # 2-6. 최대 낙폭 (Max Drawdown)
        max_drawdowns = []
        for ticker in self.tickers:
            prices_hist = self.price_data[ticker]['Close'].iloc[start_idx:date_idx+1]
            cumulative = prices_hist / prices_hist.iloc[0]
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            max_drawdowns.append(abs(max_dd) if not np.isnan(max_dd) else 0.0)
        max_drawdowns = np.array(max_drawdowns)
        
        # ===========================================================
        # 3. 펀더멘털 팩터 (PER, PBR 등)
        # ===========================================================
        n_stocks = len(self.tickers)
        
        if self.fundamental_data:
            # 실제 펀더멘털 데이터 사용 (PER, PBR)
            fundamental_for_date = self.fundamental_collector.get_batch_fundamental_for_date(
                self.tickers, 
                date, 
                self.fundamental_data
            )
            
            per_ratios = np.array([fundamental_for_date[ticker]['per'] for ticker in self.tickers])
            pbr_ratios = np.array([fundamental_for_date[ticker]['pbr'] for ticker in self.tickers])
            
            # NaN 처리
            per_ratios = np.nan_to_num(per_ratios, nan=15.0)
            pbr_ratios = np.nan_to_num(pbr_ratios, nan=1.5)
            
            # 이상치 처리
            per_ratios = np.clip(per_ratios, 0.1, 100)
            pbr_ratios = np.clip(pbr_ratios, 0.1, 10)
        else:
            # 가상 데이터 사용 (fallback)
            per_ratios = np.random.rand(n_stocks) * 20 + 5  # 5~25
            pbr_ratios = np.random.rand(n_stocks) * 3 + 0.5  # 0.5~3.5
        
        # ROE, 부채비율: OpenDart 재무제표 사용 (실제 데이터!)
        if self.financial_ratios:
            roe_list = []
            debt_ratio_list = []
            
            for ticker in self.tickers:
                ratios = self.opendart_collector.get_financial_ratio_for_date(
                    ticker, date, self.financial_ratios
                )
                
                if ratios:
                    roe_list.append(ratios['roe'])
                    debt_ratio_list.append(ratios['debt_ratio'])
                else:
                    # 데이터 없으면 fallback
                    roe_list.append(np.nan)
                    debt_ratio_list.append(np.nan)
            
            roe = np.array(roe_list)
            debt_ratio = np.array(debt_ratio_list)
            
            # NaN 처리 (OpenDart 데이터 없는 종목)
            roe = np.nan_to_num(roe, nan=0.10)
            debt_ratio = np.nan_to_num(debt_ratio, nan=0.80)
            
            # 이상치 처리
            roe = np.clip(roe, -0.5, 2.0)  # -50% ~ 200%
            debt_ratio = np.clip(debt_ratio, 0.0, 5.0)  # 0% ~ 500%
        else:
            # OpenDart 없으면 추정값 사용
            debt_ratio = self.financial_estimator.get_batch_debt_ratios(self.tickers)
            
            # ROE: pykrx의 EPS/BPS 사용 (차선책)
            if self.fundamental_data:
                roe_values = np.array([fundamental_for_date[ticker].get('roe', 0.10) for ticker in self.tickers])
                roe = np.nan_to_num(roe_values, nan=0.10)
                roe = np.clip(roe, -0.5, 1.0)
            else:
                roe = np.ones(n_stocks) * 0.10  # 기본값 10%
        
        # ===========================================================
        # 4. 시장 지표 (실제 데이터 기반)
        # ===========================================================
        market_per = per_ratios.mean()
        
        # 시장 변동성 및 수익률
        market_volatility = market_returns.std() * np.sqrt(252)
        market_return = market_returns.mean()
        
        # 시장 심리: 변동성과 수익률 기반 계산
        market_sentiment = MarketSentimentEstimator.calculate_sentiment(
            market_volatility, 
            market_return
        )
        
        # VIX (변동성 지수): 실제 계산
        vix = volatilities.mean() * 100  # 간이 VIX
        
        market_beta = 1.0  # 시장 자체는 베타 1
        
        # 무위험 이자율: 실제 기준금리
        interest_rate = InterestRateProvider.get_interest_rate(date)
        
        # ===========================================================
        # 5. 리스크 지표 (VaR 등)
        # ===========================================================
        var_95 = []
        for ticker in self.tickers:
            stock_returns = self.price_data[ticker]['Close'].pct_change().iloc[start_idx:date_idx]
            returns_clean = stock_returns.dropna()
            if len(returns_clean) > 0:
                var = np.percentile(returns_clean, 5)  # 5% VaR
                var_95.append(abs(var) if not np.isnan(var) else 0.05)
            else:
                var_95.append(0.05)  # 기본값
        var_95 = np.array(var_95)
        
        # ===========================================================
        # 최종 반환
        # ===========================================================
        return {
            # 가격
            'prices': prices,
            
            # 밸류에이션 (Value Agent용 - 가상)
            'per': per_ratios,
            'pbr': pbr_ratios,
            
            # 품질 (Quality Agent용 - 가상)
            'roe': roe,
            'debt_ratio': debt_ratio,
            
            # 리스크 (Portfolio Agent용 - 실제)
            'volatility': volatilities,
            'beta': betas,
            'sharpe': sharpes,
            
            # 헷징 (Hedging Agent용 - 실제)
            'correlation': correlations,
            'max_drawdown': max_drawdowns,
            'var_95': var_95,
            
            # 시장 지표
            'market_per': market_per,
            'market_sentiment': market_sentiment,
            'vix': vix,
            'market_volatility': market_volatility,
            'market_return': market_return,
            'market_beta': market_beta,
            'interest_rate': interest_rate,
        }
    
    def get_date_range(self):
        """
        사용 가능한 날짜 범위 반환
        
        Returns:
            tuple: (start_date, end_date)
        """
        if self.common_dates:
            return self.common_dates[0], self.common_dates[-1]
        return None, None
    
    def set_backtest_period(self, start_date, end_date):
        """
        백테스트 기간 설정 (전체 데이터에서 슬라이싱)
        
        Args:
            start_date (str): 백테스트 시작일 (YYYY-MM-DD)
            end_date (str): 백테스트 종료일 (YYYY-MM-DD)
        """
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # all_dates에서 해당 기간만 필터링 (전체 데이터 범위에서)
        self.backtest_dates = [
            d for d in self.all_dates 
            if start_date <= d <= end_date
        ]
        
        if len(self.backtest_dates) == 0:
            raise ValueError(f"백테스트 기간 {start_date.date()} ~ {end_date.date()}에 거래일이 없습니다!")
        
        print(f"\n[백테스트 기간 설정]")
        print(f"  기간: {self.backtest_dates[0].date()} ~ {self.backtest_dates[-1].date()}")
        print(f"  거래일: {len(self.backtest_dates)}일")
    
    def get_backtest_dates(self):
        """
        현재 설정된 백테스트 기간의 날짜 리스트 반환
        
        Returns:
            list: 백테스트 날짜 리스트
        """
        return self.backtest_dates if self.backtest_dates else self.common_dates
    
    def get_ticker_names(self):
        """
        종목 코드 리스트 반환
        
        Returns:
            list: 종목 코드 리스트
        """
        return self.tickers if self.tickers else []

