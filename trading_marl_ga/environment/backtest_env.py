"""
백테스트 환경

실제 또는 합성 주가 데이터로 트레이딩 전략 백테스트
"""

import numpy as np
from utils.observation import construct_observations
from environment.reward_calculator_independent import IndependentRewardCalculator
from utils.metrics import calculate_all_metrics, print_metrics
from config import config

# 실제 데이터 사용 시
if config.DATA_SOURCE == "real":
    from data.market_data_manager import MarketDataManager


def priority_integer_allocation(target_weights, prices, available_cash):
    """
    우선순위 기반 정수 주식 배분 (현실적 제약)
    
    1. 모든 종목 내림으로 시작
    2. 남은 현금으로 목표 비중에 가깝도록 1주씩 추가
    3. 현금을 최대한 활용하면서 목표 비중 유지
    
    Args:
        target_weights (np.ndarray): 목표 비중 (n_stocks,), 합=1
        prices (np.ndarray): 주가 (n_stocks,)
        available_cash (float): 사용 가능한 현금
    
    Returns:
        tuple: (shares, remaining_cash)
            - shares (np.ndarray): 정수 주식 수
            - remaining_cash (float): 남은 현금
    """
    n = len(target_weights)
    
    # 1. 목표 금액 계산
    target_values = target_weights * available_cash
    
    # 2. 초기 배분 (내림)
    shares = np.floor(target_values / (prices + 1e-8))
    used_cash = (shares * prices).sum()
    remaining_cash = available_cash - used_cash
    
    # 3. 남은 현금으로 추가 매수 (목표 비중에 가깝도록)
    max_iterations = 1000  # 무한 루프 방지
    for _ in range(max_iterations):
        # 살 수 있는 종목 체크
        can_buy = remaining_cash >= prices
        if not np.any(can_buy):
            break  # 더 이상 살 수 없음
        
        # 현재 비중 계산
        current_values = shares * prices
        total_value = current_values.sum()
        if total_value < 1e-8:
            # 아직 아무것도 안 샀으면 가장 비중 높은 것부터
            best_idx = np.argmax(target_weights)
            if can_buy[best_idx]:
                shares[best_idx] += 1
                remaining_cash -= prices[best_idx]
            else:
                break
            continue
        
        current_weights = current_values / total_value
        
        # 각 종목에 1주 추가 시 비중 오차 개선도 계산
        improvements = np.full(n, -np.inf)
        for i in range(n):
            if can_buy[i]:
                # 1주 추가 시 새 비중
                new_shares = shares.copy()
                new_shares[i] += 1
                new_values = new_shares * prices
                new_total = new_values.sum()
                new_weights = new_values / new_total
                
                # 전체 비중 오차 (L2 norm)
                old_error = np.sum((current_weights - target_weights) ** 2)
                new_error = np.sum((new_weights - target_weights) ** 2)
                improvements[i] = old_error - new_error
        
        # 가장 개선도 높은 종목 매수
        best_idx = np.argmax(improvements)
        if improvements[best_idx] <= 0:
            break  # 더 이상 개선 안 됨
        
        shares[best_idx] += 1
        remaining_cash -= prices[best_idx]
    
    return shares.astype(int), remaining_cash


class BacktestEnv:
    """
    백테스트 환경
    
    기능:
    1. 실제 or 합성 주가 데이터 제공
    2. 포트폴리오 시뮬레이션
    3. 성과 지표 계산 (Sharpe, Max DD, Total Return)
    
    데이터 소스:
    - config.DATA_SOURCE = "real": MarketDataManager로 실제 데이터
    - config.DATA_SOURCE = "synthetic": Random Walk로 합성 데이터
    """
    
    def __init__(self, n_days=50):
        """
        Args:
            n_days (int): 백테스트 기간 (거래일 수)
        """
        self.n_days = n_days
        self.n_stocks = config.N_STOCKS
        self.initial_capital = config.INITIAL_CAPITAL
        
        print(f"\n{'='*60}")
        print(f"BacktestEnv 초기화")
        print(f"{'='*60}")
        
        # ===========================================================
        # 데이터 소스 초기화
        # ===========================================================
        if config.DATA_SOURCE == "real":
            print("[OK] 데이터 소스: 실제 시장 데이터")
            self.data_manager = MarketDataManager(cache_dir=config.CACHE_DIR)
            self.data_manager.initialize(
                start_date=config.DATA_START_DATE,
                end_date=config.DATA_END_DATE,
                n_stocks=config.N_STOCKS
            )
            self.trading_days = self.data_manager.common_dates
            self.n_days = min(n_days, len(self.trading_days))
            print(f"[OK] 백테스트 기간: {self.n_days}일 (최대: {len(self.trading_days)}일)")
        else:
            print("[OK] 데이터 소스: 합성 데이터 (Random Walk)")
            self.data_manager = None
            self.trading_days = None
            self.price_history = self._generate_synthetic_prices()
            print(f"[OK] 백테스트 기간: {self.n_days}일")
        
        print(f"[OK] 종목 수: {self.n_stocks}")
        print(f"[OK] 초기 자본: {self.initial_capital:,.0f}원")
        
        # ===========================================================
        # 보상 계산기 초기화
        # ===========================================================
        self.reward_calculator = IndependentRewardCalculator()
        print(f"[OK] 보상 계산기: 독립 지표 기반")
        print(f"{'='*60}\n")
        
        self.reset()
    
    def _generate_synthetic_prices(self):
        """
        합성 주가 생성 (Random Walk) - 테스트용
        
        Returns:
            np.ndarray: (n_days, n_stocks) 가격 데이터
        """
        prices = np.zeros((self.n_days, self.n_stocks))
        prices[0] = np.random.rand(self.n_stocks) * 50000 + 10000  # 10,000~60,000
        
        for t in range(1, self.n_days):
            # 일일 수익률 (평균 0%, 표준편차 2%)
            returns = np.random.randn(self.n_stocks) * 0.02
            prices[t] = prices[t-1] * (1 + returns)
        
        return prices
    
    def reset(self):
        """
        환경 리셋
        
        Returns:
            dict: 초기 관측
        """
        self.current_day = 0
        self.cash = self.initial_capital
        self.positions = np.zeros(self.n_stocks)  # 주식 보유 수량
        self.portfolio_value = self.initial_capital
        self.portfolio_history = [self.initial_capital]
        self.last_rebalance_day = 0  # 마지막 리밸런싱 날짜
        
        return self._get_observation()
    
    def _get_observation(self):
        """
        현재 시점의 관측 생성
        
        Returns:
            dict: 4개 에이전트의 관측
                - 'value_obs': (23,)
                - 'quality_obs': (23,)
                - 'portfolio_obs': (56,)
                - 'hedging_obs': (42,)
        """
        if config.DATA_SOURCE == "real":
            # 실제 시장 데이터
            current_date = self.trading_days[self.current_day]
            market_data = self.data_manager.get_market_data_for_date(
                current_date,
                lookback=config.LOOKBACK_DAYS
            )
        else:
            # 합성 데이터 (최소한의 정보만)
            market_data = {
                'prices': self.price_history[self.current_day]
            }
        
        return construct_observations(market_data)
    
    def step(self, actions):
        """
        1일 거래 실행
        
        Args:
            actions (dict): MultiAgentSystem.act() 출력
                - 'value_scores': (n_stocks,)
                - 'quality_scores': (n_stocks,)
                - 'portfolio_weights': (n_stocks,)
                - 'hedge_ratios': (n_stocks,)
                - 'final_weights': (n_stocks,) ← 사용
        
        Returns:
            tuple: (next_obs, reward, done, info)
                - next_obs (dict): 다음 관측 (or None if done)
                - reward (float): 일 수익률
                - done (bool): 에피소드 종료 여부
                - info (dict): 추가 정보
        """
        # 현재 가격 및 시장 데이터
        if config.DATA_SOURCE == "real":
            current_date = self.trading_days[self.current_day]
            current_market_data = self.data_manager.get_market_data_for_date(current_date)
            current_prices = current_market_data['prices']
        else:
            current_market_data = None
            current_prices = self.price_history[self.current_day]
        
        # 최종 비중으로 거래
        final_weights = actions['final_weights']
        
        # =============================================================
        # 리밸런싱 주기 체크
        # =============================================================
        should_rebalance = (self.current_day - self.last_rebalance_day) >= config.REBALANCE_PERIOD
        
        if should_rebalance:
            # 리밸런싱 실행
            self.last_rebalance_day = self.current_day
            
            # =============================================================
            # 정수 주식 제약 + 현금 부족 방지 (현실적 제약)
            # =============================================================
            # 1. 현재 보유 주식 매도로 확보 가능한 현금
            sell_positions = np.maximum(0, self.positions)  # 매도 가능한 주식
            sell_value = (sell_positions * current_prices).sum()
            total_cash = self.cash + sell_value  # 총 사용 가능 금액
            
            # 2. 수수료 여유분 확보 (최대 거래액의 0.3% 여유)
            # 예상 최대 수수료를 고려하여 safe margin 확보
            safety_margin = total_cash * (config.TRANSACTION_COST * 1.5)
            available_cash = total_cash - safety_margin
            
            # 3. 정수 제약 적용 (우선순위 기반)
            target_positions, remaining_cash = priority_integer_allocation(
                final_weights,
                current_prices,
                available_cash
            )
            
            # 4. 실제 남은 현금 (safety_margin 복원)
            remaining_cash += safety_margin
            
            # 5. 거래 실행
            trades = target_positions - self.positions  # 거래량 (양수=매수, 음수=매도)
            trade_values = trades * current_prices  # 거래 금액
            trade_costs = np.abs(trade_values).sum() * config.TRANSACTION_COST  # 수수료
            
            # 포지션 및 현금 업데이트
            self.positions = target_positions
            
            # 현금 재계산 (정수 제약 적용 후)
            # remaining_cash는 이미 거래 후 남은 현금
            # 수수료만 추가로 차감
            self.cash = remaining_cash - trade_costs
            
            # 거래 직후 포트폴리오 가치 업데이트 (당일 가격 기준)
            stock_value_after_trade = (self.positions * current_prices).sum()
            self.portfolio_value = self.cash + stock_value_after_trade
            # portfolio_history는 다음 날 가격 기준으로 추가될 것임
        else:
            # 리밸런싱 주기가 아닌 경우: 포지션 유지, 가격 변동만 반영
            # 거래 비용 없음, 포지션 그대로
            trade_costs = 0.0  # 리밸런싱 없으면 거래 비용 없음
            stock_value = (self.positions * current_prices).sum()
            self.portfolio_value = self.cash + stock_value
        
        # 다음 날로 이동
        self.current_day += 1
        done = (self.current_day >= self.n_days - 1)
        
        if not done:
            # 다음 날 가격으로 포트폴리오 가치 재계산
            if config.DATA_SOURCE == "real":
                next_date = self.trading_days[self.current_day]
                next_market_data = self.data_manager.get_market_data_for_date(next_date)
                next_prices = next_market_data['prices']
            else:
                next_prices = self.price_history[self.current_day]
            
            stock_value = (self.positions * next_prices).sum()
            self.portfolio_value = self.cash + stock_value
            self.portfolio_history.append(self.portfolio_value)
            
            # ===========================================================
            # 보상 계산 (독립 지표 기반)
            # ===========================================================
            # 포트폴리오 수익률
            portfolio_return = (
                (self.portfolio_value - self.portfolio_history[-2]) / 
                (self.portfolio_history[-2] + 1e-8)
            )
            
            # 샤프 비율 (최근 N일 기준)
            recent_window = min(20, len(self.portfolio_history) - 1)
            recent_returns = np.diff(self.portfolio_history[-recent_window:]) / np.array(self.portfolio_history[-recent_window:-1])
            if len(recent_returns) > 1 and recent_returns.std() > 0:
                sharpe_ratio = recent_returns.mean() / recent_returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            # 최대 낙폭 (최근 N일 기준)
            recent_values = np.array(self.portfolio_history[-recent_window:])
            running_max = np.maximum.accumulate(recent_values)
            drawdowns = (recent_values - running_max) / (running_max + 1e-8)
            max_drawdown = drawdowns.min()
            
            # 개별 종목 수익률
            stock_returns = (next_prices - current_prices) / (current_prices + 1e-8)
            
            # 에이전트별 보상 계산
            rewards = self.reward_calculator.calculate_rewards(
                actions=actions,
                portfolio_return=portfolio_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                stock_returns=stock_returns,
                market_data=current_market_data if config.DATA_SOURCE == "real" else {
                    'volatility': np.ones(self.n_stocks) * 0.2,
                    'beta': np.ones(self.n_stocks),
                }
            )
        else:
            # 에피소드 종료 - 변수 초기화
            portfolio_return = 0.0
            sharpe_ratio = 0.0
            max_drawdown = 0.0
            rewards = {
                'value_reward': 0.0,
                'quality_reward': 0.0,
                'portfolio_reward': 0.0,
                'hedging_reward': 0.0
            }
        
        # 다음 관측
        next_obs = self._get_observation() if not done else None
        
        # 추가 정보
        info = {
            'portfolio_value': self.portfolio_value,
            'day': self.current_day,
            'trade_costs': trade_costs,
            'portfolio_return': portfolio_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
        }
        
        return next_obs, rewards, done, info
    
    def get_performance_metrics(self, risk_free_rate=0.0):
        """
        백테스트 성과 지표 계산 (metrics.py 사용)
        
        Args:
            risk_free_rate (float): 무위험 이자율 (연간, 기본값 0%)
        
        Returns:
            dict: 모든 성과 지표
                - 'total_return': 총 수익률
                - 'sharpe_ratio': 샤프 비율
                - 'max_drawdown': 최대 낙폭
                - 'win_rate': 승률
                - 'volatility': 변동성 (연환산)
                - 'calmar_ratio': 칼마 비율
                - 'sortino_ratio': 소르티노 비율
                - 'final_value': 최종 자산
                - 'n_periods': 거래일 수
        """
        portfolio_values = np.array(self.portfolio_history)
        
        # metrics.py의 함수 사용
        metrics = calculate_all_metrics(
            portfolio_values=portfolio_values,
            risk_free_rate=risk_free_rate,
            periods_per_year=252
        )
        
        return metrics
    
    def render(self):
        """성과 출력 (metrics.py의 print_metrics 사용)"""
        metrics = self.get_performance_metrics()
        print(f"\n백테스트 진행: Day {self.current_day}/{self.n_days}")
        print_metrics(metrics)

