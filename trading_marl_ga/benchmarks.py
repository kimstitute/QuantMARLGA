"""
벤치마크 전략들

GA-MARL 시스템과 비교할 기준 전략들
"""

import numpy as np
import pandas as pd
import FinanceDataReader as fdr
from environment.backtest_env import BacktestEnv
from utils.metrics import calculate_all_metrics, print_metrics


class BuyAndHold:
    """
    Buy & Hold 전략
    
    초기에 모든 종목을 동일 비중으로 매수하고 보유
    """
    def __init__(self, n_stocks):
        self.n_stocks = n_stocks
        self.name = "Buy & Hold"
    
    def act(self, obs):
        """동일 비중 배분"""
        equal_weight = 1.0 / self.n_stocks
        return {
            'value_scores': np.ones(self.n_stocks) * 0.5,
            'quality_scores': np.ones(self.n_stocks) * 0.5,
            'portfolio_weights': np.ones(self.n_stocks) * equal_weight,
            'hedge_ratios': np.zeros(self.n_stocks),  # 헷징 없음
            'final_weights': np.ones(self.n_stocks) * equal_weight
        }


def run_kospi_index_benchmark(start_date, end_date):
    """
    실제 KOSPI 지수 수익률 계산
    
    Args:
        start_date (str): 시작일 (YYYY-MM-DD)
        end_date (str): 종료일 (YYYY-MM-DD)
    
    Returns:
        dict: 성과 지표
    """
    # KOSPI 지수 데이터 가져오기
    kospi = fdr.DataReader('KS11', start_date, end_date)
    
    if kospi.empty or len(kospi) < 2:
        print(f"[ERROR] KOSPI 지수 데이터 로드 실패")
        return None
    
    # 종가 기준 포트폴리오 가치 계산 (초기 10,000,000원)
    initial_capital = 10_000_000
    kospi_prices = kospi['Close'].values
    portfolio_values = (kospi_prices / kospi_prices[0]) * initial_capital
    
    # 성과 지표 계산
    metrics = calculate_all_metrics(portfolio_values.tolist())
    
    print(f"[OK] KOSPI 지수: {len(kospi)}일, " +
          f"{kospi.index[0].date()} ~ {kospi.index[-1].date()}")
    
    return metrics


class RandomAgent:
    """
    Random Agent
    
    매일 랜덤하게 포트폴리오 구성
    """
    def __init__(self, n_stocks):
        self.n_stocks = n_stocks
        self.name = "Random Agent"
    
    def act(self, obs):
        """랜덤 비중 배분"""
        weights = np.random.dirichlet(np.ones(self.n_stocks))
        hedge = np.random.rand(self.n_stocks) * 0.3  # 0-30% 헷징
        final_weights = weights * (1 - hedge)
        
        return {
            'value_scores': np.random.rand(self.n_stocks),
            'quality_scores': np.random.rand(self.n_stocks),
            'portfolio_weights': weights,
            'hedge_ratios': hedge,
            'final_weights': final_weights
        }




def run_benchmark(strategy, env, verbose=False):
    """
    벤치마크 전략 실행
    
    Args:
        strategy: 전략 객체 (act 메서드 필요)
        env (BacktestEnv): 백테스트 환경
        verbose (bool): 상세 출력
    
    Returns:
        dict: 성과 지표
    """
    obs = env.reset()
    done = False
    step_count = 0
    
    while not done:
        # 전략 행동
        actions = strategy.act(obs)
        
        # 환경 진행
        next_obs, rewards, done, info = env.step(actions)
        step_count += 1
        
        if verbose and step_count % 20 == 0:
            print(f"  Day {step_count}: 자산 {info['portfolio_value']:,.0f}원")
        
        if not done:
            obs = next_obs
    
    # 성과 지표 계산
    metrics = env.get_performance_metrics()
    
    return metrics


def compare_strategies(strategies, n_days=200, verbose=True):
    """
    여러 전략 비교
    
    Args:
        strategies (list): 전략 객체 리스트
        n_days (int): 백테스트 기간
        verbose (bool): 상세 출력
    
    Returns:
        dict: 각 전략의 성과 지표
    """
    results = {}
    
    for strategy in strategies:
        if verbose:
            print(f"\n{'='*60}")
            print(f"[전략] {strategy.name}")
            print(f"{'='*60}")
        
        # 백테스트 환경 (각 전략마다 새로 생성)
        env = BacktestEnv(n_days=n_days)
        
        # 벤치마크 실행
        metrics = run_benchmark(strategy, env, verbose=verbose)
        
        if verbose:
            print(f"\n[성과 지표]")
            print(f"  총 수익률:    {metrics['total_return']*100:>8.2f}%")
            print(f"  샤프 비율:    {metrics['sharpe_ratio']:>8.3f}")
            print(f"  최대 낙폭:    {metrics['max_drawdown']*100:>8.2f}%")
            print(f"  승률:         {metrics['win_rate']*100:>8.2f}%")
            print(f"  변동성 (연):  {metrics['volatility']*100:>8.2f}%")
            print(f"  칼마 비율:    {metrics['calmar_ratio']:>8.3f}")
        
        results[strategy.name] = metrics
    
    return results


def print_comparison_table(results):
    """
    성과 비교 테이블 출력
    
    Args:
        results (dict): {전략명: 성과 지표}
    """
    print(f"\n{'='*80}")
    print(f"성과 비교표")
    print(f"{'='*80}")
    
    # 헤더
    print(f"{'전략':<25} {'수익률':>10} {'샤프':>8} {'MDD':>10} {'승률':>8} {'변동성':>10}")
    print(f"{'-'*80}")
    
    # 각 전략
    for name, metrics in results.items():
        print(f"{name:<25} "
              f"{metrics['total_return']*100:>9.2f}% "
              f"{metrics['sharpe_ratio']:>8.3f} "
              f"{metrics['max_drawdown']*100:>9.2f}% "
              f"{metrics['win_rate']*100:>7.2f}% "
              f"{metrics['volatility']*100:>9.2f}%")
    
    print(f"{'='*80}")
    
    # 최고 성능 강조
    best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
    best_return = max(results.items(), key=lambda x: x[1]['total_return'])
    
    print(f"\n[최고 성과]")
    print(f"  샤프 비율: {best_sharpe[0]} ({best_sharpe[1]['sharpe_ratio']:.3f})")
    print(f"  총 수익률: {best_return[0]} ({best_return[1]['total_return']*100:.2f}%)")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    """벤치마크 테스트"""
    from config import config
    
    print("="*60)
    print("벤치마크 전략 비교")
    print("="*60)
    
    # 포트폴리오 전략 리스트
    strategies = [
        BuyAndHold(config.N_STOCKS),
        RandomAgent(config.N_STOCKS),
    ]
    
    # 비교 실행
    results = compare_strategies(strategies, n_days=100, verbose=True)
    
    # KOSPI 지수 (실제 시장 벤치마크)
    print(f"\n{'='*60}")
    print(f"[전략] KOSPI Index")
    print(f"{'='*60}")
    
    kospi_metrics = run_kospi_index_benchmark(
        start_date=config.DATA_START_DATE,
        end_date=config.DATA_END_DATE
    )
    
    if kospi_metrics:
        print(f"\n[성과 지표]")
        print(f"  총 수익률:    {kospi_metrics['total_return']*100:>8.2f}%")
        print(f"  샤프 비율:    {kospi_metrics['sharpe_ratio']:>8.3f}")
        print(f"  최대 낙폭:    {kospi_metrics['max_drawdown']*100:>8.2f}%")
        print(f"  승률:         {kospi_metrics['win_rate']*100:>8.2f}%")
        print(f"  변동성 (연):  {kospi_metrics['volatility']*100:>8.2f}%")
        print(f"  칼마 비율:    {kospi_metrics['calmar_ratio']:>8.3f}")
        
        results['KOSPI Index'] = kospi_metrics
    
    # 비교 테이블
    print_comparison_table(results)

