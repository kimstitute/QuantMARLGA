"""
성과 지표 계산 함수들

트레이딩 전략의 성능을 평가하는 다양한 메트릭
"""

import numpy as np


def calculate_returns(portfolio_values):
    """
    포트폴리오 가치에서 수익률 계산
    
    Args:
        portfolio_values (np.array): 포트폴리오 가치 시계열 (T,)
    
    Returns:
        np.array: 일별 수익률 (T-1,)
    """
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    return returns


def calculate_total_return(portfolio_values):
    """
    총 수익률 계산
    
    Args:
        portfolio_values (np.array): 포트폴리오 가치 시계열 (T,)
    
    Returns:
        float: 총 수익률 (예: 0.15 = 15%)
    """
    if len(portfolio_values) < 2:
        return 0.0
    
    total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    return total_return


def calculate_sharpe_ratio(portfolio_values, risk_free_rate=0.0, periods_per_year=252):
    """
    샤프 비율 계산
    
    Args:
        portfolio_values (np.array): 포트폴리오 가치 시계열 (T,)
        risk_free_rate (float): 무위험 이자율 (연간)
        periods_per_year (int): 연간 거래일 수
    
    Returns:
        float: 샤프 비율 (연환산)
    """
    if len(portfolio_values) < 2:
        return 0.0
    
    returns = calculate_returns(portfolio_values)
    
    if len(returns) == 0:
        return 0.0
    
    # 일별 무위험 수익률
    daily_rf = risk_free_rate / periods_per_year
    
    # 초과 수익률
    excess_returns = returns - daily_rf
    
    # 평균 초과 수익률
    mean_excess = np.mean(excess_returns)
    
    # 변동성
    std_returns = np.std(excess_returns, ddof=1)
    
    if std_returns == 0 or np.isnan(std_returns):
        return 0.0
    
    # 샤프 비율 (연환산)
    sharpe = (mean_excess / std_returns) * np.sqrt(periods_per_year)
    
    return sharpe


def calculate_max_drawdown(portfolio_values):
    """
    최대 낙폭 (Maximum Drawdown) 계산
    
    Args:
        portfolio_values (np.array): 포트폴리오 가치 시계열 (T,)
    
    Returns:
        float: 최대 낙폭 (예: -0.15 = -15%)
    """
    if len(portfolio_values) < 2:
        return 0.0
    
    # 누적 최고점
    cummax = np.maximum.accumulate(portfolio_values)
    
    # 낙폭 계산
    drawdowns = (portfolio_values - cummax) / cummax
    
    # 최대 낙폭
    max_dd = np.min(drawdowns)
    
    return max_dd


def calculate_win_rate(portfolio_values):
    """
    승률 계산 (수익이 난 날의 비율)
    
    Args:
        portfolio_values (np.array): 포트폴리오 가치 시계열 (T,)
    
    Returns:
        float: 승률 (예: 0.55 = 55%)
    """
    if len(portfolio_values) < 2:
        return 0.0
    
    returns = calculate_returns(portfolio_values)
    
    if len(returns) == 0:
        return 0.0
    
    win_rate = np.sum(returns > 0) / len(returns)
    
    return win_rate


def calculate_volatility(portfolio_values, periods_per_year=252):
    """
    변동성 계산 (연환산 표준편차)
    
    Args:
        portfolio_values (np.array): 포트폴리오 가치 시계열 (T,)
        periods_per_year (int): 연간 거래일 수
    
    Returns:
        float: 변동성 (연환산)
    """
    if len(portfolio_values) < 2:
        return 0.0
    
    returns = calculate_returns(portfolio_values)
    
    if len(returns) == 0:
        return 0.0
    
    volatility = np.std(returns, ddof=1) * np.sqrt(periods_per_year)
    
    return volatility


def calculate_calmar_ratio(portfolio_values, periods_per_year=252):
    """
    칼마 비율 (Calmar Ratio) 계산
    = 연환산 수익률 / abs(최대 낙폭)
    
    Args:
        portfolio_values (np.array): 포트폴리오 가치 시계열 (T,)
        periods_per_year (int): 연간 거래일 수
    
    Returns:
        float: 칼마 비율
    """
    if len(portfolio_values) < 2:
        return 0.0
    
    total_return = calculate_total_return(portfolio_values)
    n_periods = len(portfolio_values) - 1
    
    # 연환산 수익률
    annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
    
    # 최대 낙폭
    max_dd = calculate_max_drawdown(portfolio_values)
    
    if max_dd == 0 or np.isnan(max_dd):
        return 0.0
    
    calmar = annualized_return / abs(max_dd)
    
    return calmar


def calculate_sortino_ratio(portfolio_values, risk_free_rate=0.0, periods_per_year=252):
    """
    소르티노 비율 (Sortino Ratio) 계산
    = (평균 수익률 - 무위험 수익률) / 하방 변동성
    
    Args:
        portfolio_values (np.array): 포트폴리오 가치 시계열 (T,)
        risk_free_rate (float): 무위험 이자율 (연간)
        periods_per_year (int): 연간 거래일 수
    
    Returns:
        float: 소르티노 비율 (연환산)
    """
    if len(portfolio_values) < 2:
        return 0.0
    
    returns = calculate_returns(portfolio_values)
    
    if len(returns) == 0:
        return 0.0
    
    # 일별 무위험 수익률
    daily_rf = risk_free_rate / periods_per_year
    
    # 초과 수익률
    excess_returns = returns - daily_rf
    
    # 평균 초과 수익률
    mean_excess = np.mean(excess_returns)
    
    # 하방 변동성 (손실만 고려)
    negative_returns = excess_returns[excess_returns < 0]
    
    if len(negative_returns) == 0:
        return np.inf if mean_excess > 0 else 0.0
    
    downside_std = np.std(negative_returns, ddof=1)
    
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    
    # 소르티노 비율 (연환산)
    sortino = (mean_excess / downside_std) * np.sqrt(periods_per_year)
    
    return sortino


def calculate_all_metrics(portfolio_values, risk_free_rate=0.0, periods_per_year=252):
    """
    모든 성과 지표를 한 번에 계산
    
    Args:
        portfolio_values (np.array): 포트폴리오 가치 시계열 (T,)
        risk_free_rate (float): 무위험 이자율 (연간)
        periods_per_year (int): 연간 거래일 수
    
    Returns:
        dict: 모든 성과 지표
    """
    metrics = {
        'total_return': calculate_total_return(portfolio_values),
        'sharpe_ratio': calculate_sharpe_ratio(portfolio_values, risk_free_rate, periods_per_year),
        'max_drawdown': calculate_max_drawdown(portfolio_values),
        'win_rate': calculate_win_rate(portfolio_values),
        'volatility': calculate_volatility(portfolio_values, periods_per_year),
        'calmar_ratio': calculate_calmar_ratio(portfolio_values, periods_per_year),
        'sortino_ratio': calculate_sortino_ratio(portfolio_values, risk_free_rate, periods_per_year),
        'final_value': portfolio_values[-1] if len(portfolio_values) > 0 else 0.0,
        'n_periods': len(portfolio_values) - 1 if len(portfolio_values) > 0 else 0,
    }
    
    return metrics


def print_metrics(metrics):
    """
    성과 지표를 보기 좋게 출력
    
    Args:
        metrics (dict): calculate_all_metrics() 결과
    """
    print("\n" + "="*60)
    print("성과 지표")
    print("="*60)
    print(f"총 수익률:        {metrics['total_return']*100:>8.2f}%")
    print(f"최종 자산:        {metrics['final_value']:>8,.0f}원")
    print(f"샤프 비율:        {metrics['sharpe_ratio']:>8.3f}")
    print(f"최대 낙폭:        {metrics['max_drawdown']*100:>8.2f}%")
    print(f"승률:             {metrics['win_rate']*100:>8.2f}%")
    print(f"변동성 (연):      {metrics['volatility']*100:>8.2f}%")
    print(f"칼마 비율:        {metrics['calmar_ratio']:>8.3f}")
    print(f"소르티노 비율:    {metrics['sortino_ratio']:>8.3f}")
    print(f"거래일 수:        {metrics['n_periods']:>8d}일")
    print("="*60 + "\n")

