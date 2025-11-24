"""
멀티 에이전트 트레이딩 시스템을 위한 관측 구성

시장 데이터로부터 4개 에이전트를 위한 계층적 관측을 생성합니다.
"""

import numpy as np
from config import config


def construct_observations(market_data, value_scores=None, quality_scores=None, portfolio_weights=None):
    """
    시장 데이터로부터 4개 에이전트의 관측 생성
    
    병렬 + 융합 구조:
    - Value obs: 독립적 (PER, PBR, market)
    - Quality obs: 독립적 (ROE, Debt, market) - Value와 병렬
    - Portfolio obs: 융합 (value_scores + quality_scores + risk + market)
    - Hedging obs: 방어 (portfolio_weights + market_risk)
    
    Args:
        market_data (dict): 시장 데이터:
            - 'prices': (n_stocks,) 현재 주가
            - 'per': (n_stocks,) PER (선택사항)
            - 'pbr': (n_stocks,) PBR (선택사항)
            - 'roe': (n_stocks,) ROE (선택사항)
            - 'debt_ratio': (n_stocks,) 부채비율 (선택사항)
            - ... 기타 시장 지표
        value_scores (np.ndarray, optional): Value Agent 출력 (n_stocks,)
        quality_scores (np.ndarray, optional): Quality Agent 출력 (n_stocks,)
        portfolio_weights (np.ndarray, optional): Portfolio Agent 출력 (n_stocks,)
    
    Returns:
        dict: 모든 에이전트의 관측 (N_STOCKS=30 기준):
            - 'value_obs': (63,) = N*2+3 = 30*2+3
            - 'quality_obs': (63,) = N*2+3 = 30*2+3
            - 'portfolio_obs': (156,) = N*2+(N*3+2)+4 = 30*2+92+4
            - 'hedging_obs': (122,) = N+(N*3+2) = 30+92
    """
    n_stocks = config.N_STOCKS
    
    # 시장 데이터 추출 또는 생성
    prices = market_data.get('prices', np.random.rand(n_stocks) * 100 + 10)
    
    # ===========================================================
    # 1. Value 관측 (23 차원) - 독립적
    # ===========================================================
    # 밸류에이션 전문: PER, PBR
    
    # PER 비율 (n_stocks개)
    per_ratios = market_data.get('per', np.random.rand(n_stocks) * 20 + 5)
    # [0, 1]로 정규화: PER이 낮을수록 좋음
    per_normalized = 1.0 / (1.0 + per_ratios / 20.0)
    
    # PBR 비율 (n_stocks개)
    pbr_ratios = market_data.get('pbr', np.random.rand(n_stocks) * 3 + 0.5)
    # [0, 1]로 정규화: PBR이 낮을수록 좋음
    pbr_normalized = 1.0 / (1.0 + pbr_ratios / 2.0)
    
    # 시장 지표 (3개 특성)
    market_per = market_data.get('market_per', np.random.rand() * 20 + 10)
    market_sentiment = market_data.get('market_sentiment', np.random.rand())
    vix = market_data.get('vix', np.random.rand() * 30 + 10)
    
    # 스칼라를 배열로 변환
    market_indicators = np.array([
        market_per,
        market_sentiment,
        vix
    ])
    
    value_obs = np.concatenate([
        per_normalized,       # n_stocks
        pbr_normalized,       # n_stocks
        market_indicators     # 3
    ])  # Total: n_stocks*2+3
    
    # ===========================================================
    # 2. Quality 관측 (23 차원) - 독립적 (Value와 병렬)
    # ===========================================================
    # 품질 전문: ROE, 부채비율
    
    # ROE (자기자본이익률, n_stocks개)
    roe = market_data.get('roe', np.random.rand(n_stocks) * 0.3)
    # [0, 1]로 정규화: ROE가 높을수록 좋음
    roe_normalized = roe / 0.3
    
    # 부채비율 (n_stocks개)
    debt_ratio = market_data.get('debt_ratio', np.random.rand(n_stocks) * 1.5)
    # [0, 1]로 정규화: 부채비율이 낮을수록 좋음
    debt_normalized = 1.0 / (1.0 + debt_ratio)
    
    # 동일한 시장 지표 사용 (3개)
    # (Value와 같은 시장 환경 인식)
    
    quality_obs = np.concatenate([
        roe_normalized,       # n_stocks
        debt_normalized,      # n_stocks
        market_indicators     # 3
    ])  # Total: n_stocks*2+3
    
    # ===========================================================
    # 3. Portfolio 관측 (56 차원) - 융합
    # ===========================================================
    # Value & Quality 출력을 입력으로 받음
    
    # Value & Quality Agent 출력 (각 n_stocks개)
    if value_scores is None:
        value_scores = np.random.rand(n_stocks)  # placeholder
    if quality_scores is None:
        quality_scores = np.random.rand(n_stocks)  # placeholder
    
    # 리스크 지표 (n_stocks*3+2개)
    volatility = market_data.get('volatility', np.random.rand(n_stocks) * 0.3)
    beta = market_data.get('beta', np.random.rand(n_stocks) * 2)
    sharpe = market_data.get('sharpe', np.random.randn(n_stocks))
    
    risk_metrics = np.concatenate([
        volatility,           # n_stocks
        beta,                 # n_stocks
        sharpe,               # n_stocks
        np.random.rand(2)     # 2
    ])  # Total: n_stocks*3+2
    
    # 시장 상태 (4개)
    market_vol = market_data.get('market_volatility', np.random.rand() * 0.3)
    market_beta = market_data.get('market_beta', 1.0)
    market_return = market_data.get('market_return', np.random.randn() * 0.01)
    interest_rate = market_data.get('interest_rate', np.random.rand() * 0.05)
    
    # 스칼라를 배열로 변환
    market_state = np.array([
        market_vol,
        market_beta,
        market_return,
        interest_rate
    ])  # Total: 4
    
    portfolio_obs = np.concatenate([
        value_scores,         # n_stocks
        quality_scores,       # n_stocks
        risk_metrics,         # n_stocks*3+2
        market_state          # 4
    ])  # Total: n_stocks*2+(n_stocks*3+2)+4
    
    # ===========================================================
    # 4. Hedging 관측 (42 차원) - 방어
    # ===========================================================
    # Portfolio Agent 출력 + 시장 리스크
    
    # Portfolio weights (n_stocks개)
    if portfolio_weights is None:
        portfolio_weights = np.random.rand(n_stocks)  # placeholder
        portfolio_weights = portfolio_weights / portfolio_weights.sum()
    
    # 시장 리스크 지표 (n_stocks*3+2개)
    correlation_with_market = market_data.get('correlation', np.random.rand(n_stocks))
    max_drawdown = market_data.get('max_drawdown', np.random.rand(n_stocks) * 0.5)
    var_95 = market_data.get('var_95', np.random.rand(n_stocks) * 0.1)
    
    market_risk = np.concatenate([
        correlation_with_market,  # n_stocks
        max_drawdown,             # n_stocks
        var_95,                   # n_stocks
        np.random.rand(2)         # 2
    ])  # Total: n_stocks*3+2
    
    hedging_obs = np.concatenate([
        portfolio_weights,    # n_stocks
        market_risk           # n_stocks*3+2
    ])  # Total: n_stocks+(n_stocks*3+2)
    
    return {
        'value_obs': value_obs.astype(np.float32),
        'quality_obs': quality_obs.astype(np.float32),
        'portfolio_obs': portfolio_obs.astype(np.float32),
        'hedging_obs': hedging_obs.astype(np.float32)
    }

