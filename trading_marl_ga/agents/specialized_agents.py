"""
특화된 에이전트들

4개의 전문화된 트레이딩 에이전트:
- ValueAgent: 가치 평가
- QualityAgent: 품질 평가
- PortfolioAgent: 포트폴리오 구성
- HedgingAgent: 리스크 헷징
"""

from agents.base_agent import BaseAgent
from config import config


class ValueAgent(BaseAgent):
    """
    가치 평가 에이전트
    저평가된 주식을 식별
    """
    def __init__(self):
        super().__init__(
            obs_dim=config.VALUE_OBS_DIM,
            action_dim=config.ACTION_DIM,
            name="value"
        )


class QualityAgent(BaseAgent):
    """
    품질 평가 에이전트
    재무 품질로 필터링
    """
    def __init__(self):
        super().__init__(
            obs_dim=config.QUALITY_OBS_DIM,
            action_dim=config.ACTION_DIM,
            name="quality"
        )


class PortfolioAgent(BaseAgent):
    """
    포트폴리오 구성 에이전트
    포지션 크기 결정
    """
    def __init__(self):
        super().__init__(
            obs_dim=config.PORTFOLIO_OBS_DIM,
            action_dim=config.ACTION_DIM,
            name="portfolio"
        )


class HedgingAgent(BaseAgent):
    """
    헷징 에이전트
    리스크 노출 관리
    """
    def __init__(self):
        super().__init__(
            obs_dim=config.HEDGING_OBS_DIM,
            action_dim=config.ACTION_DIM,
            name="hedging"
        )

