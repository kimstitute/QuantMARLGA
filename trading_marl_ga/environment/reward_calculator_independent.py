"""
독립 지표 기반 보상 계산기

각 에이전트의 역할에 맞는 독립적인 지표를 측정하여 보상 계산
병렬 구조에 최적화된 설계

RACE와의 차이:
- RACE: 팀 보상만 (f = Σ r)
- 우리: 에이전트별 독립 지표 + 팀 성과 혼합
"""

import numpy as np
import warnings
from scipy.stats import spearmanr, ConstantInputWarning

# ConstantInputWarning 무시 (초기 학습 시 정상)
warnings.filterwarnings('ignore', category=ConstantInputWarning)


class IndependentRewardCalculator:
    """
    독립 지표 기반 보상 계산기
    
    각 에이전트의 전문성에 맞는 독립 지표 계산:
    - Value Agent: 예측력 (순위 상관계수, Top-K 정확도)
    - Quality Agent: 안전성 (리스크 조정 수익, 하방 보호)
    - Portfolio Agent: 최적화 (포트폴리오 수익, 융합 품질)
    - Hedging Agent: 방어력 (보호 효과, 정밀 헷징)
    
    보상 구성: 50% 독립 지표 + 50% 팀 성과 (에이전트별 가중치 차등)
    """
    
    def __init__(self):
        """초기화"""
        # 에이전트별 팀 성과 가중치
        self.team_weights = {
            'value': {'return': 0.3, 'sharpe': 0.3, 'drawdown': 0.4},
            'quality': {'return': 0.2, 'sharpe': 0.3, 'drawdown': 0.5},
            'portfolio': {'return': 0.5, 'sharpe': 0.3, 'drawdown': 0.2},
            'hedging': {'return': 0.2, 'sharpe': 0.4, 'drawdown': 0.4},
        }
    
    def reset(self):
        """
        리셋 (상태 저장 없음, 호환성 위해 구현)
        """
        pass
    
    def calculate_rewards(self, actions, portfolio_return, sharpe_ratio, max_drawdown, 
                         stock_returns, market_data):
        """
        모든 에이전트의 보상 계산
        
        Args:
            actions (dict): {
                'value_scores': (n_stocks,),
                'quality_scores': (n_stocks,),
                'portfolio_weights': (n_stocks,),
                'hedge_ratios': (n_stocks,),
                'final_weights': (n_stocks,)
            }
            portfolio_return (float): 포트폴리오 수익률
            sharpe_ratio (float): 샤프 비율
            max_drawdown (float): 최대 낙폭 (음수)
            stock_returns (np.array): 개별 종목 수익률 (n_stocks,)
            market_data (dict): 시장 데이터 (volatility, beta 등)
        
        Returns:
            dict: {
                'value_reward': float,
                'quality_reward': float,
                'portfolio_reward': float,
                'hedging_reward': float
            }
        """
        # 팀 성과 정규화
        team_return_norm = np.tanh(portfolio_return * 10)  # -1~1
        team_sharpe_norm = np.tanh(sharpe_ratio * 0.5)  # -1~1
        team_drawdown_norm = np.tanh(-max_drawdown * 10)  # 낙폭 작을수록 좋음, -1~1
        
        # 각 에이전트 독립 지표 계산
        value_indiv = self._value_agent_metrics(
            actions['value_scores'], 
            stock_returns, 
            market_data
        )
        
        quality_indiv = self._quality_agent_metrics(
            actions['quality_scores'], 
            stock_returns, 
            market_data
        )
        
        portfolio_indiv = self._portfolio_agent_metrics(
            actions['value_scores'],
            actions['quality_scores'],
            actions['portfolio_weights'],
            stock_returns,
            market_data
        )
        
        hedging_indiv = self._hedging_agent_metrics(
            actions['portfolio_weights'],
            actions['hedge_ratios'],
            portfolio_return,
            max_drawdown,
            market_data
        )
        
        # 각 에이전트 팀 성과 (가중치 차등)
        value_team = (
            self.team_weights['value']['return'] * team_return_norm +
            self.team_weights['value']['sharpe'] * team_sharpe_norm +
            self.team_weights['value']['drawdown'] * team_drawdown_norm
        )
        
        quality_team = (
            self.team_weights['quality']['return'] * team_return_norm +
            self.team_weights['quality']['sharpe'] * team_sharpe_norm +
            self.team_weights['quality']['drawdown'] * team_drawdown_norm
        )
        
        portfolio_team = (
            self.team_weights['portfolio']['return'] * team_return_norm +
            self.team_weights['portfolio']['sharpe'] * team_sharpe_norm +
            self.team_weights['portfolio']['drawdown'] * team_drawdown_norm
        )
        
        hedging_team = (
            self.team_weights['hedging']['return'] * team_return_norm +
            self.team_weights['hedging']['sharpe'] * team_sharpe_norm +
            self.team_weights['hedging']['drawdown'] * team_drawdown_norm
        )
        
        # 최종 보상: 50% 독립 + 50% 팀
        return {
            'value_reward': 0.5 * value_indiv + 0.5 * value_team,
            'quality_reward': 0.5 * quality_indiv + 0.5 * quality_team,
            'portfolio_reward': 0.5 * portfolio_indiv + 0.5 * portfolio_team,
            'hedging_reward': 0.5 * hedging_indiv + 0.5 * hedging_team,
        }
    
    # ===========================================================
    # Value Agent: 예측력
    # ===========================================================
    def _value_agent_metrics(self, value_scores, stock_returns, market_data):
        """
        Value Agent 독립 지표
        
        1. 순위 상관계수: value_scores와 실제 수익률의 스피어만 상관계수
        2. Top-K 정확도: 높은 점수를 준 종목이 실제로 좋았나?
        3. 가중 수익률: value_scores를 비중으로 했을 때의 수익률
        
        Returns:
            float: -1~1 사이의 독립 지표 점수
        """
        # 1. 순위 상관계수 (Spearman)
        if len(value_scores) > 1 and not np.all(value_scores == value_scores[0]):
            try:
                rank_corr, _ = spearmanr(value_scores, stock_returns)
                if np.isnan(rank_corr):
                    rank_corr = 0.0
            except:
                rank_corr = 0.0
        else:
            rank_corr = 0.0
        
        # 2. Top-K 정확도 (상위 30% 종목이 실제로 수익 냈나?)
        k = max(1, int(len(value_scores) * 0.3))
        top_k_indices = np.argsort(value_scores)[-k:]
        top_k_returns = stock_returns[top_k_indices]
        top_k_accuracy = np.mean(top_k_returns > 0)  # 0~1
        top_k_accuracy_norm = 2 * top_k_accuracy - 1  # -1~1
        
        # 3. 가중 수익률 (value_scores를 소프트맥스 비중으로)
        weights = np.exp(value_scores) / (np.sum(np.exp(value_scores)) + 1e-8)
        weighted_return = np.sum(weights * stock_returns)
        weighted_return_norm = np.tanh(weighted_return * 10)  # -1~1
        
        # 종합 (균등 가중)
        return (rank_corr * 0.4 + top_k_accuracy_norm * 0.3 + weighted_return_norm * 0.3)
    
    # ===========================================================
    # Quality Agent: 안전성
    # ===========================================================
    def _quality_agent_metrics(self, quality_scores, stock_returns, market_data):
        """
        Quality Agent 독립 지표
        
        1. 리스크 조정 수익: 높은 quality_scores 종목이 변동성 대비 수익 좋은가?
        2. 안전성 상관관계: quality_scores와 (샤프 비율) 상관관계
        3. 하방 보호: 높은 quality_scores 종목이 손실 작은가?
        
        Returns:
            float: -1~1 사이의 독립 지표 점수
        """
        volatility = market_data.get('volatility', np.ones(len(quality_scores)) * 0.2)
        
        # 1. 리스크 조정 수익
        risk_adjusted_returns = stock_returns / (volatility + 1e-8)
        if len(quality_scores) > 1 and not np.all(quality_scores == quality_scores[0]):
            try:
                risk_adj_corr, _ = spearmanr(quality_scores, risk_adjusted_returns)
                if np.isnan(risk_adj_corr):
                    risk_adj_corr = 0.0
            except:
                risk_adj_corr = 0.0
        else:
            risk_adj_corr = 0.0
        
        # 2. 안전성 상관관계 (높은 quality = 낮은 변동성)
        if len(quality_scores) > 1 and not np.all(quality_scores == quality_scores[0]):
            try:
                safety_corr, _ = spearmanr(quality_scores, -volatility)  # 변동성 낮을수록 좋음
                if np.isnan(safety_corr):
                    safety_corr = 0.0
            except:
                safety_corr = 0.0
        else:
            safety_corr = 0.0
        
        # 3. 하방 보호 (높은 quality 종목의 손실률)
        k = max(1, int(len(quality_scores) * 0.3))
        top_k_indices = np.argsort(quality_scores)[-k:]
        top_k_downside = np.minimum(stock_returns[top_k_indices], 0)  # 손실만
        downside_protection = -np.mean(top_k_downside)  # 손실 작을수록 좋음 (0~무한대)
        downside_protection_norm = np.tanh(downside_protection * 10)  # 0~1
        
        # 종합
        return (risk_adj_corr * 0.4 + safety_corr * 0.3 + downside_protection_norm * 0.3)
    
    # ===========================================================
    # Portfolio Agent: 최적화
    # ===========================================================
    def _portfolio_agent_metrics(self, value_scores, quality_scores, 
                                  portfolio_weights, stock_returns, market_data):
        """
        Portfolio Agent 독립 지표
        
        1. 포트폴리오 수익: 실제 비중으로 계산한 수익률
        2. 융합 품질: Value와 Quality를 얼마나 잘 융합했나?
        3. 분산화: 비중이 적절히 분산되었나?
        
        Returns:
            float: -1~1 사이의 독립 지표 점수
        """
        # 1. 포트폴리오 수익
        portfolio_return = np.sum(portfolio_weights * stock_returns)
        portfolio_return_norm = np.tanh(portfolio_return * 10)  # -1~1
        
        # 2. 융합 품질: Value와 Quality 모두 높은 종목에 비중 할당?
        fusion_score = value_scores + quality_scores  # 융합 신호
        if len(fusion_score) > 1 and not np.all(fusion_score == fusion_score[0]):
            try:
                fusion_corr, _ = spearmanr(fusion_score, portfolio_weights)
                if np.isnan(fusion_corr):
                    fusion_corr = 0.0
            except:
                fusion_corr = 0.0
        else:
            fusion_corr = 0.0
        
        # 3. 분산화 (엔트로피 기반)
        weights_norm = portfolio_weights / (np.sum(portfolio_weights) + 1e-8)
        entropy = -np.sum(weights_norm * np.log(weights_norm + 1e-8))
        max_entropy = np.log(len(portfolio_weights))  # 균등 분산 시
        diversification = entropy / max_entropy  # 0~1
        diversification_norm = 2 * diversification - 1  # -1~1
        
        # 종합
        return (portfolio_return_norm * 0.5 + fusion_corr * 0.3 + diversification_norm * 0.2)
    
    # ===========================================================
    # Hedging Agent: 방어력
    # ===========================================================
    def _hedging_agent_metrics(self, portfolio_weights, hedge_ratios, 
                                portfolio_return, max_drawdown, market_data):
        """
        Hedging Agent 독립 지표
        
        1. 보호 효과: 헷징으로 인한 낙폭 감소
        2. 상황 적합성: 위험 높을 때 헷징 강화했나?
        3. 정밀 헷징: 위험한 종목만 선택적으로 헷징?
        
        Returns:
            float: -1~1 사이의 독립 지표 점수
        """
        volatility = market_data.get('volatility', np.ones(len(hedge_ratios)) * 0.2)
        
        # 1. 보호 효과 (낙폭 작을수록 좋음)
        protection_effect = np.tanh(-max_drawdown * 10)  # 0~1
        
        # 2. 상황 적합성 (변동성 높을 때 헷징 강화?)
        if len(hedge_ratios) > 1 and not np.all(hedge_ratios == hedge_ratios[0]):
            try:
                situation_corr, _ = spearmanr(volatility, hedge_ratios)
                if np.isnan(situation_corr):
                    situation_corr = 0.0
            except:
                situation_corr = 0.0
        else:
            situation_corr = 0.0
        
        # 3. 정밀 헷징 (위험한 종목에 집중?)
        # 위험도 = volatility + beta
        beta = market_data.get('beta', np.ones(len(hedge_ratios)))
        risk_score = volatility + beta
        if len(hedge_ratios) > 1 and not np.all(hedge_ratios == hedge_ratios[0]):
            try:
                precision_corr, _ = spearmanr(risk_score, hedge_ratios)
                if np.isnan(precision_corr):
                    precision_corr = 0.0
            except:
                precision_corr = 0.0
        else:
            precision_corr = 0.0
        
        # 기회비용 패널티 (과도한 헷징 방지)
        avg_hedge = np.mean(hedge_ratios)
        opportunity_cost = -np.abs(avg_hedge - 0.3) * 0.5  # 30% 근처가 최적, -0.35~0
        
        # 종합
        return (protection_effect * 0.4 + situation_corr * 0.2 + 
                precision_corr * 0.2 + opportunity_cost * 0.2)

