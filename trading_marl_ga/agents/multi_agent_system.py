"""
멀티 에이전트 트레이딩 시스템

4개의 전문화된 에이전트 조율:
1. Value Agent: 저평가된 주식 식별
2. Quality Agent: 품질 지표로 필터링
3. Portfolio Agent: 포지션 크기 결정
4. Hedging Agent: 리스크 노출 관리
"""

import copy
import numpy as np

from agents.specialized_agents import ValueAgent, QualityAgent, PortfolioAgent, HedgingAgent
from utils.replay_buffer import SharedReplayBuffer
from config import config


class MultiAgentSystem:
    """
    4개의 협력 에이전트로 구성된 완전한 트레이딩 시스템
    
    순차 실행: Value -> Quality -> Portfolio -> Hedging
    모든 에이전트는 효율적인 학습을 위해 하나의 리플레이 버퍼를 공유
    """
    
    def __init__(self, system_id=None):
        """
        Args:
            system_id (int, optional): GA population에서 이 시스템의 식별자
        """
        self.system_id = system_id
        
        # 4개 에이전트 초기화
        self.value_agent = ValueAgent()
        self.quality_agent = QualityAgent()
        self.portfolio_agent = PortfolioAgent()
        self.hedging_agent = HedgingAgent()
        
        # 공유 리플레이 버퍼
        self.replay_buffer = SharedReplayBuffer(config.BUFFER_CAPACITY)
        
        # Fitness (GA용)
        self.fitness = None
    
    def act(self, market_state):
        """
        최종 포트폴리오 비중 생성을 위한 4-에이전트 파이프라인 실행
        
        ✅ 병렬 + 융합 구조:
        1. Value & Quality: 병렬 실행 (독립적)
        2. Portfolio: Value & Quality 출력을 융합
        3. Hedging: Portfolio 출력으로 방어
        
        Args:
            market_state (dict): 시장 관측 데이터:
                - 'value_obs': (23,) - PER, PBR, market
                - 'quality_obs': (23,) - ROE, Debt, market
                - 'portfolio_obs': (56,) - value_scores, quality_scores, risk, market
                - 'hedging_obs': (42,) - portfolio_weights, market_risk
                OR
                원시 시장 데이터 (dict) - construct_observations 호출용
        
        Returns:
            dict: 트레이딩 결정:
                - 'value_scores': (n_stocks,) 밸류에이션 매력도 [0,1]
                - 'quality_scores': (n_stocks,) 품질 점수 [0,1]
                - 'portfolio_weights': (n_stocks,) 포트폴리오 배분 [sum=1]
                - 'hedge_ratios': (n_stocks,) 헷징 강도 [0,1]
                - 'final_weights': (n_stocks,) 헷징 후 최종 포지션
        """
        # ===========================================================
        # Phase 1: Value & Quality 병렬 실행 (독립적) ✅
        # ===========================================================
        
        # 1-1. Value Agent: 저평가된 주식 식별 (독립적)
        value_scores = self.value_agent.act(market_state['value_obs'])
        
        # 1-2. Quality Agent: 품질 평가 (독립적, Value와 병렬)
        quality_scores = self.quality_agent.act(market_state['quality_obs'])
        
        # ===========================================================
        # Phase 2: Portfolio 융합 (Value + Quality 출력 사용) ✅
        # ===========================================================
        
        # Portfolio obs 동적 재구성
        # market_state['portfolio_obs']의 앞부분에 value_scores, quality_scores 삽입
        from utils.observation import construct_observations
        
        # 원시 시장 데이터가 있으면 사용, 없으면 기존 obs 사용
        if 'prices' in market_state:
            # 원시 데이터로 관측 재구성
            obs_dict = construct_observations(
                market_state, 
                value_scores=value_scores,
                quality_scores=quality_scores
            )
            portfolio_obs = obs_dict['portfolio_obs']
        else:
            # 기존 obs 사용 (앞 20개를 value_scores, quality_scores로 교체)
            portfolio_obs = market_state['portfolio_obs'].copy()
            portfolio_obs[:config.N_STOCKS] = value_scores
            portfolio_obs[config.N_STOCKS:config.N_STOCKS*2] = quality_scores
        
        # 2. Portfolio Agent: 포지션 크기 결정 (융합)
        portfolio_weights = self.portfolio_agent.act(portfolio_obs)
        
        # 포트폴리오 비중을 합이 1이 되도록 정규화
        portfolio_weights = portfolio_weights / (portfolio_weights.sum() + 1e-8)
        
        # ===========================================================
        # Phase 3: Hedging 방어 (Portfolio 출력 사용) ✅
        # ===========================================================
        
        # Hedging obs 동적 재구성
        if 'prices' in market_state:
            obs_dict = construct_observations(
                market_state,
                value_scores=value_scores,
                quality_scores=quality_scores,
                portfolio_weights=portfolio_weights
            )
            hedging_obs = obs_dict['hedging_obs']
        else:
            hedging_obs = market_state['hedging_obs'].copy()
            hedging_obs[:config.N_STOCKS] = portfolio_weights
        
        # 3. Hedging Agent: 리스크 관리 (방어)
        hedge_ratios = self.hedging_agent.act(hedging_obs)
        
        # 헷징 후 최종 포지션 계산
        final_weights = portfolio_weights * (1 - hedge_ratios)
        
        return {
            'value_scores': value_scores,
            'quality_scores': quality_scores,
            'portfolio_weights': portfolio_weights,
            'hedge_ratios': hedge_ratios,
            'final_weights': final_weights
        }
    
    def store_transition(self, obs, actions, rewards, next_obs, done):
        """
        경험을 리플레이 버퍼에 저장
        
        Args:
            obs (dict): 현재 관측 (4개 에이전트)
            actions (dict): 행동 (value_scores, quality_scores, portfolio_weights, hedge_ratios, final_weights)
            rewards (dict): 보상 (value_reward, quality_reward, portfolio_reward, hedging_reward)
            next_obs (dict): 다음 관측 (4개 에이전트)
            done (bool): 에피소드 종료 여부
        """
        transition = {
            'value_obs': obs['value_obs'],
            'value_action': actions['value_scores'],
            'value_reward': rewards['value_reward'],
            'value_next_obs': next_obs['value_obs'],
            'quality_obs': obs['quality_obs'],
            'quality_action': actions['quality_scores'],
            'quality_reward': rewards['quality_reward'],
            'quality_next_obs': next_obs['quality_obs'],
            'portfolio_obs': obs['portfolio_obs'],
            'portfolio_action': actions['portfolio_weights'],
            'portfolio_reward': rewards['portfolio_reward'],
            'portfolio_next_obs': next_obs['portfolio_obs'],
            'hedging_obs': obs['hedging_obs'],
            'hedging_action': actions['hedge_ratios'],
            'hedging_reward': rewards['hedging_reward'],
            'hedging_next_obs': next_obs['hedging_obs'],
            'done': float(done)
        }
        self.replay_buffer.add(transition)
    
    def update_all(self, n_updates=10):
        """
        공유 리플레이 버퍼를 사용하여 모든 4개 에이전트 업데이트
        
        Args:
            n_updates (int): 수행할 그래디언트 업데이트 횟수
            
        Returns:
            dict: 모든 에이전트의 학습 손실, 버퍼 부족 시 None
        """
        if len(self.replay_buffer) < config.BATCH_SIZE:
            return None
        
        # 손실 추적 초기화
        losses = {
            'value': {'critic': [], 'actor': []},
            'quality': {'critic': [], 'actor': []},
            'portfolio': {'critic': [], 'actor': []},
            'hedging': {'critic': [], 'actor': []}
        }
        
        # 업데이트 수행
        for _ in range(n_updates):
            batch = self.replay_buffer.sample(config.BATCH_SIZE)
            
            # 각 에이전트 업데이트
            value_loss = self.value_agent.update(batch, 'value')
            quality_loss = self.quality_agent.update(batch, 'quality')
            portfolio_loss = self.portfolio_agent.update(batch, 'portfolio')
            hedging_loss = self.hedging_agent.update(batch, 'hedging')
            
            # 손실 기록
            losses['value']['critic'].append(value_loss['critic_loss'])
            losses['value']['actor'].append(value_loss['actor_loss'])
            losses['quality']['critic'].append(quality_loss['critic_loss'])
            losses['quality']['actor'].append(quality_loss['actor_loss'])
            losses['portfolio']['critic'].append(portfolio_loss['critic_loss'])
            losses['portfolio']['actor'].append(portfolio_loss['actor_loss'])
            losses['hedging']['critic'].append(hedging_loss['critic_loss'])
            losses['hedging']['actor'].append(hedging_loss['actor_loss'])
        
        # 평균 계산
        for agent_name in losses:
            for loss_type in losses[agent_name]:
                losses[agent_name][loss_type] = np.mean(losses[agent_name][loss_type])
        
        return losses
    
    def update_all_from_batch(self, batch):
        """
        외부 batch에서 직접 학습 (RACE 방식: Shared Buffer 사용)
        
        Args:
            batch (dict): Shared Replay Buffer에서 샘플링한 배치
        
        Returns:
            dict: 모든 에이전트의 학습 손실
        """
        # 각 에이전트 업데이트
        value_loss = self.value_agent.update(batch, 'value')
        quality_loss = self.quality_agent.update(batch, 'quality')
        portfolio_loss = self.portfolio_agent.update(batch, 'portfolio')
        hedging_loss = self.hedging_agent.update(batch, 'hedging')
        
        # 손실 집계
        losses = {
            'value': value_loss,
            'quality': quality_loss,
            'portfolio': portfolio_loss,
            'hedging': hedging_loss,
            'total': (value_loss['critic_loss'] + value_loss['actor_loss'] +
                     quality_loss['critic_loss'] + quality_loss['actor_loss'] +
                     portfolio_loss['critic_loss'] + portfolio_loss['actor_loss'] +
                     hedging_loss['critic_loss'] + hedging_loss['actor_loss']) / 8.0
        }
        
        return losses
    
    def mutate(self, mutation_prob=0.2, mutation_scale=0.02, verbose=False):
        """
        모든 에이전트 네트워크 변이 (GA용) - 가우시안 노이즈 기반
        
        Args:
            mutation_prob (float): 각 파라미터가 변이할 확률 (0.0~1.0)
            mutation_scale (float): 가우시안 노이즈의 표준편차 (σ)
            verbose (bool): 변이 과정 로그 출력
        
        Returns:
            list: 변이된 에이전트 이름 목록
        """
        mutated = []
        
        # 각 에이전트에 동일한 변이 설정 적용
        self.value_agent.mutate(mutation_prob, mutation_scale)
        mutated.append("Value")
        
        self.quality_agent.mutate(mutation_prob, mutation_scale)
        mutated.append("Quality")
        
        self.portfolio_agent.mutate(mutation_prob, mutation_scale)
        mutated.append("Portfolio")
        
        self.hedging_agent.mutate(mutation_prob, mutation_scale)
        mutated.append("Hedging")
        
        if verbose:
            print(f"      변이: {', '.join(mutated)} (확률={mutation_prob*100:.0f}%, scale={mutation_scale})")
        
        return mutated
    
    def clone(self):
        """
        이 멀티 에이전트 시스템의 깊은 복사본 생성
        
        Returns:
            MultiAgentSystem: 복제된 시스템
        """
        new_system = MultiAgentSystem(system_id=self.system_id)
        new_system.value_agent = self.value_agent.clone()
        new_system.quality_agent = self.quality_agent.clone()
        new_system.portfolio_agent = self.portfolio_agent.clone()
        new_system.hedging_agent = self.hedging_agent.clone()
        new_system.fitness = self.fitness
        return new_system
    
    def save(self, path):
        """모든 에이전트를 디스크에 저장"""
        import os
        os.makedirs(path, exist_ok=True)
        
        self.value_agent.save(f"{path}/value_agent.pth")
        self.quality_agent.save(f"{path}/quality_agent.pth")
        self.portfolio_agent.save(f"{path}/portfolio_agent.pth")
        self.hedging_agent.save(f"{path}/hedging_agent.pth")
    
    def load(self, path):
        """디스크에서 모든 에이전트 로드"""
        self.value_agent.load(f"{path}/value_agent.pth")
        self.quality_agent.load(f"{path}/quality_agent.pth")
        self.portfolio_agent.load(f"{path}/portfolio_agent.pth")
        self.hedging_agent.load(f"{path}/hedging_agent.pth")

