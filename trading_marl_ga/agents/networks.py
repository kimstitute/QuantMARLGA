"""
에이전트를 위한 신경망 아키텍처

트레이딩 에이전트를 위한 Actor-Critic 네트워크:
- ActorNetwork: 정책 네트워크 (상태 -> 행동)
- CriticNetwork: 가치 네트워크 (상태 -> 가치)
"""

import torch
import torch.nn as nn


class ActorNetwork(nn.Module):
    """
    Actor 네트워크: 관측을 행동으로 매핑
    
    Sigmoid를 사용하여 [0, 1] 범위로 출력:
    - Value Agent: 종목별 가치 점수
    - Quality Agent: 종목별 품질 점수
    - Portfolio Agent: 포트폴리오 비중 (정규화 전)
    - Hedging Agent: 종목별 헷징 비율
    """
    
    def __init__(self, obs_dim, action_dim, hidden_dim=64):
        """
        Args:
            obs_dim (int): 관측 차원
            action_dim (int): 행동 차원 (종목 개수)
            hidden_dim (int): 은닉층 차원
        """
        super(ActorNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid()  # [0, 1] 범위 출력
        )
    
    def forward(self, obs):
        """
        Args:
            obs (torch.Tensor): 관측 (batch_size, obs_dim)
            
        Returns:
            torch.Tensor: 행동 (batch_size, action_dim), [0, 1] 범위
        """
        return self.net(obs)


class CriticNetwork(nn.Module):
    """
    Critic 네트워크: 상태 가치 추정
    
    관측을 스칼라 가치 추정값으로 매핑
    Actor 학습에서 advantage 추정에 사용
    """
    
    def __init__(self, obs_dim, hidden_dim=64):
        """
        Args:
            obs_dim (int): 관측 차원
            hidden_dim (int): 은닉층 차원
        """
        super(CriticNetwork, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, obs):
        """
        Args:
            obs (torch.Tensor): 관측 (batch_size, obs_dim)
            
        Returns:
            torch.Tensor: 가치 추정 (batch_size, 1)
        """
        return self.net(obs)

