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
        
        # 네트워크 정의
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # 가중치 초기화
        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        가중치 초기화 (Orthogonal initialization)
        
        RL에서 일반적으로 사용:
        - Hidden layers: orthogonal with gain=sqrt(2)
        - Output layer: orthogonal with small gain (탐색 촉진)
        """
        # Hidden layers: Orthogonal init
        nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc1.bias, 0.0)
        
        nn.init.orthogonal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc2.bias, 0.0)
        
        # Output layer: Small initialization for exploration
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
        nn.init.constant_(self.fc3.bias, 0.0)
    
    def forward(self, obs):
        """
        Args:
            obs (torch.Tensor): 관측 (batch_size, obs_dim)
            
        Returns:
            torch.Tensor: 행동 (batch_size, action_dim), [0, 1] 범위
        """
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x
    
    @property
    def net(self):
        """하위 호환성을 위한 속성 (사용하지 말 것)"""
        return nn.Sequential(
            self.fc1, self.relu,
            self.fc2, self.relu,
            self.fc3, self.sigmoid
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
        
        # 네트워크 정의
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        
        # 가중치 초기화
        self._initialize_weights()
        
    def _initialize_weights(self):
        """
        가중치 초기화 (Orthogonal initialization)
        
        RL Critic에서 일반적으로 사용:
        - All layers: orthogonal with gain=sqrt(2)
        """
        # Hidden layers
        nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc1.bias, 0.0)
        
        nn.init.orthogonal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc2.bias, 0.0)
        
        # Output layer
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        nn.init.constant_(self.fc3.bias, 0.0)
    
    def forward(self, obs):
        """
        Args:
            obs (torch.Tensor): 관측 (batch_size, obs_dim)
            
        Returns:
            torch.Tensor: 가치 추정 (batch_size, 1)
        """
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    @property
    def net(self):
        """하위 호환성을 위한 속성 (사용하지 말 것)"""
        return nn.Sequential(
            self.fc1, self.relu,
            self.fc2, self.relu,
            self.fc3
        )

