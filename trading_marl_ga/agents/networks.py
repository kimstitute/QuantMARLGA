"""
에이전트를 위한 신경망 아키텍처

RACE 스타일 공유 인코더 구조:
- SharedEncoder: 타입별 공유 인코더 (obs -> features)
- ActorHead: 시스템별 독립 헤드 (features -> actions)
- CriticHead: 시스템별 독립 헤드 (features -> value)

레거시:
- ActorNetwork: 통합 Actor (하위 호환용)
- CriticNetwork: 통합 Critic (하위 호환용)
"""

import torch
import torch.nn as nn


class SharedEncoder(nn.Module):
    """
    타입별 공유 인코더 (RACE 스타일)
    
    같은 타입의 에이전트들이 인코더를 공유:
    - 모든 시스템의 Value 에이전트: value_encoder 공유
    - 모든 시스템의 Quality 에이전트: quality_encoder 공유
    - 모든 시스템의 Portfolio 에이전트: portfolio_encoder 공유
    - 모든 시스템의 Hedging 에이전트: hedging_encoder 공유
    
    장점:
    - 샘플 효율 11배 증가 (11개 시스템 경험 공유)
    - Off-policy learning 정당화 (같은 feature space)
    - 메모리 효율 향상
    """
    
    def __init__(self, obs_dim, hidden_dim=64):
        """
        Args:
            obs_dim (int): 관측 차원
            hidden_dim (int): 출력 feature 차원
        """
        super(SharedEncoder, self).__init__()
        
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        nn.init.orthogonal_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc1.bias, 0.0)
        
        nn.init.orthogonal_(self.fc2.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(self.fc2.bias, 0.0)
    
    def forward(self, obs):
        """
        Args:
            obs (torch.Tensor): 관측 (batch_size, obs_dim)
            
        Returns:
            torch.Tensor: Feature (batch_size, hidden_dim)
        """
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        return x


class ActorHead(nn.Module):
    """
    시스템별 독립 Actor 헤드
    
    SharedEncoder의 출력을 행동으로 변환
    각 시스템마다 독립적인 헤드를 가짐
    """
    
    def __init__(self, hidden_dim, action_dim):
        """
        Args:
            hidden_dim (int): 입력 feature 차원
            action_dim (int): 행동 차원 (종목 개수)
        """
        super(ActorHead, self).__init__()
        
        self.fc = nn.Linear(hidden_dim, action_dim)
        self.sigmoid = nn.Sigmoid()
        
        # 작은 초기화로 탐색 촉진
        nn.init.orthogonal_(self.fc.weight, gain=0.01)
        nn.init.constant_(self.fc.bias, 0.0)
    
    def forward(self, features):
        """
        Args:
            features (torch.Tensor): Encoder 출력 (batch_size, hidden_dim)
            
        Returns:
            torch.Tensor: 행동 (batch_size, action_dim), [0, 1] 범위
        """
        return self.sigmoid(self.fc(features))


class CriticHead(nn.Module):
    """
    시스템별 독립 Critic 헤드
    
    SharedEncoder의 출력을 가치로 변환
    각 시스템마다 독립적인 헤드를 가짐
    """
    
    def __init__(self, hidden_dim):
        """
        Args:
            hidden_dim (int): 입력 feature 차원
        """
        super(CriticHead, self).__init__()
        
        self.fc = nn.Linear(hidden_dim, 1)
        
        nn.init.orthogonal_(self.fc.weight, gain=1.0)
        nn.init.constant_(self.fc.bias, 0.0)
    
    def forward(self, features):
        """
        Args:
            features (torch.Tensor): Encoder 출력 (batch_size, hidden_dim)
            
        Returns:
            torch.Tensor: 가치 추정 (batch_size, 1)
        """
        return self.fc(features)


class GlobalEncoders:
    """
    전역 공유 인코더 관리자 (RACE 스타일)
    
    4개의 타입별 공유 인코더를 관리:
    - value_encoder: 모든 Value 에이전트 공유
    - quality_encoder: 모든 Quality 에이전트 공유
    - portfolio_encoder: 모든 Portfolio 에이전트 공유
    - hedging_encoder: 모든 Hedging 에이전트 공유
    
    Singleton 패턴으로 전역 인스턴스 하나만 생성
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GlobalEncoders, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not GlobalEncoders._initialized:
            self.value_encoder = None
            self.quality_encoder = None
            self.portfolio_encoder = None
            self.hedging_encoder = None
            # 글로벌 optimizers
            self.value_optimizer = None
            self.quality_optimizer = None
            self.portfolio_optimizer = None
            self.hedging_optimizer = None
            GlobalEncoders._initialized = True
    
    def initialize(self, value_obs_dim, portfolio_obs_dim, hedging_obs_dim, hidden_dim=64, device='cpu', lr=3e-4):
        """
        인코더 초기화
        
        Args:
            value_obs_dim (int): Value/Quality 관측 차원
            portfolio_obs_dim (int): Portfolio 관측 차원
            hedging_obs_dim (int): Hedging 관측 차원
            hidden_dim (int): Feature 차원
            device (str): 디바이스
            lr (float): 인코더 learning rate
        """
        from torch.optim import Adam
        
        self.value_encoder = SharedEncoder(value_obs_dim, hidden_dim).to(device)
        self.quality_encoder = SharedEncoder(value_obs_dim, hidden_dim).to(device)
        self.portfolio_encoder = SharedEncoder(portfolio_obs_dim, hidden_dim).to(device)
        self.hedging_encoder = SharedEncoder(hedging_obs_dim, hidden_dim).to(device)
        self.device = device
        
        # 각 인코더마다 독립 optimizer
        self.value_optimizer = Adam(self.value_encoder.parameters(), lr=lr)
        self.quality_optimizer = Adam(self.quality_encoder.parameters(), lr=lr)
        self.portfolio_optimizer = Adam(self.portfolio_encoder.parameters(), lr=lr)
        self.hedging_optimizer = Adam(self.hedging_encoder.parameters(), lr=lr)
        
        print(f"[GlobalEncoders] 초기화 완료")
        print(f"  Value/Quality: {value_obs_dim} -> {hidden_dim}")
        print(f"  Portfolio: {portfolio_obs_dim} -> {hidden_dim}")
        print(f"  Hedging: {hedging_obs_dim} -> {hidden_dim}")
        print(f"  Device: {device}")
        print(f"  Encoder LR: {lr}")
    
    def get_encoder(self, agent_type):
        """
        에이전트 타입에 맞는 인코더 반환
        
        Args:
            agent_type (str): 'value', 'quality', 'portfolio', 'hedging'
            
        Returns:
            SharedEncoder: 해당 타입의 공유 인코더
        """
        encoders = {
            'value': self.value_encoder,
            'quality': self.quality_encoder,
            'portfolio': self.portfolio_encoder,
            'hedging': self.hedging_encoder
        }
        return encoders[agent_type]
    
    def get_optimizer(self, agent_type):
        """
        에이전트 타입에 맞는 인코더 optimizer 반환
        
        Args:
            agent_type (str): 'value', 'quality', 'portfolio', 'hedging'
            
        Returns:
            torch.optim.Optimizer: 해당 타입의 인코더 optimizer
        """
        optimizers = {
            'value': self.value_optimizer,
            'quality': self.quality_optimizer,
            'portfolio': self.portfolio_optimizer,
            'hedging': self.hedging_optimizer
        }
        return optimizers[agent_type]
    
    def step_encoder(self, agent_type):
        """
        특정 타입의 인코더 optimizer step
        
        헤드 학습 후 gradient가 인코더에 누적되어 있음
        이 메서드로 인코더 파라미터 업데이트
        
        Args:
            agent_type (str): 'value', 'quality', 'portfolio', 'hedging'
        """
        optimizer = self.get_optimizer(agent_type)
        optimizer.step()
    
    def zero_grad_encoder(self, agent_type):
        """
        특정 타입의 인코더 gradient 초기화
        
        Args:
            agent_type (str): 'value', 'quality', 'portfolio', 'hedging'
        """
        optimizer = self.get_optimizer(agent_type)
        optimizer.zero_grad()
    
    def reset(self):
        """전역 인코더 리셋 (테스트용)"""
        self.value_encoder = None
        self.quality_encoder = None
        self.portfolio_encoder = None
        self.hedging_encoder = None
        self.value_optimizer = None
        self.quality_optimizer = None
        self.portfolio_optimizer = None
        self.hedging_optimizer = None
        GlobalEncoders._initialized = False


# ==================================================
# 레거시 네트워크 (하위 호환용)
# ==================================================

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

