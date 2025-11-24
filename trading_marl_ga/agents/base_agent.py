"""
Base Agent 클래스

Actor-Critic 기반 MARL 에이전트
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
import copy
import numpy as np
from agents.networks import ActorNetwork, CriticNetwork
from config import config


class BaseAgent:
    """
    기본 에이전트 클래스
    
    Actor-Critic 구조:
    - Actor: 행동 선택 (정책)
    - Critic: 가치 평가 (Q-value)
    - Target Critic: 안정적 학습용
    """
    
    def __init__(self, obs_dim, action_dim, name="agent"):
        """
        Args:
            obs_dim (int): 관측 차원
            action_dim (int): 행동 차원
            name (str): 에이전트 이름
        """
        self.name = name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # ============================================
        # Networks
        # ============================================
        self.device = config.DEVICE
        self.actor = ActorNetwork(obs_dim, action_dim, config.HIDDEN_DIM).to(self.device)
        self.critic = CriticNetwork(obs_dim, config.HIDDEN_DIM).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        
        # Freeze target network
        for param in self.critic_target.parameters():
            param.requires_grad = False
        
        # ============================================
        # Optimizers
        # ============================================
        self.actor_optimizer = Adam(
            self.actor.parameters(), 
            lr=config.LEARNING_RATE_ACTOR
        )
        self.critic_optimizer = Adam(
            self.critic.parameters(),
            lr=config.LEARNING_RATE_CRITIC
        )
        
        # ============================================
        # Mixed Precision Training (Colab 최적화)
        # ============================================
        self.use_amp = config.USE_AMP
        if self.use_amp:
            # PyTorch 2.0+ API: 명시적으로 device 지정
            self.scaler = GradScaler('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.scaler = None
    
    def act(self, obs, deterministic=False):
        """
        행동 선택
        
        Args:
            obs (np.ndarray): 관측 (obs_dim,)
            deterministic (bool): 결정적 행동 여부
        
        Returns:
            np.ndarray: 행동 (action_dim,)
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # (1, obs_dim) -> GPU
            action = self.actor(obs_tensor)  # (1, action_dim)
            
            if not deterministic:
                # 탐험을 위한 노이즈 추가
                noise = torch.randn_like(action) * 0.1
                action = action + noise
                action = torch.clamp(action, 0, 1)
            
            return action.squeeze(0).cpu().numpy()
    
    def update(self, batch, agent_prefix):
        """
        에이전트 업데이트 (Actor + Critic)
        
        Args:
            batch (dict): Replay buffer에서 샘플링한 배치
            agent_prefix (str): 'value', 'quality', 'portfolio', 'hedging'
        
        Returns:
            dict: 손실 값들
        """
        # 배치에서 데이터 추출 및 GPU 전송
        obs = batch[f'{agent_prefix}_obs'].to(self.device)
        actions = batch[f'{agent_prefix}_action'].to(self.device)
        rewards = batch[f'{agent_prefix}_reward'].to(self.device)
        next_obs = batch[f'{agent_prefix}_next_obs'].to(self.device)
        dones = batch['done'].to(self.device)
        
        # Shape 확인 및 수정 (broadcasting 문제 방지)
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(1)  # (batch_size,) → (batch_size, 1)
        if dones.dim() == 1:
            dones = dones.unsqueeze(1)  # (batch_size,) → (batch_size, 1)
        
        # ============================================
        # Critic Update (with Mixed Precision if enabled)
        # ============================================
        with torch.no_grad():
            # TD target 계산
            next_values = self.critic_target(next_obs)  # (batch_size, 1)
            td_target = rewards + (1 - dones) * config.GAMMA * next_values  # (batch_size, 1)
        
        # Mixed Precision Training (Colab 최적화)
        if self.use_amp:
            with autocast():
                # Current Q-value
                current_values = self.critic(obs)
                # Critic loss (MSE)
                critic_loss = F.mse_loss(current_values, td_target)
            
            # Critic 업데이트 (with GradScaler)
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()
            self.scaler.unscale_(self.critic_optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                config.GRAD_CLIP
            )
            self.scaler.step(self.critic_optimizer)
            self.scaler.update()
        else:
            # 일반 학습 (Local)
            current_values = self.critic(obs)
            critic_loss = F.mse_loss(current_values, td_target)
            
            # Critic 업데이트
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                config.GRAD_CLIP
            )
            self.critic_optimizer.step()
        
        # ============================================
        # Actor Update (with Mixed Precision if enabled)
        # ============================================
        with torch.no_grad():
            # Advantage 계산
            advantages = td_target - current_values
            # 정규화 (학습 안정화)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Mixed Precision Training (Colab 최적화)
        if self.use_amp:
            with autocast():
                # Actor 출력
                predicted_actions = self.actor(obs)
                # Actor loss (Advantage-weighted MSE)
                action_loss = F.mse_loss(predicted_actions, actions, reduction='none')
                actor_loss = (action_loss.mean(dim=1, keepdim=True) * advantages.abs()).mean()
            
            # Actor 업데이트 (with GradScaler)
            self.actor_optimizer.zero_grad()
            self.scaler.scale(actor_loss).backward()
            self.scaler.unscale_(self.actor_optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                config.GRAD_CLIP
            )
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            # 일반 학습 (Local)
            predicted_actions = self.actor(obs)
            action_loss = F.mse_loss(predicted_actions, actions, reduction='none')
            actor_loss = (action_loss.mean(dim=1, keepdim=True) * advantages.abs()).mean()
            
            # Actor 업데이트
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                config.GRAD_CLIP
            )
            self.actor_optimizer.step()
        
        # ============================================
        # Soft Update Target Network
        # ============================================
        self.soft_update_target()
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item()
        }
    
    def soft_update_target(self):
        """
        Target network 소프트 업데이트
        
        θ' <- τ * θ + (1 - τ) * θ'
        """
        for param, target_param in zip(
            self.critic.parameters(),
            self.critic_target.parameters()
        ):
            target_param.data.copy_(
                config.TAU * param.data + (1 - config.TAU) * target_param.data
            )
    
    def mutate(self, mutation_prob=0.2, mutation_scale_ratio=0.05):
        """
        GA 변이 (Mutation) - 상대적 크기 기반 가우시안 노이즈
        
        각 파라미터의 실제 크기에 비례하는 노이즈를 추가하여
        레이어별 가중치 크기 차이에 자동으로 적응
        
        Args:
            mutation_prob (float): 각 파라미터가 변이할 확률 (0.0~1.0)
            mutation_scale_ratio (float): 가중치 크기 대비 노이즈 비율
                예: 0.05 = 가중치 평균 절댓값의 5%
        
        Note:
            - Actor: 더 적극적으로 변이 (전체 mutation_prob, ratio)
            - Critic: 더 보수적으로 변이 (mutation_prob * 0.5, ratio * 0.5)
            - 상대적 크기 기반으로 모든 레이어에 균형잡힌 변이
        """
        with torch.no_grad():
            # Actor 변이 (정책 탐색)
            for param in self.actor.parameters():
                if np.random.rand() < mutation_prob:
                    # 현재 파라미터 크기에 비례하는 노이즈
                    param_scale = param.abs().mean() + 1e-8  # 0 방지
                    noise_std = param_scale * mutation_scale_ratio
                    noise = torch.randn_like(param) * noise_std
                    param.add_(noise)
            
            # Critic 변이 (가치 함수는 더 보수적으로)
            for param in self.critic.parameters():
                if np.random.rand() < mutation_prob * 0.5:
                    param_scale = param.abs().mean() + 1e-8
                    noise_std = param_scale * (mutation_scale_ratio * 0.5)
                    noise = torch.randn_like(param) * noise_std
                    param.add_(noise)
    
    def clone(self):
        """
        에이전트 복사 (Deep copy)
        
        Returns:
            BaseAgent: 복사된 에이전트
        """
        new_agent = BaseAgent(self.obs_dim, self.action_dim, self.name)
        
        # 네트워크 가중치 복사
        new_agent.actor.load_state_dict(self.actor.state_dict())
        new_agent.critic.load_state_dict(self.critic.state_dict())
        new_agent.critic_target.load_state_dict(self.critic_target.state_dict())
        
        return new_agent
    
    def save(self, path):
        """
        에이전트 저장
        
        Args:
            path (str): 저장 경로
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """
        에이전트 로드
        
        Args:
            path (str): 로드 경로
        """
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

