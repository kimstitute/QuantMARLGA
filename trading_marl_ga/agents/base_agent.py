"""
Base Agent 클래스

Actor-Critic 기반 MARL 에이전트

구조:
- USE_SHARED_ENCODER=True: RACE 스타일 공유 인코더 + 독립 헤드
- USE_SHARED_ENCODER=False: 독립 네트워크 (레거시)
"""

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.cuda.amp import autocast, GradScaler
import copy
import numpy as np
from agents.networks import (
    ActorNetwork, CriticNetwork,
    GlobalEncoders, ActorHead, CriticHead
)
from config import config


class BaseAgent:
    """
    기본 에이전트 클래스
    
    Actor-Critic 구조:
    - USE_SHARED_ENCODER=True: 공유 인코더 + 독립 헤드
    - USE_SHARED_ENCODER=False: 독립 네트워크
    """
    
    def __init__(self, obs_dim, action_dim, name="agent"):
        """
        Args:
            obs_dim (int): 관측 차원
            action_dim (int): 행동 차원
            name (str): 에이전트 타입 ('value', 'quality', 'portfolio', 'hedging')
        """
        self.name = name
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = config.DEVICE
        self.use_shared_encoder = config.USE_SHARED_ENCODER
        
        # ============================================
        # Networks
        # ============================================
        if self.use_shared_encoder:
            # RACE 스타일: 공유 인코더 + 독립 헤드
            global_encoders = GlobalEncoders()
            
            # 타입별 공유 인코더 참조
            self.encoder = global_encoders.get_encoder(name)
            
            # 독립 헤드 생성
            hidden_dim = config.ENCODER_HIDDEN_DIM
            self.actor_head = ActorHead(hidden_dim, action_dim).to(self.device)
            self.critic_head = CriticHead(hidden_dim).to(self.device)
            self.critic_head_target = copy.deepcopy(self.critic_head).to(self.device)
            
            # Target 헤드 동결
            for param in self.critic_head_target.parameters():
                param.requires_grad = False
            
            # Optimizer (헤드만 독립 학습, 인코더는 글로벌 optimizer 사용)
            self.actor_optimizer = Adam(
                self.actor_head.parameters(),  # 헤드만!
                lr=config.LEARNING_RATE_ACTOR
            )
            self.critic_optimizer = Adam(
                self.critic_head.parameters(),  # 헤드만!
                lr=config.LEARNING_RATE_CRITIC
            )
            
            # 글로벌 인코더 optimizer 참조
            self.encoder_optimizer = global_encoders.get_optimizer(name)
            
            # 하위 호환성을 위한 래핑
            self.actor = None  # 직접 사용 안 함
            self.critic = None
            self.critic_target = None
            
        else:
            # 레거시: 독립 네트워크
            self.actor = ActorNetwork(obs_dim, action_dim, config.HIDDEN_DIM).to(self.device)
            self.critic = CriticNetwork(obs_dim, config.HIDDEN_DIM).to(self.device)
            self.critic_target = copy.deepcopy(self.critic).to(self.device)
            
            # Freeze target network
            for param in self.critic_target.parameters():
                param.requires_grad = False
            
            # Optimizers
            self.actor_optimizer = Adam(
                self.actor.parameters(), 
                lr=config.LEARNING_RATE_ACTOR
            )
            self.critic_optimizer = Adam(
                self.critic.parameters(),
                lr=config.LEARNING_RATE_CRITIC
            )
            
            # 공유 모드 변수 초기화
            self.encoder = None
            self.actor_head = None
            self.critic_head = None
            self.critic_head_target = None
        
        # ============================================
        # Mixed Precision Training
        # ============================================
        self.use_amp = config.USE_AMP
        if self.use_amp:
            self.scaler = GradScaler()
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
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)  # (1, obs_dim)
            
            if self.use_shared_encoder:
                # 공유 인코더 모드
                features = self.encoder(obs_tensor)  # (1, hidden_dim)
                action = self.actor_head(features)  # (1, action_dim)
            else:
                # 독립 네트워크 모드
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
        if self.use_shared_encoder:
            # 공유 인코더 모드
            with torch.no_grad():
                # TD target 계산
                features_next = self.encoder(next_obs)
                next_values = self.critic_head_target(features_next)  # (batch_size, 1)
                td_target = rewards + (1 - dones) * config.GAMMA * next_values
            
            if self.use_amp:
                with autocast():
                    features = self.encoder(obs)
                    current_values = self.critic_head(features)
                    critic_loss = F.mse_loss(current_values, td_target)
                
                # 헤드 + 인코더 gradient 계산
                self.encoder_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                self.scaler.scale(critic_loss).backward()
                
                # Gradient clipping (분리)
                self.scaler.unscale_(self.encoder_optimizer)
                self.scaler.unscale_(self.critic_optimizer)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), config.GRAD_CLIP)
                torch.nn.utils.clip_grad_norm_(self.critic_head.parameters(), config.GRAD_CLIP)
                
                # 업데이트 (분리)
                self.scaler.step(self.encoder_optimizer)  # 글로벌 인코더
                self.scaler.step(self.critic_optimizer)   # 로컬 헤드
                self.scaler.update()
            else:
                features = self.encoder(obs)
                current_values = self.critic_head(features)
                critic_loss = F.mse_loss(current_values, td_target)
                
                # 헤드 + 인코더 gradient 계산
                self.encoder_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                
                # Gradient clipping (분리)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), config.GRAD_CLIP)
                torch.nn.utils.clip_grad_norm_(self.critic_head.parameters(), config.GRAD_CLIP)
                
                # 업데이트 (분리)
                self.encoder_optimizer.step()  # 글로벌 인코더
                self.critic_optimizer.step()   # 로컬 헤드
        else:
            # 독립 네트워크 모드
            with torch.no_grad():
                next_values = self.critic_target(next_obs)  # (batch_size, 1)
                td_target = rewards + (1 - dones) * config.GAMMA * next_values
            
            if self.use_amp:
                with autocast():
                    current_values = self.critic(obs)
                    critic_loss = F.mse_loss(current_values, td_target)
                
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
                current_values = self.critic(obs)
                critic_loss = F.mse_loss(current_values, td_target)
                
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
            advantages = td_target - current_values.detach()
            # 정규화 (학습 안정화)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        if self.use_shared_encoder:
            # 공유 인코더 모드
            if self.use_amp:
                with autocast():
                    features = self.encoder(obs)
                    predicted_actions = self.actor_head(features)
                    action_loss = F.mse_loss(predicted_actions, actions, reduction='none')
                    actor_loss = (action_loss.mean(dim=1, keepdim=True) * advantages.abs()).mean()
                
                # 헤드 + 인코더 gradient 계산
                self.encoder_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()
                self.scaler.scale(actor_loss).backward()
                
                # Gradient clipping (분리)
                self.scaler.unscale_(self.encoder_optimizer)
                self.scaler.unscale_(self.actor_optimizer)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), config.GRAD_CLIP)
                torch.nn.utils.clip_grad_norm_(self.actor_head.parameters(), config.GRAD_CLIP)
                
                # 업데이트 (분리)
                self.scaler.step(self.encoder_optimizer)  # 글로벌 인코더
                self.scaler.step(self.actor_optimizer)    # 로컬 헤드
                self.scaler.update()
            else:
                features = self.encoder(obs)
                predicted_actions = self.actor_head(features)
                action_loss = F.mse_loss(predicted_actions, actions, reduction='none')
                actor_loss = (action_loss.mean(dim=1, keepdim=True) * advantages.abs()).mean()
                
                # 헤드 + 인코더 gradient 계산
                self.encoder_optimizer.zero_grad()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                
                # Gradient clipping (분리)
                torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), config.GRAD_CLIP)
                torch.nn.utils.clip_grad_norm_(self.actor_head.parameters(), config.GRAD_CLIP)
                
                # 업데이트 (분리)
                self.encoder_optimizer.step()  # 글로벌 인코더
                self.actor_optimizer.step()    # 로컬 헤드
        else:
            # 독립 네트워크 모드
            if self.use_amp:
                with autocast():
                    predicted_actions = self.actor(obs)
                    action_loss = F.mse_loss(predicted_actions, actions, reduction='none')
                    actor_loss = (action_loss.mean(dim=1, keepdim=True) * advantages.abs()).mean()
                
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
                predicted_actions = self.actor(obs)
                action_loss = F.mse_loss(predicted_actions, actions, reduction='none')
                actor_loss = (action_loss.mean(dim=1, keepdim=True) * advantages.abs()).mean()
                
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
        if self.use_shared_encoder:
            # 공유 모드: Critic 헤드만 업데이트 (인코더는 공유)
            for param, target_param in zip(
                self.critic_head.parameters(),
                self.critic_head_target.parameters()
            ):
                target_param.data.copy_(
                    config.TAU * param.data + (1 - config.TAU) * target_param.data
                )
        else:
            # 독립 모드: Critic 전체 업데이트
            for param, target_param in zip(
                self.critic.parameters(),
                self.critic_target.parameters()
            ):
                target_param.data.copy_(
                    config.TAU * param.data + (1 - config.TAU) * target_param.data
                )
    
    def mutate(self, mutation_prob=0.2, mutation_scale_ratio=0.05):
        """
        GA 변이 (Mutation)
        
        공유 모드: 인코더는 변이 안 함, 헤드만 변이
        독립 모드: 전체 네트워크 변이
        
        Args:
            mutation_prob (float): 각 파라미터가 변이할 확률
            mutation_scale_ratio (float): 가중치 크기 대비 노이즈 비율
        
        Note:
            - 공유 인코더는 모든 시스템이 공유하므로 변이 안 함!
            - 헤드만 독립적으로 변이
        """
        with torch.no_grad():
            if self.use_shared_encoder:
                # 공유 모드: 헤드만 변이 (인코더는 공유이므로 변이 안 함!)
                for param in self.actor_head.parameters():
                    if np.random.rand() < mutation_prob:
                        param_scale = param.abs().mean() + 1e-8
                        noise_std = param_scale * mutation_scale_ratio
                        noise = torch.randn_like(param) * noise_std
                        param.add_(noise)
                
                for param in self.critic_head.parameters():
                    if np.random.rand() < mutation_prob * 0.5:
                        param_scale = param.abs().mean() + 1e-8
                        noise_std = param_scale * (mutation_scale_ratio * 0.5)
                        noise = torch.randn_like(param) * noise_std
                        param.add_(noise)
                
                # Target 헤드 동기화
                self.critic_head_target.load_state_dict(self.critic_head.state_dict())
            else:
                # 독립 모드: 전체 네트워크 변이
                for param in self.actor.parameters():
                    if np.random.rand() < mutation_prob:
                        param_scale = param.abs().mean() + 1e-8
                        noise_std = param_scale * mutation_scale_ratio
                        noise = torch.randn_like(param) * noise_std
                        param.add_(noise)
                
                for param in self.critic.parameters():
                    if np.random.rand() < mutation_prob * 0.5:
                        param_scale = param.abs().mean() + 1e-8
                        noise_std = param_scale * (mutation_scale_ratio * 0.5)
                        noise = torch.randn_like(param) * noise_std
                        param.add_(noise)
                
                # Target 동기화
                self.critic_target.load_state_dict(self.critic.state_dict())
    
    def clone(self):
        """
        에이전트 복사
        
        공유 모드: 인코더는 참조 유지, 헤드만 deep copy
        독립 모드: 전체 네트워크 deep copy
        
        Returns:
            BaseAgent: 복사된 에이전트
        """
        new_agent = BaseAgent(self.obs_dim, self.action_dim, self.name)
        
        if self.use_shared_encoder:
            # 공유 모드: 헤드만 복사 (인코더는 자동으로 같은 참조)
            new_agent.actor_head.load_state_dict(self.actor_head.state_dict())
            new_agent.critic_head.load_state_dict(self.critic_head.state_dict())
            new_agent.critic_head_target.load_state_dict(self.critic_head_target.state_dict())
        else:
            # 독립 모드: 전체 네트워크 복사
            new_agent.actor.load_state_dict(self.actor.state_dict())
            new_agent.critic.load_state_dict(self.critic.state_dict())
            new_agent.critic_target.load_state_dict(self.critic_target.state_dict())
        
        return new_agent
    
    def save(self, path):
        """
        에이전트 저장
        
        공유 모드: 헤드만 저장 (인코더는 전역)
        독립 모드: 전체 네트워크 저장
        
        Args:
            path (str): 저장 경로
        """
        if self.use_shared_encoder:
            torch.save({
                'use_shared_encoder': True,
                'actor_head': self.actor_head.state_dict(),
                'critic_head': self.critic_head.state_dict(),
                'critic_head_target': self.critic_head_target.state_dict(),
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
            }, path)
        else:
            torch.save({
                'use_shared_encoder': False,
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
        checkpoint = torch.load(path, map_location=self.device)
        
        if checkpoint.get('use_shared_encoder', False):
            # 공유 모드: 헤드만 로드
            self.actor_head.load_state_dict(checkpoint['actor_head'])
            self.critic_head.load_state_dict(checkpoint['critic_head'])
            self.critic_head_target.load_state_dict(checkpoint['critic_head_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        else:
            # 독립 모드: 전체 네트워크 로드
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])

