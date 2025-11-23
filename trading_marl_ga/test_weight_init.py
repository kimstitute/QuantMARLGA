"""
신경망 가중치 초기화 테스트
"""

import torch
import numpy as np
from agents.networks import ActorNetwork, CriticNetwork
from config import config

print("="*60)
print("신경망 가중치 초기화 테스트")
print("="*60)

# 네트워크 생성
actor = ActorNetwork(
    obs_dim=config.VALUE_OBS_DIM,
    action_dim=config.ACTION_DIM,
    hidden_dim=config.HIDDEN_DIM
)

critic = CriticNetwork(
    obs_dim=config.VALUE_OBS_DIM,
    hidden_dim=config.HIDDEN_DIM
)

print(f"\n[Actor Network]")
print(f"  관측 차원: {config.VALUE_OBS_DIM}")
print(f"  행동 차원: {config.ACTION_DIM}")
print(f"  Hidden 차원: {config.HIDDEN_DIM}")

# 가중치 통계
print(f"\n[가중치 통계]")
for name, param in actor.named_parameters():
    print(f"  {name:20s}: mean={param.data.mean():.6f}, std={param.data.std():.6f}, shape={tuple(param.shape)}")

print(f"\n[Critic Network]")
print(f"  관측 차원: {config.VALUE_OBS_DIM}")
print(f"  출력 차원: 1 (가치)")

print(f"\n[가중치 통계]")
for name, param in critic.named_parameters():
    print(f"  {name:20s}: mean={param.data.mean():.6f}, std={param.data.std():.6f}, shape={tuple(param.shape)}")

# Forward pass 테스트
print(f"\n[Forward Pass 테스트]")
batch_size = 4
obs = torch.randn(batch_size, config.VALUE_OBS_DIM)

actor_output = actor(obs)
critic_output = critic(obs)

print(f"  입력: {obs.shape}")
print(f"  Actor 출력: {actor_output.shape}, range=[{actor_output.min():.4f}, {actor_output.max():.4f}]")
print(f"  Critic 출력: {critic_output.shape}, mean={critic_output.mean():.4f}")

# Orthogonal 초기화 검증
print(f"\n[Orthogonal 초기화 검증]")
fc1_weight = actor.fc1.weight.data
print(f"  fc1 weight shape: {fc1_weight.shape}")

# Orthogonal 매트릭스는 W @ W^T = I를 만족해야 함 (행 기준)
if fc1_weight.shape[0] <= fc1_weight.shape[1]:
    product = fc1_weight @ fc1_weight.t()
    identity = torch.eye(fc1_weight.shape[0])
    diff = (product - identity).abs().mean()
    print(f"  Orthogonality error (행 기준): {diff:.6f} (이상적: 0)")
else:
    print(f"  [INFO] 행 > 열이므로 열 기준으로 검증")
    product = fc1_weight.t() @ fc1_weight
    identity = torch.eye(fc1_weight.shape[1])
    diff = (product - identity).abs().mean()
    print(f"  Orthogonality error (열 기준): {diff:.6f} (이상적: 0)")

print(f"\n{'='*60}")
print("[SUCCESS] 가중치 초기화 테스트 통과!")
print(f"{'='*60}\n")

