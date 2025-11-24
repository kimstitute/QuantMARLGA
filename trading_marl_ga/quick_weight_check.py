"""
간단한 가중치 크기 확인
"""

import torch
from agents.networks import ActorNetwork, CriticNetwork
import config as cfg

print("="*60)
print("가중치 vs 노이즈 크기 간단 분석")
print("="*60)

# Actor 생성
actor = ActorNetwork(
    obs_dim=cfg.Config.VALUE_OBS_DIM,
    action_dim=cfg.Config.ACTION_DIM,
    hidden_dim=cfg.Config.HIDDEN_DIM
)

# Critic 생성
critic = CriticNetwork(
    obs_dim=cfg.Config.VALUE_OBS_DIM,
    hidden_dim=cfg.Config.HIDDEN_DIM
)

print(f"\n[네트워크 크기]")
print(f"  Input: {cfg.Config.VALUE_OBS_DIM}")
print(f"  Hidden: {cfg.Config.HIDDEN_DIM}")
print(f"  Output: {cfg.Config.ACTION_DIM}")

# Actor 가중치
print(f"\n{'='*60}")
print("Actor 가중치")
print("="*60)

actor_weights = []
for name, param in actor.named_parameters():
    if 'weight' in name:
        mean_abs = param.data.abs().mean().item()
        std = param.data.std().item()
        actor_weights.append(mean_abs)
        print(f"{name:20s}: 평균 |W| = {mean_abs:.6f}, std = {std:.6f}")

actor_avg = sum(actor_weights) / len(actor_weights)
print(f"\n전체 평균 |W|: {actor_avg:.6f}")

# Critic 가중치
print(f"\n{'='*60}")
print("Critic 가중치")
print("="*60)

critic_weights = []
for name, param in critic.named_parameters():
    if 'weight' in name:
        mean_abs = param.data.abs().mean().item()
        std = param.data.std().item()
        critic_weights.append(mean_abs)
        print(f"{name:20s}: 평균 |W| = {mean_abs:.6f}, std = {std:.6f}")

critic_avg = sum(critic_weights) / len(critic_weights)
print(f"\n전체 평균 |W|: {critic_avg:.6f}")

# 노이즈 비율 계산
print(f"\n{'='*60}")
print("노이즈 비율 분석")
print("="*60)

mutation_scale = cfg.Config.MUTATION_SCALE
print(f"\n현재 mutation_scale: {mutation_scale}")

actor_ratio = mutation_scale / actor_avg
critic_ratio = (mutation_scale * 0.5) / critic_avg

print(f"\nActor:")
print(f"  가중치 평균: {actor_avg:.6f}")
print(f"  노이즈 σ:    {mutation_scale:.6f}")
print(f"  비율:        {actor_ratio:.2%}")

print(f"\nCritic:")
print(f"  가중치 평균: {critic_avg:.6f}")
print(f"  노이즈 σ:    {mutation_scale * 0.5:.6f}")
print(f"  비율:        {critic_ratio:.2%}")

# 판단
print(f"\n{'='*60}")
print("평가")
print("="*60)

print(f"\n일반적인 ES/GA 가이드라인:")
print(f"  1-5%:   매우 작음 (수렴 느림)")
print(f"  5-10%:  적절한 범위")
print(f"  10-20%: 큰 편 (초기 탐색)")
print(f"  >20%:   너무 큼 (불안정)")

if actor_ratio > 0.20:
    print(f"\n⚠️  Actor 비율 {actor_ratio:.1%}는 너무 큽니다!")
    recommended = actor_avg * 0.10  # 10% 목표
    print(f"   권장: mutation_scale = {recommended:.4f} (현재의 {recommended/mutation_scale:.1f}배)")
elif actor_ratio > 0.10:
    print(f"\n⚠️  Actor 비율 {actor_ratio:.1%}는 큰 편입니다.")
    recommended = actor_avg * 0.07  # 7% 목표
    print(f"   권장: mutation_scale = {recommended:.4f} (현재의 {recommended/mutation_scale:.1f}배)")
elif actor_ratio > 0.05:
    print(f"\n✓  Actor 비율 {actor_ratio:.1%}는 적절한 범위입니다.")
else:
    print(f"\n⚠️  Actor 비율 {actor_ratio:.1%}는 작습니다.")
    print(f"   탐색이 느릴 수 있습니다.")

print(f"\n{'='*60}")

