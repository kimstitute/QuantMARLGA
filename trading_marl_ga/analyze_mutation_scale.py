"""
변이 노이즈 크기 분석

신경망 가중치의 실제 크기와 변이 노이즈 크기를 비교
"""

import torch
import numpy as np
from agents.base_agent import BaseAgent
import config as cfg

print("="*80)
print("변이 노이즈 크기 분석")
print("="*80)

# Value Agent 생성 (예시)
agent = BaseAgent(
    obs_dim=cfg.Config.VALUE_OBS_DIM,
    action_dim=cfg.Config.ACTION_DIM,
    name="value"
)

print(f"\n[네트워크 구조]")
print(f"  Input: {cfg.Config.VALUE_OBS_DIM}")
print(f"  Hidden: {cfg.Config.HIDDEN_DIM}")
print(f"  Output: {cfg.Config.ACTION_DIM}")

# Actor 네트워크 가중치 분석
print(f"\n{'='*80}")
print("Actor 네트워크 가중치 분석")
print("="*80)

actor_stats = []
for name, param in agent.actor.named_parameters():
    if 'weight' in name:
        # 가중치 통계
        mean = param.data.abs().mean().item()
        std = param.data.std().item()
        min_val = param.data.min().item()
        max_val = param.data.max().item()
        
        actor_stats.append({
            'name': name,
            'shape': tuple(param.shape),
            'mean_abs': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'num_params': param.numel()
        })
        
        print(f"\n[{name}]")
        print(f"  Shape: {param.shape}")
        print(f"  평균 절댓값: {mean:.6f}")
        print(f"  표준편차: {std:.6f}")
        print(f"  범위: [{min_val:.6f}, {max_val:.6f}]")
        print(f"  파라미터 수: {param.numel():,}")

# Critic 네트워크 가중치 분석
print(f"\n{'='*80}")
print("Critic 네트워크 가중치 분석")
print("="*80)

critic_stats = []
for name, param in agent.critic.named_parameters():
    if 'weight' in name:
        mean = param.data.abs().mean().item()
        std = param.data.std().item()
        min_val = param.data.min().item()
        max_val = param.data.max().item()
        
        critic_stats.append({
            'name': name,
            'shape': tuple(param.shape),
            'mean_abs': mean,
            'std': std,
            'min': min_val,
            'max': max_val,
            'num_params': param.numel()
        })
        
        print(f"\n[{name}]")
        print(f"  Shape: {param.shape}")
        print(f"  평균 절댓값: {mean:.6f}")
        print(f"  표준편차: {std:.6f}")
        print(f"  범위: [{min_val:.6f}, {max_val:.6f}]")
        print(f"  파라미터 수: {param.numel():,}")

# 노이즈 크기 분석
print(f"\n{'='*80}")
print("변이 노이즈 크기 분석")
print("="*80)

mutation_scale = cfg.Config.MUTATION_SCALE
print(f"\n현재 설정: mutation_scale = {mutation_scale}")
print(f"노이즈 분포: N(0, {mutation_scale}²) = N(0, {mutation_scale**2:.6f})")

# 노이즈 샘플 생성
sample_size = 10000
noise_samples = torch.randn(sample_size) * mutation_scale

print(f"\n[노이즈 샘플 통계] (n={sample_size:,})")
print(f"  평균 절댓값: {noise_samples.abs().mean().item():.6f}")
print(f"  표준편차: {noise_samples.std().item():.6f}")
print(f"  범위: [{noise_samples.min().item():.6f}, {noise_samples.max().item():.6f}]")

# 이론적 기댓값
theoretical_mean = mutation_scale * np.sqrt(2/np.pi)  # E[|N(0,σ²)|] = σ√(2/π)
print(f"  이론적 E[|noise|]: {theoretical_mean:.6f}")

# 가중치 대비 노이즈 비율
print(f"\n{'='*80}")
print("가중치 대비 노이즈 비율 (Actor)")
print("="*80)

for stat in actor_stats:
    ratio = mutation_scale / stat['mean_abs']
    print(f"\n[{stat['name']}]")
    print(f"  가중치 평균 절댓값: {stat['mean_abs']:.6f}")
    print(f"  노이즈 표준편차:   {mutation_scale:.6f}")
    print(f"  비율 (noise/weight): {ratio:.2%}")
    
    if ratio < 0.01:
        print(f"  → 매우 작은 변이 (< 1%)")
    elif ratio < 0.05:
        print(f"  → 작은 변이 (1-5%)")
    elif ratio < 0.1:
        print(f"  → 중간 변이 (5-10%)")
    elif ratio < 0.2:
        print(f"  → 큰 변이 (10-20%)")
    else:
        print(f"  → 매우 큰 변이 (> 20%)")

print(f"\n{'='*80}")
print("가중치 대비 노이즈 비율 (Critic)")
print("="*80)

critic_mutation_scale = mutation_scale * 0.5  # Critic은 절반
for stat in critic_stats:
    ratio = critic_mutation_scale / stat['mean_abs']
    print(f"\n[{stat['name']}]")
    print(f"  가중치 평균 절댓값: {stat['mean_abs']:.6f}")
    print(f"  노이즈 표준편차:   {critic_mutation_scale:.6f}")
    print(f"  비율 (noise/weight): {ratio:.2%}")
    
    if ratio < 0.01:
        print(f"  → 매우 작은 변이 (< 1%)")
    elif ratio < 0.05:
        print(f"  → 작은 변이 (1-5%)")
    elif ratio < 0.1:
        print(f"  → 중간 변이 (5-10%)")
    elif ratio < 0.2:
        print(f"  → 큰 변이 (10-20%)")
    else:
        print(f"  → 매우 큰 변이 (> 20%)")

# 전체 요약
print(f"\n{'='*80}")
print("전체 요약")
print("="*80)

# Actor
actor_mean_weight = np.mean([s['mean_abs'] for s in actor_stats])
actor_noise_ratio = mutation_scale / actor_mean_weight

print(f"\n[Actor]")
print(f"  평균 가중치 절댓값: {actor_mean_weight:.6f}")
print(f"  노이즈 표준편차:    {mutation_scale:.6f}")
print(f"  전체 비율:          {actor_noise_ratio:.2%}")

# Critic
critic_mean_weight = np.mean([s['mean_abs'] for s in critic_stats])
critic_noise_ratio = critic_mutation_scale / critic_mean_weight

print(f"\n[Critic]")
print(f"  평균 가중치 절댓값: {critic_mean_weight:.6f}")
print(f"  노이즈 표준편차:    {critic_mutation_scale:.6f}")
print(f"  전체 비율:          {critic_noise_ratio:.2%}")

# 권장사항
print(f"\n{'='*80}")
print("권장사항")
print("="*80)

print(f"\n일반적인 진화 알고리즘 가이드라인:")
print(f"  - 작은 변이 (1-5%): 수렴 단계, 미세 조정")
print(f"  - 중간 변이 (5-15%): 균형적 탐색")
print(f"  - 큰 변이 (15-30%): 초기 탐색, 다양성 증가")
print(f"  - 매우 큰 변이 (>30%): 과도할 수 있음, 수렴 방해")

if actor_noise_ratio < 0.05:
    print(f"\n현재 Actor 변이: {actor_noise_ratio:.2%} (작은 변이)")
    print(f"  → 미세 조정에 적합, 수렴이 느릴 수 있음")
elif actor_noise_ratio < 0.15:
    print(f"\n현재 Actor 변이: {actor_noise_ratio:.2%} (중간 변이)")
    print(f"  → 균형적인 설정 ✓")
else:
    print(f"\n현재 Actor 변이: {actor_noise_ratio:.2%} (큰 변이)")
    print(f"  → 초기 탐색에 적합, 수렴 후 감소 권장")

print(f"\n현재 설정 (mutation_scale={mutation_scale}):")
if actor_noise_ratio < 0.05:
    print(f"  - 유지: 안정적 학습, 느린 탐색")
    print(f"  - 증가 옵션: {mutation_scale*2:.3f} (2배) → {actor_noise_ratio*2:.2%}")
elif actor_noise_ratio < 0.15:
    print(f"  - 적절한 범위 ✓")
else:
    print(f"  - 감소 옵션: {mutation_scale*0.5:.3f} (1/2) → {actor_noise_ratio*0.5:.2%}")

print(f"\n{'='*80}")

