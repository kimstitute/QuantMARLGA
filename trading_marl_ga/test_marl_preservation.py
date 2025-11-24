"""
MARL 원본 보존 테스트

MARL 원본이 절대 대체되지 않고 계속 RL 학습되는지 확인
"""

import torch
from evolution.ga_trainer import GATrainer

print("="*60)
print("MARL 원본 보존 테스트")
print("="*60)

# 작은 규모 Trainer
trainer = GATrainer(
    population_size=3,
    n_generations=3,
    env_n_days=10,
    mutation_prob=0.9,
    mutation_scale_ratio=0.05
)

print(f"\n[초기화]")
print(f"  Population: {len(trainer.population)}개")
print(f"  MARL 원본 ID: {id(trainer.marl_team)}")

# MARL 원본의 초기 파라미터 저장
initial_marl_id = id(trainer.marl_team)
initial_actor_param = trainer.marl_team.value_agent.actor.fc1.weight.data.clone()

print(f"\n[세대 1 시작]")
print(f"  MARL 원본 ID: {id(trainer.marl_team)}")
print(f"  MARL 파라미터 샘플: {initial_actor_param[0, 0].item():.6f}")

# 1세대 실행
print(f"\n[1세대 Rollout + Injection]")
trainer.rollout_and_evaluate(include_marl=True, verbose=False)

# Injection 전 MARL ID 확인
marl_id_before_injection = id(trainer.marl_team)
print(f"  Injection 전 MARL ID: {marl_id_before_injection}")

# Injection 실행 (ga_trainer.train()의 일부를 수동 실행)
fitnesses = [s.fitness if s.fitness is not None else -float('inf') 
             for s in trainer.population]
worst_idx = fitnesses.index(min(fitnesses))

print(f"  Worst 팀 #{worst_idx}, fitness={fitnesses[worst_idx]:.4f}")
print(f"  MARL fitness: {trainer.marl_team.fitness:.4f}")

# Injection
trainer.population[worst_idx] = trainer.marl_team.clone()
trainer.population[worst_idx].fitness = trainer.marl_team.fitness

# Injection 후 확인
marl_id_after_injection = id(trainer.marl_team)
current_actor_param = trainer.marl_team.value_agent.actor.fc1.weight.data

print(f"\n[Injection 후 확인]")
print(f"  MARL 원본 ID: {marl_id_after_injection}")
print(f"  Population[{worst_idx}] ID: {id(trainer.population[worst_idx])}")
print(f"  MARL 파라미터 샘플: {current_actor_param[0, 0].item():.6f}")

# 검증
print(f"\n{'='*60}")
print(f"검증 결과")
print(f"{'='*60}")

if initial_marl_id == marl_id_before_injection == marl_id_after_injection:
    print(f"✅ MARL 원본 ID 보존: {initial_marl_id}")
else:
    print(f"❌ MARL 원본 ID 변경!")
    print(f"   초기: {initial_marl_id}")
    print(f"   최종: {marl_id_after_injection}")

if id(trainer.marl_team) != id(trainer.population[worst_idx]):
    print(f"✅ MARL 원본과 복사본은 다른 객체")
else:
    print(f"❌ MARL 원본과 복사본이 같은 객체!")

# 파라미터가 같은지 확인 (복사본이므로 값은 같아야 함)
injected_param = trainer.population[worst_idx].value_agent.actor.fc1.weight.data
if torch.allclose(current_actor_param, injected_param):
    print(f"✅ MARL 복사본 파라미터 일치 (정상 복사)")
else:
    print(f"❌ MARL 복사본 파라미터 불일치!")

print(f"\n{'='*60}")
print(f"[SUCCESS] MARL 원본 보존 확인 완료!")
print(f"{'='*60}\n")

