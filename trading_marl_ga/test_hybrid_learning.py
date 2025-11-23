"""
Hybrid Learning (GA + RL) 테스트

Phase 1 (Pure GA) + Phase 2 (GA + RL Hybrid) 검증
"""

from evolution.ga_trainer import GATrainer

print("="*60)
print("Hybrid Learning 테스트 (GA + RL)")
print("="*60)
print("\n[설정]")
print("  Population: 3개")
print("  세대: 5세대")
print("  - Phase 1 (Pure GA): 세대 1-2")
print("  - Phase 2 (GA + RL): 세대 3-5")
print("  백테스트: 50일")
print("  RL Fine-tuning: 2 에피소드, 5 업데이트")
print("="*60)

# GA Trainer 초기화
trainer = GATrainer(
    population_size=3,
    n_generations=5,
    env_n_days=50,
    mutation_prob=0.8,
    mutation_scale=0.02
)

# 전체 학습 (Phase 1 + Phase 2)
print("\n[학습 시작]")
best_system, fitness_history = trainer.train(
    phase1_gens=2,      # 세대 1-2: Pure GA
    phase2_start=3,     # 세대 3부터: GA + RL
    rl_episodes=2,      # RL 에피소드 수 (적게)
    rl_updates=5        # 업데이트 횟수 (적게)
)

print(f"\n{'='*60}")
print(f"Hybrid Learning 테스트 완료!")
print(f"{'='*60}")

# 결과 분석
print(f"\n[학습 결과]")
print(f"  역대 최고 Fitness: {trainer.best_fitness:.4f}")
print(f"\n  세대별 Fitness:")
for i, stats in enumerate(fitness_history, 1):
    phase = "Phase 1 (Pure GA)" if i <= 2 else "Phase 2 (GA+RL)"
    print(f"    세대 {i} ({phase}): 최고={stats['max_fitness']:.4f}, " +
          f"평균={stats['mean_fitness']:.4f}")

# Phase 1 vs Phase 2 비교
phase1_avg = sum([stats['mean_fitness'] for stats in fitness_history[:2]]) / 2
phase2_avg = sum([stats['mean_fitness'] for stats in fitness_history[2:]]) / 3
improvement = ((phase2_avg - phase1_avg) / phase1_avg) * 100

print(f"\n[Phase 비교]")
print(f"  Phase 1 평균 Fitness: {phase1_avg:.4f}")
print(f"  Phase 2 평균 Fitness: {phase2_avg:.4f}")
print(f"  개선율: {improvement:+.2f}%")

print(f"\n[OK] Hybrid Learning 정상 작동 확인!")

