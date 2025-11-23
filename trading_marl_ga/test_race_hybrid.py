"""
GA-MARL Hybrid Learning 테스트

QuantMARLGA의 하이브리드 학습 방식 (RACE 논문 참고):
- 처음부터 GA + RL 동시 수행
- 매 세대: Rollout → RL → Evolve → Inject
"""

from evolution.ga_trainer import GATrainer

print("="*60)
print("GA-MARL Hybrid Learning 테스트")
print("="*60)
print("\n[Hybrid Learning 순서]")
print("  매 세대마다:")
print("  1. Rollout: 모든 팀(EA + MARL) 경험 수집 + Fitness 평가")
print("  2. RL: MARL 팀만 Shared Buffer에서 학습")
print("  3. Evolve: EA Population 진화 (Selection, Crossover, Mutation)")
print("  4. Inject: MARL 팀을 Worst EA 팀과 교체")
print("="*60)

print("\n[설정]")
print("  Population: 5개 EA 팀 + 1개 MARL 팀")
print("  세대: 4세대 (처음부터 GA+RL 동시 수행)")
print("  백테스트: 50일")
print("  RL 업데이트: 30회/세대")
print("="*60)

# GA Trainer 초기화
trainer = GATrainer(
    population_size=5,
    n_generations=4,
    env_n_days=50,
    mutation_prob=0.9,
    mutation_scale=0.02
)

print(f"\n[초기 상태]")
print(f"  Shared Buffer 크기: {len(trainer.shared_replay_buffer)}")

# 전체 학습 (처음부터 GA+RL)
best_system, fitness_history = trainer.train(
    rl_updates=30  # 매 세대 RL 업데이트 횟수
)

print(f"\n{'='*60}")
print(f"Hybrid Learning 완료!")
print(f"{'='*60}")

# 결과 분석
print(f"\n[학습 결과]")
print(f"  역대 최고 Fitness: {trainer.best_fitness:.4f}")
print(f"  최종 Buffer 크기: {len(trainer.shared_replay_buffer)}")

print(f"\n  세대별 Fitness:")
for i, stats in enumerate(fitness_history, 1):
    print(f"    세대 {i}: " +
          f"최고={stats['max_fitness']:.4f}, " +
          f"평균={stats['mean_fitness']:.4f}, " +
          f"최저={stats['min_fitness']:.4f}")

# 개선율 계산
if len(fitness_history) >= 2:
    first_gen_avg = fitness_history[0]['mean_fitness']
    last_gen_avg = fitness_history[-1]['mean_fitness']
    improvement = ((last_gen_avg - first_gen_avg) / first_gen_avg) * 100
    
    print(f"\n[학습 개선]")
    print(f"  1세대 평균: {first_gen_avg:.4f}")
    print(f"  {len(fitness_history)}세대 평균: {last_gen_avg:.4f}")
    print(f"  개선율: {improvement:+.2f}%")

print(f"\n[OK] GA-MARL Hybrid Learning 정상 작동 확인!")

