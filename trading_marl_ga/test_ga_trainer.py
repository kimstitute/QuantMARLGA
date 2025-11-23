"""
GA Trainer 테스트

작은 규모로 GA Trainer가 정상 작동하는지 검증
- Population: 3개 (빠른 테스트)
- 세대: 3세대 (Phase 1만)
- 백테스트: 30일
"""

from evolution.ga_trainer import GATrainer

print("="*60)
print("GA Trainer 테스트")
print("="*60)
print("\n[설정]")
print("  Population: 3개")
print("  세대: 3세대 (Pure GA만)")
print("  백테스트: 30일")
print("="*60)

# GA Trainer 초기화
trainer = GATrainer(
    population_size=3,
    n_generations=3,
    env_n_days=30,
    mutation_prob=0.8,
    mutation_scale=0.02
)

print("\n[테스트 1] 초기 Population Fitness 평가")
print("-"*60)
fitnesses = trainer.evaluate_population(verbose=False)
print(f"\n[OK] 초기 Fitness:")
for i, fitness in enumerate(fitnesses):
    print(f"  시스템 {i}: {fitness:.4f}")

print(f"\n[테스트 2] 1세대 진화")
print("-"*60)
stats = trainer.evolve_generation()
print(f"\n[OK] 진화 완료:")
print(f"  최고: {stats['max_fitness']:.4f}")
print(f"  평균: {stats['mean_fitness']:.4f}")
print(f"  최저: {stats['min_fitness']:.4f}")

print(f"\n[테스트 3] 진화 후 Fitness 재평가")
print("-"*60)
new_fitnesses = trainer.evaluate_population(verbose=False)
print(f"\n[OK] 진화 후 Fitness:")
for i, fitness in enumerate(new_fitnesses):
    print(f"  시스템 {i}: {fitness:.4f}")

print(f"\n[테스트 4] Agent-level Crossover")
print("-"*60)
parent1 = trainer.population[0]
parent2 = trainer.population[1]
child = trainer.agent_level_crossover(parent1, parent2)
print(f"[OK] 자식 생성 완료")
print(f"  부모1 Fitness: {parent1.fitness:.4f}")
print(f"  부모2 Fitness: {parent2.fitness:.4f}")

# 자식 평가
child_fitness = trainer.evaluate_fitness(child)
print(f"  자식 Fitness:  {child_fitness:.4f}")

print(f"\n[테스트 5] 전체 학습 (3세대)")
print("-"*60)

# 새로운 Trainer로 전체 학습
trainer2 = GATrainer(
    population_size=3,
    n_generations=3,
    env_n_days=30
)

best_system, fitness_history = trainer2.train(
    phase1_gens=3,  # 3세대 모두 Phase 1
    phase2_start=4
)

print(f"\n{'='*60}")
print(f"모든 테스트 통과!")
print(f"{'='*60}")
print(f"\n[결과]")
print(f"  역대 최고 Fitness: {trainer2.best_fitness:.4f}")
print(f"  세대별 Fitness:")
for i, stats in enumerate(fitness_history, 1):
    print(f"    세대 {i}: 최고={stats['max_fitness']:.4f}, " +
          f"평균={stats['mean_fitness']:.4f}")

print(f"\n[OK] GA Trainer 정상 작동 확인!")

