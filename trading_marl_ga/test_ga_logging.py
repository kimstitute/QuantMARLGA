"""
GA 진화 과정 상세 로그 테스트

Selection, Crossover, Mutation이 어떻게 일어나는지 상세 로그 확인
"""

from evolution.ga_trainer import GATrainer

print("="*80)
print("GA 진화 과정 상세 로그 테스트")
print("="*80)

# 작은 규모로 테스트
trainer = GATrainer(
    population_size=3,
    n_generations=2,
    env_n_days=10,
    mutation_prob=0.9,  # 높은 확률로 변이 발생
    mutation_scale_ratio=0.05
)

print(f"\n[설정]")
print(f"  Population: {trainer.population_size}개")
print(f"  세대: {trainer.n_generations}세대")
print(f"  백테스트: {trainer.env_n_days}일")
print(f"  변이 확률: {trainer.mutation_prob*100:.0f}%")
print(f"  상대적 노이즈 비율: {trainer.mutation_scale_ratio*100:.1f}%")

print(f"\n{'='*80}")
print("학습 시작 (상세 로그 활성화)")
print("="*80)

# 1세대만 실행해서 로그 확인
best_system, fitness_history = trainer.train(
    rl_updates=10
)

print(f"\n{'='*80}")
print("테스트 완료")
print("="*80)
print(f"최고 Fitness: {trainer.best_fitness:.4f}")
print(f"전체 Fitness 히스토리: {len(fitness_history)}세대")

