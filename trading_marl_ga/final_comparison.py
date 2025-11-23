"""
최종 비교: QuantMARLGA vs 벤치마크

GA-MARL Hybrid (처음부터 GA + RL 동시) vs Buy & Hold, Random Agent, KOSPI Index
"""

from evolution.ga_trainer import GATrainer
from benchmarks import BuyAndHold, RandomAgent, run_benchmark, run_kospi_index_benchmark, print_comparison_table
from environment.backtest_env import BacktestEnv
from config import config

print("="*80)
print("최종 비교: QuantMARLGA vs 벤치마크")
print("="*80)

# 설정 (config.py에서 가져오기)
N_DAYS = 245  # 전체 2023년 데이터 (약 245거래일)
POPULATION_SIZE = config.POPULATION_SIZE
N_GENERATIONS = config.N_GENERATIONS
RL_UPDATES = config.RL_UPDATES

print(f"\n[설정]")
print(f"  백테스트 기간: {N_DAYS}일")
print(f"  Population: {POPULATION_SIZE}개 EA 팀 + 1개 MARL 팀")
print(f"  세대: {N_GENERATIONS}세대 (처음부터 GA+RL 동시)")
print(f"  RL 업데이트: {RL_UPDATES}회/세대")
print("="*80)

# ========================================
# 1. GA-MARL 학습
# ========================================
print(f"\n{'='*80}")
print(f"[1/2] GA-MARL 시스템 학습")
print(f"{'='*80}")

trainer = GATrainer(
    population_size=POPULATION_SIZE,
    n_generations=N_GENERATIONS,
    env_n_days=N_DAYS,
    mutation_prob=config.MUTATION_PROB,
    mutation_alpha=config.MUTATION_ALPHA
)

best_system, fitness_history = trainer.train(
    rl_updates=RL_UPDATES  # config.py에서 가져옴
)

# 학습 결과 요약
print(f"\n[학습 완료]")
print(f"  역대 최고 Fitness: {trainer.best_fitness:.4f}")

# 초기 vs 최종 비교
if len(fitness_history) >= 2:
    first_avg = fitness_history[0]['mean_fitness']
    last_avg = fitness_history[-1]['mean_fitness']
    print(f"  1세대 평균 Fitness: {first_avg:.4f}")
    print(f"  {N_GENERATIONS}세대 평균 Fitness: {last_avg:.4f}")
    print(f"  개선율: {((last_avg - first_avg) / first_avg * 100):+.2f}%")

# ========================================
# 2. 최고 시스템 성과 평가
# ========================================
print(f"\n{'='*80}")
print(f"[2/2] 최종 성과 비교")
print(f"{'='*80}")

# 2.1 GA-MARL 최고 시스템 평가
print(f"\n[GA-MARL 최고 시스템 평가]")
env = BacktestEnv(n_days=N_DAYS)
obs = env.reset()
done = False

while not done:
    actions = best_system.act(obs)
    next_obs, rewards, done, info = env.step(actions)
    if not done:
        obs = next_obs

ga_marl_metrics = env.get_performance_metrics()

# 2.2 벤치마크 평가
print(f"\n[벤치마크 평가]")

results = {"GA-MARL (Best)": ga_marl_metrics}

# 포트폴리오 전략들
benchmarks = [
    ("Buy & Hold", BuyAndHold(config.N_STOCKS)),
    ("Random Agent", RandomAgent(config.N_STOCKS)),
]

for name, strategy in benchmarks:
    env = BacktestEnv(n_days=N_DAYS)
    metrics = run_benchmark(strategy, env, verbose=False)
    results[name] = metrics
    print(f"  {name}: 샤프 {metrics['sharpe_ratio']:.3f}, 수익률 {metrics['total_return']*100:.2f}%")

# KOSPI 지수 (실제 시장)
print(f"  KOSPI Index: 계산 중...")
kospi_metrics = run_kospi_index_benchmark(
    start_date=config.DATA_START_DATE,
    end_date=config.DATA_END_DATE
)
if kospi_metrics:
    results["KOSPI Index"] = kospi_metrics
    print(f"  KOSPI Index: 샤프 {kospi_metrics['sharpe_ratio']:.3f}, 수익률 {kospi_metrics['total_return']*100:.2f}%")

# ========================================
# 3. 비교 테이블
# ========================================
print(f"\n{'='*80}")
print(f"최종 성과 비교")
print(f"{'='*80}")

# 상세 테이블
print_comparison_table(results)

# ========================================
# 4. 승자 발표
# ========================================
best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
best_return = max(results.items(), key=lambda x: x[1]['total_return'])

print(f"\n{'='*80}")
print(f"최종 결과")
print(f"{'='*80}")

if best_sharpe[0] == "GA-MARL (Best)":
    print(f"[우승] GA-MARL 시스템이 샤프 비율에서 우승!")
    print(f"  GA-MARL: {best_sharpe[1]['sharpe_ratio']:.3f}")
    improvement = ((best_sharpe[1]['sharpe_ratio'] - results['Buy & Hold']['sharpe_ratio']) / 
                   results['Buy & Hold']['sharpe_ratio'] * 100)
    print(f"  Buy & Hold 대비 개선: {improvement:+.2f}%")
else:
    print(f"[결과] {best_sharpe[0]} 시스템이 샤프 비율 최고")
    print(f"  {best_sharpe[0]}: {best_sharpe[1]['sharpe_ratio']:.3f}")
    print(f"  GA-MARL: {results['GA-MARL (Best)']['sharpe_ratio']:.3f}")

print(f"{'='*80}\n")

# ========================================
# 5. 학습 곡선
# ========================================
print(f"\n{'='*80}")
print(f"학습 곡선 (세대별 Fitness)")
print(f"{'='*80}")

for i, stats in enumerate(fitness_history, 1):
    phase = "Phase 1" if i <= 3 else "Phase 2"
    bar_length = int(stats['mean_fitness'] * 10)
    bar = "█" * bar_length
    print(f"  세대 {i:2d} ({phase}): {stats['mean_fitness']:.4f} {bar}")

print(f"{'='*80}\n")

print(f"[완료] 모든 비교 완료!")

