"""
GA-MARL 시스템 학습 (Rolling Window)

학습: 2021-2023 (분기별 Rolling Window, 과적합 방지)
학습 완료 후 모델 자동 저장: models/best_system/
"""

import os
from evolution.ga_trainer import GATrainer
from config import config
from select_universe import load_universe

print("="*80)
print("GA-MARL 시스템 학습 (Rolling Window)")
print("="*80)

# ========================================
# 종목 Universe 로드
# ========================================
print(f"\n[종목 Universe 로드]")
print("-" * 80)

try:
    universe_data = load_universe("data/selected_tickers.pkl")
    tickers = universe_data['tickers']
    print(f"✅ 종목 리스트 로드 완료: {len(tickers)}개")
    print(f"   선정 일시: {universe_data['selection_date']}")
    print(f"   전체 기간: {universe_data['full_start_date']} ~ {universe_data['full_end_date']}")
    print(f"   상위 5개: {', '.join(tickers[:5])}")
except FileNotFoundError as e:
    print(f"❌ 종목 Universe 파일이 없습니다!")
    print(f"   먼저 'python trading_marl_ga/select_universe.py'를 실행하세요.")
    exit(1)

print("-" * 80)

# 설정
POPULATION_SIZE = config.POPULATION_SIZE
N_GENERATIONS = config.N_GENERATIONS
RL_UPDATES = config.RL_UPDATES

print(f"\n[설정]")
print(f"  학습 기간: 2021-2023 (분기별 Rolling Window)")
print(f"  세대 수: {N_GENERATIONS}세대")
print(f"  Population: {POPULATION_SIZE}개 EA 팀 + 1개 MARL 팀")
print(f"  RL 업데이트: {RL_UPDATES}회/세대")
print("="*80)

# ========================================
# 학습 실행
# ========================================
print(f"\n{'='*80}")
print(f"[학습 시작]")
print(f"{'='*80}")

trainer = GATrainer(
    population_size=POPULATION_SIZE,
    n_generations=N_GENERATIONS,
    child_mutation_prob=config.CHILD_MUTATION_PROB,
    param_mutation_prob=config.PARAM_MUTATION_PROB,
    mutation_scale_ratio=config.MUTATION_SCALE_RATIO,
    tickers=tickers  # 선정된 종목 리스트 전달
)

best_system, fitness_history = trainer.train(
    rl_updates=RL_UPDATES
)

# ========================================
# 학습 결과 요약
# ========================================
print(f"\n{'='*80}")
print(f"[학습 완료]")
print(f"{'='*80}")
print(f"학습 기간: 2021-2023 ({N_GENERATIONS}개 분기)")
print(f"역대 최고 Fitness: {trainer.best_fitness:.4f}")

# 초기 vs 최종 비교
if len(fitness_history) >= 2:
    first_avg = fitness_history[0]['mean_fitness']
    last_avg = fitness_history[-1]['mean_fitness']
    improvement = ((last_avg - first_avg) / abs(first_avg) * 100) if first_avg != 0 else 0
    print(f"\n1세대 평균 Fitness: {first_avg:.4f}")
    print(f"{N_GENERATIONS}세대 평균 Fitness: {last_avg:.4f}")
    print(f"개선율: {improvement:+.2f}%")

# ========================================
# 학습 곡선
# ========================================
print(f"\n{'='*80}")
print(f"학습 곡선 (세대별 Fitness)")
print(f"{'='*80}")

for i, stats in enumerate(fitness_history, 1):
    year = 2021 + ((i - 1) // 4)
    quarter = ((i - 1) % 4) + 1
    period = f"{year}-Q{quarter}"
    bar_length = int(abs(stats['mean_fitness']) * 2)
    bar = "█" * min(bar_length, 50) if bar_length > 0 else ""
    print(f"세대 {i:2d} ({period}): {stats['mean_fitness']:7.4f} {bar}")

# ========================================
# 모델 저장 (필수)
# ========================================
print(f"\n{'='*80}")
print(f"[모델 저장]")
print(f"{'='*80}")

MODEL_DIR = "models/best_system"
os.makedirs(MODEL_DIR, exist_ok=True)

# best_system 저장
best_system.save(MODEL_DIR)
print(f"[OK] 최고 시스템 저장: {MODEL_DIR}/")

# fitness_history 저장 (시각화용)
import pickle
history_path = "models/fitness_history.pkl"
with open(history_path, 'wb') as f:
    pickle.dump(fitness_history, f)
print(f"[OK] 학습 곡선 저장: {history_path}")

# 메타데이터 저장
metadata = {
    'n_generations': N_GENERATIONS,
    'population_size': POPULATION_SIZE,
    'best_fitness': trainer.best_fitness,
    'final_mean_fitness': fitness_history[-1]['mean_fitness'] if fitness_history else 0,
    'train_period': '2021-2023',
    'tickers': tickers,  # 학습에 사용된 종목 리스트
    'num_tickers': len(tickers),
}
metadata_path = "models/metadata.pkl"
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)
print(f"[OK] 메타데이터 저장: {metadata_path}")

print(f"\n{'='*80}")
print(f"[완료] 학습 종료")
print(f"{'='*80}")
print(f"다음 단계: python test.py 실행하여 성능 평가")
print(f"{'='*80}\n")

