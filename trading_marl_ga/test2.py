"""
학습된 GA-MARL 시스템 테스트 (학습 종목과 동일)

학습 시 사용한 30개 종목으로 테스트
결측값은 forward fill 처리
"""

import os
import pickle
import numpy as np
from agents.multi_agent_system import MultiAgentSystem
from benchmarks import BuyAndHold, RandomAgent, run_benchmark, run_kospi_index_benchmark, print_comparison_table
from environment.backtest_env2 import BacktestEnv2
from config import config

print("="*80)
print("GA-MARL 시스템 테스트 (학습 종목과 동일)")
print("="*80)

# 학습 시 사용한 종목 30개 (로그 기반)
TRAIN_TICKERS = [
    '005930',  # 1. 삼성전자
    '000660',  # 2. SK하이닉스
    '207940',  # 4. LG에너지솔루션
    '005380',  # 5. 현대차
    '105560',  # 7.
    '034020',  # 8.
    '012450',  # 9.
    '000270',  # 10.
    '068270',  # 11.
    '035420',  # 12.
    '055550',  # 13.
    '028260',  # 14.
    '042660',  # 16.
    '015760',  # 17.
    '009540',  # 18.
    '032830',  # 19.
    '267260',  # 20.
    '012330',  # 21.
    '035720',  # 22.
    '086790',  # 23.
    '051910',  # 24.
    '005490',  # 25.
    '006400',  # 26.
    '010140',  # 27.
    '000810',  # 28.
    '010130',  # 29.
    '138040',  # 30.
    '096770',  # 31.
    '064350',  # 32.
    '034730',  # 33.
]

# 설정
TEST_START = config.TEST_START
TEST_END = config.TEST_END
MODEL_DIR = "models/best_system"

print(f"\n[설정]")
print(f"  학습 종목: {len(TRAIN_TICKERS)}개 (학습 시와 동일)")
print(f"  테스트 기간: {TEST_START} ~ {TEST_END}")
print(f"  모델 경로: {MODEL_DIR}/")
print(f"  종목 예시: {TRAIN_TICKERS[:5]}")
print("="*80)

# ========================================
# 1. 모델 로드
# ========================================
print(f"\n{'='*80}")
print(f"[1/3] 모델 로드")
print(f"{'='*80}")

# 메타데이터 로드
metadata_path = "models/metadata.pkl"
if os.path.exists(metadata_path):
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    print(f"\n[학습 정보]")
    print(f"  학습 기간: {metadata['train_period']}")
    print(f"  세대 수: {metadata['n_generations']}")
    print(f"  Population: {metadata['population_size']}")
    print(f"  최고 Fitness: {metadata['best_fitness']:.4f}")
    print(f"  최종 평균 Fitness: {metadata['final_mean_fitness']:.4f}")

# 모델 로드
best_system = MultiAgentSystem()
best_system.load(MODEL_DIR)
print(f"\n[OK] 모델 로드 완료: {MODEL_DIR}/")

# ========================================
# 2. 테스트 환경 생성 및 평가
# ========================================
print(f"\n{'='*80}")
print(f"[2/3] 테스트 기간 성과 평가 ({TEST_START} ~ {TEST_END})")
print(f"{'='*80}")

# 테스트 환경 생성 (학습 종목 사용)
print(f"\n[테스트 환경 생성]")
test_env = BacktestEnv2(
    start_date=TEST_START, 
    end_date=TEST_END,
    tickers=TRAIN_TICKERS  # 학습 시와 동일한 종목!
)
print(f"  기간: {TEST_START} ~ {TEST_END}")
print(f"  종목: 학습 시와 동일 ({len(TRAIN_TICKERS)}개)")
print(f"  [OK] 데이터 로드 완료 (결측값 forward fill)")

# GA-MARL 최고 시스템 테스트
print(f"\n[GA-MARL 최고 시스템 테스트]")
obs = test_env.reset()
done = False
step = 0

while not done:
    actions = best_system.act(obs)
    next_obs, rewards, done, info = test_env.step(actions)
    if not done:
        obs = next_obs
    step += 1
    
    # 진행 상황 표시
    if step % 20 == 0:
        print(f"  진행: {step}/{len(test_env.trading_days)} 거래일")

ga_marl_metrics = test_env.get_performance_metrics()
print(f"[OK] GA-MARL 테스트 완료")

# 벤치마크 평가
print(f"\n[벤치마크 평가]")
results = {"GA-MARL (Best)": ga_marl_metrics}

# 포트폴리오 전략들
benchmarks = [
    ("Buy & Hold", BuyAndHold(config.N_STOCKS)),
    ("Random Agent", RandomAgent(config.N_STOCKS)),
]

for name, strategy in benchmarks:
    metrics = run_benchmark(strategy, test_env, verbose=False)
    results[name] = metrics
    print(f"  {name}: 샤프 {metrics['sharpe_ratio']:.3f}, 수익률 {metrics['total_return']*100:.2f}%")

# KOSPI 지수
print(f"  KOSPI Index: 계산 중...")
kospi_metrics = run_kospi_index_benchmark(
    start_date=TEST_START,
    end_date=TEST_END
)
if kospi_metrics:
    results["KOSPI Index"] = kospi_metrics
    print(f"  KOSPI Index: 샤프 {kospi_metrics['sharpe_ratio']:.3f}, 수익률 {kospi_metrics['total_return']*100:.2f}%")

# ========================================
# 3. 비교 테이블 및 최종 결과
# ========================================
print(f"\n{'='*80}")
print(f"[3/3] 최종 성과 비교 (테스트 기간: {TEST_START} ~ {TEST_END})")
print(f"{'='*80}")

# 상세 테이블
print_comparison_table(results)

# 승자 발표
best_sharpe = max(results.items(), key=lambda x: x[1]['sharpe_ratio'])
best_return = max(results.items(), key=lambda x: x[1]['total_return'])

print(f"\n{'='*80}")
print(f"최종 결과")
print(f"{'='*80}")

if best_sharpe[0] == "GA-MARL (Best)":
    print(f"🏆 [우승] GA-MARL 시스템이 샤프 비율에서 우승!")
    print(f"  GA-MARL: {best_sharpe[1]['sharpe_ratio']:.3f}")
    if 'Buy & Hold' in results:
        improvement = ((best_sharpe[1]['sharpe_ratio'] - results['Buy & Hold']['sharpe_ratio']) / 
                       abs(results['Buy & Hold']['sharpe_ratio']) * 100)
        print(f"  Buy & Hold 대비 개선: {improvement:+.2f}%")
else:
    print(f"[결과] {best_sharpe[0]} 시스템이 샤프 비율 최고")
    print(f"  {best_sharpe[0]}: {best_sharpe[1]['sharpe_ratio']:.3f}")
    print(f"  GA-MARL: {results['GA-MARL (Best)']['sharpe_ratio']:.3f}")

print(f"\n[중요] 학습 종목과 동일한 종목으로 테스트")
print(f"  학습: 2021-2023 (30개 종목)")
print(f"  테스트: {TEST_START} ~ {TEST_END} (동일 30개 종목, 결측값 forward fill)")

# ========================================
# 4. 결과 저장
# ========================================
print(f"\n{'='*80}")
print(f"[결과 저장]")
print(f"{'='*80}")

results_data = {
    'results': results,
    'train_tickers': TRAIN_TICKERS,
    'test_period': f"{TEST_START} ~ {TEST_END}",
}

results_path = "models/test2_results.pkl"
with open(results_path, 'wb') as f:
    pickle.dump(results_data, f)
print(f"[OK] 테스트 결과 저장: {results_path}")

print(f"\n{'='*80}")
print(f"[완료] 테스트 종료")
print(f"{'='*80}\n")

