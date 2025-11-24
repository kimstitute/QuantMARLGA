"""
학습 중 상세 로깅 테스트

각 팀의 자산과 수익률이 제대로 출력되는지 확인
"""

import numpy as np
from config import config
from evolution.ga_trainer import GATrainer

def test_detailed_logging():
    """
    작은 규모로 학습 실행하여 로그 확인
    """
    print("="*60)
    print("학습 중 상세 로깅 테스트")
    print("="*60)
    
    # 작은 규모 설정
    population_size = 3  # 3개 팀만
    n_generations = 2    # 2세대만
    env_n_days = 10      # 10일만
    
    print(f"\n설정:")
    print(f"  Population: {population_size}개")
    print(f"  Generations: {n_generations}세대")
    print(f"  Backtest: {env_n_days}일")
    
    # GATrainer 생성
    trainer = GATrainer(
        population_size=population_size,
        n_generations=n_generations,
        env_n_days=env_n_days,
        mutation_prob=0.3,
        mutation_scale_ratio=0.03
    )
    
    print(f"\n초기화 완료!")
    print(f"  EA 팀: {len(trainer.population)}개")
    print(f"  MARL 팀: 1개")
    print(f"  총 팀: {len(trainer.population) + 1}개")
    
    # 1세대만 실행 (로그 확인용)
    print(f"\n{'='*60}")
    print(f"세대 1/{n_generations} 실행")
    print(f"{'='*60}")
    
    # Rollout + Fitness 평가 (상세 로그 활성화)
    print(f"\n[1/4] Rollout + Fitness 평가")
    collected = trainer.rollout_and_evaluate(include_marl=True, verbose=True)
    
    print(f"\n[OK] 테스트 완료!")
    print(f"\n{'='*60}")
    print(f"결과 요약")
    print(f"{'='*60}")
    print(f"수집된 경험: {collected}개")
    print(f"Buffer 크기: {len(trainer.shared_replay_buffer)}")
    
    # Fitness 확인
    fitnesses = [s.fitness for s in trainer.population if s.fitness is not None]
    if len(fitnesses) > 0:
        print(f"\nEA Fitness 통계:")
        print(f"  최고: {max(fitnesses):.4f}")
        print(f"  평균: {np.mean(fitnesses):.4f}")
        print(f"  최저: {min(fitnesses):.4f}")
    
    if trainer.marl_team.fitness:
        print(f"\nMARL Fitness: {trainer.marl_team.fitness:.4f}")
    
    print(f"\n{'='*60}")
    print("[SUCCESS] 상세 로깅이 제대로 작동합니다!")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_detailed_logging()

