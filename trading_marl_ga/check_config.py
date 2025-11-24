"""
Config 설정 확인 스크립트
"""

from config import config

print("="*60)
print("현재 Config 설정")
print("="*60)

print(f"\n[환경 설정]")
print(f"  종목 수 (N_STOCKS):        {config.N_STOCKS}")
print(f"  초기 자본:                 {config.INITIAL_CAPITAL:,}원")
print(f"  거래 비용:                 {config.TRANSACTION_COST*100:.2f}%")
rebalance_label = {1: '매일', 5: '주간', 20: '월간'}.get(config.REBALANCE_PERIOD, f'{config.REBALANCE_PERIOD}일')
print(f"  리밸런싱 주기:             {config.REBALANCE_PERIOD}거래일 ({rebalance_label})")

print(f"\n[관측 차원]")
print(f"  Value 관측:                {config.VALUE_OBS_DIM}")
print(f"  Quality 관측:              {config.QUALITY_OBS_DIM}")
print(f"  Portfolio 관측:            {config.PORTFOLIO_OBS_DIM}")
print(f"  Hedging 관측:              {config.HEDGING_OBS_DIM}")
print(f"  Action 차원:               {config.ACTION_DIM}")

print(f"\n[네트워크 구조]")
print(f"  Hidden 차원:               {config.HIDDEN_DIM}")

print(f"\n[학습 하이퍼파라미터]")
print(f"  Replay Buffer 크기:        {config.BUFFER_CAPACITY:,}")
print(f"  Batch 크기:                {config.BATCH_SIZE}")
print(f"  RL 학습 최소 버퍼:         {config.MIN_BUFFER_FOR_RL}")
print(f"  Learning Rate (Actor):     {config.LEARNING_RATE_ACTOR}")
print(f"  Learning Rate (Critic):    {config.LEARNING_RATE_CRITIC}")

print(f"\n[GA 설정]")
print(f"  Population 크기:           {config.POPULATION_SIZE}")
print(f"  총 팀 수:                  {config.POPULATION_SIZE + 1} (EA + MARL)")
print(f"  Elite 비율:                {config.ELITE_FRACTION*100:.0f}% ({max(1, int(config.POPULATION_SIZE * config.ELITE_FRACTION))}개 보존)")
print(f"  세대 수:                   {config.N_GENERATIONS}")
print(f"  돌연변이 확률:             {config.MUTATION_PROB}")
print(f"  상대적 노이즈 비율:        {config.MUTATION_SCALE_RATIO*100:.1f}% (가중치 대비)")

print(f"\n[백테스트 설정]")
print(f"  데이터 소스:               {config.DATA_SOURCE}")

print(f"\n[RL 설정]")
print(f"  RL 업데이트 횟수:          {config.RL_UPDATES}")
print(f"  Gamma:                     {config.GAMMA}")
print(f"  Tau:                       {config.TAU}")

print(f"\n[환경 최적화]")
print(f"  실행 환경:                 {'Colab' if config.ENV['is_colab'] else 'Local'}")
print(f"  GPU:                       {'Yes' if config.ENV['has_gpu'] else 'CPU Only'}")
if config.ENV['has_gpu']:
    print(f"  GPU 메모리:                {config.ENV['gpu_memory_gb']:.1f} GB")
print(f"  혼합 정밀도 (FP16):        {'활성화' if config.USE_AMP else '비활성화'}")
print(f"  병렬 백테스트:             {'활성화' if config.USE_PARALLEL_BACKTEST else '비활성화'}")

print(f"\n{'='*60}")
print("예상 계산량 (세대당)")
print("="*60)

# 네트워크 파라미터 수 추정
actor_params = config.VALUE_OBS_DIM * config.HIDDEN_DIM + config.HIDDEN_DIM**2 + config.HIDDEN_DIM * config.ACTION_DIM
critic_params = config.VALUE_OBS_DIM * config.HIDDEN_DIM + config.HIDDEN_DIM**2 + config.HIDDEN_DIM

print(f"  Actor 파라미터 (Value):    ~{actor_params:,}")
print(f"  Critic 파라미터 (Value):   ~{critic_params:,}")
print(f"  총 팀 수:                  {config.POPULATION_SIZE + 1}")
print(f"  세대당 RL 업데이트:        {config.RL_UPDATES} updates")

print(f"\n{'='*60}\n")

