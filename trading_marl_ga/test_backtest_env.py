"""
BacktestEnv 테스트

실제 데이터 및 랜덤 에이전트로 전체 파이프라인 검증
"""

import numpy as np
from environment.backtest_env import BacktestEnv
from agents.multi_agent_system import MultiAgentSystem
from config import config

print("="*60)
print("BacktestEnv 테스트 시작")
print("="*60)

# =================================================================
# Test 1: 환경 초기화 테스트 (실제 데이터)
# =================================================================
print("\n[Test 1] 환경 초기화 (실제 데이터)")
print("-"*60)

# Config를 실제 데이터 모드로 설정
original_data_source = config.DATA_SOURCE
config.DATA_SOURCE = "real"

try:
    env = BacktestEnv(n_days=50)  # 50일 백테스트
    print("[OK] 환경 초기화 성공")
    print(f"  - 종목 수: {env.n_stocks}")
    print(f"  - 백테스트 기간: {env.n_days}일")
    print(f"  - 초기 자본: {env.initial_capital:,.0f}원")
    
    if config.DATA_SOURCE == "real":
        print(f"  - 거래일: {len(env.trading_days)}일")
        print(f"  - 시작일: {env.trading_days[0]}")
        print(f"  - 종료일: {env.trading_days[min(env.n_days, len(env.trading_days)-1)]}")
except Exception as e:
    print(f"[ERROR] 환경 초기화 실패: {e}")
    import traceback
    traceback.print_exc()
    config.DATA_SOURCE = original_data_source
    exit(1)


# =================================================================
# Test 2: Reset 및 Observation 테스트
# =================================================================
print("\n[Test 2] Reset 및 Observation")
print("-"*60)

try:
    obs = env.reset()
    print("[OK] 환경 리셋 성공")
    print(f"  - 현재 날짜 인덱스: {env.current_day}")
    print(f"  - 현금: {env.cash:,.0f}원")
    print(f"  - 포지션: {env.positions.sum():.2f}주")
    
    print("\n[OK] 관측 구조:")
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            print(f"  - {key}: shape={value.shape}, mean={value.mean():.4f}, std={value.std():.4f}")
        else:
            print(f"  - {key}: {type(value)}")
    
    # 관측 차원 검증
    assert obs['value_obs'].shape == (config.VALUE_OBS_DIM,), f"Value obs 차원 오류: {obs['value_obs'].shape}"
    assert obs['quality_obs'].shape == (config.QUALITY_OBS_DIM,), f"Quality obs 차원 오류: {obs['quality_obs'].shape}"
    assert obs['portfolio_obs'].shape == (config.PORTFOLIO_OBS_DIM,), f"Portfolio obs 차원 오류: {obs['portfolio_obs'].shape}"
    assert obs['hedging_obs'].shape == (config.HEDGING_OBS_DIM,), f"Hedging obs 차원 오류: {obs['hedging_obs'].shape}"
    print("\n[OK] 모든 관측 차원 검증 완료")
    
except Exception as e:
    print(f"[ERROR] Reset 실패: {e}")
    import traceback
    traceback.print_exc()
    config.DATA_SOURCE = original_data_source
    exit(1)


# =================================================================
# Test 3: 랜덤 에이전트로 1 Step 테스트
# =================================================================
print("\n[Test 3] 랜덤 에이전트 1 Step")
print("-"*60)

try:
    # 랜덤 행동 생성
    random_actions = {
        'value_scores': np.random.rand(config.N_STOCKS),
        'quality_scores': np.random.rand(config.N_STOCKS) * 2 - 1,  # -1 ~ 1
        'portfolio_weights': np.random.dirichlet(np.ones(config.N_STOCKS)),  # 합=1
        'hedge_ratios': np.random.rand(config.N_STOCKS),
        'final_weights': np.random.dirichlet(np.ones(config.N_STOCKS)) * 0.8  # 80% 투자
    }
    
    print("[OK] 랜덤 행동 생성:")
    print(f"  - portfolio_weights 합: {random_actions['portfolio_weights'].sum():.4f}")
    print(f"  - final_weights 합: {random_actions['final_weights'].sum():.4f}")
    
    next_obs, rewards, done, info = env.step(random_actions)
    
    print("\n[OK] Step 실행 성공:")
    print(f"  - 포트폴리오 가치: {info['portfolio_value']:,.0f}원")
    print(f"  - 일 수익률: {info['portfolio_return']*100:.4f}%")
    print(f"  - 거래 비용: {info['trade_costs']:,.0f}원")
    print(f"  - Done: {done}")
    
    print("\n[OK] 보상:")
    for agent, reward in rewards.items():
        print(f"  - {agent}: {reward:.6f}")
    
    assert not done, "1 Step 후 종료되면 안됨"
    assert next_obs is not None, "next_obs가 None이면 안됨"
    
    print("\n[OK] 1 Step 테스트 통과")
    
except Exception as e:
    print(f"[ERROR] Step 실패: {e}")
    import traceback
    traceback.print_exc()
    config.DATA_SOURCE = original_data_source
    exit(1)


# =================================================================
# Test 4: 랜덤 에이전트로 전체 에피소드 실행
# =================================================================
print("\n[Test 4] 랜덤 에이전트 전체 에피소드")
print("-"*60)

try:
    # 환경 리셋
    obs = env.reset()
    done = False
    step_count = 0
    
    print("[OK] 에피소드 시작")
    print(f"  초기 자본: {env.portfolio_value:,.0f}원")
    
    # 10일마다 진행 상황 출력
    while not done:
        # 랜덤 행동
        random_actions = {
            'value_scores': np.random.rand(config.N_STOCKS),
            'quality_scores': np.random.rand(config.N_STOCKS) * 2 - 1,
            'portfolio_weights': np.random.dirichlet(np.ones(config.N_STOCKS)),
            'hedge_ratios': np.random.rand(config.N_STOCKS),
            'final_weights': np.random.dirichlet(np.ones(config.N_STOCKS)) * 0.8
        }
        
        next_obs, rewards, done, info = env.step(random_actions)
        step_count += 1
        
        # 10일마다 출력
        if step_count % 10 == 0:
            print(f"  Day {step_count:3d}: 자산 {info['portfolio_value']:>12,.0f}원 " +
                  f"(수익률 {info['portfolio_return']*100:>6.2f}%)")
        
        if not done:
            obs = next_obs
    
    print(f"\n[OK] 에피소드 완료 ({step_count}일)")
    print(f"  최종 자산: {env.portfolio_value:,.0f}원")
    
    # 성과 지표 출력
    metrics = env.get_performance_metrics()
    
    print("\n[OK] 성과 지표:")
    print(f"  - 총 수익률:      {metrics['total_return']*100:>8.2f}%")
    print(f"  - 샤프 비율:      {metrics['sharpe_ratio']:>8.3f}")
    print(f"  - 최대 낙폭:      {metrics['max_drawdown']*100:>8.2f}%")
    print(f"  - 승률:           {metrics['win_rate']*100:>8.2f}%")
    print(f"  - 변동성 (연):    {metrics['volatility']*100:>8.2f}%")
    print(f"  - 칼마 비율:      {metrics['calmar_ratio']:>8.3f}")
    print(f"  - 소르티노 비율:  {metrics['sortino_ratio']:>8.3f}")
    
    # 합리성 검증
    assert -0.9 < metrics['total_return'] < 2.0, f"총 수익률이 비정상: {metrics['total_return']}"
    assert -5.0 < metrics['sharpe_ratio'] < 5.0, f"샤프 비율이 비정상: {metrics['sharpe_ratio']}"
    assert -1.0 <= metrics['max_drawdown'] <= 0.0, f"최대 낙폭이 비정상: {metrics['max_drawdown']}"
    
    print("\n[OK] 성과 지표 합리성 검증 완료")
    
except Exception as e:
    print(f"[ERROR] 전체 에피소드 실행 실패: {e}")
    import traceback
    traceback.print_exc()
    config.DATA_SOURCE = original_data_source
    exit(1)


# =================================================================
# Test 5: MultiAgentSystem과 통합 테스트
# =================================================================
print("\n[Test 5] MultiAgentSystem 통합 테스트")
print("-"*60)

try:
    # MultiAgentSystem 초기화
    mas = MultiAgentSystem(system_id=0)
    print("[OK] MultiAgentSystem 초기화 완료")
    
    # 환경 리셋
    obs = env.reset()
    done = False
    step_count = 0
    
    print("[OK] 에피소드 시작 (MultiAgentSystem)")
    print(f"  초기 자본: {env.portfolio_value:,.0f}원")
    
    # 20일 실행
    while not done and step_count < 20:
        # MultiAgentSystem으로 행동 생성
        actions = mas.act(obs)
        
        # 환경 진행
        next_obs, rewards, done, info = env.step(actions)
        step_count += 1
        
        # 경험 저장 (학습은 안 함)
        if not done:
            mas.store_transition(obs, actions, rewards, next_obs, done)
            obs = next_obs
        
        # 5일마다 출력
        if step_count % 5 == 0:
            print(f"  Day {step_count:3d}: 자산 {info['portfolio_value']:>12,.0f}원 " +
                  f"(투자비중 {actions['final_weights'].sum()*100:>5.1f}%)")
    
    print(f"\n[OK] MultiAgentSystem 통합 테스트 완료 ({step_count}일)")
    print(f"  최종 자산: {env.portfolio_value:,.0f}원")
    print(f"  Replay Buffer 크기: {len(mas.replay_buffer)}")
    
    # 버퍼에서 배치 샘플링 테스트
    if len(mas.replay_buffer) >= 5:
        batch = mas.replay_buffer.sample(batch_size=5)
        print(f"\n[OK] Replay Buffer 샘플링 성공:")
        print(f"  - value_obs shape: {batch['value_obs'].shape}")
        print(f"  - value_reward shape: {batch['value_reward'].shape}")
    
    print("\n[OK] 전체 통합 테스트 통과!")
    
except Exception as e:
    print(f"[ERROR] MultiAgentSystem 통합 실패: {e}")
    import traceback
    traceback.print_exc()
    config.DATA_SOURCE = original_data_source
    exit(1)


# =================================================================
# 마무리
# =================================================================
print("\n" + "="*60)
print("모든 테스트 통과!")
print("="*60)
print("\n[요약]")
print(f"  - 환경 초기화: OK")
print(f"  - 관측 구조: OK")
print(f"  - 랜덤 에이전트 1 Step: OK")
print(f"  - 랜덤 에이전트 전체 에피소드: OK")
print(f"  - MultiAgentSystem 통합: OK")
print(f"  - 성과 지표 계산: OK")
print(f"\n[데이터 파이프라인 → 환경 → 에이전트] 전체 흐름 검증 완료")

# Config 복원
config.DATA_SOURCE = original_data_source

print("\n[OK] BacktestEnv 테스트 완료!\n")

