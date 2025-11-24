# QuantMARLGA

**Multi-Agent Reinforcement Learning with Genetic Algorithm for Quantitative Trading**

RACE 논문 기반 GA+MARL 하이브리드 한국 주식 트레이딩 시스템

---

## 📋 프로젝트 개요

### 핵심 아이디어
- **4개 전문화 에이전트**: Value, Quality, Portfolio, Hedging
- **병렬 + 융합 구조**: Value/Quality 병렬 → Portfolio → Hedging 순차
- **RACE 방식 학습**: EA Population (GA 진화) + MARL 팀 (RL 학습)
- **실전 데이터**: 한국 주식 시장 실제 데이터 (KOSPI 상위 종목)

### 주요 특징
✅ RACE 논문 방식 완전 구현 (EA vs MARL 분리)  
✅ Shared Replay Buffer (모든 팀 경험 공유)  
✅ Dynamic Injection (MARL → EA worst 교체)  
✅ 실제 시장 데이터 파이프라인  
✅ 차별화 보상 함수 (에이전트별 기여도)  
✅ 7종 성과 지표 (Sharpe, MDD, Calmar 등)  

---

## 🏗️ 시스템 구조

```
┌─────────────────────────────────────────┐
│        EA Population (n개)              │
│  ┌──────┐ ┌──────┐      ┌──────┐       │
│  │ EA 0 │ │ EA 1 │ ...  │ EA n │       │
│  └──────┘ └──────┘      └──────┘       │
│     (GA 진화만)                          │
└─────────────────────────────────────────┘
            │
            ├─── Rollout → Shared Buffer
            │
┌─────────────────────────────────────────┐
│         MARL 팀 (1개)                   │
│  ┌──────────────────────────────────┐  │
│  │ Value → Quality → Portfolio →    │  │
│  │                    Hedging       │  │
│  └──────────────────────────────────┘  │
│     (RL 학습만)                         │
└─────────────────────────────────────────┘
            │
            ├─── Rollout → Shared Buffer
            │
            └─── RL Update ← Shared Buffer
            │
            └─── Injection → EA worst
```

---

## 📊 학습 루프

### Phase 1: Pure GA (세대 1-30)
1. Fitness 평가 (백테스트 → Sharpe Ratio)
2. Selection (Tournament + Elitism)
3. Crossover (Agent-level)
4. Mutation (Gaussian Noise)

### Phase 2: RACE Hybrid (세대 31-100)
1. **Fitness 평가** (EA Population)
2. **GA 진화** (EA Population만)
3. **Rollout** (EA n개 + MARL 1개 → Shared Buffer)
4. **RL 학습** (MARL 팀만, 모든 경험 활용)
5. **Injection** (MARL → EA worst 교체)
6. **다음 세대 준비** (EA best → MARL 복제)

---

## 🚀 빠른 시작

### 환경 설정
```bash
# Conda 환경 생성
conda create -n quantagents python=3.10
conda activate quantagents

# 패키지 설치
pip install torch numpy pandas scipy
pip install FinanceDataReader pykrx OpenDartReader
pip install python-dotenv tqdm
```

### OpenDart API 키 설정
```bash
# trading_marl_ga/.env 파일 생성
OPENDART_API_KEY=your_api_key_here
```

### 데이터 파이프라인 테스트
```bash
cd trading_marl_ga
python test_data_pipeline.py
```

### 백테스트 환경 테스트
```bash
python test_backtest_env.py
```

### GA Trainer 테스트
```bash
python test_ga_trainer.py        # Pure GA
python test_race_hybrid.py       # RACE Hybrid
```

### 벤치마크 비교
```bash
python benchmarks.py              # Buy & Hold vs Equal Weight vs Random
python final_comparison.py        # GA-MARL vs Benchmarks
```

---

## 📁 프로젝트 구조

```
QuantMARLGA/
├── trading_marl_ga/
│   ├── agents/                    # 에이전트
│   │   ├── base_agent.py         # BaseAgent (Actor-Critic + GA)
│   │   ├── networks.py           # 신경망 (Actor, Critic)
│   │   └── multi_agent_system.py # 4-Agent 시스템
│   │
│   ├── data/                      # 데이터 파이프라인
│   │   ├── collectors/           # 데이터 수집기
│   │   │   ├── price_collector.py
│   │   │   ├── fundamental_collector.py
│   │   │   ├── opendart_collector.py
│   │   │   └── financial_estimator.py
│   │   └── market_data_manager.py # 통합 관리자
│   │
│   ├── environment/               # 백테스트 환경
│   │   ├── backtest_env.py       # 매매 시뮬레이션
│   │   └── reward_calculator_independent.py
│   │
│   ├── evolution/                 # GA + RACE
│   │   └── ga_trainer.py         # RACE 방식 GA Trainer
│   │
│   ├── utils/                     # 유틸리티
│   │   ├── observation.py        # 관측 구성
│   │   ├── replay_buffer.py      # Shared Replay Buffer
│   │   └── metrics.py            # 성과 지표
│   │
│   ├── benchmarks.py              # 벤치마크 전략
│   ├── config.py                  # 설정
│   └── test_*.py                  # 테스트 파일들
│
├── 1.md - 5.md                    # 프로젝트 계획 문서
├── IMPLEMENTATION_STATUS.md       # 구현 현황
└── README.md                      # 이 파일
```

---

## 📈 성과 지표

### 구현된 지표 (7종)
- **Total Return**: 총 수익률
- **Sharpe Ratio**: 위험 대비 수익률
- **Max Drawdown**: 최대 낙폭
- **Win Rate**: 승률
- **Annualized Volatility**: 연율화 변동성
- **Calmar Ratio**: 수익률 / MDD
- **Sortino Ratio**: 하방 위험 조정 수익률

### 벤치마크 결과 (100일, 2023년)
| 전략 | 수익률 | 샤프 | MDD |
|------|--------|------|-----|
| Buy & Hold | 21.38% | 3.333 | -3.72% |
| KOSPI Index | 21.38% | 3.333 | -3.72% |
| Random Agent | 5.44% | 0.958 | -5.92% |

---

## 🔬 실험 설정

### 데이터
- **종목**: KOSPI 시가총액 상위 30개 (10 → 30 확대)
- **기간**: 2023년 (최소 200 거래일 보장)
- **Lookback**: 60 거래일 (기술적 지표 계산용)
- **리밸런싱**: 주간 (5거래일마다)

### 하이퍼파라미터 (최적화됨 - 2025.11.24)
```python
# Environment
N_STOCKS = 30  # 다양성 증가
REBALANCE_PERIOD = 5  # 거래 비용 절감

# GA
POPULATION_SIZE = 10  # 30 → 10 (효율성)
N_GENERATIONS = 100
MUTATION_PROB = 0.9
MUTATION_SCALE_RATIO = 0.05  # 상대적 노이즈
ELITE_FRACTION = 0.3  # 안정성

# RL
BATCH_SIZE = 256
BUFFER_CAPACITY = 10_000
MIN_BUFFER_FOR_RL = 256  # 즉시 학습
LEARNING_RATE_ACTOR = 3e-4
LEARNING_RATE_CRITIC = 1e-3
GAMMA = 0.99

# Hybrid
RL_UPDATES = 50  # 세대당

# GPU (자동 감지)
DEVICE = "cuda" if available else "cpu"
USE_AMP = True  # FP16 (Colab)
```

---

## 📝 문서

- **[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)**: 구현 현황 및 진행 상황
- **[2.md](2.md)**: 8시간 점진적 구현 계획
- **[3.md](3.md)**: 시스템 전체 구조 시각화
- **[4.md](4.md)**: 보상 함수 설계
- **[5.md](5.md)**: 데이터 파이프라인 설계

---

## 🎯 향후 계획

### 단기 (1-2주)
- [ ] 실전 규모 학습 (Population 30, 100세대)
- [ ] 학습 곡선 시각화
- [ ] 최종 성과 분석 리포트

### 중기 (1개월)
- [ ] 종목 확장 (10 → 50개)
- [ ] Walk-Forward Validation
- [ ] 하이퍼파라미터 자동 튜닝

### 장기 (3개월)
- [ ] Pre-training (Rule-Based Expert)
- [ ] 실시간 데이터 연동
- [ ] 자동 매매 인터페이스

---

## 📚 참고 문헌

- **RACE 논문**: Cooperative Multi-Agent Reinforcement Learning with Genetic Algorithm
- **FinanceDataReader**: https://github.com/FinanceData/FinanceDataReader
- **pykrx**: https://github.com/sharebook-kr/pykrx
- **OpenDartReader**: https://github.com/FinanceData/OpenDartReader

---

## 📄 라이선스

MIT License

---

## 👥 기여

이슈 및 PR 환영합니다!

---

**생성일**: 2025-11-23  
**최종 업데이트**: 2025-11-24 (GPU 최적화, 하이퍼파라미터 튜닝)  
**작성자**: AI Assistant