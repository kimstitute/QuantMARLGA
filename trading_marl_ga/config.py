"""
Configuration File

전역 설정 및 하이퍼파라미터
"""

import os
from pathlib import Path
from dotenv import load_dotenv
import torch

# .env 파일 로드 (trading_marl_ga/.env)
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)


def detect_environment():
    """
    실행 환경 감지 (Colab, GPU 등)
    
    Returns:
        dict: 환경 정보
    """
    env_info = {
        'is_colab': False,
        'has_gpu': torch.cuda.is_available(),
        'gpu_name': None,
        'gpu_memory_gb': 0,
        'num_cpus': os.cpu_count() or 1,
    }
    
    # Colab 환경 감지
    try:
        import google.colab
        env_info['is_colab'] = True
    except ImportError:
        pass
    
    # GPU 정보
    if env_info['has_gpu']:
        env_info['gpu_name'] = torch.cuda.get_device_name(0)
        env_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    
    return env_info


# 환경 감지
ENV = detect_environment()

# Device 설정
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Config:
    """전역 설정 클래스"""
    
    # ========================================
    # 환경 정보
    # ========================================
    ENV = ENV  # 모듈 레벨 ENV를 클래스 속성으로 노출
    DEVICE = DEVICE  # GPU/CPU 디바이스
    
    # ========================================
    # Environment Settings
    # ========================================
    N_STOCKS = 30  # 20 → 30 (더 많은 종목으로 분산 투자)
    INITIAL_CAPITAL = 10_000_000
    TRANSACTION_COST = 0.002  # 0.2% (증권사 수수료 + 세금 고려)
    
    # Rebalancing Period (리밸런싱 주기)
    # 1 = 매일, 5 = 주간 (1주), 20 = 월간 (1개월)
    # 더 긴 주기 = 거래 비용 감소, 더 안정적
    REBALANCE_PERIOD = 5  # 기본값: 주간 (5거래일)
    
    # ========================================
    # Rolling Window Training (과적합 방지)
    # ========================================
    # 각 세대마다 다른 시장 기간으로 학습하여 과적합 방지
    # 예: 세대1=2021-Q1, 세대2=2021-Q2, ..., 세대12=2023-Q4
    ROLLING_TRAIN_MONTHS = 3  # 각 세대 학습 기간 (3개월 = 분기별)
    START_YEAR = 2021         # 학습 시작 연도
    START_MONTH = 1           # 학습 시작 월
    
    # 테스트 기간 (학습에 절대 사용 안 함!)
    TEST_START = "2024-01-01"
    TEST_END = "2024-06-30"
    
    # ========================================
    # Agent Observation Dimensions (동적 조정)
    # ========================================
    # 병렬 + 융합 구조 (N_STOCKS에 자동 비례)
    VALUE_OBS_DIM = N_STOCKS * 2 + 3  # 30개: 63
    QUALITY_OBS_DIM = N_STOCKS * 2 + 3  # 30개: 63 (독립적)
    # risk_metrics: volatility(N) + beta(N) + sharpe(N) + 2 = N*3+2
    # market_risk: correlation(N) + drawdown(N) + var(N) + 2 = N*3+2
    PORTFOLIO_OBS_DIM = N_STOCKS * 2 + (N_STOCKS * 3 + 2) + 4  # 30개: 30+30+92+4 = 156
    HEDGING_OBS_DIM = N_STOCKS + (N_STOCKS * 3 + 2)  # 30개: 30+92 = 122
    
    ACTION_DIM = N_STOCKS  # 30개
    
    # ========================================
    # Network Architecture
    # ========================================
    # Hidden dimension: N_STOCKS에 비례하여 자동 조정
    # 10개: 128, 20개: 240, 30개: 360
    HIDDEN_DIM = max(128, N_STOCKS * 12)  # 최소 128, 종목당 12 뉴런
    
    # ========================================
    # Training Hyperparameters
    # ========================================
    # Replay Buffer
    BUFFER_CAPACITY = 10000  # 50000 → 10000 (현실적인 크기로 축소)
    
    # Batch Size (128)
    # Rolling window에서 1세대부터 RL 학습 가능하도록 축소
    # 3개월 × 12회 리밸런싱 × 11팀 = 132 transitions > 128 ✅
    BATCH_SIZE = 128  # 256 → 128
    
    # RL 학습 시작 최소 버퍼 크기
    # BATCH_SIZE와 동일하게 설정하여 1세대부터 RL 학습 가능
    MIN_BUFFER_FOR_RL = BATCH_SIZE  # 128
    
    # Learning Rates
    LEARNING_RATE_ACTOR = 3e-4
    LEARNING_RATE_CRITIC = 1e-3
    
    # RL Parameters
    GAMMA = 0.99
    TAU = 0.005
    GRAD_CLIP = 10.0
    
    # ========================================
    # Optimization Settings (환경별 자동 조정)
    # ========================================
    # Mixed Precision Training (FP16)
    # Colab GPU: 활성화, Local: 비활성화
    USE_AMP = ENV['is_colab'] and ENV['has_gpu']
    
    # Parallel Backtesting
    # Colab: 활성화 (멀티코어), Local: 비활성화 (안전)
    USE_PARALLEL_BACKTEST = ENV['is_colab']
    PARALLEL_WORKERS = min(ENV['num_cpus'], 8) if USE_PARALLEL_BACKTEST else 1
    
    # ========================================
    # Genetic Algorithm
    # ========================================
    POPULATION_SIZE = 10  # 20 → 10 (빠른 실험, 종목 수 증가로 보완)
    N_GENERATIONS = 12  # 100 → 12 (Rolling Window: 분기별, 2021-2023)
    
    # Elite 비율 (RACE 논문: 0.2 = 20%, 우리: 0.3 = 30%)
    # 주식 투자는 노이즈가 많으므로 더 보수적으로 설정
    ELITE_FRACTION = 0.3  # 상위 30% 보존, 70% 교체 (안정적)
    
    # Mutation 설정 (가우시안 노이즈 기반)
    MUTATION_PROB = 0.9        # 각 파라미터가 변이할 확률 (0.0~1.0)
    MUTATION_SCALE_RATIO = 0.05  # 가중치 크기 대비 노이즈 비율 (5% = 균형적 탐색)
    
    # RL Fine-tuning
    TOP_K = 10
    RL_EPISODES = 5
    RL_UPDATES = 50  # 10 → 50 (더 많은 RL 학습)
    
    # ========================================
    # Data Pipeline
    # ========================================
    DATA_SOURCE = "real"  # "synthetic" or "real"
    CACHE_DIR = "data/cache"
    
    # 기본 데이터 기간 (Rolling Window 훈련 전체 범위)
    DATA_START_DATE = "2021-01-01"  # 2023 → 2021 (Rolling Window 시작)
    DATA_END_DATE = "2023-12-31"
    LOOKBACK_DAYS = 60
    
    # OpenDart API Key (환경변수에서 로드)
    OPENDART_API_KEY = os.getenv('OPENDART_API_KEY', None)
    
    # ========================================
    # Reward Function
    # ========================================
    REWARD_MODE = "differentiated"
    REWARD_INDIVIDUAL_WEIGHT = 0.5
    REWARD_TEAM_WEIGHT = 0.5
    
    PORTFOLIO_RETURN_WEIGHT = 0.6
    HEDGING_RISK_WEIGHT = 0.5
    
    REWARD_CLIP = 1.0
    SHARPE_WINDOW = 20


# 전역 인스턴스
config = Config()
