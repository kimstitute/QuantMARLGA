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
    N_STOCKS = 30
    INITIAL_CAPITAL = 10_000_000
    TRANSACTION_COST = 0.002  # 0.2% (commission + tax)
    
    # Rebalancing: 1=daily, 5=weekly, 20=monthly
    REBALANCE_PERIOD = 5
    
    # ========================================
    # Rolling Window Training
    # ========================================
    ROLLING_TRAIN_MONTHS = 3  # 3-month window per generation
    START_YEAR = 2021
    START_MONTH = 1
    
    # Test period (held-out data)
    TEST_START = "2024-01-01"
    TEST_END = "2024-06-30"
    
    # ========================================
    # Agent Observation Dimensions
    # ========================================
    # Parallel + fusion architecture (scales with N_STOCKS)
    VALUE_OBS_DIM = N_STOCKS * 2 + 3  # 63 for N_STOCKS=30
    QUALITY_OBS_DIM = N_STOCKS * 2 + 3  # 63 (independent)
    PORTFOLIO_OBS_DIM = N_STOCKS * 2 + (N_STOCKS * 3 + 2) + 4  # 156: value(30) + quality(30) + risk(92) + market(4)
    HEDGING_OBS_DIM = N_STOCKS + (N_STOCKS * 3 + 2)  # 122: portfolio(30) + market_risk(92)
    
    ACTION_DIM = N_STOCKS
    
    # ========================================
    # Network Architecture
    # ========================================
    HIDDEN_DIM = max(128, N_STOCKS * 12)  # min 128, 12 neurons per stock
    
    # RACE-style Type-specific Shared Encoders
    USE_SHARED_ENCODER = True  # True: 타입별 공유 인코더 (RACE), False: 독립 네트워크
    ENCODER_HIDDEN_DIM = 128  # 공유 인코더 출력 차원 (64→128: 정보 손실 방지)
    
    # ========================================
    # Training Hyperparameters
    # ========================================
    BUFFER_CAPACITY = 10000
    BATCH_SIZE = 128
    MIN_BUFFER_FOR_RL = BATCH_SIZE  # Start RL training from first generation
    
    # Learning Rates
    LEARNING_RATE_ACTOR = 3e-4
    LEARNING_RATE_CRITIC = 1e-3
    
    # RL Parameters
    GAMMA = 0.99
    TAU = 0.005
    GRAD_CLIP = 10.0
    
    # ========================================
    # Optimization Settings
    # ========================================
    # Mixed precision (FP16) for GPU acceleration
    USE_AMP = ENV['is_colab'] and ENV['has_gpu']
    
    # Parallel backtesting (multi-core)
    USE_PARALLEL_BACKTEST = ENV['is_colab']
    PARALLEL_WORKERS = min(ENV['num_cpus'], 8) if USE_PARALLEL_BACKTEST else 1
    
    # ========================================
    # Genetic Algorithm
    # ========================================
    POPULATION_SIZE = 10
    N_GENERATIONS = 12  # 12 quarters (2021-2023)
    
    ELITE_FRACTION = 0.3  # Top 30% preserved
    MUTATION_PROB = 0.9  # Per-parameter mutation probability
    MUTATION_SCALE_RATIO = 0.05  # 5% relative noise
    
    # RL fine-tuning
    TOP_K = 10
    RL_EPISODES = 5
    RL_UPDATES = 50
    
    # ========================================
    # Data Pipeline
    # ========================================
    DATA_SOURCE = "real"  # "synthetic" or "real"
    CACHE_DIR = "data/cache"
    
    DATA_START_DATE = "2021-01-01"  # Full training range
    DATA_END_DATE = "2023-12-31"
    LOOKBACK_DAYS = 60
    
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
