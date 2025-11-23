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


class Config:
    """전역 설정 클래스"""
    
    # ========================================
    # 환경 정보
    # ========================================
    ENV = ENV  # 모듈 레벨 ENV를 클래스 속성으로 노출
    
    # ========================================
    # Environment Settings
    # ========================================
    N_STOCKS = 30  # 20 → 30 (더 많은 종목으로 분산 투자)
    INITIAL_CAPITAL = 10_000_000
    TRANSACTION_COST = 0.002  # 0.2% (증권사 수수료 + 세금 고려)
    
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
    BUFFER_CAPACITY = 50000  # 10000 → 50000 (경험 재사용 증가)
    
    # Batch Size (환경별 자동 조정)
    # Local: 256, Colab with GPU: 512-1024
    if ENV['is_colab'] and ENV['has_gpu']:
        # A100/V100: 대용량 배치
        if ENV['gpu_memory_gb'] >= 30:  # A100 (40GB)
            BATCH_SIZE = 1024
        elif ENV['gpu_memory_gb'] >= 15:  # V100 (16GB)
            BATCH_SIZE = 512
        else:  # T4 (16GB)
            BATCH_SIZE = 256
    else:
        # Local CPU/GPU
        BATCH_SIZE = 256
    
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
    N_GENERATIONS = 100
    MUTATION_PROB = 0.9
    MUTATION_ALPHA = 0.2
    
    # RL Fine-tuning
    TOP_K = 10
    RL_EPISODES = 5
    RL_UPDATES = 50  # 10 → 50 (더 많은 RL 학습)
    
    # ========================================
    # Data Pipeline
    # ========================================
    DATA_SOURCE = "real"  # "synthetic" or "real"
    CACHE_DIR = "data/cache"
    
    DATA_START_DATE = "2023-01-01"
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
