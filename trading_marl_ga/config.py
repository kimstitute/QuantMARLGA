"""
Configuration File

전역 설정 및 하이퍼파라미터
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드 (trading_marl_ga/.env)
env_path = Path(__file__).parent / '.env'
if env_path.exists():
    load_dotenv(env_path)


class Config:
    """전역 설정 클래스"""
    
    # ========================================
    # Environment Settings
    # ========================================
    N_STOCKS = 10
    INITIAL_CAPITAL = 10_000_000
    TRANSACTION_COST = 0.001  # 0.1%
    
    # ========================================
    # Agent Observation Dimensions
    # ========================================
    # 병렬 + 융합 구조
    VALUE_OBS_DIM = N_STOCKS * 2 + 3  # 23
    QUALITY_OBS_DIM = N_STOCKS * 2 + 3  # 23 (독립적)
    PORTFOLIO_OBS_DIM = N_STOCKS * 2 + 32 + 4  # 56 (융합)
    HEDGING_OBS_DIM = N_STOCKS + 32  # 42
    
    ACTION_DIM = N_STOCKS
    
    # ========================================
    # Network Architecture
    # ========================================
    HIDDEN_DIM = 128
    
    # ========================================
    # Training Hyperparameters
    # ========================================
    # Replay Buffer
    BUFFER_CAPACITY = 10000
    BATCH_SIZE = 256
    
    # Learning Rates
    LEARNING_RATE_ACTOR = 3e-4
    LEARNING_RATE_CRITIC = 1e-3
    
    # RL Parameters
    GAMMA = 0.99
    TAU = 0.005
    GRAD_CLIP = 10.0
    
    # ========================================
    # Genetic Algorithm
    # ========================================
    POPULATION_SIZE = 30
    N_GENERATIONS = 100
    MUTATION_PROB = 0.9
    MUTATION_ALPHA = 0.2
    
    # RL Fine-tuning
    TOP_K = 10
    RL_EPISODES = 5
    RL_UPDATES = 10
    
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
