"""
Google Colab용 QuantMARLGA 전체 파이프라인
- 종목 선정 → 학습 → 테스트 → 결과 저장

셀별로 복사해서 Colab에서 실행하세요.
"""

# ============================================================================
# 📦 CELL 1: 환경 설정 및 프로젝트 다운로드
# ============================================================================
"""
!pip install -q yfinance pandas numpy torch scipy opendart-python

import os
import sys
from getpass import getpass

# GitHub 레포지토리 클론 (Private이므로 Token 필요)
if not os.path.exists('/content/QuantMARLGA'):
    print("=" * 80)
    print("GitHub Personal Access Token이 필요합니다")
    print("=" * 80)
    print("1. https://github.com/settings/tokens 방문")
    print("2. 'Generate new token (classic)' 클릭")
    print("3. 'repo' 권한 체크")
    print("4. 생성된 토큰 복사")
    print("=" * 80)
    
    token = getpass("GitHub Token 입력: ")
    
    if not token:
        print("❌ Token이 비어있습니다!")
        sys.exit(1)
    
    # 토큰으로 클론
    repo_url = f"https://{token}@github.com/kimstitute/QuantMARLGA.git"
    result = os.system(f"git clone {repo_url} 2>&1 | grep -v {token}")
    
    if result != 0:
        print("❌ 클론 실패! Token을 확인하세요.")
        sys.exit(1)
    
    print("✅ 레포지토리 클론 완료!")
else:
    print("✅ 레포지토리가 이미 존재합니다.")

# 작업 디렉토리 이동
os.chdir('/content/QuantMARLGA')
sys.path.insert(0, '/content/QuantMARLGA')

print(f"✅ 현재 경로: {os.getcwd()}")
"""


# ============================================================================
# 🔑 CELL 2: OpenDart API Key 설정 (선택사항)
# ============================================================================
"""
import os
from getpass import getpass

# OpenDart API Key (https://opendart.fss.or.kr/)
# 없으면 PER/PBR 데이터만 사용, ROE/부채비율은 0으로 처리
print("=" * 80)
print("OpenDart API Key 설정 (선택사항)")
print("=" * 80)
print("• Key가 있으면: 전체 펀더멘털 데이터 사용")
print("• Key가 없으면: PER/PBR만 사용 (Yahoo Finance)")
print("=" * 80)

api_key = getpass("OpenDart API Key (없으면 Enter): ")

if api_key:
    os.environ['OPENDART_API_KEY'] = api_key
    print("✅ OpenDart API Key 설정 완료")
else:
    print("⚠️  OpenDart Key 없음 - PER/PBR만 사용")
"""


# ============================================================================
# 📊 CELL 3: 종목 Universe 선정 (2021-2024 전체 기간)
# ============================================================================
"""
print("=" * 80)
print("Step 1: 종목 Universe 선정")
print("=" * 80)
print("전체 기간(2021-2024)에 결측치 없는 상위 30개 종목을 선정합니다.")
print("=" * 80)

!python trading_marl_ga/select_universe.py

# 선정 결과 확인
import pickle
with open('data/selected_tickers.pkl', 'rb') as f:
    universe_data = pickle.load(f)

print("\n" + "=" * 80)
print("✅ 종목 선정 완료!")
print("=" * 80)
print(f"선정 종목 수: {len(universe_data['tickers'])}개")
print(f"선정 일시: {universe_data['selection_date']}")
print(f"전체 기간: {universe_data['full_start_date']} ~ {universe_data['full_end_date']}")
print("\n상위 10개 종목:")
for i, ticker in enumerate(universe_data['tickers'][:10], 1):
    print(f"  {i:2d}. {ticker}")
print("=" * 80)
"""


# ============================================================================
# 🎯 CELL 4: 학습 실행 (2021-2023, Rolling Window)
# ============================================================================
"""
print("=" * 80)
print("Step 2: 학습 실행 (2021-2023)")
print("=" * 80)
print("선정된 30개 종목으로 Rolling Window 학습을 시작합니다.")
print("예상 소요 시간: 30-40분 (GPU 사용 시)")
print("=" * 80)

!python trading_marl_ga/train.py

print("\n" + "=" * 80)
print("✅ 학습 완료!")
print("=" * 80)
"""


# ============================================================================
# 🧪 CELL 5: 테스트 실행 (2024, Out-of-sample)
# ============================================================================
"""
print("=" * 80)
print("Step 3: 테스트 실행 (2024)")
print("=" * 80)
print("학습과 동일한 30개 종목으로 Out-of-sample 테스트를 시작합니다.")
print("=" * 80)

!python trading_marl_ga/test.py

print("\n" + "=" * 80)
print("✅ 테스트 완료!")
print("=" * 80)
"""


# ============================================================================
# 💾 CELL 6: Google Drive에 결과 저장
# ============================================================================
"""
from google.colab import drive
import shutil
import os

print("=" * 80)
print("Google Drive에 결과 저장")
print("=" * 80)

# Google Drive 마운트
drive.mount('/content/drive')

# 저장 경로
save_dir = '/content/drive/MyDrive/QuantMARLGA_results/'
os.makedirs(save_dir, exist_ok=True)

# 저장할 파일/폴더 목록
items_to_save = [
    ('models/best_system/', 'best_system/'),           # 최고 모델
    ('models/metadata.pkl', 'metadata.pkl'),           # 학습 메타데이터
    ('models/fitness_history.pkl', 'fitness_history.pkl'),  # 학습 곡선
    ('models/test_results.pkl', 'test_results.pkl'),   # 테스트 결과
    ('data/selected_tickers.pkl', 'selected_tickers.pkl'),  # 선정 종목
]

print("저장 중...")
for source, dest in items_to_save:
    source_path = f'/content/QuantMARLGA/{source}'
    dest_path = f'{save_dir}{dest}'
    
    if os.path.exists(source_path):
        if os.path.isdir(source_path):
            # 디렉토리는 전체 복사
            if os.path.exists(dest_path):
                shutil.rmtree(dest_path)
            shutil.copytree(source_path, dest_path)
            print(f"  ✅ {source} → {dest}")
        else:
            # 파일은 개별 복사
            shutil.copy2(source_path, dest_path)
            print(f"  ✅ {source} → {dest}")
    else:
        print(f"  ⚠️  {source} 없음 (건너뜀)")

print("\n" + "=" * 80)
print("✅ Google Drive 저장 완료!")
print(f"저장 경로: {save_dir}")
print("=" * 80)
"""


# ============================================================================
# 📈 CELL 7: 결과 시각화 (선택사항)
# ============================================================================
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np

print("=" * 80)
print("결과 시각화")
print("=" * 80)

# 학습 곡선 로드
with open('models/fitness_history.pkl', 'rb') as f:
    fitness_history = pickle.load(f)

# 테스트 결과 로드
with open('models/test_results.pkl', 'rb') as f:
    test_results = pickle.load(f)

# 1. 학습 곡선
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 좌측: Fitness 진화
ax = axes[0]
generations = list(range(1, len(fitness_history) + 1))
mean_fitness = [stats['mean_fitness'] for stats in fitness_history]
max_fitness = [stats['max_fitness'] for stats in fitness_history]

ax.plot(generations, mean_fitness, 'o-', label='Mean Fitness', linewidth=2)
ax.plot(generations, max_fitness, 's-', label='Max Fitness', linewidth=2)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Generation', fontsize=12)
ax.set_ylabel('Fitness', fontsize=12)
ax.set_title('GA-MARL Training Progress (2021-2023)', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# 우측: 테스트 성과 비교
ax = axes[1]
strategies = list(test_results.keys())
sharpe_ratios = [test_results[s]['sharpe_ratio'] for s in strategies]
colors = ['#FF6B6B' if s == 'GA-MARL (Best)' else '#4ECDC4' for s in strategies]

bars = ax.bar(strategies, sharpe_ratios, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_ylabel('Sharpe Ratio', fontsize=12)
ax.set_title('Out-of-sample Performance Comparison (2024)', fontsize=14, fontweight='bold')
ax.tick_params(axis='x', rotation=45)

# 값 표시
for bar, val in zip(bars, sharpe_ratios):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.3f}',
            ha='center', va='bottom' if val > 0 else 'top', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('results_summary.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ 시각화 완료!")
print("=" * 80)

# 2. 상세 결과 출력
print("\n📊 최종 성과 요약")
print("=" * 80)
for strategy, metrics in test_results.items():
    print(f"\n{strategy}")
    print("-" * 80)
    print(f"  총 수익률:    {metrics['total_return']*100:>7.2f}%")
    print(f"  샤프 비율:    {metrics['sharpe_ratio']:>7.3f}")
    print(f"  최대 손실:    {metrics['max_drawdown']*100:>7.2f}%")
    print(f"  변동성:       {metrics['volatility']*100:>7.2f}%")
    print(f"  승률:         {metrics['win_rate']*100:>7.2f}%")

print("\n" + "=" * 80)

# 3. 종목 정보
with open('data/selected_tickers.pkl', 'rb') as f:
    universe = pickle.load(f)

print("\n📋 사용된 종목 Universe")
print("=" * 80)
print(f"선정 일시: {universe['selection_date']}")
print(f"전체 기간: {universe['full_start_date']} ~ {universe['full_end_date']}")
print(f"종목 수: {len(universe['tickers'])}개")
print("\n종목 리스트:")
for i, ticker in enumerate(universe['tickers'], 1):
    print(f"  {i:2d}. {ticker}", end='')
    if i % 5 == 0:
        print()
    else:
        print("  ", end='')
print("\n" + "=" * 80)
"""


# ============================================================================
# 🔄 CELL 8: 이전 결과 불러오기 (재실행 시)
# ============================================================================
"""
from google.colab import drive
import shutil
import os

print("=" * 80)
print("Google Drive에서 이전 결과 불러오기")
print("=" * 80)

# Google Drive 마운트
drive.mount('/content/drive')

# 불러올 경로
load_dir = '/content/drive/MyDrive/QuantMARLGA_results/'

if not os.path.exists(load_dir):
    print("❌ 저장된 결과가 없습니다!")
    print(f"   경로: {load_dir}")
else:
    # 불러올 파일/폴더 목록
    items_to_load = [
        ('best_system/', 'models/best_system/'),
        ('metadata.pkl', 'models/metadata.pkl'),
        ('fitness_history.pkl', 'models/fitness_history.pkl'),
        ('test_results.pkl', 'models/test_results.pkl'),
        ('selected_tickers.pkl', 'data/selected_tickers.pkl'),
    ]
    
    print("불러오는 중...")
    for source, dest in items_to_load:
        source_path = f'{load_dir}{source}'
        dest_path = f'/content/QuantMARLGA/{dest}'
        
        if os.path.exists(source_path):
            # 대상 디렉토리 생성
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            if os.path.isdir(source_path):
                # 디렉토리는 전체 복사
                if os.path.exists(dest_path):
                    shutil.rmtree(dest_path)
                shutil.copytree(source_path, dest_path)
                print(f"  ✅ {source} → {dest}")
            else:
                # 파일은 개별 복사
                shutil.copy2(source_path, dest_path)
                print(f"  ✅ {source} → {dest}")
        else:
            print(f"  ⚠️  {source} 없음 (건너뜀)")
    
    print("\n" + "=" * 80)
    print("✅ 이전 결과 불러오기 완료!")
    print("=" * 80)
    print("\n이제 CELL 7 (결과 시각화)을 실행할 수 있습니다.")
"""


# ============================================================================
# ℹ️  사용 가이드
# ============================================================================
"""
═══════════════════════════════════════════════════════════════════════════
                    QuantMARLGA Colab 실행 가이드
═══════════════════════════════════════════════════════════════════════════

📝 전체 파이프라인 (처음 실행 시)
───────────────────────────────────────────────────────────────────────────
  1. CELL 1: 환경 설정 (프로젝트 다운로드)
  2. CELL 2: OpenDart API Key 설정 (선택)
  3. CELL 3: 종목 선정 (2021-2024 전체)
  4. CELL 4: 학습 (2021-2023, ~30-40분)
  5. CELL 5: 테스트 (2024)
  6. CELL 6: Google Drive 저장
  7. CELL 7: 결과 시각화

🔄 재실행 시 (결과만 확인)
───────────────────────────────────────────────────────────────────────────
  1. CELL 1: 환경 설정
  2. CELL 8: Google Drive에서 불러오기
  3. CELL 7: 결과 시각화

⚙️ 주요 특징
───────────────────────────────────────────────────────────────────────────
  ✅ 종목 Universe 선정: 2021-2024 전체 기간 생존 종목만
  ✅ 학습-테스트 일관성: 정확히 같은 30개 종목 사용
  ✅ Out-of-sample: 2024 데이터는 학습에 전혀 사용 안 함
  ✅ 자동 저장: Google Drive에 모든 결과 자동 백업
  ✅ 재현 가능: 종목 리스트 저장으로 동일 실험 재현

⚠️  주의사항
───────────────────────────────────────────────────────────────────────────
  • GPU 사용 권장: 런타임 → 런타임 유형 변경 → GPU
  • Private Repo: GitHub Personal Access Token 필요
  • 학습 시간: GPU 기준 30-40분 (CPU는 2-3시간)
  • Colab 세션: 90분 무작동 시 초기화 (중간 저장 필수!)

🔗 참고 링크
───────────────────────────────────────────────────────────────────────────
  • GitHub: https://github.com/kimstitute/QuantMARLGA
  • Token 생성: https://github.com/settings/tokens
  • OpenDart: https://opendart.fss.or.kr/

═══════════════════════════════════════════════════════════════════════════
"""

