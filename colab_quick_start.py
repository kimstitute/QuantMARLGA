"""
Colab 빠른 시작 스크립트

이 파일 전체를 Colab 셀에 복사해서 실행하세요!
또는 셀별로 나눠서 실행하세요.
"""

# ============================================================
# 셀 1: 환경 확인
# ============================================================
print("="*80)
print("환경 확인")
print("="*80)

# GPU 확인
import torch
print(f"\nPyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠️ GPU 없음! 런타임 유형을 GPU로 변경하세요!")

# ============================================================
# 셀 2: 프로젝트 다운로드
# ============================================================
print("\n" + "="*80)
print("프로젝트 다운로드")
print("="*80)

# 이미 있으면 업데이트, 없으면 다운로드
import os
if os.path.exists('/content/QuantMARLGA'):
    print("\n이미 존재함 → 업데이트 중...")
    os.chdir('/content/QuantMARLGA')
    !git pull origin main
else:
    print("\n다운로드 중...")
    !git clone https://github.com/YOUR_USERNAME/QuantMARLGA.git
    os.chdir('/content/QuantMARLGA')

os.chdir('/content/QuantMARLGA/trading_marl_ga')
print(f"✅ 작업 디렉토리: {os.getcwd()}")

# ============================================================
# 셀 3: 의존성 설치
# ============================================================
print("\n" + "="*80)
print("의존성 설치")
print("="*80)

!pip install -q -r requirements.txt
print("\n✅ 설치 완료!")

# ============================================================
# 셀 4: Config 확인
# ============================================================
print("\n" + "="*80)
print("Config 확인")
print("="*80)

!python check_config.py

# ============================================================
# 셀 5: 학습 실행 (1-2시간)
# ============================================================
print("\n" + "="*80)
print("학습 시작!")
print("="*80)
print("⏰ 예상 시간: 1-2시간 (12 세대, GPU)")
print("⚠️ 브라우저를 닫지 마세요! (또는 Colab Pro 사용)")
print("="*80 + "\n")

!python train.py

# ============================================================
# 셀 6: 학습 결과 시각화
# ============================================================
print("\n" + "="*80)
print("학습 결과 시각화")
print("="*80)

import pickle
import matplotlib.pyplot as plt
import numpy as np

# Fitness history 로드
with open('models/fitness_history.pkl', 'rb') as f:
    history = pickle.load(f)

# 데이터 추출
generations = list(range(1, len(history) + 1))
max_fitness = [s['max_fitness'] for s in history]
mean_fitness = [s['mean_fitness'] for s in history]
min_fitness = [s['min_fitness'] for s in history]

# 플롯
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 왼쪽: Fitness 진화
ax1.plot(generations, max_fitness, 'g-o', linewidth=2, markersize=8, label='Best')
ax1.plot(generations, mean_fitness, 'b-s', linewidth=2, markersize=8, label='Mean')
ax1.plot(generations, min_fitness, 'r-^', linewidth=2, markersize=8, label='Worst')
ax1.fill_between(generations, min_fitness, max_fitness, alpha=0.15, color='blue')
ax1.set_xlabel('Generation', fontsize=13)
ax1.set_ylabel('Fitness (Sharpe Ratio)', fontsize=13)
ax1.set_title('Training Progress (2021-2023)', fontsize=15, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# 오른쪽: 개선율
improvement = [(mean_fitness[i] - mean_fitness[0]) / abs(mean_fitness[0]) * 100 
               for i in range(len(mean_fitness))]
colors = ['green' if x > 0 else 'red' for x in improvement]
ax2.bar(generations, improvement, color=colors, alpha=0.7, edgecolor='black')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Generation', fontsize=13)
ax2.set_ylabel('Improvement (%)', fontsize=13)
ax2.set_title('Mean Fitness Improvement', fontsize=15, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('models/training_curve.png', dpi=150, bbox_inches='tight')
plt.show()

# 통계
print(f"\n{'='*80}")
print("학습 통계")
print("="*80)
print(f"시작 평균 Fitness: {mean_fitness[0]:.4f}")
print(f"최종 평균 Fitness: {mean_fitness[-1]:.4f}")
print(f"개선율: {improvement[-1]:+.2f}%")
print(f"역대 최고: {max(max_fitness):.4f} (세대 {generations[max_fitness.index(max(max_fitness))]})")
print("="*80)

# ============================================================
# 셀 7: 테스트 실행
# ============================================================
print("\n" + "="*80)
print("테스트 실행 (2024-H1)")
print("="*80)

!python test.py

# ============================================================
# 셀 8: 테스트 결과 시각화
# ============================================================
print("\n" + "="*80)
print("테스트 결과 시각화")
print("="*80)

import pandas as pd

# 결과 로드
with open('models/test_results.pkl', 'rb') as f:
    results = pickle.load(f)

# DataFrame
df = pd.DataFrame(results).T
df = df.sort_values('sharpe_ratio', ascending=False)

# 차트
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 총 수익률
ax = axes[0, 0]
returns = df['total_return'] * 100
colors = ['green' if x > 0 else 'red' for x in returns]
ax.barh(returns.index, returns, color=colors, alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Total Return (%)', fontsize=11)
ax.set_title('Total Return', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# 2. Sharpe Ratio
ax = axes[0, 1]
sharpe = df['sharpe_ratio']
colors = ['green' if x > 0 else 'red' for x in sharpe]
ax.barh(sharpe.index, sharpe, color=colors, alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Sharpe Ratio', fontsize=11)
ax.set_title('Sharpe Ratio', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# 3. Max Drawdown
ax = axes[1, 0]
mdd = df['max_drawdown'] * 100
ax.barh(mdd.index, mdd, color='orange', alpha=0.7, edgecolor='black')
ax.set_xlabel('Max Drawdown (%)', fontsize=11)
ax.set_title('Max Drawdown', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# 4. Calmar Ratio
ax = axes[1, 1]
calmar = df['calmar_ratio']
colors = ['green' if x > 0 else 'red' for x in calmar]
ax.barh(calmar.index, calmar, color=colors, alpha=0.7, edgecolor='black')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.set_xlabel('Calmar Ratio', fontsize=11)
ax.set_title('Calmar Ratio', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('models/test_results.png', dpi=150, bbox_inches='tight')
plt.show()

# 결과 테이블
print(f"\n{'='*80}")
print("테스트 결과 상세 (2024-H1)")
print("="*80)
print(df[['total_return', 'sharpe_ratio', 'max_drawdown', 'calmar_ratio']].to_string())
print("="*80)

# 승자
best = df.index[0]
print(f"\n🏆 Winner: {best}")
print(f"   Sharpe Ratio: {df.loc[best, 'sharpe_ratio']:.4f}")
print(f"   Total Return: {df.loc[best, 'total_return']*100:.2f}%")
print(f"   Max Drawdown: {df.loc[best, 'max_drawdown']*100:.2f}%")

# ============================================================
# 셀 9: 모델 저장 (Google Drive)
# ============================================================
print("\n" + "="*80)
print("Google Drive에 저장")
print("="*80)

from google.colab import drive
drive.mount('/content/drive')

# 모델 복사
!mkdir -p /content/drive/MyDrive/QuantMARLGA_models
!cp -r models/* /content/drive/MyDrive/QuantMARLGA_models/

print("\n✅ 모델이 Google Drive에 저장되었습니다!")
print("   경로: MyDrive/QuantMARLGA_models/")

print("\n" + "="*80)
print("🎉 모든 작업 완료!")
print("="*80)

