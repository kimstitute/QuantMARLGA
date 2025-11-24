# 🚀 Colab 실행 가이드

**QuantMARLGA - Multi-Agent Reinforcement Learning + Genetic Algorithm for Trading**

---

## 📋 빠른 시작 (5분)

### 1️⃣ Colab 노트북 생성

[Google Colab](https://colab.research.google.com/)에서 새 노트북 생성

---

### 2️⃣ GPU 설정

상단 메뉴: `런타임` → `런타임 유형 변경` → `하드웨어 가속기: GPU (T4)`

```python
# GPU 확인
!nvidia-smi
```

---

### 3️⃣ 프로젝트 다운로드

```python
# 처음 실행 시
!git clone https://github.com/YOUR_USERNAME/QuantMARLGA.git
%cd QuantMARLGA/trading_marl_ga

# 이미 다운로드했다면
%cd /content/QuantMARLGA
!git pull origin main
%cd trading_marl_ga
```

---

### 4️⃣ 의존성 설치

```python
!pip install -q -r requirements.txt
print("✅ 설치 완료!")
```

---

### 5️⃣ Config 확인

```python
!python check_config.py
```

**Expected Output:**
```
[GA 설정]
  Population 크기:           10
  세대 수:                   12
  자식 변이 확률:            90%
  파라미터 변이 확률:        10%
  실제 변이율:               9.0% (sparse mutation)

[환경 최적화]
  Device:                    cuda
  GPU:                       Yes
  혼합 정밀도 (FP16):        활성화
```

---

### 6️⃣ 학습 시작! 🚀

```python
# Rolling Window Training (2021-2023)
# 예상 시간: 1-2시간
!python train.py
```

**학습 중 로그 예시:**
```
세대 1/12
학습 기간: 2021-01-01 ~ 2021-03-31
============================================================

[1/4] Rollout + Fitness 평가
EA # 1: 종료자산= 10,133,726원 | 기간수익률= 1.34% | Fitness= 0.5796
...
MARL : 종료자산= 9,679,284원 | 기간수익률= -3.21% | Fitness=-0.8440

[2/4] RL: MARL 팀 학습
[OK] MARL 팀 학습 완료 (50회 업데이트)

[3/4] Injection: MARL 팀을 Population에 주입
[OK] Injection 완료

[4/4] 진화 (Selection, Crossover, Mutation)
[진화 완료] 총 7개 자식 생성
```

---

### 7️⃣ 학습 결과 시각화

```python
import pickle
import matplotlib.pyplot as plt

# Fitness history 로드
with open('models/fitness_history.pkl', 'rb') as f:
    history = pickle.load(f)

# 시각화
generations = list(range(1, len(history) + 1))
max_fitness = [s['max_fitness'] for s in history]
mean_fitness = [s['mean_fitness'] for s in history]

plt.figure(figsize=(10, 6))
plt.plot(generations, max_fitness, 'g-o', linewidth=2, label='Best', markersize=8)
plt.plot(generations, mean_fitness, 'b-s', linewidth=2, label='Mean', markersize=8)
plt.xlabel('Generation', fontsize=14)
plt.ylabel('Fitness (Sharpe Ratio)', fontsize=14)
plt.title('GA-MARL Training Progress (2021-2023)', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 통계
print(f"\n시작 평균: {mean_fitness[0]:.4f}")
print(f"최종 평균: {mean_fitness[-1]:.4f}")
print(f"개선율: {(mean_fitness[-1] - mean_fitness[0]) / abs(mean_fitness[0]) * 100:+.2f}%")
```

---

### 8️⃣ 테스트 실행 🧪

```python
# Out-of-sample Test (2024-H1)
# 학습에 사용하지 않은 새로운 데이터
!python test.py
```

**테스트 결과 예시:**
```
[테스트 결과]
================================================================================
                    Total Return  Sharpe Ratio  Max Drawdown  Calmar Ratio
--------------------------------------------------------------------------------
GA-MARL (Best)          +12.3%         1.234        -8.5%          1.45
Buy & Hold              +8.5%          0.987        -12.3%         0.69
Random Agent            -2.1%         -0.234        -15.2%        -0.14
KOSPI Index             +5.2%          0.654        -10.8%         0.48
================================================================================

🏆 Winner: GA-MARL (Best)
   Sharpe Ratio: 1.234
```

---

### 9️⃣ 결과 다운로드 (선택)

```python
# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 모델 저장
!cp -r models /content/drive/MyDrive/QuantMARLGA_models

print("✅ 모델이 Google Drive에 저장되었습니다!")
```

---

## 🎯 전체 실행 스크립트 (원클릭)

모든 단계를 한 번에 실행하려면:

```python
# === 원클릭 실행 스크립트 ===

# 1. 다운로드
!git clone https://github.com/YOUR_USERNAME/QuantMARLGA.git 2>/dev/null || echo "이미 존재"
%cd /content/QuantMARLGA/trading_marl_ga

# 2. 의존성
!pip install -q -r requirements.txt

# 3. Config 확인
!python check_config.py

# 4. 학습
print("\n" + "="*80)
print("학습 시작! (예상 시간: 1-2시간)")
print("="*80)
!python train.py

# 5. 테스트
print("\n" + "="*80)
print("테스트 시작!")
print("="*80)
!python test.py

# 6. 시각화
import pickle
import matplotlib.pyplot as plt

with open('models/fitness_history.pkl', 'rb') as f:
    history = pickle.load(f)

generations = list(range(1, len(history) + 1))
max_fitness = [s['max_fitness'] for s in history]
mean_fitness = [s['mean_fitness'] for s in history]

plt.figure(figsize=(12, 5))

# 학습 곡선
plt.subplot(1, 2, 1)
plt.plot(generations, max_fitness, 'g-o', linewidth=2, label='Best')
plt.plot(generations, mean_fitness, 'b-s', linewidth=2, label='Mean')
plt.xlabel('Generation', fontsize=12)
plt.ylabel('Fitness', fontsize=12)
plt.title('Training Progress', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)

# 개선율
plt.subplot(1, 2, 2)
improvement = [(mean_fitness[i] - mean_fitness[0]) / abs(mean_fitness[0]) * 100 
               for i in range(len(mean_fitness))]
plt.bar(generations, improvement, color='skyblue', edgecolor='navy')
plt.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
plt.xlabel('Generation', fontsize=12)
plt.ylabel('Improvement (%)', fontsize=12)
plt.title('Mean Fitness Improvement', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('models/training_summary.png', dpi=150)
plt.show()

print("\n✅ 모든 단계 완료!")
print("결과는 models/ 폴더에 저장되었습니다.")
```

---

## ⚙️ 고급 설정

### Config 수정

`config.py`를 직접 수정:

```python
# Colab 에디터로 열기
%load config.py
```

또는 직접 수정:

```python
# 예: Population 크기 증가
!sed -i 's/POPULATION_SIZE = 10/POPULATION_SIZE = 20/' config.py

# 예: 세대 수 증가
!sed -i 's/N_GENERATIONS = 12/N_GENERATIONS = 24/' config.py
```

---

## 🐛 Troubleshooting

### 1. Out of Memory

```python
# config.py 수정
BATCH_SIZE = 128  # 256 → 128
```

### 2. 학습이 너무 느림

```python
# GPU 확인
import torch
print(torch.cuda.is_available())  # True여야 함
print(torch.cuda.get_device_name(0))

# FP16 확인
from config import config
print(config.USE_AMP)  # True여야 함
```

### 3. 데이터 다운로드 오류

```python
# 인터넷 연결 확인
!ping -c 3 google.com

# Yahoo Finance 접근 확인
import yfinance as yf
test = yf.download("005930.KS", start="2024-01-01", end="2024-01-10")
print(test)
```

### 4. 모듈 import 오류

```python
# 경로 확인
import sys
print(sys.path)

# 수동 경로 추가
sys.path.insert(0, '/content/QuantMARLGA/trading_marl_ga')
```

---

## 📊 예상 성능

### 학습 시간 (Colab T4 GPU)
- 12 세대: **1-2시간**
- 24 세대: **2-4시간**

### GPU 메모리
- 사용량: **약 4GB**
- 권장: **8GB 이상**
- Colab 무료: **15GB** ✅

### 기대 결과
- **Sharpe Ratio**: 0.5 ~ 2.0
- **수익률**: 5% ~ 20%
- **Max Drawdown**: -5% ~ -15%

---

## 💡 Tips

1. **중간 저장**: 학습 중 세션이 끊기면 처음부터 다시 시작해야 함
   ```python
   # Google Drive 자동 저장 (선택)
   from google.colab import drive
   drive.mount('/content/drive')
   !ln -s /content/drive/MyDrive/QuantMARLGA_backup models
   ```

2. **로그 저장**: 터미널 출력을 파일로 저장
   ```python
   !python train.py 2>&1 | tee training.log
   ```

3. **백그라운드 실행**: 브라우저를 닫아도 계속 실행
   ```python
   # Colab Pro 필요
   # 또는 tmux 사용
   ```

---

## 🎓 다음 단계

1. ✅ **학습 완료** → `models/` 폴더 확인
2. 📊 **성능 분석** → Sharpe Ratio, Drawdown 확인
3. 🔧 **하이퍼파라미터 튜닝** → config.py 수정
4. 🚀 **실전 적용** → 최신 데이터로 재학습

---

**Happy Trading! 📈🚀**

*문제가 있으면 GitHub Issues에 올려주세요!*

