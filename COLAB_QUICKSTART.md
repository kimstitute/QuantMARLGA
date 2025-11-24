# 🚀 Google Colab 빠른 시작 가이드

**QuantMARLGA를 Google Colab에서 실행하는 방법**

---

## 📋 목차

1. [준비사항](#준비사항)
2. [빠른 실행](#빠른-실행)
3. [단계별 실행](#단계별-실행)
4. [문제 해결](#문제-해결)

---

## 준비사항

### 1. GitHub Personal Access Token (필수)

Private 레포지토리 클론을 위해 필요합니다.

**생성 방법:**
1. https://github.com/settings/tokens 접속
2. `Generate new token (classic)` 클릭
3. `repo` 권한 체크
4. `Generate token` 클릭
5. 생성된 토큰 복사 (한 번만 표시됨!)

### 2. OpenDart API Key (선택사항)

펀더멘털 데이터(ROE, 부채비율)를 위해 권장됩니다.

**생성 방법:**
1. https://opendart.fss.or.kr/ 접속
2. 회원가입 및 로그인
3. `인증키 신청/관리` 메뉴에서 발급

> 💡 **Tip**: OpenDart Key가 없어도 실행 가능 (PER/PBR만 사용)

---

## 빠른 실행

### Step 1: Colab 노트북 생성

1. https://colab.research.google.com/ 접속
2. `새 노트북` 클릭
3. **GPU 활성화** (권장):
   - 메뉴: `런타임` → `런타임 유형 변경` → `T4 GPU` 선택

### Step 2: 코드 복사

`colab_train_test.py` 파일의 각 CELL을 순서대로 Colab의 새 셀에 복사합니다.

### Step 3: 실행

```python
# CELL 1부터 순서대로 실행 (Shift + Enter)
```

**예상 소요 시간:**
- 환경 설정: ~2분
- 종목 선정: ~5분
- 학습 (GPU): ~30-40분
- 테스트: ~3분
- **총합: 약 40-50분**

---

## 단계별 실행

### CELL 1: 환경 설정 ⚙️

**내용:**
- 필요한 패키지 설치
- GitHub 레포지토리 클론
- 작업 경로 설정

**입력 필요:**
- GitHub Personal Access Token

**예상 시간:** 2분

---

### CELL 2: OpenDart API Key 🔑

**내용:**
- OpenDart API Key 설정

**입력 필요:**
- OpenDart API Key (선택사항)

**예상 시간:** 10초

**Skip 가능**: Key가 없어도 진행 가능

---

### CELL 3: 종목 Universe 선정 📊

**내용:**
- 2021-2024 전체 기간 데이터 로드
- 결측치 없는 상위 30개 종목 선정
- `data/selected_tickers.pkl` 저장

**출력 예시:**
```
선정 종목: 30개
상위 5개: 005930, 000660, 373220, 207940, 005380
```

**예상 시간:** 5분

---

### CELL 4: 학습 실행 🎯

**내용:**
- 2021-2023 Rolling Window 학습
- 12세대 진화 (분기별)
- Population: 10개 EA + 1개 MARL

**출력 예시:**
```
세대  1 (2021-Q1): 0.5234 ████████████
세대  2 (2021-Q2): 0.7891 ██████████████
...
세대 12 (2023-Q4): 3.3220 ████████████████████████████████
```

**예상 시간:** 30-40분 (GPU), 2-3시간 (CPU)

**저장:**
- `models/best_system/` - 최고 성능 모델
- `models/metadata.pkl` - 학습 정보 + 종목 리스트
- `models/fitness_history.pkl` - 학습 곡선

---

### CELL 5: 테스트 실행 🧪

**내용:**
- 2024 Out-of-sample 테스트
- 학습과 동일한 30개 종목 사용
- 벤치마크 비교 (Buy & Hold, Random, KOSPI)

**출력 예시:**
```
전략              수익률   샤프   MDD
GA-MARL (Best)   21.46%  2.264  -8.26%
Buy & Hold       22.14%  2.332  -7.85%
KOSPI Index       4.79%  0.688  -8.76%
```

**예상 시간:** 3분

**저장:**
- `models/test_results.pkl` - 테스트 결과

---

### CELL 6: Google Drive 저장 💾

**내용:**
- 모든 결과를 Google Drive에 백업
- Colab 세션 종료 후에도 보존

**저장 경로:**
```
MyDrive/QuantMARLGA_results/
├── best_system/          # 모델 파라미터
├── metadata.pkl          # 학습 정보
├── fitness_history.pkl   # 학습 곡선
├── test_results.pkl      # 테스트 결과
└── selected_tickers.pkl  # 종목 리스트
```

**예상 시간:** 1분

---

### CELL 7: 결과 시각화 📈

**내용:**
- 학습 곡선 그래프
- 테스트 성과 비교 차트
- 상세 결과 출력

**예상 시간:** 30초

---

### CELL 8: 이전 결과 불러오기 🔄

**내용:**
- Google Drive에서 이전 결과 불러오기
- 재실행 없이 시각화만 확인

**사용 시나리오:**
- Colab 세션이 끊긴 경우
- 다른 Colab 노트북에서 확인
- 재학습 없이 결과만 확인

**예상 시간:** 30초

---

## 문제 해결

### 1. "fatal: could not read Username" 오류

**원인:** GitHub Token이 잘못되었거나 만료됨

**해결:**
1. 새 Token 생성 (https://github.com/settings/tokens)
2. `repo` 권한 확인
3. CELL 1 다시 실행

---

### 2. "CUDA out of memory" 오류

**원인:** GPU 메모리 부족

**해결:**
```python
# config.py 수정 (CELL 1 실행 후)
!sed -i 's/BATCH_SIZE = 64/BATCH_SIZE = 32/' trading_marl_ga/config.py
```

또는:
- 런타임 재시작: `런타임` → `런타임 다시 시작`
- CPU 사용 (느리지만 안정적)

---

### 3. "No module named 'yfinance'" 오류

**원인:** 패키지 설치 실패

**해결:**
```python
!pip install --upgrade yfinance pandas numpy torch scipy opendart-python
```

---

### 4. 학습 중 Colab 세션 끊김

**원인:** 90분 이상 무작동

**해결:**
1. **사전 예방**: 학습 중 가끔 새 셀 실행
   ```python
   import time
   print(f"Keep alive: {time.time()}")
   ```

2. **끊긴 후**: CELL 8로 이전 결과 불러오기
   - Google Drive에 저장된 마지막 모델 확인
   - 필요 시 해당 세대부터 재학습

---

### 5. 종목 데이터 수집 실패

**원인:** Yahoo Finance API 오류

**해결:**
```python
# 캐시 삭제 후 재시도
!rm -rf data/cache
!python trading_marl_ga/select_universe.py
```

---

## 💡 유용한 팁

### 1. 중간 체크포인트 저장

학습 중간에 CELL 6 실행하여 Google Drive에 저장:
```python
# 매 4세대마다 실행 권장
```

### 2. GPU 사용량 확인

```python
# 새 셀에서 실행
!nvidia-smi
```

### 3. 실행 시간 측정

```python
# 셀 맨 위에 추가
%%time
```

### 4. 로그 저장

```python
# 셀 맨 위에 추가
%%capture output
# ... 실행 코드 ...

# 셀 맨 아래에 추가
with open('logs/training.log', 'w') as f:
    f.write(output.stdout)
```

---

## 📊 기대 성과

### 학습 진행

```
세대  1: Fitness  0.50 (초기화)
세대  3: Fitness  1.20 (개선 중)
세대  6: Fitness  2.10 (수렴 중)
세대 12: Fitness  3.30 (최종)
```

### 테스트 결과

**목표:**
- 샤프 비율: 2.0 이상
- 총 수익률: 15-25%
- MDD: -10% 이내

**벤치마크 대비:**
- KOSPI 대비: +15-20%p
- Buy & Hold 근접

---

## 🔗 참고 링크

- **GitHub**: https://github.com/kimstitute/QuantMARLGA
- **Token 생성**: https://github.com/settings/tokens
- **OpenDart**: https://opendart.fss.or.kr/
- **Colab**: https://colab.research.google.com/

---

## 📝 라이센스

MIT License

---

**마지막 업데이트**: 2025-11-24  
**작성자**: AI Assistant

