# PA2: Active Data Curation Challenge

**Scene Classification for Autonomous Driving**

> *"Deep Learning is not just about the model; it's about the data that feeds it."*

본 저장소는 대학원 수업 **AUE8088 — 딥러닝의 이해 및 활용 (2026년)** 의 두 번째 Programming Assignment(PA2) 안내 및 스켈레톤 코드입니다. 학생 여러분은 본 README를 정독한 후 과제를 시작해 주세요.

---

## 1. 과제 개요 (Overview)

자율주행 시스템은 단순히 차량/보행자 같은 **객체(Object)** 를 인식하는 것을 넘어, 시스템이 처한 **환경(Scene)** — 날씨, 장소, 시간대 등 — 을 정확히 파악할 수 있어야 합니다. 본 과제에서는 자율주행 표준 데이터셋인 **BDD100K** 의 속성(Attribute) 라벨을 활용하여 **Scene Classification** 모델을 구축하고, 더 나아가 **데이터의 분포와 품질이 모델 성능에 미치는 영향** 을 직접 탐구합니다.

본 과제의 두 축은 다음과 같습니다.

1. **Model-centric**: 고전적 CNN(VGG, ResNet)부터 최신 Vision Transformer(ViT, Swin)까지 직접 구현·비교하며, 백본의 발전사를 체감합니다.
2. **Data-centric**: 한정된 라벨링 자원(1,000장) 안에서 *어떤* 데이터를 추가로 학습시킬 때 성능이 가장 효율적으로 향상되는지 — 즉 **Data Curation 전략** — 을 직접 설계합니다.

> *"이번 PA는 **'최고의 모델'**을 찾는 경쟁이자, 동시에 **'가장 가치 있는 데이터'**를 찾는 보물찾기입니다."*

---

## 2. 데이터셋 (Dataset)

본 과제는 [BDD100K](https://bdd-data.berkeley.edu/) 의 Scene Attribute(Weather / Scene Type / Time of Day) 라벨을 사용하며, **모든 이미지는 224×224 로 리사이징** 되어 배포됩니다 (Colab T4 환경 고려).

### 2.1 분류 대상 (Classes) — **Multi-task**

본 과제는 BDD100K의 3가지 Scene 속성을 **동시에 예측하는 Multi-task Learning** 으로 구성됩니다. 모델은 단일 백본 위에 3개의 분류 head를 가지며, 손실은 3개 task의 Cross-Entropy 합(또는 가중합)으로 정의됩니다.

| 속성 (Attribute) | 클래스 수 | 클래스 (Classes) |
|---|---|---|
| **Weather** | 6 | `clear`, `overcast`, `rainy`, `snowy`, `foggy`, `partly cloudy` |
| **Scene Type** | 3 | `city street`, `highway`, `residential` |
| **Time of Day** | 3 | `daytime`, `night`, `dawn/dusk` |

> **Multi-task의 의미**: 자율주행 시스템이 처한 환경은 단순히 "비가 온다" 가 아니라 "*고속도로*에서 *밤*에 *비*가 온다" 같이 **다중 속성의 조합**으로 정의됩니다. 본 과제는 이 조합적 환경 인식을 직접 다룹니다.

각 속성은 독립적으로 라벨링되어 있으나, 분포는 강한 상관(예: snowy + night)을 갖습니다. Level 5에서 "어떤 *조합*이 부족한가" 를 분석하는 것이 Curation 전략의 핵심이 됩니다.

### 2.2 데이터 구성 (Set A & Set B)

전체 데이터는 두 개의 풀(Pool)로 분리되어 제공됩니다.

| 구분 | 성격 | 내용 | 학생 활용 범위 |
|---|---|---|---|
| **Set A** *(Standard)* | 베이스라인 학습용 | Train (라벨) / Val (라벨) / **Test (라벨 비공개)**, **Imbalanced** | Level 1 ~ 4 |
| **Set B** *(Mining Pool)* | 데이터 광산 | BDD100K 이미지 약 1.5만 장, **라벨 공개** | Level 5 (최대 **1,000장** 선별) |

**Set A의 Class Imbalance**: 실제 도로 환경의 빈도를 반영하여 클래스 불균형이 의도적으로 강하게 설정되어 있습니다 (예: `clear` 60%+ vs `snowy`/`foggy` ~1%). 단순 Cross-Entropy로는 다수 클래스에 편향됩니다.

**Set A `test`**: 이미지만 공개되고 라벨은 Kaggle 채점용으로 비공개입니다. 학생은 이 split 에 대해 inference 후 Kaggle Leaderboard 로만 성능 검증이 가능합니다.

**Set B 라벨 공개**: Level 5 의 평가 본질은 *"주어진 풀에서 어떤 1,000장이 가장 가치 있는가"* 이므로, 학생은 Set B 의 라벨을 자유롭게 활용해 선별 전략(class balancing, hard example mining, diversity 등) 을 설계할 수 있습니다.

### 2.3 데이터 다운로드 및 디렉토리 구조

```bash
# 학기 초 공지된 링크를 통해 다운로드 후 압축 해제
data/
├── set_a/
│   ├── train/             # Imbalanced (~5,000 장)
│   ├── val/               # ~1,000 장
│   ├── test/              # 이미지만 공개, 라벨은 Kaggle 채점용으로 비공개
│   ├── labels.json        # train + val 의 (weather, scene, timeofday) 만 포함
│   ├── train_ids.txt
│   ├── val_ids.txt
│   └── test_ids.txt       # test 는 이미지 ID 만
├── set_b/
│   ├── images/            # ~15,000 장
│   ├── labels.json        # 전체 (image_id, weather, scene, timeofday) 라벨 공개
│   └── metadata.json      # 이미지 ID 리스트
└── README.md
```

---

## 3. 과제 단계 (Levels)

학생 여러분은 아래 5개 Level을 순차적으로 수행하며, 각 단계의 결과를 **하나의 통합 리포트** 와 **Jupyter 노트북** 에 정리합니다.

> **모델 구현 정책 (Level 1 ~ 2 공통)**
> `torchvision.models`, `timm`, 기타 사전 정의된 모델 라이브러리의 **모델 코드 사용은 금지** 합니다 (예: `torchvision.models.resnet18()`, `timm.create_model(...)` 등). 공식 구현체나 논문 코드를 **참고하여 직접 타이핑** 하는 것은 허용합니다 — 한 줄씩 따라 적는 과정에서 백본의 구조를 체화하는 것이 본 단계의 목적입니다.

### Level 1 — Milestone Model Implementation (Classic CNNs) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IRCVLab/2026-HYU-AUE8088-PA2/blob/main/notebooks/level1_classic_cnns.ipynb)


- **VGG16** 과 **ResNet-18 / ResNet-50** 을 PyTorch로 **직접 구현**.
- 백본 위에 **3개의 분류 head (Weather 6-class, Scene 3-class, Time 3-class)** 를 추가하여 Multi-task로 학습.
- **분석 포인트**: (a) Skip Connection 유무가 깊은 네트워크의 수렴에 미치는 영향, (b) 3개 task의 loss 가중치 설정이 결과에 미치는 영향.

### Level 2 — SOTA Model Application (Vision Transformers) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IRCVLab/2026-HYU-AUE8088-PA2/blob/main/notebooks/level2_transformers.ipynb)

- **ViT-S/16** 또는 **Swin-Tiny** 를 PyTorch로 **직접 구현** (Level 1과 동일하게 3-head Multi-task).
- ImageNet **pretrained weight 텐서(.pth) 로드는 허용** — 단, 사용 여부와 출처를 명시. (모델 라이브러리를 import하는 것이 아니라, 본인이 구현한 모델의 `state_dict` 에 외부에서 다운로드한 weight를 매핑하여 로드하는 방식입니다.)
- **분석 포인트**: CNN 대비 Attention 기반 모델의 (a) 데이터 효율성, (b) Inductive bias 부재가 소규모·불균형 데이터셋에서 갖는 영향.

### Level 3 — Handling Imbalance & Advanced Augmentation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IRCVLab/2026-HYU-AUE8088-PA2/blob/main/notebooks/level3_imbalance.ipynb)

Set A의 Class Imbalance를 정면으로 다룹니다. 아래 기법 중 **최소 2가지 이상** 을 조합하여 실험.

- **Loss-level**: Weighted Cross-Entropy, Focal Loss, LDAM, Class-Balanced Loss → **속성별로 다른 loss를 적용해도 무방**
- **Sampling-level**: Class-Balanced Sampler. *Multi-task의 까다로움* — 어느 속성 기준으로 sampling 가중치를 줄지 직접 설계 필요.
- **Augmentation-level**: RandAugment, Mixup, CutMix, AugMix. (Mixup/CutMix의 라벨 처리 방식을 Multi-task에 맞게 확장하는 것이 본 Level의 도전 과제)

각 기법이 **소수 클래스**와 **다수 클래스**, 그리고 **각 속성** 에 미치는 영향을 분리하여 보고하세요.

### Level 4 — Deep Analysis & Interpretation [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IRCVLab/2026-HYU-AUE8088-PA2/blob/main/notebooks/level4_xai_efficiency.ipynb)

성능 숫자 너머 *왜 이런 결과가 나오는지* 를 설명할 수 있어야 합니다.

- **XAI (Grad-CAM)**: 동일 이미지에 대해 **세 head가 각각 어디를 보는지** 시각화. (`weather` head는 하늘을, `scene` head는 도로 구조를, `timeofday` head는 광원을 보는지?) Multi-task 학습이 head 간 attention을 어떻게 분산시키는지 관찰.
- **Confusion Matrix 분석**: 속성별 3개 Confusion Matrix 를 모두 그리고, 어떤 클래스 쌍이 가장 혼동되는지·원인이 *텍스처*인지 *광원*인지 가설 수립.
- **Efficiency Trade-off**: T4 GPU 기준 모델별 **FPS** 와 Avg-Macro-F1 의 Trade-off를 Pareto front 형태로 작성. (Params·FLOPs는 선택적으로 첨부 가능)

### Level 5 — Data Mining Challenge: *"The 1,000-Pick"* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/IRCVLab/2026-HYU-AUE8088-PA2/blob/main/notebooks/level5_data_mining.ipynb)

> 이 단계가 본 PA의 핵심이자 **차별화 점수**입니다.

- **규칙**: Set B(라벨 미공개)에서 최대 **1,000장**을 선택하여 Set A에 추가, 다시 학습.
- **선택 자유도**: 어떤 기준으로 고를지는 **전적으로 학생 자율**.
  - 후보 전략: Class Balancing(*어느 속성 기준?*), Hard Example Mining(Uncertainty), Diversity(Feature-space clustering), Pseudo-labeling 기반 Active Learning, **속성 조합(예: rainy+night) 우선 탐색**, ...
  - Multi-task인 만큼 **3개 속성 중 어디에 우선순위를 둘지** 가 중요한 의사결정. 예를 들어 weather의 snowy를 채우는 것과 timeofday의 dawn/dusk를 채우는 것 중 어느 쪽이 Avg-Macro-F1을 더 끌어올릴지를 설계해야 합니다.
- **제출**: 선택한 이미지 ID 리스트(`level5_picks.json`) + 선택 사유 및 알고리즘이 담긴 **Curation Report**.
- **평가 (Data Intelligence Score)**: 무작위로 1,000장을 뽑은 Random Baseline 대비 **얼마나 더 큰 Avg-Macro-F1 향상** 을 이끌어냈는지를 정량 평가.

---

## 4. Kaggle Competition

본 PA의 최종 성능 평가는 별도로 개설된 **Kaggle In-Class Competition** 을 통해 이루어집니다.

| 구분 | 비중 | 구성 |
|---|---|---|
| **Public Leaderboard** | 40% | Set A 의 분포와 유사한 일반 도로 환경 |
| **Private Leaderboard** | 60% | **OOD(Out-of-Distribution)** 및 Edge Case 집중 — 역광 터널 출구, 폭설로 차선이 안 보이는 도로, 렌즈 오염 등 |

> **주의**: Public 점수에 과적합(LB Overfitting)되는 순간 Private에서 무너집니다. 진정한 성능은 **Robustness** 에서 나옵니다.

- **Submission Format**: `image_id, weather` 컬럼을 가진 CSV (예시는 `submission/sample_submission.csv` 참고).
- **제출 횟수**: 일 5회, 최종 2개 submission 선택.
- **외부 데이터/사전학습 가중치**: ImageNet pretrained weight는 허용하되, BDD100K 원본 라벨이나 외부 라벨 데이터 사용은 금지.

---

## 5. 평가 메트릭 (Evaluation Metrics)

Class Imbalance가 강한 만큼 **Accuracy 단일 지표는 사용하지 않습니다**.

- **Primary (Kaggle 공식)**: **Average Macro-F1** — 3개 속성(Weather/Scene/Time) 각각의 Macro-F1 을 산출한 뒤 단순 평균.

  ```
  Avg-MF1 = (MF1_weather + MF1_scene + MF1_timeofday) / 3
  ```

- **Secondary**: **mAP (mean Average Precision)** — 각 속성별로 클래스별 AP를 평균낸 뒤 다시 3개 속성에 대해 평균. Class Imbalance 하에서 모델의 *순위 매기기* 품질을 보완 평가.
- **필수 시각화 (모든 Level의 리포트에 포함)**:
  - **각 속성별 Confusion Matrix (정규화)** — 총 3개. 어떤 클래스 쌍이 혼동되는지 반드시 분석.
  - 각 속성별 Per-class Precision / Recall / F1 표
  - Top-1 Accuracy (속성별) 및 Worst-class Accuracy
- **Efficiency (Level 4)**: **FPS (Frames Per Second)** on T4 GPU. (배치 크기 1, 224×224 입력, warm-up 후 측정값의 평균.) — Multi-task 모델은 하나의 forward pass로 3개 head를 모두 출력하므로 단일 FPS 값.
- **Data Intelligence Score (Level 5)**:

  ```
  DI = (Macro-F1[student picks] − Macro-F1[random picks]) / Macro-F1[random picks]
  ```

  Random Baseline은 조교가 동일 시드로 Set B에서 1,000장을 뽑아 학습한 모델로 사전 산출하여 공지합니다.

---

## 6. 제출물 (Deliverables)

다음 3가지를 LMS에 제출합니다. **Kaggle 제출은 학생 본인이 Kaggle 페이지에 직접 업로드** 하며, LMS에는 별도 CSV를 제출하지 않습니다.

1. **Jupyter Notebook** (`pa2_<학번>_<이름>.ipynb`)
   - Level 1 ~ 5 의 모든 실험이 재현 가능한 형태로 포함.
   - Colab T4에서 처음부터 끝까지 실행되어야 함 (`Run All`).
2. **Final Report (PPT, `.pptx`)** — `report_<학번>.pptx`
   - 슬라이드 약 **15 ~ 25장** 권장.
   - 권장 구성: 표지 → Level별 핵심 결과(모델 비교 표, 학습 곡선, Confusion Matrix, Grad-CAM) → Level 5 Curation 전략과 근거 → 결론·한계.
   - 그림·표·수치 위주로 시각적으로 명확하게 구성.
3. **Curation Artifacts** — `level5_picks.json` (선택한 이미지 ID 리스트와 선택 사유 메타데이터)

> **선택 과제 (Optional, Bonus +5%)**: **Set B 활용 Self-Supervised Pre-training**. 라벨 없는 Set B 이미지로 SimCLR / MoCo / MAE 등 SSL 기법을 사용하여 백본을 사전학습한 뒤, Set A로 Fine-tuning. Set B를 "1,000장 라벨 후보"가 아니라 "라벨 없이도 가치있는 큰 자원" 으로 활용하는 관점입니다.

### 6.1 재현성 정책 (Reproducibility) — **★ 중요**

채점 과정에서 **조교가 학생의 코드를 받아 실제로 다시 학습/평가를 진행** 합니다. 다음을 반드시 준수해 주세요.

- **재현 불가 시 취득 점수의 50%만 인정** 됩니다. (예: 80점을 받을 만한 결과물이 재현 불가라면 40점 처리)
- **재현 가능성 체크리스트**:
  - [ ] 본 저장소가 제공하는 `requirements.txt` 기준 가상환경에서 노트북이 처음부터 끝까지 에러 없이 실행됨
  - [ ] **랜덤 시드 고정** (`torch`, `numpy`, `random`, `cudnn.deterministic` 등) 및 시드 값 README/노트북에 명시
  - [ ] 데이터 경로는 상대 경로(`./data/...`) 또는 환경 변수로 처리
  - [ ] 외부 가중치(.pth) 사용 시 다운로드 스크립트 또는 URL을 노트북에 포함
  - [ ] 학습 시간이 1시간을 초과하는 경우 사전 학습된 체크포인트(`.pth`)를 함께 제출하여 평가 단계만 재현 가능하도록 구성

> 동일 시드·동일 환경에서 **메트릭 수치가 ±1.0 Macro-F1 이내로 일치** 하면 재현 성공으로 간주합니다.

---

## 7. 채점 기준 (Grading)

총 **100점** + 선택 과제 보너스 **5점**.

| 항목 | 배점 | 세부 |
|---|---|---|
| Level 1 — Classic CNN 구현·학습 | 10 | 직접 구현 정확성, 학습 안정성, 분석 |
| Level 2 — Transformer 적용 | 10 | 적용 적절성, CNN과의 정량 비교 |
| Level 3 — Imbalance 대응 | 15 | 기법 다양성, 소수 클래스 개선 폭 |
| Level 4 — XAI & Efficiency | 15 | Grad-CAM 해석 깊이, Trade-off 분석 |
| **Level 5 — Data Mining (★ 핵심)** | **25** | Curation 전략의 창의성·논리성, DI Score |
| Kaggle Private LB 순위 | 15 | 상위 백분율 기반 (예: top 10% 만점) |
| Final Report 품질 | 10 | 구조, 시각화, 논리 전개, 가독성 |
| **Bonus**: SSL Pre-training | +5 | Set B 활용 Self-Supervised Pre-training |

### 7.1 금지 사항 (Restrictions)

- **모델 라이브러리 사용 금지**: `torchvision.models`, `timm` 등 사전 정의된 백본을 import하여 그대로 사용하는 것은 모든 Level에서 금지. (참고하여 직접 타이핑하는 것은 허용)
- **BDD100K 사전학습 가중치 사용 금지**: ImageNet pretrained는 허용하나, BDD100K 자체로 사전학습된 weight는 사용 불가.
- **Set B 라벨 역추정 금지**: 메타데이터(timestamp, GPS 등)나 외부 BDD100K 라벨을 통해 Set B의 정답을 추정하는 행위는 0점 처리.
- **외부 라벨 데이터 사용 금지**: BDD100K 외부의 추가 라벨링 데이터를 학습에 사용하는 것은 금지.
- **표절 (Plagiarism)**: 타인의 코드/리포트 표절은 즉시 0점 처리. 외부 코드 참조 시 README와 노트북 셀에 **출처를 반드시 명시**. 같은 강의 수강생 간 코드 공유도 금지입니다.
- **재현 불가**: §6.1 재현성 정책에 따라 취득 점수의 50%만 인정.

---

## 8. 일정 (Schedule)

| 마일스톤 | 일자 |
|---|---|
| 과제 공지 및 데이터 배포 | TBA |
| Kaggle Competition Open | TBA |
| **최종 제출 마감** | TBA *(약 4주 후)* |
| Kaggle Final LB 공개 | 마감 직후 |
| 우수 사례 발표 | 마감 + 1주 |

> 정확한 날짜는 LMS 공지를 참조하세요.

---

## 9. 개발 환경 (Environment)

- **Computing**: Google Colab (T4 GPU 권장) 또는 동급 로컬 환경
- **Framework**: PyTorch ≥ 2.1
- **환경 관리**: `pip` + Python `venv` (재현성을 위한 단일 소스: `requirements.txt`)
- **주요 라이브러리**: `torchvision`(데이터·트랜스폼만 사용, 모델 import 금지), `captum`(XAI), `albumentations`(augmentation), `wandb`(선택)
- **금지 라이브러리**: `timm`, `torchvision.models` (모델 코드 import 용도)

### 9.1 로컬 환경 구성 — Python `venv` 사용

다른 프로젝트와의 의존성 충돌을 막기 위해 **반드시 가상환경 안에서 작업**하세요. Python 3.10 이상을 사용합니다.

```bash
# 저장소 루트에서 — 프로젝트 전용 가상환경 생성 (.venv 디렉토리)
python3.10 -m venv .venv

# 활성화 (macOS / Linux)
source .venv/bin/activate
# 활성화 (Windows PowerShell)
# .venv\Scripts\Activate.ps1

# pip 최신화
pip install --upgrade pip

# (1) PyTorch + torchvision 을 본인 환경에 맞춰 별도 설치
#     CUDA 버전은 https://pytorch.org/get-started/locally 에서 확인 후 명령어 복사.
#     예시 (CUDA 12.1):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#     CPU 전용:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# (2) 나머지 의존성 설치
pip install -r requirements.txt

# 작업 종료 시 비활성화
deactivate
```

> **PyTorch 를 별도로 설치하는 이유**: `requirements.txt` 에 torch 를 핀하면 (a) Colab 의 사전 설치본과 충돌하고 (b) Python 버전·CUDA 버전 조합마다 wheel 가용성이 달라 학생 환경에서 설치 실패가 잦아집니다. 본 저장소의 코드는 **torch ≥ 2.3** 에서 검증되었습니다.

> 가상환경이 활성화되면 프롬프트 앞에 `(.venv)` 가 표시됩니다. 모든 `python`, `pip`, `jupyter` 명령은 이 환경 안에서만 실행됩니다 — 시스템 Python 이나 다른 프로젝트의 패키지에 영향을 주지 않습니다.

> 이미 conda 사용자라면 동일한 패턴으로 `conda create -n aue8088-pa2 python=3.10` 후 `conda activate aue8088-pa2 && pip install -r requirements.txt` 가능합니다 — 단, 의존성 정의는 `requirements.txt` 한 곳에서만 관리됩니다.

### 9.2 Colab 환경 구성 (브라우저)

Colab은 venv가 필요 없습니다 (런타임 자체가 일회성). PyTorch 는 Colab 에 **이미 사전 설치**되어 있으므로 추가 설치 불필요. 노트북 첫 셀에서:

```python
!pip install -r requirements.txt
```

> Colab 의 사전 설치 torch 를 그대로 사용하는 것이 가장 안전합니다. `pip install torch==<version>` 으로 강제 다운그레이드하면 CUDA stack 이 깨질 수 있습니다.

### 9.3 VS Code 에서 Colab Runtime 사용 (권장)

본 PA는 `src/` 아래에 모듈이 여러 파일로 흩어져 있어, 단일 노트북 위주의 브라우저 Colab 보다는 **VS Code 의 멀티-파일 편집 환경** 이 훨씬 편리합니다. Google 이 제공하는 공식 확장을 사용하면 VS Code 안에서 Colab 의 T4 GPU 런타임에 직접 연결할 수 있습니다.

**사전 요구**:
- VS Code 1.85 이상
- Google 계정 (Colab 무료 계층 가능)

**설치 및 연결 절차**:

1. VS Code 의 Extensions 패널 (`⌘+Shift+X` / `Ctrl+Shift+X`) 에서 **"Google Colab"** 으로 검색해 **Google 공식 확장** (Publisher: Google) 을 설치합니다.
2. `File → Open Folder...` 로 본 저장소 루트 (`2026-aue8088-pa2/`) 를 엽니다.
3. Command Palette (`⌘+Shift+P` / `Ctrl+Shift+P`) 에서 **`Colab`** 으로 검색하면 사용 가능한 명령들이 표시됩니다. `Colab: Sign In` (또는 유사한 이름) 으로 Google 계정 로그인.
4. 같은 Command Palette 에서 `Colab: Connect to Runtime` 실행 → 런타임 유형으로 **T4 GPU** 선택.
5. `notebooks/level1_classic_cnns.ipynb` 등을 열고, 노트북 우상단의 **커널(Kernel) 선택** 드롭다운에서 방금 연결한 **Colab Runtime** 을 선택합니다.
6. 노트북의 첫 셀에서 의존성 설치 — `!pip install -r requirements.txt` (브라우저 Colab 과 동일).

> 확장의 UI/명령 이름은 버전에 따라 조금씩 달라질 수 있습니다. 최신 사용법은 확장 페이지의 README 를 참고하세요.

**데이터 / 체크포인트 동기화**:
Colab 런타임의 `/content/` 파일은 세션이 끊기면 사라집니다. `data/` 와 `checkpoints/` 는 다음 중 하나로 보존하세요.

```python
# 옵션 A: Google Drive 마운트 (가장 일반적)
from google.colab import drive
drive.mount(\"/content/drive\")
# 이후 ./data 를 /content/drive/MyDrive/aue8088-pa2/data 로 심볼릭 링크
```

또는 본인 fork 한 GitHub 저장소를 `git clone` 으로 받아오는 방식도 가능합니다.

**장점**:
- VS Code 의 자동완성·정의 이동(F12)·멀티-파일 편집·Git 패널·Copilot 등을 모두 사용하면서 Colab 의 T4 GPU 활용
- `src/` 의 헬퍼 모듈을 노트북과 나란히 띄워놓고 양방향 편집 가능
- 학습 로그가 VS Code 의 출력 패널 / 터미널에 직접 표시되어 추적 용이

**유의 사항**:
- Colab 무료 계층은 약 90분~수 시간 후 세션이 자동 종료될 수 있으므로 **매 N epoch 체크포인트 저장** 을 습관화하세요.
- T4 GPU 할당은 시간대/가용량에 따라 거절될 수 있습니다 — 잠시 후 재시도.

처음 사용하는 학생은 §9.2 의 브라우저 Colab 으로 친숙해진 뒤, 코드 분량이 늘어날 때 본 절차로 옮겨오는 것을 권장합니다.

### 9.4 Weights & Biases (wandb) 로깅

본 저장소의 학습 노트북(Level 1·2·3·5)은 `MultiTaskTrainer` 가 매 epoch 의 train loss / val Avg-Macro-F1 / 속성별 MF1 / learning rate 를 자동으로 wandb에 기록하도록 통합되어 있습니다. Level 4 분석 노트북은 confusion matrix·Grad-CAM 패널·FPS 표를 산출물로 업로드합니다.

**활성화 (권장)**:

```bash
pip install wandb       # 이미 requirements.txt 에 포함됨
wandb login             # 최초 1회 — https://wandb.ai/authorize 에서 API key 복사
```

각 노트북 상단의 `WANDB_PROJECT = \"aue8088-pa2\"` 를 본인 프로젝트 이름으로 변경하세요. 같은 프로젝트에 누적된 Run 들은 wandb 대시보드에서 자동으로 비교 가능합니다.

**비활성화**: 노트북 상단의 `WANDB_PROJECT = None` 으로 두면 모든 wandb 호출이 자동으로 no-op 으로 동작합니다 (학습 자체에는 영향 없음). 환경변수로도 끌 수 있습니다:

```python
import os; os.environ[\"WANDB_DISABLED\"] = \"true\"
```

> **재현성 채점 시**: 조교가 학생 코드를 실행할 때는 `WANDB_DISABLED=true` 로 설정한 상태에서 평가합니다 — 학생이 wandb 활성화 상태로 코드를 두어도 채점에는 영향이 없습니다.

---

## 10. 학습 설정 (Training Configuration)

### 10.1 랜덤 시드 (Random Seed)

모든 노트북에서 `SEED = 42` 로 통일하여 실험 재현성을 보장합니다. 각 노트북의 `setup` 셀에서 선언됩니다.

```python
SEED = 42   # 전체 실험 고정 시드
set_seed(SEED, deterministic=True)
```

`src/utils/seed.py` 의 `set_seed` 함수는 다음을 모두 고정합니다.

| 대상 | 처리 방식 |
|---|---|
| Python 표준 `random` | `random.seed(SEED)` |
| NumPy | `np.random.seed(SEED)` |
| PyTorch CPU | `torch.manual_seed(SEED)` |
| PyTorch CUDA | `torch.cuda.manual_seed_all(SEED)` |
| cuDNN 결정론적 연산 | `torch.backends.cudnn.deterministic = True` |
| cuDNN autotuner 비활성화 | `torch.backends.cudnn.benchmark = False` |

DataLoader 의 worker 시드는 `seed_worker` 함수를 `worker_init_fn` 으로 전달하여 추가 고정합니다.

```python
g = torch.Generator()
g.manual_seed(SEED)
DataLoader(..., worker_init_fn=seed_worker, generator=g)
```

> 동일 시드·동일 환경에서 Avg Macro-F1 이 ±1.0 이내로 재현되면 재현 성공으로 간주합니다 (§6.1).

---

### 10.2 레벨별 학습 하이퍼파라미터

#### Level 1 — Classic CNNs (VGG-16, ResNet-18/50)

| 파라미터 | 값 |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 5e-4 |
| LR Scheduler | CosineAnnealingLR (T_max = epochs) |
| Epochs | 30 |
| Batch Size | 64 |
| Loss | CrossEntropyLoss (각 속성 균등 가중합) |
| Sampler | shuffle (불균형 보정 없음) |
| Augmentation | `train_transform()` — RandomResizedCrop(224) + RandomHFlip + Normalize |
| Gradient Clip | 없음 |

- UncertaintyWeighting 실험: `log_σ` 파라미터를 AdamW 에 함께 포함하여 학습.

---

#### Level 2 — Vision Transformers (ViT-S/16, Swin-Tiny)

| 파라미터 | 값 |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 5e-4 |
| LR Scheduler | CosineAnnealingLR (T_max = 30) |
| Epochs | 30 |
| Batch Size | 64 |
| Loss | CrossEntropyLoss |
| Sampler | shuffle |
| Augmentation | Level 1 과 동일 |
| Gradient Clip | 없음 |

- **DeiT-S/16 사전학습 가중치** (선택): `https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth`
- **Swin-Tiny 사전학습 가중치** (선택): `https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth`
- 두 가중치 모두 노트북 내 `wget` 스크립트로 자동 다운로드됩니다. `USE_PRETRAINED = False` (기본값) 로 두면 스크래치 학습.

---

#### Level 3 — Imbalance & Advanced Augmentation

| 파라미터 | 값 |
|---|---|
| Optimizer | AdamW |
| Learning Rate | 5e-4 |
| Weight Decay | 5e-2 |
| LR Scheduler | CosineAnnealingLR (T_max = 30) |
| Epochs | 30 |
| Batch Size | 64 |
| Loss | **FocalLoss** (속성별 γ 자동 계산) |
| Sampler | **IR-weighted WeightedRandomSampler** (3속성 통합) |
| Augmentation | RandAugment (n=2, m=9) + ColorJitter + **CutMix (α=1.0)** |
| Gradient Clip | norm = 1.0 |

**FocalLoss γ 자동 계산 공식**:

```
IR_a  = max_count_a / min_count_a   (속성 a 의 Imbalance Ratio)
γ_a   = round(1 + ln(IR_a), 1)

Set A 기준: weather IR ≈ 15.5 → γ = 3.7
            scene   IR ≈  5.4 → γ = 2.7
            timeofday IR ≈ 6.3 → γ = 2.8
```

**IR-weighted Sampler 가중치**:

```
w_i = Σ_a  (IR_a / ΣIR) × (1 / count(label_a[i]))
```

각 샘플 가중치는 3개 속성 모두의 희귀도를 IR 비례로 통합하여 계산합니다.

**CutMix + IR-weighted Loss**:

```python
x_mix, ya, yb, lam = cutmix_data(images, targets, alpha=1.0)
loss = Σ_a  (IR_a / ΣIR) × [lam × FocalLoss(logit_a, ya_a)
                             + (1-lam) × FocalLoss(logit_a, yb_a)]
```

**학습 대상 실험 목록**:

| 실험명 | 체크포인트 파일 |
|---|---|
| Full strategy (FocalLoss + IR sampler + RandAug + CutMix) | `level3_ir-focal+ir-sampler+randaug+cutmix.pth` |
| Ablation B: IR sampler only | `level3_ir-sampler-only.pth` |
| Ablation C: CutMix only | `level3_cutmix-only.pth` |
| Pretrained ViT + Full strategy (Level 5 베이스) | `level3_vit-pretrained+ir-focal+ir-sampler+randaug+cutmix.pth` |

---

#### Level 4 — XAI & Efficiency

학습 없음. Level 1~3 에서 저장된 체크포인트를 로드하여 평가·분석만 수행합니다.

| 사용 체크포인트 | 용도 |
|---|---|
| `level1_vgg16_uw.pth` | Pareto plot (VGG-16 + UW) |
| `level1_resnet18_uw.pth` | Pareto plot (ResNet-18 + UW) |
| `level1_resnet50_uw.pth` | Pareto plot (ResNet-50 + UW) |
| `level2_swin_tiny.pth` | Pareto plot (Swin-Tiny) |
| `level3_ir-focal+ir-sampler+randaug+cutmix.pth` | Best 모델 Grad-CAM + Confusion Matrix |

---

#### Level 5 — Data Mining (전략 재학습)

Level 3 best 체크포인트(`level3_vit-pretrained+...pth`)에서 파인튜닝합니다.

| 파라미터 | 값 |
|---|---|
| 초기화 | `level3_vit-pretrained+ir-focal+ir-sampler+randaug+cutmix.pth` |
| Optimizer | AdamW |
| Learning Rate | **1e-4** (Level 3 대비 1/5 — 이미 수렴된 모델) |
| Weight Decay | 5e-2 |
| LR Scheduler | CosineAnnealingLR (T_max = 30) |
| Epochs | 30 |
| Batch Size | 64 |
| Loss | FocalLoss (IR 재계산 — Set A + picks 합산 기준) |
| Sampler | IR-weighted WeightedRandomSampler (재계산) |
| Augmentation | RandAugment (n=2, m=9) + ColorJitter + CutMix (α=1.0) |
| Gradient Clip | norm = 1.0 |

**학습 데이터**: Set A train (5,000장) + 선별된 K장 (기본 K=1,000).

IR과 FocalLoss γ는 merged dataset 기준으로 자동 재계산되므로, picks 분포에 따라 각 속성의 γ가 자동 조정됩니다.

**선별 전략별 기록된 결과** (`SEED=42`, 30 epochs):

| 전략 | Avg-MF1 | DI Score |
|---|---|---|
| Random baseline (K=1000) | 0.6546 | 0.00% |
| IR-rarity + Uncertainty (Strategy A) | 0.6580 | +0.52% |
| Pure Uncertainty | 0.6650 | +1.60% |
| **Feature Diversity** *(best)* | **0.6739** | **+2.94%** |
| High-Loss (pure CE) | 0.6439 | −1.64% |
| Minority-Error | 0.6450 | −1.46% |

**Strategy A 절제 실험 (K 크기별)**:

| K | Avg-MF1 |
|---|---|
| 250 | 0.6467 |
| 500 | 0.6597 |
| 1000 | 0.6580 |

---

### 10.3 평가 단계만 재현하기 (SKIP_TRAINING)

학습 시간이 1시간을 초과하는 Level 1·2·3·5 노트북에는 `SKIP_TRAINING` 플래그가 있습니다. `True` 로 설정하면 학습을 건너뛰고 Drive 에 저장된 체크포인트를 직접 로드합니다.

```python
# 각 노트북의 setup 셀에서
SKIP_TRAINING = True   # 평가 단계만 재현
```

| Level | 체크포인트 위치 | 예상 학습 시간 (T4) |
|---|---|---|
| Level 1 | `/drive/MyDrive/aue8088-pa2/checkpoints/level1_*.pth` | ~3시간 (6개 모델) |
| Level 2 | `/drive/MyDrive/aue8088-pa2/checkpoints/level2_*.pth` | ~1시간 (2개 모델) |
| Level 3 | `/drive/MyDrive/aue8088-pa2/checkpoints/level3_*.pth` | ~2시간 (4개 실험) |
| Level 5 | `/drive/MyDrive/aue8088-pa2/checkpoints/level5_final.pth` | ~5시간 (6개 전략) |

> 체크포인트가 Drive 에 없으면 `FileNotFoundError` 가 발생하며 경로를 안내합니다.

---

## 11. 저장소 구조 (Repository Structure)

```
2026-aue8088-pa2/
├── README.md                       # ← 지금 이 문서
├── requirements.txt                # 의존성 정의 (단일 소스)
├── data/                           # 학생이 직접 다운로드 후 배치
│   ├── set_a/
│   └── set_b/
├── notebooks/
│   ├── level1_classic_cnns.ipynb   # VGG / ResNet 스켈레톤
│   ├── level2_transformers.ipynb   # ViT / Swin 스켈레톤
│   ├── level3_imbalance.ipynb      # Loss / Sampler / Aug 실험
│   ├── level4_xai_efficiency.ipynb # Grad-CAM, FPS 측정
│   └── level5_data_mining.ipynb    # ★ 자유 설계
├── src/
│   ├── datasets/
│   │   ├── bdd_attr.py             # Set A / Set B 로더
│   │   └── samplers.py
│   ├── models/
│   │   ├── vgg.py                  # (구현 영역 — TODO)
│   │   ├── resnet.py               # (구현 영역 — TODO)
│   │   └── transformers.py
│   ├── losses/
│   │   └── imbalanced.py           # Focal, LDAM, CB-Loss
│   ├── augment/
│   │   └── mix.py                  # Mixup / CutMix
│   ├── xai/
│   │   └── gradcam.py
│   └── utils/
│       ├── metrics.py              # Macro-F1, per-class
│       ├── trainer.py
│       └── seed.py
└── submission/
    └── sample_submission.csv
```

> 스켈레톤 코드의 `# TODO:` 주석을 채워 나가는 방식으로 진행하시면 됩니다. 모델 구현 부분은 의도적으로 비워져 있습니다 (Level 1).

---

## 12. 시작하기 (Quick Start)

```bash
# 1. 저장소 클론
git clone https://github.com/<course-org>/2026-aue8088-pa2.git
cd 2026-aue8088-pa2

# 2. 가상환경 구성 (로컬 — Colab 사용 시 생략)
python3.10 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
#    Colab 사용 시: 노트북 첫 셀에서 !pip install -r requirements.txt

# 3. 데이터 다운로드 (LMS 공지 링크 사용)
#    압축 해제 후 ./data/ 아래에 set_a/, set_b/ 배치

# 4. 베이스라인 학습 (Level 1, ResNet-18)
jupyter notebook notebooks/level1_classic_cnns.ipynb

# 5. Kaggle 제출
#    노트북에서 생성된 submission CSV를 Kaggle 페이지에 직접 업로드
```

---

## 13. 자주 묻는 질문 (FAQ)

**Q1. Pretrained weight를 써도 되나요?**
A. ImageNet **pretrained 가중치 텐서(`.pth`/`.bin`)** 는 모든 Level에서 허용됩니다. 다만 모델 라이브러리(`torchvision.models`, `timm` 등)를 import해서 모델 객체째로 가져오는 것은 금지입니다 — 본인이 구현한 모델의 `state_dict` 키에 맞춰 외부 가중치를 **로드(map)** 하는 형태로 사용하세요. 사용 시 출처(예: `timm` 에서 export한 어느 체크포인트)를 노트북과 리포트에 명시해 주세요. BDD100K 자체로 pretrain된 가중치는 사용 금지입니다.

**Q2. Set B의 이미지를 1,000장 *미만*만 선택해도 되나요?**
A. 가능합니다. "1,000장 추가" 가 아닌 "**최대 1,000장까지** 추가" 입니다. 더 적게 골라 더 큰 효과를 봤다면 그 자체가 좋은 큐레이션 결과입니다.

**Q3. Set B의 라벨을 어떻게든 알아낼 방법이 있나요? (예: 메타데이터의 timestamp로 추정)**
A. **금지** 합니다. 평가의 본질은 *모델/이미지 분석을 통한 데이터 가치 판단* 입니다. 메타데이터 기반 휴리스틱이 의심되면 0점 처리될 수 있습니다.

**Q4. Colab 무료 T4로 충분한가요?**
A. 본 PA는 클래스당 ~1,000장 / 총 5~6천장 / 224×224 / 20~30 epoch 기준으로 설계되어, ResNet-50급 모델 1회 학습이 **약 30분 ~ 1시간** 입니다. Colab Pro가 없어도 무리 없이 수행 가능합니다.

**Q5. 어떤 백본까지 의무인가요?**
A. **VGG + ResNet (Level 1) + ViT 또는 Swin 중 하나 (Level 2)** 가 의무입니다. 그 외(Inception, ResNeXt, ConvNeXt, EfficientNet 등)는 자유롭게 추가 비교하면 가산점.

**Q6. Multi-task 학습에서 3개 task의 loss는 어떻게 합치나요?**
A. 가장 간단한 시작은 단순 합 `L = L_weather + L_scene + L_timeofday` 입니다. 더 정교하게는 **Uncertainty Weighting (Kendall et al., CVPR 2018)** 이나 **GradNorm** 등을 적용해 볼 수 있습니다. 가중치 설정에 따라 성능이 크게 달라지므로 Level 1 또는 Level 3 의 분석에 포함하세요.

**Q7. Multi-task인데 각 task의 Macro-F1이 너무 다르면 어떻게 하나요?**
A. 그게 정상입니다. Time of Day(3 클래스, 비교적 균형)는 Macro-F1 가 높게, Weather(6 클래스, 극심한 불균형)는 낮게 나옵니다. 채점은 **3개의 평균(Avg-Macro-F1)** 으로 이루어지므로, 어느 한 task에 과적합되지 않도록 Level 3에서 균형을 맞추는 것이 핵심입니다.

**Q8. WandB 등 외부 로깅 툴을 써도 되나요?**
A. 자유입니다. 사용 시 리포트에 학습 곡선 스크린샷이나 공유 링크를 첨부하면 가독성이 좋아집니다.

---

## 14. 참고 자료 (References)

- **Dataset**: Yu et al., *"BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning"*, CVPR 2020.
- **Backbones**:
  - Simonyan & Zisserman, *VGG*, ICLR 2015
  - He et al., *ResNet*, CVPR 2016
  - Dosovitskiy et al., *ViT*, ICLR 2021
  - Liu et al., *Swin Transformer*, ICCV 2021
- **Imbalance**:
  - Lin et al., *Focal Loss*, ICCV 2017
  - Cao et al., *LDAM-DRW*, NeurIPS 2019
- **Augmentation**:
  - Cubuk et al., *RandAugment*, NeurIPS 2020
  - Zhang et al., *Mixup*, ICLR 2018
  - Yun et al., *CutMix*, ICCV 2019
- **XAI**:
  - Selvaraju et al., *Grad-CAM*, ICCV 2017
- **Active Learning / Curation**:
  - Sener & Savarese, *Core-Set*, ICLR 2018
  - Beluch et al., *Power of Ensembles for Active Learning*, CVPR 2018

---

## 15. 문의 (Contact)

- **수업 관련**: LMS 공지 게시판
- **과제/스켈레톤 코드 버그**: 본 저장소의 [Issues](../../issues) 에 등록
- **Kaggle Competition 관련**: Kaggle 페이지의 Discussion

---

> *여러분의 창의적인 데이터 큐레이션 전략을 기대합니다.*
