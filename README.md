# K-PopularOpinionDataSet

**한국어 정치 여론 댓글의 대상(후보자)·감성·유해성 동시 분석을 위한 멀티태스크 데이터셋**

[![Conference](https://img.shields.io/badge/Conference-KDBC%202025-blue.svg)](#citation)
[![Task](https://img.shields.io/badge/Task-Multi--Task%20NLP%20%7C%20ABSA%20%7C%20Toxicity-green.svg)](#dataset-structure)
[![Base Model](https://img.shields.io/badge/Benchmark-KLUE--RoBERTa--base-orange.svg)](#benchmark-performance)

## 개요 (Overview)

**K-PopularOpinionDataSet**은 온라인 정치 담론에서 발생하는 복합적인 여론 양상을 정밀하게 분석하기 위해 구축된 한국어 자연어 처리(NLP) 벤치마크 데이터셋입니다.

기존의 단일 감성 분석(Sentiment Analysis)이나 유해성 탐지(Toxicity Detection) 데이터셋은 **"특정 댓글이 누구를 대상으로 어떤 감성과 유해성을 띠는지"** 를 동시에 파악하기 어렵다는 한계가 있었습니다. 본 데이터셋은 2025년 대통령 선거 후보자 토론 영상의 실제 유튜브 댓글을 기반으로, **① 언급 대상(후보자)** **② 감성(Sentiment)** **③ 유해성(Toxicity)** 의 3가지 속성을 동시에 학습 및 평가할 수 있도록 설계되었습니다.

---

## 데이터셋 구축 및 전처리 (Dataset Construction)

### 1. 데이터 수집 (Data Collection)
* **수집 도구:** `YouTube Data API v3`
* **수집 출처:** 국내 주요 5개 방송사 공식 유튜브 채널(YTN, JTBC, KBS, MBC, SBS)의 2025 대통령 선거 후보자 토론 영상 상위 댓글 **총 30,647건**
* **균등 샘플링:** 채널별 편향을 방지하기 위해 정제된 데이터 중 방송사별 1,000건씩 중복 제외 **총 5,000건**을 무작위 추출하여 라벨링 수행

### 2. 텍스트 정제 및 라벨링 기준 (Cleaning & Annotation)
* **노이즈 필터링:** 500자 초과 장문 댓글 제외, 과도한 공백 및 줄바꿈 정규화, 이모지(Emoji) 텍스트 변환, 중복 댓글 제거
* **라벨링:** 심리학 기반 감정 척도(`I-PANAS-SF`) 및 AI를 활용한 1차 감정 분류 후, 문맥적 대상·감성·유해성 2차 자체 검수

### 3. 도메인 특화 전처리 파이프라인 (Domain-Specific Preprocessing)
온라인 정치 댓글 특유의 은어, 변형 별명, 다중 대상 언급 문제를 해결하기 위해 다음 4단계 전처리를 적용했습니다.
1. **후보자 별명 사전 구축:** 실제 댓글 내 변형 별명 및 은어(예: `'재매이'`)를 공식 후보자명과 매핑하는 도메인 사전 구축
2. **단일 대상 분리 (Single-Target Decomposition):** 하나의 댓글에 복수의 후보자가 언급된 경우 데이터를 복제하여 각 행(Row)이 단일 후보자에 대한 독립적인 감성·유해성을 갖도록 변환
3. **별명 정규화 (Nickname Normalization):** 구축된 사전을 기반으로 본문 내 비표준 별명을 공식 명칭으로 정규화
4. **엔티티 마킹 (Entity Marking):** 타겟 후보자 개체 앞뒤에 특수 토큰(`[ENT]`, `[/ENT]`)을 삽입하여 인코더가 대상 중심의 문맥적 관계를 집중 학습하도록 유도

---

## 데이터 구조 및 스키마 (Dataset Structure)

저장소 내 데이터 파일(merged_final.csv → preprocessed_data.csv → input.csv)은 다음과 같습니다.
| 파일명 (File) | 설명 (Description) |
| :--- | :---: |
| 'merged_final.csv' | '댓글 데이터 원본' |
| 'preprocessed_data.csv' | '댓글 데이터 전처리 후 파일' |
| 'input.csv' | '분류 모델 입력용 파일' |

저장소 내 데이터 파일(`CSV`)은 다음과 같은 필드로 구성되어 있습니다.

| 컬럼명 (Field) | 타입 (Type) | 설명 (Description) | 클래스 / 예시 (Values) |
| :--- | :---: | :--- | :--- |
| `source` | `string` | 댓글이 수집된 방송사 채널 출처 | `YTN`, `JTBC`, `KBS`, `MBC`, `SBS` |
| `comment` | `string` | 초기 노이즈 필터링만 거친 원본 댓글 텍스트 | `"ㅋㅋㅋㅋ 재매이 ㅋㅋㅋㅋ"` |
| `input_text` | `string` | 정제 및 엔티티 마킹(`[ENT]`)이 적용된 댓글 | `"ㅋㅋㅋㅋ [ENT]이재명[/ENT] ㅋㅋㅋㅋ"` |
| `emotion` | `string` / `int` | 해당 대상에 대한 화자의 감성 극성 (3진 분류) | `긍정(Positive: 0)`, `부정(Negative: 1)`, `중립(Neutral: 2)` |
| `toxicity` | `string` / `int` | 욕설·비방 등 유해 표현 포함 여부 (이진 분류) | `정상(Clean: 0)`, `욕설(Toxic: 1)` |
| `entity` | `string` / `int` | 댓글이 지목하는 평가 대상 후보자 (다중 분류) | 후보자 지칭 대상 `0: 이재명`, `1: 이준석`, `2: 김문수`, `3: 권영국`, `4: 불명(Unknown)` |

---

## 사용 예시 (Quick Start)

```python
import pandas as pd

# 데이터셋 로드 (파일 경로에 맞게 수정)
df = pd.read_csv("data/merged_final.csv")

print(f"총 데이터 샘플 수: {len(df)}")
print(df[['comment', 'emotion', 'toxicity', 'entity']].head())
