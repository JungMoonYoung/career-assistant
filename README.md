# Career Insight — 채용공고 RAG 검색 + 연봉 예측 플랫폼

> **수집 → 정제 → 분석 → 저장 → 서빙의 End-to-End 파이프라인을 100% 구축**

국내 주요 채용 플랫폼 공고 6,332건을 로컬 임베딩 + Gemini API 하이브리드 RAG 구조로 인덱싱하고, Random Forest 기반 연봉 예측 모델을 결합한 커리어 분석 플랫폼입니다.

---

## Links

- **Live Demo**: https://career-assistant-6kmsehtkmf5ilfbatjev6l.streamlit.app/
- **Portfolio (Notion)**: https://www.notion.so/33a404a59bef81d8925ede4e73da998a

---

## 프로젝트 개요

| 항목 | 내용 |
|---|---|
| 구분 | 개인 프로젝트 |
| 기간 | 2025.12 ~ 2026.04 |
| 기여도 | 100% |
| 역할 | 데이터 파이프라인 설계, 크롤링 자동화, ML 모델 학습, RAG 시스템 구축, 대시보드 배포 |
| 배포 URL | https://career-assistant-6kmsehtkmf5ilfbatjev6l.streamlit.app/ |

---

## 문제 정의 & 해결 방향

### 문제 정의

취업 준비생이 채용 시장을 파악하려면 여러 사이트를 돌아다니며 공고를 하나씩 확인해야 합니다. 직무별 연봉, 요구 기술, 지역별 차이를 **체계적으로 비교할 방법이 없습니다**.

### 해결 방향

- 채용 공고를 자동 수집하고, 통계적으로 검증된 인사이트를 제공
- 자연어로 공고를 검색할 수 있는 플랫폼을 구축

---

## 기술 스택

| 영역 | 스택 |
|---|---|
| 수집 | BeautifulSoup4, Requests (국내 주요 채용 플랫폼 크롤링) |
| 분석·모델링 | Pandas, NumPy, SciPy (ANOVA), Scikit-learn (RandomForest) |
| RAG | LangChain, FAISS, Sentence Transformers (ko-sroberta), Google Gemini API |
| 시각화·배포 | Streamlit, Plotly |

---

## 시스템 아키텍처

```
[크롤링]                    [전처리]                   [인덱싱 / 모델 학습]          [서빙]
BeautifulSoup4  →  직무명 정규화 · 교차검증  →  ko-sroberta 로컬 임베딩   →   Streamlit
Requests           3,044건 필터링              FAISS 벡터 DB                RAG 채팅
                                               RandomForest 연봉 모델        연봉 예측 UI
```

---

## 핵심 기능 구현

### 1. 로컬 임베딩 + Gemini API 하이브리드 RAG

| 선택지 | 임베딩 비용 | 답변 품질 | 채택 |
|---|---|---|---|
| 전부 API (Gemini) | 데이터 업데이트마다 비용 증가 | 높음 | ✗ |
| 전부 로컬 | $0 | 답변 생성 품질 제한 | ✗ |
| **로컬 임베딩 + API 답변** | **검색은 $0, 답변만 API** | **높음** | **O** |

**기대 효과**:

- 신규 데이터 임베딩부터 벡터 DB 재빌드까지 로컬에서 처리하여 **외부 API 비용 $0으로 운영**
- Gemini API는 최종 답변 생성에만 사용하여 비용 최소화
- 데이터가 6만 건, 60만 건으로 늘어나도 임베딩 비용은 동일하게 $0

### 2. 데이터 파이프라인

**데이터 품질 관리**

- 플랫폼마다 다른 직무 명칭(*'데이터 분석가', 'Data Analyst', 'BI 전문가'*)을 정규화하는 로직 설계
- 크롤링 파싱 오류 **3,044건 자동 필터링** (회사명 유효성 + 교차 검증)
- 정규화 적용 후 모델 노이즈 감소

**자동 업데이트 전략**

- 매주 신규 데이터만 크롤링 → 기존 데이터와 URL 기반 중복 제거 → 벡터 DB 재빌드
- 현재 6,332건이지만, 60만 건이 들어와도 동일한 파이프라인으로 처리 가능한 구조
- 주간 500건 자동 수집 → DB가 점진적으로 성장하며 검색 품질 지속 향상

### 3. 연봉 예측 모델링

- **모델 설계**: 데이터 패턴(Kaggle 2022 ML/DS Survey)
  - Kaggle로 직무·경력·지역 간 관계 패턴을 학습한 뒤, 한국 평균 연봉(7,751만 원) 기준 scale_factor를 적용하여 국내 수준으로 변환
  - 외부 데이터 사용 이유: 주요 플랫폼에서 연봉 정보가 공개되지 않음
- **Random Forest 채택 이유**:
  - RF, XGBoost, LightGBM, GradientBoosting 4종 비교 실험
  - 설명력 차이 0.006 이내로 성능 동등
  - **변수 중요도 해석이 직관적인 RandomForest 채택**
- **Feature Engineering**
  - 3개 이상 글로벌 직무 → 데이터 분석가, 데이터 엔지니어, ML 엔지니어 등 9개 국내 직무로 매핑
  - 경력(신입/경력), 지역(서울 50%, 경기 25% 등 가중), 회사 규모(대/중/소), 원격 비율 적용
- **모델 검증**
  - Train/Test 80:20 분할 + 5-Fold CV (R² ± 0.01)

### 4. 공고 검색 시스템

**벡터 DB 구축**

- ko-sroberta-multitask 모델로 임베딩 (외부 API 비용 $0)
- 채용 공고를 LangChain Document로 변환
- 500건 단위 배치 처리로 대량 데이터 인덱싱

**검색 및 답변 생성**

- 코사인 유사도 기반 FAISS 검색 (상위 3건 검색)
- 유사도 점수 → 신뢰도 변환: **80% 이상은 "추천", 미만은 "유사 공고"**로 분류
- Gemini Flash: 구조화된 답변 생성(회사명, 스킬, 경력, 매칭 이유 포함)

---

## 핵심 결과

### 연봉 예측 모델

| 지표 | 수치 |
|---|---|
| R² Score | **0.766** |
| MAE | ±789만 원 |
| 교차검증 (5-Fold) | 0.769 ± 0.01 |

**특성 중요도 인사이트**:

| 특성 | 인사이트 |
|---|---|
| 직무 | 같은 경력이라도 **직무 전환이 연봉 변화의 가장 큰 요인** |
| 경력 | 경력 초기에는 직무 선택이, 중후반에는 경력 축적이 핵심 |
| 규모 | 대기업 vs 스타트업 연봉 격차는 통상 생각하는 차이보다는 적음 |
| 지역 | 지역별로 큰 격차는 없으므로 수도권 지역을 고집할 필요 없음 |

### 공고 검색 시스템 (RAG)

| 지표 | 수치 |
|---|---|
| 임베딩 비용 | **$0** |
| 인덱싱 공고 수 | 6,332건 |
| 검색 응답 시간 | ~2-3초 |

- 신뢰도 80% 이상 → "추천"
- 신뢰도 80% 미만 → "유사 공고"로 자동 분류
- 회사명, 스킬, 경력, 매칭 이유를 포함한 구조화된 답변 제공

---

## 문제 해결 경험

### 1. 비정형 데이터의 품질 확보

- **문제**: 다양한 채용 플랫폼의 비정형 데이터를 하나의 기준으로 통합하는 과정에서 데이터 오류 발생
- **해결**: 회사명 유효성 검사 + 필드 교차 검증 + 직무명 정규화 로직 설계
- **결과**: **3,044건 제거, 유효 6,332건 확보**. 정규화 적용으로 모델 노이즈 감소 및 학습 품질 향상
- **배운 점**: 무조건 많은 데이터가 중요한 것이 아니라, **사용 가능한 깨끗한 데이터**가 되어야 사용이 가능하다.

### 2. 임베딩 비용 vs 답변 품질

- **문제**: 전체를 API 기반으로 구축하면 데이터 갱신 시 비용이 기하급수적으로 증가
- **해결**: 검색은 로컬 모델, 답변 생성만 Gemini API → **하이브리드 구조** 설계
- **결과**: 주간 재임베딩 비용 **$0 + 고품질 답변 유지**. 데이터 규모 확장에도 비용 구조 동일
- **배운 점**: 비즈니스적으로 비용이 가장 1순위인데 결과 품질만 좋게 하려 하면 안 되고, **최소한의 비용으로 최대한의 결과**를 낼 수 있는 방안이 필수이다.

---

## 실행 방법

### Local 실행

```bash
# 1. 저장소 클론
git clone https://github.com/JungMoonYoung/career-assistant.git
cd career-assistant

# 2. 가상환경 및 의존성
python -m venv venv
# Windows
.\venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt

# 3. 환경변수 설정 (.env.example 참고)
cp .env.example .env
# .env 파일에 GEMINI_API_KEY 입력

# 4. 대시보드 실행
streamlit run app.py
```

### 폴더 구조

```
career-assistant/
├── app.py                      # Streamlit 메인 앱
├── src/
│   ├── analysis/               # Kaggle 분석 (연봉 모델링)
│   │   └── kaggle_analyzer.py
│   ├── data_collection/        # 크롤링
│   │   ├── auto_crawl.py
│   │   ├── crawl_all.py
│   │   ├── saramin_api.py
│   │   ├── saramin_crawler.py
│   │   └── worknet_api.py
│   ├── preprocessing/          # 전처리·정규화
│   │   └── preprocess.py
│   ├── indexing/               # 벡터 DB
│   │   └── vector_store.py
│   └── rag/                    # RAG 파이프라인
│       └── rag_pipeline.py
├── data/
│   ├── KAGGLE_DATA.csv
│   └── raw/                    # 원본 크롤링 데이터
├── vector_db/                  # FAISS 인덱스
│   └── faiss_index/
├── .env.example
├── requirements.txt
└── README.md
```

---

## 성장 포인트

> 데이터 분석과 엔지니어링은 분리된 영역이 아님을 직접 체감했습니다 — 수집·정제·갱신이 자동화되지 않으면 분석 모델은 일회성으로 끝납니다.
>
> 분석가가 **파이프라인 설계까지 수행할 수 있을 때 비로소 분석이 지속 가능한 서비스**가 된다는 것을 확인했습니다.
>
> 크롤링부터 전처리, 벡터 DB 구축, 배포까지 End-to-End로 직접 구현하며 **분석이 서비스로 전환되는 과정**을 경험했습니다.

---

## Contact

- **GitHub**: [JungMoonYoung](https://github.com/JungMoonYoung)
- **Email**: kobing7122@gmail.com
- **Portfolio**: https://www.notion.so/252404a59bef802b8693d40f30b48d82
