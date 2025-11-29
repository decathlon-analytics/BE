# Decathlon Analytics Backend

**Decathlon Korea 제품 리뷰 데이터 분석 및 RAG 기반 챗봇 백엔드 시스템**

리뷰 데이터를 PostgreSQL에 적재하고, 통계 API 및 **대화형 제품 추천 챗봇**을 제공하는 **FastAPI 기반 백엔드 서버**입니다.




## 주요 기능

### 1. 데이터 관리
- 리뷰 및 요약 데이터 CSV 업로드 및 DB 자동 적재
- PostgreSQL 기반 제품/리뷰 데이터 관리
- 카테고리별 통계 분석 (러닝/하이킹)

### 2. 통계 분석 API
- 카테고리별 TOP 제품 조회
- 리뷰 수/가격 분포 분석
- 평점 통계

### 3. 대화형 챗봇
- **하이브리드 RAG 검색**: 상품정보(60%) + 리뷰(40%) 가중치
- **OpenAI GPT-4o-mini** 기반 자연어 응답 생성
- **세션 기반 대화**: 쿠키를 통한 24시간 대화 이력 유지
- **다양한 추천 방식**: 모델번호, 제품명, 카테고리, 레벨별 세트

## 기술 스택

| 항목 | 내용 |
|------|------|
| **언어/런타임** | Python 3.11 |
| **웹 프레임워크** | FastAPI |
| **ORM / DB** | SQLAlchemy + PostgreSQL |
| **데이터 처리** | pandas, numpy |
| **임베딩** | OpenAI text-embedding-3-small |
| **LLM** | OpenAI GPT-4o-mini |
| **배포** | Render (Backend) / Vercel (Frontend) |

## 프로젝트 구조
```bash
decathlon-analytics/
├── core/               # DB 연결, 환경 변수
│   ├── config.py
│   └── db.py
├── routers/            # API 엔드포인트
│   ├── analytics.py    # 통계 분석
│   ├── chatbot.py      # RAG 챗봇
│   └── debug.py
├── services/           # 비즈니스 로직
│   ├── ingest.py
│   ├── utils.py
│   ├── cards.py
│   └── rag_index.py    # RAG 인덱스
├── data/
│   └── embeddings.pkl  # 임베딩 캐시 (gzip 압축, 42MB)
├── main.py
├── requirements.txt
└── .env
```

## 주요 API

### 데이터 업로드
| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/ingest/summary` | POST | 상품 요약 데이터 업로드 |
| `/ingest/reviews` | POST | 리뷰 데이터 업로드 |

### 통계 분석
| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/total/*` | GET | 전체 통계 (TOP 제품, 리뷰 수, 가격 분포) |
| `/running/*` | GET | 러닝 카테고리 통계 |
| `/hiking/*` | GET | 하이킹 카테고리 통계 |

### 챗봇
| 엔드포인트 | 메서드 | 설명 |
|-----------|--------|------|
| `/chatbot/health` | GET | 챗봇 상태 및 인덱스 정보 |
| `/chatbot/reindex` | POST | 임베딩 인덱스 재구축 |
| `/chatbot/chat` | POST | 메시지 전송 및 응답 수신 |

## 챗봇 상세 기능
사용자 질문을 자동으로 분석하여 최적의 응답 방식을 선택합니다.

| 질문 유형 | 예시 | 처리 방식 |
|----------|------|----------|
| 모델번호 검색 | "8926414" | DB 직접 조회 |
| 제품명 검색 | "남성 러닝 윈드 자켓 런 100" | 띄어쓰기 무시 매칭 |
| 레벨 세트 추천 | "초심자용 러닝 세트", "500 추천" | 상의+하의+장갑 조합 |
| 인기 제품 | "요즘 인기있는 제품" | 리뷰 수 기준 정렬 |
| 일반 추천 | "가벼운 러닝화 추천" | 하이브리드 RAG 검색 |



