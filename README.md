# Decathlon Analytics Backend

**Decathlon Korea 제품 리뷰 데이터 분석 및 RAG 기반 챗봇 백엔드 시스템**

리뷰 데이터를 PostgreSQL에 적재하고, 통계 API 및 대화형 추천 기능을 제공하는 **FastAPI 기반 백엔드 서버**입니다.

## 주요 기능

- 리뷰 및 요약 데이터 CSV 업로드 및 DB 자동 적재
- 하이브리드 RAG 검색 (상품정보 60% + 리뷰 40%)
- 통계 분석 API (카테고리별 TOP 상품, 리뷰 수, 가격 분포)
- RAG 기반 대화형 챗봇 (LLM 연동)
- 쿠키 기반 세션 관리
- Render 배포 지원

## 기술 스택

| 항목 | 내용 |
|--|--|
| **언어/런타임** | Python 3.11 |
| **웹 프레임워크** | FastAPI |
| **ORM / DB** | SQLAlchemy + PostgreSQL |
| **데이터 처리** | pandas, numpy |
| **검색 / RAG** | OpenAI embeddings (gzip 압축) |
| **임베딩 모델** | text-embedding-3-small |
| **LLM** | GPT-4o-mini |
| **배포** | Render (Backend) / Vercel (Frontend) |

## 프로젝트 구조

decathlon-analytics/
├── core/ # DB 연결, 환경 변수
│ ├── config.py
│ ├── db.py
├── routers/ # API 엔드포인트
│ ├── ingest.py # CSV 데이터 적재
│ ├── analytics.py # 통계 분석
│ ├── chatbot.py # RAG 챗봇
│ ├── debug.py
├── services/ # 비즈니스 로직
│ ├── ingest.py
│ ├── utils.py
│ ├── cards.py
│ ├── rag_index.py # RAG 인덱스 (gzip 압축)
├── data/
│ ├── embeddings.pkl # 임베딩 캐시 (gzip 압축, 42MB)
├── main.py
├── requirements.txt
└── .env

text

## 로컬 실행

1. 가상환경 생성
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

2. 패키지 설치
pip install -r requirements.txt

3. 환경변수 설정 (.env 파일)
DATABASE_URL=postgresql://user:pass@localhost/decathlon
OPENAI_API_KEY=sk-...

4. 서버 실행
uvicorn main:app --reload

5. 임베딩 생성 (최초 1회)
curl -X POST http://localhost:8000/chatbot/reindex

text

## 주요 API

| 구분 | 엔드포인트 | 설명 |
|--|--|--|
| **Ingest** | `POST /ingest/summary` | 상품 요약 데이터 업로드 |
|  | `POST /ingest/reviews` | 리뷰 데이터 업로드 |
| **Analytics** | `GET /total/*` | 전체 통계 조회 |
|  | `GET /running/*` | 러닝 카테고리 통계 |
|  | `GET /hiking/*` | 하이킹 카테고리 통계 |
| **Chatbot** | `GET /chatbot/health` | 챗봇 상태 확인 |
|  | `POST /chatbot/reindex` | 임베딩 인덱스 재구축 |
|  | `POST /chatbot/chat` | 메시지 전송 및 응답 |

## 챗봇 기능

- **하이브리드 검색**: 상품정보(60%) + 리뷰(40%) 가중치
- **인텐트 분류**: 모델번호 검색, 레벨별 세트 추천, 후속 질문, 일반 추천
- **세트 추천**: 100/500/900 레벨별 상의+하의+장갑 조합
- **세션 관리**: 쿠키 기반 대화 이력 유지 (24시간 TTL)

## 배포

- **Backend**: Render
- **Frontend**: Vercel
- **CORS**: 정확한 Origin 지정 + `allow_credentials=True`
- **쿠키**: `HttpOnly`, `SameSite=None; Secure`

## 임베딩 파일

`embeddings.pkl`은 gzip 압축되어 42MB로 저장됩니다.
서버 배포 시 자동으로 로드되며, 재생성이 필요한 경우:

curl -X POST https://your-api.com/chatbot/reindex

text

## 라이센스

MIT
