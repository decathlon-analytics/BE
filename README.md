# Decathlon Analytics Backend

**Decathlon Korea 제품 리뷰 데이터 분석 및 RAG 기반 챗봇 백엔드 시스템**

리뷰 데이터를 PostgreSQL(pgvector)에 적재하고, 통계 API 및 대화형 추천 기능을 제공하는 **FastAPI 기반 백엔드 서버**입니다.  
AI, 데이터 시각화, 자연어 질의 응답을 통합적으로 제공하여 프론트엔드 대시보드 및 챗봇 기능을 지원합니다.

---

## 🚀 주요 기능

- 리뷰 및 요약 데이터 CSV 업로드 및 DB 자동 적재
- PostgreSQL(pgvector) 기반 리뷰 임베딩 검색
- 통계 분석 API (카테고리별 TOP 상품, 리뷰 수, 가격 분포 등)
- RAG 기반 대화형 챗봇 (LLM 연동)
- 쿠키 기반 세션 관리 (로그인 불필요)
- Render / Vercel 환경 분리 배포 지원

---

## 🧩 기술 스택

| 항목 | 내용 |
|------|------|
| **언어/런타임** | Python 3.11 |
| **웹 프레임워크** | FastAPI |
| **ORM / DB** | SQLAlchemy + PostgreSQL (로컬은 SQLite 지원) |
| **데이터 처리** | pandas, numpy |
| **검색 / RAG** | PostgreSQL pgvector 확장 기반 임베딩 검색 |
| **임베딩 모델** | OpenAI `text-embedding-3-small` |
| **LLM** | GPT-4o-mini (환경변수로 주입) |
| **배포** | Render (Backend) / Vercel (Frontend) |
| **기타** | python-dotenv, requests, CORS middleware |

> ⚙️ Render 환경은 pgvector 미지원으로, 현재는 로컬에서 생성한 벡터를  
> `embeddings.pkl` 형태로 캐시 저장하여 사용합니다.  
> 추후 Supabase 또는 Neon DB로 마이그레이션 예정입니다.

---

## 📂 프로젝트 구조
~~~
decathlon-analytics/
├── core/                      # 백엔드 핵심 설정 (DB 연결, 환경 변수, 메타정보)
│   ├── config.py              # 환경 변수 및 앱 메타 설정
│   ├── db.py                  # PostgreSQL(pgvector) 엔진 및 테이블 생성 관리
│
├── routers/                   # FastAPI 라우터 (API 엔드포인트 정의)
│   ├── ingest.py              # CSV 리뷰·요약 데이터 적재 API
│   ├── analytics.py           # 통계 분석 및 대시보드용 데이터 조회 API
│   ├── chatbot.py             # RAG 기반 챗봇 API (pgvector 검색 + LLM 응답)
│   ├── debug.py               # 디버깅 및 헬스체크용 엔드포인트
│
├── services/                  # 데이터 처리 및 비즈니스 로직
│   ├── ingest.py              # 리뷰·요약 데이터프레임 → DB 적재 및 집계
│   ├── utils.py               # 문자열, 날짜, 숫자 변환 등 공용 유틸 함수
│   ├── cards.py               # 리뷰·요약 통합 카드 데이터 구성
│   ├── rag_index.py           # RAG 인덱스 관리 (pgvector 기반 + pkl 캐시 저장)
│
├── data/                      # CSV 및 임베딩 캐시 저장소
│   ├── embeddings.pkl         # pgvector 임베딩 캐시 파일 (로컬 테스트용)
│
├── main.py                    # FastAPI 진입점 – CORS 설정 및 라우터 등록
├── .env                       # 환경 변수 설정 파일
├── requirements.txt           # 패키지 의존성 목록
└── .gitignore                 # Git 추적 제외 설정
~~~

---

## 🔁 데이터 파이프라인

1. **요약(Summary) / 리뷰(Review) 분리 적재**
   - `/ingest/summary(file)`  
     - 최초 업로드 시: 모든 필드 저장  
     - 이후 업로드 시: 신규 상품만 전체 필드 삽입, 기존 상품은 메타정보만 유지 (가격 포함)  
     - 숫자 집계는 리뷰 테이블에서 자동 관리
   - `/ingest/reviews(file)`  
     - 리뷰 적재 후 `reviews` 테이블에 insert  
     - 해당 `product_id`의 `product_summary` 내  
       `total_reviews`, `positive/mixed/negative_reviews`, `avg_rating` 자동 재계산

2. **분석 API (analytics)**
   - 카테고리별 TOP 상품, 리뷰 수 TOP, 월별 리뷰 추이, 가격대 히스토그램 등  
     통계 데이터를 JSON 형태로 가공하여 프론트엔드 대시보드에 전달

3. **RAG 인덱싱 (chatbot)**
   - `/chatbot/reindex`  
     DB 내 리뷰 텍스트를 OpenAI 임베딩 모델로 벡터화 후 pgvector 컬럼에 저장  
     (배포 환경에서는 `embeddings.pkl` 캐시 파일 사용)
   - 질의 시 pgvector 유사도 검색 결과를 기반으로  
     리뷰 문맥을 추출하고, LLM이 자연어 응답 생성

---

## 💬 챗봇 세션 구조

- **쿠키 기반 세션 유지**
  - 최초 `/chatbot/chat` 호출 시 쿠키가 없어도 서버가 `session_id`를 발급하고  
    `Set-Cookie`로 반환  
  - 프론트엔드는 `credentials: 'include'` 옵션만 설정하면  
    브라우저가 쿠키를 자동 전송  
- **DB 테이블 구성**
  - `sessions`, `messages`, `session_summaries`
- **세션 TTL**
  - 기본 24시간 — 동일 사용자의 연속 대화 시 맥락 유지 가능

---

## ⚙️ 주요 API

| 구분 | 엔드포인트 | 설명 |
|------|-------------|------|
| **Ingest** | `/ingest/summary(file)` | 상품 요약 데이터 업로드 |
| | `/ingest/reviews(file)` | 리뷰 데이터 업로드 |
| **Analytics** | `/total/*`, `/running/*`, `/hiking/*` | 카테고리별 통계 조회 |
| **Chatbot** | `GET /chatbot/health` | 챗봇 상태 확인 |
| | `POST /chatbot/reindex` | 임베딩 인덱스 재구축 |
| | `POST /chatbot/chat` | 메시지 입력 후 응답 생성 |

---

## 🧠 시스템 아키텍처 개요

- FastAPI 기반 REST API
- 데이터는 `reviews`(원문)과 `product_summary`(메타/집계)로 분리 저장
- 리뷰 적재 시 자동 재집계로 항상 최신 리뷰 통계 유지
- 분석 API는 SQL 기반 집계 결과를 JSON 형태로 변환
- 챗봇은 pgvector 기반 RAG 검색을 통해 의미적으로 유사한 리뷰 문맥을 추출 후 응답 생성

---

## 🔒 보안 및 배포

- **CORS**: 정확한 Origin 지정 + `allow_credentials=True` 설정  
- **쿠키 설정**: `HttpOnly`, `SameSite=None; Secure`  
- **배포 환경**: Render(Backend), Vercel(Frontend)  
- **PII 수집 없음**: 사용자 개인정보 없이 리뷰/상품 메타만 저장

---
