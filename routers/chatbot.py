# routers/chatbot.py (최종 수정)
from fastapi import APIRouter, Depends, Request, Response, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime, timedelta
import os, time, json, re

from sqlalchemy.orm import Session
from sqlalchemy import text
from core.db import get_db

from services.rag_index import hybrid_search, search_by_model_id, search_by_name_fast, recommend_level_set, index_meta
from openai import OpenAI

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

ENV = os.getenv("ENV", "dev").lower()
COOKIE_NAME = "session_id"
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "24"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatIn(BaseModel):
    message: str

class ChatOut(BaseModel):
    answer: str
    recommendations: Optional[List[Dict[str, Any]]] = None
    set_info: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

def _set_session_cookie(response: Response, sid: str):
    secure = (ENV == "prod")
    samesite = "none" if secure else "lax"
    response.set_cookie(
        key=COOKIE_NAME,
        value=sid,
        httponly=True,
        secure=secure,
        samesite=samesite,
        max_age=SESSION_TTL_HOURS * 3600,
    )

def _ensure_session(request: Request, response: Response, db: Session) -> str:
    sid = request.cookies.get(COOKIE_NAME)
    now = datetime.utcnow()
    exp = now + timedelta(hours=SESSION_TTL_HOURS)
    
    upsert_sql = text("""
        INSERT INTO sessions (id, started_at, last_active, expires_at, last_recommended_ids)
        VALUES (:id, NOW(), NOW(), :exp, '')
        ON CONFLICT (id) DO UPDATE
            SET last_active = EXCLUDED.last_active,
                expires_at  = EXCLUDED.expires_at;
    """)
    
    if not sid:
        sid = str(uuid4())
        _set_session_cookie(response, sid)
    
    try:
        db.execute(upsert_sql, {"id": sid, "exp": exp})
        db.commit()
    except Exception as e:
        from core.db import ensure_tables
        ensure_tables()
        db.execute(upsert_sql, {"id": sid, "exp": exp})
        db.commit()
    
    return sid

def _load_recent_context(db: Session, session_id: str, limit_turns: int = 6) -> List[Dict[str, Any]]:
    rows = db.execute(text("""
      SELECT role, content
      FROM messages
      WHERE session_id=:sid
      ORDER BY created_at DESC
      LIMIT :lim
    """), {"sid": session_id, "lim": limit_turns}).fetchall()
    return [{"role": r[0], "content": r[1]} for r in rows[::-1]]

def _save_message(db, session_id, role, content, meta=None):
    stmt = text("""
        INSERT INTO messages (session_id, role, content, meta)
        VALUES (:sid, :role, :content, CAST(:meta AS JSONB))
    """)
    db.execute(stmt, {
        "sid": str(session_id),
        "role": role,
        "content": content,
        "meta": json.dumps(meta or {}, ensure_ascii=False),
    })
    db.commit()

def _get_last_recommended_ids(db, session_id) -> List[str]:
    try:
        row = db.execute(text("SELECT last_recommended_ids FROM sessions WHERE id=:sid"), {"sid": session_id}).first()
        if not row or not row[0]: return []
        return [x for x in row[0].split(",") if x]
    except:
        return []

def _save_recommended_ids(db, session_id, product_ids: List[str]):
    ids_str = ",".join([str(x) for x in product_ids if x])
    try:
        db.execute(text("UPDATE sessions SET last_recommended_ids=:ids WHERE id=:sid"), {"ids": ids_str, "sid": session_id})
        db.commit()
    except:
        pass

def _get_last_user_query(history: List[Dict[str, str]]) -> str:
    for msg in reversed(history):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if not any(k in content.lower() for k in ["다른", "더", "또", "추가"]):
                return content
    return ""

def _get_last_product_type(db, session_id) -> str:
    try:
        ids = _get_last_recommended_ids(db, session_id)
        if not ids:
            return ""
        
        row = db.execute(text("""
            SELECT product_name, category 
            FROM product_summary 
            WHERE product_id = :pid
        """), {"pid": ids[0]}).mappings().first()
        
        if not row:
            return ""
        
        name = row["product_name"] or ""
        
        if any(k in name.lower() for k in ["티", "자켓", "재킷", "베스트", "상의"]):
            return "상의"
        elif any(k in name.lower() for k in ["바지", "쇼츠", "팬츠", "레깅스", "하의"]):
            return "하의"
        elif any(k in name.lower() for k in ["러닝화", "등산화", "슈즈", "신발"]):
            return "신발"
        elif "가방" in name.lower() or "백팩" in name.lower():
            return "가방"
        
        return row.get("category", "")
    except:
        return ""

def _detect_gender(query: str) -> str:
    q = query.lower()
    if any(k in q for k in ["여성", "여자", "우먼", "woman", "레이디"]):
        return "female"
    elif any(k in q for k in ["남성", "남자", "맨", "man"]):
        return "male"
    return "neutral"

def _classify_intent(query: str, history: List[Dict]) -> str:
    q = query.lower()
    
    # 1순위: 모델번호
    model_match = re.search(r"(\d{7})", q)
    if model_match:
        return "model_search"
    
    # 2순위: 세트 요청 (명확하게)
    set_keywords = ["세트", "풀세트", "조합", "구성", "셋트", "패키지", "set"]
    level_keywords = ["초심자", "중급자", "고수", "입문", "비기너", "프로", "상급", "초보"]
    
    if any(k in q for k in set_keywords):
        return "level_set"
    
    if any(k in q for k in level_keywords) and "추천" in q:
        return "level_set"
    
    if re.search(r"\b(100|500|900)\b", q) and not any(k in q for k in ["티", "자켓", "바지", "신발", "러닝", "하이킹"]):
        if any(k in q for k in ["추천", "알려", "찾아"]):
            return "level_set"
    
    # 3순위: 인기 제품
    if any(k in q for k in ["인기", "탑", "많이 팔리는", "잘 나가는", "요즘"]):
        return "popular"
    
    # 4순위: 후속 질문
    if any(k in q for k in ["다른", "또", "더", "그 외", "다른거", "다른 거", "하나 더", "추가"]):
        return "followup"
    
    # 5순위: 스몰톡 (제품 키워드 체크 포함)
    greeting_keywords = ["안녕", "하이", "ㅎㅇ", "고마워", "감사", "땡큐", "thank", "hello"]
    product_keywords = [
        "백팩", "가방", "자켓", "신발", "티셔츠", "바지", "러닝", "하이킹", "등산",
        "제품", "상품", "추천", "찾", "검색", "구매", "nh100", "nh", "아르페나즈"  # 🔥 "알려" 제거
    ]
    
    has_greeting = any(k in q for k in greeting_keywords)
    has_product = any(k in q for k in product_keywords)
    
    print(f"🔍 Query: {q}")
    print(f"🔍 has_greeting: {has_greeting}, has_product: {has_product}")
    
    if has_greeting and not has_product:
        print("🔍 Result: greeting")
        return "greeting"
    
    print("🔍 Result: recommendation")
    return "recommendation"


def _call_llm(system_prompt: str, messages: List[Dict[str, str]]) -> str:
    try:
        comp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "system", "content": system_prompt}] + messages,
            temperature=0.3,
            max_tokens=400,
        )
        return comp.choices[0].message.content.strip()
    except Exception as e:
        return f"답변 생성 중 오류가 발생했습니다: {str(e)}"

SYSTEM_PROMPT = (
    "당신은 데카트론 상품 추천 어시스턴트입니다. "
    "상품 정보와 리뷰를 바탕으로 2~4문장으로 간단히 답변하세요. "
    "제품명, 가격, 핵심 특징 1~2개를 자연스럽게 포함하세요."
)

def _extract_product_name_keywords(query: str) -> str:
    """
    사용자 쿼리에서 제품명 추출
    1단계: 조사 제거 (가장 먼저)
    2단계: 불필요한 문구 제거
    """
    
    # 1단계: 조사 제거 (먼저 처리)
    # "런 100에 대해서" → "런 100 대해서"
    josa_pattern = r"(에서|에게|으로|로|와|과|이랑|랑|이나|나|만|도|조차|마저|에|은|는|을|를|이|가)(\s+|$)"
    cleaned = re.sub(josa_pattern, " ", query)
    
    # 2단계: 불필요한 문구 제거
    noise_words = [
        "대해서", "대해", "제품", "상품", 
        "알려줘", "알려주세요", "알려", "알려달라",
        "추천해줘", "추천해주세요", "찾아줘", "찾아주세요", 
        "어때", "어때요", "괜찮아", "괜찮은지", "좋아", "좋은지", 
        "어떤지", "정보", "소개", "보여줘", "보여주세요", 
        "있어", "있나요", "뭐야", "뭔가요", "어디", "어디서", 
        "구매", "살", "사려고", "사고싶어", "검색"
    ]
    
    for word in noise_words:
        cleaned = cleaned.replace(word, " ")
    
    # 3단계: 다중 공백 제거
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    print(f"🧹 원본 쿼리: {query}")
    print(f"🧹 정제된 제품명: {cleaned}")
    
    return cleaned


def _search_product_by_name_fuzzy(query: str, db: Session):
    """
    제품명 유사도 검색 (정확도 높임)
    1순위: 정확한 제품명 매칭
    2순위: 부분 문자열 매칭 (70% 이상)
    """
    # 제품명 키워드 추출
    clean_query = _extract_product_name_keywords(query)
    
    # 1단계: 정확 매칭 (search_by_name_fast 사용)
    exact_match = search_by_name_fast(clean_query, db)
    if exact_match:
        return exact_match
    
    # 2단계: 부분 매칭 (PostgreSQL LIKE 또는 similarity 사용)
    # 제품명에 쿼리의 주요 키워드가 모두 포함된 제품 찾기
    keywords = [w for w in clean_query.split() if len(w) > 1]
    
    if not keywords:
        return None
    
    # SQL: 모든 키워드가 제품명에 포함되어 있는지 확인
    like_conditions = " AND ".join([f"ps.product_name ILIKE '%{kw}%'" for kw in keywords])
    
    sql = text(f"""
        SELECT 
            ps.product_id, ps.product_name, ps.price, ps.avg_rating,
            ps.total_reviews, ps.url, ps.thumbnail_url,
            pi.explanation, pi.technical_info, pi.management_guidelines,
            LENGTH(ps.product_name) as name_length
        FROM product_summary ps
        LEFT JOIN product_information pi USING (product_id)
        WHERE {like_conditions}
        ORDER BY 
            ABS(LENGTH(ps.product_name) - :query_length) ASC,  -- 🔥 쿼리 길이와 가장 유사한 제품명
            ps.total_reviews DESC, 
            ps.avg_rating DESC
        LIMIT 1
    """)
    
    try:
        row = db.execute(sql, {"query_length": len(clean_query)}).mappings().first()
        if row:
            return {
                "product_id": row["product_id"],
                "name": row["product_name"],
                "price": row.get("price"),
                "rating": row.get("avg_rating"),
                "review_count": row.get("total_reviews"),
                "url": row.get("url"),
                "explanation": row.get("explanation"),
                "technical_info": row.get("technical_info"),
                "management": row.get("management_guidelines"),
            }
    except Exception as e:
        print(f"fuzzy search error: {e}")
        return None
    
    return None


@router.get("/health")
def health():
    try:
        meta = index_meta()
    except Exception as e:
        meta = {"error": str(e)}
    return {"ok": True, "index": meta, "env": ENV}

@router.post("/reindex")
def reindex(db: Session = Depends(get_db)):
    from services.rag_index import build_index_from_db
    info = build_index_from_db(db)
    return {"ok": True, "index": info}

@router.post("/chat", response_model=ChatOut)
def chat(
    req: ChatIn,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
    debug: bool = Query(False),
):
    t0 = time.time()
    session_id = _ensure_session(request, response, db)
    history = _load_recent_context(db, session_id, limit_turns=10)
    
    intent = _classify_intent(req.message, history)
    gender = _detect_gender(req.message)
    
    # ======== 1. 스몰톡 ========
    if intent == "greeting":
        if any(k in req.message.lower() for k in ["안녕", "하이", "ㅎㅇ", "hello"]):
            answer = (
                "안녕하세요! 데카트론 제품 추천 챗봇입니다.\n\n"
                "예시 질문:\n"
                "• '러닝화 추천해줘'\n"
                "• '방수 자켓 찾아줘'\n"
                "• '모델 번호 검색 정보 (8926414)'\n"
                "• '제품 이름 검색 정보 (남성 러닝 윈드 자켓 런 100)'\n"
                "• '초심자용 러닝 세트'\n\n"
                "※ 데카트론 제품 관련 질문만 답변 가능합니다."
                "예시:\n"
            )
        else:
            answer = "도움이 되셨다면 기쁩니다! 더 궁금한 점이 있으면 언제든 물어보세요!"
        
        _save_message(db, session_id, "user", req.message, meta={"route": intent})
        _save_message(db, session_id, "assistant", answer, meta={"route": intent})
        
        return ChatOut(answer=answer, recommendations=None, meta={
            "latency_ms": int((time.time() - t0) * 1000),
            "route": intent,
            "has_more": False
        }, session_id=session_id)
    
    # ======== 2. 모델번호 검색 ========
    if intent == "model_search":
        model_id = re.search(r"(\d{7})", req.message).group(1)
        product = search_by_model_id(model_id, db)
        
        if not product:
            answer = f"모델번호 {model_id}에 해당하는 제품을 찾을 수 없습니다."
            recs = None
        else:
            price_str = f"{product.get('price'):,}원" if product.get('price') else "가격 미정"
            answer = (
                f"네! 제품을 찾았습니다.\n\n"
                f"{product['name']}\n\n"
                f"가격: {price_str}\n"
                f"평점: {product.get('rating', 0)}/5.0 (리뷰 {product.get('review_count', 0)}개)"
            )
            
            if product.get("explanation"):
                answer += f"\n\n{product['explanation'][:150]}..."
            
            recs = [{
                "product_id": product["product_id"],
                "name": product["name"],
                "price": product.get("price"),
                "rating": product.get("rating"),
                "review_count": product.get("review_count"),
                "link": product.get("url"),
                "product_info": {
                    "explanation": product.get("explanation"),
                    "technical_info": product.get("technical_info"),
                    "management": product.get("management"),
                }
            }]
            _save_recommended_ids(db, session_id, [product["product_id"]])
        
        _save_message(db, session_id, "user", req.message, meta={"route": intent})
        _save_message(db, session_id, "assistant", answer, meta={"route": intent})
        
        return ChatOut(answer=answer, recommendations=recs, meta={
            "latency_ms": int((time.time() - t0) * 1000),
            "route": intent,
            "has_more": False,
            "exact_match": True
        }, session_id=session_id)
    
    # ======== 3. 인기 제품 추천 ========
    if intent == "popular":
        last_ids = _get_last_recommended_ids(db, session_id)
        
        gender_filter = ""
        if gender == "female":
            gender_filter = "AND ps.product_name LIKE '여성%'"
        elif gender == "male":
            gender_filter = "AND ps.product_name LIKE '남성%'"
        else:
            gender_filter = "AND ps.product_name NOT LIKE '여성%'"
        
        exclude_filter = ""
        if last_ids:
            exclude_list = "','".join(last_ids)
            exclude_filter = f"AND ps.product_id NOT IN ('{exclude_list}')"
        
        sql = text(f"""
            SELECT 
                ps.product_id, ps.product_name, ps.price, ps.avg_rating,
                ps.total_reviews, ps.url, ps.thumbnail_url, ps.category,
                pi.explanation, pi.technical_info, pi.management_guidelines
            FROM product_summary ps
            LEFT JOIN product_information pi USING (product_id)
            WHERE ps.total_reviews > 0
            {gender_filter}
            {exclude_filter}
            ORDER BY ps.total_reviews DESC, ps.avg_rating DESC
            LIMIT 1
        """)
        
        row = db.execute(sql).mappings().first()
        
        if not row:
            answer = "더 이상 추천할 인기 제품이 없습니다."
            recs = None
        else:
            answer = f"네! 가장 인기 있는 제품을 추천해드리겠습니다.\n\n{row['product_name']} (리뷰 {row['total_reviews']}개)"
            
            recs = [{
                "product_id": row["product_id"],
                "name": row["product_name"],
                "price": row.get("price"),
                "rating": row.get("avg_rating"),
                "review_count": row.get("total_reviews"),
                "link": row.get("url"),
                "product_info": {
                    "explanation": row.get("explanation"),
                    "technical_info": row.get("technical_info"),
                    "management": row.get("management_guidelines"),
                }
            }]
            
            new_ids = last_ids + [row["product_id"]]
            _save_recommended_ids(db, session_id, new_ids)
        
        _save_message(db, session_id, "user", req.message, meta={"route": intent})
        _save_message(db, session_id, "assistant", answer, meta={"route": intent})
        
        return ChatOut(answer=answer, recommendations=recs, meta={
            "latency_ms": int((time.time() - t0) * 1000),
            "route": intent,
            "has_more": False
        }, session_id=session_id)
    
    # ======== 4. 레벨 세트 추천 ========
    if intent == "level_set":
        level = "100"
        if any(k in req.message.lower() for k in ["중급", "500", "레귤러", "중간"]):
            level = "500"
        elif any(k in req.message.lower() for k in ["고수", "상급", "900", "프로", "고급", "전문"]):
            level = "900"
        
        category = "RUNNING"
        if any(k in req.message for k in ["하이킹", "등산", "트레킹", "hiking"]):
            category = "HIKING"
        
        try:
            result = recommend_level_set(level, category, db)
            items = result["items"]
            set_info = result["set_info"]
        except Exception as e:
            answer = f"세트 조회 중 오류가 발생했습니다: {str(e)}"
            _save_message(db, session_id, "user", req.message, meta={"route": intent, "error": str(e)})
            _save_message(db, session_id, "assistant", answer, meta={"route": intent})
            return ChatOut(answer=answer, recommendations=None, meta={
                "latency_ms": int((time.time() - t0) * 1000),
                "route": intent,
                "error": str(e)
            }, session_id=session_id)
        
        if not items or len(items) < 2:
            answer = (
                f"네! 하지만 {level} 레벨 세트 구성이 어렵습니다.\n"
                f"단일 제품 추천으로 전환하시겠어요? '러닝화 추천'처럼 구체적으로 물어보세요."
            )
            recs = None
            set_info = None
        else:
            level_kr = {"100": "초심자", "500": "중급자", "900": "고수"}[level]
            cat_kr = {"RUNNING": "러닝", "HIKING": "하이킹"}[category]
            
            answer = (
                f"네! {level_kr}용 {cat_kr} 세트를 추천해드리겠습니다!\n\n"
                f"구성: {', '.join(set_info['item_types'])}\n"
                f"총 가격: {set_info['total_price']:,}원"
            )
            
            recs = []
            for item in items:
                recs.append({
                    "type": item.get("type", "기타"),
                    "product_id": item["product_id"],
                    "name": item["name"],
                    "price": item.get("price", 0),
                    "rating": item.get("rating"),
                    "review_count": item.get("review_count", 0),
                    "link": item.get("url"),
                    "product_info": {
                        "explanation": item.get("explanation") or "상세 정보 준비 중",
                        "technical_info": item.get("technical_info"),
                        "management": item.get("management"),
                    }
                })
            
            _save_recommended_ids(db, session_id, [i["product_id"] for i in items])
        
        _save_message(db, session_id, "user", req.message, meta={"route": intent, "level": level, "category": category})
        _save_message(db, session_id, "assistant", answer, meta={"route": intent})
        
        return ChatOut(answer=answer, recommendations=recs, set_info=set_info, meta={
            "latency_ms": int((time.time() - t0) * 1000),
            "route": intent,
            "has_more": False
        }, session_id=session_id)
    
    # ======== 5. 후속 질문 ========
    if intent == "followup":
        last_query = _get_last_user_query(history)
        last_type = _get_last_product_type(db, session_id)
        last_ids = _get_last_recommended_ids(db, session_id)
        
        if not last_query:
            answer = "이전 추천 기록이 없습니다. 먼저 제품을 검색해주세요."
            recs = None
        else:
            try:
                last_gender = _detect_gender(last_query)
                
                search_query = last_query
                if last_type:
                    search_query = f"{last_type} {last_query}"
                
                results = hybrid_search(search_query, top_k=5, exclude_ids=last_ids, gender_filter=last_gender)
            except Exception as e:
                print(f"followup error: {e}")
                results = []
            
            if not results:
                answer = "더 이상 추천할 제품이 없습니다. 새로운 검색어로 다시 물어보세요."
                recs = None
            else:
                import random
                product = random.choice(results)
                price_str = f"{product.get('price'):,}원" if product.get('price') else "가격 미정"
                answer = f"네! 다른 제품도 있습니다. {product['name']} ({price_str})는 어떠세요?"
                
                recs = [{
                    "product_id": product["product_id"],
                    "name": product["name"],
                    "price": product.get("price"),
                    "rating": product.get("rating"),
                    "review_count": product.get("review_count"),
                    "link": product.get("url"),
                    "product_info": {
                        "explanation": product.get("explanation") or product.get("info_snippet", ""),
                        "technical_info": product.get("technical_info"),
                        "management": product.get("management"),
                    },
                    "top_reviews": [{"text": product.get("review_snippet", ""), "rating": product.get("rating")}]
                }]
                
                new_ids = last_ids + [product["product_id"]]
                _save_recommended_ids(db, session_id, new_ids)
        
        _save_message(db, session_id, "user", req.message, meta={"route": intent})
        _save_message(db, session_id, "assistant", answer, meta={"route": intent})
        
        return ChatOut(answer=answer, recommendations=recs, meta={
            "latency_ms": int((time.time() - t0) * 1000),
            "route": intent,
            "has_more": bool(results) if 'results' in locals() else False
        }, session_id=session_id)
    
    # ======== 6. 일반 추천 (제품명 우선 검색) ========
    product_match = _search_product_by_name_fuzzy(req.message, db)
    
    if product_match:
        price_str = f"{product_match.get('price'):,}원" if product_match.get('price') else "가격 미정"
        answer = (
            f"네! 제품을 찾았습니다.\n\n"
            f"{product_match['name']}\n\n"
            f"가격: {price_str}\n"
            f"평점: {product_match.get('rating', 0)}/5.0 (리뷰 {product_match.get('review_count', 0)}개)"
        )
        
        if product_match.get("explanation"):
            answer += f"\n\n{product_match['explanation'][:200]}..."
        
        recs = [{
            "product_id": product_match["product_id"],
            "name": product_match["name"],
            "price": product_match.get("price"),
            "rating": product_match.get("rating"),
            "review_count": product_match.get("review_count"),
            "link": product_match.get("url"),
            "product_info": {
                "explanation": product_match.get("explanation"),
                "technical_info": product_match.get("technical_info"),
                "management": product_match.get("management"),
            }
        }]
        _save_recommended_ids(db, session_id, [product_match["product_id"]])
        
        _save_message(db, session_id, "user", req.message, meta={"route": "product_name_match"})
        _save_message(db, session_id, "assistant", answer, meta={"route": "product_name_match"})
        
        return ChatOut(answer=answer, recommendations=recs, meta={
            "latency_ms": int((time.time() - t0) * 1000),
            "route": "product_name_match",
            "has_more": False,
            "exact_match": True
        }, session_id=session_id)
    
    # 데카트론 제품 관련 키워드 체크
    product_keywords = [
        # 제품 카테고리
        "러닝", "하이킹", "등산", "자전거", "수영", "요가", "헬스", "축구", "농구", "배드민턴",
        # 제품 타입
        "신발", "화", "슈즈", "티", "자켓", "재킷", "바지", "팬츠", "쇼츠", "레깅스", "가방", "백팩",
        "모자", "장갑", "양말", "선글라스", "시계", "텐트", "침낭", "매트", "배낭",
        # 일반 키워드
        "운동", "스포츠", "아웃도어", "트레이닝", "피트니스", "decathlon", "데카트론",
        "추천", "제품", "상품", "구매", "찾", "알려"
    ]
    
    q_lower = req.message.lower()
    has_product_keyword = any(k in q_lower for k in product_keywords)
    
    if not has_product_keyword:
        answer = (
            "죄송합니다. 데카트론 스포츠 용품과 관련된 질문을 입력해주세요.\n\n"
            "예시:\n"
            "• '러닝화 추천해줘'\n"
            "• '방수 자켓 찾아줘'\n"
            "• '초심자용 요가 매트'\n"
            "• '8926414 제품 정보'\n\n"
            "데카트론 제품 관련 질문만 답변 가능합니다."
        )
        recs = None
        
        _save_message(db, session_id, "user", req.message, meta={"route": "non_product_query"})
        _save_message(db, session_id, "assistant", answer, meta={"route": "non_product_query"})
        
        return ChatOut(answer=answer, recommendations=None, meta={
            "latency_ms": int((time.time() - t0) * 1000),
            "route": "non_product_query",
            "has_more": False
        }, session_id=session_id)
    
    # 임베딩 검색
    try:
        results = hybrid_search(req.message, top_k=5, gender_filter=gender)
    except Exception as e:
        answer = f"검색 중 오류가 발생했습니다: {str(e)}"
        _save_message(db, session_id, "user", req.message, meta={"route": intent, "error": str(e)})
        _save_message(db, session_id, "assistant", answer, meta={"route": intent})
        return ChatOut(answer=answer, recommendations=None, meta={
            "latency_ms": int((time.time() - t0) * 1000),
            "route": intent,
            "error": str(e)
        }, session_id=session_id)
    
    # 유사도 점수 체크 (너무 낮으면 제품 못찾음)
    if not results or (results and results[0].get("score", 0) < 0.3):
        answer = (
            "죄송합니다. 요청하신 조건에 맞는 제품을 찾을 수 없습니다.\n\n"
            "다음과 같이 질문해주세요:\n"
            "• 모델번호: '8926414'\n"
            "• 제품명: '남성 러닝 윈드 자켓 런 100'\n"
            "• 카테고리: '러닝화 추천', '방수 자켓'\n"
            "• 세트: '초심자용 러닝 세트', '500 세트'"
        )
        recs = None
    else:
        import random
        top = random.choice(results)
        explanation = top.get('explanation') or top.get('info_snippet', '')
        review = top.get('review_snippet', '')
        
        prompt = (
            f"사용자 질문: {req.message}\n\n"
            f"추천 제품:\n"
            f"- 이름: {top['name']}\n"
            f"- 가격: {top.get('price', '미정')}원\n"
            f"- 설명: {explanation[:200]}\n"
            f"- 리뷰: {review}\n\n"
            f"'네! 제품 추천해드리겠습니다'로 시작하여 2~3문장으로 간단히 추천 사유를 설명하세요."
        )
        
        llm_messages = [{"role": "user", "content": prompt}]
        answer = _call_llm(SYSTEM_PROMPT, llm_messages)
        
        recs = [{
            "product_id": top["product_id"],
            "name": top["name"],
            "price": top.get("price"),
            "rating": top.get("rating"),
            "review_count": top.get("review_count"),
            "link": top.get("url"),
            "product_info": {
                "explanation": top.get("explanation") or top.get("info_snippet", ""),
                "technical_info": top.get("technical_info"),
                "management": top.get("management"),
            },
            "top_reviews": [{"text": top.get("review_snippet", ""), "rating": top.get("rating")}]
        }]
        
        _save_recommended_ids(db, session_id, [top["product_id"]])
    
    _save_message(db, session_id, "user", req.message, meta={"route": intent})
    _save_message(db, session_id, "assistant", answer, meta={"route": intent})
    
    return ChatOut(answer=answer, recommendations=recs, meta={
        "latency_ms": int((time.time() - t0) * 1000),
        "route": intent,
        "has_more": len(results) > 1 if results else False
    }, session_id=session_id)
