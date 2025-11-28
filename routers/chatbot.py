# routers/chatbot.py (완전 개선 버전)
from fastapi import APIRouter, Depends, Request, Response, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from uuid import uuid4
from datetime import datetime, timedelta
import os, time, json, re

from sqlalchemy.orm import Session
from sqlalchemy import text
from core.db import get_db

from services.rag_index import hybrid_search, search_by_model_id, recommend_level_set, index_meta
from openai import OpenAI

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

# 환경
ENV = os.getenv("ENV", "dev").lower()
COOKIE_NAME = "session_id"
SESSION_TTL_HOURS = int(os.getenv("SESSION_TTL_HOURS", "24"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 스키마
class ChatIn(BaseModel):
    message: str

class ChatOut(BaseModel):
    answer: str
    recommendations: Optional[List[Dict[str, Any]]] = None
    set_info: Optional[Dict[str, Any]] = None
    meta: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

# ======== 세션 관리 (개선) ========
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
        # 테이블 없을 때 방어
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
    """세션에 저장된 마지막 추천 제품 ID 목록"""
    try:
        row = db.execute(text("SELECT last_recommended_ids FROM sessions WHERE id=:sid"), {"sid": session_id}).first()
        if not row or not row[0]: return []
        return [x for x in row[0].split(",") if x]
    except:
        return []

def _save_recommended_ids(db, session_id, product_ids: List[str]):
    """추천한 제품 ID들을 세션에 저장"""
    ids_str = ",".join([str(x) for x in product_ids if x])
    try:
        db.execute(text("UPDATE sessions SET last_recommended_ids=:ids WHERE id=:sid"), {"ids": ids_str, "sid": session_id})
        db.commit()
    except:
        pass

# ======== 개선된 인텐트 분류 ========
def _classify_intent(query: str, history: List[Dict]) -> str:
    """
    우선순위:
    1. 모델번호 (7자리 숫자)
    2. 세트/레벨 요청
    3. 후속 질문
    4. 스몰톡
    5. 일반 추천
    """
    q = query.lower()
    
    # 1순위: 모델번호
    if re.search(r"\b\d{7}\b", q):
        return "model_search"
    
    # 2순위: 세트 요청 (강화!)
    set_keywords = ["세트", "풀세트", "조합", "구성", "셋트", "패키지", "set"]
    level_keywords = ["초심자", "중급자", "고수", "입문", "비기너", "프로", "상급", "초보", "100", "500", "900"]
    
    # 세트 또는 레벨 키워드 → 무조건 세트 모드
    if any(k in q for k in set_keywords):
        return "level_set"
    if any(k in q for k in level_keywords) and any(k2 in q for k2 in ["추천", "알려", "찾아"]):
        return "level_set"
    
    # 3순위: 후속 질문
    if any(k in q for k in ["다른", "또", "더", "그 외", "다른거", "다른 거", "하나 더", "추가"]):
        return "followup"
    
    # 4순위: 스몰톡
    if any(k in q for k in ["안녕", "하이", "ㅎㅇ", "고마워", "감사", "땡큐", "thank"]):
        return "greeting"
    
    # 기본: 일반 추천
    return "recommendation"

# ======== LLM 호출 ========
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

# ======== Routes ========
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
    history = _load_recent_context(db, session_id, limit_turns=6)
    
    # 인텐트 분류
    intent = _classify_intent(req.message, history)
    
    # ======== 1. 스몰톡 ========
    if intent == "greeting":
        if any(k in req.message.lower() for k in ["안녕", "하이", "ㅎㅇ", "hello"]):
            answer = (
                "안녕하세요! 데카트론 제품 추천 챗봇입니다 \n\n"
                "예시 질문:\n"
                "• '초심자용 러닝 세트 추천해줘'\n"
                "• '방수 자켓 찾아줘'\n"
                "• '8751038 제품 알려줘'\n\n"
                "※ 데카트론 제품 관련 질문만 답변 가능합니다."
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
        model_id = re.search(r"\b(\d{7})\b", req.message).group(1)
        product = search_by_model_id(model_id, db)
        
        if not product:
            answer = f"모델번호 {model_id}에 해당하는 제품을 찾을 수 없습니다."
            recs = None
        else:
            price_str = f"{product.get('price'):,}원" if product.get('price') else "가격 미정"
            answer = (
                f"{product['name']}\n\n"
                f"가격: {price_str}\n"
                f"평점: {product.get('rating', 0)}/5.0 (리뷰 {product.get('review_count', 0)}개)"
            )
            
            if product.get("explanation"):
                answer += f"\n\n {product['explanation'][:150]}..."
            
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
    
    # ======== 3. 레벨 세트 추천 (핵심 개선!) ========
    if intent == "level_set":
        # 레벨 추출 (기본값 100)
        level = "100"  # ← 기본값!
        if any(k in req.message.lower() for k in ["중급", "500", "레귤러", "중간"]):
            level = "500"
        elif any(k in req.message.lower() for k in ["고수", "상급", "900", "프로", "고급", "전문"]):
            level = "900"
        elif any(k in req.message.lower() for k in ["초심자", "입문", "100", "비기너", "초보", "beginner"]):
            level = "100"
        
        # 카테고리 추출 (기본값 RUNNING)
        category = "RUNNING"
        if any(k in req.message for k in ["하이킹", "등산", "트레킹", "hiking"]):
            category = "HIKING"
        
        # 세트 조회
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
        
        # 최소 2개는 있어야 세트!
        if not items or len(items) < 2:
            answer = (
                f"{level} 레벨 세트 구성이 어렵습니다.\n"
                f"단일 제품 추천으로 전환하시겠어요? '러닝화 추천'처럼 구체적으로 물어보세요."
            )
            recs = None
            set_info = None
        else:
            level_kr = {"100": "초심자", "500": "중급자", "900": "고수"}[level]
            cat_kr = {"RUNNING": "러닝", "HIKING": "하이킹"}[category]
            
            answer = (
                f"{level_kr}용 {cat_kr} 세트를 추천드립니다!\n\n"
                f"구성: {', '.join(set_info['item_types'])}\n"
                f"총 가격: {set_info['total_price']:,}원"
            )
            
            recs = []
            for item in items:
                price_val = item.get("price")
                recs.append({
                    "type": item.get("type", "기타"),
                    "product_id": item["product_id"],
                    "name": item["name"],
                    "price": price_val if price_val else 0,
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
    
    # ======== 4. 후속 질문 ========
    if intent == "followup":
        last_ids = _get_last_recommended_ids(db, session_id)
        offset = len(last_ids)
        
        # 마지막 검색 쿼리가 있으면 재사용 (간소화: 여기선 현재 쿼리 사용)
        try:
            results = hybrid_search(req.message, top_k=1, offset=offset)
        except:
            results = []
        
        if not results:
            answer = "더 이상 추천할 제품이 없습니다. 새로운 검색어로 다시 물어보세요."
            recs = None
        else:
            product = results[0]
            price_str = f"{product.get('price'):,}원" if product.get('price') else "가격 미정"
            answer = f"{offset+1}번째 추천: {product['name']} ({price_str})"
            
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
            "has_more": len(results) > 0,
            "offset": offset + 1
        }, session_id=session_id)
    
    # ======== 5. 일반 추천 ========
    try:
        results = hybrid_search(req.message, top_k=3)
    except Exception as e:
        answer = f"검색 중 오류가 발생했습니다: {str(e)}"
        _save_message(db, session_id, "user", req.message, meta={"route": intent, "error": str(e)})
        _save_message(db, session_id, "assistant", answer, meta={"route": intent})
        return ChatOut(answer=answer, recommendations=None, meta={
            "latency_ms": int((time.time() - t0) * 1000),
            "route": intent,
            "error": str(e)
        }, session_id=session_id)
    
    if not results:
        answer = "요청과 맞는 제품을 찾기 어렵네요. 카테고리(러닝/하이킹)나 예산을 조금 더 알려주세요."
        recs = None
    else:
        top = results[0]
        
        # 상품정보 우선, 없으면 리뷰 스니펫
        explanation = top.get('explanation') or top.get('info_snippet', '')
        review = top.get('review_snippet', '')
        
        prompt = (
            f"사용자 질문: {req.message}\n\n"
            f"추천 제품:\n"
            f"- 이름: {top['name']}\n"
            f"- 가격: {top.get('price', '미정')}원\n"
            f"- 설명: {explanation[:200]}\n"
            f"- 리뷰: {review}\n\n"
            f"2~3문장으로 간단히 추천 사유를 설명하세요."
        )
        
        llm_messages = [{"role": "user", "content": prompt}]
        answer = _call_llm(SYSTEM_PROMPT, llm_messages)
        
        recs = []
        for r in results:
            recs.append({
                "product_id": r["product_id"],
                "name": r["name"],
                "price": r.get("price"),
                "rating": r.get("rating"),
                "review_count": r.get("review_count"),
                "link": r.get("url"),
                "product_info": {
                    "explanation": r.get("explanation") or r.get("info_snippet", ""),
                    "technical_info": r.get("technical_info"),
                    "management": r.get("management"),
                },
                "top_reviews": [{"text": r.get("review_snippet", ""), "rating": r.get("rating")}]
            })
        
        _save_recommended_ids(db, session_id, [r["product_id"] for r in results])
    
    _save_message(db, session_id, "user", req.message, meta={"route": intent})
    _save_message(db, session_id, "assistant", answer, meta={"route": intent})
    
    return ChatOut(answer=answer, recommendations=recs, meta={
        "latency_ms": int((time.time() - t0) * 1000),
        "route": intent,
        "has_more": len(results) > 0
    }, session_id=session_id)
