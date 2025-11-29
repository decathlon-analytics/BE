# services/rag_index.py (gzip 압축 버전)
from __future__ import annotations
from typing import List, Dict, Any, Optional
import os, re, pickle, gzip
from datetime import datetime

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session
from sqlalchemy import text
from openai import OpenAI

# 환경설정
PKL_PATH = os.environ.get("RAG_EMB_PATH", "data/embeddings.pkl")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# 전역 캐시 추가
_INDEX_CACHE = None

# ======== 유틸리티 ========
def _basic_clean(text_: str) -> str:
    if not isinstance(text_, str): return ""
    t = text_.strip()
    bad_patterns = [
        r"홈>.*", r"장바구니.*", r"사이즈를 선택하세요.*", r"쿠폰코드\s*:\s*\w+",
        r"판매자\s*DECATHLON.*", r"\b\d+/?\d+\b", r"[|｜]\s*쿠폰.*", r"리뷰 작성하기.*"
    ]
    for p in bad_patterns:
        t = re.sub(p, " ", t)
    t = re.sub(r"^[A-Za-z0-9_]+[\s:]+", "", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

def _cosine_sim(query_vec: np.ndarray, mat: np.ndarray) -> np.ndarray:
    q = query_vec.astype(float)
    M = mat.astype(float)
    qn = np.linalg.norm(q) + 1e-12
    mn = np.linalg.norm(M, axis=1) + 1e-12
    return (M @ q) / (mn * qn)

def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

# ======== 제품 타입/레벨 추출 ========
def _extract_level(name: str) -> Optional[str]:
    if not name: return None
    m = re.search(r"\b(100|500|900)\b", name)
    return m.group(1) if m else None

def _classify_item_type(name: str) -> str:
    if not name: return "기타"
    n = name.lower()
    if any(k in n for k in ["티", "자켓", "재킷", "베스트", "싱글렛", "긴팔", "반팔", "하프집"]):
        return "상의"
    if any(k in n for k in ["바지", "쇼츠", "팬츠", "레깅스"]):
        return "하의"
    if any(k in n for k in ["러닝화", "등산화", "슈즈", "신발"]):
        return "신발"
    if "장갑" in n:
        return "장갑"
    return "기타"

def _load_index():
    """전역 캐싱으로 메모리 절약"""
    global _INDEX_CACHE
    if _INDEX_CACHE is None:
        if not os.path.exists(PKL_PATH):
            return None
        with gzip.open(PKL_PATH, "rb") as f:
            _INDEX_CACHE = pickle.load(f)
    return _INDEX_CACHE

# ======== 통합 인덱스 빌드 (리뷰 + 상품정보) ========
def build_index_from_db(db: Session, *, limit: Optional[int] = None, batch: int = 128) -> Dict[str, Any]:
    
    global _INDEX_CACHE
    _INDEX_CACHE = None  # 캐시 초기화
    
    # 1. 리뷰 데이터
    review_df = pd.read_sql(text("""
        SELECT
            r.product_id,
            COALESCE(r.category, ps.category) AS category,
            COALESCE(r.product_name, ps.product_name) AS name,
            ps.price, ps.avg_rating, ps.total_reviews,
            ps.url, ps.thumbnail_url,
            r.review_text
        FROM reviews r
        LEFT JOIN product_summary ps USING (product_id)
        WHERE r.review_text IS NOT NULL AND length(r.review_text) >= 10
    """), db.bind)
    
    if limit:
        review_df = review_df.head(int(limit))
    
    review_df["clean_text"] = review_df["review_text"].map(_basic_clean)
    review_df = review_df[review_df["clean_text"].str.len() >= 10].copy()
    
    review_docs, review_texts = [], []
    for _, r in review_df.iterrows():
        review_docs.append({
            "product_id": r.get("product_id"),
            "category": r.get("category"),
            "name": r.get("name"),
            "price": r.get("price"),
            "rating": r.get("avg_rating"),
            "review_count": r.get("total_reviews"),
            "url": r.get("url"),
            "thumbnail_url": r.get("thumbnail_url"),
            "text": r.get("clean_text"),
            "source": "review"
        })
        review_texts.append(r.get("clean_text"))
    
    # 2. 상품정보 데이터
    info_df = pd.read_sql(text("""
        SELECT
            pi.product_id,
            pi.product_name AS name,
            pi.brand,
            pi.explanation,
            pi.technical_info,
            pi.management_guidelines,
            pi.url,
            ps.price, ps.avg_rating, ps.total_reviews,
            ps.category, ps.thumbnail_url
        FROM product_information pi
        LEFT JOIN product_summary ps USING (product_id)
        WHERE pi.explanation IS NOT NULL
    """), db.bind)
    
    if limit:
        info_df = info_df.head(int(limit))
    
    info_df["combined_text"] = (
        info_df["name"].fillna("") + " " +
        info_df["explanation"].fillna("") + " " +
        info_df["technical_info"].fillna("")
    ).str.strip()
    info_df = info_df[info_df["combined_text"].str.len() >= 20].copy()
    
    info_docs, info_texts = [], []
    for _, r in info_df.iterrows():
        info_docs.append({
            "product_id": r.get("product_id"),
            "name": r.get("name"),
            "category": r.get("category"),
            "price": r.get("price"),
            "rating": r.get("avg_rating"),
            "review_count": r.get("total_reviews"),
            "url": r.get("url"),
            "thumbnail_url": r.get("thumbnail_url"),
            "explanation": r.get("explanation"),
            "technical_info": r.get("technical_info"),
            "management": r.get("management_guidelines"),
            "text": r.get("combined_text"),
            "source": "product_info"
        })
        info_texts.append(r.get("combined_text"))
    
    # 3. 임베딩 생성
    all_texts = review_texts + info_texts
    all_embeddings = []
    for i in range(0, len(all_texts), batch):
        chunk = all_texts[i:i+batch]
        resp = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=chunk)
        all_embeddings.extend([d.embedding for d in resp.data])
    
    # 4. 분리
    review_embeddings = all_embeddings[:len(review_texts)]
    info_embeddings = all_embeddings[len(review_texts):]
    
    # 5. gzip 압축 저장
    os.makedirs(os.path.dirname(PKL_PATH) or ".", exist_ok=True)
    payload = {
        "review_embeddings": review_embeddings,
        "review_docs": review_docs,
        "info_embeddings": info_embeddings,
        "info_docs": info_docs,
        "built_at": _now_iso(),
        "embed_model": f"openai:{OPENAI_EMBED_MODEL}",
    }
    
    with gzip.open(PKL_PATH, "wb") as f:
        pickle.dump(payload, f)
    
    return {
        "review_chunks": len(review_docs),
        "info_chunks": len(info_docs),
        "total_chunks": len(review_docs) + len(info_docs),
        "built_at": payload["built_at"],
        "path": PKL_PATH
    }

# ======== 하이브리드 검색 ========
def hybrid_search(
    query: str,
    top_k: int = 3,
    *,
    exclude_ids: List[str] = None,  # offset 대신 이걸로
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
) -> List[Dict[str, Any]]:
    idx = _load_index()
    if not idx:
        return []
    
    q_emb = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=query).data[0].embedding
    q_vec = np.array(q_emb)
    
    info_results = []
    if idx.get("info_embeddings"):
        # numpy 변환 없이 직접 계산
        info_embs = idx["info_embeddings"]  # list 그대로
        
        for i, emb in enumerate(info_embs):
            emb_vec = np.array(emb, dtype=np.float32)
            sim = np.dot(q_vec, emb_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(emb_vec) + 1e-12)
            
            d = idx["info_docs"][i]
            info_results.append({
                "product_id": d["product_id"],
                "score": float(sim) * 0.6,
                "name": d["name"],
                "price": d["price"],
                "rating": d["rating"],
                "review_count": d["review_count"],
                "url": d["url"],
                "thumbnail_url": d["thumbnail_url"],
                "explanation": d.get("explanation"),
                "technical_info": d.get("technical_info"),
                "management": d.get("management"),
                "info_snippet": d["text"][:200],
            })
    
    # 리뷰도 동일
    review_results = []
    if idx.get("review_embeddings"):
        review_embs = idx["review_embeddings"]
        
        for i, emb in enumerate(review_embs):
            emb_vec = np.array(emb, dtype=np.float32)
            sim = np.dot(q_vec, emb_vec) / (np.linalg.norm(q_vec) * np.linalg.norm(emb_vec) + 1e-12)
            
            d = idx["review_docs"][i]
            review_results.append({
                "product_id": d["product_id"],
                "score": float(sim) * 0.4,
                "review_snippet": d["text"][:150],
            })
    
    # 3. 제품별 통합
    merged = {}
    for r in info_results:
        pid = r["product_id"]
        merged[pid] = r
    
    for r in review_results:
        pid = r["product_id"]
        if pid in merged:
            merged[pid]["score"] += r["score"]
            merged[pid]["review_snippet"] = r["review_snippet"]
    
    # 4. 가격 필터
    filtered = []
    exclude_set = set(exclude_ids or [])
    
    for pid, data in merged.items():
        if pid in exclude_set:  # 이미 추천한 제품 제외
            continue
        
        price = data.get("price")
        if min_price and (not price or price < min_price): continue
        if max_price and (not price or price > max_price): continue
        filtered.append(data)
    
    # 5. 정렬
    filtered.sort(key=lambda x: x["score"], reverse=True)
    filtered = filtered[:top_k]
    
    return filtered

# ======== 모델번호 직접 검색 ========
def search_by_model_id(model_id: str, db: Session) -> Optional[Dict[str, Any]]:
    row = db.execute(text("""
        SELECT
            ps.product_id, ps.product_name, ps.price, ps.avg_rating,
            ps.total_reviews, ps.url, ps.thumbnail_url, ps.category,
            pi.explanation, pi.technical_info, pi.management_guidelines
        FROM product_summary ps
        LEFT JOIN product_information pi USING (product_id)
        WHERE ps.product_id = :pid
    """), {"pid": model_id}).mappings().first()
    
    if not row: return None
    
    return {
        "product_id": row["product_id"],
        "name": row["product_name"],
        "price": row["price"],
        "rating": row["avg_rating"],
        "review_count": row["total_reviews"],
        "url": row["url"],
        "thumbnail_url": row["thumbnail_url"],
        "category": row["category"],
        "explanation": row.get("explanation"),
        "technical_info": row.get("technical_info"),
        "management": row.get("management_guidelines"),
        "source": "exact_match",
        "score": 1.0,
    }

# ======== 레벨별 세트 추천 ========
def recommend_level_set(level: str, category: str, db: Session) -> Dict[str, Any]:
    top_row = db.execute(text("""
        SELECT product_id FROM product_summary
        WHERE category = :cat
          AND product_name LIKE :lv
          AND (product_name LIKE '%티%' OR product_name LIKE '%자켓%' OR product_name LIKE '%재킷%')
        ORDER BY total_reviews DESC, avg_rating DESC
        LIMIT 1
    """), {"cat": category.upper(), "lv": f"%{level}%"}).mappings().first()
    
    bottom_row = db.execute(text("""
        SELECT product_id FROM product_summary
        WHERE category = :cat
          AND product_name LIKE :lv
          AND (product_name LIKE '%바지%' OR product_name LIKE '%쇼츠%' OR product_name LIKE '%팬츠%')
        ORDER BY total_reviews DESC, avg_rating DESC
        LIMIT 1
    """), {"cat": category.upper(), "lv": f"%{level}%"}).mappings().first()
    
    glove_row = db.execute(text("""
        SELECT product_id FROM product_summary
        WHERE category = :cat
          AND product_name LIKE '%장갑%'
        ORDER BY total_reviews DESC
        LIMIT 1
    """), {"cat": category.upper()}).mappings().first()
    
    items = []
    total_price = 0
    
    for row, item_type in [(top_row, "상의"), (bottom_row, "하의"), (glove_row, "장갑")]:
        if not row: continue
        detail = search_by_model_id(row["product_id"], db)
        if detail:
            detail["type"] = item_type
            items.append(detail)
            total_price += (detail.get("price") or 0)
    
    return {
        "items": items,
        "set_info": {
            "level": level,
            "category": category,
            "total_price": total_price,
            "item_types": [i["type"] for i in items]
        }
    }

# ======== 메타 ========
def index_meta() -> Dict[str, Any]:
    try:
        idx = _load_index()
        if not idx:
            return {"error": "Index not found", "path": PKL_PATH}

        return {
            "review_chunks": len(idx.get("review_docs", [])),
            "info_chunks": len(idx.get("info_docs", [])),
            "total_chunks": len(idx.get("review_docs", [])) + len(idx.get("info_docs", [])),
            "built_at": idx.get("built_at"),
            "embed_model": idx.get("embed_model"),
            "path": PKL_PATH
        }
    except Exception as e:
        return {"error": str(e), "path": PKL_PATH}

__all__ = ["build_index_from_db", "hybrid_search", "search_by_model_id", "recommend_level_set", "index_meta"]
