# services/ingest.py
from typing import Dict, List, Optional
import pandas as pd
from sqlalchemy import text
import json
from core.db import (
    engine,
    ensure_tables,          
    ensure_summary_table,
    ensure_product_information_table,  
)
from services.utils import (
    safe_str, parse_date_any, deterministic_review_id,
    normalize_category, now_iso, to_int_any, to_float_any
)

# ── 컬럼 매핑
REVIEW_MAP = {
    "product_id":   ["product_id","pid","p_id"],
    "product_name": ["product_name","name","title","product"],
    "category":     ["category","cat"],
    "subcategory":  ["subcategory","sub_cat","subCategory"],
    "brand":        ["brand","maker","product_brand"],
    "rating":       ["rating","score","stars"],
    "review_text":  ["review_text","content","body","excerpt"],
    "sentiment":    ["sentiment"],
    "review_date":  ["date","review_date","created_at","createdAt"],
}

SUMMARY_MAP = {
    "product_id":       ["product_id","pid","p_id"],
    "product_name":     ["product_name","name","title"],
    "category":         ["category","cat"],
    "subcategory":      ["subcategory","sub_cat","subCategory"],
    "brand":            ["brand","maker","product_brand"],
    "price":            ["price","가격","product_price"],
    "total_reviews":    ["total_reviews","review_count","reviews_total"],
    "positive_reviews": ["positive_reviews","positive"],
    "mixed_reviews":    ["mixed_reviews","neutral","mixed"],
    "negative_reviews": ["negative_reviews","negative"],
    "avg_rating":       ["avg_rating","rating_avg","avgScore"],
    "url":              ["url","product_url"],
    "thumbnail_url":    ["thumbnail_url","thumbnail","thumb_url","image","image_url"],
    "updated_at":       ["updated_at","as_of","refreshed_at"],
}

PRODUCT_INFO_MAP = {
    "product_id":           ["product_id", "pid", "p_id"],
    "product_name":         ["product_name", "name", "title"],
    "brand":                ["brand", "maker", "product_brand"],
    "explanation":          ["explanation", "desc", "description"],
    "technical_info":       ["technical_info", "tech_info", "details"],
    "management_guidelines":["management_guidelines", "care", "관리지침"],
    "url":                  ["url", "URL", "product_url"],
}


PRODUCT_INFO_STD_COLS = ["product_id", "product_name", "category", "subcategory", "brand"]

def map_columns(df: pd.DataFrame, mapping: Dict[str, List[str]]) -> pd.DataFrame:
    rename_map = {}
    lower = {c.lower(): c for c in df.columns}
    for std, cands in mapping.items():
        src = None
        for c in cands:
            if c in df.columns: src = c; break
            if c.lower() in lower: src = lower[c.lower()]; break
        if src:
            rename_map[src] = std
        else:
            df[std] = None
    df = df.rename(columns=rename_map)
    cols = list(mapping.keys())
    for k in cols:
        if k not in df.columns:
            df[k] = None
    return df[cols]

# ── 리뷰 → 요약 숫자 재집계
def _recalc_summary_counts(conn, product_ids: Optional[List[str]] = None):
    params = {}
    where = ""
    if product_ids:
        placeholders = ",".join([f":pid{i}" for i in range(len(product_ids))])
        for i, v in enumerate(product_ids):
            params[f"pid{i}"] = str(v)
        where = f"WHERE product_id IN ({placeholders})"

    sql = f"""
    WITH agg AS (
        SELECT
            product_id,
            COUNT(*) AS total_reviews,
            AVG(NULLIF(rating, 0)) AS avg_rating,
            SUM(CASE
                WHEN COALESCE(sentiment,'') ILIKE 'pos%%' THEN 1
                WHEN rating >= 4.5 THEN 1 ELSE 0 END) AS positive_reviews,
            SUM(CASE WHEN COALESCE(sentiment,'') ILIKE 'mix%%' THEN 1 ELSE 0 END) AS mixed_reviews,
            SUM(CASE
                WHEN COALESCE(sentiment,'') ILIKE 'neg%%' THEN 1
                WHEN rating <= 2.5 THEN 1 ELSE 0 END) AS negative_reviews
        FROM reviews
        {where}
        GROUP BY product_id
    )
    INSERT INTO product_summary AS ps (
        product_id, total_reviews, avg_rating, positive_reviews, mixed_reviews, negative_reviews
    )
    SELECT product_id, total_reviews, COALESCE(avg_rating,0), positive_reviews, mixed_reviews, negative_reviews
    FROM agg
    ON CONFLICT (product_id) DO UPDATE
      SET total_reviews    = EXCLUDED.total_reviews,
          avg_rating       = EXCLUDED.avg_rating,
          positive_reviews = EXCLUDED.positive_reviews,
          mixed_reviews    = EXCLUDED.mixed_reviews,
          negative_reviews = EXCLUDED.negative_reviews;
    """
    conn.execute(text(sql), params)

# ── Reviews Ingest
def ingest_reviews_df(df: pd.DataFrame):
    ensure_tables()

    df = map_columns(df, REVIEW_MAP)

    # 타입 변환
    df["product_id"]   = df["product_id"].apply(safe_str)
    df["product_name"] = df["product_name"].apply(safe_str)
    df["category"]     = df["category"].apply(normalize_category)
    df["subcategory"]  = df["subcategory"].apply(safe_str)
    df["brand"]        = df["brand"].apply(safe_str)
    df["rating"]       = pd.to_numeric(df["rating"], errors="coerce")
    df["review_text"]  = df["review_text"].apply(safe_str)
    df["sentiment"]    = df["sentiment"].apply(safe_str)
    df["review_date"]  = df["review_date"].apply(parse_date_any)

    # 첫 업로드 여부
    with engine.begin() as conn:
        total_in_db = conn.execute(text("SELECT COUNT(*) FROM reviews;")).scalar() or 0
        is_first_upload = (total_in_db == 0)

    # 리뷰ID 생성(파일 내부 충돌 방지용 idx 포함) — utils에서 idx 인자 지원해야 함
    df["review_id"] = [
        deterministic_review_id(r.get("product_id"), r.get("review_date"), r.get("review_text"), idx=i)
        for i, r in df.iterrows()
    ]

    # 파일 내부 중복 제거는 하지 않음(요청 사항)
    # df = df.drop_duplicates(subset=["review_id"]).reset_index(drop=True)

    inserted, affected_ids = 0, []
    with engine.begin() as conn:
        # DB 중복 제거는 “첫 업로드가 아닐 때”만 수행
        if not is_first_upload:
            existing = pd.read_sql(text("SELECT review_id FROM reviews;"), conn)
            if not existing.empty:
                have = set(existing["review_id"].astype(str).tolist())
                df = df[~df["review_id"].astype(str).isin(have)]

        if not df.empty:
            df.to_sql("reviews", con=conn, if_exists="append", index=False, method="multi", chunksize=1000)
            inserted = int(len(df))
            affected_ids = df["product_id"].dropna().astype(str).unique().tolist()

        # 리뷰 적재 후 자동 재집계는 “첫 업로드가 아닐 때만”
        if (not is_first_upload) and affected_ids:
            # 숫자 집계 넣을 표가 없을 수도 있으니 그때 보장(반드시 conn과 함께)
            ensure_summary_table(conn)
            _recalc_summary_counts(conn, affected_ids)

    return {
        "ok": True,
        "inserted": inserted,
        "recounted_products": 0 if is_first_upload else len(affected_ids),
        "first_upload": is_first_upload
    }

# ---- Summary Ingest (메타 전용, 단 ‘첫 업로드’는 숫자도 반영)
# services/ingest.py 내부
def ingest_summary_df(df: pd.DataFrame):
    ensure_tables()

    df = map_columns(df, SUMMARY_MAP)

    df["product_id"]   = df["product_id"].apply(safe_str)
    df = df[df["product_id"] != ""]
    df["product_name"] = df["product_name"].apply(safe_str)
    df["category"]     = df["category"].apply(normalize_category)
    df["subcategory"]  = df["subcategory"].apply(safe_str)
    df["brand"]        = df["brand"].apply(safe_str)

    # 숫자/가격 안전 변환
    df["price"]            = df["price"].apply(to_int_any)
    df["total_reviews"]    = df["total_reviews"].apply(to_int_any)
    df["positive_reviews"] = df["positive_reviews"].apply(to_int_any)
    df["mixed_reviews"]    = df["mixed_reviews"].apply(to_int_any)
    df["negative_reviews"] = df["negative_reviews"].apply(to_int_any)
    df["avg_rating"]       = df["avg_rating"].apply(to_float_any)
    df["url"]              = df["url"].apply(safe_str)
    df["thumbnail_url"]    = df["thumbnail_url"].apply(safe_str)

    # TIMESTAMP 컬럼 변환 (UTC 기준)
    df["updated_at"] = pd.to_datetime(df["updated_at"], errors="coerce")
    df["updated_at"] = df["updated_at"].fillna(pd.Timestamp.utcnow())

    # 동일 product_id 중 최신만 유지
    df = df.sort_values(by=["product_id","updated_at"]).drop_duplicates(subset=["product_id"], keep="last")
    if df.empty:
        return {"ok": True, "upserted": 0}

    with engine.begin() as conn:
        cur = conn.execute(text("SELECT COUNT(*) FROM product_summary;")).scalar() or 0
        first_summary_upload = (cur == 0)

        if first_summary_upload:
            # 최초 업로드: 모든 필드 신뢰 (숫자 포함)
            rows = df.to_dict(orient="records")
            upsert_all = text("""
                INSERT INTO product_summary (
                    product_id, product_name, category, subcategory, brand,
                    price, total_reviews, positive_reviews, mixed_reviews, negative_reviews, avg_rating,
                    url, thumbnail_url, updated_at
                ) VALUES (
                    :product_id, :product_name, :category, :subcategory, :brand,
                    :price, :total_reviews, :positive_reviews, :mixed_reviews, :negative_reviews, :avg_rating,
                    :url, :thumbnail_url, :updated_at
                )
                ON CONFLICT (product_id) DO UPDATE SET
                    product_name = EXCLUDED.product_name,
                    category     = EXCLUDED.category,
                    subcategory  = EXCLUDED.subcategory,
                    brand        = EXCLUDED.brand,
                    price        = EXCLUDED.price,
                    total_reviews    = EXCLUDED.total_reviews,
                    positive_reviews = EXCLUDED.positive_reviews,
                    mixed_reviews    = EXCLUDED.mixed_reviews,
                    negative_reviews = EXCLUDED.negative_reviews,
                    avg_rating       = EXCLUDED.avg_rating,
                    url           = EXCLUDED.url,
                    thumbnail_url = EXCLUDED.thumbnail_url,
                    updated_at    = EXCLUDED.updated_at;
            """)
            conn.execute(upsert_all, rows)

        else:
            # 이후 업로드: 기존/신규 분리
            exist_ids = set(pd.read_sql(text("SELECT product_id FROM product_summary;"), conn)["product_id"].astype(str))
            df_new = df[~df["product_id"].astype(str).isin(exist_ids)]

            # 신규 상품만 “모든 필드” 채워서 INSERT
            if not df_new.empty:
                rows_new = df_new.to_dict(orient="records")
                insert_full = text("""
                    INSERT INTO product_summary (
                        product_id, product_name, category, subcategory, brand,
                        price, total_reviews, positive_reviews, mixed_reviews, negative_reviews, avg_rating,
                        url, thumbnail_url, updated_at
                    ) VALUES (
                        :product_id, :product_name, :category, :subcategory, :brand,
                        :price, :total_reviews, :positive_reviews, :mixed_reviews, :negative_reviews, :avg_rating,
                        :url, :thumbnail_url, :updated_at
                    )
                    ON CONFLICT (product_id) DO NOTHING;
                """)
                conn.execute(insert_full, rows_new)

        

    return {"ok": True, "upserted": int(len(df))}

def ingest_product_info_df(df: pd.DataFrame):
    """
    product_infomation.csv 적재
    기대 컬럼:
      - product_id
      - product_name
      - brand
      - explanation
      - technical_info
      - management_guidelines
      - URL (대소문자 상관X)
    """
    ensure_tables()

    # 컬럼 이름 표준화 (대소문자/살짝 다른 이름도 대비)
    df = map_columns(df, PRODUCT_INFO_MAP)

    # 타입 정리
    df["product_id"]             = df["product_id"].apply(safe_str)
    df["product_name"]           = df["product_name"].apply(safe_str)
    df["brand"]                  = df["brand"].apply(safe_str)
    df["explanation"]            = df["explanation"].apply(safe_str)
    df["technical_info"]         = df["technical_info"].apply(safe_str)
    df["management_guidelines"]  = df["management_guidelines"].apply(safe_str)
    df["url"]                    = df["url"].apply(safe_str)

    # product_id 없는 행은 버림
    df = df[df["product_id"] != ""]
    if df.empty:
        return {"ok": True, "upserted": 0}

    records = df.to_dict(orient="records")

    insert_sql = text("""
        INSERT INTO product_information (
            product_id, product_name, brand,
            explanation, technical_info, management_guidelines, url, updated_at
        )
        VALUES (
            :product_id, :product_name, :brand,
            :explanation, :technical_info, :management_guidelines, :url, NOW()
        )
        ON CONFLICT (product_id) DO UPDATE SET
            product_name          = EXCLUDED.product_name,
            brand                 = EXCLUDED.brand,
            explanation           = EXCLUDED.explanation,
            technical_info        = EXCLUDED.technical_info,
            management_guidelines = EXCLUDED.management_guidelines,
            url                   = EXCLUDED.url,
            updated_at            = NOW();
    """)

    with engine.begin() as conn:
        conn.execute(insert_sql, records)

    return {"ok": True, "upserted": len(records)}