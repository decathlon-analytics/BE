# core/db.py

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker
import os

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def _ensure_chat_tables_sql(conn):
    """챗봇 테이블 생성 + 마이그레이션"""
    
    # 1. sessions 테이블
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS sessions (
        id UUID PRIMARY KEY,
        started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        last_active TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        expires_at TIMESTAMPTZ NOT NULL,
        last_recommended_ids TEXT DEFAULT ''
    );
    """))
    
    # 2. 기존 테이블에 컬럼 추가 (마이그레이션)
    inspector = inspect(conn)
    if "sessions" in inspector.get_table_names():
        columns = [c["name"] for c in inspector.get_columns("sessions")]
        if "last_recommended_ids" not in columns:
            conn.execute(text("ALTER TABLE sessions ADD COLUMN last_recommended_ids TEXT DEFAULT '';"))
    
    # 3. messages 테이블
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS messages (
        id BIGSERIAL PRIMARY KEY,
        session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
        role VARCHAR(16) NOT NULL,
        content TEXT NOT NULL,
        meta JSONB,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """))
    
    # 4. session_summaries 테이블
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS session_summaries (
        session_id UUID PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
        rolling_summary TEXT,
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    """))
    
    # 5. 인덱스
    conn.execute(text("""
    CREATE INDEX IF NOT EXISTS idx_messages_session_created
    ON messages(session_id, created_at DESC);
    """))

def ensure_tables():
    """모든 테이블 생성 (코어 + 챗봇)"""
    with engine.begin() as conn:
        # 코어 테이블 (reviews, product_summary)
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS reviews (
            review_id VARCHAR(128) PRIMARY KEY,
            product_id VARCHAR(32),
            product_name TEXT,
            category VARCHAR(32),
            subcategory VARCHAR(64),
            brand VARCHAR(64),
            rating NUMERIC(3,1),
            review_text TEXT,
            sentiment VARCHAR(16),
            review_date DATE
        );
        """))
        
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS product_summary (
            product_id VARCHAR(32) PRIMARY KEY,
            product_name TEXT,
            category VARCHAR(32),
            subcategory VARCHAR(64),
            brand VARCHAR(64),
            price INTEGER,
            avg_rating NUMERIC(3,2),
            total_reviews INTEGER,
            positive_reviews INTEGER,
            mixed_reviews INTEGER,
            negative_reviews INTEGER,
            url TEXT,
            thumbnail_url TEXT,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """))
        
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS product_information (
            product_id VARCHAR(32) PRIMARY KEY,
            product_name TEXT,
            brand VARCHAR(64),
            explanation TEXT,
            technical_info TEXT,
            management_guidelines TEXT,
            url TEXT,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        """))
        
        # 챗봇 테이블 + 마이그레이션
        _ensure_chat_tables_sql(conn)

def ensure_summary_table(conn):
    """product_summary만 보장 (ingest.py에서 호출)"""
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS product_summary (
        product_id VARCHAR(32) PRIMARY KEY,
        product_name TEXT,
        category VARCHAR(32),
        subcategory VARCHAR(64),
        brand VARCHAR(64),
        price INTEGER,
        avg_rating NUMERIC(3,2),
        total_reviews INTEGER,
        positive_reviews INTEGER,
        mixed_reviews INTEGER,
        negative_reviews INTEGER,
        url TEXT,
        thumbnail_url TEXT,
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    """))

def ensure_product_information_table(conn):
    """product_information만 보장"""
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS product_information (
        product_id VARCHAR(32) PRIMARY KEY,
        product_name TEXT,
        brand VARCHAR(64),
        explanation TEXT,
        technical_info TEXT,
        management_guidelines TEXT,
        url TEXT,
        updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    """))
