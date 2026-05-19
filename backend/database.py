import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "reviews.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # so I can access columns by name not index
    return conn


def init_db():
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS reviews (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                movie      TEXT NOT NULL,
                review     TEXT NOT NULL,
                sentiment  TEXT NOT NULL,
                confidence REAL NOT NULL,
                created_at TEXT NOT NULL
            )
        """)


def save_review(movie: str, review: str, sentiment: str, confidence: float) -> int:
    with get_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO reviews (movie, review, sentiment, confidence, created_at) VALUES (?, ?, ?, ?, ?)",
            (movie, review, sentiment, confidence, datetime.utcnow().isoformat(timespec="seconds")),
        )
        return cursor.lastrowid


def get_reviews(limit: int = 50) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM reviews ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(row) for row in rows]
