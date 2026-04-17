"""
db_manager.py — SafeGuard AI Database Layer (SQLite + Supabase Postgres)
========================================================================
Automatically uses Postgres when DATABASE_URL env variable is set
(Railway + Supabase), and falls back to SQLite for local development.
"""

import os
import datetime
from pathlib import Path

# ── Detect environment ──────────────────────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")
USE_POSTGRES = bool(DATABASE_URL)

if USE_POSTGRES:
    # Railway production: use psycopg2 with Supabase Postgres
    try:
        import psycopg2
        import psycopg2.extras
        _POSTGRES_AVAILABLE = True
    except ImportError:
        _POSTGRES_AVAILABLE = False
        USE_POSTGRES = False

if not USE_POSTGRES:
    # Local development: use SQLite
    import sqlite3
    DATABASE_PATH = Path(r"E:\4TH YEAR PROJECT\outputs\safeguard.db")
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)


# ── Connection helpers ──────────────────────────────────────────────────────
def _pg_conn():
    return psycopg2.connect(DATABASE_URL)


def _sqlite_conn():
    return sqlite3.connect(str(DATABASE_PATH), check_same_thread=False)


def _conn():
    return _pg_conn() if USE_POSTGRES else _sqlite_conn()


# ── Schema init ─────────────────────────────────────────────────────────────
def init_db():
    """Create tables if they don't exist (idempotent, runs on import)."""
    if USE_POSTGRES:
        with _conn() as conn:
            cur = conn.cursor()
            cur.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id          SERIAL PRIMARY KEY,
                    ts          TEXT,
                    session     TEXT,
                    source      TEXT,
                    frame       INTEGER,
                    alert_type  TEXT,
                    tool_name   TEXT,
                    timer_s     REAL,
                    missing_ppe TEXT
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id          SERIAL PRIMARY KEY,
                    session     TEXT UNIQUE,
                    started     TEXT,
                    source      TEXT,
                    frames      INTEGER DEFAULT 0,
                    alerts      INTEGER DEFAULT 0,
                    compliance  REAL DEFAULT 100.0
                )
            """)
            conn.commit()
    else:
        with _sqlite_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT, session TEXT, source TEXT, frame INTEGER,
                    alert_type TEXT, tool_name TEXT, timer_s REAL, missing_ppe TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session TEXT UNIQUE, started TEXT, source TEXT,
                    frames INTEGER DEFAULT 0, alerts INTEGER DEFAULT 0, compliance REAL DEFAULT 100.0
                )
            """)
            conn.commit()


def log_alert(session: str, source: str, frame: int,
              alert_type: str, tool_name: str = "",
              timer_s: float = 0.0, missing_ppe: list = None):
    """Insert one alert event."""
    # Guard against mutable default argument mutation across calls
    if missing_ppe is None:
        missing_ppe = []
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    try:
        if USE_POSTGRES:
            with _conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO alerts (ts,session,source,frame,alert_type,tool_name,timer_s,missing_ppe) "
                    "VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
                    (ts, session, source, frame, alert_type, tool_name, timer_s, str(missing_ppe))
                )
                conn.commit()
        else:
            with _sqlite_conn() as conn:
                conn.execute(
                    "INSERT INTO alerts (ts,session,source,frame,alert_type,tool_name,timer_s,missing_ppe) "
                    "VALUES (?,?,?,?,?,?,?,?)",
                    (ts, session, source, frame, alert_type, tool_name, timer_s, str(missing_ppe))
                )
                conn.commit()
    except Exception as e:
        print(f"DB warning (log_alert): {e}")


def upsert_session(session: str, source: str, frames: int, alerts: int, compliance: float):
    """Insert or update a session record."""
    started = datetime.datetime.now().isoformat(timespec="seconds")
    try:
        if USE_POSTGRES:
            with _conn() as conn:
                cur = conn.cursor()
                cur.execute(
                    """INSERT INTO sessions (session, started, source, frames, alerts, compliance)
                       VALUES (%s,%s,%s,%s,%s,%s)
                       ON CONFLICT(session) DO UPDATE SET
                         frames=EXCLUDED.frames, alerts=EXCLUDED.alerts, compliance=EXCLUDED.compliance""",
                    (session, started, source, frames, alerts, compliance)
                )
                conn.commit()
        else:
            with _sqlite_conn() as conn:
                conn.execute(
                    """INSERT INTO sessions (session,started,source,frames,alerts,compliance)
                       VALUES (?,?,?,?,?,?)
                       ON CONFLICT(session) DO UPDATE SET
                         frames=excluded.frames, alerts=excluded.alerts, compliance=excluded.compliance""",
                    (session, started, source, frames, alerts, compliance)
                )
                conn.commit()
    except Exception as e:
        print(f"DB warning (upsert_session): {e}")


def get_all_alerts() -> list:
    """Return all alert records, newest first."""
    try:
        if USE_POSTGRES:
            with _conn() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cur.execute("SELECT * FROM alerts ORDER BY id DESC")
                return [dict(r) for r in cur.fetchall()]
        else:
            with _sqlite_conn() as conn:
                rows = conn.execute("SELECT * FROM alerts ORDER BY id DESC").fetchall()
            cols = ["id","ts","session","source","frame","type","tool","timer_s","missing_ppe"]
            return [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        print(f"DB warning (get_all_alerts): {e}")
        return []


def get_all_sessions() -> list:
    """Return all session records, newest first."""
    try:
        if USE_POSTGRES:
            with _conn() as conn:
                cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
                cur.execute("SELECT * FROM sessions ORDER BY id DESC")
                return [dict(r) for r in cur.fetchall()]
        else:
            with _sqlite_conn() as conn:
                rows = conn.execute("SELECT * FROM sessions ORDER BY id DESC").fetchall()
            cols = ["id","session","started","source","frames","alerts","compliance"]
            return [dict(zip(cols, r)) for r in rows]
    except Exception as e:
        print(f"DB warning (get_all_sessions): {e}")
        return []


def get_alert_counts_by_hour() -> list:
    """Return (hour, count) tuples grouped by hour for analytics charts."""
    try:
        if USE_POSTGRES:
            with _conn() as conn:
                cur = conn.cursor()
                cur.execute("""
                    SELECT SUBSTRING(ts FROM 12 FOR 2) AS hour, COUNT(*) AS count
                    FROM alerts GROUP BY hour ORDER BY hour
                """)
                return cur.fetchall()
        else:
            with _sqlite_conn() as conn:
                return conn.execute("""
                    SELECT substr(ts,12,2) AS hour, COUNT(*) AS count
                    FROM alerts GROUP BY hour ORDER BY hour
                """).fetchall()
    except Exception as e:
        print(f"DB warning (get_alert_counts_by_hour): {e}")
        return []


# Initialise the database on module import
init_db()
