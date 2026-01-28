from __future__ import annotations

import os
import time
import hashlib
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple, Callable, Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, IntegrityError

# =========================================================
# 基本設定
# =========================================================
APP_TITLE = "収益ナビ"
DB_PATH = "data.db"

# 「前年の記録が打てない」対策：日付入力の最小値を明示（必要ならもっと昔でもOK）
MIN_DAY = date(1970, 1, 1)

ENGINE = create_engine(
    f"sqlite:///{DB_PATH}",
    future=True,
    connect_args={"check_same_thread": False, "timeout": 30},
)

# 表示名（日本語）
CURRENCY_NAME_JA = {
    "JPY": "円",
    "USD": "米ドル",
    "EUR": "ユーロ",
    "GBP": "英ポンド",
    "AUD": "豪ドル",
    "CAD": "カナダドル",
    "CHF": "スイスフラン",
    "CNY": "人民元",
    "KRW": "韓国ウォン",
    "HKD": "香港ドル",
    "SGD": "シンガポールドル",
}

# 入力セレクト（円を先頭）
CURRENCY_OPTIONS = ["JPY", "USD", "EUR", "GBP", "AUD", "CAD", "CHF", "CNY", "KRW", "HKD", "SGD"]

# 為替設定の優先順（USD/EUR/JPYを先頭）
CURRENCY_ORDER = ["USD", "EUR", "JPY", "GBP", "AUD", "CAD", "CHF", "CNY", "KRW", "HKD", "SGD"]

# 収益：会社/給料を先頭
DEFAULT_PLATFORMS = ["会社", "YouTube", "TikTok", "Instagram", "X", "ブログ", "クライアント", "その他"]
DEFAULT_EARN_CATEGORIES = ["給料", "広告", "案件", "アフィリエイト", "商品販売", "投資", "その他"]

# 経費カテゴリ（その他は自由入力対応）
DEFAULT_EXP_CATEGORIES = ["サブスク", "機材", "広告費", "交通", "外注", "通信", "教育", "税金", "その他"]

# =========================================================
# OpenAI（環境変数 or Streamlit secrets 両対応）
# - 「キーがなくてもユーザーは1回無料」を成立させるために、
#   サービスキーは env / secrets のどちらでも拾えるようにする。
# =========================================================
def _get_secret(name: str, default: str = "") -> str:
    try:
        v = st.secrets.get(name, default)
        return (str(v) if v is not None else default).strip()
    except Exception:
        return default


def get_service_openai_key() -> str:
    # 優先：Streamlit secrets → 環境変数
    k = _get_secret("OPENAI_API_KEY", "")
    if k:
        return k
    return os.getenv("OPENAI_API_KEY", "").strip()


def get_openai_base_url() -> str:
    # secrets か env。未設定なら公式の v1
    v = _get_secret("OPENAI_BASE_URL", "") or os.getenv("OPENAI_BASE_URL", "")
    return (v.strip() or "https://api.openai.com/v1").strip()


def get_openai_model() -> str:
    # secrets か env。未設定なら軽量モデル
    v = _get_secret("OPENAI_MODEL", "") or os.getenv("OPENAI_MODEL", "")
    return (v.strip() or "gpt-5-mini").strip()


# =========================================================
# ユーティリティ
# =========================================================
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def today_date() -> date:
    return datetime.now().date()


def month_range(d: date) -> Tuple[date, date]:
    start = d.replace(day=1)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1, day=1) - timedelta(days=1)
    else:
        end = start.replace(month=start.month + 1, day=1) - timedelta(days=1)
    return start, end


def yen(x) -> str:
    # 小数点問題：表示は常に整数円へ丸める（内部はfloatでもOK）
    try:
        v = 0.0 if x is None else float(x)
    except Exception:
        v = 0.0
    return f"¥{int(round(v)):,}"


def currency_ja(code: str) -> str:
    code = (code or "JPY").upper()
    return CURRENCY_NAME_JA.get(code, code)


def pin_hash(username: str, pin: str) -> str:
    salt = f"{username}::revenue_navi"
    raw = (salt + "::" + pin).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# =========================================================
# 矢印・色（赤/緑）を「すべての場所で」確実に統一するための関数
# =========================================================
def delta_style(delta: float) -> Tuple[str, str]:
    """
    return: (arrow, color_hex)
    - プラス：緑、マイナス：赤
    """
    if delta > 0:
        return "↑", "#2e7d32"
    if delta < 0:
        return "↓", "#c62828"
    return "—", "#666666"


def _pct_text(delta: float, base: float) -> str:
    if base == 0:
        return "(±0%)"
    rate = (delta / abs(base)) * 100.0
    return f"({int(rate):+d}%)"


def html_delta_badge(delta: float, base: float, big: bool = False) -> str:
    """
    Streamlit標準の delta 色/矢印が場所によってブレる問題を避けるため、
    HTMLで100%制御する。
    """
    arrow, color = delta_style(delta)
    pct_txt = _pct_text(delta, base)

    size = "16px" if big else "13px"
    weight = "800" if big else "700"
    return (
        f"<span style='color:{color}; font-weight:{weight}; font-size:{size};'>"
        f"{arrow} {yen(delta)} {pct_txt}"
        f"</span>"
    )


# =========================================================
# SQLiteロック対策（PRAGMA + リトライ）
# =========================================================
def apply_sqlite_pragmas(conn):
    conn.execute(text("PRAGMA journal_mode=WAL"))
    conn.execute(text("PRAGMA synchronous=NORMAL"))
    conn.execute(text("PRAGMA busy_timeout=5000"))


def run_with_retry(fn: Callable[[], Any], tries: int = 8, base_sleep: float = 0.12):
    last_err = None
    for i in range(tries):
        try:
            return fn()
        except OperationalError as e:
            last_err = e
            msg = str(e).lower()
            if ("database is locked" in msg) or ("database locked" in msg) or ("locked" in msg):
                time.sleep(base_sleep * (i + 1))
                continue
            raise
    raise last_err


# =========================================================
# DBユーティリティ
# =========================================================
def table_columns(conn, table: str) -> List[str]:
    rows = conn.execute(text(f"PRAGMA table_info({table})")).fetchall()
    return [r[1] for r in rows]


def add_column_if_missing(conn, table: str, col: str, col_type: str, default_sql: Optional[str] = None):
    cols = table_columns(conn, table)
    if col in cols:
        return
    if default_sql is None:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"))
    else:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type} DEFAULT {default_sql}"))


def pick_date_column(conn, table: str) -> str:
    cols = table_columns(conn, table)
    if "day" in cols:
        return "day"
    if "date" in cols:
        return "date"
    raise RuntimeError(f"{table} テーブルに日付列（day/date）が見つかりません。")


def ensure_day_date_compat(conn, table: str):
    cols = table_columns(conn, table)
    if "day" not in cols:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN day TEXT"))
    if "date" not in cols:
        conn.execute(text(f"ALTER TABLE {table} ADD COLUMN date TEXT"))
    conn.execute(text(f"UPDATE {table} SET day = COALESCE(day, date)"))
    conn.execute(text(f"UPDATE {table} SET date = COALESCE(date, day)"))


def pick_first_existing_column(conn, table: str, candidates: List[str]) -> Optional[str]:
    cols = set(table_columns(conn, table))
    for c in candidates:
        if c in cols:
            return c
    return None


# =========================================================
# DB 初期化＆マイグレーション
# - ai_free_used: 「キーなし無料1回」をユーザーごとに管理
# =========================================================
def init_db_and_migrate():
    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)

            # users
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                pin_hash TEXT,
                password_hash TEXT,
                created_at TEXT NOT NULL
            )
            """))
            add_column_if_missing(conn, "users", "pin_hash", "TEXT")
            add_column_if_missing(conn, "users", "password_hash", "TEXT")
            add_column_if_missing(conn, "users", "created_at", "TEXT", "'1970-01-01 00:00:00'")
            conn.execute(text("UPDATE users SET pin_hash = COALESCE(pin_hash, password_hash)"))
            conn.execute(text("UPDATE users SET password_hash = COALESCE(password_hash, pin_hash)"))

            # user_settings
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS user_settings (
                user_id INTEGER PRIMARY KEY,
                monthly_goal_jpy REAL NOT NULL DEFAULT 100000,
                fixed_cost_jpy REAL NOT NULL DEFAULT 0,
                base_currency TEXT NOT NULL DEFAULT 'JPY',
                ai_free_used INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """))
            add_column_if_missing(conn, "user_settings", "monthly_goal_jpy", "REAL", "100000")
            add_column_if_missing(conn, "user_settings", "fixed_cost_jpy", "REAL", "0")
            add_column_if_missing(conn, "user_settings", "base_currency", "TEXT", "'JPY'")
            add_column_if_missing(conn, "user_settings", "ai_free_used", "INTEGER", "0")
            add_column_if_missing(conn, "user_settings", "created_at", "TEXT", "'1970-01-01 00:00:00'")
            add_column_if_missing(conn, "user_settings", "updated_at", "TEXT", "'1970-01-01 00:00:00'")

            # fx_rates
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS fx_rates (
                currency TEXT PRIMARY KEY,
                rate_to_jpy REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
            """))
            add_column_if_missing(conn, "fx_rates", "currency", "TEXT")
            add_column_if_missing(conn, "fx_rates", "rate_to_jpy", "REAL", "1.0")
            add_column_if_missing(conn, "fx_rates", "updated_at", "TEXT", "'1970-01-01 00:00:00'")

            # earnings
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS earnings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                day TEXT,
                date TEXT,
                platform TEXT NOT NULL,
                category TEXT NOT NULL,
                currency TEXT NOT NULL,
                amount REAL NOT NULL,
                jpy_amount REAL NOT NULL,
                memo TEXT,
                created_at TEXT NOT NULL
            )
            """))
            add_column_if_missing(conn, "earnings", "jpy_amount", "REAL", "0")
            add_column_if_missing(conn, "earnings", "memo", "TEXT", "''")
            add_column_if_missing(conn, "earnings", "created_at", "TEXT", "'1970-01-01 00:00:00'")
            ensure_day_date_compat(conn, "earnings")

            # expenses
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS expenses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                day TEXT,
                date TEXT,
                vendor TEXT NOT NULL,
                category TEXT NOT NULL,
                currency TEXT NOT NULL,
                amount REAL NOT NULL,
                jpy_amount REAL NOT NULL,
                memo TEXT,
                created_at TEXT NOT NULL
            )
            """))
            add_column_if_missing(conn, "expenses", "jpy_amount", "REAL", "0")
            add_column_if_missing(conn, "expenses", "memo", "TEXT", "''")
            add_column_if_missing(conn, "expenses", "created_at", "TEXT", "'1970-01-01 00:00:00'")
            ensure_day_date_compat(conn, "expenses")
            add_column_if_missing(conn, "expenses", "vendor", "TEXT", "''")

            # vendor別名→vendorへ寄せる
            cols = table_columns(conn, "expenses")
            if "vendor" in cols:
                if "payee" in cols:
                    conn.execute(text("UPDATE expenses SET vendor = COALESCE(NULLIF(vendor,''), payee)"))
                if "shop" in cols:
                    conn.execute(text("UPDATE expenses SET vendor = COALESCE(NULLIF(vendor,''), shop)"))
                if "merchant" in cols:
                    conn.execute(text("UPDATE expenses SET vendor = COALESCE(NULLIF(vendor,''), merchant)"))
                if "支払先" in cols:
                    conn.execute(text('UPDATE expenses SET vendor = COALESCE(NULLIF(vendor,""), "支払先")'))

            # assets_snapshots（資産）
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS assets_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                day TEXT NOT NULL,
                cash_jpy REAL NOT NULL DEFAULT 0,
                stocks_jpy REAL NOT NULL DEFAULT 0,
                other_jpy REAL NOT NULL DEFAULT 0,
                other_name TEXT,
                total_jpy REAL NOT NULL DEFAULT 0,
                memo TEXT,
                created_at TEXT NOT NULL
            )
            """))
            add_column_if_missing(conn, "assets_snapshots", "cash_jpy", "REAL", "0")
            add_column_if_missing(conn, "assets_snapshots", "stocks_jpy", "REAL", "0")
            add_column_if_missing(conn, "assets_snapshots", "other_jpy", "REAL", "0")
            add_column_if_missing(conn, "assets_snapshots", "other_name", "TEXT", "''")
            add_column_if_missing(conn, "assets_snapshots", "total_jpy", "REAL", "0")
            add_column_if_missing(conn, "assets_snapshots", "memo", "TEXT", "''")
            add_column_if_missing(conn, "assets_snapshots", "created_at", "TEXT", "'1970-01-01 00:00:00'")

            # 初期FX
            defaults = {"JPY": 1.0, "USD": 150.0, "EUR": 165.0, "AUD": 100.0}
            for cur, rate in defaults.items():
                conn.execute(text("""
                INSERT INTO fx_rates(currency, rate_to_jpy, updated_at)
                VALUES(:c, :r, :u)
                ON CONFLICT(currency) DO NOTHING
                """), {"c": cur, "r": float(rate), "u": now_str()})

    run_with_retry(_do)


# =========================================================
# ユーザー / 設定
# =========================================================
def get_user_by_username(username: str) -> Optional[dict]:
    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            row = conn.execute(
                text("SELECT id, username, pin_hash, password_hash FROM users WHERE username = :u"),
                {"u": username.strip()},
            ).fetchone()
            if not row:
                return None
            ph = row[2] or row[3]
            return {"id": row[0], "username": row[1], "pin_hash": ph}

    return run_with_retry(_do)


def ensure_user_defaults(user_id: int):
    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            row = conn.execute(
                text("SELECT user_id FROM user_settings WHERE user_id=:uid"),
                {"uid": user_id},
            ).fetchone()
            if row:
                conn.execute(text("""
                UPDATE user_settings
                SET created_at = COALESCE(created_at, :c),
                    updated_at = COALESCE(updated_at, :u),
                    ai_free_used = COALESCE(ai_free_used, 0)
                WHERE user_id=:uid
                """), {"uid": user_id, "c": now_str(), "u": now_str()})
            else:
                conn.execute(text("""
                INSERT INTO user_settings(user_id, monthly_goal_jpy, fixed_cost_jpy, base_currency, ai_free_used, created_at, updated_at)
                VALUES(:uid, 100000, 0, 'JPY', 0, :c, :u)
                """), {"uid": user_id, "c": now_str(), "u": now_str()})

    run_with_retry(_do)


def create_user(username: str, pin: str) -> int:
    username = username.strip()
    ph = pin_hash(username, pin)

    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            conn.execute(
                text("""
                INSERT INTO users(username, pin_hash, password_hash, created_at)
                VALUES(:u,:p,:p,:c)
                """),
                {"u": username, "p": ph, "c": now_str()},
            )
            uid = conn.execute(text("SELECT id FROM users WHERE username=:u"), {"u": username}).fetchone()[0]
            return int(uid)

    uid = run_with_retry(_do)
    ensure_user_defaults(uid)
    return int(uid)


def get_user_settings(user_id: int) -> dict:
    ensure_user_defaults(user_id)

    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            row = conn.execute(text("""
            SELECT monthly_goal_jpy, fixed_cost_jpy, base_currency, ai_free_used
            FROM user_settings
            WHERE user_id=:uid
            """), {"uid": user_id}).fetchone()
            return row

    row = run_with_retry(_do)
    return {
        "monthly_goal_jpy": float(row[0]) if row else 100000.0,
        "fixed_cost_jpy": float(row[1]) if row else 0.0,
        "base_currency": (row[2] if row else "JPY") or "JPY",
        "ai_free_used": int(row[3]) if row else 0,
    }


def save_user_settings(user_id: int, monthly_goal_jpy: float, fixed_cost_jpy: float, base_currency: str):
    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            conn.execute(text("""
            UPDATE user_settings
            SET monthly_goal_jpy=:g,
                fixed_cost_jpy=:f,
                base_currency=:b,
                updated_at=:u
            WHERE user_id=:uid
            """), {
                "uid": user_id,
                "g": float(monthly_goal_jpy),
                "f": float(fixed_cost_jpy),
                "b": (base_currency or "JPY").strip().upper(),
                "u": now_str(),
            })

    run_with_retry(_do)


def mark_ai_free_used(user_id: int):
    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            conn.execute(text("""
            UPDATE user_settings
            SET ai_free_used=1, updated_at=:u
            WHERE user_id=:uid
            """), {"uid": user_id, "u": now_str()})

    run_with_retry(_do)


# =========================================================
# AI（Responses API）呼び出し
# - 1回無料（キー無し）＋自由質問モード（チャット）どちらも同じ仕組みで動かす
# =========================================================
def _responses_api_call(api_key: str, messages: List[dict]) -> str:
    """
    OpenAI Responses API を requests で叩く（依存最小化）
    """
    import requests

    base_url = get_openai_base_url()
    model = get_openai_model()

    url = f"{base_url}/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Responses API は input にメッセージ配列を渡せる
    payload = {
        "model": model,
        "input": messages,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI API エラー {r.status_code}: {r.text}")

    data = r.json()

    # テキスト抽出（複数出力に対応）
    out = data.get("output", [])
    texts: List[str] = []
    for item in out:
        for c in (item.get("content", []) or []):
            t = c.get("text")
            if t:
                texts.append(t)

    if not texts:
        return str(data)
    return "\n".join(texts).strip()


def can_use_service_ai(user_id: int) -> Tuple[bool, str]:
    """
    サービスキー（運営側のキー）でAIを使えるか？
    ・ユーザーごとに「1回だけ」無料（ai_free_usedで管理）
    """
    settings = get_user_settings(user_id)
    free_used = int(settings.get("ai_free_used", 0)) == 1

    service_key = get_service_openai_key()
    if not service_key:
        # ここは「無料1回」の根幹なので、原因が分かる文言にする（英語なし）
        return False, "運営側のOpenAIキーが未設定です（管理者設定が必要です）。"

    if free_used:
        return False, "無料（1回）は使用済みです。続ける場合はサイドバーでご自身のOpenAIキーを入力してください。"

    return True, ""


def get_effective_api_key(user_id: int, user_supplied_key: str) -> Tuple[Optional[str], str, bool]:
    """
    返り値: (api_key, 状態メッセージ, サービスキー使用フラグ)
    ・ユーザーがキーを入れたらそれを最優先（回数制限なし想定）
    ・なければ運営側キーで「1回だけ」無料
    """
    user_supplied_key = (user_supplied_key or "").strip()
    if user_supplied_key:
        return user_supplied_key, "ok", False

    ok, reason = can_use_service_ai(user_id)
    if not ok:
        return None, reason, False

    return get_service_openai_key(), "ok", True


def run_ai_with_limits(user_id: int, user_supplied_key: str, messages: List[dict]) -> Tuple[Optional[str], str]:
    """
    無料枠（1回）管理を含めてAI実行
    """
    api_key, status, using_service = get_effective_api_key(user_id, user_supplied_key)
    if not api_key:
        return None, status

    try:
        txt = _responses_api_call(api_key, messages)
        if using_service:
            mark_ai_free_used(user_id)
        return txt, "ok"
    except Exception as e:
        return None, f"AIの実行に失敗しました：{e}"
# =========================================================
# 為替
# =========================================================
def get_fx_rates() -> Dict[str, float]:
    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            rows = conn.execute(text("SELECT currency, rate_to_jpy FROM fx_rates")).fetchall()
            return {r[0]: float(r[1]) for r in rows}

    return run_with_retry(_do)


def upsert_fx_rate(currency: str, rate_to_jpy: float):
    currency = (currency or "JPY").strip().upper()

    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            conn.execute(text("""
            INSERT INTO fx_rates(currency, rate_to_jpy, updated_at)
            VALUES(:c, :r, :u)
            ON CONFLICT(currency) DO UPDATE SET
                rate_to_jpy=excluded.rate_to_jpy,
                updated_at=excluded.updated_at
            """), {"c": currency, "r": float(rate_to_jpy), "u": now_str()})

    run_with_retry(_do)


def compute_jpy(amount: float, currency: str, fx: Dict[str, float]) -> float:
    currency = (currency or "JPY").upper()
    rate = fx.get(currency, 1.0)
    return float(amount) * float(rate)


# =========================================================
# 収益・経費 CRUD
# =========================================================
def insert_earning(user_id: int, day_: date, platform: str, category: str, currency: str, amount: float, memo: str):
    fx = get_fx_rates()
    jpy_amount = compute_jpy(amount, currency, fx)

    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            cols = table_columns(conn, "earnings")

            insert_cols = ["user_id", "platform", "category", "currency", "amount", "jpy_amount", "memo", "created_at"]
            params = {
                "uid": int(user_id),
                "p": (platform or "").strip() or "その他",
                "cat": (category or "").strip() or "その他",
                "cur": (currency or "JPY").upper(),
                "amt": float(amount),
                "jpy": float(jpy_amount),
                "m": (memo or "").strip(),
                "c": now_str(),
                "d": day_.isoformat(),
            }
            if "day" in cols:
                insert_cols.insert(1, "day")
            if "date" in cols:
                insert_cols.insert(1, "date")

            values_map = {
                "user_id": ":uid",
                "date": ":d",
                "day": ":d",
                "platform": ":p",
                "category": ":cat",
                "currency": ":cur",
                "amount": ":amt",
                "jpy_amount": ":jpy",
                "memo": ":m",
                "created_at": ":c",
            }

            conn.execute(
                text(
                    f"INSERT INTO earnings({', '.join(insert_cols)}) "
                    f"VALUES({', '.join(values_map[c] for c in insert_cols)})"
                ),
                params,
            )

    run_with_retry(_do)


def update_earning(user_id: int, earning_id: int, day_: date, platform: str, category: str, currency: str, amount: float, memo: str):
    fx = get_fx_rates()
    jpy_amount = compute_jpy(amount, currency, fx)

    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            dc = pick_date_column(conn, "earnings")
            conn.execute(text(f"""
            UPDATE earnings
            SET {dc}=:d,
                platform=:p,
                category=:cat,
                currency=:cur,
                amount=:amt,
                jpy_amount=:jpy,
                memo=:m
            WHERE id=:id AND user_id=:uid
            """), {
                "uid": int(user_id),
                "id": int(earning_id),
                "d": day_.isoformat(),
                "p": (platform or "").strip() or "その他",
                "cat": (category or "").strip() or "その他",
                "cur": (currency or "JPY").upper(),
                "amt": float(amount),
                "jpy": float(jpy_amount),
                "m": (memo or "").strip(),
            })

    run_with_retry(_do)


def delete_earning(user_id: int, earning_id: int):
    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            conn.execute(
                text("DELETE FROM earnings WHERE id=:id AND user_id=:uid"),
                {"id": int(earning_id), "uid": int(user_id)},
            )

    run_with_retry(_do)


def insert_expense(user_id: int, day_: date, vendor: str, category: str, currency: str, amount: float, memo: str):
    fx = get_fx_rates()
    jpy_amount = compute_jpy(amount, currency, fx)

    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            cols = table_columns(conn, "expenses")

            insert_cols = ["user_id", "vendor", "category", "currency", "amount", "jpy_amount", "memo", "created_at"]
            params = {
                "uid": int(user_id),
                "v": (vendor or "").strip() or "未入力",
                "cat": (category or "").strip() or "その他",
                "cur": (currency or "JPY").upper(),
                "amt": float(amount),
                "jpy": float(jpy_amount),
                "m": (memo or "").strip(),
                "c": now_str(),
                "d": day_.isoformat(),
            }
            if "day" in cols:
                insert_cols.insert(1, "day")
            if "date" in cols:
                insert_cols.insert(1, "date")

            values_map = {
                "user_id": ":uid",
                "date": ":d",
                "day": ":d",
                "vendor": ":v",
                "category": ":cat",
                "currency": ":cur",
                "amount": ":amt",
                "jpy_amount": ":jpy",
                "memo": ":m",
                "created_at": ":c",
            }

            conn.execute(
                text(
                    f"INSERT INTO expenses({', '.join(insert_cols)}) "
                    f"VALUES({', '.join(values_map[c] for c in insert_cols)})"
                ),
                params,
            )

    run_with_retry(_do)


def update_expense(user_id: int, expense_id: int, day_: date, vendor: str, category: str, currency: str, amount: float, memo: str):
    fx = get_fx_rates()
    jpy_amount = compute_jpy(amount, currency, fx)

    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            dc = pick_date_column(conn, "expenses")
            vcol = pick_first_existing_column(conn, "expenses", ["vendor", "payee", "shop", "merchant", "支払先"]) or "vendor"
            conn.execute(text(f"""
            UPDATE expenses
            SET {dc}=:d,
                {vcol}=:v,
                category=:cat,
                currency=:cur,
                amount=:amt,
                jpy_amount=:jpy,
                memo=:m
            WHERE id=:id AND user_id=:uid
            """), {
                "uid": int(user_id),
                "id": int(expense_id),
                "d": day_.isoformat(),
                "v": (vendor or "").strip() or "未入力",
                "cat": (category or "").strip() or "その他",
                "cur": (currency or "JPY").upper(),
                "amt": float(amount),
                "jpy": float(jpy_amount),
                "m": (memo or "").strip(),
            })

    run_with_retry(_do)


def delete_expense(user_id: int, expense_id: int):
    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            conn.execute(
                text("DELETE FROM expenses WHERE id=:id AND user_id=:uid"),
                {"id": int(expense_id), "uid": int(user_id)},
            )

    run_with_retry(_do)


def load_earnings(user_id: int, start: date, end: date) -> pd.DataFrame:
    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            dc = pick_date_column(conn, "earnings")
            rows = conn.execute(text(f"""
            SELECT id, {dc} AS d, platform, category, currency, amount, jpy_amount, memo
            FROM earnings
            WHERE user_id=:uid AND {dc} >= :s AND {dc} <= :e
            ORDER BY {dc} ASC, id ASC
            """), {"uid": int(user_id), "s": start.isoformat(), "e": end.isoformat()}).fetchall()
            return rows

    rows = run_with_retry(_do)
    df = pd.DataFrame(rows, columns=["ID", "日付", "プラットフォーム", "カテゴリ", "通貨", "金額", "円換算", "メモ"])
    if not df.empty:
        df["日付"] = df["日付"].astype(str)
        df["通貨コード"] = df["通貨"].astype(str).str.upper()
        df["通貨"] = df["通貨コード"].map(currency_ja)
        # 小数点問題：画面表示は整数に統一
        df["金額"] = df["金額"].map(lambda x: int(round(float(x))) if pd.notna(x) else 0)
        df["円換算"] = df["円換算"].map(lambda x: int(round(float(x))) if pd.notna(x) else 0)
        df["メモ"] = df["メモ"].fillna("").astype(str)
    return df


def load_expenses(user_id: int, start: date, end: date) -> pd.DataFrame:
    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            dc = pick_date_column(conn, "expenses")
            vcol = pick_first_existing_column(conn, "expenses", ["vendor", "payee", "shop", "merchant", "支払先"]) or "vendor"
            rows = conn.execute(text(f"""
            SELECT id, {dc} AS d, {vcol} AS vendor, category, currency, amount, jpy_amount, memo
            FROM expenses
            WHERE user_id=:uid AND {dc} >= :s AND {dc} <= :e
            ORDER BY {dc} ASC, id ASC
            """), {"uid": int(user_id), "s": start.isoformat(), "e": end.isoformat()}).fetchall()
            return rows

    rows = run_with_retry(_do)
    df = pd.DataFrame(rows, columns=["ID", "日付", "支払先", "カテゴリ", "通貨", "金額", "円換算", "メモ"])
    if not df.empty:
        df["日付"] = df["日付"].astype(str)
        df["通貨コード"] = df["通貨"].astype(str).str.upper()
        df["通貨"] = df["通貨コード"].map(currency_ja)
        df["金額"] = df["金額"].map(lambda x: int(round(float(x))) if pd.notna(x) else 0)
        df["円換算"] = df["円換算"].map(lambda x: int(round(float(x))) if pd.notna(x) else 0)
        df["メモ"] = df["メモ"].fillna("").astype(str)
        df["支払先"] = df["支払先"].fillna("").astype(str)
    return df


# =========================================================
# 資産 CRUD
# =========================================================
def upsert_assets_snapshot(user_id: int, day_: date, cash_jpy: float, stocks_jpy: float, other_jpy: float, other_name: str, memo: str):
    total = float(cash_jpy) + float(stocks_jpy) + float(other_jpy)

    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            conn.execute(text("""
            INSERT INTO assets_snapshots(user_id, day, cash_jpy, stocks_jpy, other_jpy, other_name, total_jpy, memo, created_at)
            VALUES(:uid, :d, :c, :s, :o, :on, :t, :m, :ca)
            """), {
                "uid": int(user_id),
                "d": day_.isoformat(),
                "c": float(cash_jpy),
                "s": float(stocks_jpy),
                "o": float(other_jpy),
                "on": (other_name or "").strip(),
                "t": float(total),
                "m": (memo or "").strip(),
                "ca": now_str(),
            })

    run_with_retry(_do)


def update_assets_snapshot(user_id: int, snap_id: int, day_: date, cash_jpy: float, stocks_jpy: float, other_jpy: float, other_name: str, memo: str):
    total = float(cash_jpy) + float(stocks_jpy) + float(other_jpy)

    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            conn.execute(text("""
            UPDATE assets_snapshots
            SET day=:d,
                cash_jpy=:c,
                stocks_jpy=:s,
                other_jpy=:o,
                other_name=:on,
                total_jpy=:t,
                memo=:m
            WHERE id=:id AND user_id=:uid
            """), {
                "uid": int(user_id),
                "id": int(snap_id),
                "d": day_.isoformat(),
                "c": float(cash_jpy),
                "s": float(stocks_jpy),
                "o": float(other_jpy),
                "on": (other_name or "").strip(),
                "t": float(total),
                "m": (memo or "").strip(),
            })

    run_with_retry(_do)


def delete_assets_snapshot(user_id: int, snap_id: int):
    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            conn.execute(
                text("DELETE FROM assets_snapshots WHERE id=:id AND user_id=:uid"),
                {"id": int(snap_id), "uid": int(user_id)},
            )

    run_with_retry(_do)


def load_latest_assets(user_id: int) -> Optional[dict]:
    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            row = conn.execute(text("""
            SELECT id, day, cash_jpy, stocks_jpy, other_jpy, other_name, total_jpy, memo
            FROM assets_snapshots
            WHERE user_id=:uid
            ORDER BY day DESC, id DESC
            LIMIT 1
            """), {"uid": int(user_id)}).fetchone()
            return row

    r = run_with_retry(_do)
    if not r:
        return None
    return {
        "id": int(r[0]),
        "day": str(r[1]),
        "cash_jpy": float(r[2]),
        "stocks_jpy": float(r[3]),
        "other_jpy": float(r[4]),
        "other_name": str(r[5] or ""),
        "total_jpy": float(r[6]),
        "memo": str(r[7] or ""),
    }


def load_previous_assets(user_id: int) -> Optional[dict]:
    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            rows = conn.execute(text("""
            SELECT id, day, cash_jpy, stocks_jpy, other_jpy, other_name, total_jpy, memo
            FROM assets_snapshots
            WHERE user_id=:uid
            ORDER BY day DESC, id DESC
            LIMIT 2
            """), {"uid": int(user_id)}).fetchall()
            return rows

    rows = run_with_retry(_do)
    if not rows or len(rows) < 2:
        return None
    r = rows[1]
    return {
        "id": int(r[0]),
        "day": str(r[1]),
        "cash_jpy": float(r[2]),
        "stocks_jpy": float(r[3]),
        "other_jpy": float(r[4]),
        "other_name": str(r[5] or ""),
        "total_jpy": float(r[6]),
        "memo": str(r[7] or ""),
    }


def load_assets_history(user_id: int, limit: int = 30) -> pd.DataFrame:
    def _do():
        with ENGINE.begin() as conn:
            apply_sqlite_pragmas(conn)
            rows = conn.execute(text("""
            SELECT id, day, cash_jpy, stocks_jpy, other_jpy, other_name, total_jpy, memo
            FROM assets_snapshots
            WHERE user_id=:uid
            ORDER BY day DESC, id DESC
            LIMIT :lim
            """), {"uid": int(user_id), "lim": int(limit)}).fetchall()
            return rows

    rows = run_with_retry(_do)
    df = pd.DataFrame(rows, columns=["ID", "日付", "現金", "株式", "その他", "その他名", "合計", "メモ"])
    if df.empty:
        return df

    df = df.sort_values(["日付", "ID"], ascending=[False, False]).reset_index(drop=True)
    df["合計増減"] = df["合計"].diff(-1) * -1
    df["現金増減"] = df["現金"].diff(-1) * -1
    df["株式増減"] = df["株式"].diff(-1) * -1
    df["その他増減"] = df["その他"].diff(-1) * -1

    for c in ["現金", "株式", "その他", "合計", "合計増減", "現金増減", "株式増減", "その他増減"]:
        df[c] = df[c].map(lambda x: int(round(float(x))) if pd.notna(x) else 0)

    df["メモ"] = df["メモ"].fillna("").astype(str)
    df["その他名"] = df["その他名"].fillna("").astype(str)
    return df


# =========================================================
# 複利計算
# =========================================================
def compound_projection(principal: float, monthly_contrib: float, annual_rate_pct: float, years: int) -> Tuple[float, pd.DataFrame]:
    r = float(annual_rate_pct) / 100.0
    m = r / 12.0
    months = int(years) * 12

    value = float(principal)
    rows = []
    for mo in range(1, months + 1):
        value = value * (1.0 + m) + float(monthly_contrib)
        if mo % 12 == 0:
            y = mo // 12
            invested = float(principal) + float(monthly_contrib) * mo
            gain = value - invested
            rows.append([y, int(round(value)), int(round(invested)), int(round(gain))])

    df = pd.DataFrame(rows, columns=["年", "将来の資産（円）", "元本（入金合計）", "増えた分（利益）"])
    return value, df


# =========================================================
# サマライズ（AI用）
# =========================================================
def summarize(earn_df: pd.DataFrame, exp_df: pd.DataFrame, goal: float, fixed: float) -> dict:
    income = float(earn_df["円換算"].sum()) if (earn_df is not None and not earn_df.empty) else 0.0
    expense = float(exp_df["円換算"].sum()) if (exp_df is not None and not exp_df.empty) else 0.0
    profit = income - expense
    return {
        "income_jpy": income,
        "expense_jpy": expense,
        "profit_jpy": profit,
        "goal_jpy": float(goal),
        "fixed_cost_jpy": float(fixed),
    }


# =========================================================
# UI: 期間
# =========================================================
def period_selector() -> Tuple[date, date, str]:
    today = today_date()
    this_start, this_end = month_range(today)
    last_month_end = this_start - timedelta(days=1)
    last_start, last_end = month_range(last_month_end)

    mode = st.sidebar.selectbox("期間", ["今月", "先月", "直近30日", "カスタム"], index=0)

    if mode == "今月":
        return this_start, this_end, "今月"
    if mode == "先月":
        return last_start, last_end, "先月"
    if mode == "直近30日":
        s = today - timedelta(days=29)
        e = today
        return s, e, "直近30日"

    s = st.sidebar.date_input("開始日", value=this_start, min_value=MIN_DAY)
    e = st.sidebar.date_input("終了日", value=this_end, min_value=MIN_DAY)
    if s > e:
        st.sidebar.error("開始日が終了日より後です。")
        return e, s, "カスタム"
    return s, e, "カスタム"


# =========================================================
# UI: ログイン
# =========================================================
def render_login(in_sidebar: bool = True):
    """
    ログインUIを表示（サイドバーまたはexpander内で使用可能）
    """
    container = st.sidebar if in_sidebar else st
    
    container.markdown("### 🔐 ログイン（簡易）")
    username = container.text_input("ユーザー名", value="", placeholder="例：suzuki", key="login_username")
    pin = container.text_input("PIN（4〜8桁推奨）", value="", type="password", key="login_pin")

    col1, col2 = container.columns(2)
    with col1:
        if container.button("ログイン", use_container_width=True, key="login_btn"):
            if not username.strip() or not pin.strip():
                container.error("ユーザー名とPINを入力してください。")
                return
            user = get_user_by_username(username)
            if not user:
                container.error("ユーザーが存在しません（新規登録してください）。")
                return
            if user["pin_hash"] != pin_hash(username.strip(), pin.strip()):
                container.error("PINが違います。")
                return
            st.session_state["user_id"] = int(user["id"])
            st.session_state["username"] = user["username"]
            st.session_state.pop("is_guest", None)  # ゲストフラグをクリア
            st.rerun()

    with col2:
        if container.button("新規登録", use_container_width=True, key="register_btn"):
            if not username.strip() or not pin.strip():
                container.error("ユーザー名とPINを入力してください。")
                return
            user = get_user_by_username(username)
            if user:
                container.error("そのユーザー名は既に使われています。")
                return
            try:
                uid = create_user(username, pin)
            except IntegrityError as e:
                container.error(f"登録に失敗しました（DB互換の可能性）：{e}")
                return
            st.session_state["user_id"] = int(uid)
            st.session_state["username"] = username.strip()
            st.session_state.pop("is_guest", None)  # ゲストフラグをクリア
            st.rerun()


def render_sidebar_after_login(user_id: int):
    is_guest = st.session_state.get("is_guest", False)
    username = st.session_state.get('username', '')
    
    if is_guest:
        st.sidebar.markdown("### 👤 試用中（ゲスト）")
        st.sidebar.info(f"ユーザー：{username}")
        st.sidebar.warning("💡 データを保存するには、ユーザー名とPINを設定してログインしてください。")
        
        with st.sidebar.expander("🔐 ログイン設定（データ保存用）", expanded=False):
            new_username = st.text_input("ユーザー名", value="", placeholder="例：suzuki", key="guest_set_username")
            new_pin = st.text_input("PIN（4〜8桁推奨）", value="", type="password", key="guest_set_pin")
            
            if st.button("設定してログイン", use_container_width=True, key="guest_register_btn"):
                if not new_username.strip() or not new_pin.strip():
                    st.error("ユーザー名とPINを入力してください。")
                else:
                    # 既存ユーザー名チェック
                    existing = get_user_by_username(new_username)
                    if existing:
                        st.error("そのユーザー名は既に使われています。")
                    else:
                        # ゲストユーザーを正式ユーザーに変更（ユーザー名とPINを更新）
                        try:
                            # 現在のゲストユーザーを削除して新規作成
                            # （簡易実装：実際はUPDATEが理想だが、ここでは新規作成）
                            uid = create_user(new_username.strip(), new_pin.strip())
                            # データ移行（簡易版：ここでは新規ユーザーとして開始）
                            st.session_state["user_id"] = int(uid)
                            st.session_state["username"] = new_username.strip()
                            st.session_state.pop("is_guest", None)
                            st.success("ログイン設定が完了しました！")
                            st.rerun()
                        except Exception as e:
                            st.error(f"設定に失敗しました：{e}")
    else:
        st.sidebar.markdown("### 🔓 ログイン中")
        st.sidebar.success(f"ユーザー：{username}")
    
    if st.sidebar.button("ログアウト", use_container_width=True):
        st.session_state.pop("user_id", None)
        st.session_state.pop("username", None)
        st.session_state.pop("user_api_key", None)
        st.session_state.pop("chat_history", None)
        st.session_state.pop("is_guest", None)
        st.session_state.pop("onboarding_step", None)
        st.rerun()

    st.sidebar.markdown("---")

    # 期間
    start, end, _label = period_selector()

    # ユーザー設定（目標は「利益」）
    st.sidebar.markdown("### ⚙️ ユーザー設定")
    settings = get_user_settings(user_id)

    goal = st.sidebar.number_input(
        "今月の目標（利益・円）",
        min_value=0.0,
        value=float(settings["monthly_goal_jpy"]),
        step=1000.0,
        format="%.0f",
    )
    fixed = st.sidebar.number_input(
        "固定費（設定・円）",
        min_value=0.0,
        value=float(settings["fixed_cost_jpy"]),
        step=500.0,
        format="%.0f",
    )
    base_currency = st.sidebar.selectbox(
        "基準通貨（表示）",
        options=CURRENCY_OPTIONS,
        index=CURRENCY_OPTIONS.index(settings["base_currency"]) if settings["base_currency"] in CURRENCY_OPTIONS else 0,
        format_func=lambda c: currency_ja(c),
    )
    if st.sidebar.button("設定を保存", use_container_width=True):
        save_user_settings(user_id, float(goal), float(fixed), base_currency)
        st.sidebar.success("保存しました。")

    st.sidebar.markdown("---")

        # AI（一般ユーザーにキーを見せない）
    st.sidebar.markdown("### 🤖 AI")
    st.sidebar.caption("AIは現在テスト中です。必要な人だけ設定できます。")

    with st.sidebar.expander("上級者向け：AI設定（任意）", expanded=False):
        st.caption("ご自身のOpenAIキーを入れると回数制限なしで使えます（任意）。")
        user_key = st.text_input(
            "OpenAIキー（任意）",
            value=st.session_state.get("user_api_key", ""),
            type="password",
        )
        st.session_state["user_api_key"] = (user_key or "").strip()

        settings = get_user_settings(user_id)
        free_used = int(settings.get("ai_free_used", 0)) == 1
        service_key_ok = bool(get_service_openai_key())

        if user_key.strip():
            st.success("キー：入力済み")
        else:
            if service_key_ok and not free_used:
                st.info("キーなし無料：このユーザーは1回だけAIを実行できます")
            elif service_key_ok and free_used:
                st.warning("無料1回は使用済みです（続けるならキーを入力）")
            else:
                st.error("運営側のキー未設定のため、無料実行できません。")

    st.sidebar.markdown("---")

    # 為替（優先順＋DBにある通貨を後ろへ）
    st.sidebar.markdown("### 💱 為替レート（1通貨→円）")
    fx = get_fx_rates()
    db_curs = sorted(set(list(fx.keys())))
    ordered: List[str] = []
    for c in CURRENCY_ORDER:
        if c in db_curs:
            ordered.append(c)
    for c in db_curs:
        if c not in ordered:
            ordered.append(c)
    for c in CURRENCY_OPTIONS:
        if c not in ordered:
            ordered.append(c)

    cur = st.sidebar.selectbox(
        "通貨",
        options=ordered,
        index=0 if ordered else 0,
        format_func=lambda c: currency_ja(c),
    )
    rate = st.sidebar.number_input(
        "レート（円）",
        min_value=0.0,
        value=float(fx.get(cur, 1.0)),
        step=0.1,
        format="%.4f",
    )
    if st.sidebar.button("為替を更新", use_container_width=True):
        upsert_fx_rate(cur, float(rate))
        st.sidebar.success("更新しました。")
        st.rerun()

    return start, end, float(goal), float(fixed), (user_key or "").strip()


# =========================================================
# UI: 共通（自由入力）
# =========================================================
def pick_with_other(label: str, options: List[str], key: str, other_label: str = "自由入力（その他）") -> str:
    if "その他" not in options:
        options = options + ["その他"]

    sel = st.selectbox(label, options, index=0, key=f"{key}_sel")
    if sel == "その他":
        txt = st.text_input(other_label, value="", key=f"{key}_other")
        return (txt.strip() or "その他")
    return sel


# =========================================================
# UI: 直近カード（収益/経費）編集・削除
# =========================================================
def render_recent_earnings_edit_delete(user_id: int, start: date, end: date, limit: int = 3):
    df = load_earnings(user_id, start, end)
    if df.empty:
        st.caption("直近の収益はまだありません。")
        return

    recent = df.sort_values(["日付", "ID"], ascending=[False, False]).head(limit)
    st.markdown("#### 🕘 直近の収益（すぐ編集/削除）")

    for r in recent.itertuples(index=False):
        left, b1, b2 = st.columns([0.74, 0.13, 0.13])
        with left:
            st.caption(f"{r.日付}｜{r.プラットフォーム}｜{r.カテゴリ}｜{yen(r.円換算)}")
        with b1:
            if st.button("編集", key=f"edit_earn_{r.ID}", use_container_width=True):
                st.session_state["editing_earning_id"] = int(r.ID)
                st.rerun()
        with b2:
            if st.button("削除", key=f"del_earn_{r.ID}", use_container_width=True):
                delete_earning(user_id, int(r.ID))
                st.toast("収益を削除しました")
                st.rerun()

    eid = st.session_state.get("editing_earning_id")
    if eid:
        row = df[df["ID"] == eid]
        if row.empty:
            st.session_state.pop("editing_earning_id", None)
            return

        rr = row.iloc[0]
        st.markdown("##### ✏️ 収益を編集")
        with st.container(border=True):
            c1, c2, c3, c4, c5, c6 = st.columns([1.1, 1.4, 1.2, 1.0, 0.9, 1.4])
            with c1:
                e_day = st.date_input("日付", value=date.fromisoformat(rr["日付"]), min_value=MIN_DAY, key="edit_e_day")
            with c2:
                e_platform = pick_with_other("プラットフォーム", DEFAULT_PLATFORMS, key="edit_e_platform")
            with c3:
                e_cat = pick_with_other("カテゴリ", DEFAULT_EARN_CATEGORIES, key="edit_e_cat")
            with c4:
                e_amt = st.number_input("金額", min_value=0.0, value=float(rr["金額"]), step=1.0, format="%.0f", key="edit_e_amt")
            with c5:
                cur_code = (rr.get("通貨コード") or "JPY")
                idx = CURRENCY_OPTIONS.index(cur_code) if cur_code in CURRENCY_OPTIONS else 0
                e_cur = st.selectbox("通貨", CURRENCY_OPTIONS, index=idx, key="edit_e_cur", format_func=currency_ja)
            with c6:
                e_memo = st.text_input("メモ（任意）", value=str(rr["メモ"] or ""), key="edit_e_memo")

            fx = get_fx_rates()
            st.caption(
                f"円換算（概算）：{yen(compute_jpy(e_amt, e_cur, fx))}"
                f"（1{currency_ja(e_cur)}={int(round(fx.get(e_cur, 1.0)))}円）"
            )

            a, b = st.columns(2)
            with a:
                if st.button("保存（収益）", key=f"earn_save_{eid}", use_container_width=True):
                    update_earning(user_id, eid, e_day, e_platform, e_cat, e_cur, float(e_amt), e_memo)
                    st.session_state.pop("editing_earning_id", None)
                    st.success("更新しました。")
                    st.rerun()
            with b:
                if st.button("キャンセル", key=f"earn_cancel_{eid}", use_container_width=True):
                    st.session_state.pop("editing_earning_id", None)
                    st.rerun()


def render_recent_expenses_edit_delete(user_id: int, start: date, end: date, limit: int = 3):
    df = load_expenses(user_id, start, end)
    if df.empty:
        st.caption("直近の経費はまだありません。")
        return

    recent = df.sort_values(["日付", "ID"], ascending=[False, False]).head(limit)
    st.markdown("#### 🕘 直近の経費（すぐ編集/削除）")

    for r in recent.itertuples(index=False):
        left, b1, b2 = st.columns([0.74, 0.13, 0.13])
        with left:
            st.caption(f"{r.日付}｜{r.支払先}｜{r.カテゴリ}｜{yen(r.円換算)}")
        with b1:
            if st.button("編集", key=f"edit_exp_{r.ID}", use_container_width=True):
                st.session_state["editing_expense_id"] = int(r.ID)
                st.rerun()
        with b2:
            if st.button("削除", key=f"del_exp_{r.ID}", use_container_width=True):
                delete_expense(user_id, int(r.ID))
                st.toast("経費を削除しました")
                st.rerun()

    

# =========================================================
# UI: 資産（直近 編集/削除 + 前回比：矢印/色を統一）
# =========================================================
def render_assets_section(user_id: int):
    st.subheader("💰 資産と複利（投資の見える化）")
    latest = load_latest_assets(user_id)
    prev = load_previous_assets(user_id)

    with st.container(border=True):
        st.markdown("#### ① 資産を保存（現金・株式・その他）")

        if latest:
            st.caption(
                f"最新：{latest['day']}｜合計 {yen(latest['total_jpy'])}（現金 {yen(latest['cash_jpy'])} / "
                f"株式 {yen(latest['stocks_jpy'])} / その他 {yen(latest['other_jpy'])}）"
            )
        else:
            st.caption("まだ資産データがありません。")

        a1, a2, a3, a4, a5, a6 = st.columns([1.1, 1.1, 1.1, 1.1, 1.2, 1.4])
        with a1:
            a_day = st.date_input("日付（資産）", value=today_date(), min_value=MIN_DAY, key="a_day")
        with a2:
            cash = st.number_input("現金（円）", min_value=0.0, value=float(latest["cash_jpy"]) if latest else 0.0, step=1000.0, format="%.0f", key="a_cash")
        with a3:
            stocks = st.number_input("株式（円）", min_value=0.0, value=float(latest["stocks_jpy"]) if latest else 0.0, step=1000.0, format="%.0f", key="a_stocks")
        with a4:
            other = st.number_input("その他（円）", min_value=0.0, value=float(latest["other_jpy"]) if latest else 0.0, step=1000.0, format="%.0f", key="a_other")
        with a5:
            other_name = st.text_input("その他名（任意）", value=str(latest["other_name"]) if latest else "不動産等", key="a_other_name")
        with a6:
            a_memo = st.text_input("メモ（任意）", value="", key="a_memo")

        st.caption(f"合計（計算）：{yen(cash + stocks + other)}")

        if st.button("資産を保存", use_container_width=True):
            upsert_assets_snapshot(user_id, a_day, float(cash), float(stocks), float(other), other_name, a_memo)
            st.success("資産を保存しました。")
            st.rerun()

        st.markdown("---")
        st.markdown("#### 🧾 直近の資産（すぐ編集/削除）")
        if not latest:
            st.caption("直近の資産はまだありません。")
        else:
            prev_cash = float(prev["cash_jpy"]) if prev else 0.0
            prev_stocks = float(prev["stocks_jpy"]) if prev else 0.0
            prev_other = float(prev["other_jpy"]) if prev else 0.0
            prev_total = float(prev["total_jpy"]) if prev else 0.0

            st.markdown(
                f"""
<div class="asset-recent-block">
  <div class="asset-recent-line">
    <b>{latest['day']}</b>｜合計 <b>{yen(latest['total_jpy'])}</b>
    （現金 {yen(latest['cash_jpy'])} / 株式 {yen(latest['stocks_jpy'])} / その他 {yen(latest['other_jpy'])}）
  </div>

  <div class="asset-recent-delta">
    前回比：
    合計 {html_delta_badge(float(latest['total_jpy']) - prev_total, prev_total, big=True)} ／
    現金 {html_delta_badge(float(latest['cash_jpy']) - prev_cash, prev_cash, big=True)} ／
    株式 {html_delta_badge(float(latest['stocks_jpy']) - prev_stocks, prev_stocks, big=True)} ／
    その他 {html_delta_badge(float(latest['other_jpy']) - prev_other, prev_other, big=True)}
  </div>
</div>
                """,
                unsafe_allow_html=True,
            )

            b1, b2 = st.columns([0.5, 0.5])
            with b1:
                if st.button("編集（直近資産）", use_container_width=True):
                    st.session_state["editing_asset_id"] = int(latest["id"])
                    st.rerun()
            with b2:
                if st.button("削除（直近資産）", use_container_width=True):
                    delete_assets_snapshot(user_id, int(latest["id"]))
                    st.toast("資産を削除しました")
                    st.rerun()

            aid = st.session_state.get("editing_asset_id")
            if aid == int(latest["id"]):
                st.markdown("##### ✏️ 資産を編集")
                with st.container(border=True):
                    e1, e2, e3, e4, e5, e6 = st.columns([1.1, 1.1, 1.1, 1.1, 1.2, 1.4])
                    with e1:
                        eday = st.date_input("日付", value=date.fromisoformat(latest["day"]), min_value=MIN_DAY, key="edit_a_day")
                    with e2:
                        ecash = st.number_input("現金（円）", min_value=0.0, value=float(latest["cash_jpy"]), step=1000.0, format="%.0f", key="edit_a_cash")
                    with e3:
                        estocks = st.number_input("株式（円）", min_value=0.0, value=float(latest["stocks_jpy"]), step=1000.0, format="%.0f", key="edit_a_stocks")
                    with e4:
                        eother = st.number_input("その他（円）", min_value=0.0, value=float(latest["other_jpy"]), step=1000.0, format="%.0f", key="edit_a_other")
                    with e5:
                        eother_name = st.text_input("その他名（任意）", value=str(latest["other_name"]), key="edit_a_other_name")
                    with e6:
                        ememo = st.text_input("メモ（任意）", value=str(latest["memo"] or ""), key="edit_a_memo")

                    st.caption(f"合計（計算）：{yen(ecash + estocks + eother)}")

                    x, y = st.columns(2)
                    with x:
                        asset_id = int(latest["id"]) if latest and "id" in latest else 0
                        if st.button("保存（資産）", key=f"asset_save_{asset_id}", use_container_width=True):
                            update_assets_snapshot(user_id, int(latest["id"]), eday, float(ecash), float(estocks), float(eother), eother_name, ememo)
                            st.session_state.pop("editing_asset_id", None)
                            st.success("更新しました。")
                            st.rerun()

                    with y:
                        if st.button("キャンセル（資産）", key=f"asset_cancel_{asset_id}", use_container_width=True):
                            st.session_state.pop("editing_asset_id", None)
                            st.rerun()

        with st.expander("資産の履歴（最新30件・増減つき）", expanded=False):
            hist = load_assets_history(user_id, limit=30)
            if hist.empty:
                st.info("まだ資産履歴がありません。")
            else:
                st.dataframe(hist, use_container_width=True, hide_index=True)

    with st.container(border=True):
        st.markdown("#### ② 複利計算（積立＋利回り）")
        latest2 = load_latest_assets(user_id)
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            principal = st.number_input("元本（いまの資産・円）", min_value=0.0, value=float(latest2["total_jpy"]) if latest2 else 0.0, step=10000.0, format="%.0f")
        with b2:
            monthly = st.number_input("毎月の積立（円）", min_value=0.0, value=50000.0, step=5000.0, format="%.0f")
        with b3:
            annual = st.number_input("年利（%）", min_value=0.0, value=7.0, step=0.5, format="%.1f")
        with b4:
            years = st.number_input("年数（年）", min_value=1, value=10, step=1)

        fv, df = compound_projection(float(principal), float(monthly), float(annual), int(years))
        invested_total = float(principal) + float(monthly) * int(years) * 12
        gain = float(fv) - invested_total

        c1, c2, c3 = st.columns(3)
        c1.metric("将来の資産（予測）", yen(fv))
        c2.metric("入金合計", yen(invested_total))
        c3.metric("増えた分（利益）", yen(gain))
        st.caption("※これは“期待リターン”のシミュレーションです（確定ではありません）。")

        with st.expander("年ごとの内訳（表）", expanded=False):
            st.dataframe(df, use_container_width=True, hide_index=True)
# =========================================================
# AI（キー無しでも1回無料 + 自由質問モード）
# =========================================================
def _responses_api_call(api_key: str, messages: List[dict]) -> str:
    """
    OpenAI Responses API を requests で叩く（依存最小化）
    """
    import requests

    # ★Pylance警告を100%消しつつ、動作も安定させるために関数内で確定させる
    base_url = os.getenv("OPENAI_BASE_URL", globals().get("OPENAI_BASE_URL", "https://api.openai.com/v1")).strip()
    model = os.getenv("OPENAI_MODEL", globals().get("OPENAI_MODEL", "gpt-5-mini")).strip()

    url = f"{base_url}/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "input": messages,
    }

    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if r.status_code >= 400:
        raise RuntimeError(f"OpenAI API error {r.status_code}: {r.text}")

    data = r.json()

    out = data.get("output", [])
    texts: List[str] = []
    for item in out:
        for c in item.get("content", []) or []:
            t = c.get("text")
            if t:
                texts.append(t)

    if not texts:
        return str(data)
    return "\n".join(texts).strip()


def get_service_openai_key() -> str:
    """
    運営側キー（環境変数）取得。未設定なら空文字。
    """
    return (os.getenv("OPENAI_API_KEY", "") or "").strip()


def can_use_service_ai(user_id: int) -> Tuple[bool, str]:
    """
    サービスキー（OPENAI_API_KEY）でAIを使えるか？
    無料ユーザー：1人1回だけ（ai_free_usedで管理）
    """
    settings = get_user_settings(user_id)
    free_used = int(settings.get("ai_free_used", 0)) == 1
    service_key = get_service_openai_key()
    if not service_key:
        return False, "運営側のOpenAIキーが未設定です。"
    if free_used:
        return False, "無料枠（1回）は既に使用済みです。"
    return True, ""


def get_effective_api_key(user_id: int, user_supplied_key: str) -> Tuple[Optional[str], str, bool]:
    """
    返り値: (api_key, 状態メッセージ, using_service_key)
    - user_supplied_key があればそれを優先（回数制限なし）
    - なければサービスキー（無料1回）を使う
    """
    user_supplied_key = (user_supplied_key or "").strip()
    if user_supplied_key:
        return user_supplied_key, "ユーザーキー", False

    ok, reason = can_use_service_ai(user_id)
    if not ok:
        return None, reason, False

    return get_service_openai_key(), "無料1回", True


def run_ai_with_limits(user_id: int, user_supplied_key: str, messages: List[dict]) -> Tuple[Optional[str], str]:
    """
    無料枠管理を含めてAI実行。
    """
    api_key, mode, using_service = get_effective_api_key(user_id, user_supplied_key)
    if not api_key:
        return None, mode

    try:
        txt = _responses_api_call(api_key, messages)
        if using_service:
            mark_ai_free_used(user_id)
        return txt, "ok"
    except Exception as e:
        return None, f"AI呼び出しに失敗：{e}"


# =========================================================
# UI: AI（分析＋自由質問チャット）
# =========================================================
def render_ai_section(user_id: int, goal: float, fixed: float, user_key: str):
    st.subheader("🤖 AI（分析 / 自由質問）")

    tab1, tab2 = st.tabs(["📌 AI分析（今月）", "💬 AIに質問（チャット）"])

    # 共通：今月データ
    today = today_date()
    m_start, m_end = month_range(today)
    m_earn = load_earnings(user_id, m_start, m_end)
    m_exp = load_expenses(user_id, m_start, m_end)
    summ_m = summarize(m_earn, m_exp, goal, fixed)

    # ---------- 分析 ----------
    with tab1:
        st.caption("今月の数字から「優先アクション」を提案します。")

        settings = get_user_settings(user_id)
        free_used = int(settings.get("ai_free_used", 0)) == 1
        service_key_ok = bool(get_service_openai_key())

        if (user_key or "").strip():
            st.info("ユーザーキーが入っているため、AI分析は回数制限なしで実行できます。")
        else:
            if service_key_ok and not free_used:
                st.success("このユーザーは「キーなし」で無料1回だけAIを実行できます。")
            elif service_key_ok and free_used:
                st.warning("無料1回は使用済みです。続けるならサイドバーでユーザーキーを入力してください。")
            else:
                st.error("運営側のOpenAIキーが未設定のため、AIを実行できません。")

        if st.button("AI分析を実行", key="run_ai_analysis"):
            system = (
                "あなたは収益管理・家計改善・副業の実行計画に強いコーチです。"
                "曖昧に褒めず、数字を根拠に、具体的な改善策・優先順位・次の一手まで落とし込みます。"
                "文章は短すぎないように。日本語のみ。"
            )
            user_prompt = f"""
以下はユーザーの今月のデータです。これを根拠に「今月の優先アクション」を提案してください。
制約：やることは増やしすぎない（最大でも5アクション）。ただし説明は丁寧に。
最後に必ず「次に入力すべきデータ」を1つだけ指定してください。

【今月の数値（円）】
- 収益：{int(summ_m.get('income_jpy', 0))}
- 経費：{int(summ_m.get('expense_jpy', 0))}
- 利益：{int(summ_m.get('profit_jpy', 0))}
- 目標（利益）：{int(summ_m.get('goal_jpy', 0))}
- 固定費（設定）：{int(summ_m.get('fixed_cost_jpy', 0))}

【参考】
- 収益明細（最大10件）：{m_earn.head(10).to_dict(orient="records") if not m_earn.empty else []}
- 経費明細（最大10件）：{m_exp.head(10).to_dict(orient="records") if not m_exp.empty else []}
"""

            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ]

            with st.spinner("AIが分析中…"):
                txt, status = run_ai_with_limits(user_id, user_key, messages)

            if txt:
                st.success("✅ AI分析結果")
                st.markdown(txt)
            else:
                st.error(status)

    # ---------- チャット（自由質問） ----------
    with tab2:
        st.caption("収益/経費/資産/投資/副業など、自由に質問できます。")

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        # 既存表示
        for m in st.session_state["chat_history"]:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_msg = st.chat_input("例：今月の赤字を最短で止めるには？ / 収益を増やす打ち手は？")
        if user_msg:
            st.session_state["chat_history"].append({"role": "user", "content": user_msg})
            with st.chat_message("user"):
                st.markdown(user_msg)

            system = (
                "あなたは「収益ナビ」の専属コーチです。"
                "ユーザーの数字を踏まえて、具体的・実務的に答えてください。"
                "最後に「次の一手」を1つだけ提示してください。日本語のみ。"
            )
            context = f"""
【今月サマリー（円）】
- 収益：{int(summ_m.get('income_jpy', 0))}
- 経費：{int(summ_m.get('expense_jpy', 0))}
- 利益：{int(summ_m.get('profit_jpy', 0))}
- 目標（利益）：{int(summ_m.get('goal_jpy', 0))}
- 固定費：{int(summ_m.get('fixed_cost_jpy', 0))}

【明細の参考】
- 収益（最大10件）：{m_earn.head(10).to_dict(orient="records") if not m_earn.empty else []}
- 経費（最大10件）：{m_exp.head(10).to_dict(orient="records") if not m_exp.empty else []}
"""

            # 直近だけ入れて長文化を防ぐ
            short_hist = st.session_state["chat_history"][-8:]
            messages = [{"role": "system", "content": system}]
            messages.append({"role": "user", "content": context})
            messages.extend(short_hist)

            with st.chat_message("assistant"):
                with st.spinner("AIが返信中…"):
                    txt, status = run_ai_with_limits(user_id, user_key, messages)
                if txt:
                    st.markdown(txt)
                    st.session_state["chat_history"].append({"role": "assistant", "content": txt})
                else:
                    st.error(status)


# =========================================================
# UI: スクロール用ユーティリティ（確実に動作するように）
# =========================================================
def scroll_to_section(anchor_id: str, delay_ms: int = 300):
    """
    指定したアンカーIDのセクションへ確実にスクロール（スマホ対応・components.html使用）
    
    Args:
        anchor_id: スクロール先のアンカーID（例：「expense-section」）
        delay_ms: スクロール実行までの遅延（ミリ秒、Streamlitのレンダリング完了を待つ）
    """
    scroll_js = f"""
    <script>
    (function() {{
        function scrollToTarget() {{
            const element = document.getElementById('{anchor_id}');
            if (element) {{
                // scrollIntoViewを使用（スマホ対応・確実に動作）
                element.scrollIntoView({{
                    behavior: 'smooth',
                    block: 'start'
                }});
                return true;
            }}
            return false;
        }}
        
        // 初回試行（Streamlitのレンダリング完了を待つ）
        setTimeout(function() {{
            if (!scrollToTarget()) {{
                // 要素が見つからない場合は再試行
                setTimeout(function() {{
                    scrollToTarget();
                }}, 200);
            }}
        }}, {delay_ms});
    }})();
    </script>
    """
    components.html(scroll_js, height=0)


# =========================================================
# UI: スクロール要求ヘルパー（scroll_to に統一）
# =========================================================
def request_scroll(anchor_id: str) -> None:
    """
    次回レンダリング時に指定したアンカーIDへスクロールさせる要求をセット。
    """
    st.session_state["scroll_to"] = anchor_id


def perform_scroll_if_requested() -> None:
    """
    ページ描画の最後に1回だけ呼び出し、
    必要であれば scrollIntoView を実行してから state を必ずクリアする。
    """
    anchor_id = st.session_state.pop("scroll_to", None)
    if not anchor_id:
        return

    scroll_js = f"""
    <script>
    (function() {{
        var targetId = '{anchor_id}';
        var retries = 0;
        var maxRetries = 40; // 最大約2秒（50ms × 40回）

        // iOS Safari の自動スクロール復元を抑止（可能なら）
        try {{
            if ('scrollRestoration' in history) {{
                history.scrollRestoration = 'manual';
            }}
        }} catch (e) {{}}

        function doScroll() {{
            requestAnimationFrame(function() {{
                requestAnimationFrame(function() {{
                    var el = document.getElementById(targetId);
                    if (el) {{
                        try {{
                            el.scrollIntoView({{ behavior: 'auto', block: 'start' }});
                        }} catch (e) {{
                            // scrollIntoView が不安定な環境向けフォールバック
                            try {{
                                var rect = el.getBoundingClientRect();
                                var currentY = window.pageYOffset || document.documentElement.scrollTop || 0;
                                var y = currentY + rect.top - 8;
                                window.scrollTo(0, y);
                            }} catch (e2) {{}}
                        }}
                    }} else if (retries < maxRetries) {{
                        retries += 1;
                        setTimeout(doScroll, 50);
                    }}
                }});
            }});
        }}

        // 初回は短い遅延を置いてから開始
        setTimeout(doScroll, 50);
    }})();
    </script>
    """
    components.html(scroll_js, height=0)


# =========================================================
# UI: 成功メッセージ＋次アクションCTA（共通関数）
# =========================================================
def render_success_with_next_action(
    success_message: str,
    next_action_label: str,
    cta_button_label: str,
    cta_button_key: str,
    target_anchor_id: str,
    flag_key: str,
    scroll_flag_key: str,
    on_cta_click_callback: Optional[Callable] = None
):
    """
    成功メッセージと次アクションCTAを画面上部に表示（スマホ最優先）
    
    Args:
        success_message: 成功メッセージ（例：「✅ 収益を1件追加しました！」）
        next_action_label: 次アクションの説明（例：「次：経費を1件追加（約1分）」）
        cta_button_label: CTAボタンのラベル（例：「✍️ 経費入力セクションへ移動」）
        cta_button_key: CTAボタンのキー（一意である必要がある）
        target_anchor_id: スクロール先のアンカーID（例：「expense-section」）
        flag_key: 成功フラグのキー（例：「income_added」）
        scroll_flag_key: スクロールフラグのキー（例：「scroll_to_expense」）
        on_cta_click_callback: CTA押下時の追加処理（オプション）
    """
    # 画面上部に成功メッセージ＋CTAを表示（必ず見える位置）
    # 注意：トーストはrender_mainの最上部で表示されるため、ここでは表示しない
    with st.container(border=True):
        st.success(success_message)
        st.markdown(f"**{next_action_label}**")
        
        # CTAボタン（ユーザーが押した時だけスクロール）
        if st.button(cta_button_label, type="primary", use_container_width=True, key=cta_button_key):
            # スクロールフラグを設定（次回レンダリングでスクロール実行）
            st.session_state[scroll_flag_key] = True
            
            # 追加処理があれば実行
            if on_cta_click_callback:
                on_cta_click_callback()
            
            # フラグをクリア
            st.session_state[flag_key] = False
            st.rerun()
    
    # スクロールフラグが立っている場合はスクロール実行
    if st.session_state.get(scroll_flag_key, False):
        scroll_to_section(target_anchor_id, delay_ms=300)
        st.session_state[scroll_flag_key] = False


# =========================================================
# UI: メイン（赤字/黒字の矢印・色を統一 / 英語排除）
# =========================================================
def render_main(user_id: int, start: date, end: date, goal: float, fixed: float, user_key: str):
    st.markdown(f"## {APP_TITLE}")
    
    # =========================================================
    # step制の初期化（状態管理）
    # =========================================================
    if "step" not in st.session_state:
        st.session_state["step"] = "income"
    
    # オンボーディング（ゲストユーザー向け）
    is_guest = st.session_state.get("is_guest", False)
    onboarding_step = st.session_state.get("onboarding_step", 0)
    
    if is_guest and onboarding_step > 0:
        today = today_date()
        m_start, m_end = month_range(today)
        m_earn = load_earnings(user_id, m_start, m_end)
        m_exp = load_expenses(user_id, m_start, m_end)
        
        with st.container(border=True):
            # ガイド文言（目的を1点に絞る）
            st.markdown("### 🎯 まずは収益を1件だけ入力してください（約1分）")
            st.markdown(
                """
                <div style='margin-top: 8px; margin-bottom: 16px; font-size: 14px; color: var(--rn-subtext);'>
                このあと分かること：<br>
                ・今月の収支バランス<br>
                ・一番ムダな支出<br>
                ・改善アクション（AI）
                </div>
                """,
                unsafe_allow_html=True
            )
            
            step1_done = not m_earn.empty
            step2_done = not m_exp.empty
            step3_done = step1_done and step2_done
            
            # 進捗アンロック方式：完了したステップと次のステップのみ表示
            if not step1_done:
                # 初期：①のみ表示
                st.markdown(f"**① 収益を1件追加**")
            elif step1_done and not step2_done:
                # ①完了後：①✅と②を表示
                st.markdown(f"**✅ 収益を1件追加**（完了！）")
                st.markdown("---")
                st.markdown(f"**② 経費を1件追加**")
            elif step1_done and step2_done and not step3_done:
                # ①②完了後：①②✅と③を表示
                st.markdown(f"**✅ 収益を1件追加**（完了！）")
                st.markdown(f"**✅ 経費を1件追加**（完了！）")
                st.markdown("---")
                st.markdown(f"**③ 結果を見る**")
            else:
                # すべて完了
                st.markdown(f"**✅ 収益を1件追加**（完了！）")
                st.markdown(f"**✅ 経費を1件追加**（完了！）")
                st.markdown(f"**✅ 結果を見る**（完了！）")
            
            if step3_done:
                st.markdown("---")
                st.success("🎉 試用完了！データを保存するには、サイドバー(>>)からログイン（ユーザー名/PIN）を設定してください。")
                if st.button("オンボーディングを閉じる", key="close_onboarding"):
                    st.session_state["onboarding_step"] = 0
                    st.rerun()

    # 収益セクションのアンカーを配置（スクロールターゲット用・確実なID）
    st.markdown('<div id="income-section"></div>', unsafe_allow_html=True)
    
    st.subheader("➕ 収益を追加")
    with st.container(border=True):
        # ログイン前は最小フォーム、ログイン後は全項目表示
        is_guest = st.session_state.get("is_guest", False)
        
        if is_guest:
            # ログイン前：最小フォーム（金額・カテゴリのみ）
            # 日付はデフォルトで今日（任意）
            e_day = today_date()
            e_platform = "未設定"
            e_memo = ""
            
            col1, col2 = st.columns(2)
            with col1:
                # フォーム値リセット対応：追加成功後は金額を0にリセット
                current_step = st.session_state.get("step", "income")
                default_amt = 0.0 if current_step == "income_done" else st.session_state.get("e_amt_value", 0.0)
                e_amt = st.number_input("金額（必須）", min_value=0.0, value=default_amt, step=1.0, format="%.0f", key="e_amt")
                # 現在の値を保存（リセット用）
                if current_step != "income_done":
                    st.session_state["e_amt_value"] = e_amt
            with col2:
                e_cat = pick_with_other("カテゴリ（必須）", DEFAULT_EARN_CATEGORIES, key="e_cat")
            
            # 詳細設定（折りたたみ）
            with st.expander("📝 詳細設定（任意）", expanded=False):
                e_day = st.date_input("日付", value=e_day, min_value=MIN_DAY, key="e_day")
                e_platform = pick_with_other("プラットフォーム", DEFAULT_PLATFORMS, key="e_platform")
                e_memo = st.text_input("メモ", value="", key="e_memo")
                fx = get_fx_rates()
                jpy_cur = "JPY"
                st.caption(
                    f"円換算（概算）：{yen(compute_jpy(e_amt, jpy_cur, fx))}（1円=1円）"
                )
            
            # デフォルト値設定（ログイン前）
            e_cur = "JPY"  # 円固定
            if not e_platform or e_platform.strip() == "":
                e_platform = "未設定"
            if not e_memo:
                e_memo = ""
            
            # 送信ボタン
            if st.button("収益を追加", key="add_earning", use_container_width=True):
                insert_earning(user_id, e_day, e_platform, e_cat, e_cur, float(e_amt), e_memo)
                st.session_state["step"] = "income_done"  # step制：収益追加成功
                st.session_state["e_amt_value"] = 0.0
                request_scroll("expense-section")  # 経費セクション先頭へ自動スクロール
                st.rerun()
        else:
            # ログイン後：全項目表示（既存のフォーム）
            # 日付（1カラム）
            e_day = st.date_input("日付", value=today_date(), min_value=MIN_DAY, key="e_day")
            
            # プラットフォーム×カテゴリ（2カラム）
            col1, col2 = st.columns(2)
            with col1:
                e_platform = pick_with_other("プラットフォーム", DEFAULT_PLATFORMS, key="e_platform")
            with col2:
                e_cat = pick_with_other("カテゴリ", DEFAULT_EARN_CATEGORIES, key="e_cat")
            
            # 金額×通貨（2カラム）
            col3, col4 = st.columns(2)
            with col3:
                # フォーム値リセット対応：追加成功後は金額を0にリセット
                current_step = st.session_state.get("step", "income")
                default_amt = 0.0 if current_step == "income_done" else st.session_state.get("e_amt_value", 0.0)
                e_amt = st.number_input("金額", min_value=0.0, value=default_amt, step=1.0, format="%.0f", key="e_amt")
                # 現在の値を保存（リセット用）
                if current_step != "income_done":
                    st.session_state["e_amt_value"] = e_amt
            with col4:
                e_cur = st.selectbox("通貨", CURRENCY_OPTIONS, index=0, key="e_cur", format_func=currency_ja)
            
            # メモ（1カラム）
            e_memo = st.text_input("メモ（任意）", value="", key="e_memo")
            
            # 円換算（小さく表示）
            fx = get_fx_rates()
            st.caption(
                f"円換算（概算）：{yen(compute_jpy(e_amt, e_cur, fx))}（1{currency_ja(e_cur)}={int(round(fx.get(e_cur, 1.0)))}円）"
            )
            
            # 送信ボタン（1カラム）
            if st.button("収益を追加", key="add_earning", use_container_width=True):
                insert_earning(user_id, e_day, e_platform, e_cat, e_cur, float(e_amt), e_memo)
                # step制：収益追加成功
                st.session_state["step"] = "income_done"
                # フォーム値をリセット（金額を0に）
                st.session_state["e_amt_value"] = 0.0
                request_scroll("expense-section")  # 経費セクション先頭へ自動スクロール
                st.rerun()
    
    with st.expander("🕘 直近の収益（編集/削除）", expanded=False):
        render_recent_earnings_edit_delete(user_id, start, end, limit=3)

    # =========================================================
    # 収益追加成功メッセージ（経費セクション直前・スクロール先付近に表示）
    # =========================================================
    if st.session_state.get("step") == "income_done":
        with st.container(border=True):
            st.success("✅ 収益を1件追加しました！")
            st.markdown("**次：経費を1件追加（約1分）**")
            if st.button("✍️ 経費入力セクションへ移動", type="primary", use_container_width=True, key="goto_expense_btn"):
                st.session_state["step"] = "expense"  # step制：経費入力へ
                request_scroll("expense-section")  # スクロールターゲット設定
                st.rerun()

    # 経費入力フォームの見出し直前にアンカーを配置（スクロールターゲット用・確実なID）
    st.markdown('<div id="expense-section"></div>', unsafe_allow_html=True)
    
    st.subheader("➖ 経費を追加")
    with st.container(border=True):
        # ログイン前は最小フォーム、ログイン後は全項目表示
        is_guest = st.session_state.get("is_guest", False)
        
        if is_guest:
            # ログイン前：最小フォーム（金額・カテゴリのみ）
            # 日付はデフォルトで今日（任意）
            x_day = today_date()
            x_vendor = "未設定"
            x_memo = ""
            
            col1, col2 = st.columns(2)
            with col1:
                x_amt = st.number_input("金額（必須）", min_value=0.0, value=0.0, step=1.0, format="%.0f", key="x_amt")
            with col2:
                x_cat = pick_with_other("カテゴリ（必須）", DEFAULT_EXP_CATEGORIES, key="x_cat")
            
            # 詳細設定（折りたたみ）
            with st.expander("📝 詳細設定（任意）", expanded=False):
                x_day = st.date_input("日付", value=x_day, min_value=MIN_DAY, key="x_day")
                x_vendor = st.text_input("支払先", value="", key="x_vendor")
                x_memo = st.text_input("メモ", value="", key="x_memo")
                fx = get_fx_rates()
                jpy_cur = "JPY"
                st.caption(
                    f"円換算（概算）：{yen(compute_jpy(x_amt, jpy_cur, fx))}（1円=1円）"
                )
            
            # デフォルト値設定（ログイン前）
            x_cur = "JPY"  # 円固定
            if not x_vendor or x_vendor.strip() == "":
                x_vendor = "未設定"
            if not x_memo:
                x_memo = ""
            
            # 送信ボタン
            if st.button("経費を追加", key="add_expense", use_container_width=True):
                insert_expense(user_id, x_day, x_vendor, x_cat, x_cur, float(x_amt), x_memo)
                st.session_state["step"] = "expense_done"  # step制：経費追加成功
                request_scroll("expense-success-section")  # 「結果を見る」ボタン位置へ自動スクロール
                st.rerun()
        else:
            # ログイン後：全項目表示（既存のフォーム）
            # 日付（1カラム）
            x_day = st.date_input("日付", value=today_date(), min_value=MIN_DAY, key="x_day")
            
            # 支払先×カテゴリ（2カラム）
            col1, col2 = st.columns(2)
            with col1:
                x_vendor = st.text_input("支払先", value="ChatGPT", key="x_vendor")
            with col2:
                x_cat = pick_with_other("カテゴリ（経費）", DEFAULT_EXP_CATEGORIES, key="x_cat")
            
            # 金額×通貨（2カラム）
            col3, col4 = st.columns(2)
            with col3:
                x_amt = st.number_input("金額（経費）", min_value=0.0, value=0.0, step=1.0, format="%.0f", key="x_amt")
            with col4:
                x_cur = st.selectbox("通貨（経費）", CURRENCY_OPTIONS, index=0, key="x_cur", format_func=currency_ja)
            
            # メモ（1カラム）
            x_memo = st.text_input("メモ（任意）", value="", key="x_memo")
            
            # 円換算（小さく表示）
            fx = get_fx_rates()
            st.caption(
                f"円換算（概算）：{yen(compute_jpy(x_amt, x_cur, fx))}（1{currency_ja(x_cur)}={int(round(fx.get(x_cur, 1.0)))}円）"
            )
            
            # 送信ボタン（1カラム）
            if st.button("経費を追加", key="add_expense", use_container_width=True):
                insert_expense(user_id, x_day, x_vendor, x_cat, x_cur, float(x_amt), x_memo)
                # step制：経費追加成功
                st.session_state["step"] = "expense_done"
                request_scroll("expense-success-section")  # 「結果を見る」ボタン位置へ自動スクロール
                st.rerun()
    
    # =========================================================
    # 経費追加成功メッセージ（フォーム直下に固定表示・「結果を見る」ボタンが見える位置）
    # =========================================================
    if st.session_state.get("step") == "expense_done":
        # アンカーIDを設定（自動スクロールのターゲット）
        st.markdown('<div id="expense-success-section"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.success("✅ 経費を1件追加しました！")
            st.markdown("**結果を見る準備ができました**")
            if st.button("📊 結果を見る", type="primary", use_container_width=True, key="view_results_btn"):
                st.session_state["step"] = "result"  # step制：結果表示へ
                st.session_state["show_results_section"] = True
                request_scroll("results-section")  # 結果セクションへスクロール
                st.rerun()

    with st.expander("🕘 直近の経費（編集/削除）", expanded=False):
        render_recent_expenses_edit_delete(user_id, start, end, limit=3)

    # =========================================================
    # 結果セクション表示（step制で制御）
    # =========================================================
    is_guest = st.session_state.get("is_guest", False)
    current_step = st.session_state.get("step", "income")
    
    # stepが"result"の場合、または既存のshow_results_sectionフラグが立っている場合に結果を表示
    if is_guest and (current_step == "result" or st.session_state.get("show_results_section", False)):
        st.markdown("---")
        
        # ミニ結果（最上部に大きく表示）
        today = today_date()
        m_start, m_end = month_range(today)
        m_earn = load_earnings(user_id, m_start, m_end)
        m_exp = load_expenses(user_id, m_start, m_end)
        
        income = float(m_earn["円換算"].sum()) if not m_earn.empty else 0.0
        expense = float(m_exp["円換算"].sum()) if not m_exp.empty else 0.0
        profit = income - expense
        
        with st.container(border=True):
            st.markdown("### 📊 結果（今月の収支）")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("収益", yen(income), delta=None)
            with col2:
                st.metric("経費", yen(expense), delta=None)
            with col3:
                profit_color = "#2e7d32" if profit >= 0 else "#c62828"
                st.markdown(
                    f"<div style='text-align: center;'><div style='font-size: 12px; color: var(--rn-subtext); margin-bottom: 4px;'>利益</div><div style='font-size: 28px; font-weight: 900; color: {profit_color};'>{yen(profit)}</div></div>",
                    unsafe_allow_html=True
                )
            
            if profit < 0:
                st.warning("⚠️ 今月は赤字です（経費が収益を上回っています）")
            else:
                st.success("✅ 今月は黒字です")
        
        st.markdown("---")
        
        # 結果セクションのアンカーを配置（スクロールターゲット用・確実なID）
        st.markdown('<div id="results-section"></div>', unsafe_allow_html=True)
        
        # 詳細結果（今月の状況）
        st.subheader("📊 今月の状況（詳細）")
        st.caption("※ここは「今月だけ」の速報。下の「サマリー」は、左サイドバーで選んだ期間の集計です。")
        
        # 前月
        prev_last_day = m_start - timedelta(days=1)
        prev_start, prev_end = month_range(prev_last_day)
        p_earn = load_earnings(user_id, prev_start, prev_end)
        p_exp = load_expenses(user_id, prev_start, prev_end)
        prev_profit = (float(p_earn["円換算"].sum()) if not p_earn.empty else 0.0) - (float(p_exp["円換算"].sum()) if not p_exp.empty else 0.0)
        delta_profit = profit - prev_profit
        
        remain_to_goal = max(0.0, float(goal) - float(profit))
        achieve = 0.0
        if float(goal) > 0:
            achieve = max(0.0, (profit / float(goal)) * 100.0)
        
        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric("収益", yen(income))
        r1c2.metric("経費", yen(expense))
        r1c3.metric("利益", yen(profit))
        
        r2c1, r2c2 = st.columns(2)
        r2c1.metric("目標まで（利益）", yen(remain_to_goal))
        r2c2.metric("達成率（利益）", f"{int(achieve)}%")
        
        st.markdown(
            f"<div style='margin-top:-8px; font-size:15px;'>前月比：{html_delta_badge(delta_profit, prev_profit, big=True)}</div>",
            unsafe_allow_html=True,
        )
        
        st.markdown("---")
        
        # 試用完了メッセージ（控えめに）
        st.info("💡 データを保存するには、サイドバーからログイン（ユーザー名/PIN）を設定してください。")

    with st.expander("🧾 収益一覧（編集・削除）", expanded=False):
        earn_df= load_earnings(user_id, start, end)
        exp_df  = load_expenses(user_id, start, end)

        # -------------------------
        # 収益一覧
        # -------------------------
        st.markdown("##### 収益一覧")
        if earn_df.empty:
            st.info("この期間の収益データはありません。")
        else:
            st.dataframe(
                earn_df.drop(columns=["通貨コード"], errors="ignore"),
                use_container_width=True,
                hide_index=True,
            )

            earn_df2 = earn_df.copy()
            earn_df2["ID"] = earn_df2["ID"].astype(int)

            earn_labels = {
                int(r.ID): f"ID {int(r.ID)}｜{r.日付}｜{r.プラットフォーム}｜{r.カテゴリ}｜{yen(r.円換算)}"
                for r in earn_df2.itertuples(index=False)
            }
            earn_ids = list(earn_labels.keys())

            chosen_id = st.selectbox(
                "編集/削除する収益を選択",
                options=earn_ids,
                format_func=lambda x: earn_labels[x],
                key="pick_earn_id",
            )

            colA, colB = st.columns(2)
            with colA:
                if st.button("この収益を編集", key=f"btn_open_edit_earn_{chosen_id}"):
                    st.session_state["editing_earning_id"] = int(chosen_id)
                    st.rerun()
            with colB:
                if st.button("この収益を削除", key=f"btn_del_earn_{chosen_id}"):
                    delete_earning(user_id, int(chosen_id))
                    st.success("削除しました。")
                    st.rerun()
    # =========================
    # 収益編集フォーム
    # =========================
    if "editing_earning_id" in st.session_state:
        eid = st.session_state["editing_earning_id"]

        row = load_earnings(user_id, start, end)
        row = row[row["ID"] == eid]

        if row.empty:
            st.error("編集対象の収益が見つかりません")
        else:
            rr = row.iloc[0]

            st.markdown("### ✏️ 収益を編集")

            with st.container(border=True):
                c1, c2, c3, c4, c5, c6 = st.columns(6)

                # 日付
                with c1:
                    e_day = st.date_input(
                        "日付",
                        value=date.fromisoformat(rr["日付"]),
                        key=f"edit_e_day_{eid}",
                    )

                # プラットフォーム
                with c2:
                    e_platform = st.text_input(
                        "プラットフォーム",
                        value=rr["プラットフォーム"],
                        key=f"edit_e_platform_{eid}",
                    )

                # カテゴリ
                with c3:
                    e_cat = st.text_input(
                        "カテゴリ",
                        value=rr["カテゴリ"],
                        key=f"edit_e_cat_{eid}",
                    )

                # 金額（★少数点なし）
                with c4:
                    e_amt = st.number_input(
                        "金額",
                        min_value=0,
                        value=int(rr["金額"]),
                        step=1,
                        format="%d",
                        key=f"edit_e_amt_{eid}",
                    )

                # 通貨（★日本語表示）
                with c5:
                    cur_code = rr.get("通貨コード") or "JPY"
                    idx = (
                        CURRENCY_OPTIONS.index(cur_code)
                        if cur_code in CURRENCY_OPTIONS
                        else 0
                    )

                    e_cur = st.selectbox(
                        "通貨",
                        options=CURRENCY_OPTIONS,
                        index=idx,
                        format_func=currency_ja,  # ← 日本語表示
                        key=f"edit_e_cur_{eid}",
                    )

                # メモ
                with c6:
                    e_memo = st.text_input(
                        "メモ",
                        value=str(rr["メモ"] or ""),
                        key=f"edit_e_memo_{eid}",
                    )

                # ボタン
                b1, b2 = st.columns(2)

                with b1:
                    if st.button("保存（収益）", key=f"save_earn_{eid}"):
                        update_earning(
                            user_id,
                            eid,
                            e_day,
                            e_platform,
                            e_cat,
                            e_cur,
                            int(e_amt),  # 念のため int
                            e_memo,
                        )
                        st.session_state.pop("editing_earning_id")
                        st.success("収益を更新しました")
                        st.rerun()

                with b2:
                    if st.button("キャンセル", key=f"cancel_earn_{eid}"):
                        st.session_state.pop("editing_earning_id")
                        st.rerun()

        st.markdown("---")

    with st.expander("🧾 経費一覧（編集・削除）", expanded=False):
        # -------------------------
        # 経費一覧
        # -------------------------
        st.markdown("##### 経費一覧")
        if exp_df.empty:
            st.info("この期間の経費データはありません。")
        else:
            st.dataframe(
                exp_df.drop(columns=["通貨コード"], errors="ignore"),
                use_container_width=True,
                hide_index=True,
            )

            exp_df2 = exp_df.copy()
            exp_df2["ID"] = exp_df2["ID"].astype(int)

            exp_labels = {
                int(r.ID): f"ID {int(r.ID)}｜{r.日付}｜{r.支払先}｜{r.カテゴリ}｜{yen(r.円換算)}"
                for r in exp_df2.itertuples(index=False)
            }
            exp_ids = list(exp_labels.keys())

            chosen_exp_id = st.selectbox(
                "編集/削除する経費を選択",
                options=exp_ids,
                format_func=lambda x: exp_labels[x],
                key="pick_exp_id",
            )

            colA, colB = st.columns(2)
            with colA:
                if st.button("この経費を編集", key=f"btn_open_edit_exp_{chosen_exp_id}"):
                    st.session_state["editing_expense_id"] = int(chosen_exp_id)
                    st.rerun()
            with colB:
                if st.button("この経費を削除", key=f"btn_del_exp_{chosen_exp_id}"):
                    delete_expense(user_id, int(chosen_exp_id))
                    st.success("削除しました。")
                    st.rerun()

    # =========================
    # 経費編集フォーム（←expander の外！！）
    # =========================
    if "editing_expense_id" in st.session_state:
        eid = st.session_state["editing_expense_id"]

        row = load_expenses(user_id, start, end)
        row = row[row["ID"] == eid]

        if row.empty:
            st.error("編集対象の経費が見つかりません")
        else:
            rr = row.iloc[0]
            st.markdown("### ✏️ 経費を編集")

            with st.container(border=True):
                c1, c2, c3, c4, c5, c6 = st.columns(6)

                with c1:
                    e_day = st.date_input("日付", value=date.fromisoformat(rr["日付"]), key=f"exp_day_{eid}")
                with c2:
                    e_vendor = st.text_input("支払先", value=rr["支払先"], key=f"exp_vendor_{eid}")
                with c3:
                    e_cat = st.text_input("カテゴリ", value=rr["カテゴリ"], key=f"exp_cat_{eid}")
                with c4:
                    e_amt = st.number_input(
                        "金額",
                        min_value=0,
                        value=int(rr["金額"]),
                        step=1,
                        format="%d",
                        key=f"exp_amt_{eid}",
                    )
                with c5:
                    cur_code = (rr.get("通貨コード") or "JPY")
                    idx = CURRENCY_OPTIONS.index(cur_code) if cur_code in CURRENCY_OPTIONS else 0
                    e_cur = st.selectbox(
                        "通貨",
                        options=CURRENCY_OPTIONS,
                        index=idx,
                        format_func=currency_ja,
                        key=f"exp_cur_{eid}",
                    )
                with c6:
                    e_memo = st.text_input("メモ", value=str(rr["メモ"] or ""), key=f"exp_memo_{eid}")

                b1, b2 = st.columns(2)
                with b1:
                    if st.button("保存（経費）", key=f"exp_save_{eid}"):
                        update_expense(user_id, eid, e_day, e_vendor, e_cat, e_cur, e_amt, e_memo)
                        st.session_state.pop("editing_expense_id", None)
                        st.success("更新しました")
                        st.rerun()
                with b2:
                    if st.button("キャンセル", key=f"exp_cancel_{eid}"):
                        st.session_state.pop("editing_expense_id", None)
                        st.rerun()
    st.markdown("---")

    # -------------------------
    # 今月の状況（矢印・色を自前HTMLで確実に）
    # ゲストユーザーで結果セクションを既に表示している場合はスキップ
    # -------------------------
    current_step_for_results = st.session_state.get("step", "income")
    show_results_section = st.session_state.get("show_results_section", False)
    if not (is_guest and (current_step_for_results == "result" or show_results_section)):
        st.caption("※ここは「今月だけ」の速報。下の「サマリー」は、左サイドバーで選んだ期間の集計です。")

        today = today_date()
        m_start, m_end = month_range(today)
        m_earn = load_earnings(user_id, m_start, m_end)
        m_exp = load_expenses(user_id, m_start, m_end)

        income = float(m_earn["円換算"].sum()) if not m_earn.empty else 0.0
        expense = float(m_exp["円換算"].sum()) if not m_exp.empty else 0.0
        profit = income - expense

        # 前月
        prev_last_day = m_start - timedelta(days=1)
        prev_start, prev_end = month_range(prev_last_day)
        p_earn = load_earnings(user_id, prev_start, prev_end)
        p_exp = load_expenses(user_id, prev_start, prev_end)
        prev_profit = (float(p_earn["円換算"].sum()) if not p_earn.empty else 0.0) - (float(p_exp["円換算"].sum()) if not p_exp.empty else 0.0)

        delta_profit = profit - prev_profit

        remain_to_goal = max(0.0, float(goal) - float(profit))
        achieve = 0.0
        if float(goal) > 0:
            achieve = max(0.0, (profit / float(goal)) * 100.0)

        r1c1, r1c2, r1c3 = st.columns(3)
        r1c1.metric("収益", yen(income))
        r1c2.metric("経費", yen(expense))
        r1c3.metric("利益", yen(profit))

        r2c1, r2c2 = st.columns(2)
        r2c1.metric("目標まで（利益）", yen(remain_to_goal))
        r2c2.metric("達成率（利益）", f"{int(achieve)}%")

        st.markdown(
            f"<div style='margin-top:-8px; font-size:15px;'>前月比：{html_delta_badge(delta_profit, prev_profit, big=True)}</div>",
            unsafe_allow_html=True,
        )

        if profit < 0:
            st.warning("⚠️ 今月は赤字です（経費が収益を上回っています）")
        else:
            st.success("✅ 今月は黒字です")

        st.markdown("---")


    # -------------------------
    # サマリー（選択した期間）
    # -------------------------
    st.subheader("📌 サマリー（選択した期間）")
    st.caption("※左サイドバーで選んだ期間（今月/先月/直近30日/カスタム）の集計です。")

    earn_df = load_earnings(user_id, start, end)
    exp_df = load_expenses(user_id, start, end)

    period_income = float(earn_df["円換算"].sum()) if not earn_df.empty else 0.0
    period_expense = float(exp_df["円換算"].sum()) if not exp_df.empty else 0.0
    period_profit = period_income - period_expense

    s1, s2, s3, s4 = st.columns(4)
    s1.metric("期間内 収益（円）", yen(period_income))
    s2.metric("期間内 経費（円）", yen(period_expense))
    s3.metric("期間内 利益（円）", yen(period_profit))
    s4.metric("固定費（設定・円）", yen(fixed))

    st.markdown("---")

    # 資産
    render_assets_section(user_id)

    st.markdown("---")

    # AI（分析＋自由質問）
    render_ai_section(user_id, goal, fixed, user_key)

    st.success("狙い：入力→編集/削除→可視化→AI提案が1画面で回る")

    # =========================================================
    # ページ末尾でスクロール要求があれば1回だけ実行
    # =========================================================
    perform_scroll_if_requested()


# =========================================================
# 見た目（字が薄い問題＆直近資産の文字サイズUP）
# =========================================================
def inject_css():
    st.markdown(
        """
<style>
/* =========================
   1) ライト / ダーク 自動追従
   ========================= */
@media (prefers-color-scheme: light) {
  :root{
    --rn-text: rgba(0,0,0,0.88);
    --rn-subtext: rgba(0,0,0,0.72);
    --rn-border: rgba(0,0,0,0.10);
    --rn-card: rgba(250,250,250,0.92);
  }
}
@media (prefers-color-scheme: dark) {
  :root{
    --rn-text: rgba(255,255,255,0.92);
    --rn-subtext: rgba(255,255,255,0.75);
    --rn-border: rgba(255,255,255,0.14);
    --rn-card: rgba(255,255,255,0.06);
  }
}

/* =========================
   2) “薄い”だけ直す（黒固定しない）
   ========================= */
.stMarkdown, .stMarkdown * {
  color: var(--rn-text) !important;
  opacity: 1 !important;
}

label, label span {
  color: var(--rn-subtext) !important;
  opacity: 1 !important;
  font-weight: 650 !important;
}

.stCaption, .stCaption * ,
div[data-testid="stCaptionContainer"],
div[data-testid="stCaptionContainer"] * {
  color: var(--rn-subtext) !important;
  opacity: 1 !important;
  font-weight: 600 !important;
}

div[data-testid="stAlert"] * {
  color: var(--rn-text) !important;
  font-weight: 650 !important;
  opacity: 1 !important;
}

div[data-testid="stMetricLabel"] {
  color: var(--rn-subtext) !important;
  font-weight: 800 !important;
  opacity: 1 !important;
}
div[data-testid="stMetricValue"] {
  color: var(--rn-text) !important;
  font-weight: 900 !important;
  opacity: 1 !important;
}

details summary, details * {
  color: var(--rn-text) !important;
  opacity: 1 !important;
}

/* selectbox文字の薄さ対策 */
div[role="combobox"] * {
  color: var(--rn-text) !important;
  opacity: 1 !important;
}

/* 入力欄高さ（任意） */
div[data-testid="stNumberInput"] input,
div[data-testid="stTextInput"] input,
div[data-testid="stDateInput"] input,
div[data-testid="stSelectbox"] div[role="combobox"] {
  min-height: 40px !important;
  height: 40px !important;
  padding-top: 6px !important;
  padding-bottom: 6px !important;
}

/* ---- 直近資産カード ---- */
.asset-recent-block{
  margin-top: 6px;
  padding: 10px 12px;
  border-radius: 10px;
  border: 1px solid var(--rn-border);
  background: var(--rn-card);
}
.asset-recent-line{
  font-size: 16px;
  font-weight: 750;
  line-height: 1.6;
}
.asset-recent-delta{
  margin-top: 6px;
  font-size: 15px;
  font-weight: 800;
  line-height: 1.8;
}
</style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# メイン（未定義エラー回避：ここで全部そろっている前提）
# =========================================================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    inject_css()
    init_db_and_migrate()

    user_id = st.session_state.get("user_id", None)
    if not user_id:
        # サイドバー：ログイン（目立たない位置づけ）
        with st.sidebar:
            with st.expander("🔐 ログイン（既存ユーザー）", expanded=False):
                render_login(in_sidebar=False)  # expander内なのでsidebar=False
        
        # メイン画面：ヒーロー領域（価値提案＋CTA）
        st.markdown(f"# {APP_TITLE}")
        
        # サブコピー
        st.markdown(
            """
            <div style='font-size: 20px; font-weight: 500; color: var(--rn-subtext); margin-top: -8px; margin-bottom: 24px; line-height: 1.6;'>
            収支・副業・SNS収益を "次にやる一手" まで可視化
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # ベネフィット箇条書き
        st.markdown(
            """
            <div style='margin-bottom: 32px;'>
            <ul style='list-style: none; padding-left: 0;'>
            <li style='margin-bottom: 12px; font-size: 16px;'>✓ 収入/支出を一瞬で整理</li>
            <li style='margin-bottom: 12px; font-size: 16px;'>✓ ムダをAIが1行で指摘</li>
            <li style='margin-bottom: 12px; font-size: 16px;'>✓ 改善アクションが分かる</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # CTA（1つだけ、強調）
        col_cta, _ = st.columns([0.4, 0.6])
        with col_cta:
            if st.button("今すぐ分析する", type="primary", use_container_width=True):
                import random
                import string
                # ゲストユーザー名を自動生成
                guest_username = f"guest_{''.join(random.choices(string.ascii_lowercase + string.digits, k=8))}"
                guest_pin = "1234"  # 簡単なPIN
                try:
                    uid = create_user(guest_username, guest_pin)
                    st.session_state["user_id"] = int(uid)
                    st.session_state["username"] = guest_username
                    st.session_state["is_guest"] = True  # ゲストフラグ
                    st.session_state["onboarding_step"] = 1  # オンボーディング開始
                    st.rerun()
                except Exception as e:
                    st.error(f"試用開始に失敗しました：{e}")
        
        # 補足（小さく）
        st.markdown(
            """
            <div style='margin-top: 16px; font-size: 13px; color: var(--rn-subtext);'>
            登録は後でOK / データは外部公開されません
            </div>
            """,
            unsafe_allow_html=True
        )
        
        return

    start, end, goal, fixed, user_key = render_sidebar_after_login(int(user_id))
    render_main(int(user_id), start, end, goal, fixed, user_key)


if __name__ == "__main__":
    main()
