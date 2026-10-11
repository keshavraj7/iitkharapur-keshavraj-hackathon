import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from app.schemas.signal import RiskSignal


DB_PATH = Path(__file__).resolve().parent / "risk_signals.sqlite3"


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(str(DB_PATH), timeout=10)

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db() -> None:
    with _connect() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS risk_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signal_id TEXT NOT NULL UNIQUE,
                event_time TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)

        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_risk_signals_event_time
            ON risk_signals(event_time)
        """)


def _serialize(signal: RiskSignal) -> str:
    if hasattr(signal, "model_dump_json"):
        return signal.model_dump_json()

    return signal.json()


def _deserialize(payload: str) -> RiskSignal:
    if hasattr(RiskSignal, "model_validate_json"):
        return RiskSignal.model_validate_json(payload)

    return RiskSignal.parse_raw(payload)


def insert_signal(signal: RiskSignal) -> bool:
    """Insert only if this signal ID does not already exist."""
    with _connect() as connection:
        cursor = connection.execute(
            """
            INSERT OR IGNORE INTO risk_signals
                (signal_id, event_time, payload)
            VALUES (?, ?, ?)
            """,
            (
                signal.signal_id,
                signal.event_time.isoformat(),
                _serialize(signal),
            ),
        )

        return cursor.rowcount == 1


def get_signal(signal_id: str) -> RiskSignal | None:
    with _connect() as connection:
        row = connection.execute(
            """
            SELECT payload
            FROM risk_signals
            WHERE signal_id = ?
            """,
            (signal_id,),
        ).fetchone()

    return _deserialize(row[0]) if row else None


def list_signals(limit: int = 50) -> list[RiskSignal]:
    with _connect() as connection:
        rows = connection.execute(
            """
            SELECT payload
            FROM risk_signals
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [_deserialize(row[0]) for row in rows]


init_db()
