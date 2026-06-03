# api/inference_logger.py
"""
Inference Logger — SQLite
──────────────────────────
Logs every prediction made by the API for:
  • monitoring (latency, fraud rate trends, confidence distribution)
  • drift detection (feature distributions over time)
  • audit trail (transaction_id, model version, timestamp)

Schema:
    inference_logs(
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp      TEXT,
        transaction_id TEXT,
        fraud_probability REAL,
        is_fraud       INTEGER,
        risk_level     TEXT,
        latency_ms     REAL,
        model_version  TEXT,
        amount         REAL,
        features_json  TEXT     -- JSON array of 30 floats
    )
"""

import sqlite3
import json
import os
from datetime import datetime, timezone
from typing import Optional


class InferenceLogger:
    def __init__(self, db_path: str = "inference_logs.db"):
        self.db_path = db_path
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS inference_logs (
                    id                INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp         TEXT NOT NULL,
                    transaction_id    TEXT NOT NULL,
                    fraud_probability REAL NOT NULL,
                    is_fraud          INTEGER NOT NULL,
                    risk_level        TEXT NOT NULL,
                    latency_ms        REAL NOT NULL,
                    model_version     TEXT NOT NULL,
                    amount            REAL NOT NULL,
                    features_json     TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp
                ON inference_logs(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_is_fraud
                ON inference_logs(is_fraud)
            """)
            conn.commit()

    # ── Write ─────────────────────────────────────────────────────────────────
    def log(
        self,
        transaction_id:    str,
        features:          list,
        amount:            float,
        fraud_probability: float,
        is_fraud:          bool,
        risk_level:        str,
        latency_ms:        float,
        model_version:     str,
    ):
        """Insert one prediction record."""
        try:
            with self._connect() as conn:
                conn.execute(
                    """INSERT INTO inference_logs
                       (timestamp, transaction_id, fraud_probability, is_fraud,
                        risk_level, latency_ms, model_version, amount, features_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        datetime.now(timezone.utc).isoformat(),
                        transaction_id,
                        float(fraud_probability),
                        int(is_fraud),
                        risk_level,
                        float(latency_ms),
                        model_version,
                        float(amount),
                        json.dumps(features),
                    )
                )
                conn.commit()
        except Exception as e:
            # Logging must never crash the prediction endpoint
            print(f"[InferenceLogger] write error: {e}")

    # ── Read ──────────────────────────────────────────────────────────────────
    def get_recent(self, limit: int = 100) -> list[dict]:
        """Return the most recent N predictions (newest first)."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT id, timestamp, transaction_id, fraud_probability,
                          is_fraud, risk_level, latency_ms, model_version, amount
                   FROM inference_logs
                   ORDER BY id DESC LIMIT ?""",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_stats(self) -> dict:
        """Aggregate stats for the monitoring dashboard."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*)                            AS total_predictions,
                    SUM(is_fraud)                       AS total_fraud_flagged,
                    AVG(fraud_probability)              AS avg_fraud_probability,
                    AVG(latency_ms)                     AS avg_latency_ms,
                    MIN(latency_ms)                     AS min_latency_ms,
                    MAX(latency_ms)                     AS max_latency_ms,
                    MIN(timestamp)                      AS first_prediction,
                    MAX(timestamp)                      AS last_prediction
                FROM inference_logs
            """).fetchone()

            # Hourly fraud rate trend (last 24 hours)
            trend = conn.execute("""
                SELECT
                    strftime('%Y-%m-%d %H:00', timestamp) AS hour,
                    COUNT(*)                               AS total,
                    SUM(is_fraud)                          AS frauds,
                    AVG(latency_ms)                        AS avg_latency,
                    AVG(fraud_probability)                 AS avg_proba
                FROM inference_logs
                WHERE timestamp >= datetime('now', '-24 hours')
                GROUP BY hour
                ORDER BY hour
            """).fetchall()

            # Confidence distribution buckets
            buckets = conn.execute("""
                SELECT
                    CAST(fraud_probability * 10 AS INTEGER) * 10 AS bucket_start,
                    COUNT(*) AS count
                FROM inference_logs
                GROUP BY bucket_start
                ORDER BY bucket_start
            """).fetchall()

            # Model version breakdown
            versions = conn.execute("""
                SELECT model_version, COUNT(*) AS count
                FROM inference_logs
                GROUP BY model_version
                ORDER BY count DESC
            """).fetchall()

        stats = dict(row) if row else {}
        stats["fraud_rate"] = (
            stats["total_fraud_flagged"] / stats["total_predictions"]
            if stats.get("total_predictions", 0) > 0 else 0.0
        )
        stats["hourly_trend"] = [dict(r) for r in trend]
        stats["confidence_distribution"] = [dict(r) for r in buckets]
        stats["model_versions"] = [dict(r) for r in versions]

        return stats

    def get_features_for_drift(self, limit: int = 500) -> list[list]:
        """Return last N feature vectors (for drift detection in retrain.py)."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT features_json FROM inference_logs ORDER BY id DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [json.loads(r["features_json"]) for r in rows]

    def get_fraud_rate(self, limit: int = 500) -> Optional[float]:
        """Current fraud rate over the last N predictions."""
        with self._connect() as conn:
            row = conn.execute(
                """SELECT AVG(is_fraud) AS rate
                   FROM (SELECT is_fraud FROM inference_logs ORDER BY id DESC LIMIT ?)""",
                (limit,)
            ).fetchone()
        return row["rate"] if row and row["rate"] is not None else None
