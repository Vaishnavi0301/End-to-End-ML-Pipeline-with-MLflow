# api/main.py
"""
Fraud Detection API — FastAPI
──────────────────────────────
Endpoints:
    POST /predict          — Single transaction prediction (rate limited, cached)
    POST /predict-batch    — Batch prediction (up to 100 transactions)
    GET  /explain          — SHAP feature importance for a transaction
    GET  /health           — API + model health
    GET  /model-info       — Production model metrics from MLflow
    GET  /threshold-info   — Precision-recall tradeoff explanation
    GET  /monitoring/stats — Inference log aggregate stats
    GET  /monitoring/recent — Recent predictions
    GET  /drift-report     — Live drift detection on recent inference data
"""

from api.inference_logger import InferenceLogger
import os
import sys
import json
import time
import hashlib
import pickle
import numpy as np
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

# ── Rate Limiter ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

# ── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fraud Detection API",
    description="Real-time credit card fraud detection — MLflow · SHAP · Drift Detection",
    version="2.0.0"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
SCALERS_PATH = os.getenv("SCALERS_PATH", "scalers.pkl")
REFERENCE_STATS_PATH = os.getenv(
    "REFERENCE_STATS_PATH", "reference_stats.json")
INFERENCE_DB_PATH = os.getenv("INFERENCE_DB_PATH", "inference_logs.db")

mlflow.set_tracking_uri(MLFLOW_URI)

# ── Prediction Cache ──────────────────────────────────────────────────────────
# In-memory LRU-style cache. Key = MD5 of (features, amount, time).
# TTL = 5 minutes. Not suitable for features that change meaning over time
# (e.g. time-sensitive fraud patterns) — kept short intentionally.
_prediction_cache: dict[str, tuple] = {}   # key → (result_dict, expires_at)
CACHE_TTL_SECONDS = 300


def _cache_key(features: list, amount: float, t: float) -> str:
    raw = f"{features}|{round(amount, 2)}|{round(t, 2)}"
    return hashlib.md5(raw.encode()).hexdigest()


def _cache_get(key: str) -> Optional[dict]:
    entry = _prediction_cache.get(key)
    if entry and time.time() < entry[1]:
        return entry[0]
    _prediction_cache.pop(key, None)
    return None


def _cache_set(key: str, value: dict):
    _prediction_cache[key] = (value, time.time() + CACHE_TTL_SECONDS)
    # Evict expired entries if cache grows large
    if len(_prediction_cache) > 1000:
        now = time.time()
        expired = [k for k, (_, exp) in _prediction_cache.items() if now > exp]
        for k in expired:
            _prediction_cache.pop(k, None)


# ── Load Model + Scalers at Startup ──────────────────────────────────────────
try:
    model = mlflow.sklearn.load_model("models:/fraud-detector@production")
    scalers = pickle.load(open(SCALERS_PATH, "rb"))
    amount_scaler = scalers["amount"]
    time_scaler = scalers["time"]
    MODEL_LOADED = True
    print("✅ Model and scalers loaded.")
except Exception as e:
    MODEL_LOADED = False
    print(f"❌ Model load failed: {e}")

# ── SHAP Explainer (lazy init) ────────────────────────────────────────────────
_shap_explainer = None


def get_shap_explainer():
    global _shap_explainer
    if _shap_explainer is None and MODEL_LOADED:
        try:
            import shap
            _shap_explainer = shap.TreeExplainer(model)
            print("SHAP explainer ready.")
        except Exception as e:
            print(f"SHAP init failed: {e}")
    return _shap_explainer


# ── Inference Logger ──────────────────────────────────────────────────────────
inference_logger = InferenceLogger(db_path=INFERENCE_DB_PATH)

# ── Feature Names ─────────────────────────────────────────────────────────────
FEATURE_NAMES = [f'V{i}' for i in range(
    1, 29)] + ['Amount_scaled', 'Time_scaled']


# ── Schemas ───────────────────────────────────────────────────────────────────
class TransactionRequest(BaseModel):
    features: list[float]   # V1–V28, exactly 28 values
    amount:   float
    time:     float

    model_config = {
        "json_schema_extra": {
            "example": {
                "features": [
                    -1.3598, -0.0728, 2.5363, 1.3782, -0.3383,
                    0.4624,  0.2396, 0.0987, 0.3638, 0.0908,
                    -0.5516, -0.6178, -0.9914, -0.3112, 1.4682,
                    -0.4704,  0.2080, 0.0258, 0.4040, 0.2514,
                    -0.0183,  0.2778, -0.1105, 0.0669, 0.1285,
                    -0.1891,  0.1336, -0.0211
                ],
                "amount": 149.62,
                "time":   0.0
            }
        }
    }


class FraudResponse(BaseModel):
    transaction_id:    str
    is_fraud:          bool
    fraud_probability: float
    risk_level:        str
    latency_ms:        float
    model_version:     str
    cache_hit:         bool


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest]


class HealthResponse(BaseModel):
    status:       str
    model_loaded: bool
    mlflow_uri:   str
    cache_size:   int
    total_predictions: int


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_risk_level(proba: float) -> str:
    if proba < 0.3:
        return "LOW"
    elif proba < 0.7:
        return "MEDIUM"
    return "HIGH"


def get_production_version() -> str:
    try:
        client = MlflowClient()
        v = client.get_model_version_by_alias("fraud-detector", "production")
        return f"v{v.version}"
    except Exception:
        return "unknown"


def build_feature_vector(transaction: TransactionRequest) -> np.ndarray:
    """Scale amount + time and concatenate with V1–V28."""
    amount_scaled = amount_scaler.transform([[transaction.amount]])[0][0]
    time_scaled = time_scaler.transform([[transaction.time]])[0][0]
    return np.array(transaction.features + [amount_scaled, time_scaled]).reshape(1, -1)


def _predict_one(transaction: TransactionRequest, model_version: str) -> dict:
    """Core prediction logic — used by both /predict and /predict-batch."""
    fv = build_feature_vector(transaction)
    pred = model.predict(fv)[0]
    proba = model.predict_proba(fv)[0][1]

    return {
        "is_fraud":          bool(pred == 1),
        "fraud_probability": round(float(proba), 4),
        "risk_level":        get_risk_level(proba),
        "model_version":     model_version,
        "feature_vector":    fv[0].tolist(),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Fraud Detection API v2 — visit /docs"}


@app.get("/health", response_model=HealthResponse)
async def health():
    stats = inference_logger.get_stats()
    return HealthResponse(
        status="ok" if MODEL_LOADED else "degraded",
        model_loaded=MODEL_LOADED,
        mlflow_uri=MLFLOW_URI,
        cache_size=len(_prediction_cache),
        total_predictions=stats.get("total_predictions", 0) or 0,
    )


@app.post("/predict", response_model=FraudResponse)
@limiter.limit("30/minute")
async def predict(request: Request, transaction: TransactionRequest):
    """
    Single transaction fraud prediction.
    Rate limited to 30 req/min per IP.
    Repeated identical inputs are served from cache (TTL 5 min).
    """
    if not MODEL_LOADED:
        raise HTTPException(503, "Model not loaded.")
    if len(transaction.features) != 28:
        raise HTTPException(
            422, f"Expected 28 features, got {len(transaction.features)}")

    # ── Cache check ───────────────────────────────────────────────────────
    cache_key = _cache_key(transaction.features,
                           transaction.amount, transaction.time)
    cached = _cache_get(cache_key)
    if cached:
        return FraudResponse(**cached, cache_hit=True)

    # ── Inference ─────────────────────────────────────────────────────────
    start = time.time()
    model_version = get_production_version()
    result = _predict_one(transaction, model_version)
    latency = round((time.time() - start) * 1000, 2)
    txn_id = f"txn_{int(time.time() * 1000)}"

    # ── Log to SQLite ─────────────────────────────────────────────────────
    inference_logger.log(
        transaction_id=txn_id,
        features=result["feature_vector"],
        amount=transaction.amount,
        fraud_probability=result["fraud_probability"],
        is_fraud=result["is_fraud"],
        risk_level=result["risk_level"],
        latency_ms=latency,
        model_version=model_version,
    )

    response_data = {
        "transaction_id":    txn_id,
        "is_fraud":          result["is_fraud"],
        "fraud_probability": result["fraud_probability"],
        "risk_level":        result["risk_level"],
        "latency_ms":        latency,
        "model_version":     model_version,
    }

    _cache_set(cache_key, response_data)
    return FraudResponse(**response_data, cache_hit=False)


@app.post("/predict-batch")
@limiter.limit("10/minute")
async def predict_batch(request: Request, batch: BatchRequest):
    """
    Batch prediction for up to 100 transactions.
    Returns list of results in the same order as input.
    """
    if not MODEL_LOADED:
        raise HTTPException(503, "Model not loaded.")
    if len(batch.transactions) > 100:
        raise HTTPException(422, "Batch size limit is 100 transactions.")
    if len(batch.transactions) == 0:
        raise HTTPException(422, "Batch is empty.")

    model_version = get_production_version()
    results = []
    start_total = time.time()

    for txn in batch.transactions:
        if len(txn.features) != 28:
            results.append(
                {"error": f"Expected 28 features, got {len(txn.features)}"})
            continue

        start = time.time()
        result = _predict_one(txn, model_version)
        latency = round((time.time() - start) * 1000, 2)
        txn_id = f"txn_{int(time.time() * 1000)}"

        inference_logger.log(
            transaction_id=txn_id,
            features=result["feature_vector"],
            amount=txn.amount,
            fraud_probability=result["fraud_probability"],
            is_fraud=result["is_fraud"],
            risk_level=result["risk_level"],
            latency_ms=latency,
            model_version=model_version,
        )

        results.append({
            "transaction_id":    txn_id,
            "is_fraud":          result["is_fraud"],
            "fraud_probability": result["fraud_probability"],
            "risk_level":        result["risk_level"],
            "latency_ms":        latency,
        })

    total_latency = round((time.time() - start_total) * 1000, 2)
    return {
        "model_version":  model_version,
        "total_latency_ms": total_latency,
        "batch_size":     len(batch.transactions),
        "results":        results,
    }


@app.post("/explain")
@limiter.limit("10/minute")
async def explain(request: Request, transaction: TransactionRequest):
    """
    SHAP feature importance for a single prediction.
    Shows which features most influenced the fraud decision.
    TreeExplainer is used — no background dataset needed.
    """
    if not MODEL_LOADED:
        raise HTTPException(503, "Model not loaded.")
    if len(transaction.features) != 28:
        raise HTTPException(
            422, f"Expected 28 features, got {len(transaction.features)}")

    explainer = get_shap_explainer()
    if explainer is None:
        raise HTTPException(
            503, "SHAP explainer not available. Install shap package.")

    try:
        fv = build_feature_vector(transaction)
        shap_values = explainer.shap_values(fv)

        # For RandomForestClassifier shap_values is a list [class0, class1]
        # We want class 1 (fraud) SHAP values
        fraud_shap = (
            shap_values[1][0]
            if isinstance(shap_values, list)
            else shap_values[0]
        )

        base_value = (
            explainer.expected_value[1]
            if hasattr(explainer.expected_value, '__len__')
            else float(explainer.expected_value)
        )

        # Build sorted dict by absolute impact
        shap_dict = {
            name: round(float(val), 5)
            for name, val in zip(FEATURE_NAMES, fraud_shap)
        }
        top_features = sorted(
            shap_dict.items(), key=lambda x: abs(x[1]), reverse=True
        )

        # Also run the prediction so we return full context
        pred = model.predict(fv)[0]
        proba = model.predict_proba(fv)[0][1]

        return {
            "fraud_probability": round(float(proba), 4),
            "is_fraud":          bool(pred == 1),
            "base_value":        round(base_value, 5),
            "shap_sum":          round(float(np.sum(fraud_shap)), 5),
            "top_10_features": [
                {"feature": k, "shap_value": v,
                    "direction": "↑fraud" if v > 0 else "↓fraud"}
                for k, v in top_features[:10]
            ],
            "all_shap_values": shap_dict,
        }
    except Exception as e:
        raise HTTPException(500, f"SHAP computation failed: {e}")


@app.get("/model-info")
async def model_info():
    if not MODEL_LOADED:
        raise HTTPException(503, "Model not loaded")
    client = MlflowClient()
    try:
        latest = client.get_model_version_by_alias(
            "fraud-detector", "production")
        run = client.get_run(latest.run_id)
        return {
            "model_name":        "fraud-detector",
            "version":           latest.version,
            "stage":             "Production",
            "sampling_strategy": run.data.params.get("sampling_strategy"),
            "model_type":        run.data.params.get("model_type"),
            "metrics": {
                "recall_fraud":    run.data.metrics.get("recall_fraud"),
                "precision_fraud": run.data.metrics.get("precision_fraud"),
                "roc_auc":         run.data.metrics.get("roc_auc"),
                "fraud_caught":    run.data.metrics.get("fraud_caught"),
                "fraud_missed":    run.data.metrics.get("fraud_missed"),
            }
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/threshold-info")
async def threshold_info():
    return {
        "current_threshold": 0.5,
        "model_recall":      0.8673,
        "model_precision":   0.7658,
        "explanation": (
            "Lowering threshold catches more fraud (higher recall) "
            "but increases false alarms (lower precision). "
            "For fraud detection, recall is prioritised — "
            "missing real fraud is more costly than a false alarm."
        ),
        "tradeoffs": {
            "threshold_0.3": "Higher recall, more false alarms — use when fraud cost is very high",
            "threshold_0.5": "Current default — balanced",
            "threshold_0.7": "Fewer false alarms, misses more fraud — use when review capacity is limited"
        }
    }


@app.get("/monitoring/stats")
async def monitoring_stats():
    """Aggregate inference stats — used by the Streamlit monitoring dashboard."""
    return inference_logger.get_stats()


@app.get("/monitoring/recent")
async def monitoring_recent(limit: int = 100):
    """Recent N predictions. Default 100, max 1000."""
    limit = min(limit, 1000)
    return {"predictions": inference_logger.get_recent(limit=limit)}


@app.get("/drift-report")
async def drift_report():
    """
    Run drift detection on the most recent 500 inference records.
    Returns PSI, KS stats per feature, and an overall should_retrain flag.
    """
    try:
        from drift_detector import DriftDetector
    except ImportError:
        raise HTTPException(
            503, "drift_detector module not found. Check sys.path.")

    if not os.path.exists(REFERENCE_STATS_PATH):
        raise HTTPException(
            404, "No reference_stats.json found. Run train.py first.")

    # Check for a pre-computed drift report from retrain.py
    cached_report_path = "latest_drift_report.json"
    if os.path.exists(cached_report_path):
        with open(cached_report_path) as f:
            cached = json.load(f)
        # Return cached if less than 5 minutes old
        report_time = cached.get("timestamp", "")
        try:
            from datetime import datetime
            rt = datetime.strptime(report_time, '%Y-%m-%d %H:%M:%S')
            if (datetime.now() - rt).seconds < 300:
                cached["source"] = "cached_from_retrain_scheduler"
                return cached
        except Exception:
            pass

    # Compute fresh
    features_data = inference_logger.get_features_for_drift(limit=500)
    if len(features_data) < 50:
        return {
            "status":  "insufficient_data",
            "message": f"Only {len(features_data)} inference records. Need ≥50 for drift analysis.",
            "should_retrain": False,
        }

    fraud_rate = inference_logger.get_fraud_rate(limit=500)
    X = np.array(features_data)

    detector = DriftDetector(REFERENCE_STATS_PATH)
    report = detector.full_report(X, fraud_rate or 0.0)
    report["source"] = "live_computation"
    report["n_records_analyzed"] = len(features_data)

    return report
