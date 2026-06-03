# src/retrain.py
"""
Scheduled + Drift-Triggered Retraining
────────────────────────────────────────
Runs in two modes:
  1. Scheduled    — every 24 hours regardless of drift
  2. Drift-triggered — immediately if feature or prediction drift is detected
     in the last N inference records

Run:
    python src/retrain.py
"""

from drift_detector import DriftDetector
from train import train_and_log
import os
import sys
import json
import time
import numpy as np
import schedule
import mlflow
from mlflow.tracking import MlflowClient

sys.path.append(os.path.dirname(__file__))

mlflow.set_tracking_uri("http://localhost:5000")

DATA_PATH = os.path.join(os.path.dirname(
    __file__), "..", "data", "creditcard.csv")
INFERENCE_DB_PATH = os.getenv("INFERENCE_DB_PATH", "inference_logs.db")
REFERENCE_STATS_PATH = "reference_stats.json"
DRIFT_WINDOW_SIZE = 500    # number of recent predictions to check for drift
DRIFT_CHECK_INTERVAL = 60     # seconds between drift checks (1 min)

BEST_CONFIG = {
    "run_name":     "scheduled-retrain",
    "model":        "RandomForest",
    "sampling":     "smote",
    "n_estimators": 100,
    "max_depth":    10,
    "class_weight": None,
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_production_metrics():
    """Fetch current production model recall + AUC. Returns (recall, auc, version)."""
    client = MlflowClient()
    try:
        # FIX: use alias-based lookup — get_latest_versions(stages=) is deprecated in MLflow 3.x
        mv = client.get_model_version_by_alias("fraud-detector", "production")
        run = client.get_run(mv.run_id)
        recall = float(run.data.metrics["recall_fraud"])
        auc = float(run.data.metrics["roc_auc"])
        return recall, auc, mv.version
    except Exception as e:
        print(f"  No production model found: {e}")
        return 0.0, 0.0, None


def promote_to_production(client: MlflowClient):
    """Promote the latest registered version to the production alias."""
    try:
        # FIX: use set_registered_model_alias — transition_model_version_stage is deprecated
        all_versions = client.search_model_versions("name='fraud-detector'")
        if not all_versions:
            print("  No versions found to promote.")
            return
        latest = max(all_versions, key=lambda v: int(v.version))
        client.set_registered_model_alias(
            "fraud-detector", "production", str(latest.version))
        print(f"  Promoted v{latest.version} → @production alias")
    except Exception as e:
        print(f"  Promotion failed: {e}")


def load_recent_inference_data(n: int = DRIFT_WINDOW_SIZE):
    """
    Load the last N predictions from the SQLite inference log.
    Returns (feature_matrix, fraud_rate) or (None, None) if not enough data.
    """
    try:
        import sqlite3
        import json as _json

        if not os.path.exists(INFERENCE_DB_PATH):
            return None, None

        conn = sqlite3.connect(INFERENCE_DB_PATH)
        rows = conn.execute(
            "SELECT features_json, is_fraud FROM inference_logs ORDER BY id DESC LIMIT ?",
            (n,)
        ).fetchall()
        conn.close()

        if len(rows) < 50:   # not enough data to assess drift reliably
            print(
                f"  Only {len(rows)} inference records — need ≥50 for drift check. Skipping.")
            return None, None

        feature_rows = [_json.loads(r[0]) for r in rows]
        is_fraud_arr = [r[1] for r in rows]

        X = np.array(feature_rows)
        fraud_rate = float(np.mean(is_fraud_arr))
        return X, fraud_rate

    except Exception as e:
        print(f"  Could not load inference data for drift check: {e}")
        return None, None


# ── Drift Check ───────────────────────────────────────────────────────────────
def check_drift() -> bool:
    """
    Run drift detection on recent inference data.
    Returns True if retraining should be triggered immediately.
    """
    print(f"\n  [Drift Check] {time.strftime('%Y-%m-%d %H:%M:%S')}")

    X_recent, fraud_rate = load_recent_inference_data()
    if X_recent is None:
        print("  Insufficient inference data — drift check skipped.")
        return False

    detector = DriftDetector(REFERENCE_STATS_PATH)
    if not detector.is_ready():
        print("  No reference stats — drift check skipped.")
        return False

    report = detector.full_report(X_recent, fraud_rate)

    print(
        f"  Feature drift:    {report['feature_drift']['overall_drift_detected']}")
    print(
        f"  Prediction drift: {report['prediction_drift']['drift_detected']}")
    print(
        f"  Mean PSI:         {report['feature_drift'].get('mean_psi', 'N/A')}")
    print(f"  Current fraud %:  {fraud_rate*100:.3f}%")
    print(f"  Trigger reason:   {report['trigger_reason']}")

    # Save drift report for the dashboard
    with open("latest_drift_report.json", "w") as f:
        json.dump({
            "timestamp":        time.strftime('%Y-%m-%d %H:%M:%S'),
            "should_retrain":   report["should_retrain"],
            "trigger_reason":   report["trigger_reason"],
            "mean_psi":         report["feature_drift"].get("mean_psi"),
            "current_fraud_rate": fraud_rate,
            "feature_summary": {
                k: {
                    "psi":      v["psi"],
                    "severity": v["severity"],
                    "drift":    v["drift_detected"],
                }
                for k, v in report["feature_drift"].get("feature_drift", {}).items()
            },
            "prediction_drift": report["prediction_drift"],
        }, f, indent=2)

    return report["should_retrain"]


# ── Retrain Job ───────────────────────────────────────────────────────────────
def retrain_job(reason: str = "scheduled"):
    print(f"\n{'='*55}")
    print(
        f"  RETRAIN — {time.strftime('%Y-%m-%d %H:%M:%S')} [{reason.upper()}]")
    print(f"{'='*55}")

    client = MlflowClient()
    current_recall, current_auc, current_version = get_production_metrics()

    if current_version:
        print(f"  Current Production: v{current_version}")
        print(f"  Recall: {current_recall:.4f} | ROC-AUC: {current_auc:.4f}")
    else:
        print("  No production model — training from scratch.")

    # ── Retrain ───────────────────────────────────────────────────────────
    print("\n  Training...")
    BEST_CONFIG["run_name"] = f"retrain-{reason}-{time.strftime('%Y%m%d-%H%M')}"
    new_metrics = train_and_log(BEST_CONFIG, data_path=DATA_PATH)

    # ── Compare and Promote ───────────────────────────────────────────────
    new_recall = new_metrics["recall_fraud"]
    improved = new_recall > current_recall

    if improved or current_version is None:
        print(
            f"\n  ✅ New model better! Recall: {current_recall:.4f} → {new_recall:.4f}")
        promote_to_production(client)
    else:
        print(f"\n  Current model still best.")
        print(
            f"  Old recall: {current_recall:.4f} | New recall: {new_recall:.4f}")
        print(f"  Keeping v{current_version} in production.")


# ── Drift Monitor Loop ────────────────────────────────────────────────────────
def drift_monitor_loop():
    """Separate tight loop that checks for drift every DRIFT_CHECK_INTERVAL seconds."""
    while True:
        should_retrain = check_drift()
        if should_retrain:
            print("\n  🚨 Drift detected — triggering immediate retraining.")
            retrain_job(reason="drift-triggered")
        time.sleep(DRIFT_CHECK_INTERVAL)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import threading

    print("Retrain scheduler started.")
    print(f"  Scheduled retraining: every 24 hours")
    print(f"  Drift check interval: every {DRIFT_CHECK_INTERVAL}s")
    print(f"  Drift window: last {DRIFT_WINDOW_SIZE} predictions\n")

    # Run initial retrain immediately
    retrain_job(reason="initial")

    # Schedule 24-hour retrain
    schedule.every(24).hours.do(retrain_job, reason="scheduled")

    # Run drift monitor in a background thread
    drift_thread = threading.Thread(target=drift_monitor_loop, daemon=True)
    drift_thread.start()

    # Main loop for scheduled jobs
    while True:
        schedule.run_pending()
        time.sleep(60)
