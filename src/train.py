# src/train.py
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)
from preprocess import preprocess

sys.path.append(os.path.dirname(__file__))

# ── MLflow Setup ──────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("credit-card-fraud-detection")

FEATURE_NAMES = [f'V{i}' for i in range(
    1, 29)] + ['Amount_scaled', 'Time_scaled']

# ── Experiment Configs ────────────────────────────────────────────────────────
EXPERIMENTS = [
    {
        "run_name":    "RF-baseline-no-sampling",
        "model":       "RandomForest",
        "sampling":    "none",
        "n_estimators": 100,
        "max_depth":   10,
        "class_weight": None,
    },
    {
        "run_name":    "RF-class-weight-balanced",
        "model":       "RandomForest",
        "sampling":    "class_weight",
        "n_estimators": 100,
        "max_depth":   10,
        "class_weight": "balanced",
    },
    {
        "run_name":    "RF-smote",
        "model":       "RandomForest",
        "sampling":    "smote",
        "n_estimators": 100,
        "max_depth":   10,
        "class_weight": None,
    },
    {
        "run_name":    "RF-smote-tuned",
        "model":       "RandomForest",
        "sampling":    "smote",
        "n_estimators": 200,
        "max_depth":   20,
        "class_weight": None,
    },
    {
        "run_name":    "GBM-smote",
        "model":       "GradientBoosting",
        "sampling":    "smote",
        "n_estimators": 150,
        "learning_rate": 0.1,
        "class_weight": None,
    },
]


# ── Reference Stats for Drift Detection ──────────────────────────────────────
def save_reference_stats(X_train, y_train, path="reference_stats.json"):
    """
    Save training distribution as reference for drift detection.
    Samples 1000 rows per feature — enough for PSI/KS without huge file size.
    Called only when a model qualifies for registration (recall > 0.85).
    """
    rng = np.random.default_rng(42)
    n_samples = min(1000, X_train.shape[0])
    idx = rng.choice(X_train.shape[0], size=n_samples, replace=False)
    sampled = X_train[idx]

    features_data = {}
    for i, name in enumerate(FEATURE_NAMES):
        col = sampled[:, i].tolist()
        features_data[name] = {
            "samples": col,
            "mean":    float(np.mean(col)),
            "std":     float(np.std(col)),
            "p10":     float(np.percentile(col, 10)),
            "p25":     float(np.percentile(col, 25)),
            "p50":     float(np.percentile(col, 50)),
            "p75":     float(np.percentile(col, 75)),
            "p90":     float(np.percentile(col, 90)),
        }

    stats = {
        "feature_names":   FEATURE_NAMES,
        "features":        features_data,
        "fraud_rate":      float(y_train.mean()),
        "n_train_samples": int(len(y_train)),
        "saved_at":        str(np.datetime64("now")),
    }

    with open(path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Reference stats saved → {path}  ({n_samples} samples/feature)")


# ── Confusion Matrix Plot ─────────────────────────────────────────────────────
def save_confusion_matrix(y_test, y_pred, run_id):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legit', 'Fraud'],
                yticklabels=['Legit', 'Fraud'])
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

    fraud_caught = cm[1][1]
    fraud_missed = cm[1][0]
    ax.text(0.5, -0.12,
            f"Fraud caught: {fraud_caught} | Fraud missed: {fraud_missed}",
            transform=ax.transAxes, ha='center', fontsize=10, color='darkred')

    plt.tight_layout()
    path = f"confusion_matrix_{run_id[:8]}.png"
    plt.savefig(path, dpi=120, bbox_inches='tight')
    plt.close()
    return path


# ── Core Training Function ────────────────────────────────────────────────────
def train_and_log(config, data_path="data/creditcard.csv"):
    print(f"\n{'='*55}")
    print(f"  Running: {config['run_name']}")
    print(f"{'='*55}")

    X_train, X_test, y_train, y_test = preprocess(
        data_path, sampling_strategy=config["sampling"]
    )

    with mlflow.start_run(run_name=config["run_name"]):
        run_id = mlflow.active_run().info.run_id

        # ── 1. Log Parameters ─────────────────────────────────────────────
        mlflow.log_params({
            "model_type":        config["model"],
            "sampling_strategy": config["sampling"],
            "n_estimators":      config.get("n_estimators", "N/A"),
            "max_depth":         config.get("max_depth", "N/A"),
            "learning_rate":     config.get("learning_rate", "N/A"),
            "class_weight":      str(config.get("class_weight")),
            "train_size":        X_train.shape[0],
            "test_size":         X_test.shape[0],
            "fraud_in_test":     int(sum(y_test == 1)),
        })

        # ── 2. Build Model ────────────────────────────────────────────────
        if config["model"] == "RandomForest":
            model = RandomForestClassifier(
                n_estimators=config["n_estimators"],
                max_depth=config["max_depth"],
                class_weight=config.get("class_weight"),
                random_state=42,
                n_jobs=-1
            )
        else:
            model = GradientBoostingClassifier(
                n_estimators=config["n_estimators"],
                learning_rate=config["learning_rate"],
                random_state=42
            )

        print(f"Training {config['model']}...")
        model.fit(X_train, y_train)

        # ── 3. Evaluate ───────────────────────────────────────────────────
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy":        round(accuracy_score(y_test, y_pred), 6),
            "precision_fraud": round(precision_score(y_test, y_pred, zero_division=0), 6),
            "recall_fraud":    round(recall_score(y_test, y_pred, zero_division=0), 6),
            "f1_fraud":        round(f1_score(y_test, y_pred, zero_division=0), 6),
            "roc_auc":         round(roc_auc_score(y_test, y_proba), 6),
            "pr_auc":          round(average_precision_score(y_test, y_proba), 6),
            "fraud_caught":    int(sum((y_pred == 1) & (y_test == 1))),
            "fraud_missed":    int(sum((y_pred == 0) & (y_test == 1))),
            "false_alarms":    int(sum((y_pred == 1) & (y_test == 0))),
        }
        mlflow.log_metrics(metrics)

        # ── 4. Log Artifacts ──────────────────────────────────────────────
        cm_path = save_confusion_matrix(y_test, y_pred, run_id)
        mlflow.log_artifact(cm_path)
        os.remove(cm_path)

        if os.path.exists("scalers.pkl"):
            mlflow.log_artifact("scalers.pkl")

        # ── 5. Log Model ──────────────────────────────────────────────────
        mlflow.sklearn.log_model(
            model, artifact_path="model", registered_model_name=None)

        # ── 6. Register if Good Enough ────────────────────────────────────
        # recall > 0.85 AND roc_auc > 0.97
        if metrics["recall_fraud"] > 0.85 and metrics["roc_auc"] > 0.97:
            mlflow.register_model(
                model_uri=f"runs:/{run_id}/model",
                name="fraud-detector"
            )
            print("✅ Registered to MLflow Model Registry")

            # Save reference distribution for drift detection.
            # Only saved for models that qualify — ties reference to production quality.
            save_reference_stats(X_train, y_train, path="reference_stats.json")
            if os.path.exists("reference_stats.json"):
                mlflow.log_artifact("reference_stats.json")

        # ── 7. Print Summary ──────────────────────────────────────────────
        print(f"\n  Results:")
        print(
            f"  Accuracy:       {metrics['accuracy']:.4f}  ← misleading on imbalanced data")
        print(
            f"  Recall (fraud): {metrics['recall_fraud']:.4f}  ← most important")
        print(f"  Precision:      {metrics['precision_fraud']:.4f}")
        print(f"  ROC-AUC:        {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:         {metrics['pr_auc']:.4f}")
        print(
            f"  Fraud caught:   {metrics['fraud_caught']} / {sum(y_test == 1)}")
        print(f"  Fraud missed:   {metrics['fraud_missed']}")
        print(f"  False alarms:   {metrics['false_alarms']}")

        return metrics


# ── Run All Experiments ───────────────────────────────────────────────────────
if __name__ == "__main__":
    data_path = os.path.join(os.path.dirname(
        __file__), "..", "data", "creditcard.csv")

    all_results = []
    for config in EXPERIMENTS:
        metrics = train_and_log(config, data_path=data_path)
        all_results.append({
            "run":           config["run_name"],
            "recall":        metrics["recall_fraud"],
            "roc_auc":       metrics["roc_auc"],
            "fraud_missed":  metrics["fraud_missed"],
        })

    print(f"\n{'='*55}")
    print("  FINAL COMPARISON")
    print(f"{'='*55}")
    print(f"{'Run':<30} {'Recall':>8} {'ROC-AUC':>9} {'Missed':>8}")
    print("-" * 55)
    for r in all_results:
        print(
            f"{r['run']:<30} {r['recall']:>8.4f} {r['roc_auc']:>9.4f} {r['fraud_missed']:>8}")

    print(f"\n✅ All experiments complete. Open http://localhost:5000 to compare.")
