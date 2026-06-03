# dashboard/app.py
import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.tracking import MlflowClient
import time

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

MLFLOW_URI = "http://localhost:5000"
API_URL = "http://localhost:8000"
mlflow.set_tracking_uri(MLFLOW_URI)

DARK_BG = '#0e1117'
CHART_GRID = '#1c1f26'


def api_get(path, timeout=5):
    try:
        r = requests.get(f"{API_URL}{path}", timeout=timeout)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def dark_fig(figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')
    return fig, ax


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["📊 Experiment Results",
         "🔮 Live Prediction",
         "📈 Model Info",
         "📡 Monitoring",
         "🧠 Drift Detection"],
        label_visibility="collapsed"
    )

    st.divider()

    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            st.success("API: Online ✅")
        else:
            st.error("API: Error ❌")
        health = r.json()
        st.caption(
            f"Total predictions: {health.get('total_predictions', 0):,}")
        st.caption(f"Cache entries: {health.get('cache_size', 0)}")
    except Exception:
        st.error("API: Offline ❌")
        health = {}

    try:
        client = MlflowClient()
        client.search_experiments()
        st.success("MLflow: Online ✅")
    except Exception:
        st.error("MLflow: Offline ❌")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Experiment Results
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Experiment Results":
    st.header("📊 MLflow Experiment Results")
    st.markdown(
        "Comparing 5 experiments across different imbalance handling strategies.")

    try:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(
            "credit-card-fraud-detection")

        if experiment is None:
            st.warning("No experiment found. Run train.py first.")
        else:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["metrics.recall_fraud DESC"]
            )
            if not runs:
                st.warning("No runs found.")
            else:
                rows = []
                for run in runs:
                    m, p = run.data.metrics, run.data.params
                    rows.append({
                        "Run":          run.info.run_name,
                        "Sampling":     p.get("sampling_strategy", "-"),
                        "Model":        p.get("model_type", "-"),
                        "Recall":       round(m.get("recall_fraud", 0), 4),
                        "Precision":    round(m.get("precision_fraud", 0), 4),
                        "ROC-AUC":      round(m.get("roc_auc", 0), 4),
                        "PR-AUC":       round(m.get("pr_auc", 0), 4),
                        "Fraud Missed": int(m.get("fraud_missed", 0)),
                        "False Alarms": int(m.get("false_alarms", 0)),
                    })

                df = pd.DataFrame(rows)
                best = df.loc[df["Recall"].idxmax()]

                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Best Recall",    f"{best['Recall']:.4f}")
                col2.metric("Best ROC-AUC",   f"{df['ROC-AUC'].max():.4f}")
                col3.metric("Min Fraud Missed",
                            f"{int(df['Fraud Missed'].min())} / 98")
                col4.metric("Total Experiments", len(df))

                st.divider()
                st.subheader("All Runs")

                def highlight_best(row):
                    return (["background-color: #1a3a1a"] * len(row)
                            if row["Recall"] == df["Recall"].max() else [""] * len(row))

                st.dataframe(
                    df.style.apply(highlight_best, axis=1).format(
                        {"Recall": "{:.4f}", "Precision": "{:.4f}",
                         "ROC-AUC": "{:.4f}", "PR-AUC": "{:.4f}"}),
                    use_container_width=True, hide_index=True
                )
                st.caption(
                    "🟢 Highlighted row = best recall (most fraud caught)")

                st.divider()
                col_left, col_right = st.columns(2)

                with col_left:
                    st.subheader("Recall by Experiment")
                    fig, ax = dark_fig()
                    colors = [
                        "#2ecc71" if r == df["Recall"].max() else
                        "#e74c3c" if r == df["Recall"].min() else "#3498db"
                        for r in df["Recall"]
                    ]
                    ax.barh(df["Run"], df["Recall"], color=colors, alpha=0.85)
                    ax.set_xlabel("Recall (Fraud)")
                    ax.set_xlim(0.7, 1.0)
                    ax.axvline(x=0.85, color='white', linestyle='--',
                               alpha=0.5, label='Target (0.85)')
                    ax.legend(facecolor='#1c1f26', labelcolor='white')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

                with col_right:
                    st.subheader("Fraud Missed vs False Alarms")
                    fig, ax = dark_fig()
                    x, w = np.arange(len(df)), 0.35
                    ax.bar(x - w/2, df["Fraud Missed"], w,
                           label="Fraud Missed", color="#e74c3c", alpha=0.85)
                    ax.bar(x + w/2, df["False Alarms"], w,
                           label="False Alarms", color="#f39c12", alpha=0.85)
                    ax.set_xticks(x)
                    ax.set_xticklabels(
                        [r.replace("RF-", "").replace("-smote", "+S")
                         for r in df["Run"]],
                        rotation=30, ha='right', fontsize=8
                    )
                    ax.legend(facecolor='#1c1f26', labelcolor='white')
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()

    except Exception as e:
        st.error(f"MLflow connection error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Live Prediction
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Prediction":
    st.header("🔮 Live Prediction")

    FRAUD_EXAMPLE = {
        "features": [-2.3122, 1.9520, -1.6099, 3.9979, -0.5222,
                     -1.4265, -2.5374, 1.3917, -2.7701, -2.7723,
                     3.2020, -2.8999, -0.5952, -4.2893, 0.3897,
                     -1.1407, -2.8301, -0.0168, 0.4170, 0.1269,
                     0.5172, -0.0350, -0.4652, 0.3202, 0.0445,
                     0.1778, 0.2611, -0.1433],
        "amount": 0.0, "time": 406.0
    }
    NORMAL_EXAMPLE = {
        "features": [1.2, 0.5, -0.3, 0.8, 0.2, -0.1, 0.4, 0.3, -0.2, 0.1,
                     0.5, -0.3, 0.2, 0.1, -0.4, 0.3, 0.2, -0.1, 0.4, 0.2,
                     -0.1, 0.3, 0.1, -0.2, 0.4, 0.1, -0.3, 0.2],
        "amount": 50.0, "time": 1000.0
    }

    col_input, col_result = st.columns([1, 1])

    with col_input:
        st.subheader("Transaction Input")
        qcol1, qcol2 = st.columns(2)
        if qcol1.button("Load Normal", use_container_width=True):
            st.session_state["example"] = NORMAL_EXAMPLE
        if qcol2.button("Load Suspicious", use_container_width=True):
            st.session_state["example"] = FRAUD_EXAMPLE

        example = st.session_state.get("example", NORMAL_EXAMPLE)
        amount = st.number_input("Amount ($)", min_value=0.0, max_value=50000.0,
                                 value=float(example["amount"]), step=0.01)
        time_val = st.number_input("Time (seconds since first transaction)",
                                   min_value=0.0, value=float(example["time"]), step=1.0)
        st.markdown("**V1–V28 Features**")
        features_str = st.text_area(
            "Features (28 comma-separated values)",
            value=", ".join([str(round(f, 4)) for f in example["features"]]),
            height=120, label_visibility="collapsed"
        )

        predict_btn = st.button(
            "🔍 Predict", type="primary", use_container_width=True)
        explain_btn = st.button("🧠 Explain (SHAP)", use_container_width=True)

    with col_result:
        st.subheader("Prediction Result")

        if predict_btn or explain_btn:
            try:
                features = [float(x.strip()) for x in features_str.split(",")]
                if len(features) != 28:
                    st.error(f"Need exactly 28 features, got {len(features)}")
                else:
                    payload = {"features": features,
                               "amount": amount, "time": time_val}

                    if predict_btn:
                        with st.spinner("Calling API..."):
                            response = requests.post(
                                f"{API_URL}/predict", json=payload, timeout=10)

                        if response.status_code == 200:
                            result = response.json()
                            if result["is_fraud"]:
                                st.error("## 🚨 FRAUD DETECTED")
                            else:
                                st.success("## ✅ LEGITIMATE TRANSACTION")

                            m1, m2, m3 = st.columns(3)
                            m1.metric("Fraud Probability",
                                      f"{result['fraud_probability']*100:.2f}%")
                            m2.metric("Risk Level", result["risk_level"])
                            m3.metric(
                                "Latency", f"{result['latency_ms']:.1f}ms")

                            if result.get("cache_hit"):
                                st.caption("⚡ Served from cache")

                            # Gauge bar
                            proba = result["fraud_probability"]
                            fig, ax = dark_fig((6, 1.5))
                            ax.barh([""], [proba], color="#e74c3c" if proba > 0.5 else "#2ecc71",
                                    alpha=0.85, height=0.5)
                            ax.barh([""], [1 - proba], left=[proba],
                                    color="#333", alpha=0.5, height=0.5)
                            ax.axvline(x=0.5, color='white',
                                       linestyle='--', alpha=0.7)
                            ax.set_xlim(0, 1)
                            ax.set_xlabel("Fraud Probability")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                            st.caption(
                                f"Transaction: `{result['transaction_id']}` · Model: `{result['model_version']}`")
                        else:
                            st.error(f"API error {response.status_code}")
                            st.json(response.json())

                    if explain_btn:
                        with st.spinner("Computing SHAP values..."):
                            response = requests.post(
                                f"{API_URL}/explain", json=payload, timeout=30)

                        if response.status_code == 200:
                            shap_result = response.json()
                            st.subheader("🧠 SHAP Feature Importance")
                            st.caption(
                                f"Base value: {shap_result['base_value']:.4f} | "
                                f"Fraud probability: {shap_result['fraud_probability']*100:.2f}%"
                            )

                            top10 = shap_result["top_10_features"]
                            shap_df = pd.DataFrame(top10)

                            fig, ax = dark_fig((7, 5))
                            colors = [
                                "#e74c3c" if v > 0 else "#2ecc71" for v in shap_df["shap_value"]]
                            ax.barh(shap_df["feature"], shap_df["shap_value"],
                                    color=colors, alpha=0.85)
                            ax.axvline(x=0, color='white',
                                       linestyle='-', alpha=0.4)
                            ax.set_xlabel(
                                "SHAP Value (impact on fraud probability)")
                            ax.set_title("Top 10 Feature Contributions")
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close()

                            st.caption(
                                "🔴 Red = pushes toward fraud  |  🟢 Green = pushes toward legit")
                        else:
                            st.error(
                                f"SHAP error {response.status_code}: {response.json()}")

            except ValueError:
                st.error("Invalid features. Use comma-separated numbers.")
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API.")
        else:
            st.info("Fill in transaction details and click Predict.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Model Info
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Info":
    st.header("📈 Production Model Info")

    try:
        response = requests.get(f"{API_URL}/model-info", timeout=5)
        if response.status_code == 200:
            info = response.json()
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Model Details")
                st.markdown(
                    f"**Model:** `{info.get('model_name', 'fraud-detector')}`")
                st.markdown(f"**Version:** `v{info.get('version', '?')}`")
                st.markdown(
                    f"**Type:** `{info.get('model_type', 'RandomForest')}`")
                st.markdown(
                    f"**Sampling:** `{info.get('sampling_strategy', 'smote')}`")

                st.divider()
                st.subheader("Threshold Analysis")
                t = api_get("/threshold-info")
                if t:
                    st.markdown(
                        f"**Current threshold:** `{t['current_threshold']}`")
                    st.info(t['explanation'])
                    for thresh, desc in t['tradeoffs'].items():
                        st.markdown(f"- **{thresh}**: {desc}")

            with col2:
                st.subheader("Production Metrics")
                metrics = info.get("metrics", {})
                st.metric("Recall (Fraud)", f"{metrics.get('recall_fraud', 0):.4f}",
                          help="Fraction of actual fraud cases caught")
                st.metric("Precision",
                          f"{metrics.get('precision_fraud', 0):.4f}")
                st.metric("ROC-AUC",        f"{metrics.get('roc_auc', 0):.4f}")
                st.metric("Fraud Caught",
                          f"{int(metrics.get('fraud_caught', 0))} / 98")
                st.metric("Fraud Missed",
                          f"{int(metrics.get('fraud_missed', 0))} cases")
        else:
            st.error("Could not fetch model info from API")

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API at port 8000.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Monitoring
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📡 Monitoring":
    st.header("📡 Inference Monitoring")
    st.caption(
        "Live stats from the SQLite inference log · Auto-updates on refresh")

    if st.button("🔄 Refresh", type="secondary"):
        st.rerun()

    stats = api_get("/monitoring/stats")

    if stats is None:
        st.error("Cannot fetch monitoring stats from API.")
        st.stop()

    total = stats.get("total_predictions") or 0
    frauds = stats.get("total_fraud_flagged") or 0
    avg_lat = stats.get("avg_latency_ms") or 0
    fraud_r = stats.get("fraud_rate") or 0

    # ── KPI row ──────────────────────────────────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Predictions", f"{total:,}")
    k2.metric("Fraud Flagged",     f"{frauds:,}")
    k3.metric("Fraud Rate",        f"{fraud_r*100:.2f}%")
    k4.metric("Avg Latency",       f"{avg_lat:.1f} ms")

    st.divider()

    # ── Hourly Trend ─────────────────────────────────────────────────────
    trend = stats.get("hourly_trend", [])
    if trend:
        trend_df = pd.DataFrame(trend)
        st.subheader("Fraud Rate & Latency — Last 24 Hours")
        col_a, col_b = st.columns(2)

        with col_a:
            fig, ax = dark_fig()
            fraud_rate_vals = (
                trend_df["frauds"] / trend_df["total"].replace(0, 1) * 100)
            ax.plot(trend_df["hour"], fraud_rate_vals, color="#e74c3c",
                    marker="o", linewidth=2)
            ax.fill_between(range(len(trend_df)), fraud_rate_vals,
                            alpha=0.15, color="#e74c3c")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Fraud Rate (%)")
            ax.set_title("Fraud Rate Trend")
            ax.set_xticks(range(len(trend_df)))
            ax.set_xticklabels(
                [h[-5:] for h in trend_df["hour"]], rotation=45, ha='right', fontsize=7
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_b:
            fig, ax = dark_fig()
            ax.plot(trend_df["hour"], trend_df["avg_latency"],
                    color="#3498db", marker="o", linewidth=2)
            ax.fill_between(range(len(trend_df)), trend_df["avg_latency"],
                            alpha=0.15, color="#3498db")
            ax.set_xlabel("Hour")
            ax.set_ylabel("Avg Latency (ms)")
            ax.set_title("Average Latency Trend")
            ax.set_xticks(range(len(trend_df)))
            ax.set_xticklabels(
                [h[-5:] for h in trend_df["hour"]], rotation=45, ha='right', fontsize=7
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    else:
        st.info("No hourly trend data yet. Make some predictions first.")

    st.divider()

    # ── Confidence Distribution ───────────────────────────────────────────
    buckets = stats.get("confidence_distribution", [])
    if buckets:
        st.subheader("Fraud Probability Distribution")
        bdf = pd.DataFrame(buckets)
        bdf["label"] = bdf["bucket_start"].apply(lambda x: f"{x}–{x+10}%")

        fig, ax = dark_fig((8, 3))
        colors = ["#e74c3c" if b >=
                  50 else "#3498db" for b in bdf["bucket_start"]]
        ax.bar(bdf["label"], bdf["count"], color=colors, alpha=0.85)
        ax.set_xlabel("Fraud Probability Bucket")
        ax.set_ylabel("Prediction Count")
        ax.set_title("Confidence Distribution")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        st.caption("🔴 Red buckets = predictions above 50% fraud threshold")

    st.divider()

    # ── Recent Predictions Table ──────────────────────────────────────────
    st.subheader("Recent Predictions")
    recent_data = api_get("/monitoring/recent?limit=50")
    if recent_data and recent_data.get("predictions"):
        rdf = pd.DataFrame(recent_data["predictions"])
        rdf["is_fraud"] = rdf["is_fraud"].map({1: "🚨 FRAUD", 0: "✅ Legit"})
        rdf["timestamp"] = pd.to_datetime(
            rdf["timestamp"]).dt.strftime("%H:%M:%S")
        rdf = rdf[["timestamp", "transaction_id", "fraud_probability",
                   "is_fraud", "risk_level", "latency_ms", "model_version"]]
        rdf.columns = ["Time", "Transaction ID", "Fraud Prob", "Result",
                       "Risk", "Latency (ms)", "Model"]
        st.dataframe(rdf, use_container_width=True, hide_index=True)
    else:
        st.info("No predictions logged yet.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Drift Detection
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Drift Detection":
    st.header("🧠 Drift Detection")
    st.markdown(
        "Monitors feature distributions against the training reference using "
        "**PSI** (Population Stability Index) and **KS tests**. "
        "Automatic retraining is triggered when drift is detected."
    )

    st.info(
        "**PSI Thresholds:** < 0.10 → Stable  |  0.10–0.20 → Moderate  |  > 0.20 → Significant (retrain)"
    )

    if st.button("🔄 Run Drift Analysis", type="primary"):
        with st.spinner("Analysing recent inference data..."):
            report = api_get("/drift-report", timeout=30)

        if report is None:
            st.error("Could not fetch drift report. Check API connection.")
            st.stop()

        if report.get("status") == "insufficient_data":
            st.warning(report.get("message",
                                  "Insufficient inference data. Make at least 50 predictions first."))
            st.stop()

        if report.get("error"):
            st.warning(report["error"])
            st.stop()

        # ── Overall status ────────────────────────────────────────────────
        should_retrain = report.get("should_retrain", False)
        trigger = report.get("trigger_reason", "none")

        if should_retrain:
            st.error(
                f"## 🚨 Drift Detected — Retrain Recommended\nTrigger: `{trigger}`")
        else:
            st.success("## ✅ No Significant Drift — Model Stable")

        feat_report = report.get("feature_drift", {})
        pred_report = report.get("prediction_drift", {})

        # ── Prediction Drift Panel ────────────────────────────────────────
        st.divider()
        st.subheader("Prediction Drift")
        if pred_report and not pred_report.get("error"):
            pd1, pd2, pd3, pd4 = st.columns(4)
            pd1.metric("Reference Fraud Rate",
                       f"{pred_report.get('reference_fraud_rate', 0)*100:.3f}%")
            pd2.metric("Current Fraud Rate",
                       f"{pred_report.get('current_fraud_rate', 0)*100:.3f}%")
            pd3.metric("Rate Change",
                       f"{pred_report.get('rate_change_pct', 0):+.2f}%")
            pd4.metric("Direction",            pred_report.get(
                "direction", "STABLE"))

            if pred_report.get("drift_detected"):
                st.warning(
                    "Prediction drift detected — fraud rate has shifted significantly.")

        st.divider()

        # ── Feature Drift Summary ─────────────────────────────────────────
        st.subheader("Feature Drift Summary")
        feature_results = feat_report.get("feature_drift", {})

        if feature_results:
            drift_rows = []
            for fname, fdata in feature_results.items():
                drift_rows.append({
                    "Feature":       fname,
                    "PSI":           fdata["psi"],
                    "KS Stat":       fdata["ks_statistic"],
                    "KS p-value":    fdata["ks_pvalue"],
                    "Severity":      fdata["severity"],
                    "Drift":         "🚨 YES" if fdata["drift_detected"] else "✅ No",
                    "Ref Mean":      fdata["ref_mean"],
                    "Cur Mean":      fdata["cur_mean"],
                })
            drift_df = pd.DataFrame(drift_rows).sort_values(
                "PSI", ascending=False)

            def color_severity(val):
                return {
                    "HIGH":   "color: #e74c3c",
                    "MEDIUM": "color: #f39c12",
                    "LOW":    "color: #2ecc71",
                }.get(val, "")

            st.dataframe(
                drift_df.style
                .format({"PSI": "{:.4f}", "KS Stat": "{:.4f}", "KS p-value": "{:.4f}"})
                .applymap(color_severity, subset=["Severity"]),
                use_container_width=True, hide_index=True
            )

            # ── PSI Bar Chart ─────────────────────────────────────────────
            st.subheader("PSI by Feature")
            fig, ax = dark_fig((10, 5))
            psi_vals = [feature_results[f]["psi"] for f in feature_results]
            feat_names = list(feature_results.keys())
            colors = [
                "#e74c3c" if v > 0.20 else
                "#f39c12" if v > 0.10 else "#2ecc71"
                for v in psi_vals
            ]
            ax.bar(feat_names, psi_vals, color=colors, alpha=0.85)
            ax.axhline(y=0.10, color='#f39c12', linestyle='--',
                       alpha=0.7, label='Moderate (0.10)')
            ax.axhline(y=0.20, color='#e74c3c', linestyle='--',
                       alpha=0.7, label='Significant (0.20)')
            ax.set_xlabel("Feature")
            ax.set_ylabel("PSI")
            ax.set_title("Population Stability Index per Feature")
            ax.legend(facecolor='#1c1f26', labelcolor='white')
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.caption(
                f"🔴 High drift  |  🟡 Moderate  |  🟢 Stable  |  "
                f"Drifted features: {feat_report.get('n_drifted', 0)}/{len(feature_results)}"
            )

            if feat_report.get("drifted_features"):
                st.warning(
                    f"Drifted: **{', '.join(feat_report['drifted_features'])}**")

        st.caption(f"Source: `{report.get('source', 'unknown')}` · "
                   f"Records analysed: {report.get('n_records_analyzed', 'N/A')}")
    else:
        st.info(
            "Click **Run Drift Analysis** to check for distribution shifts in recent predictions.")
        st.markdown("""
        **What drift detection does:**
        - Compares the distribution of features in recent predictions to the training reference
        - **PSI** measures bucket-level distribution shift (industry standard for credit risk)
        - **KS test** detects distributional differences statistically
        - **Prediction drift** monitors if the live fraud rate has diverged from training

        **When retraining is triggered automatically:**
        - Any feature shows PSI > 0.20 (significant drift)
        - KS p-value < 0.05 for any feature (distributions statistically different)
        - Live fraud rate deviates > 3% from training fraud rate
        """)
