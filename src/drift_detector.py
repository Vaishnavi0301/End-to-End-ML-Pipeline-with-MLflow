# src/drift_detector.py
"""
Drift Detection for the Fraud Detection Pipeline
─────────────────────────────────────────────────
Three detection methods:

  1. Population Stability Index (PSI)
       Industry-standard metric from credit risk.
       Compares current feature distribution to reference (training) bins.
       PSI < 0.10  → no significant drift
       PSI 0.10–0.20 → moderate drift, investigate
       PSI > 0.20  → significant drift, retrain

  2. Kolmogorov–Smirnov (KS) test
       Non-parametric test comparing two distributions.
       p-value < 0.05 → distributions differ significantly.
       Complementary to PSI — catches drift PSI bins can miss.

  3. Prediction Drift
       Monitor the live fraud rate vs the training fraud rate.
       A sudden spike or drop in predicted fraud % is a strong drift signal.

Usage:
    detector = DriftDetector("reference_stats.json")
    report   = detector.check_feature_drift(X_recent)   # numpy array
    pred_rep = detector.check_prediction_drift(0.023)
    summary  = detector.full_report(X_recent, current_fraud_rate)
"""

import json
import os
import numpy as np
from scipy import stats


# ── Thresholds ────────────────────────────────────────────────────────────────
PSI_LOW  = 0.10   # below → stable
PSI_HIGH = 0.20   # above → retrain
KS_PVAL  = 0.05   # below → drift
FRAUD_RATE_DELTA = 0.03   # absolute change in fraud rate triggers alert


class DriftDetector:
    def __init__(self, reference_stats_path: str = "reference_stats.json"):
        self.reference_stats_path = reference_stats_path
        self.reference = self._load()

    # ── Internal ─────────────────────────────────────────────────────────────
    def _load(self):
        if not os.path.exists(self.reference_stats_path):
            return None
        with open(self.reference_stats_path) as f:
            return json.load(f)

    def is_ready(self) -> bool:
        """Returns True if reference stats are available."""
        return self.reference is not None

    # ── PSI ──────────────────────────────────────────────────────────────────
    @staticmethod
    def compute_psi(reference: list, current: list, n_bins: int = 10) -> float:
        """
        Population Stability Index.

        Bins are derived from reference data percentiles so they reflect the
        natural spread of training data, not uniform width intervals.
        A small epsilon (1e-4) avoids log(0) when a bin is empty.
        """
        ref = np.array(reference)
        cur = np.array(current)

        # Build percentile-based bins from reference
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges   = np.percentile(ref, percentiles)
        bin_edges[0]  = -np.inf
        bin_edges[-1] =  np.inf

        ref_counts = np.histogram(ref, bins=bin_edges)[0]
        cur_counts = np.histogram(cur, bins=bin_edges)[0]

        # Convert to proportions, add epsilon to avoid log(0)
        eps = 1e-4
        ref_pct = (ref_counts + eps) / (len(ref) + eps * n_bins)
        cur_pct = (cur_counts + eps) / (len(cur) + eps * n_bins)

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return round(psi, 6)

    # ── Feature Drift ─────────────────────────────────────────────────────────
    def check_feature_drift(self, current_features: np.ndarray) -> dict:
        """
        Run PSI + KS test for each feature.

        Args:
            current_features: numpy array (n_samples, 30) — recent inference data

        Returns:
            dict with per-feature drift stats + overall summary
        """
        if not self.is_ready():
            return {
                "error": "No reference_stats.json found. Run train.py first.",
                "overall_drift_detected": False
            }

        feature_names   = self.reference["feature_names"]
        feature_results = {}
        drifted         = []

        for i, fname in enumerate(feature_names):
            ref_samples = self.reference["features"][fname]["samples"]
            cur_samples = current_features[:, i].tolist()

            psi = self.compute_psi(ref_samples, cur_samples)
            ks_stat, ks_pval = stats.ks_2samp(ref_samples, cur_samples)

            # Drift if PSI is high OR KS test rejects null hypothesis
            drift_detected = (psi > PSI_HIGH) or (ks_pval < KS_PVAL)

            if psi > PSI_HIGH:
                severity = "HIGH"
            elif psi > PSI_LOW:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            feature_results[fname] = {
                "psi":            psi,
                "ks_statistic":   round(float(ks_stat), 4),
                "ks_pvalue":      round(float(ks_pval), 4),
                "drift_detected": drift_detected,
                "severity":       severity,
                "ref_mean":       round(self.reference["features"][fname]["mean"], 4),
                "ref_std":        round(self.reference["features"][fname]["std"], 4),
                "cur_mean":       round(float(np.mean(cur_samples)), 4),
                "cur_std":        round(float(np.std(cur_samples)), 4),
            }

            if drift_detected:
                drifted.append(fname)

        # Overall PSI = mean across all features (common in credit risk)
        all_psi = [v["psi"] for v in feature_results.values()]
        mean_psi = round(float(np.mean(all_psi)), 4)

        return {
            "feature_drift":          feature_results,
            "drifted_features":        drifted,
            "n_drifted":               len(drifted),
            "overall_drift_detected":  len(drifted) > 0,
            "mean_psi":                mean_psi,
            "severity": (
                "HIGH"   if mean_psi > PSI_HIGH else
                "MEDIUM" if mean_psi > PSI_LOW  else
                "LOW"
            ),
            "n_samples_evaluated": current_features.shape[0],
        }

    # ── Prediction Drift ─────────────────────────────────────────────────────
    def check_prediction_drift(self, current_fraud_rate: float) -> dict:
        """
        Compare current observed fraud rate to the training fraud rate.

        Args:
            current_fraud_rate: fraction of recent predictions flagged as fraud

        Returns:
            dict with drift status and rate change stats
        """
        if not self.is_ready():
            return {"error": "No reference stats found."}

        ref_rate = self.reference["fraud_rate"]
        delta    = abs(current_fraud_rate - ref_rate)

        # PSI on a 2-bucket distribution (fraud / not-fraud)
        ref_dist = [ref_rate, 1 - ref_rate]
        cur_dist = [current_fraud_rate, 1 - current_fraud_rate]
        psi = self.compute_psi(ref_dist, cur_dist, n_bins=2)

        drift_detected = delta > FRAUD_RATE_DELTA

        return {
            "reference_fraud_rate": round(ref_rate, 6),
            "current_fraud_rate":   round(current_fraud_rate, 6),
            "absolute_delta":       round(delta, 6),
            "psi":                  psi,
            "drift_detected":       drift_detected,
            "rate_change_pct": (
                round((current_fraud_rate - ref_rate) / ref_rate * 100, 2)
                if ref_rate > 0 else 0.0
            ),
            "direction": (
                "SPIKE"  if current_fraud_rate > ref_rate + FRAUD_RATE_DELTA else
                "DROP"   if current_fraud_rate < ref_rate - FRAUD_RATE_DELTA else
                "STABLE"
            ),
        }

    # ── Full Report ───────────────────────────────────────────────────────────
    def full_report(self, current_features: np.ndarray, current_fraud_rate: float) -> dict:
        """
        Combined feature + prediction drift report.
        Used by retrain.py to decide whether to trigger retraining.

        Returns:
            dict with should_retrain flag and all drift details
        """
        feature_report    = self.check_feature_drift(current_features)
        prediction_report = self.check_prediction_drift(current_fraud_rate)

        feature_drift    = feature_report.get("overall_drift_detected", False)
        prediction_drift = prediction_report.get("drift_detected", False)
        should_retrain   = feature_drift or prediction_drift

        return {
            "should_retrain":   should_retrain,
            "trigger_reason": (
                "feature_and_prediction_drift" if (feature_drift and prediction_drift) else
                "feature_drift"                if feature_drift else
                "prediction_drift"             if prediction_drift else
                "none"
            ),
            "feature_drift":   feature_report,
            "prediction_drift": prediction_report,
        }


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    detector = DriftDetector("reference_stats.json")

    if not detector.is_ready():
        print("No reference_stats.json — run train.py first.")
        sys.exit(1)

    feature_names = detector.reference["feature_names"]
    n_features    = len(feature_names)

    print("Testing DriftDetector with synthetic current data...\n")

    # Simulate no drift — sample from same distribution
    rng = np.random.default_rng(99)
    means = np.array([detector.reference["features"][f]["mean"] for f in feature_names])
    stds  = np.array([detector.reference["features"][f]["std"]  for f in feature_names])
    X_nodrift = rng.normal(loc=means, scale=stds, size=(500, n_features))

    # Simulate drift — shift V1 and V3 significantly
    X_drift = X_nodrift.copy()
    X_drift[:, 0] += 3.0  # V1 shifted
    X_drift[:, 2] += 2.5  # V3 shifted

    print("=== No Drift Scenario ===")
    rep = detector.check_feature_drift(X_nodrift)
    print(f"Drift detected: {rep['overall_drift_detected']}")
    print(f"Mean PSI: {rep['mean_psi']}")

    print("\n=== Drift Scenario ===")
    rep = detector.check_feature_drift(X_drift)
    print(f"Drift detected: {rep['overall_drift_detected']}")
    print(f"Drifted features: {rep['drifted_features']}")
    print(f"Mean PSI: {rep['mean_psi']}")

    print("\n=== Prediction Drift ===")
    pred_rep = detector.check_prediction_drift(current_fraud_rate=0.05)
    print(json.dumps(pred_rep, indent=2))

    print("\n✅ DriftDetector OK")
