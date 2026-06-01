import numpy as np
import pandas as pd
import joblib
import json
import os
from pathlib import Path
from datetime import datetime
from loguru import logger
from sklearn.svm import SVC, OneClassSVM
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, f1_score)

from quantum.encoder import FeatureEncoder
from quantum.kernel import QuantumKernelEngine


def generate_training_data(n_samples: int = 40, seed: int = 42) -> tuple:
    """
    Realistic robot telemetry — 4 anomaly types, 3 normal operating modes.
    
    Normal modes:
      - Idle: low current, minimal vibration
      - Running: moderate all metrics
      - High-load: higher current/temp, normal vibration
      
    Anomaly types:
      - Thermal: temperature spike
      - Bearing: vibration spike
      - Electrical: current spike
      - Combined: all three
    """
    rng = np.random.RandomState(seed)
    samples = []
    labels = []

    n_anomaly = max(6, int(n_samples * 0.25))
    n_normal = n_samples - n_anomaly

    # Normal — three operating modes
    modes = [
        # (temp_range, vib_range, curr_range, accel_range)
        ((30, 36), (0.02, 0.15), (0.8, 1.4), (9.75, 9.85)),   # Idle
        ((34, 40), (0.05, 0.35), (1.3, 2.0), (9.70, 9.90)),   # Running
        ((38, 44), (0.08, 0.45), (1.8, 2.8), (9.65, 9.95)),   # High-load
    ]
    per_mode = n_normal // len(modes)
    for (tr, vr, cr, ar) in modes:
        n = per_mode
        X_m = np.column_stack([
            rng.uniform(*tr, n),
            rng.uniform(*vr, n),
            rng.uniform(*cr, n),
            rng.uniform(*ar, n),
        ])
        samples.append(X_m)
        labels.extend([0] * n)

    # Anomalies
    anomaly_types = [
        ((55, 72), (0.05, 0.40), (1.2, 2.5), (9.70, 9.90)),  # Thermal
        ((32, 40), (2.50, 5.00), (1.3, 2.2), (11.0, 16.0)),  # Bearing
        ((33, 41), (0.06, 0.38), (4.50, 8.00), (9.72, 9.88)), # Electrical
        ((52, 68), (2.00, 4.50), (4.00, 7.50), (11.0, 17.0)), # Combined
    ]
    per_type = max(1, n_anomaly // len(anomaly_types))
    for (tr, vr, cr, ar) in anomaly_types:
        n = per_type
        X_a = np.column_stack([
            rng.uniform(*tr, n),
            rng.uniform(*vr, n),
            rng.uniform(*cr, n),
            rng.uniform(*ar, n),
        ])
        samples.append(X_a)
        labels.extend([1] * n)

    X = np.vstack(samples)
    y = np.array(labels, dtype=int)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def load_training_data(data_path: str) -> tuple:
    if not os.path.exists(data_path):
        logger.warning(f"No data at {data_path} — generating synthetic data")
        return generate_training_data()
    df = pd.read_csv(data_path)
    required = ["temperature_c", "vibration_rms", "current_a", "accel_magnitude", "label"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    X = df[["temperature_c", "vibration_rms", "current_a", "accel_magnitude"]].values
    y = df["label"].values.astype(int)
    logger.info(f"Loaded {len(X)} samples: {sum(y==0)} normal, {sum(y==1)} anomaly")
    return X, y


def train_quantum_svm(
    config: dict,
    data_path: str = None,
    use_real_hardware: bool = False,
    model_dir: str = "models/",
) -> dict:
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # 1. Load data
    X, y = load_training_data(data_path) if data_path else generate_training_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train: {len(X_train)} ({sum(y_train==0)} N / {sum(y_train==1)} A)")
    logger.info(f"Test:  {len(X_test)} ({sum(y_test==0)} N / {sum(y_test==1)} A)")

    # 2. Encode
    encoder = FeatureEncoder(n_features=4)
    X_train_enc = encoder.fit_transform(X_train)
    X_test_enc = encoder.transform(X_test)

    # 3. Quantum kernel matrices (batched — single job each)
    engine = QuantumKernelEngine(config)
    engine.setup(use_real_hardware=use_real_hardware)

    logger.info("Computing training kernel matrix (batched)...")
    K_train = engine.compute_kernel_matrix(X_train_enc)
    logger.info("Computing test kernel matrix (batched)...")
    K_test = engine.compute_kernel_matrix(X_test_enc, X_train_enc)
    engine.close()

    # 4. Train SVM — tuned for small quantum datasets
    svm = SVC(
        kernel="precomputed",
        C=config.get("C", 50.0),
        probability=True,
        class_weight="balanced",
        random_state=42,
    )
    svm.fit(K_train, y_train)

    # 5. Evaluate
    y_pred = svm.predict(K_test)
    y_prob = svm.predict_proba(K_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    try:
        auc = roc_auc_score(y_test, y_prob)
    except Exception:
        auc = 0.0

    logger.info("\n" + classification_report(y_test, y_pred, zero_division=0))
    logger.info(f"Confusion matrix:\n{cm}")
    logger.info(f"ROC-AUC: {auc:.3f}")

    # 5b. Cross-validation on training kernel (robust estimate)
    cv = StratifiedKFold(n_splits=min(5, sum(y_train==1)), shuffle=True, random_state=42)
    try:
        cv_scores = cross_val_score(
            SVC(kernel="precomputed", C=config.get("C", 50.0),
                probability=True, class_weight="balanced"),
            K_train, y_train, cv=cv, scoring="f1"
        )
        logger.info(f"CV F1 scores: {cv_scores.round(3)} | mean={cv_scores.mean():.3f}")
    except Exception as e:
        logger.warning(f"CV skipped: {e}")
        cv_scores = np.array([0.0])

    # 6. Support vectors
    sv_indices = svm.support_
    support_vectors_enc = X_train_enc[sv_indices]
    logger.info(f"Support vectors: {len(sv_indices)} / {len(X_train_enc)} training points")

    # 7. Save all artifacts
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    joblib.dump(svm, f"{model_dir}/svm_latest.joblib")
    joblib.dump(encoder, f"{model_dir}/encoder_latest.joblib")
    np.save(f"{model_dir}/X_train_enc_latest.npy", X_train_enc)
    np.save(f"{model_dir}/support_vectors_latest.npy", support_vectors_enc)
    np.save(f"{model_dir}/K_train_latest.npy", K_train)

    metrics = {
        "accuracy": report["accuracy"],
        "f1_normal": report.get("0", {}).get("f1-score", 0.0),
        "f1_anomaly": report.get("1", {}).get("f1-score", 0.0),
        "roc_auc": auc,
        "cv_f1_mean": float(cv_scores.mean()),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_support_vectors": len(sv_indices),
        "confusion_matrix": cm.tolist(),
        "trained_at": ts,
        "backend": config.get("backend", "simulator"),
        "feature_map_reps": config.get("feature_map_reps", 2),
        "shots": config.get("shots", 2048),
    }

    with open(f"{model_dir}/model_info.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.success(f"Model saved → {model_dir}")
    logger.success(
        f"Accuracy={metrics['accuracy']:.1%} | "
        f"F1-anomaly={metrics['f1_anomaly']:.3f} | "
        f"AUC={auc:.3f}"
    )
    return metrics
