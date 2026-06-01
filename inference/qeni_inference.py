import numpy as np
import joblib
import json
import time
import os
from datetime import datetime
from typing import Optional, Callable, List
from dataclasses import dataclass, field
from loguru import logger

from quantum.kernel import QuantumKernelEngine
from quantum.encoder import FeatureEncoder
from sensors.sensor_manager import SensorManager


@dataclass
class InferenceResult:
    timestamp: datetime
    unit_id: int
    # Raw sensor data — stays on device, never transmitted
    temperature_c: float
    vibration_rms: float
    current_a: float
    health_score: float
    is_injected_anomaly: bool
    # Quantum computation outputs
    features_encoded: np.ndarray
    kernel_row: np.ndarray
    svm_decision: float
    prediction: int              # 0=NORMAL 1=ANOMALY
    prediction_label: str
    confidence: float
    # Latency breakdown
    inference_latency_sec: float
    quantum_latency_sec: float
    classical_latency_sec: float

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp.isoformat(),
            "unit_id": self.unit_id,
            "temperature_c": round(self.temperature_c, 2),
            "vibration_rms": round(self.vibration_rms, 4),
            "current_a": round(self.current_a, 3),
            "health_score": round(self.health_score, 1),
            "is_injected_anomaly": self.is_injected_anomaly,
            "features_encoded": [round(v, 6) for v in self.features_encoded.tolist()],
            "kernel_row": [round(v, 6) for v in self.kernel_row.tolist()],
            "kernel_max": round(float(self.kernel_row.max()), 4),
            "kernel_mean": round(float(self.kernel_row.mean()), 4),
            "svm_decision": round(float(self.svm_decision), 6),
            "prediction": self.prediction,
            "prediction_label": self.prediction_label,
            "confidence": round(float(self.confidence), 4),
            "inference_latency_sec": round(self.inference_latency_sec, 3),
            "quantum_latency_sec": round(self.quantum_latency_sec, 3),
            "classical_latency_sec": round(self.classical_latency_sec, 4),
        }


class QENIInferenceEngine:
    """
    Production inference engine for QENI.
    
    Architecture:
      1. Read sensor features (on-device, private)
      2. Encode to [0, 2π]^4 (on-device, classical)
      3. Compute K(x_new, x_train_i) via IBM QPU or simulator (batched)
      4. SVM decision function (on-device, classical, microseconds)
      5. Emit result to dashboard via callback
    
    Privacy guarantee:
      Only step 3 involves any network communication.
      Only encoded feature vectors leave the device.
      Raw sensor readings never transmitted.
    """
    def __init__(self, config: dict, model_dir: str = "models/"):
        self.config = config
        self.model_dir = model_dir
        self.svm = None
        self.encoder: Optional[FeatureEncoder] = None
        self.X_train_enc: Optional[np.ndarray] = None
        self.quantum_engine: Optional[QuantumKernelEngine] = None
        self._callbacks: List[Callable] = []
        self._running = False
        self._stats = {
            "total_inferences": 0,
            "total_anomalies": 0,
            "total_quantum_time": 0.0,
            "session_start": datetime.utcnow().isoformat(),
        }

    def load_model(self) -> bool:
        try:
            self.svm = joblib.load(f"{self.model_dir}/svm_latest.joblib")
            self.encoder = joblib.load(f"{self.model_dir}/encoder_latest.joblib")
            self.X_train_enc = np.load(f"{self.model_dir}/X_train_enc_latest.npy")

            with open(f"{self.model_dir}/model_info.json") as f:
                info = json.load(f)

            logger.success(
                f"Model loaded | trained={info.get('trained_at','?')} | "
                f"acc={info.get('accuracy',0):.1%} | "
                f"n_train={len(self.X_train_enc)}"
            )
            return True
        except FileNotFoundError as e:
            logger.error(f"Model not found: {e}")
            logger.error("Run: python scripts/train.py")
            return False

    def setup_quantum(self, use_real_hardware: bool = None) -> bool:
        self.quantum_engine = QuantumKernelEngine(self.config["quantum"])
        ok = self.quantum_engine.setup(use_real_hardware=use_real_hardware)
        return ok

    def on_result(self, callback: Callable[[InferenceResult], None]):
        self._callbacks.append(callback)

    def get_stats(self) -> dict:
        n = self._stats["total_inferences"]
        return {
            **self._stats,
            "anomaly_rate": (
                self._stats["total_anomalies"] / n if n > 0 else 0.0
            ),
            "avg_quantum_latency": (
                self._stats["total_quantum_time"] / n if n > 0 else 0.0
            ),
        }

    def run_single_inference(self, sensor_manager: SensorManager) -> InferenceResult:
        t_total = time.time()

        # 1. Read sensors — private, stays on device
        reading = sensor_manager.get_latest_reading()
        x_raw = reading.to_feature_vector()

        # 2. Encode — classical, on-device
        t_classical = time.time()
        x_encoded = self.encoder.transform(x_raw)[0]
        t_enc = time.time() - t_classical

        # 3. Quantum kernel — batched, only encoded features transmitted
        t_q = time.time()
        kernel_row = self.quantum_engine.compute_inference_kernel_row(
            x_encoded, self.X_train_enc
        )
        t_quantum = time.time() - t_q

        # 4. SVM decision — classical, microseconds, on-device
        t_svm = time.time()
        K_inf = kernel_row.reshape(1, -1)
        prediction = int(self.svm.predict(K_inf)[0])
        decision = float(self.svm.decision_function(K_inf)[0])
        prob = self.svm.predict_proba(K_inf)[0]
        confidence = float(prob[prediction])
        t_classical_total = t_enc + (time.time() - t_svm)

        result = InferenceResult(
            timestamp=datetime.utcnow(),
            unit_id=sensor_manager.unit_id,
            temperature_c=reading.temperature_c,
            vibration_rms=reading.vibration_rms,
            current_a=reading.current_a,
            health_score=reading.health_score(),
            is_injected_anomaly=reading.is_injected_anomaly,
            features_encoded=x_encoded,
            kernel_row=kernel_row,
            svm_decision=decision,
            prediction=prediction,
            prediction_label="ANOMALY" if prediction == 1 else "NORMAL",
            confidence=confidence,
            inference_latency_sec=time.time() - t_total,
            quantum_latency_sec=t_quantum,
            classical_latency_sec=t_classical_total,
        )

        # Update stats
        self._stats["total_inferences"] += 1
        if prediction == 1:
            self._stats["total_anomalies"] += 1
        self._stats["total_quantum_time"] += t_quantum

        label = result.prediction_label
        marker = "⚠" if prediction == 1 else "✓"
        logger.info(
            f"{marker} Unit {sensor_manager.unit_id}: {label} | "
            f"dec={decision:+.3f} conf={confidence:.0%} | "
            f"Q={t_quantum:.2f}s cls={t_classical_total*1000:.1f}ms"
        )

        for cb in self._callbacks:
            try:
                cb(result)
            except Exception as e:
                logger.error(f"Callback error: {e}")

        return result

    def run_continuous(
        self,
        sensor_managers: List[SensorManager],
        interval_sec: float = 45.0,
    ):
        self._running = True
        logger.info(
            f"QENI inference loop started | "
            f"{len(sensor_managers)} units | interval={interval_sec}s"
        )

        # Prime sensor history
        for sm in sensor_managers:
            sm.read_fused()

        while self._running:
            # Read all sensors first
            for sm in sensor_managers:
                sm.read_fused()

            # Run inference for all units
            for sm in sensor_managers:
                if not self._running:
                    break
                try:
                    self.run_single_inference(sm)
                except Exception as e:
                    logger.error(f"Inference error unit {sm.unit_id}: {e}")

            if self._running:
                logger.debug(f"Sleeping {interval_sec}s")
                time.sleep(interval_sec)

    def stop(self):
        self._running = False
        if self.quantum_engine:
            self.quantum_engine.close()
        logger.info("Inference engine stopped")
