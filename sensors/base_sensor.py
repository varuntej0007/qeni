from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
import time


@dataclass
class SensorReading:
    timestamp: datetime
    unit_id: int
    temperature_c: float
    vibration_rms: float
    current_a: float
    accel_x: float = 0.0
    accel_y: float = 0.0
    accel_z: float = 0.0
    gyro_x: float = 0.0
    gyro_y: float = 0.0
    gyro_z: float = 0.0
    is_injected_anomaly: bool = False

    def to_feature_vector(self) -> np.ndarray:
        """4-dim feature vector used by quantum kernel."""
        accel_mag = np.sqrt(self.accel_x**2 + self.accel_y**2 + self.accel_z**2)
        return np.array([
            self.temperature_c,
            self.vibration_rms,
            self.current_a,
            accel_mag if accel_mag > 0 else abs(self.vibration_rms),
        ], dtype=np.float64)

    def health_score(self) -> float:
        """Simple rule-based health score 0-100 for display."""
        score = 100.0
        if self.temperature_c > 45: score -= 30
        if self.temperature_c > 60: score -= 40
        if self.vibration_rms > 2.0: score -= 20
        if self.vibration_rms > 4.0: score -= 30
        if self.current_a > 3.5: score -= 20
        if self.current_a > 6.0: score -= 40
        return max(0.0, score)


class BaseSensor(ABC):
    def __init__(self, unit_id: int, config: dict):
        self.unit_id = unit_id
        self.config = config
        self.hardware_ok = False

    @abstractmethod
    def read(self) -> SensorReading:
        pass

    @abstractmethod
    def setup(self) -> bool:
        pass
