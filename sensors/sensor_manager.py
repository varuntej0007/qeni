import numpy as np
import random
import time
from collections import deque
from datetime import datetime
from loguru import logger

from sensors.tmp102_sensor import TMP102Sensor
from sensors.mpu6050_sensor import MPU6050Sensor
from sensors.acs712_sensor import ACS712Sensor
from sensors.base_sensor import SensorReading


class SensorManager:
    """
    Fuses three sensors into a single SensorReading per robot unit.
    Supports realistic anomaly injection for demo and training data collection.
    
    Anomaly types modelled:
      - Thermal runaway (motor overheating)
      - Bearing failure (high vibration)  
      - Electrical fault (current spike)
      - Combined failure (all three)
    """
    ANOMALY_TYPES = ["thermal", "bearing", "electrical", "combined"]

    def __init__(self, unit_id: int, config: dict):
        self.unit_id = unit_id
        self.config = config
        self.window_size = config.get("window_size", 10)
        self._history: deque = deque(maxlen=self.window_size)
        self._inject_rate = config.get("anomaly_inject_rate", 0.12)
        self._anomaly_counter = 0
        self._anomaly_type = None

        self.temp_sensor = TMP102Sensor(unit_id, config)
        self.imu_sensor = MPU6050Sensor(unit_id, config)
        self.current_sensor = ACS712Sensor(unit_id, config)

    def setup(self) -> bool:
        self.temp_sensor.setup()
        self.imu_sensor.setup()
        self.current_sensor.setup()
        hw = sum([self.temp_sensor.hardware_ok,
                  self.imu_sensor.hardware_ok,
                  self.current_sensor.hardware_ok])
        logger.info(f"Unit {self.unit_id}: {hw}/3 hardware sensors online")
        return True

    def _should_inject(self) -> tuple:
        """Returns (inject_bool, anomaly_type)."""
        if self._anomaly_counter > 0:
            self._anomaly_counter -= 1
            return True, self._anomaly_type
        if random.random() < self._inject_rate:
            self._anomaly_type = random.choice(self.ANOMALY_TYPES)
            self._anomaly_counter = random.randint(2, 5)  # Anomaly persists
            return True, self._anomaly_type
        return False, None

    def read_fused(self) -> SensorReading:
        inject, atype = self._should_inject()

        t_inject = inject and atype in ("thermal", "combined")
        v_inject = inject and atype in ("bearing", "combined")
        c_inject = inject and atype in ("electrical", "combined")

        t_r = self.temp_sensor.read(inject_anomaly=t_inject)
        i_r = self.imu_sensor.read(inject_anomaly=v_inject)
        c_r = self.current_sensor.read(inject_anomaly=c_inject)

        fused = SensorReading(
            timestamp=datetime.utcnow(),
            unit_id=self.unit_id,
            temperature_c=t_r.temperature_c,
            vibration_rms=i_r.vibration_rms,
            current_a=c_r.current_a,
            accel_x=i_r.accel_x, accel_y=i_r.accel_y, accel_z=i_r.accel_z,
            gyro_x=i_r.gyro_x, gyro_y=i_r.gyro_y, gyro_z=i_r.gyro_z,
            is_injected_anomaly=inject,
        )
        self._history.append(fused)
        return fused

    def get_feature_vector(self) -> np.ndarray:
        if not self._history:
            self.read_fused()
        return self._history[-1].to_feature_vector()

    def get_latest_reading(self) -> SensorReading:
        if not self._history:
            self.read_fused()
        return self._history[-1]

    def get_windowed_features(self) -> np.ndarray:
        while len(self._history) < self.window_size:
            self.read_fused()
            time.sleep(0.01)
        return np.array([r.to_feature_vector() for r in self._history])
