import random
import time
import math
import numpy as np
from datetime import datetime
from loguru import logger
from sensors.base_sensor import BaseSensor, SensorReading


class MPU6050Sensor(BaseSensor):
    PWR_MGMT_1 = 0x6B
    ACCEL_XOUT_H = 0x3B
    GYRO_XOUT_H = 0x43
    ACCEL_SCALE = 16384.0
    GYRO_SCALE = 131.0
    _sim_phase = 0.0

    def setup(self) -> bool:
        try:
            import smbus2
            self.bus = smbus2.SMBus(1)
            self.addr = self.config.get("mpu6050_i2c_addr", 0x68)
            self.bus.write_byte_data(self.addr, self.PWR_MGMT_1, 0)
            time.sleep(0.1)
            self.hardware_ok = True
            logger.success(f"MPU-6050 online at 0x{self.addr:02x}")
            return True
        except Exception as e:
            logger.warning(f"MPU-6050 not found — simulating ({e})")
            self.bus = None
            return False

    def _read_word_2c(self, reg: int) -> float:
        high = self.bus.read_byte_data(self.addr, reg)
        low = self.bus.read_byte_data(self.addr, reg + 1)
        val = (high << 8) + low
        if val >= 0x8000:
            val = -((65535 - val) + 1)
        return val

    def _simulate(self, inject_anomaly: bool = False):
        MPU6050Sensor._sim_phase += 0.05
        p = MPU6050Sensor._sim_phase
        if inject_anomaly:
            ax = random.gauss(0, 1.5)
            ay = random.gauss(0, 1.5)
            az = 9.81 + random.gauss(0, 2.0)
        else:
            ax = 0.05 * math.sin(p) + random.gauss(0, 0.02)
            ay = 0.03 * math.cos(p) + random.gauss(0, 0.02)
            az = 9.81 + 0.02 * math.sin(p * 2) + random.gauss(0, 0.05)
        gx = random.gauss(0, 0.1)
        gy = random.gauss(0, 0.1)
        gz = random.gauss(0, 0.05)
        return ax, ay, az, gx, gy, gz

    def read(self, inject_anomaly: bool = False) -> SensorReading:
        if self.hardware_ok and self.bus:
            try:
                ax = self._read_word_2c(self.ACCEL_XOUT_H) / self.ACCEL_SCALE * 9.81
                ay = self._read_word_2c(self.ACCEL_XOUT_H + 2) / self.ACCEL_SCALE * 9.81
                az = self._read_word_2c(self.ACCEL_XOUT_H + 4) / self.ACCEL_SCALE * 9.81
                gx = self._read_word_2c(self.GYRO_XOUT_H) / self.GYRO_SCALE
                gy = self._read_word_2c(self.GYRO_XOUT_H + 2) / self.GYRO_SCALE
                gz = self._read_word_2c(self.GYRO_XOUT_H + 4) / self.GYRO_SCALE
            except Exception:
                ax, ay, az, gx, gy, gz = self._simulate(inject_anomaly)
        else:
            ax, ay, az, gx, gy, gz = self._simulate(inject_anomaly)

        vib = float(np.sqrt(ax**2 + ay**2 + (az - 9.81)**2))
        return SensorReading(
            timestamp=datetime.utcnow(), unit_id=self.unit_id,
            temperature_c=0.0,
            vibration_rms=vib,
            current_a=0.0,
            accel_x=ax, accel_y=ay, accel_z=az,
            gyro_x=gx, gyro_y=gy, gyro_z=gz,
        )
