import random
import time
from datetime import datetime
from loguru import logger
from sensors.base_sensor import BaseSensor, SensorReading


class TMP102Sensor(BaseSensor):
    TMP102_REG_TEMP = 0x00
    _sim_temp = 35.0
    _sim_direction = 1

    def setup(self) -> bool:
        try:
            import smbus2
            self.bus = smbus2.SMBus(1)
            self.addr = self.config.get("temperature_i2c_addr", 0x48)
            self.bus.read_i2c_block_data(self.addr, self.TMP102_REG_TEMP, 2)
            self.hardware_ok = True
            logger.success(f"TMP102 online at 0x{self.addr:02x}")
            return True
        except Exception as e:
            logger.warning(f"TMP102 not found — simulating ({e})")
            self.bus = None
            return False

    def _read_hw(self) -> float:
        data = self.bus.read_i2c_block_data(self.addr, self.TMP102_REG_TEMP, 2)
        raw = ((data[0] << 8) | data[1]) >> 4
        if raw > 2047:
            raw -= 4096
        return raw * 0.0625

    def _simulate(self, inject_anomaly: bool = False) -> float:
        # Realistic slow thermal drift
        TMP102Sensor._sim_temp += TMP102Sensor._sim_direction * random.uniform(0.02, 0.1)
        if TMP102Sensor._sim_temp > 42: TMP102Sensor._sim_direction = -1
        if TMP102Sensor._sim_temp < 30: TMP102Sensor._sim_direction = 1
        base = TMP102Sensor._sim_temp + random.gauss(0, 0.3)
        return random.uniform(55, 72) if inject_anomaly else base

    def read(self, inject_anomaly: bool = False) -> SensorReading:
        if self.hardware_ok and self.bus:
            try:
                temp = self._read_hw()
            except Exception:
                temp = self._simulate(inject_anomaly)
        else:
            temp = self._simulate(inject_anomaly)
        return SensorReading(
            timestamp=datetime.utcnow(), unit_id=self.unit_id,
            temperature_c=temp, vibration_rms=0.0, current_a=0.0,
        )
