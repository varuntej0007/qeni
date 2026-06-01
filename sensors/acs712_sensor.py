import random
import math
from datetime import datetime
from loguru import logger
from sensors.base_sensor import BaseSensor, SensorReading


class ACS712Sensor(BaseSensor):
    SENSITIVITY_MV_PER_A = 185.0
    VCC = 3.3
    ADC_BITS = 10
    _sim_phase = 0.0

    def setup(self) -> bool:
        try:
            import spidev
            self.spi = spidev.SpiDev()
            self.spi.open(0, 0)
            self.spi.max_speed_hz = 1350000
            self.channel = self.config.get("mcp3008_channel", 0)
            self.hardware_ok = True
            logger.success(f"ACS712/MCP3008 online on SPI channel {self.channel}")
            return True
        except Exception as e:
            logger.warning(f"SPI not found — simulating ({e})")
            self.spi = None
            return False

    def _read_adc(self) -> int:
        r = self.spi.xfer2([1, (8 + self.channel) << 4, 0])
        return ((r[1] & 3) << 8) + r[2]

    def _simulate(self, inject_anomaly: bool = False) -> float:
        ACS712Sensor._sim_phase += 0.03
        p = ACS712Sensor._sim_phase
        if inject_anomaly:
            return random.uniform(5.0, 8.0)
        return 1.5 + 0.2 * math.sin(p) + random.gauss(0, 0.08)

    def read(self, inject_anomaly: bool = False) -> SensorReading:
        if self.hardware_ok and self.spi:
            try:
                adc = self._read_adc()
                voltage_mv = (adc / (2**self.ADC_BITS - 1)) * self.VCC * 1000
                current = (voltage_mv - (self.VCC * 1000 / 2.0)) / self.SENSITIVITY_MV_PER_A
            except Exception:
                current = self._simulate(inject_anomaly)
        else:
            current = self._simulate(inject_anomaly)

        return SensorReading(
            timestamp=datetime.utcnow(), unit_id=self.unit_id,
            temperature_c=0.0, vibration_rms=0.0, current_a=current,
        )
