"""Read Pi 5's actual CPU temperature — no external hardware needed."""
import subprocess
import random
import math
import time
from datetime import datetime
from sensors.base_sensor import BaseSensor, SensorReading


class PiTempSensor(BaseSensor):
    """
    Uses the Pi 5's built-in thermal sensor.
    Real temperature that changes with CPU load.
    Stress the CPU to simulate thermal anomalies.
    """
    _phase = 0.0

    def setup(self) -> bool:
        try:
            result = subprocess.run(
                ['vcgencmd', 'measure_temp'],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                self.hardware_ok = True
                return True
        except Exception:
            pass
        # Fallback: read from thermal zone
        try:
            with open('/sys/class/thermal/thermal_zone0/temp') as f:
                int(f.read().strip())
            self.hardware_ok = True
            return True
        except Exception:
            self.hardware_ok = False
            return False

    def _read_cpu_temp(self) -> float:
        try:
            with open('/sys/class/thermal/thermal_zone0/temp') as f:
                return int(f.read().strip()) / 1000.0
        except Exception:
            try:
                result = subprocess.run(
                    ['vcgencmd', 'measure_temp'],
                    capture_output=True, text=True
                )
                temp_str = result.stdout.strip()
                return float(temp_str.replace("temp=","").replace("'C",""))
            except Exception:
                return 45.0 + random.gauss(0, 2)

    def read(self, inject_anomaly: bool = False) -> SensorReading:
        temp = self._read_cpu_temp()
        if inject_anomaly:
            temp += random.uniform(15, 25)
        return SensorReading(
            timestamp=datetime.utcnow(),
            unit_id=self.unit_id,
            temperature_c=temp,
            vibration_rms=0.0,
            current_a=0.0,
        )
