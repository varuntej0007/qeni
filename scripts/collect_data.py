#!/usr/bin/env python3
"""
Collect real labeled training data from sensors.
Press N=normal, A=anomaly, Q=quit.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import yaml
import argparse
from datetime import datetime
from loguru import logger
from sensors.sensor_manager import SensorManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unit", type=int, default=1)
    parser.add_argument("--output", type=str, default="data/labeled_training.csv")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    os.makedirs("data", exist_ok=True)
    sm = SensorManager(args.unit, config["sensors"])
    sm.setup()

    samples = []
    print("\nQENI Data Collection")
    print("N=NORMAL  A=ANOMALY  Q=QUIT AND SAVE")
    print("-" * 50)

    while True:
        reading = sm.read_fused()
        fv = reading.to_feature_vector()

        print(f"\n[{len(samples)+1}] {datetime.utcnow().strftime('%H:%M:%S')}")
        print(f"  Temperature: {fv[0]:.2f}°C  |  Health: {reading.health_score():.0f}%")
        print(f"  Vibration:   {fv[1]:.4f} m/s²")
        print(f"  Current:     {fv[2]:.3f} A")
        print(f"  Accel mag:   {fv[3]:.3f} m/s²")

        key = input("  Label [N/A/Q]: ").strip().upper()
        if key == "Q":
            break
        elif key in ("N", "A"):
            label = 0 if key == "N" else 1
            samples.append({
                "timestamp": reading.timestamp.isoformat(),
                "unit_id": args.unit,
                "temperature_c": fv[0],
                "vibration_rms": fv[1],
                "current_a": fv[2],
                "accel_magnitude": fv[3],
                "label": label,
            })
            print(f"  → {'NORMAL' if label==0 else 'ANOMALY'} saved ({len(samples)} total)")
        else:
            print("  Invalid key")

    if samples:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(samples[0].keys()))
            writer.writeheader()
            writer.writerows(samples)
        n_n = sum(1 for s in samples if s["label"] == 0)
        n_a = len(samples) - n_n
        logger.success(f"Saved {len(samples)} samples → {args.output}")
        logger.info(f"Normal: {n_n}  Anomaly: {n_a}")
        logger.info(f"Next: python scripts/train.py --data {args.output}")


if __name__ == "__main__":
    main()
