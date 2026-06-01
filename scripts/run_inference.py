#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import threading
from dotenv import load_dotenv
from loguru import logger

from sensors.sensor_manager import SensorManager
from inference.qeni_inference import QENIInferenceEngine
from dashboard.app import app, socketio, push_result, set_backend_name

load_dotenv()
logger.add("logs/qeni_{time}.log", rotation="1 day", retention="14 days", compression="gz")


def main():
    parser = argparse.ArgumentParser(description="QENI Inference Runtime")
    parser.add_argument("--simulator", action="store_true",
                        help="Use AerSimulator (no IBM account needed)")
    parser.add_argument("--interval", type=float, default=45.0,
                        help="Inference interval in seconds")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("QENI — Quantum Edge Neural Inference")
    logger.info("Raspberry Pi 5 | Privacy-Preserving Distributed Quantum ML")
    logger.info("=" * 60)

    # Load model
    engine = QENIInferenceEngine(config, model_dir=config["pipeline"]["model_dir"])
    if not engine.load_model():
        logger.error("Run training first: python scripts/train.py")
        sys.exit(1)

    # Setup quantum backend
    use_real = not args.simulator and bool(os.environ.get("IBM_QUANTUM_TOKEN"))
    engine.setup_quantum(use_real_hardware=use_real)

    backend_name = config["quantum"]["backend"] if use_real else "AerSimulator"
    set_backend_name(backend_name)
    logger.info(f"Quantum backend: {backend_name}")

    # Register dashboard callback
    engine.on_result(lambda r: push_result(r.to_dict()))

    # Initialize sensor managers
    sensor_managers = []
    for uid in range(1, config["dashboard"]["robot_units"] + 1):
        sm = SensorManager(uid, config["sensors"])
        sm.setup()
        sensor_managers.append(sm)

    # Start inference loop
    def inference_thread():
        engine.run_continuous(sensor_managers, interval_sec=args.interval)

    t = threading.Thread(target=inference_thread, daemon=True)
    t.start()

    pi_ip = os.popen("hostname -I").read().strip().split()[0]
    logger.success(f"Dashboard: http://{pi_ip}:{config['dashboard']['port']}")
    logger.success(f"Also:      http://localhost:{config['dashboard']['port']}")
    logger.info(f"Inference interval: {args.interval}s")
    logger.info("Press Ctrl+C to stop")

    socketio.run(
        app,
        host=config["dashboard"]["host"],
        port=config["dashboard"]["port"],
        debug=False,
    )


if __name__ == "__main__":
    main()
