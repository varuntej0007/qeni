#!/usr/bin/env python3
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
from dotenv import load_dotenv
from loguru import logger
from core.training import train_quantum_svm

load_dotenv()
logger.add("logs/training_{time}.log", rotation="10 MB")


def main():
    parser = argparse.ArgumentParser(description="QENI Training Pipeline")
    parser.add_argument("--real-hardware", action="store_true",
                        help="Use real IBM QPU (requires IBM_QUANTUM_TOKEN in .env)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to labeled CSV (columns: temperature_c, vibration_rms, current_a, accel_magnitude, label)")
    parser.add_argument("--samples", type=int, default=40,
                        help="Number of synthetic samples if no --data provided")
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    logger.info("=" * 60)
    logger.info("QENI Training Pipeline — Production Run")
    logger.info("=" * 60)

    if args.real_hardware:
        if not os.environ.get("IBM_QUANTUM_TOKEN"):
            logger.error("IBM_QUANTUM_TOKEN not set in .env — cannot use real hardware")
            sys.exit(1)
        logger.info(f"Backend: IBM QPU ({config['quantum']['backend']})")
        logger.info("WARNING: This will use IBM Quantum credits")
    else:
        logger.info("Backend: AerSimulator (local, free, fast)")

    metrics = train_quantum_svm(
        config=config["quantum"],
        data_path=args.data,
        use_real_hardware=args.real_hardware,
        model_dir=config["pipeline"]["model_dir"],
    )

    logger.success("=" * 60)
    logger.success("TRAINING COMPLETE")
    logger.success("=" * 60)
    logger.success(f"  Accuracy:          {metrics['accuracy']:.1%}")
    logger.success(f"  F1 (normal):       {metrics['f1_normal']:.3f}")
    logger.success(f"  F1 (anomaly):      {metrics['f1_anomaly']:.3f}")
    logger.success(f"  ROC-AUC:           {metrics['roc_auc']:.3f}")
    logger.success(f"  CV F1 mean:        {metrics['cv_f1_mean']:.3f}")
    logger.success(f"  Support vectors:   {metrics['n_support_vectors']}")
    logger.success(f"  Model saved:       {config['pipeline']['model_dir']}")
    logger.info("\nNext: python scripts/run_inference.py --simulator")


if __name__ == "__main__":
    main()
