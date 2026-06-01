import numpy as np
import os
import time
from typing import Optional, List
from loguru import logger

from qiskit.circuit.library import PauliFeatureMap
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler


class QuantumKernelEngine:
    """
    Production quantum kernel engine for QENI.
    K(x_i, x_j) = |<0|U†(x_j)·U(x_i)|0>|²  (compute-uncompute)
    
    Supports:
      - IBM Quantum free tier (ibm_quantum_platform, no Session needed)
      - AerSimulator local fallback
      - Batched circuit execution for speed
    """
    def __init__(self, config: dict, use_simulator: bool = False):
        self.config = config
        self.use_simulator = use_simulator
        self.n_qubits = config.get("n_qubits", 4)
        self.reps = config.get("feature_map_reps", 2)
        self.shots = config.get("shots", 2048)
        self.backend_name = config.get("backend", "ibm_fez")
        self.pauli_terms = config.get("pauli_terms", ["Z", "ZZ"])
        self._feature_map = None
        self._sampler = None
        self._backend = None
        self._use_ibm = False
        self._transpile_cache = {}

    def _build_feature_map(self):
        return PauliFeatureMap(
            feature_dimension=self.n_qubits,
            reps=self.reps,
            paulis=self.pauli_terms,
            entanglement="full",
        )

    def _build_kernel_circuit(self, x_i: np.ndarray, x_j: np.ndarray):
        key = (tuple(np.round(x_i, 4)), tuple(np.round(x_j, 4)))
        if key in self._transpile_cache:
            return self._transpile_cache[key]
        qc_i = self._feature_map.assign_parameters(x_i)
        qc_j_inv = self._feature_map.assign_parameters(x_j).inverse()
        circuit = qc_i.compose(qc_j_inv)
        circuit.measure_all()
        transpiled = transpile(circuit, backend=self._backend, optimization_level=1)
        if len(self._transpile_cache) > 500:
            self._transpile_cache.clear()
        self._transpile_cache[key] = transpiled
        return transpiled

    def setup(self, use_real_hardware: bool = None) -> bool:
        if use_real_hardware is None:
            use_real_hardware = (
                bool(os.environ.get("IBM_QUANTUM_TOKEN"))
                and not self.use_simulator
            )

        self._feature_map = self._build_feature_map()

        if use_real_hardware:
            logger.info(f"Connecting to IBM Quantum: {self.backend_name}")
            try:
                from qiskit_ibm_runtime import QiskitRuntimeService
                from qiskit_ibm_runtime import SamplerV2 as IBMSampler

                token = os.environ.get("IBM_QUANTUM_TOKEN")
                # ibm_quantum_platform = new channel name (replaces ibm_quantum)
                service = QiskitRuntimeService(
                    channel="ibm_quantum_platform",
                    token=token,
                )
                backend = service.least_busy(
                    operational=True,
                    min_num_qubits=self.n_qubits,
                )
                logger.info(f"Selected backend: {backend.name} "
                            f"({backend.status().pending_jobs} jobs pending)")

                # Open plan: NO Session — use backend directly
                self._sampler = IBMSampler(mode=backend)
                self._backend = backend
                self._use_ibm = True
                logger.success(f"IBM Quantum ready: {backend.name} (no-session mode)")

            except Exception as e:
                logger.warning(f"IBM connection failed: {e}")
                logger.warning("Falling back to AerSimulator")
                self._backend = AerSimulator()
                self._sampler = AerSampler()
                self._use_ibm = False
        else:
            logger.info("AerSimulator mode (local)")
            self._backend = AerSimulator()
            self._sampler = AerSampler()
            self._use_ibm = False

        logger.success(
            f"QuantumKernelEngine ready | "
            f"{'IBM QPU: ' + (self._backend.name if self._use_ibm else '') if self._use_ibm else 'Simulator'} | "
            f"PauliFeatureMap reps={self.reps} shots={self.shots}"
        )
        return True

    def _batch_evaluate(self, pairs: List[tuple]) -> np.ndarray:
        circuits = [self._build_kernel_circuit(x_i, x_j) for x_i, x_j in pairs]
        job = self._sampler.run(circuits, shots=self.shots)
        result = job.result()
        zero_state = "0" * self.n_qubits
        return np.array([
            result[i].data.meas.get_counts().get(zero_state, 0) / self.shots
            for i in range(len(circuits))
        ])

    def compute_kernel_matrix(
        self, X: np.ndarray, Y: np.ndarray = None
    ) -> np.ndarray:
        if self._sampler is None:
            raise RuntimeError("Call setup() first")
        symmetric = Y is None
        if symmetric:
            Y = X
        n, m = len(X), len(Y)
        pairs = []
        pair_index = {}
        K = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                if symmetric and j < i:
                    continue
                pairs.append((X[i], Y[j]))
                pair_index[(i, j)] = len(pairs) - 1

        logger.info(
            f"Kernel matrix {n}x{m}: {len(pairs)} circuits "
            f"(symmetry={'on' if symmetric else 'off'}) — batched"
        )
        t0 = time.time()
        values = self._batch_evaluate(pairs)
        for (i, j), idx in pair_index.items():
            K[i, j] = values[idx]
            if symmetric and i != j:
                K[j, i] = values[idx]
        logger.info(f"Kernel matrix done in {time.time()-t0:.1f}s ({len(pairs)} circuits batched)")
        return K

    def compute_inference_kernel_row(
        self, x_new: np.ndarray, X_train: np.ndarray
    ) -> np.ndarray:
        if self._sampler is None:
            raise RuntimeError("Call setup() first")
        x_new = np.atleast_2d(x_new)[0]
        n = len(X_train)
        logger.info(f"Inference kernel: 1 x {n} — single batched job")
        t0 = time.time()
        pairs = [(x_new, X_train[j]) for j in range(n)]
        K_row = self._batch_evaluate(pairs)
        logger.info(f"Inference kernel done in {time.time()-t0:.2f}s ({n} circuits batched)")
        return K_row

    def close(self):
        logger.info("QuantumKernelEngine closed")
        self._transpile_cache.clear()
