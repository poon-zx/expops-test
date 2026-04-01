from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec, ed25519, padding, rsa

from expops.core import log_metric, process, step

logger = logging.getLogger(__name__)


def _b64_encode(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _b64_decode(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def _mean_ms(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=float)))


@process()
def data_generation(
    payload_size_bytes: int = 2,
    random_seed: int = 42,
) -> Dict[str, Any]:
    """
    Generate a deterministic payload for RSA encryption benchmarks.

    ``payload_size_bytes`` is normally set from ``configs/project_config.yaml`` (e.g. the
    paper-style 2 / 3 / 4 byte matrix). The payload is deterministic for a given
    ``random_seed`` and size.

    Parameters
    ----------
    payload_size_bytes : int, default=2
        Size of the generated payload in bytes.
    random_seed : int, default=42
        Seed used by the NumPy RNG to generate the payload bytes.

    Returns
    -------
    dict
        Dictionary containing:

        - **payload_b64** (`str`): Base64-encoded payload bytes.
        - **payload_bytes** (`int`): Payload size in bytes (for downstream metadata).
    """

    @step()
    def generate_payload(payload_size_bytes: int = 2, random_seed: int = 42) -> Dict[str, Any]:
        """
        Generate base64-encoded random bytes.

        Parameters
        ----------
        payload_size_bytes : int, default=2
            Size of the generated payload in bytes.
        random_seed : int, default=42
            Seed used by the NumPy RNG to generate the payload bytes.

        Returns
        -------
        dict
            Dictionary containing:

            - **payload_b64** (`str`): Base64-encoded payload bytes.
            - **payload_bytes** (`int`): Payload size in bytes.
        """
        ps_inner = int(payload_size_bytes)
        rng = np.random.default_rng(int(random_seed))
        payload = rng.integers(0, 256, size=ps_inner, dtype=np.uint8).tobytes()
        return {"payload_b64": _b64_encode(payload), "payload_bytes": ps_inner}

    return generate_payload(payload_size_bytes=payload_size_bytes, random_seed=random_seed)


@dataclass(frozen=True)
class _BenchResult:
    mean_keygen_ms: float
    mean_sign_ms: float
    mean_verify_ms: float


def _log_trial_series(metric_name: str, values_ms: list[float]) -> None:
    for i, v in enumerate(values_ms, start=1):
        log_metric(metric_name, float(v), step=i)


def _log_means(result: _BenchResult) -> None:
    log_metric("mean_keygen_ms", float(result.mean_keygen_ms))
    log_metric("mean_sign_ms", float(result.mean_sign_ms))
    log_metric("mean_verify_ms", float(result.mean_verify_ms))


_OAEP_PADDING = padding.OAEP(
    mgf=padding.MGF1(algorithm=hashes.SHA256()),
    algorithm=hashes.SHA256(),
    label=None,
)


@process()
def define_rsa_bench(
    payload_b64: str,
    key_size: int = 2048,
    num_trials: int = 100,
) -> Dict[str, Any]:
    """
    Benchmark mean RSA encryption and decryption latency (OAEP + SHA-256), paper-style.

    For each trial, generates a fresh RSA key pair (not timed), encrypts the payload
    with the public key (timed), then decrypts to verify correctness (timed).

    Parameters
    ----------
    payload_b64 : str
        Base64-encoded plaintext bytes to encrypt.
    key_size : int, default=2048
        RSA modulus size in bits (set from ``configs/project_config.yaml`` for this demo).
        OAEP-SHA256 needs a large enough modulus for the plaintext; additionally, the
        `cryptography` backend used here enforces RSA key_size ≥ 1024.
    num_trials : int, default=10
        Number of benchmark trials.

    Returns
    -------
    dict
        Dictionary containing:

        - **algorithm** (`str`): Always ``"RSA"``.
        - **key_size** (`int`): Key size in bits.
        - **payload_bytes** (`int`): Plaintext length in bytes.
        - **num_trials** (`int`): Number of executed trials.
        - **mean_encrypt_ms** (`float`): Mean encryption latency in milliseconds.
        - **mean_decrypt_ms** (`float`): Mean decryption latency in milliseconds.

    Raises
    ------
    ValueError
        If decryption does not recover the plaintext.
    """
    payload = _b64_decode(str(payload_b64))
    key_size_i = int(key_size)
    num_trials_i = int(num_trials)

    @step()
    def run_trials() -> Dict[str, Any]:
        """
        Execute RSA encryption/decryption benchmark trials and log mean latencies.

        Returns
        -------
        dict
            Benchmark summary payload (see ``define_rsa_bench``).

        Raises
        ------
        ValueError
            If decryption does not recover the plaintext.
        """
        encrypt_ms: list[float] = []
        decrypt_ms: list[float] = []

        for trial_i in range(num_trials_i):
            priv = rsa.generate_private_key(public_exponent=65537, key_size=key_size_i)
            pub = priv.public_key()

            t1 = time.perf_counter()
            ct = pub.encrypt(payload, _OAEP_PADDING)
            encrypt_ms.append((time.perf_counter() - t1) * 1000.0)

            t2 = time.perf_counter()
            pt = priv.decrypt(ct, _OAEP_PADDING)
            decrypt_ms.append((time.perf_counter() - t2) * 1000.0)
            if pt != payload:
                raise ValueError("RSA OAEP decryption did not recover plaintext")

            step_i = trial_i + 1
            log_metric("encrypt_ms", float(encrypt_ms[-1]), step=step_i)
            log_metric("decrypt_ms", float(decrypt_ms[-1]), step=step_i)

        mean_enc = _mean_ms(encrypt_ms)
        mean_dec = _mean_ms(decrypt_ms)
        log_metric("mean_encrypt_ms", float(mean_enc))
        log_metric("mean_decrypt_ms", float(mean_dec))
        return {
            "algorithm": "RSA",
            "key_size": key_size_i,
            "payload_bytes": len(payload),
            "num_trials": num_trials_i,
            "mean_encrypt_ms": mean_enc,
            "mean_decrypt_ms": mean_dec,
        }

    return run_trials()

