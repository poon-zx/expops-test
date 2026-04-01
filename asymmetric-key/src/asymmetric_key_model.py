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
    num_trials: int = 10,
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

        for _ in range(num_trials_i):
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


def _ec_curve_from_name(name: str) -> ec.EllipticCurve:
    n = str(name).strip().lower()
    if n in {"p-256", "secp256r1", "prime256v1"}:
        return ec.SECP256R1()
    if n in {"p-384", "secp384r1"}:
        return ec.SECP384R1()
    raise ValueError(f"Unsupported ECDSA curve: {name}")


@process()
def define_ecdsa_bench(
    payload_b64: str,
    curve: str = "secp256r1",
    num_trials: int = 100,
    hash_name: str = "sha256",
) -> Dict[str, Any]:
    """
    Benchmark ECDSA key generation, signing, and verification latency.

    For each trial, this process:
    - Generates a fresh ECDSA private key for the selected `curve`.
    - Signs the provided payload with ECDSA using the requested hash.
    - Verifies the signature with the corresponding public key.
    - Logs per-trial latencies and mean latencies (in milliseconds).

    Parameters
    ----------
    payload_b64 : str
        Base64-encoded payload bytes to sign and verify.
    curve : str, default="secp256r1"
        Elliptic curve name. Supported values include `"secp256r1"` (P-256) and
        `"secp384r1"` (P-384).
    num_trials : int, default=100
        Number of benchmark trials to run.
    hash_name : str, default="sha256"
        Hash algorithm for ECDSA. Supported values are `"sha256"` and `"sha384"`.

    Returns
    -------
    dict
        Dictionary containing:

        - **algorithm** (`str`): Always `"ECDSA"`.
        - **curve** (`str`): Curve name as provided.
        - **hash** (`str`): Hash name as provided.
        - **num_trials** (`int`): Number of executed trials.
        - **mean_keygen_ms** (`float`): Mean key-generation latency in milliseconds.
        - **mean_sign_ms** (`float`): Mean signing latency in milliseconds.
        - **mean_verify_ms** (`float`): Mean verification latency in milliseconds.

    Raises
    ------
    ValueError
        If an unsupported curve/hash is requested, or if signature verification
        fails for any trial.
    """
    payload = _b64_decode(str(payload_b64))
    num_trials = int(num_trials)
    curve_obj = _ec_curve_from_name(curve)

    def _hash_alg(hash_name: str) -> hashes.HashAlgorithm:
        hn = str(hash_name).strip().lower()
        if hn == "sha256":
            return hashes.SHA256()
        if hn == "sha384":
            return hashes.SHA384()
        raise ValueError(f"Unsupported hash for ECDSA: {hash_name}")

    hash_alg = _hash_alg(hash_name)

    @step()
    def run_trials() -> Dict[str, Any]:
        """
        Execute ECDSA benchmark trials and log metrics.

        Returns
        -------
        dict
            Benchmark summary payload (see `define_ecdsa_bench`).

        Raises
        ------
        ValueError
            If signature verification fails for any trial.
        """
        keygen_ms: list[float] = []
        sign_ms: list[float] = []
        verify_ms: list[float] = []

        for _ in range(num_trials):
            t0 = time.perf_counter()
            priv = ec.generate_private_key(curve_obj)
            keygen_ms.append((time.perf_counter() - t0) * 1000.0)

            msg = payload
            t1 = time.perf_counter()
            sig = priv.sign(msg, ec.ECDSA(hash_alg))
            sign_ms.append((time.perf_counter() - t1) * 1000.0)

            pub = priv.public_key()
            t2 = time.perf_counter()
            try:
                pub.verify(sig, msg, ec.ECDSA(hash_alg))
                ok = True
            except Exception:
                ok = False
            verify_ms.append((time.perf_counter() - t2) * 1000.0)
            if not ok:
                raise ValueError("ECDSA signature verification failed")

        _log_trial_series("keygen_ms", keygen_ms)
        _log_trial_series("sign_ms", sign_ms)
        _log_trial_series("verify_ms", verify_ms)

        result = _BenchResult(
            mean_keygen_ms=_mean_ms(keygen_ms),
            mean_sign_ms=_mean_ms(sign_ms),
            mean_verify_ms=_mean_ms(verify_ms),
        )
        _log_means(result)
        return {
            "algorithm": "ECDSA",
            "curve": str(curve),
            "hash": str(hash_name),
            "num_trials": num_trials,
            "mean_keygen_ms": result.mean_keygen_ms,
            "mean_sign_ms": result.mean_sign_ms,
            "mean_verify_ms": result.mean_verify_ms,
        }

    return run_trials()


@process()
def define_ed25519_bench(
    payload_b64: str,
    num_trials: int = 200,
) -> Dict[str, Any]:
    """
    Benchmark Ed25519 key generation, signing, and verification latency.

    For each trial, this process:
    - Generates a fresh Ed25519 private key.
    - Signs the provided payload.
    - Verifies the signature with the corresponding public key.
    - Logs per-trial latencies and mean latencies (in milliseconds).

    Parameters
    ----------
    payload_b64 : str
        Base64-encoded payload bytes to sign and verify.
    num_trials : int, default=200
        Number of benchmark trials to run.

    Returns
    -------
    dict
        Dictionary containing:

        - **algorithm** (`str`): Always `"Ed25519"`.
        - **num_trials** (`int`): Number of executed trials.
        - **mean_keygen_ms** (`float`): Mean key-generation latency in milliseconds.
        - **mean_sign_ms** (`float`): Mean signing latency in milliseconds.
        - **mean_verify_ms** (`float`): Mean verification latency in milliseconds.

    Raises
    ------
    ValueError
        If signature verification fails for any trial.
    """
    payload = _b64_decode(str(payload_b64))
    num_trials = int(num_trials)

    @step()
    def run_trials() -> Dict[str, Any]:
        """
        Execute Ed25519 benchmark trials and log metrics.

        Returns
        -------
        dict
            Benchmark summary payload (see `define_ed25519_bench`).

        Raises
        ------
        ValueError
            If signature verification fails for any trial.
        """
        keygen_ms: list[float] = []
        sign_ms: list[float] = []
        verify_ms: list[float] = []

        for _ in range(num_trials):
            t0 = time.perf_counter()
            priv = ed25519.Ed25519PrivateKey.generate()
            keygen_ms.append((time.perf_counter() - t0) * 1000.0)

            msg = payload
            t1 = time.perf_counter()
            sig = priv.sign(msg)
            sign_ms.append((time.perf_counter() - t1) * 1000.0)

            pub = priv.public_key()
            t2 = time.perf_counter()
            try:
                pub.verify(sig, msg)
                ok = True
            except Exception:
                ok = False
            verify_ms.append((time.perf_counter() - t2) * 1000.0)
            if not ok:
                raise ValueError("Ed25519 signature verification failed")

        _log_trial_series("keygen_ms", keygen_ms)
        _log_trial_series("sign_ms", sign_ms)
        _log_trial_series("verify_ms", verify_ms)

        result = _BenchResult(
            mean_keygen_ms=_mean_ms(keygen_ms),
            mean_sign_ms=_mean_ms(sign_ms),
            mean_verify_ms=_mean_ms(verify_ms),
        )
        _log_means(result)
        return {
            "algorithm": "Ed25519",
            "num_trials": num_trials,
            "mean_keygen_ms": result.mean_keygen_ms,
            "mean_sign_ms": result.mean_sign_ms,
            "mean_verify_ms": result.mean_verify_ms,
        }

    return run_trials()

