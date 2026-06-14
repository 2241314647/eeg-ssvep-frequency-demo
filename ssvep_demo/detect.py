from __future__ import annotations

import numpy as np
from scipy.signal import welch


def make_reference(
    frequency: float,
    sample_rate: float,
    n_samples: int,
    harmonics: int = 2,
) -> np.ndarray:
    t = np.arange(n_samples) / sample_rate
    columns = []
    for harmonic in range(1, harmonics + 1):
        columns.append(np.sin(2 * np.pi * harmonic * frequency * t))
        columns.append(np.cos(2 * np.pi * harmonic * frequency * t))
    return np.column_stack(columns)


def _first_canonical_correlation(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    qx, _ = np.linalg.qr(x)
    qy, _ = np.linalg.qr(y)
    singular_values = np.linalg.svd(qx.T @ qy, compute_uv=False)
    return float(np.clip(singular_values[0], 0.0, 1.0))


def cca_scores(
    trial: np.ndarray,
    candidate_frequencies: list[float] | tuple[float, ...],
    sample_rate: float,
    harmonics: int = 2,
) -> dict[float, float]:
    x = trial.T
    n_samples = trial.shape[1]
    scores = {}
    for frequency in candidate_frequencies:
        reference = make_reference(frequency, sample_rate, n_samples, harmonics)
        scores[float(frequency)] = _first_canonical_correlation(x, reference)
    return scores


def spectral_snr_scores(
    trial: np.ndarray,
    candidate_frequencies: list[float] | tuple[float, ...],
    sample_rate: float,
    bandwidth: float = 0.5,
    guard_band: float = 0.4,
    neighborhood: float = 2.5,
) -> dict[float, float]:
    signal = trial.mean(axis=0)
    freqs, psd = welch(signal, fs=sample_rate, nperseg=min(len(signal), 512))
    scores = {}
    for target in candidate_frequencies:
        target_mask = np.abs(freqs - target) <= bandwidth
        noise_mask = (np.abs(freqs - target) <= neighborhood) & (
            np.abs(freqs - target) >= guard_band
        )
        signal_power = float(np.mean(psd[target_mask])) if np.any(target_mask) else 0.0
        noise_power = float(np.mean(psd[noise_mask])) if np.any(noise_mask) else 1e-12
        scores[float(target)] = signal_power / max(noise_power, 1e-12)
    return scores


def predict_from_scores(scores: dict[float, float]) -> float:
    return max(scores.items(), key=lambda item: item[1])[0]
