from __future__ import annotations

from dataclasses import dataclass

import numpy as np


CHANNELS = ("O1", "O2", "Pz", "Cz")


@dataclass(frozen=True)
class SimulatedSSVEP:
    data: np.ndarray
    labels: np.ndarray
    sample_rate: float
    channels: tuple[str, ...]
    frequencies: tuple[float, ...]


def simulate_trials(
    frequencies: list[float] | tuple[float, ...] = (8.0, 10.0, 12.0, 15.0),
    trials_per_frequency: int = 6,
    sample_rate: float = 250.0,
    duration: float = 3.0,
    noise_scale: float = 0.65,
    seed: int = 7,
) -> SimulatedSSVEP:
    """Create EEG-like SSVEP trials with target-frequency and harmonic energy."""
    rng = np.random.default_rng(seed)
    n_samples = int(round(sample_rate * duration))
    t = np.arange(n_samples) / sample_rate
    channel_gain = np.array([1.0, 0.92, 0.42, 0.18])
    trials: list[np.ndarray] = []
    labels: list[float] = []

    for freq in frequencies:
        for _ in range(trials_per_frequency):
            phase = rng.uniform(0, 2 * np.pi)
            harmonic_phase = rng.uniform(0, 2 * np.pi)
            ssvep = np.sin(2 * np.pi * freq * t + phase)
            ssvep += 0.45 * np.sin(2 * np.pi * 2 * freq * t + harmonic_phase)
            slow_drift = 0.15 * np.sin(2 * np.pi * rng.uniform(0.2, 0.8) * t)
            trial = channel_gain[:, None] * ssvep[None, :]
            trial += slow_drift
            trial += noise_scale * rng.normal(size=trial.shape)
            trials.append(trial)
            labels.append(float(freq))

    order = rng.permutation(len(trials))
    data = np.stack(trials, axis=0)[order]
    label_array = np.asarray(labels, dtype=float)[order]
    return SimulatedSSVEP(
        data=data,
        labels=label_array,
        sample_rate=sample_rate,
        channels=CHANNELS,
        frequencies=tuple(float(f) for f in frequencies),
    )
