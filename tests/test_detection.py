import numpy as np

from ssvep_demo.detect import cca_scores, predict_from_scores
from ssvep_demo.simulate import simulate_trials


def test_cca_detects_clean_simulated_trials():
    simulated = simulate_trials(
        frequencies=(8.0, 10.0, 12.0),
        trials_per_frequency=3,
        noise_scale=0.3,
        seed=12,
    )
    predictions = []
    for trial in simulated.data:
        scores = cca_scores(trial, simulated.frequencies, simulated.sample_rate, harmonics=2)
        predictions.append(predict_from_scores(scores))
    accuracy = np.mean(np.asarray(predictions) == simulated.labels)
    assert accuracy >= 0.95


def test_simulator_shapes_are_consistent():
    simulated = simulate_trials(frequencies=(10.0,), trials_per_frequency=2, duration=2.0)
    assert simulated.data.shape == (2, 4, 500)
    assert simulated.labels.tolist() == [10.0, 10.0]
