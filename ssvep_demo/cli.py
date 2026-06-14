from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import welch

from .detect import cca_scores, predict_from_scores, spectral_snr_scores
from .simulate import simulate_trials


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate and detect SSVEP frequencies.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="Run the simulated SSVEP detection demo.")
    run.add_argument("--frequencies", nargs="+", type=float, default=[8.0, 10.0, 12.0, 15.0])
    run.add_argument("--trials-per-frequency", type=int, default=6)
    run.add_argument("--sample-rate", type=float, default=250.0)
    run.add_argument("--duration", type=float, default=3.0)
    run.add_argument("--noise-scale", type=float, default=0.65)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--harmonics", type=int, default=2)
    run.add_argument("--out", type=Path, default=Path("reports/demo"))
    return parser


def run_demo(args: argparse.Namespace) -> int:
    simulated = simulate_trials(
        frequencies=args.frequencies,
        trials_per_frequency=args.trials_per_frequency,
        sample_rate=args.sample_rate,
        duration=args.duration,
        noise_scale=args.noise_scale,
        seed=args.seed,
    )
    args.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, trial in enumerate(simulated.data):
        cca = cca_scores(trial, simulated.frequencies, simulated.sample_rate, args.harmonics)
        snr = spectral_snr_scores(trial, simulated.frequencies, simulated.sample_rate)
        rows.append(
            {
                "trial": idx,
                "true_frequency": simulated.labels[idx],
                "cca_prediction": predict_from_scores(cca),
                "spectral_prediction": predict_from_scores(snr),
                "cca_scores": json.dumps(cca, sort_keys=True),
                "spectral_snr_scores": json.dumps(snr, sort_keys=True),
            }
        )

    results = pd.DataFrame(rows)
    trial_manifest = pd.DataFrame(
        {
            "trial": range(len(simulated.labels)),
            "true_frequency": simulated.labels,
            "sample_rate_hz": simulated.sample_rate,
            "duration_seconds": args.duration,
            "channels": ",".join(simulated.channels),
        }
    )
    trial_manifest.to_csv(args.out / "simulated_trials.csv", index=False)
    results.to_csv(args.out / "detection_results.csv", index=False)

    summary = {
        "n_trials": int(len(results)),
        "candidate_frequencies_hz": list(simulated.frequencies),
        "cca_accuracy": float((results["cca_prediction"] == results["true_frequency"]).mean()),
        "spectral_accuracy": float((results["spectral_prediction"] == results["true_frequency"]).mean()),
        "sample_rate_hz": simulated.sample_rate,
        "duration_seconds": args.duration,
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_spectrum_plot(simulated.data[0], simulated.sample_rate, simulated.labels[0], args.out)
    print(json.dumps(summary, indent=2))
    return 0


def write_spectrum_plot(trial, sample_rate: float, true_frequency: float, out_dir: Path) -> None:
    freqs, psd = welch(trial.mean(axis=0), fs=sample_rate, nperseg=min(trial.shape[1], 512))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(freqs, psd, color="#1f77b4", linewidth=1.5)
    ax.axvline(true_frequency, color="#d62728", linestyle="--", label=f"target {true_frequency:g} Hz")
    ax.set_xlim(4, 32)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density")
    ax.set_title("Example Simulated SSVEP Trial")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_dir / "example_trial_spectrum.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return run_demo(args)
    parser.error(f"Unknown command: {args.command}")
    return 2
