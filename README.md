# EEG SSVEP Frequency Detection Demo

A compact, locally runnable Python demo for steady-state visual evoked potential
(SSVEP) target-frequency detection. The project simulates short multi-channel
occipital EEG trials, then predicts the stimulus frequency with two simple
training-free approaches:

- spectral SNR around candidate frequencies
- canonical correlation analysis (CCA) against sine/cosine references

This is intended as a research utility and teaching scaffold, not a clinical or
production BCI decoder.

## Why this exists

SSVEP BCIs often identify the attended flicker target by measuring the EEG
response at known stimulation frequencies. MNE's SSVEP tutorial demonstrates
frequency-tagged spectral analysis and SNR measurement, while Lin et al. showed
that CCA can recognize SSVEP frequencies by correlating EEG with sinusoidal
reference templates.

References:

- MNE-Python, "Frequency-tagging: Basic analysis of an SSVEP/vSSR dataset":
  https://mne.tools/stable/auto_tutorials/time-freq/50_ssvep.html
- Lin et al., "Frequency recognition based on canonical correlation analysis
  for SSVEP-based BCIs", IEEE Transactions on Biomedical Engineering, 2006:
  https://pubmed.ncbi.nlm.nih.gov/17549911/

## Install

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run the demo

```powershell
python -m ssvep_demo run --out reports/demo
```

Expected outputs:

- `reports/demo/simulated_trials.csv`
- `reports/demo/detection_results.csv`
- `reports/demo/summary.json`
- `reports/demo/example_trial_spectrum.png`

## Example CLI options

```powershell
python -m ssvep_demo run --frequencies 8 10 12 15 --trials-per-frequency 8 --duration 3 --noise-scale 0.9 --out reports/noisy_demo
```

## Verification

```powershell
python -m pytest
python -m ssvep_demo run --out reports/demo
```

## Notes

- The synthetic generator creates four EEG-like channels with strongest SSVEP
  amplitude over occipital-style channels `O1` and `O2`.
- The CCA detector uses harmonics by default because SSVEP responses commonly
  contain frequency-locked harmonic energy.
- The spectral detector reports a simple local SNR ratio around each candidate
  frequency and is useful as a transparent baseline.
