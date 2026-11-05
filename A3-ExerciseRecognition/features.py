"""Window features for exercise recognition; no labels are used here."""

import numpy as np

from util import add_features


BASELINE_FEATURES = ["avg", "max", "med", "min", "q25", "q75", "std"]
FEATURE_GROUPS = {
    "axis_motion": ["std_x", "std_y", "std_z", "axis_dominance"],
    "rhythm": ["dominant_frequency", "spectral_concentration"],
    "correlation": ["corr_xy", "corr_xz", "corr_yz"],
    "intensity": ["rms_acceleration"],
}
EXTRA_FEATURES = [name for group in FEATURE_GROUPS.values() for name in group]
# Selected by the declared validation comparisons, before model tuning.
SELECTED_EXTRA_FEATURES = [name for name in EXTRA_FEATURES if name != "corr_xz"]


def candidate_features(window, sample_rate):
    """Return the seven baseline features and all ten proposed additions."""
    features = add_features(window)
    axes = window[["x", "y", "z"]].to_numpy(dtype=float)
    centered = axes - axes.mean(axis=0)
    deviations = centered.std(axis=0)
    for name, value in zip(("std_x", "std_y", "std_z"), deviations):
        features[name] = value
    total = float(deviations.sum())
    features["axis_dominance"] = float(deviations.max() / total) if total > 1e-12 else 0.0

    # Frequency of the axis with most motion; Hann taper reduces edge effects.
    # This describes rhythm, not necessarily the true rep rate (harmonics exist).
    signal = centered[:, int(np.argmax(deviations))]
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
    frequencies = np.fft.rfftfreq(len(signal), d=1 / sample_rate)
    band = (frequencies >= 0.2) & (frequencies <= 3.0)
    power = spectrum[band]
    if len(power) and power.sum() > 1e-12:
        peak = int(np.argmax(power))
        features["dominant_frequency"] = float(frequencies[band][peak])
        features["spectral_concentration"] = float(power[peak] / power.sum())
    else:
        features["dominant_frequency"] = 0.0
        features["spectral_concentration"] = 0.0

    for name, i, j in (("corr_xy", 0, 1), ("corr_xz", 0, 2), ("corr_yz", 1, 2)):
        first, second = centered[:, i], centered[:, j]
        denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
        value = float(np.dot(first, second) / denominator) if denominator > 1e-12 else 0.0
        features[name] = np.clip(value, -1.0, 1.0)
    features["rms_acceleration"] = float(np.sqrt(np.mean(np.sum(axes ** 2, axis=1))))
    return features.replace([np.inf, -np.inf], 0).fillna(0)


def add_my_features(window, sample_rate):
    """The selected feature set used by the notebook's Function 3."""
    return candidate_features(window, sample_rate)[BASELINE_FEATURES + SELECTED_EXTRA_FEATURES]
