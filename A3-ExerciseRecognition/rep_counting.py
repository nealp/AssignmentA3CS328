"""Count repetitions from a segmented recording using smoothed motion peaks."""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


# Selected using fit recordings only; see the notebook/work log for comparisons.
REP_SETTINGS = {
    "signal_kind": "pca",
    "cutoff_hz": 1.0,
    "min_distance_sec": 0.6,
    "prominence_fraction": 1.0,
}


def rep_details(data, sample_rate, signal_kind="axis", cutoff_hz=1.0,
                min_distance_sec=0.6, prominence_fraction=0.5):
    """Return the counted signal and exact peaks so our plots explain the count."""
    if signal_kind not in ("magnitude", "axis", "pca"):
        raise ValueError("signal_kind must be magnitude, axis, or pca")
    if sample_rate <= 0 or cutoff_hz <= 0 or min_distance_sec <= 0 or prominence_fraction < 0:
        raise ValueError("Sampling rate, cutoff, and peak distance must be positive")

    empty = np.array([], dtype=int)
    if len(data) < 10:
        return {"signal": np.zeros(len(data)), "peaks": empty, "signal_name": signal_kind}

    # Filter before choosing an axis/direction so brief high-frequency noise has
    # less influence on the choice. The supplied preparation filter is at 5 Hz.
    cutoff = min(cutoff_hz, 0.95 * sample_rate / 2)
    b, a = butter(2, cutoff / (sample_rate / 2), btype="lowpass")
    if signal_kind == "magnitude":
        values = data["accel_mag"].to_numpy(dtype=float)
        signal = filtfilt(b, a, values - values.mean())
        signal_name = "magnitude"
    else:
        axes = data[["x", "y", "z"]].to_numpy(dtype=float)
        axes = filtfilt(b, a, axes - axes.mean(axis=0), axis=0)
        if signal_kind == "axis":
            index = int(np.argmax(axes.std(axis=0)))
            signal = axes[:, index]
            signal_name = "xyz"[index]
        else:
            _, _, directions = np.linalg.svd(axes, full_matrices=False)
            direction = directions[0]
            # PCA's sign is arbitrary. Fix it to make counts/figures repeatable.
            direction *= np.sign(direction[np.argmax(np.abs(direction))]) or 1
            signal = axes @ direction
            signal_name = "first principal component"

    signal = signal - signal.mean()
    scale = float(signal.std())
    if scale < 0.01:
        # There is too little movement to distinguish meaningful peaks from noise.
        peaks = empty
    else:
        peaks, _ = find_peaks(
            signal,
            distance=max(round(min_distance_sec * sample_rate), 1),
            prominence=prominence_fraction * scale,
        )
    return {"signal": signal, "peaks": peaks, "signal_name": signal_name}


def count_reps(data, sample_rate, exercise):
    """Return a nonnegative count, using the same settings for all exercises."""
    # Keep the required exercise argument, but avoid placement- or label-based
    # guesses. No reported rep count or recording identity is used here.
    return int(len(rep_details(data, sample_rate, **REP_SETTINGS)["peaks"]))
