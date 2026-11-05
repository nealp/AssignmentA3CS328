"""Select an exercise set using the recording's motion, without using labels."""

import numpy as np

from util import no_segment


ADAPTIVE_SETTINGS = {
    "energy_window_sec": 1.0,
    "quiet_percentile": 10,
    "active_percentile": 90,
    "threshold_fraction": 0.20,
    "minimum_threshold": 0.02,
    "max_gap_sec": 2.0,
    "min_bout_sec": 4.0,
    "padding_sec": 0.75,
}


def protocol_segment(data, sample_rate):
    """Assume the requested five seconds of rest at each end actually happened."""
    start, end = no_segment(data, sample_rate)
    if end - start <= 11:
        return start, end
    return float(start + 5), float(end - 5)


def _runs(mask):
    """Return inclusive start/end indices of consecutive True samples."""
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1))


def adaptive_details(data, sample_rate):
    """Compute the segment and diagnostics used to explain it in our plots."""
    settings = ADAPTIVE_SETTINGS
    times = data["time"].to_numpy()
    energy = data["filtered_accel_mag"].rolling(
        max(round(sample_rate * settings["energy_window_sec"]), 2),
        center=True,
        min_periods=2,
    ).std().fillna(0).to_numpy()
    quiet, active = np.percentile(energy, [
        settings["quiet_percentile"], settings["active_percentile"]
    ])
    threshold = max(
        settings["minimum_threshold"],
        float(quiet + settings["threshold_fraction"] * (active - quiet)),
    )
    mask = energy > threshold

    # A rep can briefly slow to a stop. Join short internal gaps before deciding
    # whether the surrounding activity is a sustained set or a handling burst.
    for start, end in _runs(~mask):
        if start > 0 and end < len(mask) - 1:
            gap_duration = times[end + 1] - times[start - 1]
            if gap_duration <= settings["max_gap_sec"]:
                mask[start:end + 1] = True

    bouts = [
        (start, end) for start, end in _runs(mask)
        if times[end] - times[start] >= settings["min_bout_sec"]
    ]
    fallback = not bouts
    if fallback:
        # With no sustained activity, there is insufficient evidence for safe
        # trimming. Keep the recording rather than inventing a set boundary.
        segment = no_segment(data, sample_rate)
    else:
        start, end = max(bouts, key=lambda pair: times[pair[1]] - times[pair[0]])
        segment = (
            max(float(times[0]), float(times[start] - settings["padding_sec"])),
            min(float(times[-1]), float(times[end] + settings["padding_sec"])),
        )
    return {
        "segment": segment,
        "energy": energy,
        "threshold": threshold,
        "quiet_level": float(quiet),
        "active_level": float(active),
        "bouts": [(float(times[start]), float(times[end])) for start, end in bouts],
        "fallback": fallback,
    }


def adaptive_segment(data, sample_rate):
    """Keep the longest sustained motion bout, with a small margin at each end."""
    return adaptive_details(data, sample_rate)["segment"]
