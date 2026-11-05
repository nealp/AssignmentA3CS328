"""Combine a classifier's window predictions into one label per recording."""

import pandas as pd


def predict_recordings(model, frames, features=None):
    """Vote by most common label; exact ties use the alphabetically first label."""
    if features is None:
        features = getattr(model, "feature_names_in_", [
            column for column in frames.columns if column not in ("exercise", "recording")
        ])
    if frames.empty:
        return pd.DataFrame(columns=["recording", "true", "predicted"])
    if frames["recording"].isna().any():
        raise ValueError("Each window must identify its recording")
    if frames.groupby("recording")["exercise"].nunique(dropna=False).ne(1).any():
        raise ValueError("All windows of one recording must have the same true exercise")

    windows = frames[["recording", "exercise"]].copy()
    windows["predicted"] = model.predict(frames[list(features)])
    # Series.mode sorts tied labels, making the tie rule explicit and repeatable.
    recordings = windows.groupby("recording", sort=False).agg(
        true=("exercise", "first"),
        predicted=("predicted", lambda labels: labels.mode().iloc[0]),
    ).reset_index()
    return recordings[["recording", "true", "predicted"]]
