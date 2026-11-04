"""Reproduce Step 1 using the supplied methods, without evaluating test data."""

import hashlib
import json
import os
import platform
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

from util import (
    baseline_count_reps,
    baseline_features,
    baseline_segment,
    build_feature_frames,
    evaluate_rep_counter,
    load_manifest,
    select_recordings,
    split_recordings,
)


BASELINE_FEATURES = ["avg", "max", "med", "min", "q25", "q75", "std"]
WINDOW_SEC = 4  # Starting point only; compare lengths in Step 4.


def count_summary(results, by=None):
    """The supplied metric definitions, saved before rounding for the report."""
    values = results.assign(
        abs_error=results["error"].abs(),
        within_2=results["error"].abs().le(2),
    )
    if by is None:
        return {
            "recordings": len(values),
            "MAE": float(values["abs_error"].mean()),
            "within_2": float(values["within_2"].mean()),
            "bias": float(values["error"].mean()),
        }
    return values.groupby(by).agg(
        recordings=("error", "size"),
        MAE=("abs_error", "mean"),
        within_2=("within_2", "mean"),
        bias=("error", "mean"),
    )


def run_baseline(output_dir="results/step1_baseline"):
    """Run from the assignment directory; return settings and measured results."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest("./data")
    train, test = split_recordings(manifest, test_size=0.3, random_state=42)
    fit, validation = split_recordings(train, test_size=0.25, random_state=42)
    splits = {"train": train, "fit": fit, "validation": validation, "test": test}

    # Check split integrity before reading signals or training a model.
    assert manifest["path"].is_unique, "Duplicate recording paths"
    assert all(Path(path).is_file() for path in manifest["path"]), "Missing CSV"
    assert (len(train), len(fit), len(validation), len(test)) == (301, 225, 76, 129)
    fit_paths, val_paths, test_paths = (
        set(part["path"]) for part in (fit, validation, test)
    )
    assert fit_paths.isdisjoint(val_paths)
    assert (fit_paths | val_paths).isdisjoint(test_paths)
    assert fit_paths | val_paths == set(train["path"])
    assert set(train["path"]) | test_paths == set(manifest["path"])
    for name, part in splits.items():
        assert set(part["exercise"]) == set(manifest["exercise"])
        part.to_csv(output / f"{name}_manifest.csv", index=False)
    split_counts = pd.DataFrame(
        {name: part.groupby("exercise").size() for name, part in splits.items()}
    )
    split_counts.to_csv(output / "split_counts.csv")
    print("Recordings:", {name: len(part) for name, part in splits.items()}, flush=True)

    print("Counting repetitions on the 301 training recordings...", flush=True)
    rep_results = evaluate_rep_counter(train, baseline_segment, baseline_count_reps)
    assert len(rep_results) == len(train)
    assert rep_results["recording"].is_unique
    assert (rep_results["pred_reps"] >= 0).all()
    rep_results.to_csv(output / "rep_predictions_train.csv", index=False)
    rep_summaries = {}
    for name, part in (("train", train), ("fit", fit), ("validation", validation)):
        results = rep_results[rep_results["recording"].isin(part["path"])]
        rep_summaries[name] = count_summary(results)
        for group in ("exercise", "placement"):
            count_summary(results, by=group).to_csv(output / f"rep_{name}_by_{group}.csv")

    print("Building baseline features for training recordings only...", flush=True)
    frames = build_feature_frames(train, WINDOW_SEC, baseline_segment, baseline_features)
    fit_frames = select_recordings(frames, fit)
    val_frames = select_recordings(frames, validation)
    assert set(fit_frames["recording"]) == fit_paths
    assert set(val_frames["recording"]) == val_paths
    assert set(frames["recording"]).isdisjoint(test_paths)
    assert np.isfinite(frames[BASELINE_FEATURES].to_numpy()).all()
    frames.to_csv(output / "features_train_4s.csv", index=False)

    # These match the notebook defaults. There is no hyperparameter search here.
    models = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, criterion="entropy", random_state=42
        ),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    model_summaries = []
    predictions = val_frames[["recording", "exercise"]].copy()
    for name, model in models.items():
        print(f"Fitting {name}; evaluating validation windows...", flush=True)
        model.fit(fit_frames[BASELINE_FEATURES], fit_frames["exercise"])
        predicted = model.predict(val_frames[BASELINE_FEATURES])
        accuracy = float(accuracy_score(val_frames["exercise"], predicted))
        slug = name.lower().replace(" ", "_")
        predictions[slug] = predicted
        report = classification_report(
            val_frames["exercise"], predicted, output_dict=True, zero_division=0
        )
        (output / f"classification_{slug}_validation.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        model_summaries.append({
            "model": name,
            "fit_windows": len(fit_frames),
            "validation_windows": len(val_frames),
            "validation_window_accuracy": accuracy,
            "parameters": model.get_params(),
        })
    predictions.to_csv(output / "window_predictions_validation.csv", index=False)

    summary = {
        "step": "1: supplied baselines; no tuning; no test predictions",
        "split_recordings": {name: len(part) for name, part in splits.items()},
        "settings": {
            "random_state": 42,
            "segmentation": "baseline_segment: 1 s moving std > 0.5 m/s^2",
            "rep_counter": "baseline_count_reps: magnitude peaks above mean, >=1 s apart",
            "rep_exercise_labels": "true exercise (provided helper default)",
            "window_seconds": WINDOW_SEC,
            "window_overlap": 0.5,
            "features": BASELINE_FEATURES,
        },
        "rep_counting": rep_summaries,
        "classification": model_summaries,
        "environment": {
            "python": platform.python_version(),
            **{package: version(package) for package in (
                "numpy", "pandas", "scipy", "matplotlib", "scikit-learn"
            )},
        },
        "util_sha256": hashlib.sha256(Path("util.py").read_bytes()).hexdigest(),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Training rep metrics:", rep_summaries["train"], flush=True)
    for model in model_summaries:
        print(f"{model['model']} validation window accuracy: "
              f"{model['validation_window_accuracy']:.2%}", flush=True)
    print(f"Results saved in {output}", flush=True)
    return summary


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    run_baseline()
