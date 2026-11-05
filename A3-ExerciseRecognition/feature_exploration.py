"""Step 4: compare window lengths, features, and segmentation on development data."""

import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from features import BASELINE_FEATURES, EXTRA_FEATURES, FEATURE_GROUPS, candidate_features
from segmentation import adaptive_segment, protocol_segment
from util import (baseline_segment, build_feature_frames, load_manifest, no_segment,
                  select_recordings, split_recordings)


def run_feature_exploration(output_dir="results/step4_features"):
    output = Path(output_dir)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    train, test = split_recordings(load_manifest(), test_size=0.3, random_state=42)
    fit, validation = split_recordings(train, test_size=0.25, random_state=42)
    assert set(train.path).isdisjoint(test.path)
    assert set(fit.path).isdisjoint(validation.path)
    frame_cache, rows, reports = {}, [], {}

    def get_frames(window_sec, segmentation="adaptive"):
        key = (window_sec, segmentation)
        if key not in frame_cache:
            segment_fn = {"adaptive": adaptive_segment, "none": no_segment,
                          "baseline": baseline_segment, "protocol_trim": protocol_segment}[segmentation]
            print(f"Building {window_sec}s windows, {segmentation} segmentation...", flush=True)
            frames = build_feature_frames(train, window_sec, segment_fn, candidate_features)
            assert set(frames.recording) == set(train.path)
            assert set(frames.recording).isdisjoint(test.path)
            assert np.isfinite(frames[BASELINE_FEATURES + EXTRA_FEATURES].to_numpy()).all()
            frames.to_csv(output / f"features_{segmentation}_{window_sec}s_train.csv", index=False)
            frame_cache[key] = frames
        return frame_cache[key]

    def evaluate(run_name, frames, feature_names, window_sec, segmentation="adaptive", phase="features"):
        fit_frames, val_frames = select_recordings(frames, fit), select_recordings(frames, validation)
        assert set(fit_frames.recording) == set(fit.path)
        assert set(val_frames.recording) == set(validation.path)
        # Fixed model settings throughout Step 4: no classifier tuning here.
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(fit_frames[feature_names], fit_frames.exercise)
        pred = model.predict(val_frames[feature_names])
        report = classification_report(val_frames.exercise, pred, output_dict=True, zero_division=0)
        reports[run_name] = report
        pd.DataFrame(val_frames[["recording", "exercise"]]).assign(predicted=pred).to_csv(
            output / f"predictions_{run_name}_validation.csv", index=False
        )
        (output / f"report_{run_name}_validation.json").write_text(json.dumps(report, indent=2) + "\n")
        row = {"run": run_name, "phase": phase, "segmentation": segmentation,
               "window_sec": window_sec, "feature_count": len(feature_names),
               "features": feature_names, "fit_windows": len(fit_frames),
               "validation_windows": len(val_frames),
               "validation_window_accuracy": float(accuracy_score(val_frames.exercise, pred))}
        rows.append(row)
        print(f"{run_name}: {row['validation_window_accuracy']:.2%}", flush=True)
        return row

    # 1. Follow the notebook order: compare lengths using baseline features only.
    window_rows = []
    for seconds in (2, 4, 6):
        window_rows.append(evaluate(f"window_{seconds}s_baseline", get_frames(seconds),
                                    BASELINE_FEATURES, seconds, phase="window"))
    selected_window = max(window_rows, key=lambda row: row["validation_window_accuracy"])["window_sec"]
    frames = get_frames(selected_window)
    print(f"Chosen baseline-feature window: {selected_window}s", flush=True)

    # 2. Controlled feature subsets at the chosen length, on identical windows.
    feature_sets = {"baseline": [], **FEATURE_GROUPS}
    for name, groups in {
        "axis_rhythm": ("axis_motion", "rhythm"),
        "axis_correlation": ("axis_motion", "correlation"),
        "axis_rhythm_correlation": ("axis_motion", "rhythm", "correlation"),
        "axis_rhythm_intensity": ("axis_motion", "rhythm", "intensity"),
    }.items():
        feature_sets[name] = [feature for group in groups for feature in FEATURE_GROUPS[group]]
    feature_sets["all"] = EXTRA_FEATURES
    for feature in EXTRA_FEATURES:
        feature_sets[f"without_{feature}"] = [name for name in EXTRA_FEATURES if name != feature]
    feature_rows = {}
    for name, extra in feature_sets.items():
        feature_rows[name] = evaluate(f"features_{name}", frames, BASELINE_FEATURES + extra, selected_window)
    eligible = [name for name, extra in feature_sets.items() if 5 <= len(extra) <= 10]
    selected_set = max(eligible, key=lambda name: (
        feature_rows[name]["validation_window_accuracy"], -len(feature_sets[name])
    ))
    selected_features = BASELINE_FEATURES + feature_sets[selected_set]
    print(f"Chosen feature set: {selected_set}", flush=True)

    # Marginal effects are conditional on the other nine additions being present.
    # They are useful diagnostics, not proof of independent causal contributions.
    effect_rows = []
    exercises = sorted(train.exercise.unique())
    full_row, full_report = feature_rows["all"], reports["features_all"]
    for feature in EXTRA_FEATURES:
        dropped = feature_rows[f"without_{feature}"]
        drop_report = reports[f"features_without_{feature}"]
        changes = {exercise: full_report[exercise]["recall"] - drop_report[exercise]["recall"]
                   for exercise in exercises}
        effect_rows.append({
            "feature": feature,
            "full_minus_without_accuracy_points": 100 * (full_row["validation_window_accuracy"] -
                                                          dropped["validation_window_accuracy"]),
            **{f"recall_change_{exercise}": value for exercise, value in changes.items()},
        })
    pd.DataFrame(effect_rows).to_csv(output / "feature_effects.csv", index=False)

    # Group-level recall changes identify which exercises gained/lost recognition.
    recall_rows = []
    for name, row in feature_rows.items():
        report = reports[row["run"]]
        baseline_report = reports[feature_rows["baseline"]["run"]]
        for exercise in exercises:
            recall_rows.append({"feature_set": name, "exercise": exercise,
                                "recall": report[exercise]["recall"],
                                "baseline_recall": baseline_report[exercise]["recall"],
                                "recall_change": report[exercise]["recall"] - baseline_report[exercise]["recall"],
                                "windows": report[exercise]["support"]})
    pd.DataFrame(recall_rows).to_csv(output / "feature_sets_by_exercise.csv", index=False)

    # 3. Same selected features/window and fixed forest for all segmenters.
    segmentation_rows = []
    for name in ("adaptive", "none", "baseline", "protocol_trim"):
        segmentation_rows.append(evaluate(f"segmentation_{name}", get_frames(selected_window, name),
                                           selected_features, selected_window, name, "segmentation"))
    # Adaptive is shared with counting, where Step 3 found it useful. Record all
    # classification alternatives; do not switch solely for a small window gain.
    selected_segmentation = "adaptive"
    selected_frames = frames[selected_features + ["exercise", "recording"]]
    selected_frames.to_csv(output / "selected_features_train.csv", index=False)
    select_recordings(selected_frames, fit).groupby("exercise")[selected_features].median().to_csv(
        output / "feature_medians_fit_by_exercise.csv"
    )

    selected_report = reports[feature_rows[selected_set]["run"]]
    baseline_report = reports[feature_rows["baseline"]["run"]]
    fig, ax = plt.subplots(figsize=(11, 4))
    positions = np.arange(len(exercises))
    ax.bar(positions - 0.2, [baseline_report[e]["recall"] for e in exercises], 0.4, label="7 baseline features")
    ax.bar(positions + 0.2, [selected_report[e]["recall"] for e in exercises], 0.4, label="selected features")
    ax.set_xticks(positions, exercises, rotation=25, ha="right")
    ax.set_ylabel("Validation window recall")
    ax.set_ylim(0, 1)
    ax.set_title(f"Feature comparison at {selected_window}s, adaptive segmentation, fixed Random Forest")
    ax.legend()
    fig.tight_layout()
    figure_path = output / "figures" / "recall_by_exercise.png"
    fig.savefig(figure_path, dpi=140)
    plt.close(fig)

    # Store lists as JSON in CSV to preserve exact feature ordering.
    table = pd.DataFrame(rows)
    table["features"] = table.features.map(json.dumps)
    table.to_csv(output / "all_comparisons.csv", index=False)
    summary = {
        "scope": "Step 4: fit/validation only; no test signals or predictions",
        "fixed_model": {"type": "RandomForestClassifier", "n_estimators": 100, "random_state": 42},
        "window_comparison": window_rows,
        "feature_comparison": list(feature_rows.values()),
        "segmentation_comparison": segmentation_rows,
        "selected_window_sec": selected_window,
        "selected_feature_set": selected_set,
        "selected_extra_features": feature_sets[selected_set],
        "selected_features": selected_features,
        "selected_segmentation": selected_segmentation,
        "selected_validation_window_accuracy": feature_rows[selected_set]["validation_window_accuracy"],
        "selected_frames_path": str(output / "selected_features_train.csv"),
        "figure": str(figure_path),
        "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                          for name in ("util.py", "features.py", "feature_exploration.py", "segmentation.py")},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(table[["run", "validation_window_accuracy"]].to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    run_feature_exploration()
