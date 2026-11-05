"""Compare segmentation while holding the counter, features, and models fixed."""

import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier

from baseline import BASELINE_FEATURES, WINDOW_SEC, count_summary
from segmentation import ADAPTIVE_SETTINGS, adaptive_details, adaptive_segment, protocol_segment
from util import (
    baseline_count_reps, baseline_features, baseline_segment, build_feature_frames,
    cut_segment, load_manifest, prepare_recording, select_recordings, split_recordings,
)


SEGMENTERS = {
    "baseline": baseline_segment,
    "protocol_trim": protocol_segment,
    "adaptive": adaptive_segment,
}

# Chosen after inspecting fit traces and fit errors. These are illustrations,
# not exceptions in the segmentation algorithm.
FAILURE_EXAMPLES = {
    "baseline_failure": "data/squats/002_2026-09-13_19-25-27.csv",
    "adaptive_failure": "data/ab_crunches/002_2026-10-13_22-50-15.csv",
}


def select_visual_examples(fit):
    """Choose one per exercise, then add any missing placements, without scores."""
    examples = fit.sort_values("path").groupby("exercise", sort=True).head(1).copy()
    for placement in sorted(set(fit["placement"]) - set(examples["placement"])):
        extra = fit[fit["placement"] == placement].sort_values("path").head(1)
        examples = pd.concat([examples, extra], ignore_index=True)
    return examples.reset_index(drop=True)


def plot_comparison(row):
    """Separate panels make it easy to see which motion each method keeps."""
    data, rate = prepare_recording(row["path"])
    details = adaptive_details(data, rate)
    fig, axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True)
    for ax, (name, segmenter) in zip(axes[:3], SEGMENTERS.items()):
        start, end = segmenter(data, rate)
        ax.plot(data["time"], data["filtered_accel_mag"], color="0.35", linewidth=0.7)
        ax.axvspan(start, end, alpha=0.2, color={
            "baseline": "tab:orange", "protocol_trim": "tab:green", "adaptive": "tab:blue"
        }[name])
        ax.set_ylabel("m/s²")
        ax.set_title(f"{name}: {start:.2f}–{end:.2f} s", loc="left", fontsize=10)
    axes[3].plot(data["time"], details["energy"], color="tab:blue", linewidth=0.8,
                 label="1-second moving standard deviation")
    axes[3].axhline(details["threshold"], color="tab:red", linestyle="--", label="adaptive threshold")
    axes[3].set_ylabel("motion (m/s²)")
    axes[3].set_xlabel("Time from recording start (seconds)")
    axes[3].legend(fontsize=8)
    fig.suptitle(f"{row['exercise']} | {row['placement']} | reported reps: {row['reps']}\n{row['path']}")
    fig.tight_layout()
    return fig


def run_segmentation_exploration(output_dir="results/step2_segmentation"):
    output = Path(output_dir)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    train, test = split_recordings(manifest, test_size=0.3, random_state=42)
    fit, validation = split_recordings(train, test_size=0.25, random_state=42)
    assert set(train["path"]).isdisjoint(test["path"])
    assert set(fit["path"]).isdisjoint(validation["path"])

    examples = select_visual_examples(fit)
    examples.to_csv(output / "visual_examples_fit.csv", index=False)
    assert len(examples) >= 6 and set(examples["path"]).issubset(fit["path"])
    assert set(examples["exercise"]) == set(fit["exercise"])
    assert set(examples["placement"]) == set(fit["placement"])
    figure_paths = []
    for _, row in examples.iterrows():
        fig = plot_comparison(row)
        filename = output / "figures" / f"{Path(row['path']).parent.name}_{Path(row['path']).stem}.png"
        fig.savefig(filename, dpi=120)
        plt.close(fig)
        figure_paths.append(str(filename))

    failure_examples = {}
    for name, recording in FAILURE_EXAMPLES.items():
        assert recording in set(fit["path"]), "Failure illustrations must be fit recordings"
        row = fit[fit["path"] == recording].iloc[0]
        fig = plot_comparison(row)
        filename = output / "figures" / f"{name}.png"
        fig.savefig(filename, dpi=120)
        plt.close(fig)
        failure_examples[name] = {"recording": recording, "figure": str(filename)}

    prepared = {}
    for _, row in train.iterrows():
        prepared[row["path"]] = prepare_recording(row["path"])

    counting_rows, classification_rows, boundary_rows = [], [], []
    for name, segmenter in SEGMENTERS.items():
        print(f"Evaluating {name}: counting on training recordings...", flush=True)
        predictions = []
        for _, row in train.iterrows():
            data, rate = prepared[row["path"]]
            start, end = segmenter(data, rate)
            assert np.isfinite([start, end]).all() and data.time.iloc[0] <= start < end <= data.time.iloc[-1]
            segment = cut_segment(data, start, end)
            count = baseline_count_reps(segment, rate, row["exercise"])
            predictions.append({
                "recording": row["path"], "exercise": row["exercise"],
                "placement": row["placement"], "true_reps": row["reps"],
                "pred_reps": count, "error": count - row["reps"],
            })
            audit = {
                "approach": name, "recording": row["path"],
                "exercise": row["exercise"], "placement": row["placement"],
                "duration_sec": float(data.time.iloc[-1] - data.time.iloc[0]),
                "start_sec": float(start), "end_sec": float(end),
                "kept_sec": float(segment.time.iloc[-1] - segment.time.iloc[0]),
            }
            if name == "adaptive":
                details = adaptive_details(data, rate)
                audit.update(fallback=details["fallback"], threshold=details["threshold"],
                             sustained_bouts=len(details["bouts"]))
            boundary_rows.append(audit)
        predictions = pd.DataFrame(predictions)
        predictions.to_csv(output / f"rep_predictions_{name}_train.csv", index=False)
        for split_name, part in (("train", train), ("fit", fit), ("validation", validation)):
            subset = predictions[predictions["recording"].isin(part["path"])]
            counting_rows.append({"approach": name, "split": split_name, **count_summary(subset)})
            for group in ("exercise", "placement"):
                count_summary(subset, group).to_csv(output / f"rep_{name}_{split_name}_by_{group}.csv")

        print(f"Evaluating {name}: features and fixed classifiers...", flush=True)
        frames = build_feature_frames(train, WINDOW_SEC, segmenter, baseline_features)
        fit_frames, val_frames = select_recordings(frames, fit), select_recordings(frames, validation)
        assert set(fit_frames.recording) == set(fit.path)
        assert set(val_frames.recording) == set(validation.path)
        assert set(frames.recording).isdisjoint(test.path)
        assert np.isfinite(frames[BASELINE_FEATURES].to_numpy()).all()
        frames.to_csv(output / f"features_{name}_train_4s.csv", index=False)
        model_predictions = val_frames[["recording", "exercise"]].copy()
        for model_name, model in (
            ("Decision Tree", DecisionTreeClassifier(max_depth=5, criterion="entropy", random_state=42)),
            ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)),
        ):
            model.fit(fit_frames[BASELINE_FEATURES], fit_frames.exercise)
            pred = model.predict(val_frames[BASELINE_FEATURES])
            slug = model_name.lower().replace(" ", "_")
            model_predictions[slug] = pred
            classification_rows.append({
                "approach": name, "model": model_name, "fit_windows": len(fit_frames),
                "validation_windows": len(val_frames),
                "validation_window_accuracy": float(accuracy_score(val_frames.exercise, pred)),
            })
            report = classification_report(val_frames.exercise, pred, output_dict=True, zero_division=0)
            (output / f"classification_{name}_{slug}_validation.json").write_text(json.dumps(report, indent=2) + "\n")
        model_predictions.to_csv(output / f"window_predictions_{name}_validation.csv", index=False)

    pd.DataFrame(counting_rows).to_csv(output / "counting_comparison.csv", index=False)
    pd.DataFrame(classification_rows).to_csv(output / "classification_comparison.csv", index=False)
    pd.DataFrame(boundary_rows).to_csv(output / "boundaries_train.csv", index=False)
    summary = {
        "scope": "Step 2 development comparison; no test signals or predictions",
        "adaptive_settings": ADAPTIVE_SETTINGS,
        "protocol_trim_seconds_per_end": 5,
        "fixed_counter": "baseline_count_reps",
        "fixed_features": BASELINE_FEATURES,
        "fixed_window_seconds": WINDOW_SEC,
        "counting": counting_rows,
        "classification": classification_rows,
        "visual_examples": figure_paths,
        "failure_examples": failure_examples,
        "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                          for name in ("util.py", "segmentation.py", "segmentation_exploration.py")},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.DataFrame(counting_rows).to_string(index=False), flush=True)
    print(pd.DataFrame(classification_rows).to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    run_segmentation_exploration()
