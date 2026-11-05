"""Step 3: choose a counter on fit data, then report development comparisons."""

import hashlib
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from baseline import count_summary
from rep_counting import REP_SETTINGS, count_reps, rep_details
from segmentation import adaptive_segment, protocol_segment
from util import (baseline_count_reps, baseline_segment, cut_segment, load_manifest,
                  no_segment, prepare_recording, split_recordings)


def plot_counting_example(row):
    data, rate = prepare_recording(row["recording"])
    start, end = adaptive_segment(data, rate)
    segment = cut_segment(data, start, end)
    details = rep_details(segment, rate, **REP_SETTINGS)
    peaks = details["peaks"]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(data.time, data.filtered_accel_mag, color="0.4", linewidth=0.7)
    axes[0].axvspan(start, end, color="tab:blue", alpha=0.2, label="adaptive segment")
    axes[0].set_ylabel("Filtered magnitude (m/s²)")
    axes[0].legend()
    axes[1].plot(segment.time, details["signal"], linewidth=0.8)
    axes[1].plot(segment.time.iloc[peaks], details["signal"][peaks], "rx", label=f"{len(peaks)} counted reps")
    axes[1].set_ylabel(f"{details['signal_name']} (m/s²)")
    axes[1].set_xlabel("Time from recording start (seconds)")
    axes[1].legend()
    fig.suptitle(f"{row['exercise']} | {row['placement']} | reported: {row['true_reps']}\n{row['recording']}")
    fig.tight_layout()
    return fig


def run_rep_counting_exploration(output_dir="results/step3_rep_counting"):
    output = Path(output_dir)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    train, test = split_recordings(load_manifest(), test_size=0.3, random_state=42)
    fit, validation = split_recordings(train, test_size=0.25, random_state=42)
    assert set(train.path).isdisjoint(test.path)
    assert set(fit.path).isdisjoint(validation.path)
    prepared = {row.path: prepare_recording(row.path) for row in train.itertuples()}
    segment_cache = {}

    def evaluate(recordings, segment_name="adaptive", settings=None, use_selected=False):
        segment_fn = {"none": no_segment, "baseline": baseline_segment,
                      "adaptive": adaptive_segment, "protocol_trim": protocol_segment}[segment_name]
        rows = []
        for row in recordings.itertuples():
            key = (segment_name, row.path)
            if key not in segment_cache:
                data, rate = prepared[row.path]
                start, end = segment_fn(data, rate)
                segment_cache[key] = (cut_segment(data, start, end), rate)
            segment, rate = segment_cache[key]
            if use_selected:
                count = count_reps(segment, rate, row.exercise)
            elif settings is None:
                count = baseline_count_reps(segment, rate, row.exercise)
            else:
                count = len(rep_details(segment, rate, **settings)["peaks"])
            rows.append({"recording": row.path, "exercise": row.exercise, "placement": row.placement,
                         "true_reps": row.reps, "pred_reps": int(count), "error": int(count) - row.reps})
        return pd.DataFrame(rows)

    # Stage 1 selects signal/cutoff on fit recordings only.
    configs = {"baseline": None}
    for kind in ("magnitude", "axis", "pca"):
        for cutoff in (1.0, 1.5):
            configs[f"{kind}_{cutoff:g}Hz"] = {
                "signal_kind": kind, "cutoff_hz": cutoff,
                "min_distance_sec": 0.6, "prominence_fraction": 0.5,
            }
    fit_predictions = {name: evaluate(fit, settings=settings) for name, settings in configs.items()}
    signal_winner = min((name for name in configs if name != "baseline"),
                        key=lambda name: count_summary(fit_predictions[name])["MAE"])
    print(f"Fit signal/cutoff choice: {signal_winner}", flush=True)

    # Stage 2 varies peak distance/prominence for that fit-selected signal.
    for distance in (0.6, 0.9, 1.2):
        for prominence in (0.5, 1.0):
            if distance == 0.6 and prominence == 0.5:
                continue  # Already in Stage 1.
            settings = {**configs[signal_winner], "min_distance_sec": distance,
                        "prominence_fraction": prominence}
            name = f"{signal_winner}_distance{distance:g}_prom{prominence:g}"
            configs[name] = settings
            fit_predictions[name] = evaluate(fit, settings=settings)
    selected_name = min(configs, key=lambda name: count_summary(fit_predictions[name])["MAE"])
    selected_settings = configs[selected_name]
    assert selected_settings == REP_SETTINGS, "Update count_reps defaults to the fit-selected settings"
    print(f"Fit-selected counter: {selected_name}; settings frozen before validation", flush=True)

    # Only now evaluate validation. These scores do not change the selection.
    comparisons, predictions = [], {}
    for name, settings in configs.items():
        val_predictions = evaluate(validation, settings=settings)
        predictions[name] = pd.concat([fit_predictions[name], val_predictions], ignore_index=True)
        assert set(predictions[name].recording) == set(train.path)
        assert predictions[name].recording.is_unique
        assert predictions[name].pred_reps.ge(0).all()
        predictions[name].to_csv(output / f"predictions_{name}_train.csv", index=False)
        for split_name, values in (("fit", fit_predictions[name]), ("validation", val_predictions)):
            comparisons.append({"counter": name, "split": split_name, **count_summary(values)})
    candidate_table = pd.DataFrame(comparisons)
    candidate_table.to_csv(output / "counter_comparison.csv", index=False)

    # Recheck segmentation with the same chosen counter; also reproduce the
    # rubric's five combinations on development data, not the reserved test set.
    combination_rows = []
    combination_predictions = {}
    for segment_name, counter_name in (("none", "baseline"), ("baseline", "baseline"),
                                       ("adaptive", "baseline"), ("baseline", "ours"),
                                       ("adaptive", "ours"), ("protocol_trim", "ours")):
        values = evaluate(train, segment_name=segment_name, use_selected=counter_name == "ours")
        key = f"{segment_name}_{counter_name}"
        combination_predictions[key] = values
        values.to_csv(output / f"combination_{key}_train.csv", index=False)
        for split_name, part in (("train", train), ("fit", fit), ("validation", validation)):
            subset = values[values.recording.isin(part.path)]
            combination_rows.append({"segmentation": segment_name, "counter": counter_name,
                                     "split": split_name, **count_summary(subset)})
    pd.DataFrame(combination_rows).to_csv(output / "segmentation_counter_comparison.csv", index=False)

    ours = combination_predictions["adaptive_ours"]
    for split_name, part in (("train", train), ("validation", validation)):
        subset = ours[ours.recording.isin(part.path)]
        for group in ("exercise", "placement"):
            count_summary(subset, by=group).to_csv(output / f"ours_{split_name}_by_{group}.csv")

    # Select illustrations on fit only, after scores are fixed.
    fit_errors = ours[ours.recording.isin(fit.path)].merge(
        predictions["baseline"][["recording", "error"]], on="recording", suffixes=("", "_baseline")
    )
    fit_errors["improvement"] = fit_errors.error_baseline.abs() - fit_errors.error.abs()
    examples = {
        "improvement": fit_errors.sort_values("improvement", ascending=False).iloc[0],
        "largest_error": fit_errors.loc[fit_errors.error.abs().idxmax()],
        "segmentation_failure": fit_errors[fit_errors.recording ==
            "data/ab_crunches/002_2026-10-13_22-50-15.csv"].iloc[0],
    }
    figures = {}
    for name, row in examples.items():
        fig = plot_counting_example(row)
        filename = output / "figures" / f"{name}.png"
        fig.savefig(filename, dpi=120)
        plt.close(fig)
        figures[name] = {"recording": row.recording, "figure": str(filename)}
    pd.DataFrame(examples).T.to_csv(output / "fit_examples.csv", index_label="example")

    summary = {
        "scope": "Step 3 development; selection on fit only; no test signals or predictions",
        "signal_stage_winner": signal_winner,
        "selected_counter": selected_name,
        "selected_settings": selected_settings,
        "candidate_settings": configs,
        "counter_comparison": comparisons,
        "segmentation_counter_comparison": combination_rows,
        "selected_training_metrics": count_summary(ours),
        "figures": figures,
        "exercise_labels": "true exercise; selected counter uses one common setting for all exercises",
        "source_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                          for name in ("util.py", "segmentation.py", "rep_counting.py", "rep_counting_exploration.py")},
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(candidate_table.to_string(index=False), flush=True)
    print(pd.DataFrame(combination_rows).query("split == 'validation'").to_string(index=False), flush=True)
    return summary


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    run_rep_counting_exploration()
